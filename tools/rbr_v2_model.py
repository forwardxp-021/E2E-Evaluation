#!/usr/bin/env python3
"""S3-P0 single-candidate shared z64 model; both heads backpropagate to encoder."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hashlib
import math
import numpy as np
import torch
from torch import nn
from tools.rbr_v2_canonical83 import ANGLES, PRESENCE


class RBRV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(167, 128)
        self.project = nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Linear(128,64))
        self.semantic = nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,13))
        self.temporal = nn.Sequential(nn.Linear(65,128), nn.ReLU(), nn.Linear(128,83))

    def forward(self, x, channel_valid, ego_valid):
        if x.ndim != 3 or tuple(x.shape[1:]) != (80,83):
            raise ValueError('Expected [N,80,83] input')
        if channel_valid.shape != x.shape or ego_valid.shape != x.shape[:2]:
            raise ValueError('Mask shape mismatch')
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite input')
        if not (ego_valid.sum(1) >= 64).all():
            raise ValueError('Input-invalid rows must be accounted for, not silently embedded')
        h = x.new_zeros((len(x),128))
        t = torch.arange(80,device=x.device,dtype=x.dtype)/79
        for i in range(80):
            inp = torch.cat([x[:,i],channel_valid[:,i].to(x.dtype),t[i].expand(len(x),1)],1)
            update = self.gru(inp,h)
            h = torch.where(ego_valid[:,i,None].bool(),update,h)
        z = self.project(h)
        rt = torch.cat([z[:,None,:].expand(-1,80,-1),t[None,:,None].expand(len(x),-1,1)],2)
        return z,self.semantic(z),self.temporal(rt)


def mask_plan(valid, identity, seed):
    """Deterministic 15% masked cells in shuffled length2 spans; no presence targets."""
    valid = np.asarray(valid,bool).copy(); valid[:,PRESENCE] = False
    digest = hashlib.sha256((str(seed)+':'+identity).encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4],'little'))
    candidates = [(t,c) for t in range(79) for c in range(83) if valid[t:t+2,c].any()]
    order = rng.permutation(len(candidates)); chosen = np.zeros((80,83),bool)
    remaining = int(math.ceil(.15*valid.sum()))
    for k in order:
        t,c = candidates[k]
        for u in (t,t+1):
            if valid[u,c] and not chosen[u,c] and remaining:
                chosen[u,c] = True; remaining -= 1
        if not remaining:
            break
    if remaining or not chosen.any():
        raise ValueError('Mask plan has insufficient valid continuous cells')
    return chosen


def objectives(pred_sem, pred_repr, truth_sem, truth_repr, sem_valid, hidden_mask):
    r = pred_repr-truth_repr
    # Angles were divided by pi, therefore normalized period is 2.
    r = r.clone(); r[:,:,ANGLES] = torch.remainder(r[:,:,ANGLES]+1,2)-1
    loss = torch.nn.functional.smooth_l1_loss(r,torch.zeros_like(r),reduction='none',beta=1)
    count = hidden_mask.sum((1,2))
    if not (count > 0).all():
        raise ValueError('Reconstruction sample has no valid masked targets')
    lr = ((loss*hidden_mask).sum((1,2))/count).mean()
    sl = torch.nn.functional.smooth_l1_loss(pred_sem,truth_sem,reduction='none',beta=1)
    sc = sem_valid.sum(0)
    if not (sc > 0).all():
        raise ValueError('Semantic target has no support; fail closed')
    ls = ((sl*sem_valid).sum(0)/sc).mean()
    return lr,ls


class Normalizer:
    """Streaming population statistics. Only explicit U_train rows can fit."""
    def __init__(self, width):
        self.count=np.zeros(width,np.int64); self.mean=np.zeros(width); self.m2=np.zeros(width)

    def update(self, values, valid, split):
        if split != 'train':
            raise ValueError('Normalizer fit is restricted to U_train')
        a = np.asarray(values,np.float64).reshape(-1,len(self.mean))
        mask = np.asarray(valid,bool).reshape(a.shape)
        if not np.isfinite(a[mask]).all():
            raise ValueError('Nonfinite normalizer observations')
        n = mask.sum(0); mean = np.where(mask,a,0).sum(0)/np.maximum(n,1)
        m2 = np.where(mask,(a-mean)**2,0).sum(0)
        total=self.count+n; delta=mean-self.mean
        self.m2 += m2+delta**2*self.count*n/np.maximum(total,1)
        self.mean += delta*n/np.maximum(total,1); self.count=total

    def state(self, physical=False):
        std=np.sqrt(self.m2/np.maximum(self.count,1)); std[std<=1e-6]=1
        mean=self.mean.copy()
        if physical:
            mean[ANGLES]=0; std[ANGLES]=np.pi
            mean[PRESENCE]=0; std[PRESENCE]=1
        return dict(count=self.count.tolist(),mean=mean.tolist(),std=std.tolist(),fit_split='train',
                    zero_support=np.flatnonzero(self.count==0).tolist())


def normalize(values, valid, state):
    out=(values-np.array(state['mean']))/np.array(state['std'])
    return np.where(valid,out,0).astype(np.float32)
