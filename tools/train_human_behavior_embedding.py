#!/usr/bin/env python3
import argparse, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class HumanDS(Dataset):
    def __init__(self,traj,feat): self.traj=traj.astype(np.float32); self.feat=feat.astype(np.float32)
    def __len__(self): return len(self.traj)
    def __getitem__(self,i): return self.traj[i], self.feat[i]

def normalize_local(tr):
    xy=tr[...,:2]; v=tr[...,2:4]
    x0=xy[:,0:1,:]; xy=xy-x0
    h=torch.atan2(v[:,0,1], v[:,0,0])
    c=torch.cos(-h)[:,None]; s=torch.sin(-h)[:,None]
    xr=xy[:,:,0]*c-xy[:,:,1]*s; yr=xy[:,:,0]*s+xy[:,:,1]*c
    vr=v[:,:,0]*c-v[:,:,1]*s; vy=v[:,:,0]*s+v[:,:,1]*c
    return torch.stack([xr,yr,vr,vy],dim=-1)

class Enc(nn.Module):
    def __init__(self, emb=64, hid=128):
        super().__init__(); self.gru=nn.GRU(4,hid,batch_first=True); self.head=nn.Sequential(nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,emb))
    def forward(self,x):
        _,h=self.gru(x); z=self.head(h[-1]); return F.normalize(z,dim=1)

def soft_loss(z,f,temp):
    f=F.normalize(f,dim=1); s_f=(f@f.T)/temp; t=F.softmax(s_f,dim=1)
    s_z=(z@z.T)/temp; lp=F.log_softmax(s_z,dim=1)
    return F.kl_div(lp,t,reduction='batchmean')

def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    if args.smoke_test:
        td=Path(tempfile.mkdtemp())/'d'; td.mkdir(parents=True)
        n,t=96,80
        traj=np.random.randn(n,t,4).astype(np.float32); feat=np.random.randn(n,16).astype(np.float32)
        split=np.array(['train']*64+['val']*16+['test']*16)
    else:
        d=Path(args.data_dir); traj=np.load(d/'traj.npy'); feat=np.load(d/'feat_style.npy'); split=np.load(d/'split.npy').astype(str)
    mu=feat[split=='train'].mean(0,keepdims=True); sd=feat[split=='train'].std(0,keepdims=True)+1e-6
    feat=(feat-mu)/sd
    dev=torch.device(args.device if (args.device!='cuda' or torch.cuda.is_available()) else 'cpu')
    tr_ds=HumanDS(traj[split=='train'],feat[split=='train']); va_ds=HumanDS(traj[split=='val'],feat[split=='val'])
    tr=DataLoader(tr_ds,batch_size=args.batch_size,shuffle=True,num_workers=0); va=DataLoader(va_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=Enc(args.embedding_dim).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    best=1e9; bad=0; logs=[]
    for ep in range(args.epochs):
        model.train(); tl=[]
        for tb,fb in tr:
            tb,fb=tb.to(dev),fb.to(dev); tb=normalize_local(tb)
            loss=soft_loss(model(tb),fb,args.temperature)
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss.item()))
        model.eval(); vl=[]
        with torch.no_grad():
            for tb,fb in va:
                tb,fb=tb.to(dev),fb.to(dev); tb=normalize_local(tb)
                vl.append(float(soft_loss(model(tb),fb,args.temperature).item()))
        t=float(np.mean(tl)); v=float(np.mean(vl)) if vl else t
        logs.append({'epoch':ep+1,'train_loss':t,'val_loss':v})
        if v<best: best=v; bad=0; torch.save({'model':model.state_dict(),'embedding_dim':args.embedding_dim}, out/'model.pt')
        else: bad+=1
        if bad>=args.patience: break
    pd.DataFrame(logs).to_csv(out/'train_log.csv',index=False)
    (out/'train_summary.json').write_text(json.dumps({'best_val_loss':best,'epochs_ran':len(logs),'embedding_dim':args.embedding_dim,'used_pseudo_labels_for_training':False},indent=2))
    (out/'val_metrics.json').write_text(json.dumps({'best_val_loss':best},indent=2))
    plt.figure(); plt.plot([x['epoch'] for x in logs],[x['train_loss'] for x in logs],label='train'); plt.plot([x['epoch'] for x in logs],[x['val_loss'] for x in logs],label='val'); plt.legend(); plt.tight_layout(); plt.savefig(out/'training_curve.png'); plt.close()
    print('train_done')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data_dir',default='outputs/waymo_human_v1_full51'); p.add_argument('--out_dir',required=True)
    p.add_argument('--embedding_dim',type=int,default=64); p.add_argument('--batch_size',type=int,default=512); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--temperature',type=float,default=0.1); p.add_argument('--device',default='cpu'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=5); p.add_argument('--smoke_test',action='store_true')
    a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); run(a)
