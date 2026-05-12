#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from train_human_behavior_embedding import Enc, AuxRegHead, load_feature_names, select_aux_target_indices
from trajectory_preprocessing import normalize_local


def run(a):
    d=Path(a.data_dir)
    traj=np.load(d/'traj.npy',allow_pickle=True).astype(np.float32)
    feat=np.load(d/'feat_style.npy').astype(np.float32)
    split=np.load(d/'split.npy',allow_pickle=True).astype(str)
    names=load_feature_names(a.data_dir)
    targets=[x.strip() for x in a.aux_targets.split(',') if x.strip()]
    idx=select_aux_target_indices(names, targets)
    y=np.clip(np.nan_to_num(feat[:,idx],nan=0.0,posinf=0.0,neginf=0.0),-a.aux_target_clip,a.aux_target_clip)
    m=split==a.eval_split
    x=torch.from_numpy(traj[m])
    y_t=torch.from_numpy(y[m])
    obj=torch.load(a.checkpoint,map_location='cpu')
    enc=Enc(int(obj.get('embedding_dim',64)))
    enc.load_state_dict(obj['model'],strict=False)
    aux=AuxRegHead(int(obj.get('embedding_dim',64)), int(obj.get('model_architecture',{}).get('aux_hidden_dim',128)), len(idx))
    aux.load_state_dict(obj.get('aux_head', {}), strict=False)
    dev=torch.device(a.device if (a.device!='cuda' or torch.cuda.is_available()) else 'cpu')
    enc.to(dev).eval(); aux.to(dev).eval()
    ds=TensorDataset(x,y_t); dl=DataLoader(ds,batch_size=a.batch_size)
    preds=[]; ys=[]
    with torch.no_grad():
      for xb,yb in dl:
        z=enc(normalize_local(xb.to(dev)))
        preds.append(aux(z).cpu().numpy()); ys.append(yb.numpy())
    p=np.concatenate(preds); t=np.concatenate(ys)
    out={}
    for i,n in enumerate(targets):
      mae=float(np.mean(np.abs(p[:,i]-t[:,i]))); rmse=float(np.sqrt(np.mean((p[:,i]-t[:,i])**2))); sp=float(spearmanr(p[:,i],t[:,i]).correlation)
      out[n]={'mae':mae,'rmse':rmse,'spearman':sp}
    Path(a.out_path).write_text(json.dumps({'eval_split':a.eval_split,'metrics':out},indent=2),encoding='utf-8')
    print('aux_eval_done')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',default='outputs/waymo_human_v1_full51')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--eval_split',default='test')
    p.add_argument('--aux_targets',default='rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw')
    p.add_argument('--aux_target_clip',type=float,default=10.0)
    p.add_argument('--batch_size',type=int,default=1024)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out_path',required=True)
    run(p.parse_args())
