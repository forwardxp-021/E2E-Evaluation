#!/usr/bin/env python3
"""Reuse the passed D1 extraction interface in six-arm batches, then export frozen z64/H30."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,copy,json
from pathlib import Path
import numpy as np
from unittest.mock import patch
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha
CFG=ROOT/'configs/rbr_v2_d2_v1/interface_contract.json';DEST=OUT/'interface'


def extract():
 import tools.rbr_v2_d1_interface as d1
 c=read(CFG);DEST.mkdir(exist_ok=False);allrows=[];guards=[]
 for start in range(0,len(c['runs']),6):
  subset=copy.deepcopy(c);subset['runs']=c['runs'][start:start+6];b=DEST/f'batch{start//6+1:03d}';b.mkdir();config=b/'contract.json';write(config,subset)
  with patch.object(d1,'OUT',b),patch.object(d1,'CFG',config):d1.extract()
  m=read(b/'input_manifest.json');allrows.extend(m['rows']);guards.append(m['zero_run_guard']);print('D-only fair representation inputs',len(allrows),'/',len(c['runs']),flush=True)
 if len(allrows)!=120 or len({(r['session_id'],r['arm']) for r in allrows})!=120:raise ValueError('Full60pair/120arm D input denominator missing')
 if any(any(g.values()) for g in guards):raise ValueError('Unexpected science call during D-only feature extraction')
 write(DEST/'input_manifest.json',dict(rows=allrows,contract_sha256=sha(CFG),status='PASS',guard='ALL_BATCH_GUARDS_ZERO',V_C_access=False))


def encode():
 import torch
 from tools.rbr_v2_model import RBRV2,normalize
 c=read(CFG);m=read(DEST/'input_manifest.json');scaler=read(ROOT/c['normalizer'])
 if sha(ROOT/c['checkpoint'])!=c['checkpoint_sha256'] or sha(ROOT/c['normalizer'])!=c['normalizer_sha256']:raise ValueError('Frozen model/normalizer identity mismatch')
 checkpoint=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False);model=RBRV2().to('mps' if torch.backends.mps.is_available() else 'cpu');model.load_state_dict(checkpoint['model']);model.eval();device=next(model.parameters()).device
 zs=[];hs=[];valid=[]
 for start in range(0,len(m['rows']),6):
  rows=m['rows'][start:start+6];data=[]
  for r in rows:
   path=ROOT/r['input_path']
   if sha(path)!=r['input_sha256']:raise ValueError('D input bytes changed')
   data.append(np.load(path));hs.append(data[-1]['H30']);valid.append(data[-1]['channel_valid'].mean())
  x=np.stack([normalize(a['x'],a['channel_valid'],scaler['input']) for a in data]);mask=np.stack([a['channel_valid'] for a in data]);ev=np.stack([a['ego_valid'] for a in data])
  def forward():
   with torch.no_grad():return model(torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device),torch.as_tensor(ev,device=device))[0].cpu().numpy()
  z=forward();np.testing.assert_array_equal(z,forward());zs.append(z);print('D-only frozen z64',start+len(rows),'/120',flush=True)
 z=np.concatenate(zs);h=np.array(hs)
 if z.shape!=(120,64) or h.shape!=(120,30) or not np.isfinite(z).all() or not np.isfinite(h).all():raise ValueError('Real D representation denominator or finite gate failure')
 np.save(DEST/'RBR64.npy',z);np.save(DEST/'H30.npy',h)
 write(DEST/'export.json',dict(status='PASS',RBR64_shape=list(z.shape),H30_shape=list(h.shape),checkpoint_sha256=c['checkpoint_sha256'],normalizer_sha256=c['normalizer_sha256'],H30_source_sha256=c['H30_source_sha256'],
      RBR64_sha256=sha(DEST/'RBR64.npy'),H30_sha256=sha(DEST/'H30.npy'),finite=True,repeated_eval_bitwise_equal=True,device=str(device),torch=torch.__version__,numpy=np.__version__,
      channel_valid_fraction=valid,z64_per_dimension_variance=z.var(0).tolist(),new_optimizer_steps=0,V_C_access=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','encode']);a=p.parse_args();extract() if a.mode=='extract' else encode()
