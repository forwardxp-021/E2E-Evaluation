#!/usr/bin/env python3
"""Fixed six preselected engineering replays: representation compatibility audit only, never a calibration replacement."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha
OUT=ROOT/'outputs/rbr_v2_d3_v1';DEST=OUT/'compatibility_representation_probe'


def extract():
 import tools.rbr_v2_d1_interface as d1
 c=read(ROOT/'configs/rbr_v2_d3_v1/interface_contract.json');c['runs']=[]
 for s in read(ROOT/'docs/stageR/rbr_v2_d3_v1/ENGINEERING_Protocol_v1.json')['arms'][2:8]:
  c['runs'].append(dict(run_id=s['run_id'],session_id=s['session_id'],log_id=s['log_id'],scenario_token=s['scenario_token'],arm=s['arm'],root='outputs/stageR/rbr_v2_d3/ENGINEERING/'+s['run_id'],new_role='D_FIXED_ENGINEERING_COMPATIBILITY_PROBE',prior_run_id=s['prior_source_run_id']))
 c['scope']='SIX_PRIOR_PHYSICAL_COMPATIBILITY_TARGETS_ONLY_NO_REPLACEMENT_CALIBRATION_OR_NEW_SOURCES';DEST.mkdir(exist_ok=False);p=DEST/'contract.json';write(p,c)
 with patch.object(d1,'OUT',DEST),patch.object(d1,'CFG',p):d1.extract()


def encode():
 import tools.rbr_v2_d3_interface as d3
 with patch.object(d3,'DEST',DEST),patch.object(d3,'CFG',DEST/'contract.json'):d3.encode()


def compare():
 from tools.rbr_v2_d1_statistics import transform
 m=read(DEST/'input_manifest.json');z=np.load(DEST/'RBR64.npy');h=np.load(DEST/'H30.npy');allrows=read(OUT/'interface/input_manifest.json')['rows'];oldz=np.load(OUT/'interface/RBR64.npy');oldh=np.load(OUT/'interface/H30.npy')
 d1rows=read(ROOT/'outputs/rbr_v2_d1_v1/input_manifest.json')['rows'];d1z=np.load(ROOT/'outputs/rbr_v2_d1_v1/RBR64.npy');d1h=np.load(ROOT/'outputs/rbr_v2_d1_v1/H30.npy');fit=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines'];rows=[]
 for i,r in enumerate(m['rows']):
  oldid=r['prior_run_id'];pool=(allrows,oldz,oldh) if any(a['run_id']==oldid for a in allrows) else (d1rows,d1z,d1h);j=next(j for j,a in enumerate(pool[0]) if a['run_id']==oldid)
  values={}
  for name,new,old in [('RBR64',z[i],pool[1][j]),('H30',h[i],pool[2][j])]:
   a=transform(new[None],fit[name])[0];b=transform(old[None],fit[name])[0];d2=float(np.dot(a-b,a-b));values[name]=dict(raw_max_abs_difference=float(np.max(np.abs(new-old))),raw_L2_difference=float(np.linalg.norm(new-old)),same_source_old_vs_repaired_frozen_kernel=float(np.exp(-d2/(2*fit[name]['sigma']**2))))
  rows.append(dict(new_engineering_run=r['run_id'],original_run=oldid,SESSION=r['session_id'],representations=values))
 write(OUT/'engineering_representation_compatibility_audit.json',dict(scope='POST_STATISTICS_FIXED_PHYSICAL_TARGET_DIAGNOSTIC_ONLY_NO_THRESHOLD_OR_POOL_CHANGE',rows=rows,targets_fixed_before_any_detector_outputs=True,
       replaced_calibration_vectors=0,refitted_scalers=0,new_SESSION=0,new_rollout=0,checkpoint_sha256=read(DEST/'contract.json')['checkpoint_sha256'],main_D_statistics_sha256_unchanged=sha(OUT/'D_statistics_results.json')))
 print('Fixed six engineering representation probes complete; no calibration replacements:',rows,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','encode','compare']);a=p.parse_args();{'extract':extract,'encode':encode,'compare':compare}[a.mode]()
