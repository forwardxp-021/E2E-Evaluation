#!/usr/bin/env python3
"""Frozen-model D3 fair80 extraction/export and fixed paired D-only diagnostics. No training or rollout entry."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,copy,csv,json,time
from unittest.mock import patch
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha
OUT=ROOT/'outputs/rbr_v2_d3_v1';DEST=OUT/'interface';CFG=ROOT/'configs/rbr_v2_d3_v1/interface_contract.json'


def extract():
 import tools.rbr_v2_d1_interface as d1
 c=read(CFG);gate=read(OUT/'ENGINEERING_gate.json')
 if gate['status']!='ENGINEERING_COMPATIBILITY_A_CERTIFIED_FOR_D_ONLY':raise PermissionError('No uncertified version mixing')
 expected=[];actual=[]
 for r in c['runs']:
  root=ROOT/r['root'];ok=(root/'execution_manifest.json').exists() and (root/'arm_science_review.json').exists()
  expected.append(dict(run_id=r['run_id'],session_id=r['session_id'],arm=r['arm'],role=r['new_role'],technical80_history_available=ok))
  if ok:actual.append(r)
 write(OUT/'interface_expected120_availability.json',dict(expected_rows=expected,expected=120,available=len(actual),incomplete_not_imputed=True,scientific_failures_not_filtered=True))
 DEST.mkdir(exist_ok=False);rows=[];guards=[]
 for start in range(0,len(actual),6):
  cfg=copy.deepcopy(c);cfg['runs']=actual[start:start+6];batch=DEST/f'batch{start//6+1:03d}';batch.mkdir();path=batch/'contract.json';write(path,cfg)
  with patch.object(d1,'OUT',batch),patch.object(d1,'CFG',path):d1.extract()
  m=read(batch/'input_manifest.json');rows.extend(m['rows']);guards.append(m['zero_run_guard']);print('D3 fair actual inputs',len(rows),'/',len(actual),flush=True)
 if any(any(g.values()) for g in guards):raise ValueError('Extraction invoked scientific propagation')
 write(DEST/'input_manifest.json',dict(status='COMPLETE120' if len(rows)==120 else 'PARTIAL_EXPLICIT_TECHNICAL_MISSINGNESS',rows=rows,expected120=True,scientific_failures_removed=False,contract_sha256=sha(CFG),guard_calls_all_zero=True))


def encode():
 import torch
 from tools.rbr_v2_model import RBRV2,normalize
 c=read(CFG);m=read(DEST/'input_manifest.json');rows=m['rows']
 if sha(ROOT/c['checkpoint'])!=c['checkpoint_sha256'] or sha(ROOT/c['normalizer'])!=c['normalizer_sha256']:raise ValueError('Frozen model/normalizer changed')
 checkpoint=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False)
 if checkpoint['committed_optimizer_steps']!=27390 or checkpoint['next_epoch']!=31:raise ValueError('Only frozen trained epoch30 permitted')
 model=RBRV2();model.load_state_dict(checkpoint['model']);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device);model.eval();normalizer=read(ROOT/c['normalizer'])
 zparts=[];hs=[];fractions=[];started=time.monotonic()
 for start in range(0,len(rows),6):
  arrays=[]
  for row in rows[start:start+6]:
   path=ROOT/row['input_path']
   if sha(path)!=row['input_sha256']:raise ValueError('Canonical83 bytes changed')
   arrays.append(np.load(path));hs.append(arrays[-1]['H30']);fractions.append(float(arrays[-1]['channel_valid'].mean()))
  x=np.stack([normalize(a['x'],a['channel_valid'],normalizer['input']) for a in arrays]);mask=np.stack([a['channel_valid'] for a in arrays]);ev=np.stack([a['ego_valid'] for a in arrays])
  def forward():
   with torch.no_grad():return model(torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device),torch.as_tensor(ev,device=device))[0].cpu().numpy()
  z=forward();np.testing.assert_array_equal(z,forward());zparts.append(z);print('D3 frozen z64',start+len(arrays),'/',len(rows),flush=True)
 z=np.concatenate(zparts);h=np.array(hs)
 if z.shape!=(len(rows),64) or h.shape!=(len(rows),30) or not np.isfinite(z).all() or not np.isfinite(h).all():raise ValueError('Frozen representation shape/finite failure')
 np.save(DEST/'RBR64.npy',z);np.save(DEST/'H30.npy',h)
 centered=z.astype(float)-z.astype(float).mean(0);singular=np.linalg.svd(centered,compute_uv=False);mass=singular**2;p=mass[mass>1e-20]/mass.sum() if mass.sum()>0 else np.array([])
 pairs={}
 for i,r in enumerate(rows):pairs.setdefault(r['session_id'],{})[r['arm']]=i
 diffs=[dict(session_id=s,RBR64_raw_L2=float(np.linalg.norm(z[v['BASELINE']]-z[v['TREATMENT']])),H30_raw_L2=float(np.linalg.norm(h[v['BASELINE']]-h[v['TREATMENT']])),H30_signed_delta=(h[v['TREATMENT']]-h[v['BASELINE']]).tolist()) for s,v in pairs.items() if set(v)=={'BASELINE','TREATMENT'}]
 write(DEST/'representation_diagnostics.json',dict(status='FINITE_FROZEN_EXPORT',rows=len(rows),RBR64_shape=list(z.shape),H30_shape=list(h.shape),RBR64_sha256=sha(DEST/'RBR64.npy'),H30_sha256=sha(DEST/'H30.npy'),checkpoint_sha256=c['checkpoint_sha256'],normalizer_sha256=c['normalizer_sha256'],
       finite=True,repeated_eval_bitwise_equal=True,per_channel_RBR64_variance=z.var(0).tolist(),centered_numerical_rank=int(np.linalg.matrix_rank(centered)),entropy_effective_rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
       valid_channel_fraction_by_arm=fractions,pair_differences=diffs,L2_caveat='Different units across representations; raw L2 magnitudes cannot rank detection utility',
       new_optimizer_steps=0,device=device,torch=torch.__version__,numpy=np.__version__,wall_seconds=time.monotonic()-started,Primary_V_C=False))


def statistics():
 from tools.rbr_v2_d1_statistics import threshold,alarms
 from tools.rbr_v2_d2_statistics import kernels,draws
 manifest=read(DEST/'input_manifest.json');rows=manifest['rows'];z=np.load(DEST/'RBR64.npy');h=np.load(DEST/'H30.npy');fit=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines']
 pairs={};groups={'cal':[],'aa':[],'ab':[]}
 for i,r in enumerate(rows):pairs.setdefault(r['session_id'],{})[r['arm']]=i
 for r in rows:
  if r['arm']=='BASELINE':groups['cal' if r['new_role']=='D_SWAP_CAL_EXISTING_B_REUSE' else 'aa' if r['new_role']=='D_AA_SWAP' else 'ab'].append(r['session_id'])
 if len(rows)!=120 or any(len(ids)!=20 for ids in groups.values()) or any(set(v)!={'BASELINE','TREATMENT'} for v in pairs.values()):
  write(OUT/'D_statistics_blocked.json',dict(status='INSUFFICIENT_COMPLETE_INDEPENDENT_M20_ROLE_SUPPORT',actual_role_counts={k:len(v) for k,v in groups.items()},missing_no_imputation=True));return
 seen=set(fit['RBR64']['fit_ids'])
 for ids in groups.values():
  if len(set(ids))!=20 or seen&set(ids):raise ValueError('D fit/cal/AA/AB overlap or duplicate SESSION')
  seen|=set(ids)
 ks={}
 for group,ids in groups.items():
  bi=[pairs[s]['BASELINE'] for s in ids];ti=[pairs[s]['TREATMENT'] for s in ids];ks[group]={k:kernels(a[bi],a[ti],fit[k]) for k,a in [('RBR64',z),('H30',h)]}
 started=time.monotonic();cal=draws(ks['cal'],groups['cal'],1999,2026091101,True);cut={k:threshold(v) for k,v in cal.items()};outcomes={}
 for name,g,seed,swap in [('AA_CONTROLLED_SWAP','aa',2026091102,True),('AB_PAIRED_PI50','ab',2026091103,False)]:
  n=10000
  while True:
   stats=draws(ks[g],groups[g],n,seed,swap);a={k:alarms(v,cut[k]) for k,v in stats.items()};delta=a['RBR64'].astype(float)-a['H30'];se=float(delta.std(ddof=1)/np.sqrt(n))
   if se<=.005 or n>=40000:break
   n*=2
  outcomes[name]=dict(alarm_probability={k:float(v.mean()) for k,v in a.items()},delta_BDD=float(delta.mean()),conditional_MC_SE=se,draws=n,conditional_pool_only=True,source_domain_CI='NOT_ESTABLISHED',operational_unpaired_FPR='NOT_ESTABLISHED')
  for k in stats:np.save(OUT/('D_'+name+'_'+k+'_statistics.npy'),stats[k]);np.save(OUT/('D_'+name+'_'+k+'_alarms.npy'),a[k])
 for k in cal:np.save(OUT/('D_SWAP_CAL_'+k+'_statistics.npy'),cal[k])
 d=read(OUT/'SCIENCE_ledger.json');failed=[p['pair_id'] for p in d['pairs'] if not p['pair_pass']]
 result=dict(status='REAL_D_ONLY_CONDITIONAL_PAIRED_SWAP_EVIDENCE_NOT_PRIMARY',outcomes=outcomes,thresholds=cut,groups=groups,m=20,pi=.5,alpha=.05,
       D_fit_scaler_sha256=sha(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json'),checkpoint_sha256=read(CFG)['checkpoint_sha256'],legacy_calibration_compatibility_sha256=sha(OUT/'ENGINEERING_gate.json'),
       new_scientific_complete_pairs=len(d['pairs']),scientific_failed_pairs=failed,failed_pairs_not_removed=True,whole_roster_qualified=not failed,
       saturation={k:dict(AB_alarm_at_least_099=outcomes['AB_PAIRED_PI50']['alarm_probability'][k]>=.99,AB_alarm_at_most_001=outcomes['AB_PAIRED_PI50']['alarm_probability'][k]<=.01,finite_pool_max_gain_over_this_pipeline=1-outcomes['AB_PAIRED_PI50']['alarm_probability'][k]) for k in cut},
       source_cluster_CI='NOT_ESTABLISHED: each role20 equals m20; source bootstrap cannot preserve20 unique originals; MC draws are not new logs',power_at_gain020_under_delta010='NOT_ESTABLISHED',interval_coverage='NOT_ESTABLISHED',
       independently_validated_population_FPR='NOT_ESTABLISHED',Primary_result='NOT_EVALUATED',no_checkpoint_H30_kernel_rule_FPR_m_pi_or_threshold_selection=True,
       kernel_diagnostics={g:{k:dict(off_diagonal_zero_fraction=float((K[np.triu_indices(40,1)]==0).mean()),off_diagonal_mean=float(K[np.triu_indices(40,1)].mean())) for k,K in val.items()} for g,val in ks.items()},wall_seconds=time.monotonic()-started)
 write(OUT/'D_statistics_results.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','encode','statistics']);a=p.parse_args();{'extract':extract,'encode':encode,'statistics':statistics}[a.mode]()
