#!/usr/bin/env python3
"""D-only paired controlled swap feasibility, never operational/unpaired FPR or Primary."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,json,time
from pathlib import Path
import numpy as np
from tools.rbr_v2_d1_statistics import transform,threshold,alarms,unique_ids,mmd2
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha


def kernels(b,t,fit):
 z=transform(np.concatenate([b,t]),fit);d=np.sum((z[:,None]-z[None,:])**2,axis=-1)
 return np.exp(-d/(2*fit['sigma']**2))


def index_stat(K,left,right):
 n=len(left);xx=K[np.ix_(left,left)];yy=K[np.ix_(right,right)];xy=K[np.ix_(left,right)]
 return float((xx.sum()-np.trace(xx))/(n*(n-1))+(yy.sum()-np.trace(yy))/(n*(n-1))-2*xy.mean())


def draws(ks,ids,draws,seed,swap_null):
 ids=unique_ids(ids,len(ids));n=len(ids)
 if n<20:raise ValueError('INSUFFICIENT_UNIQUE_SESSION_M20; no duplicate source rescue')
 rng=np.random.RandomState(seed);out={k:np.empty(draws) for k in ks}
 for j in range(draws):
  selected=rng.choice(n,20,replace=False);drift=np.zeros(20,bool);drift[rng.choice(20,10,replace=False)]=True
  left=selected.copy();right=selected+n*drift
  if swap_null:
   swap=rng.randint(0,2,20).astype(bool);old=left.copy();left[swap]=right[swap];right[swap]=old[swap]
  for k,K in ks.items():out[k][j]=index_stat(K,left,right)
 return out


def main():
 m=read(OUT/'interface/input_manifest.json');z=np.load(OUT/'interface/RBR64.npy');h=np.load(OUT/'interface/H30.npy');fit=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines']
 # Exact same pre-D scalers, no fit or selection on any D2 detection result.
 pairs={}
 for i,r in enumerate(m['rows']):pairs.setdefault(r['session_id'],{})[r['arm']]=i
 groups={'cal':[],'aa':[],'ab':[]}
 for r in m['rows']:
  if r['arm']!='BASELINE':continue
  group='cal' if r['new_role']=='D_SWAP_CAL_EXISTING_B_REUSE' else 'aa' if r['new_role']=='D_AA_SWAP' else 'ab'
  groups[group].append(r['session_id'])
 if [len(groups[k]) for k in ['cal','aa','ab']]!=[20,20,20]:raise ValueError('Three disjoint20-SESSION D roles required')
 seen=set(fit['RBR64']['fit_ids'])
 for ids in groups.values():
  if seen&set(ids):raise ValueError('D_fit/cal/AA/AB SESSION overlap')
  seen|=set(ids)
 ks={};diagnostics={}
 for group,ids in groups.items():
  bi=[pairs[s]['BASELINE'] for s in ids];ti=[pairs[s]['TREATMENT'] for s in ids];ks[group]={k:kernels(a[bi],a[ti],fit[k]) for k,a in [('RBR64',z),('H30',h)]}
  diagnostics[group]={k:dict(off_diagonal_zero_fraction=float(np.mean(K[np.triu_indices(40,1)]==0)),off_diagonal_mean=float(K[np.triu_indices(40,1)].mean())) for k,K in ks[group].items()}
 started=time.monotonic();cal=draws(ks['cal'],groups['cal'],1999,2026091101,True);cut={k:threshold(v) for k,v in cal.items()}
 def numerical_evaluation(group,seed,swap):
  n=10000
  while True:
   result=draws(ks[group],groups[group],n,seed,swap);a={k:alarms(v,cut[k]) for k,v in result.items()};d=a['RBR64'].astype(float)-a['H30'];se=float(d.std(ddof=1)/np.sqrt(n))
   if se<=.005 or n>=40000:return result
   n*=2
 aa=numerical_evaluation('aa',2026091102,True);ab=numerical_evaluation('ab',2026091103,False)
 outcomes={}
 for name,stats in [('AA_CONTROLLED_SWAP',aa),('AB_PAIRED_PI50',ab)]:
  a={k:alarms(v,cut[k]) for k,v in stats.items()};d=a['RBR64'].astype(float)-a['H30']
  outcomes[name]=dict(alarm_probability={k:float(v.mean()) for k,v in a.items()},delta_BDD=float(d.mean()),conditional_MC_SE=float(d.std(ddof=1)/np.sqrt(len(d))),
                      draws=len(d),MC_precision_target=.005,finite_pool_MC_precision_only=True,scientific_CI='NOT_ESTABLISHED',unpaired_or_operational_FPR='NOT_ESTABLISHED')
  for k in a:np.save(OUT/('D_'+name+'_'+k+'_statistics.npy'),stats[k]);np.save(OUT/('D_'+name+'_'+k+'_alarms.npy'),a[k])
 for k,v in cal.items():np.save(OUT/('D_SWAP_CAL_'+k+'_statistics.npy'),v)
 scientific=read(OUT/'science_ledger.json');failures=sum(not p['pair_pass'] for p in scientific['pairs'])
 result=dict(scope='D_ONLY_PAIRED_CONTROLLED_ASSIGNMENT_SWAP_NOT_OPERATIONAL_RELEASE_FPR',status='CONDITIONAL_NUMERICAL_RESULTS_ONLY',outcomes=outcomes,thresholds=cut,kernel_diagnostics=diagnostics,
             source_groups=groups,D_fit_source_SHA256=sha(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json'),m=20,pi=.5,alpha=.05,
             genuine_new_scientific_pairs=len(scientific['pairs']),new_scientific_pair_failures=failures,whole_roster_qualification=failures==0,
             failures_removed=False,no_model_or_H_or_threshold_rule_tuning=True,
             source_cluster_interval='NOT_ESTABLISHED: each role20 IDs equals m20; source bootstrap loses unique support; no duplicate IDs as independent logs',
             power_at_gain_020_under_delta_010='NOT_ESTABLISHED',interval_coverage='NOT_ESTABLISHED',
             independent_release_population_FPR='NOT_ESTABLISHED',Primary_success='NOT_EVALUATED',
             saturation={k:dict(observed_finite_pool_max_gain_over_this_pipeline=1-outcomes['AB_PAIRED_PI50']['alarm_probability'][k],AB_at_least_099=outcomes['AB_PAIRED_PI50']['alarm_probability'][k]>=.99,AB_at_most_001=outcomes['AB_PAIRED_PI50']['alarm_probability'][k]<=.01) for k in cut},
             wall_seconds=time.monotonic()-started)
 write(OUT/'D_statistics_results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
