#!/usr/bin/env python3
"""SAP bootstrap unique-ID occupancy only; not synthetic source expansion, power or CI coverage."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.rbr_v2_d2_download import OUT,write


def main():
 cases=[('current_D',[20,20,20],20),('paired_small',[40,40,80],20),('paired_medium',[60,60,120],20),('paired_large',[80,80,160],20),('unpaired_small',[80,80,160],40),('unpaired_medium',[120,120,240],40),('unpaired_large',[160,160,320],40)];rows=[]
 for name,ns,needed in cases:
  rng=np.random.RandomState(2026091104);support=[]
  for _ in range(2000):support.append([np.count_nonzero(rng.multinomial(n,np.full(n,1/n))) for n in ns])
  a=np.asarray(support);bad=(a<needed).any(1)
  rows.append(dict(name=name,role_pool_sizes=ns,required_original_IDs_per_release=needed,outer_replicates=2000,min_unique_by_role=a.min(0).tolist(),median_unique_by_role=np.median(a,axis=0).tolist(),max_unique_by_role=a.max(0).tolist(),replicates_with_insufficient_source_support=int(bad.sum()),source_support_gate_pass=not bad.any()))
 result=dict(scope='METADATA_ONLY_MULTINOMIAL_BOOTSTRAP_SUPPORT_NOT_POWER_OR_INTERVAL_COVERAGE',seed=2026091104,rows=rows,real_new_samples_created=0,RBR_H30_results_used=False,interpretation='Even a supported envelope still requires D-only power and coverage validation; no new sample-size rule, no changed m, no duplicate-ID rescue')
 write(OUT/'bootstrap_source_support_audit.json',result)
 for r in rows:print(r['name'],r['min_unique_by_role'],'insufficient',r['replicates_with_insufficient_source_support'],'/2000',flush=True)

if __name__=='__main__':main()
