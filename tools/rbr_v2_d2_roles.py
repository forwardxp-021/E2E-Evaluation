#!/usr/bin/env python3
"""Predetermined metadata-only candidate allocation; does not read any D alarm, representation or V/C outcome."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv,hashlib
from tools.rbr_v2_d2_download import ROOT,OUT,read,write
from tools.rbr_v2_d2_round2 import combined


def allocate():
 b=read(OUT/'D_execution_binding.json') if (OUT/'D_execution_binding.json').exists() else None
 Dnew={r['session_id'] for r in b['arms']} if b else set();cfg=read(ROOT/'configs/rbr_v2_d2_v1/interface_contract.json') if b else {'runs':[]};Dold={r['session_id'] for r in cfg['runs'] if r['new_role']=='D_SWAP_CAL_EXISTING_B_REUSE'}
 v=[dict(session_id=r['session_id'],source='D1_STATIC_CANDIDATE',scientifically_qualified=r['existing_SDC_qualification_pair']=='True') for r in csv.DictReader((ROOT/'outputs/rbr_v2_d1_v1/SESSION_Capacity_Audit.csv').open()) if r['V_claim_B_static_candidate']=='True' and r['session_id'] not in Dold]
 new=combined();v.extend(dict(session_id=sid,source='D2_STATIC_CANDIDATE',scientifically_qualified=False) for sid,r in new.items() if r['status']=='ELIGIBLE' and sid not in Dnew)
 v.sort(key=lambda r:hashlib.sha256(('D2_FINAL_CANDIDATE_ROLE_HASH_V1|'+r['session_id']).encode()).hexdigest())
 cal=[dict(session_id=r['session_id'],source='D1_E3_STATIC_D_ONLY',scientifically_qualified=False,final_calibration_independence='PENDING_EXACT_BINDING') for r in read(ROOT/'outputs/rbr_v2_d1_v1/D_E3_Static_Results.json')['rows'] if r['status']=='ELIGIBLE_D_ONLY']
 support={r['name']:r for r in read(OUT/'bootstrap_source_support_audit.json')['rows']};envs=[('unpaired_small',80,80,160),('unpaired_medium',120,120,240),('unpaired_large',160,160,320),('paired_small',40,40,80),('paired_medium',60,60,120),('paired_large',80,80,160)];audit=[];choice=None
 for name,nc,na,nb in envs:
  needed=max(0,nc-len(cal))+na+nb;enough=len(v)>=needed;record=dict(envelope=name,required_final_cal=nc,required_AA=na,required_AB=nb,nominal_distinct_total=nc+na+nb,available_D_only_static_cal=len(cal),available_V_static=len(v),counts_enough=enough,additional_static_deficit=max(0,needed-len(v)),source_bootstrap_support_gate=support[name]['source_support_gate_pass'],power_and_CI_coverage='NOT_ESTABLISHED')
  audit.append(record)
  if choice is None and enough and support[name]['source_support_gate_pass']:choice=(name,nc,na,nb)
 if choice is None:choice=next((e for e,r in zip(envs,audit) if r['counts_enough']),('paired_small',40,40,80));state='CAPACITY_OR_BOOTSTRAP_SUPPORT_BLOCKED'
 else:state='COUNTS_AND_UNIQUE_SUPPORT_ONLY_NOT_PRIMARY_READY'
 name,nc,na,nb=choice;take=max(0,nc-len(cal));finalcal=cal+v[:take];left=v[take:];aa=left[:na];ab=left[na:na+nb]
 plan=read(OUT/'acquisition_plan.json');oldC=read(ROOT/'docs/stageR/e2e_scoped_direct_coast_v1/Pool_Split_v1.json')['protected_C_SESSION'];fit={r['session_id'] for r in read(ROOT/'configs/rbr_v2_d1_v1/contract.json')['runs']}
 sets=[fit,Dnew,Dold,{r['session_id'] for r in finalcal},{r['session_id'] for r in aa},{r['session_id'] for r in ab},set(oldC),set(plan['new_C_METADATA_ONLY'])]
 for i,x in enumerate(sets):
  for y in sets[i+1:]:
   if x&y:raise ValueError('D/fit/final-cal/AA/AB/C SESSION alias conflict')
 result=dict(status=state,design_candidate_only=name,not_primary_freeze=True,Primary_V_C_authorized=False,D_fit=sorted(fit),D_existing_swap_cal=sorted(Dold),D_new_science=sorted(Dnew),
             final_calibration_candidates=finalcal,V_AA_candidates=aa,V_AB_candidates=ab,unallocated_candidates=left[na+nb:],original_C_protected=oldC,new_C_index_reserved=plan['new_C_METADATA_ONLY'],
             C_new_identity_status='OFFICIAL_INDEX_ONLY_RAW_HEADER_AND_STATIC_ELIGIBILITY_NOT_ACCESSED',envelope_audit=audit,independence_condition='Strict SESSION-disjoint, including all source chunks and aliases; static != scientific qualification',
             no_D_results_read_for_allocation=True,calibration_caveat='Ten old E3 are only potential D calibration sources; exact final-cal suitability/independence is not automatically certified',source_domain_power_and_coverage='NOT_ESTABLISHED')
 write(OUT/'candidate_role_partition_final.json',result)
 print(state,name,'Vstatic',len(v),'D-onlycalstatic',len(cal),flush=True)

if __name__=='__main__':allocate()
