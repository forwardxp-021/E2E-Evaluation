#!/usr/bin/env python3
"""Prospectively bind D-only sources and mutually exclusive future candidate roles; no V/C execution."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy,json
from pathlib import Path
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha
CFG=ROOT/'configs/rbr_v2_d2_v1';SDC=ROOT/'docs/stageR/e2e_scoped_direct_coast_v1'


def prepare():
 if (OUT/'D_execution_binding.json').exists():raise FileExistsError('D identities already frozen')
 plan=read(OUT/'acquisition_plan.json');rank={x['session_id']:x['SESSION_rank'] for x in plan['rows']};eligible=sorted([read(p) for p in (OUT/'static').glob('*.json') if read(p)['status']=='ELIGIBLE'],key=lambda r:(rank[r['session_id']],r['session_id']))
 if len(eligible)>=40:
  last=next(i for i,r in enumerate(plan['rows']) if r['session_id']==eligible[39]['session_id'])
  if not all((OUT/'static'/(r['session_id']+'.json')).exists() for r in plan['rows'][:last+1]):raise ValueError('Earlier source prefix incomplete; no D cohort selection')
 if len(eligible)<40:raise ValueError('Fewer than40 new independent static D sources; no D scientific launch')
 D=eligible[:40];remaining=eligible[40:];arms=[]
 old_tokens={r['database_log_token'] for r in read(ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json')['logs']}
 tokens=[r['log']['database_log_token'] for r in D]
 if len(set(tokens))!=40 or set(tokens)&old_tokens:raise ValueError('D raw LOG identity aliases historical or another SESSION')
 for i,r in enumerate(D,1):
  for arm in ['BASELINE','TREATMENT']:
   spec=copy.deepcopy(r['selected']['spec']);pid=f'RBR-D2-D-{i:02d}';spec.update(pair_id=pid,run_id=pid+'-'+arm,arm=arm,benchmark_version='RBR_V2_D2_SDC_D_ONLY_V1')
   arms.append(spec)
 cfg=read(ROOT/'configs/rbr_v2_d1_v1/contract.json');cfg.update(stage='RBR_V2_D2_INTERFACE',purpose='D_ONLY_PAIRED_CONTROLLED_SWAP_NUMERICAL_FEASIBILITY',runs=[],selection='FROZEN_SDC_RULE_NO_DETECTOR_OUTCOME_SELECTION')
 old=read(SDC/'QUALIFICATION_Arm_Specs_v2.json')['arms']
 for spec in old:
  cfg['runs'].append(dict(run_id=spec['run_id'],session_id=spec['session_id'],log_id=spec['log_id'],scenario_token=spec['scenario_token'],arm=spec['arm'],root='outputs/stageR/e2e_sdc_v1/QUALIFICATION/'+spec['run_id'],new_role='D_SWAP_CAL_EXISTING_B_REUSE',future_V_C_excluded_for_current_pipeline=True))
 for spec in arms:cfg['runs'].append(dict(run_id=spec['run_id'],session_id=spec['session_id'],log_id=spec['log_id'],scenario_token=spec['scenario_token'],arm=spec['arm'],root='outputs/stageR/rbr_v2_d2_D/'+spec['run_id'],new_role='D_AA_SWAP' if int(spec['pair_id'].split('-')[-1])<=20 else 'D_AB',future_V_C_excluded_for_current_pipeline=True))
 write(CFG/'interface_contract.json',cfg)
 binding=dict(stage='RBR_V2_D2_D_ONLY_V1',owner_task_authorized=True,arms=arms,run_ids=[a['run_id'] for a in arms],new_SESSION=40,first_arms=80,technical_retry=0,
              per_arm_wall_seconds=1200,total_scientific_CPU_seconds=14400,scientific_failure_policy='RETAIN_AND_CONTINUE_FROZEN_FULL_DENOMINATOR',infrastructure_failure_policy='STOP_NO_RETRY_NO_REPLACEMENT',
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tools/rbr_v2_d2_execute.py',ROOT/'tools/rbr_v2_d2_prepare.py',ROOT/'tools/rbr_v2_d2_interface.py',ROOT/'tools/rbr_v2_d2_statistics.py',ROOT/'tools/rbr_v2_d2_roles.py',ROOT/'tools/rbr_v2_model.py',ROOT/'tools/rbr_v2_canonical83.py',ROOT/'tools/s1_protocol_schema.py',ROOT/'tools/tsb_direct_coast_f2_production.py',ROOT/'tools/sdc_v1_execution.py',ROOT/'tools/tsb_direct_coast_f2_tracker.py',ROOT/'tools/tsb_coast_p1r_analysis.py']},
              physical_rule_sha256=sha(SDC/'Frozen_Qualification_Protocol_v1.json'),static_sha256={r['session_id']:sha(OUT/'static'/(r['session_id']+'.json')) for r in D},
              data_sha256={r['log']['source_file']:r['download_sha256'] for r in D},interface_contract_sha256=sha(CFG/'interface_contract.json'),
              scientific_scope='EXACT_SDC_V1_ONLY_NO_PRIMARY_V_C',D_fit='existing3 separate D1 baseline SESSION',D_cal='ALL20 existing SDC qualification B pairs prospectively assigned D, not V',
              D_AA='first20 new D SESSION controlled pair-swap pseudo-null',D_AB='last20 new D SESSION baseline-vs-exact10-treatment mixture',
              D_numerical_claim='PAIRED_CONTROLLED_SWAP_NULL_ONLY_NOT_OPERATIONAL_AA_FPR',source_domain_CI_power_coverage='NOT_ESTABLISHED_IF_UNIQUE_CLUSTER_SUPPORT_OR_VALIDATION_INSUFFICIENT',
              unchanged_model_sha256=cfg['checkpoint_sha256'],unchanged_H30_sha256=cfg['H30_source_sha256'],D_fit_scalers_sha256=sha(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json'),acquisition_plan_sha256=sha(OUT/'acquisition_plan.json'),second_source_plan_sha256=sha(OUT/'round2_source_plan.json'),D_SOURCE='FIRST40_ROUND1_ELIGIBLE_IN_COMPLETE_HASH_PREFIX; future second-source findings cannot change D cohort')
 for item in read(ROOT/'docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json')['map_binding']['map_files']:
  path=ROOT.parent/'nuplan/dataset/maps'/item['relative_path']
  if path.suffix=='.gpkg' or path.name=='nuplan-maps-v1.0.json':
   if sha(path)!=item['sha256']:raise ValueError('Frozen official map data changed '+str(path))
 for path in list((ROOT.parent/'nuplan/dataset/maps').rglob('*.gpkg'))+[ROOT.parent/'nuplan/dataset/maps/nuplan-maps-v1.0.json']:
  binding['data_sha256'][str(path.resolve())]=sha(path)
 write(OUT/'D_execution_binding.json',binding)
 # Candidate allocations are metadata only. Never use D scientific outcomes to alter these lists.
 oldD={x['session_id'] for x in cfg['runs'] if x['new_role']=='D_SWAP_CAL_EXISTING_B_REUSE'}
 import csv
 known=[r for r in csv.DictReader((ROOT/'outputs/rbr_v2_d1_v1/SESSION_Capacity_Audit.csv').open()) if r['V_claim_B_static_candidate']=='True' and r['session_id'] not in oldD]
 V=[dict(session_id=r['session_id'],source='D1_STATIC',qualified=r['existing_SDC_qualification_pair']=='True') for r in known]+[dict(session_id=r['session_id'],source='D2_NEW_STATIC',qualified=False) for r in remaining]
 V.sort(key=lambda r:__import__('hashlib').sha256(('D2_FUTURE_ROLE_CANDIDATE_V1|'+r['session_id']).encode()).hexdigest())
 d_extra=read(ROOT/'outputs/rbr_v2_d1_v1/D_E3_Static_Results.json')['rows'];cal=[dict(session_id=r['session_id'],source='D1_E3_STATIC_ONLY',qualified=False) for r in d_extra if r['status']=='ELIGIBLE_D_ONLY']
 design='UNPAIRED_PLANNING' if len(V)+len(cal)>=320 and len(V)>=240 else 'PAIRED_PLANNING' if len(V)+len(cal)>=160 and len(V)>=120 else 'CAPACITY_BLOCKED'
 size=(80,80,160) if design=='UNPAIRED_PLANNING' else (40,40,80)
 finalcal=cal+V[:max(0,size[0]-len(cal))];rest=V[max(0,size[0]-len(cal)):];aa=rest[:size[1]];ab=rest[size[1]:size[1]+size[2]]
 roles=dict(status=design,not_primary_freeze=True,not_power_certified=True,D_fit=read(ROOT/'configs/rbr_v2_d1_v1/contract.json')['runs'],D_existing_swap_cal=sorted(oldD),D_new=[r['session_id'] for r in D],
            final_calibration_candidates=finalcal,V_AA_candidates=aa,V_AB_candidates=ab,unallocated_candidates=rest[size[1]+size[2]:],C_metadata_only_new=plan['new_C_METADATA_ONLY'],C_original_protected='original five unchanged; raw/results not accessed',
            uniform_population='Conditional on official train/val and deterministic one minimum-size >=15s chunk per SESSION; not all nuPlan timeframes',required_counts=dict(zip(['final_calibration','V_AA','V_AB'],size)),actual_counts=dict(zip(['final_calibration','V_AA','V_AB'],map(len,[finalcal,aa,ab]))))
 sets=[{r['session_id'] for r in read(ROOT/'configs/rbr_v2_d1_v1/contract.json')['runs']},set(oldD),{r['session_id'] for r in D},set(plan['new_C_METADATA_ONLY'])]+[{r['session_id'] for r in a} for a in [finalcal,aa,ab]]
 for i,a in enumerate(sets):
  for b in sets[i+1:]:
   if a&b:raise ValueError('SESSION roles overlap')
 write(OUT/'candidate_role_partition_prefix_at_D_freeze.json',roles);print('D80 arms fixed; future capacity:',design,roles['actual_counts'],flush=True)

if __name__=='__main__':prepare()
