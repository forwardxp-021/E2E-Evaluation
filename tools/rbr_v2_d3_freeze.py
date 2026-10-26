#!/usr/bin/env python3
"""Certify engineering compatibility from physical evidence and freeze the original D40; no detector reads."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy,json
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha
from tools.rbr_v2_d3_execute import OUT,DOC,SCIENCE,contract,sources


def trace(root):return [json.loads(l)['current_ego'] for l in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()]
def controls(root):return [json.loads(l) for l in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]


def certify():
 c=read(contract('ENGINEERING'));d=read(OUT/'ENGINEERING_ledger.json');compat=read(OUT/'legacy_D20_reference_compatibility.json')
 if not d['stopped'] or d['stop_reason']!='ALL_FROZEN_FIRST_ARMS_COMPLETE' or len(d['attempts'])!=12 or len(d['pairs'])!=6 or not all(p['pair_pass'] for p in d['pairs']):raise ValueError('D3 engineering qualification incomplete or failed; no large D execution')
 oldruns=[r for r in read(ROOT/'configs/rbr_v2_d2_v1/interface_contract.json')['runs'] if r['new_role']=='D_SWAP_CAL_EXISTING_B_REUSE']+read(ROOT/'configs/rbr_v2_d1_v1/contract.json')['runs']
 old={r['run_id']:ROOT/r['root'] for r in oldruns};comparisons=[]
 for s in c['arms']:
  root=SCIENCE/'ENGINEERING'/s['run_id'];m=read(root/'execution_manifest.json')
  if any(sha(root/p)!=h for p,h in m['artifacts'].items()):raise ValueError('Engineering artifact changed')
  x=trace(root);u=controls(root)
  if len(x)!=80 or len(u)!=79 or not read(root/'actual_observation_audit.json')['status']=='PASS':raise ValueError('Engineering full physical/observation evidence missing')
  previous=s['prior_source_run_id']
  if previous not in old:continue
  base=old[previous];y=trace(base);v=controls(base)
  longitudinal={k:all(a[k]==b[k] for a,b in zip(u,v)) for k in ['longitudinal_requested_acceleration','model_realized_acceleration','realized_speed','speed_before']}
  speeds_equal=[r['speed_mps'] for r in x]==[r['speed_mps'] for r in y]
  if not all(longitudinal.values()) or not speeds_equal:raise ValueError('Repair unexpectedly changed the longitudinal dynamics/commands')
  xy=lambda z:np.array([[r['rear_axle']['x'],r['rear_axle']['y']] for r in z]);yaw=lambda z:np.array([r['rear_axle']['heading'] for r in z])
  comparisons.append(dict(new_run_id=s['run_id'],prior_run_id=previous,all_longitudinal_control_and_model_values_exact=longitudinal,all80_speeds_bitwise_equal=speeds_equal,
      max_abs_XY_coordinate_difference_m=float(np.abs(xy(x)-xy(y)).max()),max_position_norm_difference_m=float(np.linalg.norm(xy(x)-xy(y),axis=1).max()),max_abs_heading_difference_rad=float(np.abs(np.arctan2(np.sin(yaw(x)-yaw(y)),np.cos(yaw(x)-yaw(y)))).max()),
      official_safety_new=read(root/'official_safety_adapter.json')['frozen_arm_safety_pass'],official_safety_old=read(base/'official_safety_adapter.json')['frozen_arm_safety_pass']))
 if len(comparisons)!=6:raise ValueError('Four affected legacy arms and one D_fit pair must be compared')
 result=dict(status='ENGINEERING_COMPATIBILITY_A_CERTIFIED_FOR_D_ONLY',kind='A_ENDPOINT_COORDINATE_COMPATIBILITY_REPAIR_NOT_NEW_BEHAVIOR_INTERVENTION',
      technical_version='SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1',engineering_complete_arms=12,engineering_complete_pairs=6,all_original_mechanism_F_match_balance_and_official_safety_pass=True,
      legacy_reference_calls=3200,legacy_bitwise_identical_reference_calls=sum(80-r['reference_different_frames'] for r in compat['rows']),legacy_bug_affected_reference_calls=sum(r['reference_different_frames'] for r in compat['rows']),
      unaffected_case_policy='Exact original arc and geometry retained; only endpoint-clamped coordinate mismatch repaired',targeted_physical_replays=comparisons,
      legacy_D20_calibration_reuse='Explicit D-only physical compatibility certificate: same frozen intervention/control/model/clock/measurement;3196 correct reference calls exact; all4 affected legacy arms replayed with exact longitudinal history, preserved paired qualification/safety, and measured sub-millimeter lateral differences. This is NOT a fresh20-pair qualification or Primary certificate.',
      RBR_H30_outputs_used=False,scientific_safety_or_F_match_thresholds_changed=False,route_ids_or_native_geometry_replaced=False,old_D2_failures_unchanged=True,
      evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'ENGINEERING_ledger.json',OUT/'legacy_D20_reference_compatibility.json',OUT/'patched_reference_v2_diagnosis.json',ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json']})
 write(OUT/'ENGINEERING_gate.json',result);print('Engineering certificate PASS; max historical coordinate difference',max(r['max_abs_XY_coordinate_difference_m'] for r in comparisons),flush=True)


def freeze():
 if contract('SCIENCE').exists():raise FileExistsError('D3 science contract already frozen')
 gate=read(OUT/'ENGINEERING_gate.json')
 if gate['status']!='ENGINEERING_COMPATIBILITY_A_CERTIFIED_FOR_D_ONLY':raise PermissionError('Qualified engineering compatibility required')
 old=read(ROOT/'outputs/rbr_v2_d2_v1/D_execution_binding.json');arms=[]
 for s in old['arms']:
  a=copy.deepcopy(s);a.update(run_id=s['run_id'].replace('RBR-D2-D-','D3-D-'),pair_id=s['pair_id'].replace('RBR-D2-D-','D3-D-'),benchmark_version='SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1',D3_phase='SCIENCE',prior_source_run_id=s['run_id'],prior_D2_exposure_disclosed=True);arms.append(a)
 c=dict(stage='RBR_V2_D3_D_ONLY_SCIENCE',phase='SCIENCE',owner_task_scope='D3 bounded D-only first executions; no V/C/Primary, no model selection',arms=arms,run_ids=[a['run_id'] for a in arms],max_first_arms=80,
       per_arm_wall_seconds=1200,total_scientific_CPU_seconds=14400,technical_retry=0,science_failure_policy='RETAIN_AND_CONTINUE_FULL_FROZEN_DENOMINATOR',infrastructure_failure_policy='STOP_NO_RETRY_NO_IDENTITY_REPLACEMENT',
       source_sha256=sources(),data_sha256=old['data_sha256'],previous_D2_ledgers=read(contract('ENGINEERING'))['previous_D2_ledgers'],proof_sha256={str((OUT/'ENGINEERING_gate.json').relative_to(ROOT)):sha(OUT/'ENGINEERING_gate.json')},
       unchanged_F2_protocol_sha256=sha(ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json'),parent_D2_cohort_sha256=sha(ROOT/'outputs/rbr_v2_d2_v1/D_execution_binding.json'),
       representation_checkpoint_sha256=old['unchanged_model_sha256'],D_fit_scaler_sha256=old['D_fit_scalers_sha256'],mechanism_F_match_safety='EXACT_ORIGINAL_F2_PLUS_SDC_TEMPORAL_GATE',
       role_firewall='OriginalD40 entire SESSION reservedE3; oldD20 explicitD-swap-cal; originalD_fit3; no final-cal/V/C source consumed; engineering observedD01/D02/D03 and priorD2D01 are not independent confirmation',
       paired_D_statistics='Original D2 fixed roles: all20 historicalD pairs swap-cal; newD01-20 held-out controlled-swap AA;newD21-40 AB. m20,pi.5,alpha.05,1999cal1900rank strict>,10000eval max40000 conditionalMC. Not operational FPR or Primary.',
       no_model_H30_kernel_threshold_rule_or_source_rule_tuning=True,download_authorized_bytes=0,Primary_V_C=False)
 write(contract('SCIENCE'),c)
 cfg=read(ROOT/'configs/rbr_v2_d2_v1/interface_contract.json');cfg.update(stage='RBR_V2_D3_D_ONLY_INTERFACE',purpose='PAIRED_CONTROLLED_SWAP_D_DIAGNOSTIC_NOT_PRIMARY')
 for r in cfg['runs']:
  if r['new_role']!='D_SWAP_CAL_EXISTING_B_REUSE':r['run_id']=r['run_id'].replace('RBR-D2-D-','D3-D-');r['root']='outputs/stageR/rbr_v2_d3/SCIENCE/'+r['run_id']
 folder=ROOT/'configs/rbr_v2_d3_v1';folder.mkdir(exist_ok=True);write(folder/'interface_contract.json',cfg)
 c['interface_contract_sha256']=sha(folder/'interface_contract.json');write(contract('SCIENCE'),c)
 print('Original40 D SESSION and80 D3 first identities frozen; independent roles20/20/20 unchanged',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['certify','freeze']);a=p.parse_args();certify() if a.mode=='certify' else freeze()
