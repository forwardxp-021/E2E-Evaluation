#!/usr/bin/env python3
"""Read-only cumulative analysis view over preserved linked ledgers; never an execution/budget authority."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from pathlib import Path
from unittest.mock import patch
from tools.rbr_v2_d2_download import ROOT,read,write,sha
OUT=ROOT/'outputs/rbr_v2_d3_v1';SCIENCE=ROOT/'outputs/stageR/rbr_v2_d3/SCIENCE'


def view():
 first=read(OUT/'SCIENCE_ledger.json');cont=read(OUT/'SCIENCE_continuation_ledger.json');cfg=read(ROOT/'docs/stageR/rbr_v2_d3_v1/SCIENCE_Protocol_v1.json');rec=read(OUT/'D13_existing_artifact_recovery.json')
 if not first['stopped'] or not cont['stopped']:raise PermissionError('Scientific execution still active')
 if sha(OUT/'SCIENCE_ledger.json')!=rec['original_ledger_sha256'] or sha(OUT/'SCIENCE_ledger.json')!=cont['original_stopped_ledger_sha256']:raise ValueError('Original stop ledger changed')
 attempts=first['attempts']+cont['attempts'];ids=[r['run_id'] for r in attempts]
 if len(set(ids))!=len(ids) or ids!=cfg['run_ids'][:len(ids)] or len(ids)>80:raise ValueError('Duplicate attempt, replacement or order change')
 completed=[];pairs=[];missing=[]
 for spec in cfg['arms']:
  root=SCIENCE/spec['run_id'];path=root/'completion.json'
  if not path.exists():missing.append(spec['run_id']);continue
  q=read(path);m=read(root/'execution_manifest.json')
  if q['manifest_sha256']!=sha(root/'execution_manifest.json') or any(sha(root/p)!=h for p,h in m['artifacts'].items()):raise ValueError('Scientific artifact changed '+spec['run_id'])
  completed.append(dict(run_id=spec['run_id'],scientific_pass=q['scientific_pass'],original_monitor_status=next(a['status'] for a in attempts if a['run_id']==spec['run_id']),
       recovered_from_existing_output=spec['run_id']=='D3-D-13-TREATMENT',completion_sha256=sha(path)))
 for spec in cfg['arms'][::2]:
  p=SCIENCE/(spec['pair_id']+'-pair_review.json')
  if p.exists():pairs.append(read(p))
 measured=first['cpu_seconds']+first['driver_cpu_seconds']+cont['cpu_seconds']+cont['driver_cpu_seconds'];charged=cont['prior_conservatively_charged_CPU_seconds']+cont['cpu_seconds_budget_charged']+cont['driver_cpu_seconds']
 if charged>14400:raise ValueError('Scientific conservative CPU ceiling exceeded')
 result=dict(view_only=True,not_an_execution_ledger_or_budget_reset=True,source_sha256={p:sha(OUT/p) for p in ['SCIENCE_ledger.json','SCIENCE_continuation_ledger.json','D13_existing_artifact_recovery.json']},
      attempts=attempts,pairs=pairs,completed_arm_integrity=completed,missing_arm_ids=missing,total_first_attempts=len(attempts),total_actual_complete_arms=len(completed),total_actual_complete_pairs=len(pairs),
      science_pairs_passed=sum(p['pair_pass'] for p in pairs),scientific_retry=0,one_monitor_event_preserved_and_recovered_without_rerun=True,
      observed_measured_CPU_seconds=measured,CPU_measured_scope='Observed parent/descendant and wait4 CPU; the original late-helper uncertainty is covered by the separate full-interval hardware upper charge, not falsely claimed as exact CPU.',
      conservative_scientific_CPU_budget_charge_seconds=charged,CPU_budget_seconds=14400,science_driver_wall_seconds=first['wall_seconds']+cont['wall_seconds'])
 write(OUT/'SCIENCE_analysis_view.json',result);print('Cumulative actual data:',len(completed),'arms,',len(pairs),'pairs,',sum(p['pair_pass'] for p in pairs),'pair PASS; first attempts',len(attempts),'CPU charged',charged,flush=True)


def statistics():
 import tools.rbr_v2_d3_interface as interface
 base=interface.read
 def routed(p):
  return base(OUT/'SCIENCE_analysis_view.json') if Path(p)==OUT/'SCIENCE_ledger.json' else base(p)
 # Only the ledger metadata input is combined. Frozen MMD, calibration, source groups and model are unchanged.
 with patch.object(interface,'read',routed):interface.statistics()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['view','statistics']);a=p.parse_args();view() if a.mode=='view' else statistics()
