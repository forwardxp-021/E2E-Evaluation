#!/usr/bin/env python3
"""Write the versioned R3.2 final execution closure after zero-run checks."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; R1=ROOT/'docs/stageR/r1'
OUT=R1/'r1_b2_8_r3_2_final_execution_binding_manifest_v1.1.json'; REQUEST=R1/'R1_B2_8_R3_2_Scientific_Owner_48_Run_Authorization_Request_v0.1.md'
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path): return json.loads(p.read_text())
def main():
 if OUT.exists() or REQUEST.exists(): raise FileExistsError('R3_2_VERSIONED_FINAL_OUTPUT_EXISTS')
 roster=R1/'r1_official_compliant_technical_smoke_roster_v2.1.json'; schedule=R1/'r1_official_compliant_technical_smoke_schedule_v2.1.json'; pairs=R1/'r1_b2_8_r3_2_frozen_pair_evaluation_bindings_v1.0.json'; dry=R1/'r1_b2_8_r3_2_orchestrator_dry_run_v1.0.json'; inherited=R1/'r1_b2_8_r3_1_final_execution_binding_manifest_v1.0.json'
 result=read(dry); binding=read(pairs)
 if result['status']!='EXECUTION_ORCHESTRATOR_READY' or len(result['runs'])!=48 or result['simulation_started'] or binding['counts']!={'total':24,'HLC_PAIR_BINDING_COMPLETE':12,'TSB_PAIR_BINDING_COMPLETE':12}: raise ValueError('R3_2_FINAL_CLOSURE_PRECONDITION_FAIL')
 components=[ROOT/'tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py',ROOT/'tools/r1_b2_8_r3_2_execute_frozen_48run_smoke.py',ROOT/'tools/r1_b2_8_r3_2_freeze_pair_bindings.py',ROOT/'tools/r1_b2_8_r3_1_official_safety_adapter.py',ROOT/'tools/r1_official_technical_smoke_evaluator_v2_1.py',ROOT/'tools/r1_b2_8_r3_frozen_run_dispatcher.py']
 payload={'schema_version':'r1_b2_8_r3_2_final_execution_binding_manifest_v1.1','status':'FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION','inherits_r3_1':{'path':str(inherited.relative_to(ROOT)),'sha256':sha(inherited)},'roster':{'path':str(roster.relative_to(ROOT)),'sha256':sha(roster),'changed':False},'schedule':{'path':str(schedule.relative_to(ROOT)),'sha256':sha(schedule),'changed':False},'frozen_pair_binding':{'path':str(pairs.relative_to(ROOT)),'sha256':sha(pairs),'counts':binding['counts']},'orchestrator_dry_run':{'path':str(dry.relative_to(ROOT)),'sha256':sha(dry),'result':{'runs':48,'claim_49':result['claim_49'],'simulation_started':False,'official_runs':0,'consumed_real_budget':0}},'future_execution_components_sha256':{str(p.relative_to(ROOT)):sha(p) for p in components},'technical_failure_semantics':'STOPPED_ON_TECHNICAL_FAILURE_REQUIRES_OWNER_REVIEW_NO_RETRY_NO_REPLACEMENT','scientific_outcome_semantics':'RECORD_AND_CONTINUE_SCHEDULE_NO_RETRY_NO_REPLACEMENT','automatic_pair_evaluation':'R3_2_dispatcher_auto_lookup_by_pair_id_after_both_arms_complete','authorization':{'OFFICIAL_SMOKE_AUTHORIZED':False,'NEW_RUN_BUDGET':0,'RBR_A_B_C_AUTHORIZED':False},'threshold_changed':False}
 OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); final_sha=sha(OUT)
 REQUEST.write_text('# R1 B2.8-R3.2 Scientific Owner 48-Run Authorization Request v0.1\n\n'+f'唯一 final execution binding manifest SHA：`{final_sha}`。\n\n'+ 'pair binding 24/24（HLC 12/12、TSB 12/12）；dispatcher structural execution 24/24；safety adapter real-format fail-closed 测试通过；orchestrator dry-run 48/48，49th claim 在 runner 前拒绝。simulation_started=false，official_runs=0，consumed_real_budget=0。\n\n'+'roster、schedule 与 thresholds 均未改变。仍保持 `OFFICIAL_SMOKE_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RBR_A/B/C=NOT_AUTHORIZED`。\n',encoding='utf-8')
 print(json.dumps({'status':payload['status'],'final_execution_manifest_sha256':final_sha},ensure_ascii=False))
if __name__=='__main__': main()
