#!/usr/bin/env python3
"""Freeze R3.3 recursive authorization closure without launching simulation."""
from __future__ import annotations
import hashlib,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R1=ROOT/'docs/stageR/r1'; OUT=R1/'r1_b2_8_r3_3_final_execution_binding_manifest_v1.2.json'; REQUEST=R1/'R1_B2_8_R3_3_Scientific_Owner_48_Run_Authorization_Request_v0.1.md'
sys.path.insert(0, str(ROOT))
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path):return json.loads(p.read_text())
def main():
 if OUT.exists() or REQUEST.exists():raise FileExistsError('R3_3_VERSIONED_OUTPUT_EXISTS')
 prior=R1/'r1_b2_8_r3_2_final_execution_binding_manifest_v1.1.json'; p=read(prior); inherited=p['inherits_r3_1']; components=dict(p['future_execution_components_sha256'])
 r33=ROOT/'tools/r1_b2_8_r3_3_execute_frozen_48run_smoke.py'; components[str(r33.relative_to(ROOT))]=sha(r33)
 payload={'schema_version':'r1_b2_8_r3_3_final_execution_binding_manifest_v1.2','status':'FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION','inherits_r3_2':{'path':str(prior.relative_to(ROOT)),'sha256':sha(prior)},'inherits_r3_1':inherited,'roster':p['roster'],'schedule':p['schedule'],'frozen_pair_binding':p['frozen_pair_binding'],'future_execution_components_sha256':components,'authorization_gate_semantics':'RECURSIVE_R3_1_PLUS_R3_2_SHA_CLOSURE_ENFORCED','authorization':{'OFFICIAL_SMOKE_AUTHORIZED':False,'NEW_RUN_BUDGET':0,'RBR_A_B_C_AUTHORIZED':False},'simulation_started':False,'official_runs':0,'consumed_real_budget':0,'threshold_changed':False}
 OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 from tools.r1_b2_8_r3_3_execute_frozen_48run_smoke import authorize
 with tempfile.TemporaryDirectory(prefix='r1_r3_3_owner_dry_') as temp:
  owner=Path(temp)/'owner.json'; owner.write_text(json.dumps({'OFFICIAL_SMOKE_AUTHORIZED':True,'final_execution_manifest_sha256':sha(OUT)}),encoding='utf-8'); inherited_count=authorize(OUT,owner)
 final_sha=sha(OUT); REQUEST.write_text('# R1 B2.8-R3.3 Scientific Owner 48-Run Authorization Request v0.1\n\n'+f'唯一 final manifest SHA：`{final_sha}`。\n\n'+f'R3.1 inherited manifest SHA、其 {inherited_count} 个 runtime components，以及 R3.2/R3.3 当前层 SHA closure 均已递归核验通过。roster、schedule、pair binding 与 threshold 未改变；simulation_started=false，official_runs=0，consumed_real_budget=0。\n\n'+ '`OFFICIAL_SMOKE_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RBR_A/B/C=NOT_AUTHORIZED`。\n',encoding='utf-8')
 print(json.dumps({'status':payload['status'],'final_execution_manifest_sha256':final_sha,'inherited_runtime_component_count':inherited_count},ensure_ascii=False))
if __name__=='__main__':main()
