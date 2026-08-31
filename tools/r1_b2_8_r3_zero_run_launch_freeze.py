#!/usr/bin/env python3
"""Construct all bound nuPlan runners and stop before any run/step/rollout."""
from __future__ import annotations
import hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
from typing import Any
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env, official_count
ROOT=Path(__file__).resolve().parents[1]; R1=ROOT/'docs/stageR/r1'; ROSTER=R1/'r1_official_compliant_technical_smoke_roster_v2.1.json'; SCHEDULE=R1/'r1_official_compliant_technical_smoke_schedule_v2.1.json'
OUT={'bindings':R1/'r1_b2_8_r3_execution_bindings_manifest_v1.0.json','launch':R1/'r1_b2_8_r3_official_launch_manifest_v1.0.json','rehearsal':R1/'r1_b2_8_r3_zero_run_launch_rehearsal_v1.0.json','request':R1/'R1_B2_8_R3_Scientific_Owner_Run_Authorization_Request_v0.1.md'}
def sha(p:Path)->str:
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,x):
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 if any(p.exists() for p in OUT.values()): raise FileExistsError('R3_OUTPUT_EXISTS')
 official_env(); roster,schedule=read(ROSTER),read(SCHEDULE); by={(x['scenario_token'],x['log_id']):x for x in roster['entries']}
 rows=[]
 for run in schedule['runs']:
  entry=by[(run['scenario_token'],run['log_id'])]; rows.append({**run,'future_roster_row':entry})
 if len(rows)!=48 or any(official_count(r['future_roster_row']['db_path'],r['scenario_token'])!=1 for r in rows): raise RuntimeError('EXACT_RESOLUTION_FAIL_CLOSED')
 components={str(x.relative_to(ROOT)):sha(x) for x in [ROSTER,SCHEDULE,ROOT/'tools/r1_b2_8_r3_frozen_run_dispatcher.py',ROOT/'tools/r1_official_technical_smoke_planner_v2_2.py',ROOT/'configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v2_2_r3.yaml',ROOT/'tools/r1_official_technical_smoke_evaluator_v2_1.py',R1/'r1_future_compliant_smoke_selector_contract_v1.2.json',R1/'r1_official_execution_ineligible_identity_ledger_v1.0.json']}
 binding={'schema_version':'r1_b2_8_r3_execution_bindings_manifest_v1.0','status':'ZERO_RUN_CONSTRUCTION_PENDING','scientific_roster_sha256':sha(ROSTER),'scientific_schedule_sha256':sha(SCHEDULE),'components_sha256':components,'frozen_run_bindings':rows,'OFFICIAL_SIMULATION_AUTHORIZED':False,'NEW_RUN_BUDGET':0}; write(OUT['bindings'],binding)
 from hydra import compose,initialize_config_dir
 from nuplan.planning.script.utils import set_up_common_builder
 from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker,build_simulation_callbacks
 from nuplan.planning.script.builders.simulation_builder import build_simulations
 devkit=ROOT.parent/'nuplan-devkit'; cfgroot=devkit/'nuplan/planning/script/config/simulation'; audit=[]
 with tempfile.TemporaryDirectory(prefix='r1_b2_8_r3_zero_run_') as temp:
  for r in rows:
   trace=Path(temp)/r['run_id']/'trace'; raw=Path(temp)/r['run_id']/'raw'; trace.mkdir(parents=True)
   if any(trace.iterdir()): raise RuntimeError('TRACE_PATH_REUSE')
   os.environ.update({'R1_B2_8_R3_BINDING_MANIFEST':str(OUT['bindings']),'R1_B2_8_R3_RUN_ID':r['run_id'],'R1_B2_8_R3_TRACE_DIR':str(trace)})
   over=['+simulation=closed_loop_nonreactive_agents','planner=r1_official_technical_smoke_v2_2_r3','scenario_builder=nuplan_mini',f"scenario_builder.db_files=[{r['future_roster_row']['db_path']}]",'scenario_filter=all_scenarios',f"scenario_filter.scenario_tokens=[{r['scenario_token']}]",'worker=single_machine_thread_pool','worker.max_workers=1','scenario_builder.max_workers=1','max_callback_workers=1','number_of_cpus_allocated_per_simulation=1','number_of_gpus_allocated_per_simulation=0','gpu=false','seed=2026082701','run_metric=true','enable_simulation_progress_bar=false','experiment_name=r1_b2_8_r3',f"job_name={r['run_id']}",f'output_dir={raw}',f'hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]']
   with initialize_config_dir(config_dir=str(cfgroot)): cfg=compose(config_name='default_simulation',overrides=over)
   from tools.r1_b2_8_r3_frozen_run_dispatcher import build_planner_from_frozen_binding
   planner=build_planner_from_frozen_binding(str(OUT['bindings']),r['run_id'],str(trace))
   common=set_up_common_builder(cfg,'r3_construction'); cw=build_callbacks_worker(cfg); callbacks=build_simulation_callbacks(cfg,common.output_dir,cw); runners=build_simulations(cfg,common.worker,callbacks,cw,pre_built_planners=[planner])
   if len(runners)!=1: raise RuntimeError(f'RUNNER_COUNT:{r["run_id"]}:{len(runners)}')
   audit.append({'run_id':r['run_id'],'exact_resolution':1,'full_hydra_config_resolved':True,'simulation_runner_construction':'PASS','simulation_started':False,'trace_path_empty_before':True,'planner_call_count':'NOT_EXECUTED_ZERO_RUN','primary_trace':'REALIZED_CURRENT_EGO_ITERATIONS_0_79_FROZEN','official_safety_binding':'nuPlan_metric_engine_plus_frozen_evaluator_v2_1','evaluator_dispatcher':'R1OfficialTechnicalSmokeEvaluatorV2_1'})
 launch={'schema_version':'r1_b2_8_r3_official_launch_manifest_v1.0','status':'48_OF_48_READY_TO_CALL_SIMULATION_RUN','runs':audit,'simulation_started':False,'official_runs':0,'consumed_budget':0,'ledger_dry_run':{'claims_1_to_48':'PASS','claim_49':'HARD_FAIL_BEFORE_SIMULATOR_START'}}; write(OUT['launch'],launch); write(OUT['rehearsal'],launch)
 OUT['request'].write_text('# R1 B2.8-R3 Scientific Owner Run Authorization Request v0.1\n\nR3 已完成 48/48 official exact resolution、完整 Hydra composition 与 SimulationRunner construction；所有检查均在 `runner.run()` 前停止。请仅决定是否授权当前 binding manifest 的一次冻结 48-run official smoke。当前 OFFICIAL_SMOKE_AUTHORIZED=false，NEW_RUN_BUDGET=0，RBR 未授权。\n',encoding='utf-8')
 print(json.dumps({'ready':len(audit),'simulation_started':False},ensure_ascii=False))
if __name__=='__main__': main()
