#!/usr/bin/env python3
"""Read-only P2-Q1 export: consume stored verdicts; never re-run qualification."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tsb_direct_coast_p2_q1 import ROOT,OUT,Q0,F2,SCIENCE,LEDGER,AUTH,BINDING,RUNS,read,write,sha,specs


def csv_new(name,rows,fields):
    with (OUT/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def export():
    import numpy as np
    ledger=read(LEDGER)
    if not ledger['stopped']:raise PermissionError('Results export requires stopped ledger')
    if (OUT/'P2_Q1_Final_Qualification_Report_v1.md').exists():raise FileExistsError('One-shot read-only report exists')
    by_run={x['run_id']:x for x in ledger['attempts']};pair_reviews={x['pair_id']:x for x in ledger['pairs']}
    rows=[];matrix=[];fs=[];safety=[];exposure=[];rawhash={};unrounded=[];coast=[];all_skews=[]
    for source in sorted(SCIENCE.rglob('*')):
        if source.is_file():rawhash[str(source.relative_to(ROOT))]=sha(source)
    roster=list(csv.DictReader((Q0/'P2_Q0_Independent_Session_Roster_v1.csv').open()))
    for r in roster:
        pid=r['pair_id'];pair=pair_reviews.get(pid);arms=[];started=[]
        for name in ('BASELINE','TREATMENT'):
            run=pid+'-'+name;attempt=by_run.get(run);started.append(bool(attempt));root=SCIENCE/run
            science=read(root/'arm_science_review.json') if (root/'arm_science_review.json').is_file() else None
            official=read(root/'official_safety_adapter.json') if (root/'official_safety_adapter.json').is_file() else None
            completion=read(root/'P2_Q1_Child_Completion.json') if (root/'P2_Q1_Child_Completion.json').is_file() else None
            telemetry=[json.loads(x) for x in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()] if (root/'telemetry/actual_lqr_controller_telemetry.jsonl').is_file() else []
            trace=[json.loads(x) for x in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()] if (root/'trace/realized_current_ego.jsonl').is_file() else []
            mechanism=science['mechanism'] if science else None
            arm_status=attempt['status'] if attempt else 'NOT_RUN_AFTER_STOP'
            arms.append((arm_status,science))
            cr=[x for x in telemetry if 2100000<=x['episode_time_us']<3900000]
            if name=='TREATMENT' and telemetry:
                coast.append(dict(pair_id=pid,run_id=run,coast_control_count=len(cr),zero_count=sum(x['longitudinal_requested_acceleration']==0. for x in cr),
                    all_commands_reach_model=all(x['command_identity_reached_model'] for x in cr),all_states_advance=all(x['state_propagation_continues'] for x in cr),
                    maximum_model_filter_residual=max(abs(x['model_realized_acceleration']-x['model_expected_acceleration']) for x in telemetry),
                    first_coast_acceleration=cr[0]['model_realized_acceleration'],last_coast_acceleration=cr[-1]['model_realized_acceleration']))
            matrix.append(dict(pair_id=pid,session_id=r['session_id'],arm=name,attempt_status=arm_status,
                technical_status=completion['technical_status'] if completion else 'INCOMPLETE' if attempt else 'NOT_RUN',
                mechanism_status=('PASS' if mechanism['proposal_diagnostic_pass'] and not mechanism['reasons'] else 'FAIL') if mechanism else 'NOT_OBSERVED',
                qualified_phase_count=len(mechanism['brake_phases']) if mechanism else '',
                phases_json=json.dumps(mechanism['brake_phases']) if mechanism else '',reasons_json=json.dumps(mechanism['reasons']) if mechanism else '',
                diagnostic_gap_json=json.dumps(mechanism['near_zero_core_diagnostic_only']) if mechanism else '',
                actual_current_ego_states=len(trace) if attempt else '',controller_transitions=len(telemetry) if attempt else '',
                coast_zero_request_count=sum(x['longitudinal_requested_acceleration']==0. for x in cr) if name=='TREATMENT' and attempt else '',
                positive_command_count=sum(x['longitudinal_requested_acceleration']>0 for x in telemetry) if telemetry else '',
                minimum_actual_speed_mps=min(x['current_ego']['speed_mps'] for x in trace) if trace else '',
                cpu_seconds=attempt.get('cpu_seconds','') if attempt else '',
                artifact_manifest_sha256=sha(root/'execution_manifest.json') if (root/'execution_manifest.json').exists() else ''))
            safety.append(dict(pair_id=pid,session_id=r['session_id'],arm=name,attempt_status=arm_status,
                at_fault_collisions=official['canonical_payload']['collision']['number_of_all_at_fault_collisions_stat_value'] if official else '',
                drivable_area_compliance=official['canonical_payload']['drivable_area']['drivable_area_compliance_stat_value'] if official else '',
                safety_pass=official['frozen_arm_safety_pass'] if official else '',
                official_adapter_sha256=sha(root/'official_safety_adapter.json') if official else ''))
            if attempt:
                exposure.append(dict(pair_id=pid,session_id=r['session_id'],log_id=r['log_id'],scenario_token=r['scenario_token'],run_id=run,
                    prior_exposure_class=r['exposure_class_before_P2'],added_exposure='E2_BENCHMARK_ENGINEERING',
                    role='B_BENCHMARK_QUALIFICATION',attempt_status=arm_status,claim_before_child=True,
                    primary_outcome_exposure=False,E5=False,authorization_sha256=sha(AUTH)))
            if trace and len(trace)==80:
                v=np.array([x['current_ego']['speed_mps'] for x in trace]);t=np.array([(x['current_ego']['time_us']-trace[0]['current_ego']['time_us'])/1e6 for x in trace])
                xy=np.array([[x['current_ego']['rear_axle']['x'],x['current_ego']['rear_axle']['y']] for x in trace])
                # Transparency export only; official qualification is the stored F2 analyzer result.
                unrounded.append(dict(pair_id=pid,arm=name,mean_speed=float(v.mean()),end_minus_start_speed=float(v[-1]-v[0]),
                    path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),mean_abs_accel=float(np.abs(np.gradient(v,t,edge_order=2)).mean()),
                    scope='UNROUNDED_DESCRIPTIVE_EXPORT_NOT_REQUALIFICATION'))
            if (root/'actual_observation_audit.json').is_file():
                all_skews.extend(x['raw_minus_sim_us'] for x in read(root/'actual_observation_audit.json')['frames'])
        failed=any(s in ('SCIENTIFIC_FAILURE','INFRASTRUCTURE_FAILURE') for s,_ in arms)
        pair_status=('PASS' if pair['pair_pass'] else 'SCIENTIFIC_FAIL') if pair else ('SCIENTIFIC_FAIL' if any(s=='SCIENTIFIC_FAILURE' for s,_ in arms) else 'INFRASTRUCTURE_BLOCKED' if failed or any(started) else 'NOT_RUN_AFTER_STOP')
        f=pair['F_match'] if pair else None;extra=pair['retained_P1_balance_review'] if pair else None
        rows.append(dict(pair_id=pid,session_id=r['session_id'],log_id=r['log_id'],scenario_token=r['scenario_token'],
            baseline_technical_status='PASS' if arms[0][0] in ('COMPLETE_PASS','SCIENTIFIC_FAILURE') else arms[0][0],
            treatment_technical_status='PASS' if arms[1][0] in ('COMPLETE_PASS','SCIENTIFIC_FAILURE') else arms[1][0],
            baseline_mechanism=('PASS' if arms[0][1]['mechanism']['proposal_diagnostic_pass'] else 'FAIL') if arms[0][1] else 'NOT_OBSERVED',
            treatment_mechanism=('PASS' if arms[1][1]['mechanism']['proposal_diagnostic_pass'] else 'FAIL') if arms[1][1] else 'NOT_OBSERVED',
            coast_zero_request_count=next((x['zero_count'] for x in coast if x['pair_id']==pid),''),
            f_match_status=('PASS' if f['pass'] else 'FAIL') if f else 'NOT_EVALUATED_ARM_GATE_FAILED' if any(started) else 'NOT_RUN',
            f_match_four_values=json.dumps(f['absolute_delta']) if f else '',extra_balance=json.dumps(extra) if extra else '',
            official_safety=('PASS' if all(x[1] and x[1]['official_safety_pass'] for x in arms) else 'FAIL_OR_INCOMPLETE') if any(started) else 'NOT_RUN',
            overall_pair_status=pair_status,cpu_seconds=sum(by_run.get(pid+'-'+a,{}).get('cpu_seconds',0) for a in ('BASELINE','TREATMENT')),
            pair_review_sha256=sha(SCIENCE/(pid+'-pair_review.json')) if pair else '',
            stop_or_not_run_reason='' if pair_status=='PASS' else ledger['status']))
        for metric,caliper in dict(mean_speed=.1,end_minus_start_speed=.15,path_length=.8,mean_abs_accel=.02).items():
            fs.append(dict(pair_id=pid,session_id=r['session_id'],metric=metric,baseline=f['baseline'][metric] if f else '',
                treatment=f['treatment'][metric] if f else '',absolute_delta=f['absolute_delta'][metric] if f else '',caliper=caliper,
                margin=caliper-f['absolute_delta'][metric] if f else '',status=('PASS' if f['absolute_delta'][metric]<=caliper+1e-12 else 'FAIL') if f else rows[-1]['f_match_status']))
    fields=['pair_id','session_id','log_id','scenario_token','baseline_technical_status','treatment_technical_status','baseline_mechanism','treatment_mechanism','coast_zero_request_count','f_match_status','f_match_four_values','extra_balance','official_safety','overall_pair_status','cpu_seconds','pair_review_sha256','stop_or_not_run_reason']
    csv_new('P2_Q1_Pair_Results_v1.csv',rows,fields)
    csv_new('P2_Q1_Mechanism_Matrix_v1.csv',matrix,list(matrix[0]))
    csv_new('P2_Q1_FMatch_Results_v1.csv',fs,list(fs[0]))
    csv_new('P2_Q1_Safety_Results_v1.csv',safety,list(safety[0]))
    csv_new('P2_Q1_Exposure_Update_v1.csv',exposure,list(exposure[0]) if exposure else ['run_id','added_exposure'])
    csv_new('P2_Q1_Unrounded_Observed_Descriptors_v1.csv',unrounded,list(unrounded[0]) if unrounded else ['pair_id','arm','scope'])
    write(OUT/'P2_Q1_Execution_Ledger_v1.json',ledger,exclusive=True)
    write(OUT/'evidence/P2_Q1_Raw_Artifact_Index_v1.json',dict(artifact_count=len(rawhash),sha256=rawhash),exclusive=True)
    pair_complete=sum(all(by_run.get(r['pair_id']+'-'+a,{}).get('status') in ('COMPLETE_PASS','SCIENTIFIC_FAILURE') for a in ('BASELINE','TREATMENT')) for r in rows)
    summary=dict(status=ledger['status'],planned_pairs=20,started_pairs=sum(any(x['run_id'].startswith(r['pair_id']+'-') for x in ledger['attempts']) for r in rows),
        both_arms_technically_complete_pairs=pair_complete,formally_judged_pairs=len(pair_reviews),successful_pairs=sum(r['overall_pair_status']=='PASS' for r in rows),
        failed_pairs=sum(r['overall_pair_status']=='SCIENTIFIC_FAIL' for r in rows),infrastructure_blocked_pairs=sum(r['overall_pair_status']=='INFRASTRUCTURE_BLOCKED' for r in rows),
        not_run_pairs=sum(r['overall_pair_status']=='NOT_RUN_AFTER_STOP' for r in rows),attempts=len(ledger['attempts']),cpu_seconds=ledger['cpu_seconds_consumed'],
        treatment_coast=coast,observed_clock_skew_min_us=min(all_skews) if all_skews else None,observed_clock_skew_max_us=max(all_skews) if all_skews else None,
        complete_arms=sum(x.get('technical_status')=='PASS' for x in matrix),mechanism_pass_arms=sum(x['mechanism_status']=='PASS' for x in matrix),
        official_safety_pass_arms=sum(x['safety_pass'] is True for x in safety),first_stop=next((x for x in ledger['attempts'] if x['status']!='COMPLETE_PASS'),None),
        formal_fmatch_distributions={key:dict(min=min(x['F_match']['absolute_delta'][key] for x in pair_reviews.values()),max=max(x['F_match']['absolute_delta'][key] for x in pair_reviews.values()),
            minimum_margin=limit-max(x['F_match']['absolute_delta'][key] for x in pair_reviews.values())) for key,limit in dict(mean_speed=.1,end_minus_start_speed=.15,path_length=.8,mean_abs_accel=.02).items()} if pair_reviews else {},
        extra_balance=dict(max_negative_dose_delta=max(p['retained_P1_balance_review']['negative_dose_delta_mps'] for p in pair_reviews.values()),
            minimum_negative_dose_margin=.15-max(p['retained_P1_balance_review']['negative_dose_delta_mps'] for p in pair_reviews.values()),
            max_peak_difference=max(max(p['retained_P1_balance_review']['peak_differences_mps2']) for p in pair_reviews.values()),
            minimum_peak_margin=.2-max(max(p['retained_P1_balance_review']['peak_differences_mps2']) for p in pair_reviews.values())) if pair_reviews else {},
        raw_artifact_count=len(rawhash),original_denominator_preserved=True,scientific_reexecution=0)
    write(OUT/'evidence/P2_Q1_Actual_Summary_v1.json',summary,exclusive=True)
    text=f'''# P2-Q1 — 最终资格报告 v1

## A. Overall Verdict

`{ledger['status']}`。固定分母20 SESSION，已启动{summary['started_pairs']}对，双臂技术完整{pair_complete}对，正式pair判定{len(pair_reviews)}对，成功{summary['successful_pairs']}对，科学失败{summary['failed_pairs']}对，基础设施阻断{summary['infrastructure_blocked_pairs']}对，完全未运行{summary['not_run_pairs']}对。全部未执行成员保留在原分母，禁止替换/补跑。

## B. 科学执行与停止

仅{len(ledger['attempts'])}个原名单首次arm。每arm独立进程与新官方对象。CPU累计{ledger['cpu_seconds_consumed']:.9f}秒，预算57600秒；wall每arm1200秒；42仅累计天花板，技术retry授权0。账本stopped={ledger['stopped']}；最后状态与首次停止点见Execution Ledger和Actual Summary。不把提前停止已完成成功比例解释为20 SESSION总体通过率。

## C. 科学合同和原始证据

完全复用原F2 tracker/planner/model/recorder/lifecycle/analyzer。请求、阈值、median3+physical gradient测量、六位舍入四F及额外平衡均未变。正式F_match只读取已存pair review；arm门未通过不运行pair判定，CSV明确NOT_EVALUATED_ARM_GATE_FAILED。未执行指标留空，状态NOT_RUN_AFTER_STOP。未舍入描述符仅作透明导出，不重判资格。40arm/20pair矩阵为固定名单全集。

## D. 机制、安全与时钟

实际技术完整{summary['complete_arms']}arm，机制通过{summary['mechanism_pass_arms']}arm，官方安全通过{summary['official_safety_pass_arms']}arm；全部实际数值见Mechanism Matrix、Safety Results、Actual Summary及原始JSON。已运行观察偏差范围{summary['observed_clock_skew_min_us']}～{summary['observed_clock_skew_max_us']}微秒。每SESSION保留自己的冻结映射，实际serialized history逐帧和raw source核对。

`DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY`。同index离线回放保留原观察，不插值，不代表严格事件时间因果；碰撞、TTC及安全结论仅适用于此模型与回放条件。滑行能力为ABSTRACT_COAST_PROXY_ONLY，不证明真实踏板或轮端零扭矩。

## E. 数据暴露与防火墙

仅实际启动arm追加E2_BENCHMARK_ENGINEERING事件，保留原E1/E2历史；没有E5重标，没有未执行科学值。SESSION是独立资格单位，不声称独立driver/城市、Claim-A或Primary功效。无RBR训练、H30/RBR比较、BDD/MMD、Primary/V/C、参数搜索或F3。资格结果不得用于encoder/loss/checkpoint/seed、H30或Primary样本/成功标准选择。旧B1 20/20、C2 FAIL、F1和F2 Development PASS永久保留。

## F. 下一阶段

本轮停止。若全20对PASS，唯一建议为Scientific Owner审查资格结论及后续研究授权；若首个科学失败，唯一建议为Owner只读审查该首次失败；若技术阻断，唯一建议为Owner审查具体基础设施故障及是否另行授权修复/技术重试。均无自动恢复或额外执行。
'''
    (OUT/'P2_Q1_Final_Qualification_Report_v1.md').write_text(text)
    return summary


if __name__=='__main__':print(json.dumps(export(),indent=2))
