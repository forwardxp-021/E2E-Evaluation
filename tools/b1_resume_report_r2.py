#!/usr/bin/env python3
"""Append R2 results to the immutable original17; never re-evaluate historical science."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import b1_resume_execution_r2 as x
from tools import b1_resume_report as inherited
from tools import b1_resume_execution as r


def new_pair(roster, ledger):
    selected = {a['run_id']: a for a in ledger['attempts']}
    base, treatment = selected[roster['baseline_arm_id']], selected[roster['treatment_arm_id']]
    row = {key: roster[key] for key in ('pair_id', 'session_id', 'log_id', 'scenario_token', 'exposure_class')}
    row.update(baseline_arm_status=base['status'], treatment_arm_status=treatment['status'], technical_complete='false',
        baseline_measurement_valid='', treatment_measurement_valid='', baseline_phase_count='', treatment_phase_count='',
        release_fraction='', second_peak_ratio='', LOW_SPEED_ENDSTOP='', precontext_match='', route_match='',
        joint_mechanism_status='NOT_AVAILABLE', F_match_status='NOT_AVAILABLE', safety_status='NOT_AVAILABLE',
        joint_scientific_status='TECHNICAL_INCOMPLETE', technical_failure_reason='', scientific_failure_reason='',
        provenance='B1_R2_OWNER_AUTHORIZED_FROZEN_RESUME')
    safeties = {}
    for label, a in [('baseline', base), ('treatment', treatment)]:
        path = Path(a['output_namespace']) / 'official_safety_adapter.json'
        if path.exists():
            safe = r.load(path)
            safeties[label] = safe['frozen_arm_safety_pass']
            row[label+'_collision_count'] = safe['canonical_payload']['collision']['number_of_all_at_fault_collisions_stat_value']
            row[label+'_drivable_area'] = str(safe['canonical_payload']['drivable_area']['drivable_area_compliance_stat_value']).lower()
            row[label+'_official_safety_sha256'] = r.sha(path)
    if len(safeties) == 2:
        row['safety_status'] = 'PASS' if all(safeties.values()) else 'SCIENTIFIC_FAIL'
    complete = all(a['status'] == 'ARM_COMPLETE' for a in (base, treatment))
    if complete:
        manifests = [r.load(Path(a['output_namespace'])/'execution_manifest.json') for a in (base, treatment)]
        row['precontext_match'] = str(manifests[0]['precontext_id'] == manifests[1]['precontext_id']).lower()
        row['route_match'] = str(manifests[0]['route_id'] == manifests[1]['route_id']).lower()
        complete = row['precontext_match'] == row['route_match'] == 'true'
    row['technical_complete'] = str(complete).lower()
    metrics_path = x.OUT/'pair_evaluations'/f"{row['pair_id']}__measurements.json"
    eval_path = x.OUT/'pair_evaluations'/f"{row['pair_id']}__evaluation.json"
    failure_path = x.OUT/'pair_evaluations'/f"{row['pair_id']}__measurement_failure.json"
    if metrics_path.exists():
        metrics = r.load(metrics_path)
        b, t = metrics['BASELINE'], metrics['TREATMENT']
        from tools.r1_context_mechanism_core import qualify_tsb_pair
        mechanism = qualify_tsb_pair(b,t)
        row.update(baseline_measurement_valid=str(b['status']=='OK').lower(), treatment_measurement_valid=str(t['status']=='OK').lower(),
            baseline_phase_count=b.get('brake_phase_count'), treatment_phase_count=t.get('brake_phase_count'),
            release_fraction=t.get('interstage_release_fraction'), second_peak_ratio=t.get('second_brake_peak_ratio'),
            LOW_SPEED_ENDSTOP=str(b['status']=='LOW_SPEED_ENDSTOP' or t['status']=='LOW_SPEED_ENDSTOP').lower(),
            joint_mechanism_status='PASS' if mechanism['pass'] else 'SCIENTIFIC_FAIL', scientific_failure_reason=';'.join(mechanism['reasons']))
    if not complete:
        row['technical_failure_reason'] = str(ledger['stop'] or 'PAIR_NOT_COMPLETE')
        return row
    if failure_path.exists():
        row.update(joint_scientific_status='MEASUREMENT_INVALID', baseline_measurement_valid='false', treatment_measurement_valid='false',
                   scientific_failure_reason=r.load(failure_path)['reason'])
    elif eval_path.exists():
        if not metrics_path.exists():
            row['technical_failure_reason'] = 'R2_MEASUREMENT_ARTIFACT_MISSING'
            return row
        evaluation = r.load(eval_path)
        fmatch, mechanism = evaluation['evaluation']['f_match'], evaluation['evaluation']['mechanism']
        reasons = list(mechanism['reasons'])
        if not fmatch['pass']:
            reasons.append('F_MATCH_FAIL')
        if not evaluation['official_safety_pair_pass']:
            reasons.append('OFFICIAL_SAFETY_FAIL')
        if row['LOW_SPEED_ENDSTOP'] == 'true':
            reasons.append('LOW_SPEED_ENDSTOP')
        valid = row['baseline_measurement_valid'] == row['treatment_measurement_valid'] == 'true'
        low_or_no_phase = row['LOW_SPEED_ENDSTOP'] == 'true' or b['status']=='NO_BRAKE_PHASE' or t['status']=='NO_BRAKE_PHASE'
        row.update(F_match_status=fmatch['status'], F_match_four_descriptors_pass=str(fmatch['pass']).lower(),
            F_match_full_evaluation=json.dumps(fmatch, sort_keys=True), F_match_pass_by_feature=json.dumps(fmatch['pass_by_feature'], sort_keys=True),
            joint_mechanism_status='PASS' if mechanism['pass'] else 'SCIENTIFIC_FAIL',
            safety_status='PASS' if evaluation['official_safety_pair_pass'] else 'SCIENTIFIC_FAIL',
            joint_scientific_status='PASS' if valid and not reasons else ('SCIENTIFIC_FAIL' if valid or low_or_no_phase else 'MEASUREMENT_INVALID'),
            scientific_failure_reason=';'.join(reasons), evaluation_sha256=r.sha(eval_path))
    else:
        row['technical_failure_reason'] = str(ledger['stop'] or 'PAIR_EVALUATION_MISSING')
    return row


def audit(auth, ledger):
    x.verify_authorization(auth)
    pids = []
    r.require(len(ledger['attempts']) == 6 and sum(a['runner_entry_claimed'] for a in ledger['attempts']) <= 6, 'R2_FINAL_BUDGET')
    for a in ledger['attempts']:
        if a['status'] != 'ARM_COMPLETE':
            continue
        root = Path(a['output_namespace'])
        manifest = r.load(root/'execution_manifest.json')
        r.require(manifest['execution_status']=='ARM_COMPLETE' and manifest['lifecycle_states_complete']==list(r.executor.LIFECYCLE_ORDER), 'R2_FINAL_LIFECYCLE')
        r.require(manifest['authorization_sha256']==r.sha(x.AUTH) and manifest['attempt_id']==a['attempt_id']
                  and manifest['attempt_number']==a['attempt_number'], 'R2_FINAL_ATTEMPT_IDENTITY')
        spec = next(s for s in r.load(r.frozen.SPECS)['arms'] if s['run_id']==a['run_id'])
        r.require(all(manifest[key]==spec[key] for key in ('precontext_id','route_id','session_id','scenario_token')), 'R2_FINAL_FROZEN_IDENTITY')
        for name, expected in manifest['artifact_hashes'].items():
            r.require(r.sha(root/name)==expected, 'R2_FINAL_ARTIFACT_HASH:'+name)
        pids.append(manifest['process_id'])
    r.require(len(pids)==len(set(pids)), 'R2_PROCESS_REUSE')
    return {'status':'PASS', 'completed_process_ids':pids, 'historical_files_unchanged':len(auth['historical_artifact_hashes']),
            'protected_asset_sha256':r.sha(r.PROTECTED), 'source_config_dependency_data_hashes_valid':True,
            'new_claimed_attempts':sum(a['runner_entry_claimed'] for a in ledger['attempts'])}


def finalize():
    r.require(not (x.OUT/'B1_R2_Manifest_v1.json').exists(), 'R2_FINAL_REPORT_ALREADY_FROZEN')
    auth, ledger, preflight = r.load(x.AUTH), r.load(x.LEDGER), r.load(x.PREFLIGHT)
    try:
        final_audit = audit(auth,ledger)
    except Exception as exc:
        final_audit = {'status':'FAIL','reason':f'{type(exc).__name__}:{exc}'}
        ledger = dict(ledger,stop=ledger['stop'] or {'reason': 'POSTRUN_AUDIT_FAILURE:'+str(exc)})
    r.write_json(x.OUT/'B1_R2_Final_Audit_v1.json', final_audit)
    roster = r.rows(r.frozen.ROSTER)
    pairs = inherited.inherited_pairs() + [new_pair(row,ledger) for row in roster[17:]]
    r.require(len(pairs)==20 and len({p['session_id'] for p in pairs})==20, 'R2_FINAL_20_DENOMINATOR')
    r.csv_write(x.OUT/'B1_R2_Final_Pair_Results_v1.csv', pairs)
    summary = inherited.summarize(pairs,ledger,'B1_R_PREFLIGHT_PASS' if preflight['verdict']=='B1_R2_PREFLIGHT_PASS' else 'B1_R_PREFLIGHT_BLOCKED')
    if preflight['verdict'] != 'B1_R2_PREFLIGHT_PASS':
        summary['verdict']='B1_R2_PREFLIGHT_BLOCKED'
    r.write_json(x.OUT/'B1_R2_Summary_v1.json',summary)
    arm_results=[]
    archive=x.OUT/'execution_manifests'
    archive.mkdir(exist_ok=True)
    for a in ledger['attempts']:
        root=Path(a['output_namespace'])
        actual=root/'execution_manifest.json'
        target=archive/(a['attempt_id']+'.json')
        if actual.exists():
            shutil.copyfile(actual,target)
            manifest=r.load(actual)
        else:
            manifest={**a,'execution_status':'NOT_RUN','reason':str(ledger['stop'] or preflight['failure'])}
            r.write_json(target,manifest)
        result={**a,'manifest_sha256':r.sha(target),'source_git_commit':auth['source_git_commit'],
            'authorization_sha256':r.sha(x.AUTH),'config_hash':manifest.get('config_hash',''),
            'lifecycle_states':json.dumps(manifest.get('lifecycle_states_complete',[])),
            'precontext_id':auth['precontext_hashes'][a['run_id']], 'route_id':auth['route_identity_hashes'][a['run_id']]}
        safety=root/'official_safety_adapter.json'
        if safety.exists():
            safe=r.load(safety)
            result.update(collision_count=safe['canonical_payload']['collision']['number_of_all_at_fault_collisions_stat_value'],
                drivable_area=safe['canonical_payload']['drivable_area']['drivable_area_compliance_stat_value'],official_safety_pass=safe['frozen_arm_safety_pass'])
        arm_results.append(result)
    r.csv_write(x.OUT/'B1_R2_Arm_Results_v1.csv',arm_results)
    safety_rows=[]
    for p in pairs:
        safe={key:p[key] for key in ('pair_id','session_id','safety_status','provenance')}
        for label in ('baseline','treatment'):
            for field in ('collision_count','drivable_area'):
                safe[label+'_'+field]=p.get(label+'_'+field,'NOT_AVAILABLE')
        safety_rows.append(safe)
    r.csv_write(x.OUT/'B1_R2_Official_Safety_Summary_v1.csv',safety_rows)
    exposure=[]
    for p in pairs:
        new=[a for a in ledger['attempts'] if a['run_id'].startswith(p['pair_id']+'-') and a['runner_entry_claimed']]
        exposure.append({'session_id':p['session_id'],'pair_id':p['pair_id'],'prior_exposure':p['exposure_class'],
            'new_exposure':'E2_BENCHMARK_ENGINEERING' if new else 'NO_NEW_SCIENTIFIC_EXPOSURE_THIS_STAGE',
            'role':'B','claim':'FROZEN_TSB_BENCHMARK_QUALIFICATION','observed':p['joint_scientific_status'] if new else 'INHERITED_OR_NOT_RUN',
            'new_attempts':len(new),'decision_influenced':'NONE_FROZEN_CONTRACT_UNCHANGED',
            'RBR_vs_H_Primary_outcome_exposure':'NO','future_V_C_membership':'NOT_ASSIGNED;CLAIM_SPECIFIC_OWNER_AUDIT_REQUIRED',
            'policy':'APPROVED_S1R_OWNER_AMENDMENT_V1'})
    r.csv_write(x.OUT/'B1_R2_Exposure_Ledger_Update_v1.csv',exposure)
    counts='\n'.join(f"| {key} | {v['n']}/20 | {v['n']}/{v['evaluable_denominator']} | [{v['wilson95_full_denominator'][0]:.6f}, {v['wilson95_full_denominator'][1]:.6f}] |" for key,v in summary['counts'].items())
    detail='\n'.join(f"- {p['pair_id']}: {p['joint_scientific_status']}; mechanism={p['joint_mechanism_status']}; F_match={p['F_match_status']}; safety={p['safety_status']}; technical={p['technical_failure_reason']}; scientific={p['scientific_failure_reason']}" for p in pairs[17:])
    auth_commit=x.committed_record(x.AUTH)
    report=f'''# B1-R2 Final Qualification Report v1

正式判定：`{summary['verdict']}`。

Owner指令SHA `{r.sha(x.OWNER)}`；授权SHA `{r.sha(x.AUTH)}`。Executable source base `{auth['source_git_commit']}`；authorization record commit `{auth_commit}`。元数据提交改变HEAD不改变已冻结源码基准；逐文件hash/Git blob是权威绑定。

最终真实零运行预检：{len(preflight['results'])}/6 PASS（若verdict为BLOCKED则未完成全部准入），预检状态 `{preflight['verdict']}`。实际执行严格按18T2、18B1、19T1、19B1、20B1、20T1顺序，每arm独立OS进程，复用唯一正式B1 runner/六步生命周期、RP实际回调检查和官方安全adapter。没有修改TSB/参数/阈值。

Historical attempts=35；new attempts={summary['new_attempts']}；cumulative={summary['cumulative_attempts']}≤41；unique complete arms={summary['unique_arms_complete']}/40；complete pairs={summary['pairs_complete']}/20。原18 Treatment ATTEMPT_1 infrastructure failure永久保留，没有ATTEMPT_3或其他第二次尝试。

| 项目 | 固定分母 | 可评价分母 | 描述性Wilson95（分母20） |
|---|---|---|---|
{counts}

LOW_SPEED_ENDSTOP observed={summary['low_speed_endstop_count']}/20。未完成/不可评价成员不当作无endstop或科学PASS。只有ALL_20_PAIRS_PASS_FROZEN_JOINT_QUALIFICATION才通过，Wilson不改变标准。

{detail}

Infrastructure stop: `{ledger['stop']}`。科学失败和基础设施失败分别记录；未完成时不声称20对已完整评价。原01–17机制/F_match来自原冻结报告，official safety/full-contract来自IR2，直接继承而非重新判定。历史B1→IR→IR2→IR3→R→RP→R2完整来源均保留。

所有输入身份、配置、原始trace、passive recorder、官方parquet、官方simulation-log、独立execution manifests和日志位于 `{x.RAW}` 与本目录归档；manifest绑定全部实际文件hash。技术缺失不在本轮自行恢复、不修改代码后继续。

Final hash audit `{final_audit['status']}`；保护CSV SHA `{r.PROTECTED_SHA}`。科学源码、依赖、关键数据和历史文件按新授权逐项核对。

Exposure按SESSION×claim×observed×decision influence记录。B benchmark机制/安全暴露不自动成为RBR-vs-H Primary污染；没有改变未来V/C成员、encoder、H、kernel或Primary规则。

LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED；TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED。B1不能证明RBR>H、RBR能检测TSB、所有人工特征无效或真实量产监控有效。

RBR训练/推理、H、BDD/MMD/z64、Primary、V/C均未执行且未获授权。PASS时只建议Owner review→RBR-V2 training-entry audit→U-only training authorization decision；FAIL时Owner benchmark scientific review；INCOMPLETE时Owner review of technical blocker and remaining attempt authority。完成后停止，不开启额外实验或工程循环。
'''
    (x.OUT/'B1_R2_Final_Qualification_Report_v1.md').write_text(report,encoding='utf-8')
    manifest={'stage':x.STAGE,'verdict':summary['verdict'],'source_git_commit':auth['source_git_commit'],
        'authorization_record_commit':auth_commit,'authorization_sha256':r.sha(x.AUTH),'final_audit':final_audit,
        'budget':{key:summary[key] for key in ('historical_attempts','new_attempts','cumulative_attempts','unique_arms_complete','pairs_complete')},
        'original_18_treatment_attempt_1':{'path':str(r.frozen.RUN_ROOT/'B1-TSB-18-TREATMENT/execution_manifest.json'),
            'sha256':r.sha(r.frozen.RUN_ROOT/'B1-TSB-18-TREATMENT/execution_manifest.json')},
        'artifact_sha256':{str(p.relative_to(ROOT)):r.sha(p) for p in sorted(x.OUT.rglob('*')) if p.is_file() and p.suffix!='.lock'},
        'raw_artifact_sha256':r.hashes(x.RAW.rglob('*')),'scientific_prohibitions':summary['prohibitions']}
    r.write_json(x.OUT/'B1_R2_Manifest_v1.json',manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


ROOT=r.ROOT
