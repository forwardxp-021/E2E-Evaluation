#!/usr/bin/env python3
"""Report frozen B1 resume without rerunning or re-adjudicating the original 17 pairs."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import b1_resume_execution as r
from tools import b1_tsb_qualification as frozen
from tools.s2r_production_executor import PROTECTED, PROTECTED_SHA, sha256_file as sha, write_manifest as write_json


def inherited_pairs():
    original = r.rows(frozen.OUT / 'B1_TSB_Qualification_Pair_Results_v1.csv')[:17]
    ir2 = {x['pair_id']: x for x in r.rows(r.ROOT / 'docs/stageR/b1_ir2_adjudication/B1_IR2_Existing17_Full_Contract_Status_v1.csv')}
    arm_safety = {x['arm_id']: x for x in r.rows(r.ROOT / 'docs/stageR/b1_ir2_adjudication/B1_IR2_Recovered_Safety_Arm_Results_v1.csv')}
    result = []
    for old in original:
        row = dict(old)
        proof = ir2[row['pair_id']]
        r.require(proof['scientific_status'] == 'SCIENTIFIC_PASS', 'INHERITED_IR2_SCIENCE_MISMATCH')
        row.update(technical_complete='true', safety_status='PASS', joint_scientific_status='PASS',
                   technical_failure_reason='', provenance='INHERITED_IR2_FULL_CONTRACT;ORIGINAL_MECHANISM_F_MATCH;NO_RECOMPUTATION',
                   inherited_ir2_artifact=str(r.ROOT / 'docs/stageR/b1_ir2_adjudication/B1_IR2_Existing17_Full_Contract_Status_v1.csv'))
        for arm in ('BASELINE', 'TREATMENT'):
            safe = arm_safety[row['pair_id'] + '-' + arm]
            row[arm.lower() + '_collision_count'] = safe['collision_count']
            row[arm.lower() + '_drivable_area'] = safe['drivable_value']
        row['F_match_four_descriptors_pass'] = 'true'  # Inherited all-caliper PASS; never recomputed.
        result.append(row)
    return result


def new_pair(roster, ledger):
    pair_id = roster['pair_id']
    arms = {a['run_id']: a for a in ledger['attempts']}
    base, treat = arms[roster['baseline_arm_id']], arms[roster['treatment_arm_id']]
    row = {key: roster[key] for key in ('pair_id', 'session_id', 'log_id', 'scenario_token', 'exposure_class')}
    row.update(baseline_arm_status=base['status'], treatment_arm_status=treat['status'], technical_complete='false',
               precontext_match='', route_match='', baseline_measurement_valid='', treatment_measurement_valid='',
               baseline_phase_count='', treatment_phase_count='', release_fraction='', second_peak_ratio='',
               F_match_status='NOT_AVAILABLE', safety_status='NOT_AVAILABLE', LOW_SPEED_ENDSTOP='',
               joint_mechanism_status='NOT_AVAILABLE', joint_scientific_status='TECHNICAL_INCOMPLETE',
               technical_failure_reason='', scientific_failure_reason='', provenance='B1_R_OWNER_AUTHORIZED_RESUME')
    from tools.r1_b2_8_r3_1_official_safety_adapter import adapt_official_safety
    for label, arm in [('baseline', base), ('treatment', treat)]:
        root = Path(arm['output_namespace'])
        if arm['status'] == 'ARM_COMPLETE':
            safety = adapt_official_safety(root)
            payload = safety['canonical_payload']
            row[label + '_collision_count'] = payload['collision']['number_of_all_at_fault_collisions_stat_value']
            row[label + '_drivable_area'] = str(payload['drivable_area']['drivable_area_compliance_stat_value']).lower()
    complete = all(a['status'] == 'ARM_COMPLETE' for a in (base, treat))
    if not complete:
        row['technical_failure_reason'] = json.dumps(ledger['stop'], ensure_ascii=False) if ledger['stop'] else 'NOT_RUN_AFTER_STOP_OR_PREFLIGHT'
        return row
    bm, tm = [r.load(Path(a['output_namespace']) / 'execution_manifest.json') for a in (base, treat)]
    row['precontext_match'] = str(bm['precontext_id'] == tm['precontext_id']).lower()
    row['route_match'] = str(bm['route_id'] == tm['route_id']).lower()
    if row['precontext_match'] != 'true' or row['route_match'] != 'true':
        row['technical_failure_reason'] = 'PAIR_IDENTITY_MISMATCH'
        return row
    row['technical_complete'] = 'true'
    row['safety_status'] = 'PASS' if all(row.get(label + '_collision_count') == 0 and row.get(label + '_drivable_area') == 'true' for label in ('baseline', 'treatment')) else 'SCIENTIFIC_FAIL'
    evaluation_path = r.OUT / 'pair_evaluations' / f'{pair_id}__evaluation.json'
    measurement_path = r.OUT / 'pair_evaluations' / f'{pair_id}__measurements.json'
    if measurement_path.is_file():
        partial = r.load(measurement_path)
        b, t = partial['BASELINE'], partial['TREATMENT']
        from tools.r1_context_mechanism_core import qualify_tsb_pair
        mechanism = qualify_tsb_pair(b, t)
        row.update(baseline_measurement_valid=str(b['status'] == 'OK').lower(), treatment_measurement_valid=str(t['status'] == 'OK').lower(),
                   baseline_phase_count=b.get('brake_phase_count'), treatment_phase_count=t.get('brake_phase_count'),
                   release_fraction=t.get('interstage_release_fraction'), second_peak_ratio=t.get('second_brake_peak_ratio'),
                   LOW_SPEED_ENDSTOP=str(b['status'] == 'LOW_SPEED_ENDSTOP' or t['status'] == 'LOW_SPEED_ENDSTOP').lower(),
                   joint_mechanism_status='PASS' if mechanism['pass'] else 'SCIENTIFIC_FAIL',
                   scientific_failure_reason=';'.join(mechanism['reasons']))
    if not evaluation_path.is_file():
        failure = r.OUT / 'pair_evaluations' / f'{pair_id}__measurement_failure.json'
        if failure.is_file():
            row.update(joint_scientific_status='MEASUREMENT_INVALID', baseline_measurement_valid='false',
                       treatment_measurement_valid='false', scientific_failure_reason=r.load(failure)['reason'])
        else:
            row['technical_failure_reason'] = str(ledger['stop'])
        return row
    ev, metrics = r.load(evaluation_path), r.load(measurement_path)
    b, t = metrics['BASELINE'], metrics['TREATMENT']
    fmatch, mechanism = ev['evaluation']['f_match'], ev['evaluation']['mechanism']
    low = b['status'] == 'LOW_SPEED_ENDSTOP' or t['status'] == 'LOW_SPEED_ENDSTOP'
    valid = b['status'] == t['status'] == 'OK'
    reasons = list(mechanism['reasons'])
    if not fmatch['pass']:
        reasons.append('F_MATCH_FAIL')
    if not ev['official_safety_pair_pass']:
        reasons.append('OFFICIAL_SAFETY_FAIL')
    if low:
        reasons.append('LOW_SPEED_ENDSTOP')
    row.update(baseline_measurement_valid=str(b['status'] == 'OK').lower(), treatment_measurement_valid=str(t['status'] == 'OK').lower(),
               baseline_phase_count=b.get('brake_phase_count'), treatment_phase_count=t.get('brake_phase_count'),
               release_fraction=t.get('interstage_release_fraction'), second_peak_ratio=t.get('second_brake_peak_ratio'),
               F_match_status=fmatch['status'], F_match_four_descriptors_pass=str(fmatch['pass']).lower(),
               F_match_pass_by_feature=json.dumps(fmatch['pass_by_feature'], sort_keys=True),
               F_match_full_evaluation=json.dumps(fmatch, sort_keys=True),
               safety_status='PASS' if ev['official_safety_pair_pass'] else 'SCIENTIFIC_FAIL',
               LOW_SPEED_ENDSTOP=str(low).lower(), joint_mechanism_status='PASS' if mechanism['pass'] else 'SCIENTIFIC_FAIL',
               joint_scientific_status='PASS' if valid and not reasons else ('SCIENTIFIC_FAIL' if valid or low or b['status'] == 'NO_BRAKE_PHASE' or t['status'] == 'NO_BRAKE_PHASE' else 'MEASUREMENT_INVALID'),
               scientific_failure_reason=';'.join(reasons), evaluation_sha256=sha(evaluation_path))
    return row


def summarize(pairs, ledger, preflight_verdict):
    technical = sum(p['technical_complete'] == 'true' for p in pairs)
    evaluated = sum(p['joint_mechanism_status'] in {'PASS', 'SCIENTIFIC_FAIL'} for p in pairs)
    safe_eval = sum(p['safety_status'] in {'PASS', 'SCIENTIFIC_FAIL'} for p in pairs)
    counts = {
        'technical_completeness': (technical, 20),
        'measurement_validity': (sum(p['baseline_measurement_valid'] == p['treatment_measurement_valid'] == 'true' for p in pairs), technical),
        'baseline_one_phase': (sum(p['baseline_measurement_valid'] == 'true' and str(p['baseline_phase_count']) == '1' for p in pairs), evaluated),
        'treatment_two_phase': (sum(p['treatment_measurement_valid'] == 'true' and str(p['treatment_phase_count']) == '2' and float(p['release_fraction'] or -1) >= 0.15 and float(p['second_peak_ratio'] or -1) >= 0.5 for p in pairs), evaluated),
        'joint_mechanism': (sum(p['joint_mechanism_status'] == 'PASS' for p in pairs), evaluated),
        'F_match': (sum(p['F_match_status'] == 'F_MATCH_PASS' for p in pairs), sum(p['F_match_status'] in {'F_MATCH_PASS', 'F_MATCH_FAIL'} for p in pairs)),
        'official_safety': (sum(p['safety_status'] == 'PASS' for p in pairs), safe_eval),
        'no_LOW_SPEED_ENDSTOP': (sum(p['LOW_SPEED_ENDSTOP'] == 'false' for p in pairs), sum(p['LOW_SPEED_ENDSTOP'] in {'true', 'false'} for p in pairs)),
        'full_contract_qualification': (sum(p['joint_scientific_status'] == 'PASS' for p in pairs), sum(p['joint_scientific_status'] in {'PASS', 'SCIENTIFIC_FAIL', 'MEASUREMENT_INVALID'} for p in pairs)),
    }
    n = counts['full_contract_qualification'][0]
    if preflight_verdict != 'B1_R_PREFLIGHT_PASS':
        verdict = 'B1_R_PREFLIGHT_BLOCKED'
    elif ledger['stop'] or technical < 20 or any(p['joint_scientific_status'] == 'TECHNICAL_INCOMPLETE' for p in pairs):
        verdict = 'B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE'
    elif n == 20:
        verdict = 'B1_TSB_BENCHMARK_QUALIFICATION_PASS'
    else:
        verdict = 'B1_TSB_BENCHMARK_QUALIFICATION_FAIL'
    new = sum(a['runner_entry_claimed'] for a in ledger['attempts'])
    return dict(verdict=verdict, historical_attempts=35, new_attempts=new, cumulative_attempts=35+new,
                unique_arms_complete=34+sum(a['status'] == 'ARM_COMPLETE' for a in ledger['attempts']),
                pairs_complete=technical, low_speed_endstop_count=sum(p['LOW_SPEED_ENDSTOP'] == 'true' for p in pairs),
                counts={k: {'n': n, 'planned_denominator': 20, 'evaluable_denominator': d,
                            'wilson95_full_denominator': list(frozen._wilson(n, 20))} for k, (n, d) in counts.items()},
                full_contract_wilson95=list(frozen._wilson(n, 20)),
                LOW_ORDER_NUISANCE_ELIMINATED='NOT_ESTABLISHED', TSB_CLEAN_RESIDUAL_TASK='NOT_ESTABLISHED',
                prohibitions={key: 'NOT_EXECUTED_NOT_AUTHORIZED' for key in r.PROHIBITIONS})


def audit():
    auth, ledger = r.load(r.AUTH), r.load(r.LEDGER)
    r.verify_authorization(auth)
    r.require(len(ledger['attempts']) == 6 and sum(a['runner_entry_claimed'] for a in ledger['attempts']) <= 6, 'BUDGET')
    pids = []
    for a in ledger['attempts']:
        if a['status'] != 'ARM_COMPLETE':
            continue
        root = Path(a['output_namespace'])
        manifest = r.load(root / 'execution_manifest.json')
        r.require(manifest['execution_status'] == 'ARM_COMPLETE' and manifest['lifecycle_states_complete'] == list(r.executor.LIFECYCLE_ORDER), 'ARM_LIFECYCLE')
        r.require(manifest['attempt_id'] == a['attempt_id'] and manifest['attempt_number'] == a['attempt_number'], 'ATTEMPT_IDENTITY')
        r.require(manifest['authorization_sha256'] == sha(r.AUTH), 'ARM_AUTH_HASH')
        spec = next(s for s in r.load(frozen.SPECS)['arms'] if s['run_id'] == a['run_id'])
        r.require(manifest['precontext_id'] == spec['precontext_id'] and manifest['route_id'] == spec['route_id'], 'ARM_FROZEN_IDENTITY')
        for name, expected in manifest['artifact_hashes'].items():
            r.require(sha(root / name) == expected, f'ARM_ARTIFACT_HASH:{name}')
        r.require(len((root / 'trace/realized_current_ego.jsonl').read_text().splitlines()) == 80, 'TRACE_80')
        r.require(len((root / 'telemetry/planner_transfer.jsonl').read_text().splitlines()) == 80, 'PLANNER_80')
        r.require(len((root / 'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()) == 79, 'RECORDER_79')
        pids.append(manifest['process_id'])
    r.require(len(set(pids)) == len(pids), 'PROCESS_REUSE')
    r.require(sha(PROTECTED) == PROTECTED_SHA, 'PROTECTED')
    result = {'status': 'PASS', 'authorization_sha256': sha(r.AUTH), 'historical_files_unchanged': len(auth['historical_artifact_hashes']),
              'new_attempts_claimed': sum(a['runner_entry_claimed'] for a in ledger['attempts']), 'source_config_data_dependency_hashes_match': True,
              'original_roster_pairs': 20, 'original_unique_arms': 40, 'isolated_completed_processes': pids,
              'protected_asset_sha256': sha(PROTECTED), 'firewall': {key: 0 for key in r.PROHIBITIONS}}
    write_json(r.OUT / 'B1_R_Postrun_Audit_v1.json', result)
    print(json.dumps(result), flush=True)
    return result


def finalize():
    r.require(not (r.OUT / 'B1_R_Manifest_v1.json').exists(), 'FINAL_REPORT_ALREADY_FROZEN')
    auth, ledger = r.load(r.AUTH), r.load(r.LEDGER)
    preflight = r.load(r.PREFLIGHT)
    audit_result = audit()
    roster = r.rows(frozen.ROSTER)
    pairs = inherited_pairs() + [new_pair(row, ledger) for row in roster[17:]]
    r.require(len(pairs) == 20 and len({p['pair_id'] for p in pairs}) == 20, 'FULL_DENOMINATOR')
    r.csv_write(r.OUT / 'B1_R_Final_Pair_Results_v1.csv', pairs)
    arm_rows = []
    archive = r.OUT / 'execution_manifests'
    archive.mkdir(exist_ok=True)
    for a in ledger['attempts']:
        root = Path(a['output_namespace'])
        source = root / 'execution_manifest.json'
        if source.is_file():
            shutil.copyfile(source, archive / (a['attempt_id'] + '.json'))
            manifest = r.load(source)
        else:
            manifest = {**a, 'execution_status': a['status'], 'reason': str(ledger['stop']) if ledger['stop'] else preflight['verdict']}
            write_json(archive / (a['attempt_id'] + '.json'), manifest)
        row = {**a, 'lifecycle_states_complete': json.dumps(manifest.get('lifecycle_states_complete', [])),
               'manifest_sha256': sha(archive / (a['attempt_id'] + '.json')), 'authorization_sha256': sha(r.AUTH),
               'source_git_commit': auth['source_git_commit'], 'config_hash': manifest.get('config_hash', ''),
               'tsb_config_sha256': sha(r.PARAMETERS), 'precontext_id': auth['precontext_hashes'][a['run_id']],
               'route_id': auth['route_identity_hashes'][a['run_id']]}
        safety = root / 'official_safety_adapter.json'
        if safety.is_file():
            sf = r.load(safety)
            row.update(collision_count=sf['canonical_payload']['collision']['number_of_all_at_fault_collisions_stat_value'],
                       drivable_area=sf['canonical_payload']['drivable_area']['drivable_area_compliance_stat_value'], safety_pass=sf['frozen_arm_safety_pass'])
        arm_rows.append(row)
    r.csv_write(r.OUT / 'B1_R_Resume_Arm_Results_v1.csv', arm_rows)
    exposure = []
    for row in roster:
        attempts = [a for a in ledger['attempts'] if a['run_id'].startswith(row['pair_id'] + '-') and a['runner_entry_claimed']]
        exposure.append(dict(pair_id=row['pair_id'], session_id=row['session_id'], prior_exposure=row['exposure_class'],
                             current_exposure='E2_BENCHMARK_ENGINEERING', role='B',
                             observed='INHERITED_B1_IR2_RESULTS' if row['pair_id'] in {p['pair_id'] for p in pairs[:17]} else ('B1_R_MECHANISM_MATCHING_SAFETY_AND_TECHNICAL_RESULTS' if attempts else 'ZERO_RUN_INFRASTRUCTURE_METADATA_ONLY_THIS_STAGE'),
                             new_attempts_consumed=len(attempts), decision_influenced='NONE_SCIENTIFIC_CONTRACT_FROZEN',
                             claim_A_unseen_generalization='NOT_SUPPORTED_BY_BENCHMARK_EXPOSED_SESSION',
                             claim_B_primary_contamination='NOT_AUTOMATIC;NO_RBR_H_BDD_OUTCOME_INFLUENCE',
                             future_V_C_membership='NOT_ASSIGNED;REQUIRES_CLAIM_SPECIFIC_OWNER_AUDIT',
                             policy='APPROVED_S1R_OWNER_AMENDMENT_V1'))
    r.csv_write(r.OUT / 'B1_R_Exposure_Update_v1.csv', exposure)
    summary = summarize(pairs, ledger, preflight['verdict'])
    write_json(r.OUT / 'B1_R_Summary_v1.json', summary)
    count_lines = '\n'.join(f"| {key} | {v['n']}/20 | {v['n']}/{v['evaluable_denominator']} | [{v['wilson95_full_denominator'][0]:.6f}, {v['wilson95_full_denominator'][1]:.6f}] |" for key, v in summary['counts'].items())
    pair_lines = '\n'.join(f"- `{p['pair_id']}`：{p['joint_scientific_status']}；mechanism={p['joint_mechanism_status']}；F_match={p['F_match_status']}；safety={p['safety_status']}；{p['scientific_failure_reason'] or p['technical_failure_reason']}" for p in pairs[17:])
    text = f'''# B1-R Final Qualification Report v1

正式判定：`{summary['verdict']}`。

原20独立 SESSION、40 unique arms 保持冻结，01–17 原机制/F_match 直接继承，安全和 full-contract 继承 IR2；不重新计算或运行。历史18 treatment ATTEMPT_1 的 infrastructure failure 永久保留。IR3 静态 route gate 只是工程准入。

Historical attempts = 35；new attempts = {summary['new_attempts']}；cumulative = {summary['cumulative_attempts']} ≤ 41；unique complete arms = {summary['unique_arms_complete']}/40；complete pairs = {summary['pairs_complete']}/20。

| 项目 | 原始分母 | 可评价样本 | 描述性 Wilson 95%（分母20） |
|---|---:|---:|---|
{count_lines}

LOW_SPEED_ENDSTOP observed events = {summary['low_speed_endstop_count']}/20；不可评价样本不算无 endstop。所有 full-contract 20/20 才通过；Wilson 不改变成功标准。

{pair_lines}

Infrastructure stop: `{ledger['stop']}`。

## 来源和身份

Owner 授权：`B1_R_Owner_Conditional_Authorization_v1.json`，SHA256 `{sha(r.AUTH)}`；执行源码 commit `{auth['source_git_commit']}`。授权列出所有 tracked tools/config、外部官方实现与环境文件、DB/map、科学合同和全部历史产物的真实 SHA256。原 roster/specs/pair bindings SHA：`{auth['original_roster_hash']}` / `{auth['original_arm_specs_hash']}` / `{auth['original_pair_bindings_hash']}`。

每个实际 arm 独立 OS 进程、planner/controller/tracker/callback/recorder/random state；复用唯一 B1 正式 runner site，官方 MetricFileCallback 收尾，官方 serializer、16 metric parquets、原 safety adapter、六步 lifecycle 与 artifact hash 全部验证。独立 attempt namespace 和账本不改写原35预算。原始失败或新失败均不被科学成功覆盖。

新 raw root：`{r.RAW}`；每次 execution manifest、resolved config、trace、recorder、metric、serializer 原样保留，所有文件 hash 见本阶段 manifest；日志和执行 manifest 归档随报告提交。历史来源：原 B1 → IR offline metrics → IR2 safety/full-contract → IR3 native route repair → 本轮有限授权。

## 暴露和科学边界

Exposure Update 按批准的 S1R 记录 SESSION × claim × observed × decision influence。本轮机制/F_match/安全暴露属于 B benchmark exposure，不自动等于 Primary 污染；没有调整 detector/encoder/H/BDD/roster/阈值，也未安排未来 V/C。

LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED。
TSB_CLEAN_RESIDUAL_TASK = NOT_ESTABLISHED。
RBR/H/BDD/MMD/z64/Primary/V/C 均未执行，训练或未来执行不获授权。本阶段不能证明 RBR 优于 H、所有传统特征失效、未见场景普遍100%成功或真实量产有效性。

## 验证与下一步

Postrun audit `{audit_result['status']}`；历史 {audit_result['historical_files_unchanged']} 文件 SHA 保持一致；受保护 CSV SHA `{PROTECTED_SHA}`。

PASS 时只进入 Scientific Owner review → S3 / RBR-V2 training protocol readiness；FAIL 时只评审 benchmark 科学适用性和论文主张；INCOMPLETE 时只报告基础设施、预算和合法可选路径。不得额外 attempt，不自动启动训练。完成后停止。
'''
    (r.OUT / 'B1_R_Final_Qualification_Report_v1.md').write_text(text, encoding='utf-8')
    manifest = {'stage': 'B1_R', 'verdict': summary['verdict'], 'authorization_sha256': sha(r.AUTH),
                'source_git_commit': auth['source_git_commit'], 'postrun_audit': audit_result,
                'artifact_sha256': {str(p.relative_to(r.ROOT)): sha(p) for p in sorted(r.OUT.rglob('*')) if p.is_file() and p.suffix != '.lock'},
                'raw_artifact_sha256': r.hashes(r.RAW.rglob('*')),
                'original_attempt_1_manifest': {'path': str(frozen.RUN_ROOT / TARGET_ORIGINAL / 'execution_manifest.json'),
                    'sha256': sha(frozen.RUN_ROOT / TARGET_ORIGINAL / 'execution_manifest.json')},
                'budget': {k: summary[k] for k in ('historical_attempts', 'new_attempts', 'cumulative_attempts', 'unique_arms_complete', 'pairs_complete')},
                'prohibitions': summary['prohibitions']}
    write_json(r.OUT / 'B1_R_Manifest_v1.json', manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


TARGET_ORIGINAL = 'B1-TSB-18-TREATMENT'
