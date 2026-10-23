#!/usr/bin/env python3
"""Versioned Owner-authorized R2 accounting around the unchanged B1 production stack."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import b1_resume_execution as r
from tools import b1_resume_preflight_correction as rp
from tools import b1_production_preflight as gate
from tools.s2r_production_executor import build_fresh_arm, write_manifest as write_json

ROOT = r.ROOT
OUT = ROOT / 'docs/stageR/b1_resume_execution_r2'
RAW = ROOT / 'outputs/stageR/b1_resume_execution_r2_v1'
AUTH = OUT / 'B1_R2_Owner_Authorization_v1.json'
LEDGER = OUT / 'B1_R2_Execution_Attempt_Ledger_v1.json'
PREFLIGHT = OUT / 'B1_R2_Preflight_Evidence_v1.json'
OWNER = Path('/Users/liuqing/.codex/attachments/bde69ced-8ee5-4edc-845d-a1e0342d9650/已粘贴的文本.txt')
STAGE = 'B1_R2_FROZEN_SIX_ATTEMPT_RESUME'
NEW_SOURCE = ['tools/b1_resume_execution_r2.py', 'tools/b1_resume_report_r2.py', 'tests/test_b1_resume_execution_r2.py']


def contracts():
    return [{**a, 'output_namespace': str(RAW / a['run_id'] / f"ATTEMPT_{a['attempt_number']}")}
            for a in r.attempt_contract()]


def historical_paths():
    return [*(ROOT / p for p in rp.history_hashes()), *rp.OUT.rglob('*'), *rp.RAW.rglob('*')]


def verify_hash_group(group, label):
    for name, expected in group.items():
        path = ROOT / name
        r.require(path.is_file() and r.sha(path) == expected, f'HASH_MISMATCH:{label}:{name}')


def committed_record(path):
    relative = str(path.relative_to(ROOT))
    commit = r.git('log', '-1', '--format=%H', '--', relative)
    r.require(commit and not r.git('diff', 'HEAD', '--', relative), 'RECORD_NOT_COMMITTED:' + relative)
    r.require(r.executor._git_blob_matches(commit, relative, r.sha(path)), 'COMMITTED_RECORD_HASH:' + relative)
    return commit


def prepare():
    r.require(not OUT.exists() and not RAW.exists(), 'R2_FRESH_STAGE_REQUIRED;NO_AUTH_REBIND')
    r.require(r.git('branch', '--show-current') == r.BRANCH, 'R2_BRANCH')
    r.require(not r.git('diff', 'HEAD', '--', 'tools', 'configs', *NEW_SOURCE), 'DIRTY_EXECUTABLE_SOURCE')
    draft_path = rp.OUT / 'B1_RP_Proposed_B1_R2_Authorization_v1.json'
    draft = r.load(draft_path)
    # Verify every inherited executable/environment/data binding before conversion.
    for key in ('source_file_hashes', 'scientific_contract_hashes', 'dependency_file_hashes', 'critical_data_hashes'):
        verify_hash_group(draft[key], 'RP_' + key)
    rp.verify_inherited_inputs(r.load(r.AUTH))
    rp_manifest = r.load(rp.OUT / 'B1_RP_Manifest_v1.json')
    r.require(rp_manifest['status'] == rp.STATUS_PASS, 'RP_NOT_READY')
    verify_hash_group(rp_manifest['artifact_sha256'], 'RP_ARTIFACTS')
    config_hashes = {}
    for item in contracts():
        evidence = r.load(rp.OUT / 'zero_run_evidence' / f"{item['order']:02d}_{item['run_id']}.json")
        config_hashes[item['attempt_id']] = evidence['normalized_config_hash']
    source_commit = r.git('rev-parse', 'HEAD')
    source = dict(draft['source_file_hashes'])
    for name in NEW_SOURCE:
        source[name] = r.sha(ROOT / name)
    for name, expected in source.items():
        r.require(r.executor._git_blob_matches(source_commit, name, expected), 'UNCOMMITTED_SOURCE:' + name)
    auth = copy.deepcopy(draft)
    auth.update(stage=STAGE, authorization_status='OWNER_AUTHORIZED_CONDITIONAL_ON_FINAL_PREFLIGHT',
        AUTHORIZATION_STATUS='OWNER_AUTHORIZED_CONDITIONAL_ON_FINAL_PREFLIGHT', OWNER_APPROVAL='APPROVED',
        EXECUTION='EXACT_SIX_ATTEMPTS_ONLY_AFTER_FINAL_PREFLIGHT_PASS',
        source_git_commit=source_commit, authorized_source_git_sha=source_commit,
        EXECUTABLE_SOURCE_BASE_COMMIT=source_commit,
        authorization_record_commit_locator={'method': 'git log -1 --format=%H -- AUTHORIZATION_PATH',
            'path': str(AUTH.relative_to(ROOT)), 'resolved_in': 'preflight evidence and each attempt manifest'},
        owner_decision_provenance={'instruction_path': str(OWNER), 'instruction_sha256': r.sha(OWNER),
            'scope': 'B1_R2_EXACT_SIX_ATTEMPTS_ONLY', 'conditional_on_all_final_preflight_checks': True},
        authorization_source={'path': str(OWNER), 'sha256': r.sha(OWNER)},
        owner_authorization_scope='EXACT_SIX_ATTEMPTS_AFTER_FINAL_PREFLIGHT;NO_OTHER_RETRY',
        source_file_hashes=source, resolved_configuration_hashes=config_hashes,
        resolved_configuration_hash_contract='RP normalized_config_hash; output root and generated administrative timestamps only',
        allowed_execution_attempts=contracts(), approved_attempt_ids=[x['attempt_id'] for x in contracts()],
        approved_attempt_numbers={x['run_id']: x['attempt_number'] for x in contracts()},
        approved_execution_order=[x['run_id'] for x in contracts()],
        original_precontext_and_route_identity={'precontext': draft['precontext_hashes'], 'route': draft['route_identity_hashes']},
        historical_budget_state={'attempts': 35, 'complete_arms': 34, 'complete_pairs': 17,
            'original_ledger_path': str(r.frozen.RUN_ROOT / 'B1_Scientific_Arm_Budget_Ledger_v1.json'),
            'original_ledger_sha256': r.sha(r.frozen.RUN_ROOT / 'B1_Scientific_Arm_Budget_Ledger_v1.json')},
        new_budget_ceiling=6, cumulative_budget_ceiling=41, new_execution_budget_granted=6,
        current_stage_status='B1_R2_CONDITIONAL_AWAITING_FINAL_PREFLIGHT',
        maximum_new_attempts=6, maximum_cumulative_attempts=41,
        historical_artifact_hashes=r.hashes(historical_paths()),
        proposal_sha256=r.sha(draft_path), previous_rp_manifest_sha256=r.sha(rp.OUT / 'B1_RP_Manifest_v1.json'))
    OUT.mkdir(parents=True)
    write_json(AUTH, auth)
    write_json(LEDGER, {'stage': STAGE, 'authorization_sha256': r.sha(AUTH), 'historical_attempts': 35,
        'maximum_new_attempts': 6, 'maximum_cumulative_attempts': 41, 'stop': None,
        'attempts': [{**a, 'status': 'NOT_RUN', 'runner_entry_claimed': False} for a in contracts()]})
    (OUT / 'B1_R2_Authorization_Audit_v1.md').write_text(
        '# B1-R2 Authorization Audit v1\n\n'
        + f'Owner instruction SHA256 `{r.sha(OWNER)}`; authorization SHA256 `{r.sha(AUTH)}`.\n\n'
        + f'EXECUTABLE_SOURCE_BASE_COMMIT = `{source_commit}`. Inherited409 bindings remain unchanged; '
        + 'R2 adds only a versioned authorization/accounting/report entry around the same factory, sole B1 runner/lifecycle and frozen evaluators.\n\n'
        + 'AUTHORIZATION_RECORD_COMMIT is resolved from committed authorization path history, recorded separately in final preflight and each attempt. '
        + 'It is not embedded back into its own file. Later documentation commits do not change the source base. Source is validated by exact file hashes and Git blobs, never HEAD equality.\n\n'
        + 'Authorization permits only the original ordered six attempts; cumulative ceiling41. Original17 outcomes and all old budgets/failures remain immutable. '
        + 'Any final preflight or runtime infrastructure failure terminates this stage; no repair, rebind, rerun or additional retry. Scientific failure retains denominator and continues.\n', encoding='utf-8')
    print('B1_R2_AUTHORIZATION_CREATED; source=' + source_commit, flush=True)


def verify_authorization(auth):
    r.require(auth.get('stage') == STAGE and auth.get('authorization_status') == 'OWNER_AUTHORIZED_CONDITIONAL_ON_FINAL_PREFLIGHT'
              and auth.get('OWNER_APPROVAL') == 'APPROVED', 'R2_AUTH_NOT_ACTIVE')
    r.require(auth['allowed_execution_attempts'] == contracts(), 'R2_EXACT_ATTEMPT_SCOPE')
    r.require(auth['approved_attempt_ids'] == [a['attempt_id'] for a in contracts()]
              and auth['approved_execution_order'] == [a['run_id'] for a in contracts()]
              and auth['approved_attempt_numbers'] == {a['run_id']: a['attempt_number'] for a in contracts()}, 'R2_APPROVED_ORDER')
    r.require(auth['maximum_new_attempts'] == auth['new_budget_ceiling'] == 6
              and auth['maximum_cumulative_attempts'] == auth['cumulative_budget_ceiling'] == 41, 'R2_BUDGET_CONTRACT')
    r.require(auth['scientific_prohibitions'] == r.PROHIBITIONS and auth['stop_policy'] == [
        'COMPLETE_PRE_REGISTERED_ROSTER_ON_SCIENTIFIC_FAILURE', 'STOP_ON_INFRASTRUCTURE_FAILURE'], 'R2_FIREWALL_STOP_CONTRACT')
    r.require(auth['retry_contract']['only_authorized_retry'] == r.TARGETS[0][0]
              and auth['retry_contract']['authorized_attempt'] == 2, 'R2_RETRY_CONTRACT')
    r.require(r.git('branch', '--show-current') == auth['branch'] == r.BRANCH, 'R2_BRANCH')
    r.require(r.sha(OWNER) == auth['owner_decision_provenance']['instruction_sha256'], 'R2_OWNER_PROVENANCE')
    inherited = r.load(rp.OUT / 'B1_RP_Proposed_B1_R2_Authorization_v1.json')['source_file_hashes']
    r.require(set(inherited).union(NEW_SOURCE).issubset(auth['source_file_hashes']), 'R2_SOURCE_COVERAGE')
    for name, expected in inherited.items():
        r.require(auth['source_file_hashes'][name] == expected, 'R2_INHERITED_SOURCE_REBOUND:' + name)
    for key in ('source_file_hashes', 'scientific_contract_hashes', 'dependency_file_hashes', 'critical_data_hashes', 'historical_artifact_hashes'):
        verify_hash_group(auth[key], key)
    for path, key in ((r.frozen.ROSTER, 'original_roster_hash'), (r.frozen.SPECS, 'original_arm_specs_hash'), (r.frozen.BINDINGS, 'original_pair_bindings_hash')):
        r.require(r.sha(path) == auth[key], 'R2_IDENTITY_HASH:' + key)
    for name, expected in auth['source_file_hashes'].items():
        r.require(r.executor._git_blob_matches(auth['source_git_commit'], name, expected), 'R2_SOURCE_GIT_BLOB:' + name)
    r.require(auth['source_git_commit'] == auth['EXECUTABLE_SOURCE_BASE_COMMIT'] == auth['authorized_source_git_sha'], 'R2_SOURCE_BASE')
    r.require(not r.git('diff', 'HEAD', '--', 'tools', 'configs', *NEW_SOURCE), 'R2_DIRTY_SOURCE')
    r.require(r.sha(r.PROTECTED) == r.PROTECTED_SHA, 'R2_PROTECTED_HASH')


def check_pending(ledger, attempt_id):
    r.require(ledger['authorization_sha256'] == r.sha(AUTH), 'R2_LEDGER_AUTH_HASH')
    r.require(not ledger['stop'], 'R2_TERMINAL_INFRASTRUCTURE_STOP')
    r.require(len(ledger['attempts']) == 6 and [{k: a[k] for k in c} for a, c in zip(ledger['attempts'], contracts())] == contracts(), 'R2_LEDGER_SCOPE')
    pending = next((a for a in ledger['attempts'] if a['status'] != 'ARM_COMPLETE'), None)
    r.require(pending and pending['attempt_id'] == attempt_id and pending['status'] == 'NOT_RUN', 'R2_ORDER_DUPLICATE_OR_RETRY')
    r.require(sum(a['runner_entry_claimed'] for a in ledger['attempts']) < 6, 'R2_BUDGET_EXHAUSTED')
    return pending


@contextmanager
def locked():
    with LEDGER.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def preflight():
    r.require(not PREFLIGHT.exists(), 'R2_PREFLIGHT_ALREADY_DECIDED;NO_REPAIR_OR_REPEAT')
    auth = r.load(AUTH)
    evidence, calls, error = [], {}, None
    try:
        verify_authorization(auth)
        record_commit = committed_record(AUTH)
        rp.verify_inherited_inputs(r.load(r.AUTH))
        ledger = r.load(LEDGER)
        ledger_hash_before = r.sha(LEDGER)
        r.require(all(a['status'] == 'NOT_RUN' and not a['runner_entry_claimed'] for a in ledger['attempts']), 'R2_LEDGER_NOT_FRESH')
        r.require(not ledger['stop'], 'R2_LEDGER_STOPPED')
        specs = {s['run_id']: s for s in r.load(r.frozen.SPECS)['arms']}
        retained_arms = []  # Retain references so identity isolation is verifiable, without allocator reuse.
        with gate.zero_run_guard() as calls:
            for a in contracts():
                root = RAW / 'final_zero_run_preflight' / f"{a['order']:02d}_{a['run_id']}"
                r.require(not Path(a['output_namespace']).exists(), 'R2_SCIENTIFIC_NAMESPACE_EXISTS')
                arm = build_fresh_arm(specs[a['run_id']], root)
                callback = gate.validate_production_callback_binding(arm)
                route = r.executor.validate_pre_run_lifecycle(arm, specs[a['run_id']])
                from omegaconf import OmegaConf
                cfg = OmegaConf.to_container(arm.cfg, resolve=True)
                normalized = r.normalized_config_hash(cfg, root)
                r.require(normalized == auth['resolved_configuration_hashes'][a['attempt_id']], 'R2_RESOLVED_CONFIG_MISMATCH')
                ids = r.executor.component_identity(arm)
                for old in retained_arms:
                    other = r.executor.component_identity(old)
                    r.require(all(ids[key] != other[key] for key in ids), 'R2_PRECHECK_MUTABLE_OBJECT_REUSE')
                retained_arms.append(arm)
                row = {**a, 'status': 'PASS', 'callback_binding': callback, 'route_gate': route,
                       'normalized_config_hash': normalized, 'resolved_config': cfg, 'component_ids': ids,
                       'session_id': arm.spec['session_id'], 'scenario_token': arm.spec['scenario_token'],
                       'precontext_id': arm.precontext_id, 'route_id': arm.route_id, 'simulator_entries': 0}
                evidence.append(row)
                write_json(OUT / 'preflight_evidence' / f"{a['order']:02d}_{a['run_id']}.json", row)
                print(f"B1-R2 final preflight {len(evidence)}/6 {a['run_id']} PASS", flush=True)
        r.require(not any(calls.values()), 'R2_ZERO_RUN_VIOLATION')
        r.require(r.sha(LEDGER) == ledger_hash_before, 'R2_ZERO_RUN_BUDGET_CHANGED')
        verdict = 'B1_R2_PREFLIGHT_PASS'
    except Exception as exc:
        error, verdict = f'{type(exc).__name__}:{exc}', 'B1_R2_PREFLIGHT_BLOCKED'
        traceback.print_exc()
    write_json(PREFLIGHT, {'verdict': verdict, 'authorization_sha256': r.sha(AUTH),
        'authorization_record_commit': locals().get('record_commit'), 'source_git_commit': auth['source_git_commit'],
        'results': evidence, 'guarded_calls': calls, 'failure': error, 'simulator_entries': 0})
    (OUT / 'B1_R2_Preflight_Report_v1.md').write_text(
        '# B1-R2 Final Production Preflight v1\n\n'
        + f'判定 `{verdict}`；真实工厂通过 {len(evidence)}/6；SIMULATOR_ENTRY=0；failure={error}。\n\n'
        + f'Executable source base `{auth["source_git_commit"]}`；authorization record `{locals().get("record_commit")}`。\n\n'
        + '授权、Git逐文件绑定、原20/40身份、历史预算、全部环境/数据hash、RP真实回调链、route/precontext、recorder/serializer、新namespace、解析配置与RP hash、独立对象均检查。共享callback列表可为空。静态PASS不保证未来运行成功；失败后禁止修复并重试本轮。\n', encoding='utf-8')
    print(verdict, flush=True)
    return verdict == 'B1_R2_PREFLIGHT_PASS'


def update_status(attempt_id, status):
    with locked():
        ledger = r.load(LEDGER)
        a = next(x for x in ledger['attempts'] if x['attempt_id'] == attempt_id)
        r.require(a['status'] == 'RUNNER_ENTRY_CLAIMED', 'R2_STATUS_WITHOUT_CLAIM')
        a['status'] = status
        if status != 'ARM_COMPLETE':
            ledger['stop'] = {'attempt_id': attempt_id, 'reason': status}
        write_json(LEDGER, ledger)


def execute_attempt(attempt_id):
    auth = r.load(AUTH)
    verify_authorization(auth)
    final = r.load(PREFLIGHT)
    r.require(final['verdict'] == 'B1_R2_PREFLIGHT_PASS' and final['authorization_sha256'] == r.sha(AUTH), 'R2_PREFLIGHT_NOT_PASSED')
    auth_record = committed_record(AUTH)
    preflight_record = committed_record(PREFLIGHT)
    committed_record(OUT / 'B1_R2_Preflight_Report_v1.md')
    r.require(final['authorization_record_commit'] == auth_record, 'R2_AUTHORIZATION_RECORD_CHANGED')
    with locked():
        ledger = r.load(LEDGER)
        target = dict(check_pending(ledger, attempt_id))
        spec = next(s for s in r.load(r.frozen.SPECS)['arms'] if s['run_id'] == target['run_id'])
        r.require(not Path(target['output_namespace']).exists(), 'R2_ATTEMPT_NAMESPACE_EXISTS')
        import numpy as np
        random.seed(2026091801)
        np.random.seed(2026091801)
        arm = build_fresh_arm(spec, Path(target['output_namespace']))
        callback = gate.validate_production_callback_binding(arm)
        route = r.executor.validate_pre_run_lifecycle(arm, spec)
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(arm.cfg, resolve=True)
        r.require(r.normalized_config_hash(cfg, arm.run_root) == auth['resolved_configuration_hashes'][attempt_id], 'R2_RUN_CONFIG_MISMATCH')
        metadata = {'stage': STAGE, 'attempt_id': attempt_id, 'attempt_number': target['attempt_number'],
            'authorization_sha256': r.sha(AUTH), 'authorization_record_commit': auth_record,
            'preflight_sha256': r.sha(PREFLIGHT), 'preflight_record_commit': preflight_record,
            'process_id': os.getpid(), 'source_git_commit': auth['source_git_commit'],
            'source_file_hashes': auth['source_file_hashes'], 'callback_precheck': callback,
            'official_lifecycle_executor': 'tools.b1_tsb_qualification_executor.complete_production_arm'}
        write_json(arm.run_root / 'resolved_config.json', cfg)
        write_json(arm.run_root / 'execution_manifest.json', {**r.executor.arm_manifest(spec, arm, auth, 'READY_BEFORE_RUNNER_RUN', route_precheck=route), **metadata})
        row = next(a for a in ledger['attempts'] if a['attempt_id'] == attempt_id)
        row.update(status='RUNNER_ENTRY_CLAIMED', runner_entry_claimed=True, process_id=os.getpid(),
                   cumulative_attempt_number=35+target['order'])
        write_json(LEDGER, ledger)
    # The existing sole B1 runner site and official six-step lifecycle are unchanged.
    result = r.executor.complete_production_arm(arm, spec, auth, route, lambda status: update_status(attempt_id, status), metadata)
    print(json.dumps(result), flush=True)


def evaluate_pair(pair_id):
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import _read_trace, evaluate_frozen_pair
    from tools.r1_closed_loop_benchmark_v2_1 import calculate_tsb_option_a_v2_timestamp_aware, exact_realized_window_v1_1, trajectory_arrays_timestamp_aware
    ledger = r.load(LEDGER)
    selected = {a['run_id'].rsplit('-', 1)[-1]: a for a in ledger['attempts'] if a['run_id'].startswith(pair_id+'-')}
    r.require(len(selected) == 2 and all(a['status'] == 'ARM_COMPLETE' for a in selected.values()), 'R2_PAIR_NOT_COMPLETE')
    roots = {arm: Path(a['output_namespace']) for arm, a in selected.items()}
    spec_by_arm = {s['arm']: s for s in r.load(r.frozen.SPECS)['arms'] if s['pair_id'] == pair_id}
    manifests = {arm: r.load(root / 'execution_manifest.json') for arm, root in roots.items()}
    for field in ('precontext_id', 'route_id'):
        r.require(manifests['BASELINE'][field] == manifests['TREATMENT'][field] == spec_by_arm['BASELINE'][field] == spec_by_arm['TREATMENT'][field], 'R2_PAIR_IDENTITY:' + field)
    metrics = {}
    for arm, root in roots.items():
        states = exact_realized_window_v1_1(_read_trace(root))
        time, _, _, speed = trajectory_arrays_timestamp_aware(states)
        metrics[arm] = calculate_tsb_option_a_v2_timestamp_aware(time, speed)
    write_json(OUT / 'pair_evaluations' / f'{pair_id}__measurements.json', metrics)
    binding = next(x for x in r.load(r.frozen.BINDINGS)['pairs'] if x['pair_id'] == pair_id)
    evaluation = evaluate_frozen_pair(pair_binding=binding, baseline_run_dir=roots['BASELINE'], treatment_run_dir=roots['TREATMENT'])
    write_json(OUT / 'pair_evaluations' / f'{pair_id}__evaluation.json', evaluation)
    print(f"{pair_id}: mechanism={evaluation['evaluation']['mechanism']['pass']}; F_match={evaluation['evaluation']['f_match']['pass']}; safety={evaluation['official_safety_pair_pass']}", flush=True)


def stop(attempt_id, reason):
    with locked():
        ledger = r.load(LEDGER)
        target = next(a for a in ledger['attempts'] if a['attempt_id'] == attempt_id)
        if target['status'] == 'RUNNER_ENTRY_CLAIMED':
            target['status'] = 'TECHNICAL_INCOMPLETE'
        ledger['stop'] = {'attempt_id': attempt_id, 'reason': reason}
        write_json(LEDGER, ledger)


def execute_all():
    r.require(r.load(PREFLIGHT)['verdict'] == 'B1_R2_PREFLIGHT_PASS', 'R2_PREFLIGHT_BLOCKED')
    ledger = r.load(LEDGER)
    r.require(not ledger['stop'] and all(a['status'] == 'NOT_RUN' for a in ledger['attempts']), 'R2_ALREADY_STARTED_NO_AUTOMATIC_RESUME')
    for a in contracts():
        log = OUT / 'execution_logs' / (a['attempt_id'] + '.log')
        log.parent.mkdir(exist_ok=True)
        print(f"B1-R2 authorized execution {a['order']}/6: {a['attempt_id']}", flush=True)
        with log.open('x') as handle:
            process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'attempt', '--attempt-id', a['attempt_id']],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONHASHSEED': '2026091801'})
        if process.returncode:
            stop(a['attempt_id'], f'PROCESS_EXIT_{process.returncode};LOG={log}')
            print('B1_R2_INFRASTRUCTURE_STOP', flush=True)
            return
        current = next(v for v in r.load(LEDGER)['attempts'] if v['attempt_id'] == a['attempt_id'])
        if current['status'] != 'ARM_COMPLETE':
            stop(a['attempt_id'], 'CHILD_EXITED_WITHOUT_ARM_COMPLETE')
            return
        pair_id = a['run_id'].rsplit('-', 1)[0]
        pair = [x for x in r.load(LEDGER)['attempts'] if x['run_id'].startswith(pair_id+'-')]
        if len(pair) == 2 and all(x['status'] == 'ARM_COMPLETE' for x in pair):
            try:
                evaluate_pair(pair_id)
            except ValueError as exc:
                if 'NOT_EVALUABLE' in str(exc):
                    write_json(OUT / 'pair_evaluations' / f'{pair_id}__measurement_failure.json', {'status':'MEASUREMENT_INVALID', 'reason':str(exc)})
                else:
                    stop(a['attempt_id'], f'PAIR_EVALUATION_INFRASTRUCTURE:{type(exc).__name__}:{exc}')
                    return
            except Exception as exc:
                stop(a['attempt_id'], f'PAIR_EVALUATION_INFRASTRUCTURE:{type(exc).__name__}:{exc}')
                traceback.print_exc()
                return
        print(f"B1-R2 {a['attempt_id']} ARM_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'execute', 'attempt', 'finalize'])
    parser.add_argument('--attempt-id')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'preflight':
        return 0 if preflight() else 2
    elif args.command == 'execute':
        execute_all()
    elif args.command == 'attempt':
        execute_attempt(args.attempt_id)
    else:
        from tools.b1_resume_report_r2 import finalize
        finalize()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
