#!/usr/bin/env python3
"""Owner-scoped B1 resume: immutable authorization, zero-run gate, six attempts."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import b1_tsb_qualification as frozen
from tools import b1_tsb_qualification_executor as executor
from tools.s2r_production_executor import (
    ROOT, DEVKIT, PARAMETERS, PROTECTED, PROTECTED_SHA, build_fresh_arm,
    precontext_id, route_id, sha256_file as sha, write_manifest as write_json,
)

OUT = ROOT / 'docs/stageR/b1_resume_execution'
RAW = ROOT / 'outputs/stageR/b1_resume_execution_v1'
AUTH = OUT / 'B1_R_Owner_Conditional_Authorization_v1.json'
LEDGER = OUT / 'B1_R_Resume_Attempt_Ledger_v1.json'
PREFLIGHT = OUT / 'B1_R_Preflight_Evidence_v1.json'
OWNER = Path('/Users/liuqing/.codex/attachments/ef9f05ee-cbde-44c6-a960-65b2c0446c1d/已粘贴的文本.txt')
IR3_COMMIT = 'c5be0108924c4135f16837a1ab3fec1c1ec7f4c0'
BRANCH = '20260825_stageR_new'
TARGETS = [('B1-TSB-18-TREATMENT', 2), ('B1-TSB-18-BASELINE', 1),
           ('B1-TSB-19-TREATMENT', 1), ('B1-TSB-19-BASELINE', 1),
           ('B1-TSB-20-BASELINE', 1), ('B1-TSB-20-TREATMENT', 1)]
PROHIBITIONS = ['RBR_TRAINING', 'RBR_INFERENCE', 'H_FEATURE_EVALUATION', 'BDD',
                'MMD', 'Z64_COMPUTATION', 'PRIMARY_EVALUATION', 'V_EXECUTION', 'C_EXECUTION']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def csv_write(path, data):
    fields = list(dict.fromkeys(key for row in data for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def attempt_contract():
    return [dict(order=i, original_execution_order=34+i, run_id=run, attempt_number=n,
                 attempt_id=f'{run}__ATTEMPT_{n}',
                 output_namespace=str(RAW / run / f'ATTEMPT_{n}'))
            for i, (run, n) in enumerate(TARGETS, 1)]


def hashes(paths):
    return {str(p): sha(p) for p in sorted(set(Path(x) for x in paths)) if p.is_file()}


def source_paths():
    # Bind the full tracked tool/config tree, including transitive scientific imports.
    return [ROOT / p for p in git('ls-files', 'tools', 'configs', 'tests/test_b1_resume_execution.py', 'docs/stageR/b1_resume_execution/B1_R_Execution_Contract_v1.md').splitlines()
            if p.endswith(('.py', '.json', '.yaml', '.yml', '.md'))]


def dependency_paths():
    # Bind external official implementation/configs and the installed runtime files.
    paths = []
    for root in (DEVKIT / 'nuplan', ROOT.parent / 'tuplan_garage/tuplan_garage'):
        paths += [p for p in root.rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.yml', '.json'}]
    env = Path(sys.prefix)
    for root in (env / 'lib', env / 'conda-meta'):
        paths += [p for p in root.rglob('*') if p.is_file() and p.suffix in {'.py', '.so', '.dylib', '.json', '.yaml', '.yml'}
                  and '__pycache__' not in p.parts]
    paths += [Path(sys.executable).resolve()]
    return paths


def history_paths():
    paths = list(frozen.RUN_ROOT.rglob('*'))
    for name in ('b1_tsb_qualification', 'b1_infrastructure_remediation', 'b1_ir2_adjudication', 'b1_ir3_route_closure'):
        paths += list((ROOT / 'docs/stageR' / name).rglob('*'))
    paths.append(PROTECTED)
    return [p for p in paths if p.is_file()]


def prepare():
    require(not AUTH.exists() and not RAW.exists() and not LEDGER.exists(), 'B1_R_FRESH_STAGE_REQUIRED;REFUSE_AUTH_REBIND')
    if OUT.exists():
        require({p.name for p in OUT.iterdir()} <= {'B1_R_Execution_Contract_v1.md', 'B1_R_Validation_Record_v1.md'}, 'B1_R_UNEXPECTED_EXISTING_STAGE_OUTPUT')
    require(git('branch', '--show-current') == BRANCH, 'B1_R_WRONG_BRANCH')
    source_commit = git('rev-parse', 'HEAD')
    # Refuse to freeze dirty code, including this stage's source and tests.
    require(not git('diff', 'HEAD', '--', 'tools', 'configs', 'tests/test_b1_resume_execution.py'), 'B1_R_DIRTY_SOURCE')
    specs = load(frozen.SPECS)['arms']
    original_auth = load(frozen.AUTH)
    ir3 = load(ROOT / 'docs/stageR/b1_ir3_route_closure/B1_IR3_Manifest_v1.json')
    contracts = dict(ir3['governing_input_sha256'])
    contracts[str(PARAMETERS.relative_to(ROOT))] = sha(PARAMETERS)
    # Scientific source/config must equal their actual historical frozen blobs.
    for key in ('tools/r1_closed_loop_benchmark_v2_1.py', 'tools/r1_context_mechanism_core.py',
                'tools/r1_official_technical_smoke_evaluator_v2_1.py',
                'tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py',
                'tools/r1_b2_8_r3_1_official_safety_adapter.py', 'tools/r1_official_metric_canonicalizer.py',
                'tools/r2_b_controller_aware_generator_v1.py', 'tools/r2_bj_b0_2_passive_actual_lqr_recorder.py',
                'tools/s1_protocol_schema.py'):
        contracts[key] = sha(ROOT / key)
    data = [Path(s['db_path']) for s in specs if (s['run_id'], 2 if s['run_id'] == TARGETS[0][0] else 1) in TARGETS]
    # Official map database files used by the frozen maps; no trajectory generated.
    map_root = ROOT.parent / 'nuplan/dataset/maps'
    for name in {s['map_name'] for s in specs}:
        data += list((map_root / name).rglob('*.gpkg'))
    data += list(map_root.glob('*.json'))
    auth = dict(
        stage='B1_R_FROZEN_TSB_RESUME', authorization_status='CONDITIONALLY_AUTHORIZED',
        owner_authorization_scope='EXACT_SIX_ATTEMPTS_AFTER_ALL_PREFLIGHT_GATES_PASS',
        authorization_source={'path': str(OWNER), 'sha256': sha(OWNER)},
        source_git_commit=source_commit, authorized_source_git_sha=source_commit, branch=BRANCH,
        predecessor_ir3_commit=IR3_COMMIT,
        scientific_contract_hashes=contracts,
        original_roster_hash=sha(frozen.ROSTER), original_arm_specs_hash=sha(frozen.SPECS),
        original_pair_bindings_hash=sha(frozen.BINDINGS),
        original_authorization_hash=sha(frozen.AUTH),
        precontext_hashes={s['run_id']: s['precontext_id'] for s in specs},
        route_identity_hashes={s['run_id']: s['route_id'] for s in specs},
        source_file_hashes={str(p.relative_to(ROOT)): sha(p) for p in source_paths()},
        dependency_file_hashes=hashes(dependency_paths()), critical_data_hashes=hashes(data),
        historical_artifact_hashes=hashes(history_paths()),
        allowed_execution_attempts=attempt_contract(), maximum_new_attempts=6,
        original_attempts_consumed=35, maximum_cumulative_attempts=41,
        stop_policy=['COMPLETE_PRE_REGISTERED_ROSTER_ON_SCIENTIFIC_FAILURE', 'STOP_ON_INFRASTRUCTURE_FAILURE'],
        retry_contract={**executor.RETRY_CONTRACT, 'only_authorized_retry': TARGETS[0][0], 'authorized_attempt': 2,
                        'no_other_second_attempt': True, 'attempt_3_forbidden': True},
        no_replacement_rule='ORIGINAL_20_PAIRS_40_ARMS_NO_REPLACEMENT_NO_SURVIVOR_SELECTION',
        scientific_prohibitions=PROHIBITIONS,
        production_route_version='build_native_route_reference_v1_2_native_successor_backtracking',
        resolved_config_comparison_normalization=['absolute_output_root', 'experiment_uid_timestamp', 'experiment_derived_uid', 'metric_aggregator_filename_timestamp'],
        original_authorization_source_commit=original_auth['authorized_source_git_sha'],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(AUTH, auth)
    write_json(LEDGER, {'schema_version': 'B1_R_Resume_Attempt_Ledger_v1', 'authorization_sha256': sha(AUTH),
                       'historical_attempts': 35, 'maximum_new_attempts': 6, 'maximum_cumulative_attempts': 41,
                       'stop': None, 'attempts': [{**a, 'status': 'NOT_RUN', 'runner_entry_claimed': False}
                                                 for a in attempt_contract()]})
    print('B1_R_AUTHORIZATION_FROZEN; simulator entries=0', flush=True)


def verify_authorization(auth):
    require(auth['stage'] == 'B1_R_FROZEN_TSB_RESUME' and auth['authorization_status'] == 'CONDITIONALLY_AUTHORIZED', 'AUTH_STAGE')
    require(auth['allowed_execution_attempts'] == attempt_contract(), 'EXACT_ATTEMPT_SCOPE_OR_ORDER_MISMATCH')
    require(auth['maximum_new_attempts'] == 6 and auth['maximum_cumulative_attempts'] == 41, 'BUDGET_CONTRACT')
    require(auth['scientific_prohibitions'] == PROHIBITIONS, 'FIREWALL_CONTRACT')
    require(git('branch', '--show-current') == auth['branch'] == BRANCH, 'BRANCH_MISMATCH')
    require(sha(OWNER) == auth['authorization_source']['sha256'], 'OWNER_SOURCE_MISMATCH')
    require(sha(frozen.ROSTER) == auth['original_roster_hash'], 'ROSTER_HASH')
    require(sha(frozen.SPECS) == auth['original_arm_specs_hash'], 'SPECS_HASH')
    require(sha(frozen.BINDINGS) == auth['original_pair_bindings_hash'], 'BINDINGS_HASH')
    for group in ('source_file_hashes', 'scientific_contract_hashes', 'dependency_file_hashes',
                  'critical_data_hashes', 'historical_artifact_hashes'):
        for name, expected in auth[group].items():
            path = ROOT / name
            require(path.is_file() and sha(path) == expected, f'HASH_MISMATCH:{group}:{name}')
    # Current tracked source matches the exact committed source stack.
    dirty = git('diff', 'HEAD', '--', 'tools', 'configs')
    require(not dirty, 'UNAUTHORIZED_DIRTY_SOURCE')
    for name, expected in auth['source_file_hashes'].items():
        require(executor._git_blob_matches(auth['source_git_commit'], name, expected), f'SOURCE_COMMIT_MISMATCH:{name}')
    require(sha(PROTECTED) == PROTECTED_SHA, 'PROTECTED_ASSET_HASH')


@contextmanager
def forbid_simulation():
    from unittest.mock import patch
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from tools.r2_b_controller_aware_planner_v1 import R2BControllerAwarePlannerV1
    with patch.object(SimulationRunner, 'run', side_effect=RuntimeError('ZERO_RUN_GUARD')), \
         patch.object(TwoStageController, 'update_state', side_effect=RuntimeError('ZERO_RUN_GUARD')), \
         patch.object(R2BControllerAwarePlannerV1, 'compute_planner_trajectory', side_effect=RuntimeError('ZERO_RUN_GUARD')):
        yield


def normalized_config_hash(cfg, run_root):
    """Ignore only administrative output namespace and generated timestamp labels."""
    from tools.s2r_production_executor import canonical_sha256
    normalized = json.loads(json.dumps(cfg, sort_keys=True).replace(str(run_root), '__ATTEMPT_ROOT__'))
    timestamp_pattern = r'\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}'
    if 'experiment_uid' in normalized:
        uid = normalized['experiment_uid']
        require(re.fullmatch(timestamp_pattern, uid) is not None, 'UNEXPECTED_EXPERIMENT_UID_FORMAT')
        expected = f"{normalized['experiment_name']}/{normalized['job_name']}/{uid}"
        require(normalized['experiment'] == expected, 'UNEXPECTED_EXPERIMENT_DERIVATION')
        normalized['experiment_uid'] = '__GENERATED_TIMESTAMP__'
        normalized['experiment'] = expected[:-len(uid)] + '__GENERATED_TIMESTAMP__'
        for aggregator in normalized.get('metric_aggregator', {}).values():
            name = aggregator.get('file_name')
            if name:
                require(re.search(timestamp_pattern + r'$', name) is not None, 'UNEXPECTED_AGGREGATOR_FILENAME_FORMAT')
                aggregator['file_name'] = re.sub(timestamp_pattern + r'$', '__GENERATED_TIMESTAMP__', name)
    return canonical_sha256(normalized)


def preflight():
    require(not PREFLIGHT.exists(), 'PREFLIGHT_ALREADY_DECIDED;NO_AUTOMATIC_REPAIR_AND_CONTINUE')
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    checks, probes = [], []
    auth = load(AUTH)
    try:
        verify_authorization(auth)
        checks.append('01_CLEAN_COMMITTED_EXECUTION_SOURCE')
        roster, specs = rows(frozen.ROSTER), load(frozen.SPECS)['arms']
        require(len(roster) == 20 and len({r['session_id'] for r in roster}) == 20 and len(specs) == 40
                and len({s['run_id'] for s in specs}) == 40, '20_PAIRS_40_ARMS')
        checks.append('02_ORIGINAL_ROSTER_IDENTITIES')
        inherited = rows(ROOT / 'docs/stageR/b1_ir2_adjudication/B1_IR2_Existing17_Full_Contract_Status_v1.csv')
        require(len(inherited) == 17 and all(r['scientific_status'] == 'SCIENTIFIC_PASS' and r['full_contract_status'] == 'FULL_CONTRACT_COMPLETE' for r in inherited), 'IR2_17_FULL_CONTRACT')
        # Verify the complete historical snapshot against IR3, not merely a new snapshot.
        old_evidence = load(ROOT / 'docs/stageR/b1_ir3_route_closure/B1_IR3_Static_Evidence_v1.json')
        for name, expected in old_evidence['historical_artifact_sha256'].items():
            require(sha(ROOT / name) == expected, f'IR3_HISTORICAL_HASH:{name}')
        checks.append('03_IR2_17_RESULTS_TRACEABLE_UNCHANGED')
        old_ledger = load(frozen.RUN_ROOT / 'B1_Scientific_Arm_Budget_Ledger_v1.json')['runs']
        require(old_ledger[TARGETS[0][0]] == 'TECHNICAL_INCOMPLETE', 'ATTEMPT_1_FAILURE_NOT_PRESERVED')
        checks.append('04_ATTEMPT_1_PRESERVED')
        require(sum(v == 'NOT_RUN' for v in old_ledger.values()) == 5, 'NOT_RUN_COUNT')
        for run, _ in TARGETS[1:]:
            require(old_ledger[run] == 'NOT_RUN', f'ALREADY_RUN:{run}')
            trace = frozen.RUN_ROOT / run / 'trace/realized_current_ego.jsonl'
            require(not trace.exists(), f'UNDECLARED_EXECUTION:{run}')
        checks.append('05_FIVE_ARMS_NOT_PREVIOUSLY_EXECUTED')
        ir3 = load(ROOT / 'docs/stageR/b1_ir3_route_closure/B1_IR3_Manifest_v1.json')
        for name, expected in ir3['source_sha256'].items():
            if name != 'tools/b1_tsb_qualification_executor.py':
                require(sha(ROOT / name) == expected, f'IR3_SOURCE_CHANGED:{name}')
        checks.append('06_IR3_NATIVE_ROUTE_SOURCE_BOUND')
        for s in specs:
            require(precontext_id(s['precontext']) == s['precontext_id'] == auth['precontext_hashes'][s['run_id']], 'PRECONTEXT_HASH')
            require(route_id(s['map_name'], s['route_roadblock_ids']) == s['route_id'] == auth['route_identity_hashes'][s['run_id']], 'ROUTE_HASH')
        checks.append('07_ALL_40_ROUTE_PRECONTEXT_IDENTITIES')
        by_id = {s['run_id']: s for s in specs}
        with forbid_simulation():
            for a in attempt_contract():
                probe_root = RAW / 'zero_run_preflight' / a['attempt_id']
                arm = build_fresh_arm(by_id[a['run_id']], probe_root)
                gate = executor.validate_pre_run_lifecycle(arm, by_id[a['run_id']])
                require(arm.recorder is not None and len(arm.callbacks) > 0, 'RECORDER_CALLBACK_NOT_BOUND')
                from omegaconf import OmegaConf
                cfg = OmegaConf.to_container(arm.cfg, resolve=True)
                require('simulation_log_callback' in cfg['callback'] and cfg['run_metric'] is True, 'SERIALIZER_METRIC_CONFIG')
                write_json(probe_root / 'resolved_config.json', cfg)
                probe = {'attempt_id': a['attempt_id'], 'route_gate': gate, 'config_hash': arm.config_hash,
                         'normalized_config_hash': normalized_config_hash(cfg, arm.run_root),
                         'official_finalizer_bound': executor.official_metric_finalizer_bound(arm),
                         'recorder_class': type(arm.recorder).__module__ + '.' + type(arm.recorder).__name__,
                         'serializer_config': cfg['callback']['simulation_log_callback'],
                         'manifest_schema': executor.arm_manifest(by_id[a['run_id']], arm, auth, 'ZERO_RUN_PROBE'),
                         'simulator_entries': 0}
                probes.append(probe)
                print(f"B1-R zero-run factory {len(probes)}/6: {a['attempt_id']} COMPATIBLE", flush=True)
        checks.extend(['08_ALL_SIX_PRODUCTION_ROUTE_GATES_COMPATIBLE', '09_OFFICIAL_CALLBACK_RECORDER_SERIALIZER_MANIFEST_BOUND'])
        require(all(not Path(a['output_namespace']).exists() for a in attempt_contract()), 'ATTEMPT_NAMESPACE_COLLISION')
        checks.append('10_FRESH_ATTEMPT_NAMESPACES')
        require(len(old_ledger) == 40 and sum(v != 'NOT_RUN' for v in old_ledger.values()) == 35
                and sum(v == 'TECHNICAL_COMPLETE' for v in old_ledger.values()) == 34, 'ORIGINAL_BUDGET')
        checks.append('11_ORIGINAL_35_BUDGET_RETAINED')
        ledger = load(LEDGER)
        require(ledger['authorization_sha256'] == sha(AUTH) and len(ledger['attempts']) == 6
                and all(a['status'] == 'NOT_RUN' for a in ledger['attempts']), 'NEW_LEDGER_NOT_FRESH')
        checks.append('12_INDEPENDENT_SIX_ATTEMPT_LEDGER')
        executor.validate_retry_request(2, 'NATIVE_ROUTE_INFRASTRUCTURE_FAILURE', True)
        require(auth['retry_contract']['only_authorized_retry'] == TARGETS[0][0], 'RETRY_SCOPE')
        checks.append('13_TECHNICAL_RETRY_ONLY_NOT_SCIENTIFIC_RETRY')
        frozen.validate_frozen_parameters()
        for name, expected in auth['scientific_contract_hashes'].items():
            require(sha(ROOT / name) == expected, f'SCIENCE_CONTRACT_CHANGED:{name}')
            # Actual historical commit comparison, not a newly generated digest alone.
            blob = subprocess.check_output(['git', 'show', f'{IR3_COMMIT}:{name}'], cwd=ROOT)
            require(hashlib.sha256(blob).hexdigest() == expected, f'SCIENCE_NOT_FROZEN_AT_IR3:{name}')
        checks.append('14_FROZEN_PARAMETERS_ANALYZERS_THRESHOLDS')
        verdict, failure = 'B1_R_PREFLIGHT_PASS', None
    except Exception as exc:
        verdict, failure = 'B1_R_PREFLIGHT_BLOCKED', f'{type(exc).__name__}:{exc}'
        traceback.print_exc()
    evidence = dict(verdict=verdict, authorization_sha256=sha(AUTH), passed_checks=checks, failure=failure,
                    zero_run_probes=probes, simulator_entries=0, source_git_commit=auth['source_git_commit'])
    write_json(PREFLIGHT, evidence)
    (OUT / 'B1_R_Preflight_Report_v1.md').write_text(
        '# B1-R Zero-Run Preflight v1\n\n' + f'判定：`{verdict}`。Simulator entries = 0。\n\n'
        + '\n'.join(f'- PASS `{c}`' for c in checks) + f'\n\nFailure: {failure}\n'
        + '\n完整工厂、路由和 callback 证据见 B1_R_Preflight_Evidence_v1.json。失败后本授权不允许自动修复并恢复。\n', encoding='utf-8')
    print(verdict, flush=True)
    return verdict == 'B1_R_PREFLIGHT_PASS'


def validate_claim(ledger, attempt_id):
    require(not ledger['stop'], 'INFRASTRUCTURE_STOP_IS_TERMINAL')
    require(ledger['authorization_sha256'] == sha(AUTH), 'LEDGER_AUTH_HASH')
    require([{k: row[k] for k in attempt_contract()[i]} for i, row in enumerate(ledger['attempts'])] == attempt_contract(), 'LEDGER_SCOPE')
    pending = next((a for a in ledger['attempts'] if a['status'] != 'ARM_COMPLETE'), None)
    require(pending is not None and pending['attempt_id'] == attempt_id and pending['status'] == 'NOT_RUN', 'ATTEMPT_ORDER_OR_DUPLICATE')
    require(sum(a['runner_entry_claimed'] for a in ledger['attempts']) < 6, 'BUDGET_EXHAUSTED')
    return pending


@contextmanager
def locked_ledger():
    with LEDGER.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def update_status(attempt_id, status):
    with locked_ledger():
        ledger = load(LEDGER)
        a = next(x for x in ledger['attempts'] if x['attempt_id'] == attempt_id)
        require(a['status'] == 'RUNNER_ENTRY_CLAIMED', 'STATUS_WITHOUT_CLAIM')
        a['status'] = status
        if status != 'ARM_COMPLETE':
            ledger['stop'] = {'attempt_id': attempt_id, 'reason': 'TECHNICAL_INCOMPLETE'}
        write_json(LEDGER, ledger)


def execute_attempt(attempt_id):
    auth = load(AUTH)
    verify_authorization(auth)
    gate = load(PREFLIGHT)
    require(gate['verdict'] == 'B1_R_PREFLIGHT_PASS' and gate['authorization_sha256'] == sha(AUTH), 'PREFLIGHT_NOT_PASSED')
    # The authorization and the complete zero-run gate must be durable before exposure.
    for path in (AUTH, PREFLIGHT, OUT / 'B1_R_Preflight_Report_v1.md'):
        require(git('ls-files', '--error-unmatch', str(path.relative_to(ROOT))) and not git('diff', 'HEAD', '--', str(path.relative_to(ROOT))), 'PREFLIGHT_NOT_COMMITTED')
    with locked_ledger():
        ledger = load(LEDGER)
        target = dict(validate_claim(ledger, attempt_id))
        spec = next(s for s in load(frozen.SPECS)['arms'] if s['run_id'] == target['run_id'])
        require(not Path(target['output_namespace']).exists(), 'NAMESPACE_EXISTS')
        import random
        import numpy as np
        random.seed(2026091801)
        np.random.seed(2026091801)
        arm = build_fresh_arm(spec, Path(target['output_namespace']))
        route_gate = executor.validate_pre_run_lifecycle(arm, spec)
        metadata = {'stage': auth['stage'], 'attempt_id': attempt_id, 'attempt_number': target['attempt_number'],
                    'authorization_sha256': sha(AUTH), 'preflight_sha256': sha(PREFLIGHT), 'process_id': os.getpid(),
                    'source_file_hashes': auth['source_file_hashes'], 'official_lifecycle_executor': 'tools.b1_tsb_qualification_executor.complete_production_arm'}
        write_json(arm.run_root / 'execution_manifest.json', {**executor.arm_manifest(spec, arm, auth, 'READY_BEFORE_RUNNER_RUN', route_precheck=route_gate), **metadata})
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(arm.cfg, resolve=True)
        expected_probe = next(p for p in gate['zero_run_probes'] if p['attempt_id'] == attempt_id)
        require(normalized_config_hash(cfg, arm.run_root) == expected_probe['normalized_config_hash'], 'RESOLVED_CONFIG_PREFLIGHT_MISMATCH')
        write_json(arm.run_root / 'resolved_config.json', cfg)
        pending = next(a for a in ledger['attempts'] if a['attempt_id'] == attempt_id)
        pending.update(status='RUNNER_ENTRY_CLAIMED', runner_entry_claimed=True, process_id=os.getpid())
        write_json(LEDGER, ledger)
    result = executor.complete_production_arm(arm, spec, auth, route_gate, lambda status: update_status(attempt_id, status), metadata)
    print(json.dumps(result), flush=True)


def evaluate_new_pair(pair_id):
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import _read_trace, evaluate_frozen_pair
    from tools.r1_closed_loop_benchmark_v2_1 import (calculate_tsb_option_a_v2_timestamp_aware,
        exact_realized_window_v1_1, trajectory_arrays_timestamp_aware)
    bindings = {x['pair_id']: x for x in load(frozen.BINDINGS)['pairs']}
    specs = {s['arm']: s for s in load(frozen.SPECS)['arms'] if s['pair_id'] == pair_id}
    ledger = load(LEDGER)
    selected = {a['run_id']: a for a in ledger['attempts']}
    require(all(selected[s['run_id']]['status'] == 'ARM_COMPLETE' for s in specs.values()), 'PAIR_NOT_COMPLETE')
    roots = {arm: Path(selected[s['run_id']]['output_namespace']) for arm, s in specs.items()}
    manifests = {arm: load(root / 'execution_manifest.json') for arm, root in roots.items()}
    for field in ('precontext_id', 'route_id'):
        require(manifests['BASELINE'][field] == manifests['TREATMENT'][field] == specs['BASELINE'][field] == specs['TREATMENT'][field], f'PAIR_IDENTITY:{field}')
    metrics = {}
    for arm, root in roots.items():
        states = exact_realized_window_v1_1(_read_trace(root))
        t, _, _, speed = trajectory_arrays_timestamp_aware(states)
        metrics[arm] = calculate_tsb_option_a_v2_timestamp_aware(t, speed)
    # Keep measurement/mechanism evidence even if later safety/evaluator parsing fails.
    write_json(OUT / 'pair_evaluations' / f'{pair_id}__measurements.json', metrics)
    result = evaluate_frozen_pair(pair_binding=bindings[pair_id], baseline_run_dir=roots['BASELINE'], treatment_run_dir=roots['TREATMENT'])
    write_json(OUT / 'pair_evaluations' / f'{pair_id}__evaluation.json', result)
    return metrics, result


def execute_all():
    require(load(PREFLIGHT)['verdict'] == 'B1_R_PREFLIGHT_PASS', 'PREFLIGHT_BLOCKED')
    require(all(a['status'] == 'NOT_RUN' for a in load(LEDGER)['attempts']), 'EXECUTION_ALREADY_STARTED;NO_AUTOMATIC_RESUME')
    for a in attempt_contract():
        log = OUT / 'execution_logs' / f"{a['attempt_id']}.log"
        log.parent.mkdir(exist_ok=True)
        print(f"B1-R authorized attempt {a['order']}/6: {a['attempt_id']}", flush=True)
        with log.open('x') as handle:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'attempt', '--attempt-id', a['attempt_id']], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONHASHSEED': '2026091801'})
        error = None
        if result.returncode:
            error = f'ATTEMPT_PROCESS_EXIT_{result.returncode};see {log}'
        else:
            pair_id = a['run_id'].rsplit('-', 1)[0]
            pair = [x for x in load(LEDGER)['attempts'] if x['run_id'].startswith(pair_id + '-')]
            if len(pair) == 2 and all(x['status'] == 'ARM_COMPLETE' for x in pair):
                try:
                    evaluate_new_pair(pair_id)
                except ValueError as exc:
                    if 'NOT_EVALUABLE' in str(exc):
                        write_json(OUT / 'pair_evaluations' / f'{pair_id}__measurement_failure.json', {'status': 'MEASUREMENT_INVALID', 'reason': str(exc)})
                    else:
                        error = f'PAIR_EVALUATION_INFRASTRUCTURE:{type(exc).__name__}:{exc}'
                except Exception as exc:
                    error = f'PAIR_EVALUATION_INFRASTRUCTURE:{type(exc).__name__}:{exc}'
        if error:
            with locked_ledger():
                ledger = load(LEDGER)
                current = next(x for x in ledger['attempts'] if x['attempt_id'] == a['attempt_id'])
                if current['status'] == 'RUNNER_ENTRY_CLAIMED':
                    current['status'] = 'TECHNICAL_INCOMPLETE'
                ledger['stop'] = {'attempt_id': a['attempt_id'], 'reason': error}
                write_json(LEDGER, ledger)
            print('B1_R_INFRASTRUCTURE_STOP:' + error, flush=True)
            return
        print(f"B1-R {a['attempt_id']} ARM_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'execute', 'attempt', 'finalize', 'audit'])
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
        from tools.b1_resume_report import finalize, audit
        (finalize if args.command == 'finalize' else audit)()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
