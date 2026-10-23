#!/usr/bin/env python3
"""B1-RP engineering-only entry: official factories, callback fault matrix, no runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import b1_resume_execution as r
from tools import b1_production_preflight as gate
from tools.s2r_production_executor import build_fresh_arm, write_manifest, precontext_id, route_id

ROOT = r.ROOT
OUT = ROOT / 'docs/stageR/b1_resume_preflight_correction'
RAW = ROOT / 'outputs/stageR/b1_resume_preflight_correction_v1'
OWNER = Path('/Users/liuqing/.codex/attachments/03439929-f3eb-4930-9bcc-ce7a051bf0fb/已粘贴的文本.txt')
BASE_COMMIT = '448da65ec791b695ccd7b3695aabfa46ba575851'
ENGINEERING_CHANGED = {'tools/b1_resume_execution.py'}
STATUS_PASS = 'B1_RP_PREFLIGHT_PASS_READY_FOR_OWNER_REVIEW'


def history_hashes():
    paths = r.history_paths() + list(r.OUT.rglob('*')) + list(r.RAW.rglob('*'))
    return r.hashes(paths)


def verify_inherited_inputs(auth):
    r.require(r.git('branch', '--show-current') == r.BRANCH, 'WRONG_BRANCH')
    for group in ('source_file_hashes', 'scientific_contract_hashes', 'dependency_file_hashes',
                  'critical_data_hashes', 'historical_artifact_hashes'):
        for name, expected in auth[group].items():
            if group == 'source_file_hashes' and name in ENGINEERING_CHANGED:
                blob = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:{name}'], cwd=ROOT)
                r.require(hashlib.sha256(blob).hexdigest() == expected, 'OLD_BOUND_SOURCE_INVALID:' + name)
                continue
            r.require(r.sha(ROOT / name) == expected, f'FROZEN_INPUT_CHANGED:{group}:{name}')
    for path, key in ((r.frozen.ROSTER, 'original_roster_hash'), (r.frozen.SPECS, 'original_arm_specs_hash'),
                      (r.frozen.BINDINGS, 'original_pair_bindings_hash')):
        r.require(r.sha(path) == auth[key], 'IDENTITY_FILE_CHANGED:' + key)
    r.require(r.sha(r.PROTECTED) == r.PROTECTED_SHA, 'PROTECTED_CHANGED')
    roster, specs = r.rows(r.frozen.ROSTER), r.load(r.frozen.SPECS)['arms']
    r.require(len(roster) == 20 and len({x['session_id'] for x in roster}) == 20
              and len(specs) == len({x['run_id'] for x in specs}) == 40, 'ROSTER_20_40')
    original = r.load(r.frozen.RUN_ROOT / 'B1_Scientific_Arm_Budget_Ledger_v1.json')['runs']
    r.require(sum(s != 'NOT_RUN' for s in original.values()) == 35
              and sum(s == 'TECHNICAL_COMPLETE' for s in original.values()) == 34
              and original[r.TARGETS[0][0]] == 'TECHNICAL_INCOMPLETE', 'ORIGINAL_BUDGET_OR_FAILURE')
    r.require(all(original[run] == 'NOT_RUN' for run, n in r.TARGETS[1:]), 'FIVE_NOT_RUN_CHANGED')
    inherited = r.rows(ROOT / 'docs/stageR/b1_ir2_adjudication/B1_IR2_Existing17_Full_Contract_Status_v1.csv')
    r.require(len(inherited) == 17 and all(x['scientific_status'] == 'SCIENTIFIC_PASS' for x in inherited), 'IR2_17')
    ledger = r.load(r.LEDGER)
    r.require(len(ledger['attempts']) == 6 and all(x['status'] == 'NOT_RUN' and not x['runner_entry_claimed'] for x in ledger['attempts']), 'RESUME_BUDGET_NOT_ZERO')
    r.require(auth['allowed_execution_attempts'] == r.attempt_contract(), 'EXACT_FROZEN_SLICE')
    r.require(all(not Path(x['output_namespace']).exists() for x in r.attempt_contract()), 'SCIENTIFIC_ATTEMPT_NAMESPACE_EXISTS')
    for spec in specs:
        r.require(precontext_id(spec['precontext']) == spec['precontext_id'] == auth['precontext_hashes'][spec['run_id']], 'PRECONTEXT')
        r.require(route_id(spec['map_name'], spec['route_roadblock_ids']) == spec['route_id'] == auth['route_identity_hashes'][spec['run_id']], 'ROUTE_ID')
    r.frozen.validate_frozen_parameters()
    r.executor.validate_retry_request(2, 'NATIVE_ROUTE_INFRASTRUCTURE_FAILURE', True)
    return {'roster_pairs': 20, 'unique_arms': 40, 'historical_attempts': 35,
            'historical_complete_arms': 34, 'historical_complete_pairs': 17,
            'new_attempts': 0, 'cumulative_attempts': 35, 'all_40_identity_hashes_valid': True,
            'original_failure_preserved': True, 'five_not_run_preserved': True}


def verify_execution_body_unchanged():
    import ast
    # Only the preflight function and its evidence field change in the original resume tool.
    old = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:tools/b1_resume_execution.py'], cwd=ROOT).decode()
    new = (ROOT / 'tools/b1_resume_execution.py').read_text()
    def except_preflight(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == 'preflight')]
        return ast.dump(tree, include_attributes=False)
    r.require(except_preflight(old) == except_preflight(new), 'NON_PREFLIGHT_EXECUTION_CHANGE')
    for name in ('tools/b1_tsb_qualification_executor.py', 'tools/s2r_production_executor.py'):
        blob = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:{name}'], cwd=ROOT)
        r.require(blob == (ROOT / name).read_bytes(), 'PRODUCTION_EXECUTOR_CHANGED:' + name)
    return {'only_original_preflight_function_changed': True, 'runner_and_lifecycle_source_byte_unchanged': True}


def reject_old_authorization():
    # Use the actual old worker entry, with tripwires before factory/budget/runner.
    with patch.object(r, 'build_fresh_arm', side_effect=AssertionError('UNAUTHORIZED_FACTORY_REACHED')) as factory:
        try:
            r.execute_attempt(r.attempt_contract()[0]['attempt_id'])
        except RuntimeError as exc:
            reason = str(exc)
        else:
            raise RuntimeError('OLD_AUTHORIZATION_WAS_ACCEPTED')
    r.require('HASH_MISMATCH:source_file_hashes:tools/b1_resume_execution.py' in reason, 'WRONG_OLD_AUTH_REJECTION:' + reason)
    r.require(factory.call_count == 0, 'OLD_AUTH_REACHED_FACTORY')
    return {'status': 'REJECTED', 'reason': reason, 'factory_calls': 0, 'runner_calls': 0, 'budget_claims': 0,
            'authorization_sha256': r.sha(r.AUTH)}


def fault_matrix(arm, missing_config_root):
    """Faults applied only to freshly assembled real official objects; no ideal mocks."""
    from nuplan.planning.simulation.callback.metric_callback import MetricCallback
    from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
    main = arm.common_builder.multi_main_callback
    metric = next(x for x in arm.simulation.callback.callbacks if type(x) is MetricCallback)
    finalizer = next(x for x in main._main_callbacks if type(x) is MetricFileCallback)
    cases = [
        ('MISSING_METRIC_FILE_CALLBACK', main, '_main_callbacks', [x for x in main._main_callbacks if x is not finalizer], 'MISSING_METRIC_FILE_CALLBACK'),
        ('INVALID_CALLBACK_BINDING', metric, '_metric_engine', None, 'INVALID_CALLBACK_BINDING:METRIC_ENGINE'),
        ('MISSING_REQUIRED_FINALIZER', finalizer, 'on_run_simulation_end', None, 'MISSING_REQUIRED_FINALIZER'),
        ('INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION', finalizer, '_metric_file_output_path', arm.run_root / 'wrong_metric_output', 'INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION'),
        ('INVALID_METRIC_INPUT_CONFIGURATION', finalizer, '_scenario_metric_paths', [arm.run_root / 'wrong_metric_input'], 'INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION'),
        ('MISSING_MAIN_DISPATCH', main, 'on_run_simulation_end', None, 'MISSING_REQUIRED_FINALIZER'),
        ('CALLBACK_LIFECYCLE_CONTRACT_MISMATCH', r.executor, 'LIFECYCLE_ORDER', tuple(reversed(gate.EXPECTED_LIFECYCLE)), 'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH'),
        ('MISSING_SIMULATION_METRIC_CALLBACK', arm.simulation.callback, '_callbacks', [x for x in arm.simulation.callback.callbacks if x is not metric], 'MISSING_OFFICIAL_SIMULATION_METRIC_CALLBACK'),
        ('MISSING_SERIALIZER', arm.simulation.callback, '_callbacks', [x for x in arm.simulation.callback.callbacks if type(x) is not SimulationLogCallback], 'MISSING_OFFICIAL_SIMULATION_LOG_CALLBACK'),
        ('INVALID_SEQUENTIAL_LIFECYCLE', metric, '_pool', object(), 'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH'),
        ('UNBOUND_RECORDER', arm.recorder, '_tracker', None, 'INVALID_CALLBACK_BINDING:PASSIVE_RECORDER'),
    ]
    positive = gate.validate_production_callback_binding(arm)
    results = [{'case': name, 'expected': 'PASS', 'actual': positive['status'],
                'reason': '', 'real_official_factory': True, 'simulation_calls': 0, 'metric_finalization_calls': 0}
               for name in ('REAL_PRODUCTION_CALLBACK_COMPOSITION', 'OFFICIAL_METRIC_CALLBACK_BOUND',
                            'FINALIZATION_LIFECYCLE_AVAILABLE', 'OUTPUT_CONFIGURATION_VALID')]
    for name, owner, field, damaged, expected_code in cases:
        with patch.object(owner, field, damaged):
            try:
                gate.validate_production_callback_binding(arm)
            except gate.CallbackPreflightError as exc:
                reason = str(exc)
            else:
                raise RuntimeError('NEGATIVE_CONFIG_NOT_BLOCKED:' + name)
        r.require(reason.startswith(expected_code), 'NEGATIVE_WRONG_REASON:' + name + ':' + reason)
        results.append({'case': name, 'expected': 'BLOCK', 'actual': 'BLOCK', 'reason': reason,
                        'real_official_factory': True, 'simulation_calls': 0, 'metric_finalization_calls': 0})
    # Config fault passes through Hydra and the actual canonical factory/builder.
    import hydra
    from omegaconf import open_dict
    compose = hydra.compose
    def no_metric_file(*args, **kwargs):
        cfg = compose(*args, **kwargs)
        with open_dict(cfg):
            del cfg.main_callback['metric_file_callback']
        return cfg
    with patch.object(hydra, 'compose', no_metric_file):
        spec = arm.spec
        damaged_arm = build_fresh_arm(spec, missing_config_root)
    try:
        gate.validate_production_callback_binding(damaged_arm)
    except gate.CallbackPreflightError as exc:
        reason = str(exc)
    else:
        raise RuntimeError('MISSING_FINALIZER_CONFIG_NOT_BLOCKED')
    r.require(reason == 'MISSING_METRIC_FILE_CALLBACK', 'CONFIG_FAULT_WRONG_REASON:' + reason)
    results.append({'case': 'MISSING_METRIC_FILE_CONFIG_THROUGH_REAL_BUILDER', 'expected': 'BLOCK', 'actual': 'BLOCK', 'reason': reason,
                    'real_official_factory': True, 'simulation_calls': 0, 'metric_finalization_calls': 0})
    r.require(gate.validate_production_callback_binding(arm)['status'] == 'PASS', 'FAULT_TEST_NOT_RESTORED')
    return results


def source_transition():
    names = r.source_paths() + [ROOT / 'tools/b1_production_preflight.py', ROOT / 'tools/b1_resume_preflight_correction.py', ROOT / 'tests/test_b1_preflight_callback_gate.py']
    old_auth = r.load(r.AUTH)
    return {str(p.relative_to(ROOT)): {'before_sha256': old_auth['source_file_hashes'].get(str(p.relative_to(ROOT))),
                                      'after_sha256': r.sha(p)} for p in sorted(set(names)) if p.is_file()}


def run(output):
    r.require(not output.exists(), 'B1_RP_FRESH_REPORT_DIRECTORY_REQUIRED')
    output.mkdir(parents=True)
    old = r.load(r.AUTH)
    before = history_hashes()
    r.write_json(output / 'B1_RP_Historical_Hash_Audit_v1.json', {'before': before})
    results, negatives, guards, failure, status = [], [], {}, None, STATUS_PASS
    try:
        inputs = verify_inherited_inputs(old)
        source_proof = verify_execution_body_unchanged()
        with gate.zero_run_guard() as guards:
            rejection = reject_old_authorization()
            specs = {s['run_id']: s for s in r.load(r.frozen.SPECS)['arms']}
            for item in r.attempt_contract():
                # All six remain diagnostic names, never the scientific ATTEMPT output roots.
                root = RAW / 'real_production_preflight' / f"{item['order']:02d}_{item['run_id']}_static"
                r.require(not root.exists(), 'ZERO_RUN_NAMESPACE_ALREADY_EXISTS')
                arm = build_fresh_arm(specs[item['run_id']], root)
                callback = gate.validate_production_callback_binding(arm)
                route = r.executor.validate_pre_run_lifecycle(arm, specs[item['run_id']])
                from omegaconf import OmegaConf
                cfg = OmegaConf.to_container(arm.cfg, resolve=True)
                config_hash = r.normalized_config_hash(cfg, root)
                evidence = {'kind': 'ZERO_RUN_INSPECTION_ONLY_NOT_EXECUTION_MANIFEST', 'item': item,
                            'callback_binding': callback, 'route_gate': route, 'resolved_config': cfg,
                            'normalized_config_hash': config_hash, 'component_ids': r.executor.component_identity(arm)}
                evidence_path = output / 'zero_run_evidence' / f"{item['order']:02d}_{item['run_id']}.json"
                r.write_json(evidence_path, evidence)
                # The production manifest is assembled in memory only: do not create an ATTEMPT_2 execution record.
                manifest_schema = r.executor.arm_manifest(specs[item['run_id']], arm, old, 'ZERO_RUN_INSPECTION_ONLY')
                r.require(manifest_schema['precontext_id'] == specs[item['run_id']]['precontext_id'], 'MANIFEST_BINDING')
                results.append({**{k: item[k] for k in ('order', 'run_id', 'attempt_number')},
                                'session_id': arm.spec['session_id'], 'scenario_token': arm.spec['scenario_token'],
                                'precontext_id': arm.precontext_id, 'route_id': arm.route_id,
                                'route_status': route['route_precheck_status'], 'callback_status': callback['status'],
                                'lifecycle_status': 'STATIC_READY_RUNTIME_NOT_EXECUTED', 'recorder_status': 'BOUND_EMPTY',
                                'serializer_status': 'BOUND_NOT_RUN', 'shared_callback_count': len(arm.callbacks),
                                'normalized_config_hash': config_hash, 'namespace_status': 'DIAGNOSTIC_ONLY_NO_EXECUTION_ROOT',
                                'evidence_sha256': r.sha(evidence_path), 'status': 'PASS', 'new_attempts': 0})
                print(f"B1-RP zero-run {len(results)}/6 {item['run_id']} PASS", flush=True)
                if len(results) == 1:
                    negatives = fault_matrix(arm, RAW / 'negative_config_fixture')
        r.require(not any(guards.values()), 'ZERO_RUN_GUARD_TOUCHED')
    except gate.CallbackPreflightError as exc:
        status, failure = 'B1_RP_BLOCKED_CALLBACK_LIFECYCLE', f'{type(exc).__name__}:{exc}'
    except Exception as exc:
        status, failure = 'B1_RP_BLOCKED_ADDITIONAL_PREFLIGHT', f'{type(exc).__name__}:{exc}'
    after = history_hashes()
    r.require(before == after, 'HISTORICAL_OR_BUDGET_MUTATION')
    historical = {'before': before, 'after_hashes_match': True, 'unchanged_files': len(before), 'budgets_unchanged': True}
    r.write_json(output / 'B1_RP_Historical_Hash_Audit_v1.json', historical)
    r.write_json(output / 'B1_RP_Source_Hash_Transition_v1.json', {'source_git_commit': r.git('rev-parse', 'HEAD'),
                 'before_commit': BASE_COMMIT, 'files': source_transition(),
                 'execution_body_proof': locals().get('source_proof'), 'old_authorization_rejection': locals().get('rejection')})
    if results:
        r.csv_write(output / 'B1_RP_Real_Production_Preflight_Results_v1.csv', results)
    if negatives:
        r.csv_write(output / 'B1_RP_Positive_Negative_Test_Matrix_v1.csv', negatives)
    r.write_json(output / 'B1_RP_Zero_Run_Proof_v1.json', {'guarded_calls': guards, 'simulation': 0, 'runner_run': 0,
                 'new_scientific_attempts': 0, 'metric_finalization': 0, 'historical_attempts': 35, 'cumulative_attempts': 35,
                 'forbidden_scientific_actions': {key: 0 for key in r.PROHIBITIONS}, 'budget_hash_audit': 'UNCHANGED'})
    draft = copy.deepcopy(old)
    draft.update(stage='B1_R2_PROPOSED_ONLY', authorization_status='PROPOSED_ONLY', AUTHORIZATION_STATUS='PROPOSED_ONLY',
                 OWNER_APPROVAL='PENDING', EXECUTION='NOT_AUTHORIZED', owner_authorization_scope='NONE;DRAFT_FOR_OWNER_REVIEW_ONLY',
                 source_git_commit=r.git('rev-parse', 'HEAD'), authorized_source_git_sha=None,
                 source_file_hashes={name: value['after_sha256'] for name, value in source_transition().items()},
                 previous_b1_r_authorization_sha256=r.sha(r.AUTH),
                 authorization_source={'status': 'OWNER_APPROVAL_PENDING', 'prior_b1_r_instruction_is_not_new_authority': old['authorization_source']},
                 engineering_instruction={'path': str(OWNER), 'sha256': r.sha(OWNER), 'scope': 'ZERO_RUN_ENGINEERING_ONLY'},
                 current_stage_status=status, new_execution_budget_granted=0,
                 previous_authorization_cannot_execute_current_source=True)
    r.write_json(output / 'B1_RP_Proposed_B1_R2_Authorization_v1.json', draft)
    r.write_json(output / 'B1_RP_Run_Status_v1.json', {'status': status, 'failure': failure, 'six_item_pass_count': len(results),
                 'matrix_cases': len(negatives), 'scientific_b1_status': 'B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE',
                 'B1_R2_EXECUTION': 'NOT_AUTHORIZED', 'inputs': locals().get('inputs')})
    print(json.dumps({'status': status, 'failure': failure, 'six_item_pass_count': len(results), 'new_attempts': 0}), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    return 0 if run(args.output_dir) == STATUS_PASS else 2


if __name__ == '__main__':
    raise SystemExit(main())
