"""Budget, stop policy, shared lifecycle and inherited-result protection; no rollout."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import b1_resume_execution as r
from tools import b1_resume_report as report
from tools import b1_tsb_qualification_executor as executor


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    auth = tmp_path / 'auth.json'
    auth.write_text('{}')
    monkeypatch.setattr(r, 'AUTH', auth)
    monkeypatch.setattr(r, 'LEDGER', tmp_path / 'ledger.json')
    value = dict(authorization_sha256=r.sha(auth), historical_attempts=35, stop=None,
                 attempts=[{**a, 'status': 'NOT_RUN', 'runner_entry_claimed': False} for a in r.attempt_contract()])
    r.write_json(r.LEDGER, value)
    return value


def test_exact_six_owner_order_and_only_retry():
    attempts = r.attempt_contract()
    assert [x['original_execution_order'] for x in attempts] == list(range(35, 41))
    assert [x['attempt_number'] for x in attempts] == [2, 1, 1, 1, 1, 1]
    assert len({x['output_namespace'] for x in attempts}) == 6
    assert all('ATTEMPT_' in x['output_namespace'] for x in attempts)
    assert all(int(x['run_id'].split('-')[2]) >= 18 for x in attempts)


def test_order_duplicate_and_terminal_stop(ledger):
    first, second = ledger['attempts'][:2]
    with pytest.raises(RuntimeError, match='ORDER'):
        r.validate_claim(ledger, second['attempt_id'])
    assert r.validate_claim(ledger, first['attempt_id']) == first
    first['status'] = 'RUNNER_ENTRY_CLAIMED'
    with pytest.raises(RuntimeError, match='ORDER'):
        r.validate_claim(ledger, first['attempt_id'])
    ledger['stop'] = {'reason': 'infrastructure'}
    with pytest.raises(RuntimeError, match='TERMINAL'):
        r.validate_claim(ledger, first['attempt_id'])


def test_exhausted_six_and_no_attempt_three(ledger):
    for a in ledger['attempts']:
        a.update(status='ARM_COMPLETE', runner_entry_claimed=True)
    with pytest.raises(RuntimeError):
        r.validate_claim(ledger, ledger['attempts'][0]['attempt_id'])
    with pytest.raises(executor.B1ExecutionError):
        executor.validate_retry_request(3, 'NATIVE_ROUTE_INFRASTRUCTURE_FAILURE', True)
    with pytest.raises(executor.B1ExecutionError):
        executor.validate_retry_request(2, 'SCIENTIFIC_FAIL', True)


def test_failure_status_retained_and_not_reclaimable(ledger):
    a = ledger['attempts'][0]
    a.update(status='RUNNER_ENTRY_CLAIMED', runner_entry_claimed=True)
    r.write_json(r.LEDGER, ledger)
    r.update_status(a['attempt_id'], 'TECHNICAL_INCOMPLETE')
    value = r.load(r.LEDGER)
    assert value['stop'] and value['attempts'][0]['runner_entry_claimed']
    with pytest.raises(RuntimeError, match='TERMINAL'):
        r.validate_claim(value, a['attempt_id'])


@pytest.mark.parametrize('scientific_fail', [False, True])
def test_orchestrator_continues_science_but_stops_infrastructure(tmp_path, monkeypatch, ledger, scientific_fail):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    preflight = tmp_path / 'preflight.json'
    r.write_json(preflight, {'verdict': 'B1_R_PREFLIGHT_PASS'})
    monkeypatch.setattr(r, 'PREFLIGHT', preflight)
    calls = []
    def run(command, **kwargs):
        attempt_id = command[-1]
        calls.append(attempt_id)
        value = r.load(r.LEDGER)
        a = next(a for a in value['attempts'] if a['attempt_id'] == attempt_id)
        a.update(status='ARM_COMPLETE' if scientific_fail else 'TECHNICAL_INCOMPLETE', runner_entry_claimed=True)
        r.write_json(r.LEDGER, value)
        return SimpleNamespace(returncode=0 if scientific_fail else 1)
    monkeypatch.setattr(r.subprocess, 'run', run)
    monkeypatch.setattr(r, 'evaluate_new_pair', lambda _: ({}, {'scientific_status': 'FAIL'}))
    r.execute_all()
    assert len(calls) == (6 if scientific_fail else 1)
    assert bool(r.load(r.LEDGER)['stop']) is not scientific_fail


def test_preflight_rejection_prevents_any_process(tmp_path, monkeypatch, ledger):
    p = tmp_path / 'preflight.json'
    r.write_json(p, {'verdict': 'B1_R_PREFLIGHT_BLOCKED'})
    monkeypatch.setattr(r, 'PREFLIGHT', p)
    monkeypatch.setattr(r.subprocess, 'run', lambda *a, **k: pytest.fail('PROCESS_ENTRY'))
    with pytest.raises(RuntimeError, match='PREFLIGHT_BLOCKED'):
        r.execute_all()


@pytest.mark.parametrize('failing_stage', [None, 'recorder', 'metrics', 'serializer', 'hash'])
def test_single_shared_lifecycle_order(tmp_path, monkeypatch, failing_stage):
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    import nuplan.planning.script.utils as utils
    events, statuses = [], []
    def stage(name, value=None):
        def invoke(*args, **kwargs):
            events.append(name)
            if name == failing_stage:
                raise RuntimeError('MOCK_INFRASTRUCTURE_FAILURE')
            return value
        return invoke
    arm = SimpleNamespace(run_root=tmp_path, runner=SimpleNamespace(run=stage('runner', SimpleNamespace(succeeded=True))),
        recorder=SimpleNamespace(validate_complete=stage('recorder')),
        common_builder=SimpleNamespace(output_dir=tmp_path, multi_main_callback=SimpleNamespace(on_run_simulation_end=stage('callback'))),
        cfg=SimpleNamespace(runner_report_file='report.parquet'))
    monkeypatch.setattr(utils, 'save_runner_reports', stage('runner_report'))
    monkeypatch.setattr(executor, 'validate_official_metric_outputs', stage('metrics'))
    monkeypatch.setattr(executor, 'validate_serializer_complete', stage('serializer'))
    monkeypatch.setattr(executor, 'validate_and_hash_artifacts', stage('hash', {'file': 'digest'}))
    monkeypatch.setattr(executor, 'arm_manifest', lambda spec, arm, auth, status, lifecycle=(), *args, **kw: {'execution_status': status, 'lifecycle_states_complete': list(lifecycle)})
    if failing_stage:
        with pytest.raises(RuntimeError, match='MOCK_INFRASTRUCTURE'):
            executor.complete_production_arm(arm, {'run_id': 'TEST'}, {}, {}, statuses.append, {'attempt_id': 'TEST__ATTEMPT_2'})
        assert statuses == ['TECHNICAL_INCOMPLETE']
    else:
        executor.complete_production_arm(arm, {'run_id': 'TEST'}, {}, {}, statuses.append, {'attempt_id': 'TEST__ATTEMPT_2'})
        assert events == ['runner', 'recorder', 'runner_report', 'callback', 'metrics', 'serializer', 'hash']
        assert statuses == ['ARM_COMPLETE']
        manifest = r.load(tmp_path / 'execution_manifest.json')
        assert manifest['lifecycle_states_complete'] == list(executor.LIFECYCLE_ORDER)
        assert manifest['attempt_id'] == 'TEST__ATTEMPT_2'


def test_inherited_17_never_call_new_evaluator(monkeypatch):
    import tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher as dispatcher
    monkeypatch.setattr(dispatcher, 'evaluate_frozen_pair', lambda **kwargs: pytest.fail('HISTORICAL_REEVALUATION'))
    inherited = report.inherited_pairs()
    assert len(inherited) == 17
    assert all(p['joint_scientific_status'] == p['safety_status'] == 'PASS' for p in inherited)


def test_complete_scientific_fail_not_infrastructure():
    pairs = report.inherited_pairs()
    for i in range(3):
        p = dict(pairs[0], pair_id=f'NEW-{i}')
        pairs.append(p)
    ledger = {'stop': None, 'attempts': [{'runner_entry_claimed': True, 'status': 'ARM_COMPLETE'} for _ in range(6)]}
    assert report.summarize(pairs, ledger, 'B1_R_PREFLIGHT_PASS')['verdict'].endswith('_PASS')
    pairs[-1]['joint_scientific_status'] = 'SCIENTIFIC_FAIL'
    result = report.summarize(pairs, ledger, 'B1_R_PREFLIGHT_PASS')
    assert result['verdict'].endswith('_FAIL') and result['counts']['full_contract_qualification']['n'] == 19
    ledger['stop'] = {'reason': 'infrastructure'}
    assert report.summarize(pairs, ledger, 'B1_R_PREFLIGHT_PASS')['verdict'].endswith('_INCOMPLETE_INFRASTRUCTURE')


def test_source_contract_scientific_files_unchanged_from_ir3():
    names = ['tools/r1_closed_loop_benchmark_v2_1.py', 'tools/r1_context_mechanism_core.py',
             'tools/r2_b_controller_aware_planner_v1.py', 'tools/r2_b_controller_aware_generator_v1.py',
             'tools/r2_bj_b0_2_passive_actual_lqr_recorder.py', 'tools/b1_native_route_infrastructure.py',
             'tools/r1_b2_8_r3_1_official_safety_adapter.py', 'tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py']
    for name in names:
        assert executor._git_blob_matches(r.IR3_COMMIT, name, r.sha(r.ROOT / name))


def test_authorization_rejected_before_factory(monkeypatch):
    monkeypatch.setattr(r, 'load', lambda path: {})
    monkeypatch.setattr(r, 'verify_authorization', lambda auth: (_ for _ in ()).throw(RuntimeError('HASH_MISMATCH')))
    monkeypatch.setattr(r, 'build_fresh_arm', lambda *args: pytest.fail('UNAUTHORIZED_FACTORY'))
    with pytest.raises(RuntimeError, match='HASH_MISMATCH'):
        r.execute_attempt('B1-TSB-18-TREATMENT__ATTEMPT_2')


def test_resolved_config_paths_normalized_but_numerics_not():
    left = {'path': '/one/raw', 'parameter': -2.4}
    right = {'path': '/two/raw', 'parameter': -2.4}
    assert r.normalized_config_hash(left, '/one') == r.normalized_config_hash(right, '/two')
    right['parameter'] = -2.3
    assert r.normalized_config_hash(left, '/one') != r.normalized_config_hash(right, '/two')


def test_safety_scientific_fail_does_not_raise_infrastructure(tmp_path, monkeypatch):
    import pandas as pd
    import tools.r1_b2_8_r3_1_official_safety_adapter as safety
    for name in executor.EXPECTED_OFFICIAL_METRICS:
        path = tmp_path / 'raw/metrics' / f'{name}.parquet'
        path.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame({'value': [0]}).to_parquet(path)
    monkeypatch.setattr(safety, 'adapt_official_safety', lambda _: {'frozen_arm_safety_pass': False})
    executor.validate_official_metric_outputs(SimpleNamespace(run_root=tmp_path))
    assert r.load(tmp_path / 'official_safety_adapter.json')['frozen_arm_safety_pass'] is False


def test_historical_ir3_source_provenance_bound_to_original_commit():
    manifest = r.load(r.ROOT / 'docs/stageR/b1_ir3_route_closure/B1_IR3_Manifest_v1.json')
    for name, expected in manifest['source_sha256'].items():
        assert executor._git_blob_matches(r.IR3_COMMIT, name, expected)


def test_administrative_timestamp_normalization_preserves_labels():
    def config(uid):
        return {'experiment_uid': uid, 'experiment_name': 'production', 'job_name': 'arm',
                'experiment': 'production/arm/' + uid,
                'metric_aggregator': {'official': {'file_name': 'weighted_average_metrics_' + uid}}, 'seed': 2026091801}
    left, right = config('2026.10.23.01.02.03'), config('2026.10.23.04.05.06')
    assert r.normalized_config_hash(left, '/one') == r.normalized_config_hash(right, '/two')
    right['metric_aggregator']['official']['file_name'] = 'other_metric_2026.10.23.04.05.06'
    assert r.normalized_config_hash(left, '/one') != r.normalized_config_hash(right, '/two')
