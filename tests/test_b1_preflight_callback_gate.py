"""Real official assembly with deterministic faults; all scientific entrypoints guarded."""
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import b1_production_preflight as gate
from tools import b1_resume_preflight_correction as rp
from tools import b1_resume_execution as r
from tools.s2r_production_executor import build_fresh_arm


@pytest.fixture(scope='module')
def actual_factory(tmp_path_factory):
    base = tmp_path_factory.mktemp('b1_rp_real_factory')
    spec = next(s for s in r.load(r.frozen.SPECS)['arms'] if s['run_id'] == r.TARGETS[0][0])
    with gate.zero_run_guard() as calls:
        arm = build_fresh_arm(spec, base / 'positive')
        yield arm, calls, base
        assert not any(calls.values()), calls


def test_real_production_shared_empty_but_actual_callbacks_bound(actual_factory):
    arm, calls, _ = actual_factory
    evidence = gate.validate_production_callback_binding(arm)
    assert evidence['status'] == 'PASS'
    assert evidence['shared_callback_count'] == 0
    assert len(evidence['simulation_callback_classes']) == 2
    assert any(x.endswith('.MetricFileCallback') for x in evidence['main_callback_classes'])
    assert evidence['runtime_metric_finalization'] == 'NOT_EXECUTED_NOT_PROVEN'
    assert evidence['simulation_uninitialized']
    assert not list(arm.run_root.rglob('*.parquet'))
    assert not list(arm.run_root.rglob('*.pickle.temp'))
    assert not list(arm.run_root.rglob('*.msgpack.xz'))


def test_real_factory_all_faults_block_and_restore(actual_factory):
    arm, _, base = actual_factory
    matrix = rp.fault_matrix(arm, base / 'missing_metric_file_config')
    assert len(matrix) == 16
    assert all(row['actual'] == 'PASS' for row in matrix[:4])
    assert all(row['actual'] == 'BLOCK' and row['real_official_factory'] for row in matrix[4:])
    assert gate.validate_production_callback_binding(arm)['status'] == 'PASS'


def test_incorrect_executor_lifecycle_contract_blocks(actual_factory):
    arm, _, _ = actual_factory
    with patch.object(r.executor, 'LIFECYCLE_ORDER', tuple(reversed(gate.EXPECTED_LIFECYCLE))):
        with pytest.raises(gate.CallbackPreflightError, match='CALLBACK_LIFECYCLE_CONTRACT_MISMATCH'):
            gate.validate_production_callback_binding(arm)


def test_old_actual_authorization_rejected_without_factory_or_budget(actual_factory):
    arm, _, _ = actual_factory
    old_hash = r.sha(r.AUTH)
    ledger_hash = r.sha(r.LEDGER)
    refusal = rp.reject_old_authorization()
    assert refusal['status'] == 'REJECTED' and refusal['factory_calls'] == 0
    assert r.sha(r.AUTH) == old_hash and r.sha(r.LEDGER) == ledger_hash


def test_scientific_runner_and_lifecycle_unchanged():
    assert rp.verify_execution_body_unchanged()['runner_and_lifecycle_source_byte_unchanged']


def test_proposed_authorization_is_not_accepted(actual_factory):
    draft = r.load(r.AUTH)
    draft.update(stage='B1_R2_PROPOSED_ONLY', authorization_status='PROPOSED_ONLY', OWNER_APPROVAL='PENDING', EXECUTION='NOT_AUTHORIZED')
    with pytest.raises(RuntimeError, match='AUTH_STAGE'):
        r.verify_authorization(draft)


def test_preflight_config_validation_without_simulation(actual_factory):
    from omegaconf import OmegaConf
    arm, _, _ = actual_factory
    cfg = OmegaConf.to_container(arm.cfg, resolve=True)
    assert len(r.normalized_config_hash(cfg, arm.run_root)) == 64
    assert cfg['seed'] == 2026091801 and cfg['run_metric'] is True


def test_no_scientific_namespace_created(actual_factory):
    for item in r.attempt_contract():
        assert not Path(item['output_namespace']).exists()
