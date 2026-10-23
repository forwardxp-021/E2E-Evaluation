"""Regression evidence for S3-P0 findings; never trains or runs nuPlan."""
import struct
import numpy as np
import pytest

from tools import s3_p0_training_entry_audit as audit
from tools.stage5d_context_core import build_context_traj_from_standard_tracks


def test_independent_layout_oracle_detects_current_axis_mix():
    n=np.arange(5*80*15,dtype=np.float32).reshape(5,80,15)
    c=build_context_traj_from_standard_tracks(np.zeros((80,8),np.float32),n)
    assert c[0,23] == n[0,1,0]
    assert c[0,23] != n[1,0,0]
    assert np.count_nonzero(c[:,8:] != n.transpose(1,0,2).reshape(80,75)) == 5970


def test_actual_production_chains_and_physical_oracles(tmp_path, monkeypatch):
    # No optimizer, torch model, simulator, raw loader or H/BDD is constructed.
    monkeypatch.setattr(audit,"OUT",tmp_path)
    r=audit.parity()
    assert len(r['cases']) == 14
    assert r['cases']['straight']['pipelines']['dynamic']['semantic_allclose']
    assert r['cases']['lane_aware']['pipelines']['dynamic']['semantic_allclose']
    assert r['cases']['turn']['pipelines']['legacy']['semantic_yaw_rate_max_error'] > .039
    assert r['cases']['turn']['pipelines']['dynamic']['semantic_allclose']
    assert not r['cases']['irregular_time']['pipelines']['dynamic']['semantic_allclose']
    assert not r['cases']['sideslip']['pipelines']['dynamic']['semantic_allclose']
    assert not r['cases']['invalid_ego']['pipelines']['dynamic']['semantic_allclose']
    assert not r['cases']['slot_switch']['pipelines']['legacy']['semantic_allclose']
    assert r['cases']['slot_switch']['pipelines']['dynamic']['semantic_allclose']
    assert r['cases']['accelerating_gap']['pipelines']['legacy']['semantic_neighbor_accel_max_error'] > .29
    assert r['cases']['accelerating_gap']['pipelines']['dynamic']['semantic_allclose']
    assert not r['cases']['start_rest']['pipelines']['dynamic']['semantic_allclose']
    assert not r['cases']['angular_gap']['pipelines']['legacy']['semantic_allclose']
    assert r['cases']['epoch_time']['distinct_serialized_timestamps'] == 1
    assert not r['layout_oracle']['production_matches_time_slot_channel']
    assert r['closing_oracle']['actual_closing_mps'] == 8.
    assert abs(r['raw33_discretization_oracle']['historical_rms_lateral_accel'] * 10 -
               r['raw33_discretization_oracle']['expected_rms_including_initial_zero']) < 1e-6


def test_wire_reader_and_truncation_guards(tmp_path):
    payload=b'\x2a\x03abc'+b'\x09'+struct.pack('<d',.1)
    fields=list(audit.wire_fields(payload))
    assert fields[0] == (5,2,b'abc')
    assert struct.unpack('<d',fields[1][2])[0] == .1
    for malformed in [b'\x2a\x03a',b'\x09\x00',b'\x00',b'\x0f',b'\x80']:
        with pytest.raises(ValueError): list(audit.wire_fields(malformed))
    with pytest.raises(FileNotFoundError,match='raw directory missing'):
        audit.raw_index(tmp_path/'absent',set())
    with pytest.raises(ValueError,match='requires 51 TFRecord'):
        audit.raw_index(tmp_path,set())


def test_physical_fixture_is_deterministic_and_no_input_alias_mutation():
    a=audit.make_fixture('turn'); b=audit.make_fixture('turn')
    np.testing.assert_array_equal(a[1],b[1])
    assert a[1].shape == (80,8)
    assert np.all(np.diff(a[1][:,7])>0)
