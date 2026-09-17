import copy
import hashlib
import json
from pathlib import Path
import pytest
from tools.s2_preflight_metadata_census import session, speed_gate
from tools.s2_preflight_validate import select_q20, validate_draft, validate_fresh_root, verify_manifest
ROOT=Path(__file__).resolve().parents[1]

def candidates(n=20):
    return [{'log_id':f'FIXTURE_LOG_{i}','scenario_token':f'FIXTURE_TOKEN_{i}','session_id':f'FIXTURE_SESSION_{i}','eligibility':'PASS','exclusion_reasons':[],'evidence_sha256':'a'*64} for i in range(n)]

def test_nominal_speed_inclusive_and_nonfinite():
    assert speed_gate(3.61)
    assert not any(speed_gate(x) for x in (3.609999,0,float('nan'),float('inf'),True,None))

def test_session_groups_chunk_offsets_without_inventing_driver():
    assert session('2021.05.12.22.00.38_veh-35_01008_01518')==session('2021.05.12.22.00.38_veh-35_02000_02500')
    assert session('unknown') is None

def test_selection_is_order_invariant_and_session_disjoint():
    rows=candidates(23); rows[1]['session_id']=rows[0]['session_id']
    a=select_q20(rows); assert a==select_q20(list(reversed(rows)))
    assert len({r['session_id'] for r in a})==20

def test_no_q12_fallback_and_no_duplicate_or_outcome_input():
    for n in (0,12,19):
        with pytest.raises(ValueError,match='Q20_CAPACITY'): select_q20(candidates(n))
    rows=candidates(); rows[1]['scenario_token']=rows[0]['scenario_token']
    with pytest.raises(ValueError,match='DUPLICATE'): select_q20(rows)
    rows=candidates(); rows[0]['realized_safety']=True
    with pytest.raises(ValueError,match='NO_OUTCOMES'): select_q20(rows)
    rows=candidates(); rows[0]['eligibility']='UNKNOWN'
    with pytest.raises(ValueError,match='UNCERTIFIED'): select_q20(rows)

def test_closed_budget_missing_wrong_and_active_reject():
    value=json.loads((ROOT/'docs/stageR/s2_preflight/S2_Preflight_Budget_Authorization_DRAFT_v1.json').read_text())
    result=validate_draft(value); assert result['scientific_qualification'] is False and result['execution_possible'] is False
    for key,val in [('active_budget',40),('RUN_BUDGET_ACTIVE',True),('S2_EXECUTION_AUTHORIZED',True),('consumed_entries',True)]:
        bad=copy.deepcopy(value); bad[key]=val
        with pytest.raises(ValueError): validate_draft(bad)
    bad=copy.deepcopy(value); del bad['active_budget']
    with pytest.raises(ValueError,match='EXACT_KEYS'): validate_draft(bad)
    bad=copy.deepcopy(value); bad['zero_run_counters']['RUNNER_RUN_CALLS']=1
    with pytest.raises(ValueError,match='ZERO_RUN_COUNTERS'): validate_draft(bad)

def test_output_freshness_and_symlink_rejection(tmp_path):
    validate_fresh_root(tmp_path/'new',tmp_path)
    for target in (tmp_path,tmp_path.parent):
        with pytest.raises(ValueError): validate_fresh_root(target,tmp_path)
    (tmp_path/'used').mkdir()
    with pytest.raises(ValueError): validate_fresh_root(tmp_path/'used',tmp_path)
    (tmp_path/'link').symlink_to(tmp_path/'new')
    with pytest.raises(ValueError): validate_fresh_root(tmp_path/'link',tmp_path)

def test_hash_fail_closed_without_any_scientific_result(tmp_path):
    p=tmp_path/'fixture'; p.write_bytes(b'only static fixture')
    m={'bindings':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}],'scientific_qualification':False}
    assert verify_manifest(tmp_path,m)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA_MISMATCH'): verify_manifest(tmp_path,m)
