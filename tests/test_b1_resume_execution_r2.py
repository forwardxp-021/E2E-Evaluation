"""R2 scope, source/record binding, immutable history and stop policy; no scientific run."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import b1_resume_execution_r2 as x
from tools import b1_resume_execution as r
from tools import b1_production_preflight as gate
from tools import b1_resume_report_r2 as report


@pytest.fixture
def ledger(tmp_path,monkeypatch):
    auth=tmp_path/'auth.json'
    r.write_json(auth,{'kind':'TEST_ONLY'})
    monkeypatch.setattr(x,'AUTH',auth)
    monkeypatch.setattr(x,'LEDGER',tmp_path/'ledger.json')
    monkeypatch.setattr(x,'OUT',tmp_path)
    value={'authorization_sha256':r.sha(auth),'historical_attempts':35,'stop':None,
           'attempts':[{**a,'status':'NOT_RUN','runner_entry_claimed':False} for a in x.contracts()]}
    r.write_json(x.LEDGER,value)
    return value


def test_exact_scope_and_new_namespace():
    rows=x.contracts()
    assert [(a['run_id'],a['attempt_number']) for a in rows]==r.TARGETS
    assert len({a['output_namespace'] for a in rows})==6
    assert all(str(x.RAW) in a['output_namespace'] for a in rows)
    assert all(a['output_namespace'] != b['output_namespace'] for a,b in zip(rows,r.attempt_contract()))


def test_order_duplicate_attempt_three_and_budget(ledger):
    first,second=ledger['attempts'][:2]
    assert x.check_pending(ledger,first['attempt_id'])==first
    for bad in (second['attempt_id'],'B1-TSB-18-TREATMENT__ATTEMPT_3','B1-TSB-01-BASELINE__ATTEMPT_1'):
        with pytest.raises(RuntimeError,match='ORDER_DUPLICATE_OR_RETRY'):
            x.check_pending(ledger,bad)
    first.update(status='RUNNER_ENTRY_CLAIMED',runner_entry_claimed=True)
    with pytest.raises(RuntimeError):
        x.check_pending(ledger,first['attempt_id'])
    ledger['stop']={'reason':'infrastructure'}
    with pytest.raises(RuntimeError,match='TERMINAL'):
        x.check_pending(ledger,second['attempt_id'])


def test_stop_retains_consumed_attempt_and_future_not_run(ledger):
    first=ledger['attempts'][0]
    first.update(status='RUNNER_ENTRY_CLAIMED',runner_entry_claimed=True)
    r.write_json(x.LEDGER,ledger)
    x.update_status(first['attempt_id'],'TECHNICAL_INCOMPLETE')
    result=r.load(x.LEDGER)
    assert result['stop'] and sum(a['runner_entry_claimed'] for a in result['attempts'])==1
    assert all(a['status']=='NOT_RUN' for a in result['attempts'][1:])


@pytest.mark.parametrize('scientific_fail',[False,True])
def test_scientific_continue_infrastructure_stop(ledger,tmp_path,monkeypatch,scientific_fail):
    preflight=tmp_path/'preflight.json'
    r.write_json(preflight,{'verdict':'B1_R2_PREFLIGHT_PASS'})
    monkeypatch.setattr(x,'PREFLIGHT',preflight)
    calls=[]
    def process(command,**kwargs):
        attempt=command[-1]
        calls.append(attempt)
        value=r.load(x.LEDGER)
        a=next(a for a in value['attempts'] if a['attempt_id']==attempt)
        a.update(status='ARM_COMPLETE' if scientific_fail else 'TECHNICAL_INCOMPLETE',runner_entry_claimed=True)
        r.write_json(x.LEDGER,value)
        return SimpleNamespace(returncode=0 if scientific_fail else 1)
    monkeypatch.setattr(x.subprocess,'run',process)
    monkeypatch.setattr(x,'evaluate_pair',lambda _: {'scientific_result':'FAIL'})
    x.execute_all()
    assert len(calls)==(6 if scientific_fail else 1)
    assert bool(r.load(x.LEDGER)['stop']) != scientific_fail


def test_blocked_preflight_never_starts_process(ledger,tmp_path,monkeypatch):
    p=tmp_path/'preflight.json'
    r.write_json(p,{'verdict':'B1_R2_PREFLIGHT_BLOCKED'})
    monkeypatch.setattr(x,'PREFLIGHT',p)
    monkeypatch.setattr(x.subprocess,'run',lambda *a,**k:pytest.fail('UNAUTHORIZED_PROCESS'))
    with pytest.raises(RuntimeError,match='PREFLIGHT_BLOCKED'):
        x.execute_all()


def test_old_and_proposed_authorizations_rejected():
    for path in (r.AUTH,rp_draft_path()):
        with pytest.raises(RuntimeError,match='R2_AUTH_NOT_ACTIVE'):
            x.verify_authorization(r.load(path))


def rp_draft_path():
    return x.rp.OUT/'B1_RP_Proposed_B1_R2_Authorization_v1.json'


def test_real_hash_tampering_fails(tmp_path):
    path=tmp_path/'parameter.json'
    path.write_text('frozen')
    correct=r.sha(path)
    x.verify_hash_group({str(path):correct},'TEST')
    path.write_text('tampered')
    with pytest.raises(RuntimeError,match='HASH_MISMATCH'):
        x.verify_hash_group({str(path):correct},'TEST')


def test_worker_rejects_before_factory(ledger,monkeypatch):
    monkeypatch.setattr(x,'verify_authorization',lambda _:(_ for _ in ()).throw(RuntimeError('HASH_MISMATCH')))
    monkeypatch.setattr(x,'build_fresh_arm',lambda *a:pytest.fail('UNAUTHORIZED_FACTORY'))
    with pytest.raises(RuntimeError,match='HASH_MISMATCH'):
        x.execute_attempt(x.contracts()[0]['attempt_id'])


def test_inherited_17_are_read_only(monkeypatch):
    import tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher as dispatcher
    monkeypatch.setattr(dispatcher,'evaluate_frozen_pair',lambda **k:pytest.fail('HISTORICAL_REEVALUATION'))
    assert len(report.inherited.inherited_pairs())==17


def test_all_complete_cannot_pass_with_one_scientific_failure():
    pairs=report.inherited.inherited_pairs()
    pairs += [dict(pairs[0],pair_id=f'NEW-{n}') for n in range(3)]
    ledger={'stop':None,'attempts':[{'status':'ARM_COMPLETE','runner_entry_claimed':True} for _ in range(6)]}
    assert report.inherited.summarize(pairs,ledger,'B1_R_PREFLIGHT_PASS')['verdict'].endswith('_PASS')
    pairs[-1]['joint_scientific_status']='SCIENTIFIC_FAIL'
    assert report.inherited.summarize(pairs,ledger,'B1_R_PREFLIGHT_PASS')['verdict'].endswith('_FAIL')
    ledger['stop']={'reason':'infrastructure'}
    assert report.inherited.summarize(pairs,ledger,'B1_R_PREFLIGHT_PASS')['verdict'].endswith('_INCOMPLETE_INFRASTRUCTURE')


def test_real_factory_matches_pre_frozen_rp_config(tmp_path):
    from omegaconf import OmegaConf
    spec=next(s for s in r.load(r.frozen.SPECS)['arms'] if s['run_id']==r.TARGETS[0][0])
    expected=r.load(x.rp.OUT/'zero_run_evidence/01_B1-TSB-18-TREATMENT.json')['normalized_config_hash']
    with gate.zero_run_guard() as calls:
        arm=x.build_fresh_arm(spec,tmp_path/'static_factory')
        assert gate.validate_production_callback_binding(arm)['status']=='PASS'
        cfg=OmegaConf.to_container(arm.cfg,resolve=True)
        assert r.normalized_config_hash(cfg,arm.run_root)==expected
        assert not any(calls.values())
    assert not list(tmp_path.rglob('*.parquet'))
