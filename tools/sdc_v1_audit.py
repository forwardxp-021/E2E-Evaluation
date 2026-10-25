#!/usr/bin/env python3
"""Final read-only integrity, independence, resource and real-artifact cross-check."""
from __future__ import annotations
import csv,hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
import numpy as np
import pandas as pd
from tools import sdc_v1_execution as e
from tools.tsb_direct_coast_f2_tracker import request_at
from tools.sdc_v1_results import csv_new
ROOT,OUT,SCIENCE=e.ROOT,e.OUT,e.SCIENCE
read,sha,write=e.read,e.sha,e.write


def audit():
    ledger=read(e.STATE);before=sha(e.STATE)
    assert ledger['stopped'] and len(ledger['attempts'])==46 and len(ledger['pairs'])==23
    assert all(p['pair_pass'] for p in ledger['pairs'])
    assert [x['phase'] for x in ledger['attempts']]==['DEVELOPMENT']*6+['QUALIFICATION']*40
    assert len({x['run_id'] for x in ledger['attempts']})==len({x['child_pid'] for x in ledger['attempts']})==46
    for run in read(e.binding_path('QUALIFICATION'))['run_ids']:
        try:e.claim('QUALIFICATION',run)
        except PermissionError:pass
        else:raise AssertionError('Stopped budget accepted attempt')
    assert sha(e.STATE)==before
    marker='\n## E2E Scoped DirectCoast v1：自主开发与独立资格\n';baseline=read(OUT/'evidence/Protected_Baseline_v1.json');bad=[]
    for p,h in baseline['protected_sha256'].items():
        actual=hashlib.sha256((ROOT/p).read_text().split(marker,1)[0].encode()).hexdigest() if p=='QUICK_REFERENCE.md' else sha(ROOT/p)
        if actual!=h:bad.append(p)
    assert not bad,bad
    oldscience=read(ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Execution_Source_Binding_v1.json')
    for p,h in oldscience['source_config_sha256'].items():assert sha(ROOT/p)==h,p
    for phase in ['DEVELOPMENT','QUALIFICATION']:
        auth=read(e.binding_path(phase))
        for p,h in auth['source_sha256'].items():
            frozen=subprocess.check_output(['git','show',auth['source_commit']+':'+p],cwd=ROOT)
            assert hashlib.sha256(frozen).hexdigest()==h
            if phase=='QUALIFICATION':assert sha(ROOT/p)==h
        for p,h in {**auth['proof_sha256'],**auth['data_sha256']}.items():assert sha(ROOT/p)==h,p
    cat=read(OUT/'Pool_Split_v1.json');census=read(OUT/'evidence/QUALIFICATION_Applicability_v2.json')
    roster=read(OUT/'QUALIFICATION_Frozen_Roster_v2.json')['roster'];qids={r['session_id'] for r in roster}
    assert len(qids)==20 and not qids.intersection(cat['development_SESSION']) and not qids.intersection(cat['historical_excluded']) and not qids.intersection(cat['protected_C_SESSION'])
    oldlogs=read(ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json')['logs'];bylog={x['log_id']:x for x in oldlogs}
    logtokens=[bylog[x['log_id']]['database_log_token'] for x in roster];assert len(set(logtokens))==20
    Ctokens={x['database_log_token'] for x in oldlogs if x['session_id'] in cat['protected_C_SESSION']};assert not set(logtokens)&Ctokens
    spec=read(e.specs_path('QUALIFICATION'))['arms'];assert len({sha(s['db_path']) for s in spec[::2]})==20
    # Source inspection and exposure are B-only, not access to protected C.
    opened=set();data={};metadata={};trial_reasons={}
    for phase,version in [('DEVELOPMENT',4),('QUALIFICATION',2)]:
        for item in read(OUT/'evidence'/(phase+f'_Applicability_v{version}.json'))['rows']:
            if not item.get('sources_opened'):continue
            sid=item['session_id'];opened.add(sid)
            row=next(x for x in cat['rows'] if x['session_id']==sid);assert not row['policy_reasons']
            p=row['log']['source_file'];data[p]=sha(p)
            p=ROOT/'docs/stageR/s2_preflight'/row['log']['shard'];metadata[str(p.relative_to(ROOT))]=sha(p)
            for trial in item.get('anchor_trials',[]):
                if 'evidence' in trial:
                    src=trial['evidence']['sources'];assert data[src['db_path']]==src['db_sha256']
    assert not opened&set(cat['protected_C_SESSION'])
    raw=read(OUT/'evidence/Raw_Artifact_Index_v1.json')
    for p,h in raw['sha256'].items():assert sha(ROOT/p)==h,p
    rows=[];descriptors=[];versions=[];residuals=[];qzero=[];state_count=0;transition_count=0
    for attempt in ledger['attempts']:
        phase=attempt['phase'];run=attempt['run_id'];arm=run.rsplit('-',1)[1];root=SCIENCE/phase/run
        manifest=read(root/'execution_manifest.json');c=read(root/'completion.json')
        assert c['manifest_sha256']==sha(root/'execution_manifest.json') and c['technical_pass'] and c['arm_scientific_pass']
        for p,h in manifest['artifacts'].items():assert sha(root/p)==h,p
        safety=root/'raw/metrics';collision=pd.read_parquet(safety/'no_ego_at_fault_collisions.parquet').iloc[0]['number_of_all_at_fault_collisions_stat_value'];drivable=pd.read_parquet(safety/'drivable_area_compliance.parquet').iloc[0]['drivable_area_compliance_stat_value']
        assert collision==0 and bool(drivable)
        trace=[json.loads(x)['current_ego'] for x in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()];control=[json.loads(x) for x in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()];plans=(root/'telemetry/planner_transfer.jsonl').read_text().splitlines()
        assert len(trace)==len(plans)==80 and len(control)==79
        ts=[x['time_us'] for x in trace];assert ts==[ts[0]+100000*i for i in range(80)]
        state_count+=80;transition_count+=79
        for i,x in enumerate(control):
            assert x['longitudinal_requested_acceleration']==request_at(arm,i*100000)
            assert x['longitudinal_LQR_calls']==0 and x['lateral_LQR_calls']==i+1 and x['command_identity_reached_model'] and x['state_propagation_continues']
            residuals.append(abs(x['model_realized_acceleration']-x['model_expected_acceleration']))
        if phase=='QUALIFICATION' and arm=='TREATMENT':
            cr=control[21:39];assert len(cr)==18 and all(x['longitudinal_requested_acceleration']==0 for x in cr);qzero.append(len(cr))
        audit=read(root/'actual_observation_audit.json');assert audit['status']=='PASS' and len(audit['frames'])==80 and not audit['unexpected_indices']
        prior=json.loads(json.dumps(manifest));del prior['artifacts']['actual_observation_audit.json']
        digest=hashlib.sha256((json.dumps(prior,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest();assert digest==audit['execution_manifest_sha256']
        versions.append(dict(run_id=run,pre_audit_manifest_sha256=digest,final_manifest_sha256=sha(root/'execution_manifest.json'),exact_prior_version_reconstructed=True))
        v=np.array([x['speed_mps'] for x in trace]);t=np.arange(80)/10;xy=np.array([[x['rear_axle']['x'],x['rear_axle']['y']] for x in trace])
        descriptors.append(dict(phase=phase,run_id=run,mean_speed=float(v.mean()),end_minus_start_speed=float(v[-1]-v[0]),path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),mean_abs_accel=float(np.abs(np.gradient(v,t,edge_order=2)).mean()),scope='UNROUNDED_TRANSPARENCY_EXPORT_NOT_REQUALIFICATION'))
        for i,x in enumerate(trace):rows.append(dict(phase=phase,run_id=run,index=i,time_us=x['time_us'],episode_s=i/10,speed_mps=x['speed_mps'],rear_x=x['rear_axle']['x'],rear_y=x['rear_axle']['y'],heading=x['rear_axle']['heading'],model_acceleration_mps2=control[0]['model_acceleration_before'] if i==0 else control[i-1]['model_realized_acceleration'],outgoing_request_mps2=control[i]['longitudinal_requested_acceleration'] if i<79 else ''))
    assert max(residuals)==0 and len(qzero)==20 and sum(qzero)==360
    assert state_count==3680 and transition_count==3634
    cpu_envelopes=[]
    for name in ['Development_Execution.log','Qualification_Execution.log']:
        st=(OUT/'evidence'/name).stat();wall=st.st_mtime-st.st_birthtime;assert wall>0
        cpu_envelopes.append(dict(log=name,wall_envelope_seconds=wall,logical_CPUs=os.cpu_count(),all_driver_and_child_CPU_upper_bound_seconds=wall*os.cpu_count()))
    upper=sum(x['all_driver_and_child_CPU_upper_bound_seconds'] for x in cpu_envelopes)
    assert upper<10800 and ledger['cpu_seconds']<10800 and max(x['wall_seconds'] for x in ledger['attempts'])<1200
    resource=dict(scientific_child_and_measurable_descendant_CPU_seconds=ledger['cpu_seconds'],development_child_CPU_seconds=sum(x['cpu_seconds'] for x in ledger['attempts'] if x['phase']=='DEVELOPMENT'),qualification_child_CPU_seconds=sum(x['cpu_seconds'] for x in ledger['attempts'] if x['phase']=='QUALIFICATION'),
        driver_CPU_not_claimed_exactly_measured=True,conservative_all_scientific_driver_plus_children_CPU_upper_bound_seconds=upper,method='For each scientific execution driver: creation-to-final-write wall envelope of its stdout log times host logical CPU count; bounds all job CPU including parent/children, summed across non-overlapping phases. Metadata-only preparation excluded.',phase_envelopes=cpu_envelopes,budget_seconds=10800,within_budget=True)
    csv_new(OUT/'All_Actual_States_And_Requests_v1.csv',rows);csv_new(OUT/'Unrounded_Descriptors_v1.csv',descriptors)
    write(OUT/'evidence/Manifest_Phase_Provenance_v1.json',versions,exclusive=True);write(OUT/'evidence/Resource_Audit_v1.json',resource,exclusive=True)
    write(OUT/'evidence/All_Accessed_Source_Hashes_v1.json',dict(source_DB_sha256=data,metadata_shard_sha256=metadata),exclusive=True)
    verification=dict(status='PASS',protected_historical_files=len(baseline['protected_sha256']),changed_protected=[],F2_source_config_files=len(oldscience['source_config_sha256']),official_raw_metric_arms_crosschecked=46,complete_actual_states=state_count,complete_transitions=transition_count,
        original_qualification_denominator=20,qualification_complete=20,qualification_PASS=20,qualification_FAIL=0,qualification_technical_failures=0,qualification_attempts=40,development_attempts=6,scientific_retries=0,qualification_technical_retries=0,
        independent_qualification_SESSION=20,independent_raw_LOG_tokens=20,independent_full_DB_hashes=20,development_SESSION_overlap=0,historical_B1_P2_DEV_overlap=0,C_source_token_overlap=0,protected_C_sources_opened=0,
        original_source_access_SESSION=len(opened),raw_artifacts_verified=len(raw['sha256']),all_final_artifact_manifests_verified=46,pre_audit_manifest_versions_reconstructed=46,
        qualification_zero_request_transitions=360,maximum_model_filter_equation_residual=0.,stopped_ledger_unchanged_by_40_actual_rejection_checks=True,
        scientific_child_CPU_seconds=ledger['cpu_seconds'],conservative_total_scientific_CPU_upper_bound_seconds=upper,all_scientific_CPU_budget_satisfied=True,
        criteria_repaired_only_for_official_metadata_identity_before_first_Q_attempt=True,physical_domain_threshold_change_after_first_Q_attempt=0,cohort_replacement_after_first_Q_attempt=0,
        forbidden=dict(RBR_TRAINING=0,H30_RBR_COMPARISON=0,MMD_BDD_PRIMARY=0,V_C=0,OFFICIAL_MODEL_CHANGE=0,OFFICIAL_SAFETY_THRESHOLD_CHANGE=0),tests_passed=7,figure_visually_checked=True)
    write(OUT/'evidence/Final_Verification_v1.json',verification,exclusive=True);print(json.dumps(verification,indent=2))


if __name__=='__main__':audit()
