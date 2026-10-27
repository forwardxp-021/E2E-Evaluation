#!/usr/bin/env python3
"""D6 source verification, metadata-only capacity summaries, report provenance."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,csv,math
from pathlib import Path
from tools.rbr_v2_d6_static import ROOT,OUT,PLAN,read,write,check,table
from tools.rbr_v2_d1_interface import sha
DOC=ROOT/'docs/stageR/rbr_v2_d6_v1'


def audit():
    check();r=read(OUT/'static_census.json');verified=[]
    for sid,d in r['initial_DB_read_provenance'].items():
        actual=sha(d['path'])
        if d['raw_sha256'] and actual!=d['raw_sha256']:raise ValueError('Raw source SHA changed '+sid)
        verified.append(dict(session_id=sid,path=d['path'],sha256=actual,prior_source_hash_matched=bool(d['raw_sha256'])))
    if len({v['header_token'] for v in r['initial_DB_read_provenance'].values()})!=len(verified):raise ValueError('LOG token aliases distinct SESSION')
    maps=[p for p in (ROOT.parent/'nuplan/dataset/maps').rglob('*.gpkg') if 'us-nv-las-vegas-strip' in str(p)]
    write(OUT/'direct_source_verification.json',dict(raw_initial_frame_LOGs=verified,map_files={str(p):sha(p) for p in maps},map_index_sha256=sha(ROOT.parent/'nuplan/dataset/maps/nuplan-maps-v1.0.json'),network_requests=0,raw_C_access=0))
    rows=list(csv.DictReader((OUT/'SESSION_Task_And_Role_Census.csv').open()));strictrole=['final_calibration_candidates','V_AA_candidates','V_AB_candidates','unallocated_candidates'];roles=read(ROOT/'outputs/rbr_v2_d2_v1/candidate_role_partition_final.json');pending={v['session_id'] for v in roles['final_calibration_candidates'] if v['source']=='D1_E3_STATIC_D_ONLY'}
    rolecounts={role:{c:sum(x['prior_role']==role and x['session_id'] not in pending and x['status']=='ELIGIBLE_Q' and x['cell']==c for x in rows) for c in r['strict_future_cells']} for role in strictrole}
    newids={p.stem for p in (ROOT/'outputs/rbr_v2_d2_v1/static').glob('*.json')};newq=sum(x['session_id'] in newids and x['status']=='ELIGIBLE_Q' for x in rows);coarse=[]
    for lo,hi in [(5,10),(10,15)]:
        for role in ['D_new_science','final_calibration_candidates','V_AA_candidates','V_AB_candidates']:
            rr=[x for x in rows if x['prior_role']==role and x['map_name']=='us-nv-las-vegas-strip' and lo<=float(x['initial_speed_mps'])<hi and x.get('actor_count50','')!='']
            if not rr:continue
            topology=sum(x['lane_count']=='1' and x['connector_count']=='0' and x['out_degree']=='1' for x in rr)
            coarse.append(dict(speed_bin=[lo,hi],role=role,n=len(rr),unique_lane_single_successor_fraction=topology/len(rr),actor_count50_range=[min(int(x['actor_count50']) for x in rr),max(int(x['actor_count50']) for x in rr)],route_heading_range=[min(float(x['route_heading_range_rad']) for x in rr),max(float(x['route_heading_range_rad']) for x in rr)]))
    missing={k:sum(x.get(k,'')=='' for x in rows if x['map_name']=='us-nv-las-vegas-strip') for k in ['lane_count','connector_count','out_degree','lane_speed_limit_mps','nearest_connector_distance_m','actor_count50']}
    cost=read(ROOT/'docs/stageR/rbr_v2_d2_v1/Capacity_And_Resource_Summary_v1.json')
    # Locate the already recorded mean without assuming a changed summary schema.
    def find(obj,key):
        if isinstance(obj,dict):
            if key in obj:return obj[key]
            for v in obj.values():
                value=find(v,key)
                if value is not None:return value
        return None
    mean=find(cost,'mean_compressed_bytes_per_downloaded_LOG');rate=newq/len(newids);busy_deficit=160-r['strict_unexposed_future_q'];session_scenario=math.ceil(busy_deficit/rate)
    downloaded_logs=len(list((ROOT/'outputs/rbr_v2_d2_v1/static').glob('*.json')))+len(list((ROOT/'outputs/rbr_v2_d2_v1/round2/static').glob('*.json')))
    logs=math.ceil(session_scenario*downloaded_logs/len(newids))
    write(OUT/'capacity_details.json',dict(strict_future_role_cells=rolecounts,source_counts_before_q=dict(D_fit=3,D_development_complete=60,pending_E3_calibration=10,strict_unexposed_future=151,protected_C=25),source_counts_after_q=dict(D_fit=sum(x['prior_role']=='D_fit' and x['status']=='ELIGIBLE_Q' for x in rows),D_development_complete=sum(x['prior_role'] in ['D_existing_swap_cal','D_new_science'] and x['status']=='ELIGIBLE_Q' for x in rows),pending_E3_calibration=sum(x['session_id'] in pending and x['status']=='ELIGIBLE_Q' for x in rows),strict_unexposed_future=r['strict_unexposed_future_q']),strict320_shortfall=320-r['strict_unexposed_future_q'],optimistic320_shortfall=320-r['optimistic_including_pending_E3_cal_q'],q_loss_total=224-r['q_eligible'],strict_future_loss=151-r['strict_unexposed_future_q'],D2_scanned_distinct=len(newids),D2_new_q=newq,D2_q_yield=rate,D2_empty_cell_observed_count=0,finite_download_cost_for_full_q='NOT_ESTABLISHED: two positive target-weight cells have zero observed yield; no defensible finite extrapolation',covered_busy_component_cost_only=dict(missing_sources=busy_deficit,SESSION_scenario=session_scenario,raw_LOG_scenario=logs,observed_LOGs_per_SESSION=downloaded_logs/len(newids),compressed_GB_scenario=logs*mean/1e9,assumed_rate=rate,assumed_compressed_bytes_per_LOG=mean,not_full_q_cost=True,not_authorized=True),missing_fields_in_target_city=missing,coarse_city_speed_role_differences=coarse))
    print('D6 direct source hash and capacity audit complete')


def package():
    check();c=read(OUT/'capacity_details.json');n=read(OUT/'numerical_and_capacity.json');p=read(PLAN)
    write(DOC/'Status_v1.json',dict(status='D6_UNPAIRED_TASK_BDD_NOT_FEASIBLE_UNDER_CURRENT_RESOURCES',decision='NO_GO',q_common_support=False,true_AA_currently_constructible=False,numerical_pipeline_certified=False,source_CI_power_coverage='NOT_ESTABLISHED',q_all_static=40,q_strict_future=29,new_sources_authorized=0,new_scientific_execution=0,SAP_activated=False))
    write(DOC/'D6_Method_Contract_PROPOSED_v1.json',dict(status='PROPOSED_ONLY',scientific_execution_authorized=False,task=p['task'],cells=['5to10_empty','5to10_busy','10to15_empty','10to15_busy'],weights=[.25]*4,m_each_side=20,independent_SESSION_per_cell_per_side=5,required_unique_SESSION_each_cell_each_pool=10,AA=dict(same_frozen_version=True,target_treatment_count=0,reference_target_SESSION_overlap=0,external_calibration_evaluation_SESSION_overlap=0),AB=dict(target_treatment_count=10,assignment_seed=2026091103),calibration=dict(draws=1999,seed=2026091101,order_rank_1based=1900,comparison='strict_gt',nominal_alpha=.05),statistic='Original unbiased-form quadratic MMD2 and RBF; no paired cross-source identity in either side',representation=dict(checkpoint='outputs/rbr_v2_prod_v1/run/epoch30.pt',checkpoint_sha256=p['protected']['outputs/rbr_v2_prod_v1/run/epoch30.pt'],normalizer_sha256=p['protected']['outputs/rbr_v2_prod_v1/normalizer.json'],H30_raw_features='unchanged30',scaler_status='BLOCKED_NO_NEW_PHYSICAL_FLOOR_ADOPTED'),preflight='tools.rbr_v2_d6_contract.validate_pool; rejects missing/under10 cells, duplicate SESSION and V/C roles',executable_science_entry=None,blockers=['two empty q cells','only29 strict future sources','only6 exact-version D3 baseline sources within q','numerical preprocessing not certified','source bootstrap/CI/power not certified'],no_empty_cell_renormalization=True))
    before=(OUT/'user_Quick_before.txt').read_bytes();section=(OUT/'quick_section.md').read_bytes()
    if (ROOT/'QUICK_REFERENCE.md').read_bytes()!=before+section:raise ValueError('User Quick prefix changed')
    csvpath=ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'
    if sha(csvpath)!='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8':raise ValueError('Protected CSV changed')
    write(OUT/'protection.json',dict(user_Quick_prefix_unchanged=True,protected_CSV_sha256=sha(csvpath),old_SAP_model_H30_changed=False,C_raw_access=False,V_C_results_access=False))
    print('D6 NO_GO method contract packaged; no science entry enabled')


def manifest():
    check();target=DOC/'Manifest_v1.json';paths=list(OUT.rglob('*'))+list(DOC.rglob('*'))+list((ROOT/'configs/rbr_v2_d6_v1').rglob('*'))+list((ROOT/'tools').glob('rbr_v2_d6_*.py'))+list((ROOT/'tests').glob('test_rbr_v2_d6_*.py'));files={}
    for p in paths:
        if p.is_file() and p!=target and '__pycache__' not in p.parts and p.name!='user_Quick_before.txt' and not p.name.startswith('manifest_packaging'):files[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    write(target,dict(status='D6_UNPAIRED_TASK_BDD_NOT_FEASIBLE_UNDER_CURRENT_RESOURCES',files=files,protected=read(PLAN)['protected'],static_metadata_sources=read(OUT/'static_census.json')['static_sources'],numeric_read_sources=read(OUT/'numerical_and_capacity.json')['protected_read_sources'],source_commit=read(PLAN)['start_commit'],network_download_bytes=0,vehicle_propagation=0,new_model_forward=0,new_training=0,Primary_V_C=0))
    print('D6 provenance bound',len(files),'own files')


def verify():
    check();m=read(DOC/'Manifest_v1.json')
    for p,h in m['files'].items():
        if sha(ROOT/p)!=h['sha256']:raise ValueError('D6 output changed '+p)
    for key in ['static_metadata_sources','numeric_read_sources']:
        for p,h in m[key].items():
            path=Path(p) if Path(p).is_absolute() else ROOT/p
            if sha(path)!=h:raise ValueError('Direct historical evidence changed '+p)
    print('D6 verification PASS:',len(m['files']),'own hashes; directly relevant source metadata/numerical evidence and frozen model unchanged; NO_GO retained')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit','package','manifest','verify']);a=p.parse_args();{'audit':audit,'package':package,'manifest':manifest,'verify':verify}[a.mode]()
