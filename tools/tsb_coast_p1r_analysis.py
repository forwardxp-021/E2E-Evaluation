#!/usr/bin/env python3
"""Prospective layered proxy analysis, and unchanged four-field F definitions."""
from __future__ import annotations
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.tsb_coast_p1_measurement import validate,measure,f_descriptors,F_LIMITS
from tools.tsb_coast_p1r_recorder import fp_bound


SCIENCE_CONTRACT={
    'version':'P1R_PROXY_PROPOSAL_v1','capability':'ABSTRACT_COAST_PROXY_ONLY',
    'clock':{'frames':80,'transitions':79,'step_us':100000,'physical_integer_anchor':True},
    'coast':{'interval_us':[2100000,3900000],'request_exact_zero':True,'same_command_object_reaches_model':True,
             'state_propagation_required':True,'longitudinal_LQR_and_stopping_calls':0},
    'model':{'accel_filter_tau_s':.2,'equation':'a_next=a_before+dt/(dt+tau)*(u-a_before)',
             'velocity_equation':'vx_next=vx_before+dt*a_next','floating_check_eps_multiplier':64},
    'braking':{'speed_processing':'median3_then_gradient_edge2_physical_time',
        'phase_threshold_mps2':-.6,'minimum_support_s':.4,'minimum_peak_mps2':.8,'minimum_dose_mps':.35,
        'valid_domain_s':[1.1,5.5],'baseline_phase_count':1,'treatment_phase_count':2,'minimum_separation_s':.7},
    'low_speed':{'positive_speed_required':True,'threshold_mps':1.,'maximum_below_threshold_support_s':.5,'reference_floor_rejected':True},
    'near_zero_acceleration_rule':'DIAGNOSTIC_ONLY_NOT_QUALIFICATION',
    'positive_response':'NO_POSITIVE_COAST_REQUEST; RESIDUAL_RESPONSE_MUST_MATCH_BOUND_MODEL',
    'F_match':{'calipers':dict(F_LIMITS),'speed_processing':'raw_gradient_edge2_physical_time',
               'descriptor_decimals':6,'comparison_epsilon':1e-12,'pm_0p01_pressure_role':'SEPARATE_ROBUSTNESS_DIAGNOSTIC'},
    'additional_pair_balance':{'negative_dose_delta_max_mps':.15,'each_treatment_peak_vs_baseline_max_mps2':.20},
    'official_safety':{'at_fault_collisions':0,'drivable_area_compliance':True},
    'failure_policy':'STOP_NO_AUTOMATIC_RETRY_OR_REPLACEMENT',
}


def audit_proxy_response(time_us,speed,transitions,arm):
    """Proposal diagnostic only: Owner approval enforced separately at execution."""
    t,v=validate(time_us,speed)
    old=measure(time_us,speed,arm)
    retained={'LOW_SPEED_ENDSTOP','PHASE_EDGE_CENSORED','ONE_BRAKE_PHASE_NOT_REALIZED','TWO_BRAKE_PHASES_NOT_REALIZED'}
    reasons=[x for x in old['reasons'] if x in retained]
    if len(transitions)!=79 or [x['iteration'] for x in transitions]!=list(range(79)):
        reasons.append('TRANSITION_OBSERVABILITY_INCOMPLETE')
    for i,row in enumerate(transitions):
        if i>=79:break
        if row['time_us']!=time_us[i] or row['next_time_us']!=time_us[i+1] or not row['command_identity_reached_model'] or not row['state_propagation_continues']:
            reasons.append('CLOCK_COMMAND_OR_STATE_GATE_FAIL')
        coast=arm=='TREATMENT' and 21<=i<39
        if coast and (row['tracking_mode']!='COAST' or row['longitudinal_requested_acceleration']!=0. or not row['coast_command_reached_model']):
            reasons.append('NONZERO_OR_UNOBSERVED_COAST_REQUEST')
        if arm=='BASELINE' and row['tracking_mode']=='COAST':reasons.append('BASELINE_ADAPTER_FORBIDDEN')
        before=row['model_acceleration_before'];a=row['model_realized_acceleration'];u=row['longitudinal_requested_acceleration'];tau=row['model_filter_tau_s']
        expected=before+.1/(.1+tau)*(u-before)
        if abs(a-expected)>fp_bound(a,before,u):reasons.append('FILTER_TAIL_INCONSISTENT')
        if abs(row['realized_speed']-(row['speed_before']+.1*a))>fp_bound(row['realized_speed'],row['speed_before']):
            reasons.append('VELOCITY_RESPONSE_INCONSISTENT')
        if coast and before<=0 and a>fp_bound(a,before):reasons.append('UNEXPLAINED_POSITIVE_RESPONSE')
    if arm=='TREATMENT' and len(old['phases'])==2 and old['phases'][1]['start_s']-old['phases'][0]['end_s']<.7-1e-12:
        reasons.append('TWO_BRAKE_SEPARATION_INSUFFICIENT')
    return {'contract_version':'P1R_PROXY_PROPOSAL_v1','approval':'PENDING_OWNER','proxy':'ABSTRACT_COAST_PROXY_ONLY',
        'proposal_diagnostic_pass':not reasons,'reasons':sorted(set(reasons)),'brake_phases':old['phases'],
        'near_zero_core_diagnostic_only':old['gap'],'acceleration_from_median3_speed':old['acceleration_mps2'],
        'active_positive_recovery_rule':'COAST_REQUEST_EXACT_ZERO; REALIZED_RESPONSE_EXPLAINED_BY_BOUND_MODEL'}


def fmatch_pair(time_us,b_speed,b_xy,t_speed,t_xy):
    b=f_descriptors(time_us,b_speed,b_xy);t=f_descriptors(time_us,t_speed,t_xy)
    delta={k:round(abs(t[k]-b[k]),6) for k in F_LIMITS}
    return {'version':'P1R_UNCHANGED_FOUR_F_v1','baseline':b,'treatment':t,'calipers':F_LIMITS,
        'absolute_delta':delta,'pass':all(delta[k]<=F_LIMITS[k]+1e-12 for k in F_LIMITS)}


def metrology_model_internal_bound(speeds,time_us):
    """Conservative float64 extraction/operator bound, NOT continuous-world error."""
    t,v=validate(time_us,speeds)
    # np.gradient is a linear operator on raw speed. Read its actual row norm.
    weights=np.gradient(np.eye(80),t,axis=0,edge_order=2)
    row_norm=float(np.max(np.sum(np.abs(weights),axis=1)))
    extraction=64*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(v))))
    pair_bound=2*row_norm*extraction+1e-6
    return {'scope':'MODEL_STATE_FLOAT_SERIALIZATION_AND_DISCRETE_F_NOT_REAL_WORLD_SENSOR_ERROR',
        'max_gradient_row_L1':row_norm,'conservative_speed_roundoff_bound_mps':extraction,
        'pair_mean_abs_accel_roundoff_and_rounding_bound_mps2':pair_bound,
        'F_caliper_unchanged':.02,'synthetic_pm_0p01_is_actual_noise_bound':False,
        'pressure_test_role_proposed':'SEPARATE_ROBUSTNESS_DIAGNOSTIC_PENDING_OWNER_APPROVAL'}


def analyze_completed_arm(run_root,arm):
    """Future authorized lifecycle only; no learned or handcrafted representation."""
    import json
    from pathlib import Path
    root=Path(run_root)
    trace=[json.loads(x) for x in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()]
    if len(trace)!=80 or [r['iteration_index'] for r in trace]!=list(range(80)):
        raise ValueError('Expected one realized-current-ego row per planner call')
    rows=[json.loads(x) for x in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]
    ts=[r['current_ego']['time_us'] for r in trace];v=[r['current_ego']['speed_mps'] for r in trace]
    mechanism=audit_proxy_response(ts,v,rows,arm)
    planner=[json.loads(x) for x in (root/'telemetry/planner_transfer.jsonl').read_text().splitlines()]
    if len(planner)!=80 or any(r['reference_floor_triggered'] for r in planner):
        mechanism['reasons'].append('REFERENCE_FLOOR_OR_TELEMETRY_GATE_FAIL')
        mechanism['proposal_diagnostic_pass']=False
    safety=json.loads((root/'official_safety_adapter.json').read_text())
    return {'arm':arm,'mechanism':mechanism,'official_safety_pass':safety['frozen_arm_safety_pass'],
        'arm_pass':mechanism['proposal_diagnostic_pass'] and safety['frozen_arm_safety_pass']}


def analyze_completed_pair(baseline_root,treatment_root):
    import json
    from pathlib import Path
    roots=[Path(baseline_root),Path(treatment_root)]
    traces=[[json.loads(x)['current_ego'] for x in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()] for root in roots]
    b,t=traces;ts=[r['time_us'] for r in b]
    if b[:11]!=t[:11]:raise ValueError('Common realized precontext differs before treatment divergence')
    if ts!=[r['time_us'] for r in t]:raise ValueError('Paired physical clocks differ')
    get=lambda trace:([r['speed_mps'] for r in trace],[[r['rear_axle']['x'],r['rear_axle']['y']] for r in trace])
    bv,bxy=get(b);tv,txy=get(t);f=fmatch_pair(ts,bv,bxy,tv,txy)
    reports=[json.loads((root/'arm_science_review.json').read_text()) for root in roots]
    bm,tm=[r['mechanism'] for r in reports]
    dose=lambda m:float(np.trapz(np.maximum(-np.asarray(m['acceleration_from_median3_speed']),0),np.arange(80)/10))
    dose_delta=abs(dose(bm)-dose(tm))
    peaks=[abs(bm['brake_phases'][0]['peak_decel_mps2']-p['peak_decel_mps2']) for p in tm['brake_phases']] if len(bm['brake_phases'])==1 and len(tm['brake_phases'])==2 else [float('inf')]
    additional={'negative_dose_delta_mps':dose_delta,'peak_differences_mps2':peaks,
        'pass':dose_delta<=.15+1e-12 and all(p<=.20+1e-12 for p in peaks)}
    return {'F_match':f,'retained_P1_balance_review':additional,'pair_pass':all(r['arm_pass'] for r in reports) and f['pass'] and additional['pass']}


def production_metrology_audit():
    """Read only existing B1 numerical provenance, never requalify B1 outcomes."""
    import csv
    import hashlib
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    index=root/'docs/stageR/tsb_scientific_semantics_review/TSB_SR_20Pair_Acceleration_Characteristics_v1.csv'
    rows=[];all_speeds=[];gaps=[]
    for pair in csv.DictReader(index.open()):
        for arm in ('baseline','treatment'):
            path=root/pair[arm+'_trace'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=pair[arm+'_trace_sha256']:raise ValueError('Frozen numerical-provenance trace changed')
            trace=[json.loads(x) for x in path.read_text().splitlines()]
            if len(trace)!=80 or [x['iteration_index'] for x in trace]!=list(range(80)):
                raise ValueError('Not completed frozen eighty-state trace')
            speed=[x['current_ego']['speed_mps'] for x in trace];ts=[x['current_ego']['time_us'] for x in trace]
            if not np.isfinite(speed).all() or any(type(t) is not int for t in ts):raise ValueError('Nonfinite or noninteger source')
            exact=json.loads(json.dumps(speed))==speed
            if not exact:raise ValueError('JSON roundtrip loss')
            rows.append({'pair_id':pair['pair_id'],'arm':arm.upper(),'trace':str(path.relative_to(root)),'sha256':digest,
                'states':80,'float64_json_roundtrip_exact':exact,'off_centimeter_per_second_grid':sum(abs(v-round(v,2))>1e-12 for v in speed)})
            all_speeds.extend(speed);gaps.extend(np.diff(ts).tolist())
    if len(rows)!=40:raise ValueError('Expected forty frozen numerical source traces')
    from tools.tsb_coast_p1_measurement import fixture
    ts,b,bxy=fixture('BASELINE');_,t,txy=fixture('TREATMENT')
    clean=abs(f_descriptors(ts,t,txy)['mean_abs_accel']-f_descriptors(ts,b,bxy)['mean_abs_accel'])
    cases=[]
    for bp in range(4):
        for tp in range(4):
            nb=b+np.roll(np.tile([.01,.01,-.01,-.01],20),bp)
            nt=t+np.roll(np.tile([.01,.01,-.01,-.01],20),tp)
            bf=f_descriptors(ts,nb,bxy)['mean_abs_accel'];tf=f_descriptors(ts,nt,txy)['mean_abs_accel']
            cases.append({'baseline_phase':bp,'treatment_phase':tp,'correlated_same_pattern':bp==tp,
                'absolute_F_pair_difference_mps2':abs(tf-bf),'pair_difference_change_mps2':abs(abs(tf-bf)-clean)})
    bound=metrology_model_internal_bound(np.full(80,max(abs(v) for v in all_speeds)),ts)
    source_paths=['tools/r1_official_technical_smoke_planner_v2_2.py','tools/tsb_coast_p1_measurement.py',
        '../nuplan-devkit/nuplan/common/actor_state/dynamic_car_state.py',
        '../nuplan-devkit/nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py']
    return {'status':'MODEL_INTERNAL_DISCRETE_F_NUMERICALLY_RESOLVABLE_OWNER_AMENDMENT_PENDING',
        'scope':'FROZEN_SPEED_PRECISION_ONLY_NO_B1_SCIENTIFIC_RECOMPUTATION',
        'trace_count':40,'speed_values':len(all_speeds),'all_json_roundtrips_exact':True,
        'off_0p01_grid_values':sum(x['off_centimeter_per_second_grid'] for x in rows),
        'historical_cadence_min_us':min(gaps),'historical_cadence_max_us':max(gaps),
        'historical_clock_is_not_P1R_clock_proof':True,'source_records':rows,
        'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths},
        'model_internal_float_bound':bound,'pair_synthetic_pressure_diagnostic':cases,
        'max_synthetic_pair_difference_change_mps2':max(c['pair_difference_change_mps2'] for c in cases),
        'synthetic_pressure_is_proven_production_noise':False,
        'no_new_F_match_or_derivative_rule':True,'scientific_simulation_calls':0}


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrology-audit',required=True,help='New read-only provenance/analytic diagnostic JSON output')
    args=parser.parse_args();result=production_metrology_audit()
    with open(args.metrology_audit,'x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'])
