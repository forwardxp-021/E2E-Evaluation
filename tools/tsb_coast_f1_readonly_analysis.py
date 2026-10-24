#!/usr/bin/env python3
"""F1: describe frozen first-pair evidence. No production/analyzer/model calls."""
from __future__ import annotations
import argparse
import csv
from io import StringIO
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'docs/stageR/tsb_coast_p1r_first_pair'
OUT=ROOT/'docs/stageR/tsb_coast_f1_feasibility_closure'
LABEL='POST_HOC_DESCRIPTIVE_ONLY_NOT_FMATCH_QUALIFICATION'
STAGES=[('COMMON',0,11),('FIRST_REFERENCE_BRAKE',11,21),('COAST_REFERENCE_WINDOW',21,39),('SECOND_REFERENCE_BRAKE',39,49),('POST_REFERENCE_TAIL',49,79)]


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def filter_area_identity(u,a_before,a_next,dt=.1,tau=.2):
    """Algebra on recorded arrays, not a dynamics rollout or a counterfactual."""
    u=np.asarray(u);before=np.asarray(a_before);after=np.asarray(a_next)
    if not(len(u)==len(before)==len(after)) or not len(u):raise ValueError('Nonempty matching arrays required')
    realized=float(dt*np.sum(after));command=float(dt*np.sum(u))
    endpoint=float(tau*(before[0]-after[-1]))
    return dict(signed_realized_impulse_mps=realized,signed_command_impulse_mps=command,
                endpoint_filter_term_mps=endpoint,identity_residual_mps=realized-command-endpoint)


def motion_moments(vx,dt=.1):
    """Exact signed discrete speed-loss accounting using right endpoint times."""
    vx=np.asarray(vx,dtype=float);loss=vx[:-1]-vx[1:];right=dt*np.arange(1,len(vx))
    dose=float(loss.sum());moment=float(np.dot(right,loss))
    if dose<=0:raise ValueError('This signed centroid accounting requires positive net deceleration')
    horizon=float(dt*(len(vx)-1));centroid=moment/dose
    distance=float(dt*vx[:-1].sum())
    return dict(horizon_s=horizon,net_speed_loss_mps=dose,signed_right_endpoint_first_moment_m=moment,
                signed_net_loss_centroid_s=centroid,forward_euler_distance_m=distance,
                distance_identity_residual_m=distance-(vx[0]*horizon-horizon*dose+moment))


def decompose_motion(b,t):
    if abs(b['horizon_s']-t['horizon_s'])>1e-12:raise ValueError('Horizon mismatch')
    db,dt=b['net_speed_loss_mps'],t['net_speed_loss_mps']
    mb,mt=b['signed_net_loss_centroid_s'],t['signed_net_loss_centroid_s']
    impulse=-(b['horizon_s']-(mb+mt)/2)*(dt-db)
    timing=(db+dt)/2*(mt-mb)
    return dict(net_loss_delta_T_minus_B_mps=dt-db,signed_centroid_delta_T_minus_B_s=mt-mb,
                dose_component_m=impulse,timing_component_m=timing,sum_m=impulse+timing,
                interpretation='EXACT_SYMMETRIC_ALGEBRAIC_ACCOUNTING_NOT_COUNTERFACTUAL_CAUSAL_EFFECT')


def load_arm(arm,auth):
    root=Path(auth['scientific_output_root'])/('P1R-C2-DEV-01-'+arm)
    trace=read_jsonl(root/'trace/realized_current_ego.jsonl')
    controls=read_jsonl(root/'telemetry/actual_lqr_controller_telemetry.jsonl')
    planner=read_jsonl(root/'telemetry/planner_transfer.jsonl')
    review=json.loads((root/'arm_science_review.json').read_text())
    if len(trace)!=80 or len(controls)!=79 or len(planner)!=80:raise ValueError('Incomplete historical rows')
    time_us=[row['current_ego']['time_us'] for row in trace]
    time=np.asarray([x-time_us[0] for x in time_us])/1e6
    if time_us!=[time_us[0]+i*100000 for i in range(80)]:raise ValueError('Frozen canonical times changed')
    speed=np.asarray([row['current_ego']['speed_mps'] for row in trace])
    xy=np.asarray([[row['current_ego']['rear_axle'][key] for key in ['x','y']] for row in trace])
    u=np.asarray([row['longitudinal_requested_acceleration'] for row in controls])
    before=np.asarray([row['model_acceleration_before'] for row in controls])
    after=np.asarray([row['model_realized_acceleration'] for row in controls])
    vx=np.asarray([controls[0]['speed_before']]+[row['realized_speed'] for row in controls])
    proxy=np.asarray(review['mechanism']['acceleration_from_median3_speed'])
    rows=[r for r in csv.DictReader(StringIO((OLD/'P1R_D_Reference_Command_Realized_v1.csv').read_text())) if r['arm']==arm]
    raw_proxy=np.asarray([float(r['raw_speed_gradient_proxy_mps2']) for r in rows])
    for i,r in enumerate(rows):
        if int(r['iteration'])!=i or float(r['actual_speed_mps'])!=speed[i] or float(r['frozen_median3_speed_gradient_proxy_mps2'])!=proxy[i]:
            raise ValueError('Old curve CSV no longer agrees with actual source layers')
    for i,r in enumerate(controls):
        if r['time_us']!=time_us[i] or r['next_time_us']!=time_us[i+1]:raise ValueError('Command-state temporal misalignment')
    ref=np.asarray([r['reference_acceleration_mps2'][0] for r in planner[:79]])
    return dict(root=root,time=time,time_us=time_us,speed=speed,xy=xy,u=u,before=before,after=after,vx=vx,
                proxy=proxy,raw_proxy=raw_proxy,ref=ref,controls=controls,planner=planner,review=review)


def analyze():
    manifest=json.loads((OLD/'P1R_D_Manifest_v1.json').read_text())
    verified={}
    for group in ['source_config_sha256','data_sha256','analysis_tool_sha256','actual_raw_outputs_sha256','document_and_figure_sha256']:
        for name,digest in manifest[group].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Changed frozen evidence: '+name)
            verified[name]=digest
    auth=json.loads((OLD/'P1R_D_Owner_FirstPair_Authorization_v1.json').read_text())
    ledger_path=ROOT/auth['execution_ledger_path'];ledger=json.loads(ledger_path.read_text())
    if not ledger['stopped'] or len(ledger['attempts'])!=2 or sha256(ledger_path)!=manifest['final_ledger_sha256']:
        raise ValueError('Historical stopped ledger changed')
    data={arm:load_arm(arm,auth) for arm in ['BASELINE','TREATMENT']}
    existing=json.loads((OLD/'evidence/Actual_Descriptive_Summary_v1.json').read_text())
    curve_names=['Actual_Acceleration_Curves','Reference_Command_Realized','Realized_Speed_Comparison','Coast_Window_Detail']
    curve_rows={name:len(list(csv.DictReader(StringIO((OLD/('P1R_D_'+name+'_v1.csv')).read_text())))) for name in curve_names}
    summary={'analysis_label':LABEL,'frozen_verdict':manifest['verdict'],'accepted_control_conclusion':'ZERO_LONGITUDINAL_REQUEST_COAST_IMPLEMENTED_IN_CLOSED_LOOP',
             'original_scientific_analyzer_calls':0,'new_scientific_attempts':0,'formal_pair_qualification':'NOT_EVALUATED_ARM_GATE_FAILED_UNCHANGED',
             'curve_source_row_counts':curve_rows,'arms':{}}
    stages=[];table=[]
    for arm,d in data.items():
        cfg=json.loads((d['root']/'resolved_config.json').read_text())
        # Resolved configuration has controller key; require explicit values, no fallback.
        tracker=cfg['ego_controller']['tracker']
        horizon=tracker['tracking_horizon']*tracker['discretization_time']
        gain=tracker['q_longitudinal'][0]*horizon/(tracker['r_longitudinal'][0]+tracker['q_longitudinal'][0]*horizon*horizon)
        per=[]
        for name,s,e in STAGES:
            row=dict(arm=arm,window=name,start_s=s/10,end_s=e/10,intervals=e-s,
                reference_negative_impulse_mps=float(.1*np.maximum(-d['ref'][s:e],0).sum()),
                command_negative_impulse_mps=float(.1*np.maximum(-d['u'][s:e],0).sum()),
                command_positive_impulse_mps=float(.1*np.maximum(d['u'][s:e],0).sum()),
                model_negative_impulse_mps=float(.1*np.maximum(-d['after'][s:e],0).sum()),
                model_positive_impulse_mps=float(.1*np.maximum(d['after'][s:e],0).sum()),
                signed_net_speed_loss_mps=float(d['vx'][s]-d['vx'][e]),
                maximum_request_deceleration_mps2=float(np.maximum(-d['u'][s:e],0).max()),
                maximum_model_deceleration_mps2=float(np.maximum(-d['after'][s:e],0).max()),
                saved_proxy_negative_trapezoid_mps=float(np.trapz(np.maximum(-d['proxy'][s:e+1],0),d['time'][s:e+1])),analysis_label=LABEL)
            per.append(row);stages.append(row)
        proxy_total=float(np.trapz(np.maximum(-d['proxy'],0),d['time']))
        supports=existing[arm]['prequalification_candidates']
        old_support_dose=sum(x['dose_mps'] for x in supports)
        modelneg=float(.1*np.maximum(-d['after'],0).sum());modelpos=float(.1*np.maximum(d['after'],0).sum())
        a_states=np.r_[d['before'][0],d['after']]
        diag=dict(start_speed_magnitude_mps=float(d['speed'][0]),start_body_vx_mps=float(d['vx'][0]),
            initial_speed_magnitude_minus_body_vx_mps=float(d['speed'][0]-d['vx'][0]),
            terminal_body_vx_mps=float(d['vx'][-1]),minimum_body_vx_mps=float(d['vx'].min()),
            maximum_command_deceleration_mps2=float(np.maximum(-d['u'],0).max()),
            maximum_model_deceleration_mps2=float(np.maximum(-a_states,0).max()),
            maximum_saved_raw_gradient_deceleration_mps2=float(np.maximum(-d['raw_proxy'],0).max()),
            maximum_saved_qualification_proxy_deceleration_mps2=float(np.maximum(-d['proxy'],0).max()),
            saved_proxy_total_negative_trapezoid_mps=proxy_total,
            historical_candidate_supports=supports,historical_candidate_support_dose_sum_mps=old_support_dose,
            proxy_negative_dose_outside_historical_supports_mps=proxy_total-old_support_dose,
            actual_model_negative_impulse_mps=modelneg,actual_model_positive_impulse_mps=modelpos,
            negative_command_impulse_mps=float(.1*np.maximum(-d['u'],0).sum()),
            positive_command_impulse_mps=float(.1*np.maximum(d['u'],0).sum()),
            filter_area_identity=filter_area_identity(d['u'],d['before'],d['after']),motion_moments=motion_moments(d['vx']),
            actual_path_length_m=float(np.linalg.norm(np.diff(d['xy'],axis=0),axis=1).sum()),
            observed_position_minus_euler_speed_integral_m=float(np.linalg.norm(np.diff(d['xy'],axis=0),axis=1).sum()-.1*d['vx'][:-1].sum()),
            lqr_lookahead_s=horizon,lqr_velocity_error_coefficient_per_s=gain,
            max_saved_raw_vs_median_proxy_difference_in_brake_domain_mps2=float(np.max(np.abs(d['raw_proxy'][11:56]-d['proxy'][11:56]))),
            max_model_filter_equation_residual_mps2=float(np.max(np.abs(d['after']-(2/3*d['before']+d['u']/3)))),
            stages=per)
        summary['arms'][arm]=diag
    descriptors=[r for r in csv.DictReader(StringIO((OLD/'P1R_D_FMatch_And_Safety_v1.csv').read_text())) if r['category']=='F_SINGLE_ARM_DESCRIPTIVE_ONLY']
    published=[]
    for row in descriptors:
        bv=Decimal(row['baseline_actual']);tv=Decimal(row['treatment_actual']);delta=tv-bv;budget=Decimal(row['frozen_pair_caliper'])
        r=dict(metric=row['metric'],unit=row['unit'],window='0..7.9s',basis='ALREADY_PUBLISHED_SIX_DECIMAL_DESCRIPTORS',baseline=str(bv),treatment=str(tv),
            signed_T_minus_B=str(delta),absolute_difference=str(abs(delta)),historical_budget=str(budget),descriptive_budget_multiple=str(abs(delta)/budget),
            analysis_label=LABEL,formal_pair_result='NOT_EVALUATED_UNCHANGED',notes='Historical blank F_match fields are never filled')
        table.append(r);published.append(r)
    b,t=data.values();bm,tm=[summary['arms'][arm]['motion_moments'] for arm in data]
    decomposition=decompose_motion(bm,tm);actualdelta=summary['arms']['TREATMENT']['actual_path_length_m']-summary['arms']['BASELINE']['actual_path_length_m']
    decomposition['actual_path_delta_T_minus_B_m']=actualdelta
    decomposition['actual_path_minus_decomposed_delta_residual_m']=actualdelta-decomposition['sum_m']
    decomposition['dose_component_fraction_of_algebraic_total']=decomposition['dose_component_m']/decomposition['sum_m']
    summary['published_descriptor_differences']=published;summary['path_difference_accounting']=decomposition
    paired=[]
    for i in range(80):
        r=dict(episode_time_s=i/10,simulation_time_us=b['time_us'][i],baseline_speed_mps=float(b['speed'][i]),treatment_speed_mps=float(t['speed'][i]),
               speed_difference_T_minus_B_mps=float(t['speed'][i]-b['speed'][i]),
               path_difference_T_minus_B_m=float(np.linalg.norm(np.diff(t['xy'][:i+1],axis=0),axis=1).sum()-np.linalg.norm(np.diff(b['xy'][:i+1],axis=0),axis=1).sum()),
               baseline_net_speed_loss_from_start_mps=float(b['vx'][0]-b['vx'][i]),treatment_net_speed_loss_from_start_mps=float(t['vx'][0]-t['vx'][i]),
               analysis_label=LABEL)
        for arm,d in data.items():
            r[arm.lower()+'_request_mps2']=float(d['u'][i]) if i<79 else ''
            r[arm.lower()+'_model_accel_next_state_mps2']=float(d['after'][i]) if i<79 else ''
            r[arm.lower()+'_inferred_LQR_reference_vx_mps']=float(d['vx'][i]+d['u'][i]/summary['arms'][arm]['lqr_velocity_error_coefficient_per_s']) if i<79 and d['controls'][i]['tracking_mode']!='COAST' else ''
        paired.append(r)
    summary['speed_difference_summary']=dict(min_mps=float((t['speed']-b['speed']).min()),max_mps=float((t['speed']-b['speed']).max()),
        max_time_s=float(b['time'][np.argmax(t['speed']-b['speed'])]),terminal_mps=float(t['speed'][-1]-b['speed'][-1]),
        selected_times={str(i/10):float(t['speed'][i]-b['speed'][i]) for i in [0,11,21,30,39,49,79]})
    for name,s,e in STAGES:
        for metric in ['signed_net_speed_loss_mps','model_negative_impulse_mps','command_negative_impulse_mps','reference_negative_impulse_mps']:
            bv=summary['arms']['BASELINE']['stages'][STAGES.index((name,s,e))][metric];tv=summary['arms']['TREATMENT']['stages'][STAGES.index((name,s,e))][metric]
            table.append(dict(metric=metric,unit='m/s',window=name,basis='ACTUAL_TRANSITIONS_CURRENT_TO_NEXT_TIME',baseline=bv,treatment=tv,signed_T_minus_B=tv-bv,
                absolute_difference=abs(tv-bv),historical_budget='',descriptive_budget_multiple='',analysis_label=LABEL,formal_pair_result='NOT_EVALUATED_UNCHANGED',notes='Disjoint interval contributions; no phase requalification'))
    summary['input_sha256']=verified
    return summary,table,stages,paired


def write_outputs():
    if (OUT/'evidence/F1_Numeric_Analysis_v1.json').exists():raise FileExistsError('F1 one-shot read-only analysis already exists; do not overwrite')
    summary,table,stages,paired=analyze()
    for name,rows in [('F1_PostHoc_Motion_Comparison_v1.csv',table),('F1_Stage_Impulse_Accounting_v1.csv',stages),('F1_Paired_Speed_Difference_v1.csv',paired)]:
        with (OUT/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (OUT/'evidence/F1_Numeric_Analysis_v1.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-once',action='store_true',required=True,help='Read frozen historical arrays and write new F1 analysis; never run original qualification')
    parser.parse_args();write_outputs()
