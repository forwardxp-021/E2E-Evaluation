#!/usr/bin/env python3
"""Read-only actual first-pair evidence, observation audit and scientific plots."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from tools.s2r_production_executor import ROOT,sha256_file,canonical_sha256

OUT=ROOT/'docs/stageR/tsb_coast_p1r_first_pair'
AUTH=OUT/'P1R_D_Owner_FirstPair_Authorization_v1.json'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def arm_root(arm):
    if arm not in ('BASELINE','TREATMENT'):raise ValueError('Only first-pair Baseline/Treatment are allowed')
    return Path(json.loads(AUTH.read_text())['scientific_output_root'])/('P1R-C2-DEV-01-'+arm)


def audit_observations(arm):
    """Inspect trusted, actually produced official serialized history; never run."""
    from nuplan.planning.simulation.simulation_log import SimulationLog
    root=arm_root(arm);manifest=json.loads((root/'execution_manifest.json').read_text())
    changed=[p for p,h in manifest['artifacts'].items() if sha256_file(root/p)!=h]
    if changed:raise ValueError('Actual immutable arm artifact changed: '+repr(changed))
    auth=json.loads(AUTH.read_text());run_id=root.name
    if canonical_sha256(manifest['spec'])!=auth['approved_spec_sha256'][run_id]:raise ValueError('Actual executed spec changed')
    files=list((root/'raw/simulation_log').rglob('*.msgpack.xz'))
    if len(files)!=1:raise ValueError('Expected exactly one official simulation log')
    log=SimulationLog.load_data(files[0]);samples=log.simulation_history.data
    mapping=json.loads((root/'canonical_clock_mapping.json').read_text())['rows']
    if len(samples)!=80:raise ValueError('Official serialized history does not contain exactly 80 frames')
    def objects(observation):
        rows=[];future=0
        for obj in observation.tracked_objects.tracked_objects:
            m=obj.metadata;c=obj.box.center
            rows.append((str(m.token),str(m.track_token),int(m.timestamp_us),float(c.x),float(c.y),float(c.heading)))
            if getattr(obj,'predictions',None):future+=1
        return sorted(rows),future
    def lights(items):
        return sorted((str(x.status),str(x.lane_connector_id),int(x.timestamp)) for x in (items or []))
    evidence=[];unexpected=[]
    for i,sample in enumerate(samples):
        expected=mapping[i];raw=int(log.scenario.get_time_point(i).time_us)
        actual,forecast=objects(sample.observation)
        original,original_forecast=objects(log.scenario.get_tracked_objects_at_iteration(i))
        actual_lights=lights(sample.traffic_light_status)
        original_lights=lights(list(log.scenario.get_traffic_light_status_at_iteration(i)))
        ok=(sample.iteration.index==i and sample.iteration.time_us==expected['simulation_time_us']
            and sample.ego_state.time_us==expected['simulation_time_us'] and raw==expected['raw_log_time_us']
            and str(log.scenario._lidarpc_tokens[i])==expected['lidar_token']
            and actual==original and actual_lights==original_lights and forecast==0 and original_forecast==0)
        if not ok:unexpected.append(i)
        evidence.append(dict(observation_index=i,simulation_time_us=sample.ego_state.time_us,raw_log_time_us=raw,
            raw_minus_sim_us=raw-sample.ego_state.time_us,lidar_token=str(log.scenario._lidarpc_tokens[i]),
            tracked_object_count=len(actual),actual_observation_fingerprint=canonical_sha256(actual),
            source_observation_fingerprint=canonical_sha256(original),future_prediction_objects=forecast,
            original_future_prediction_objects=original_forecast,traffic_light_source_equal=actual_lights==original_lights,
            same_index_observation_values_equal=actual==original,status='PASS' if ok else 'FAIL'))
    result=dict(status='PASS' if not unexpected else 'ACTUAL_OBSERVATION_MAPPING_BLOCKED',arm=arm,
        scope='ACTUAL_OFFICIAL_SERIALIZED_HISTORY_AND_SAME_INDEX_RAW_SOURCE_COMPARISON',
        scientific_reexecution=False,simulation_log_sha256=sha256_file(files[0]),
        execution_manifest_sha256=sha256_file(root/'execution_manifest.json'),changed_artifacts=changed,
        unexpected_indices=unexpected,frames=evidence,
        claim='DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY')
    target=OUT/'evidence'/(arm+'_Actual_Observation_Audit_v1.json')
    with target.open('x') as f:f.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def actual_arrays(arm):
    """Align model states at next_time, and interval commands at current time."""
    import numpy as np
    from tools.tsb_coast_p1_measurement import validate
    root=arm_root(arm)
    manifest=json.loads((root/'execution_manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():
        if sha256_file(root/name)!=digest:raise ValueError('Changed immutable actual artifact: '+name)
    trace=read_jsonl(root/'trace/realized_current_ego.jsonl')
    control=read_jsonl(root/'telemetry/actual_lqr_controller_telemetry.jsonl')
    planner=read_jsonl(root/'telemetry/planner_transfer.jsonl')
    if [r['iteration_index'] for r in trace]!=list(range(80)) or [r['iteration'] for r in control]!=list(range(79)) or len(planner)!=80:
        raise ValueError('Actual 80 state / 79 transition / 80 planner row coverage incomplete')
    ts=[r['current_ego']['time_us'] for r in trace]
    t,v=validate(ts,[r['current_ego']['speed_mps'] for r in trace])
    model=np.asarray([control[0]['model_acceleration_before']]+[r['model_realized_acceleration'] for r in control])
    for i,row in enumerate(control):
        if row['time_us']!=ts[i] or row['next_time_us']!=ts[i+1] or row['model_acceleration_before']!=model[i]:
            raise ValueError('Actual model state time alignment failed')
        if abs(row['realized_speed']-v[i+1])>1e-10:raise ValueError('Realized model/trace speed disagreement')
        if planner[i]['iteration']!=i:raise ValueError('Planner interval index disagreement')
    review=json.loads((root/'arm_science_review.json').read_text())
    from tools.tsb_coast_p1_measurement import measure
    measured=measure(ts,v,arm)
    proxy=np.asarray(review['mechanism']['acceleration_from_median3_speed'])
    if not np.array_equal(proxy,np.asarray(measured['acceleration_mps2'])):
        raise ValueError('Frozen qualification proxy differs from saved scientific review')
    return dict(root=root,time_us=ts,t=t,v=v,model=model,control=control,planner=planner,trace=trace,
        raw_proxy=np.gradient(v,t,edge_order=2),proxy=proxy,review=review,
        xy=np.asarray([[r['current_ego']['rear_axle'][k] for k in ('x','y')] for r in trace]))


def descriptive_summary(data):
    """Read-only per-arm description; deliberately no pair F_match decision."""
    import numpy as np
    from tools.tsb_coast_p1_measurement import ranges,f_descriptors
    result={}
    for arm,d in data.items():
        t,a,v=d['t'],d['proxy'],d['v'];c=d['control'];candidates=[]
        for s,e in ranges(a<=-.6+1e-12):
            if t[e]<1.1-1e-12 or t[s]>5.5+1e-12:continue
            support=float(t[e]-t[s]);peak=float(-np.min(a[s:e+1]));dose=float(np.trapz(np.maximum(-a[s:e+1],0),t[s:e+1]))
            candidates.append(dict(start_s=float(t[s]),end_s=float(t[e]),support_s=support,peak_decel_mps2=peak,dose_mps=dose,
                support_pass=support>=.4-1e-12,peak_pass=peak>=.8-1e-12,dose_pass=dose>=.35-1e-12))
        safety=json.loads((d['root']/'official_safety_adapter.json').read_text())
        r=dict(arm_pass=d['review']['arm_pass'],reasons=d['review']['mechanism']['reasons'],
            qualified_phases=d['review']['mechanism']['brake_phases'],prequalification_candidates=candidates,
            F_descriptors_descriptive_only=f_descriptors(d['time_us'],v,d['xy']),official_safety=safety['canonical_payload'],
            official_safety_pass=safety['frozen_arm_safety_pass'],minimum_speed_mps=float(v.min()),
            start_speed_mps=float(v[0]),end_speed_mps=float(v[-1]),
            maximum_model_filter_equation_error_mps2=max(abs(x['model_realized_acceleration']-x['model_expected_acceleration']) for x in c),
            reference_floor_triggered=any(x['reference_floor_triggered'] for x in d['planner']))
        if arm=='TREATMENT':
            coast=c[21:39];mask=(t>=2.1-1e-12)&(t<=3.9+1e-12)
            near=ranges((np.abs(a)<=.2+1e-12)&mask)
            core=None
            if near:
                s,e=min(near,key=lambda se:(-(t[se[1]]-t[se[0]]),se[0]))
                core=dict(start_s=float(t[s]),end_s=float(t[e]),support_s=float(t[e]-t[s]),
                    raw_speed_drift_mps=float(v[e]-v[s]),old_drift_limit_mps=float(.2*(t[e]-t[s])+.05))
            positive=ranges((a>.2+1e-12)&mask)
            positive_support=max([float(t[e]-t[s]) for s,e in positive] or [0.])
            posdose=float(np.trapz(np.maximum(a[21:40],0),t[21:40]));pospeak=float(a[21:40].max())
            r['coast']=dict(window_s=[2.1,3.9],interval_count=len(coast),all_requests_exact_zero=all(x['longitudinal_requested_acceleration']==0 for x in coast),
                command_identity_all=all(x['command_identity_reached_model'] and x['coast_command_reached_model'] for x in coast),
                state_propagation_all=all(x['state_propagation_continues'] for x in coast),
                min_displacement_m=min(x['displacement_m'] for x in coast),total_displacement_m=sum(x['displacement_m'] for x in coast),
                model_acceleration_at_2p1_mps2=float(d['model'][21]),model_acceleration_at_3p9_mps2=float(d['model'][39]),
                speed_at_2p1_mps=float(v[21]),speed_at_3p9_mps=float(v[39]),speed_drift_mps=float(v[39]-v[21]),
                maximum_zero_request_decay_error_mps2=max(abs(x['model_realized_acceleration']-(2/3)*x['model_acceleration_before']) for x in coast),
                resume_time_s=3.9,resume_mode=c[39]['tracking_mode'],resume_request_mps2=c[39]['longitudinal_requested_acceleration'],
                old_qualified_interbrake_gap=d['review']['mechanism']['near_zero_core_diagnostic_only'],
                nominal_window_diagnostics=dict(scope='FIXED_REFERENCE_COAST_WINDOW_NOT_A_QUALIFIED_INTERBRAKE_GAP',
                    near_zero_core=core,positive_run_support_s=positive_support,positive_dose_mps=posdose,positive_peak_mps2=pospeak,
                    old_positive_recovery_excluded=positive_support>=.3-1e-12 or posdose>.10+1e-12 or pospeak>.40+1e-12),
                descriptive_model_body_x_peaks_mps2=[float(-d['model'][(t>=s)&(t<=e)].min()) for s,e in [(1.1,2.1),(3.9,4.9)]])
        result[arm]=r
    return result


def export_actual_plots():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not font_path.exists():raise FileNotFoundError('Chinese chart font missing: '+str(font_path))
    font=FontProperties(fname=str(font_path))
    matplotlib.font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'P1R_D_ACTUAL_v1'})
    data={a:actual_arrays(a) for a in ('BASELINE','TREATMENT')}
    summary=descriptive_summary(data)
    colors={'BASELINE':'#2468a2','TREATMENT':'#d35a31'}
    def windows(ax,labels=False):
        for s,e,color,label in [(1.1,2.1,'#f0d8be','第一次制动参考窗'),(2.1,3.9,'#d4e9df','零纵向请求窗'),(3.9,4.9,'#f0d8be','第二次制动参考窗')]:
            ax.axvspan(s,e,color=color,alpha=.45,label=label if labels else None)
        ax.axhline(0,color='#555555',lw=.8);ax.grid(alpha=.22);ax.set_ylabel('加速度（m/s²）')
    def save(fig,name,rows):
        fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)
        svg=OUT/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    allrows=[]
    for arm,d in data.items():
        for i,t in enumerate(d['t']):
            row=dict(arm=arm,iteration=i,simulation_time_us=d['time_us'][i],episode_time_s=float(t),
                actual_speed_mps=float(d['v'][i]),actual_model_body_x_accel_mps2=float(d['model'][i]),
                raw_speed_gradient_proxy_mps2=float(d['raw_proxy'][i]),frozen_median3_speed_gradient_proxy_mps2=float(d['proxy'][i]),
                rear_axle_x_m=float(d['xy'][i,0]),rear_axle_y_m=float(d['xy'][i,1]),
                current_interval_planner_ref_mps2=d['planner'][i]['reference_acceleration_mps2'][0] if i<79 else '',
                final_longitudinal_request_mps2=d['control'][i]['longitudinal_requested_acceleration'] if i<79 else '',
                tracking_mode=d['control'][i]['tracking_mode'] if i<79 else 'NO_NEXT_TRANSITION',
                longitudinal_tracking_enabled=(d['control'][i]['tracking_mode']!='COAST') if i<79 else '',
                steering_request=d['control'][i]['steering_request'] if i<79 else '')
            allrows.append(row)
    fig,ax=plt.subplots(figsize=(12.5,5.4),layout='constrained')
    windows(ax,True)
    for arm,d in data.items():
        zh='基线' if arm=='BASELINE' else '处理'
        ax.plot(d['t'],d['model'],color=colors[arm],lw=2,label=zh+'：官方模型实际加速度')
        ax.plot(d['t'],d['proxy'],color=colors[arm],ls='--',alpha=.85,label=zh+'：冻结资格测量代理')
    ax.axhline(-.8,color='#777777',lw=.9,ls=':',label='资格峰值线 −0.80（仅用于测量代理）')
    ax.set(xlim=(0,7.9),xlabel='规范物理时间（秒）',title='DEV-01 真实闭环加速度：两次减速起伏存在，但处理臂没有合格制动段')
    ax.legend(loc='lower right',ncol=2,fontsize=9)
    save(fig,'P1R_D_Actual_Acceleration_Curves_v1',allrows)
    fig,axes=plt.subplots(2,1,figsize=(12.5,8),sharex=True,layout='constrained')
    for ax,(arm,d) in zip(axes,data.items()):
        windows(ax)
        ref=[r['reference_acceleration_mps2'][0] for r in d['planner'][:79]]
        u=[r['longitudinal_requested_acceleration'] for r in d['control']]
        ax.step(d['t'],ref+[ref[-1]],where='post',ls=':',color='#777777',label='当前区间参考（非预瞄请求）')
        ax.step(d['t'],u+[u[-1]],where='post',color='#9362a4',lw=1.4,label='最终纵向请求（区间起点）')
        ax.plot(d['t'],d['model'],color=colors[arm],lw=2,label='官方模型实际状态（状态时刻）')
        ax.plot(d['t'],d['raw_proxy'],color='#2f8b72',ls='--',label='真实原始速度梯度代理')
        ax.set_title(('基线' if arm=='BASELINE' else '处理')+'：参考、控制与响应分层对照');ax.legend(ncol=2,fontsize=9)
    axes[-1].set(xlim=(0,7.9),xlabel='规范物理时间（秒）')
    save(fig,'P1R_D_Reference_Command_Realized_v1',allrows)
    fig,ax=plt.subplots(figsize=(12.5,5),layout='constrained')
    for s,e,c in [(1.1,2.1,'#f0d8be'),(2.1,3.9,'#d4e9df'),(3.9,4.9,'#f0d8be')]:ax.axvspan(s,e,color=c,alpha=.45)
    for arm,d in data.items():ax.plot(d['t'],d['v'],color=colors[arm],lw=2,label='基线实际速度' if arm=='BASELINE' else '处理实际速度')
    ax.set(xlim=(0,7.9),xlabel='规范物理时间（秒）',ylabel='实际速度（m/s）',title='DEV-01 真实速度：零请求中仍有衰减的残余减速')
    ax.grid(alpha=.22);ax.legend()
    save(fig,'P1R_D_Realized_Speed_Comparison_v1',allrows)
    d=data['TREATMENT'];u=[r['longitudinal_requested_acceleration'] for r in d['control']]
    mode=[r['tracking_mode']!='COAST' for r in d['control']]
    fig,(ax0,ax)=plt.subplots(2,1,figsize=(12.5,7),sharex=True,height_ratios=(1,2.5),layout='constrained')
    for aa in (ax0,ax):
        aa.axvspan(2.1,3.9,color='#d4e9df',alpha=.6);aa.axvline(3.9,color='#555555',ls='--');aa.grid(alpha=.2)
    ax0.step(d['t'],mode+[mode[-1]],where='post',color='#2468a2',lw=2)
    ax0.set(yticks=[0,1],yticklabels=['纵向跟踪停用／零请求','官方纵向跟踪启用'],ylim=(-.3,1.3),title='处理臂滑行细节：[2.1, 3.9) 秒零请求；3.9 秒恢复纵向跟踪')
    ax.step(d['t'],u+[u[-1]],where='post',color='#9362a4',lw=1.7,label='最终纵向请求')
    ax.plot(d['t'],d['model'],color=colors['TREATMENT'],lw=2,label='官方模型实际加速度')
    ax.plot(d['t'],d['raw_proxy'],color='#2f8b72',ls='--',label='真实速度梯度代理')
    ax.axhline(0,color='#555555',lw=.8);ax.set(xlim=(1.8,4.2),xlabel='规范物理时间（秒）',ylabel='加速度（m/s²）');ax.legend(loc='lower left')
    save(fig,'P1R_D_Coast_Window_Detail_v1',[r for r in allrows if r['arm']=='TREATMENT' and 1.8-1e-12<=r['episode_time_s']<=4.2+1e-12])
    source={}
    for d in data.values():
        for name in ['trace/realized_current_ego.jsonl','telemetry/actual_lqr_controller_telemetry.jsonl','telemetry/planner_transfer.jsonl','arm_science_review.json','execution_manifest.json']:
            path=d['root']/name;source[str(path.relative_to(ROOT))]=sha256_file(path)
    provenance=dict(status='ACTUAL_DATA_ONLY',plot_source_sha256=source,executable_source_commit=json.loads(AUTH.read_text())['executable_source_commit'],
        authorization_sha256=sha256_file(AUTH),plot_tool_sha256=sha256_file(Path(__file__)),
        model_time_alignment='MODEL_TRANSITION_OUTPUT_AT_NEXT_TIME; COMMAND_AT_CURRENT_INTERVAL_START',
        final_frame_new_command=False,shading='FIXED_REFERENCE_WINDOWS_NOT_QUALIFIED_PHASES',
        joint_F_match='NOT_EVALUATED_ARM_GATE_FAILED',scientific_reexecution=False)
    (OUT/'P1R_D_Plot_Source_Provenance_v1.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    (OUT/'evidence/Actual_Descriptive_Summary_v1.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit-arm',choices=['BASELINE','TREATMENT'],help='One-shot audit of an already serialized arm; refuses overwrite')
    mode.add_argument('--plots',action='store_true',help='Read existing immutable actual first-pair outputs; export four figure/CSV groups, never run science')
    args=parser.parse_args()
    result=export_actual_plots() if args.plots else audit_observations(args.audit_arm)
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if args.plots or result['status']=='PASS' else 2)
