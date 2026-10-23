#!/usr/bin/env python3
"""TSB-SR: descriptive review of the forty existing completed B1 traces only.

No qualifier, H feature extractor, detector, MMD, model, or simulator is called.
Historical PASS/FAIL values are inherited verbatim, never adjudicated here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.r1_context_mechanism_core import median3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/tsb_scientific_semantics_review"
B1 = ROOT / "docs/stageR/b1_resume_execution_r2"
PARAMS = ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def git(*args):
    return subprocess.check_output(["git", *args],cwd=ROOT).decode().strip()


def snapshot():
    p=OUT/"evidence/TSB_SR_Protected_Baseline_v1.json"
    if p.exists(): raise FileExistsError(f"Baseline already exists: {p}")
    old=read(ROOT/"docs/stageR/s3_p0_rbr_training_entry/evidence/Protected_Baseline_v1.json")
    paths=set(old['file_sha256'])
    paths.update(str(x.relative_to(ROOT)) for x in (ROOT/"docs/stageR/s3_p0_rbr_training_entry").rglob('*') if x.is_file())
    paths.update(['tools/s3_p0_training_entry_audit.py','tests/test_s3_p0_training_entry_audit.py',
                  'RBR-64 博士研究总体方案 v2.4.md','07_stage5_interaction_design.md'])
    paths.add('outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv')
    for name,h in old['file_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError(f"Prior protected source changed: {name}")
    write(p,{'base_git_head':git('rev-parse','HEAD'),'tracked_dirty':git('status','--porcelain','--untracked-files=no'),
             'quick_reference_initial_sha256':sha(ROOT/'QUICK_REFERENCE.md'),
             'protected_sha256':{name:sha(ROOT/name) for name in sorted(paths)}})


def verify_protected():
    b=read(OUT/"evidence/TSB_SR_Protected_Baseline_v1.json")
    bad=[p for p,h in b['protected_sha256'].items() if sha(ROOT/p)!=h]
    if bad: raise ValueError(f"Protected assets changed: {bad}")
    return {'status':'PASS','protected_file_count':len(b['protected_sha256']),'changed':[]}


def component_bounds(time, values, peak, level, below):
    """Connected threshold support containing a fixed peak; no phase count/gate.

    Interpolate threshold crossings on the observed acceleration curve. Also
    return sampled span separately; do not add nominal dt to end-start.
    """
    t=np.asarray(time,dtype=float);v=np.asarray(values,dtype=float)
    inside=v<=level if below else v>level
    if not inside[peak]: raise ValueError("Selected peak does not meet descriptive threshold")
    left=right=int(peak)
    while left>0 and inside[left-1]:left-=1
    while right+1<len(t) and inside[right+1]:right+=1
    def crossing(i,j):
        return float(t[i]+(level-v[i])*(t[j]-t[i])/(v[j]-v[i]))
    start=crossing(left-1,left) if left>0 else float(t[left])
    end=crossing(right,right+1) if right+1<len(t) else float(t[right])
    return {'first_index':left,'last_index':right,'sample_start_s':float(t[left]),'sample_end_s':float(t[right]),
            'sample_span_s':float(t[right]-t[left]),'crossing_start_s':start,'crossing_end_s':end,
            'crossing_duration_s':end-start,'edge_censored':left==0 or right==len(t)-1}


def describe(time, speed, acceleration=None):
    t=np.asarray(time,dtype=float);v=np.asarray(speed,dtype=float)
    if len(t)!=80 or t.shape!=v.shape or not np.isfinite(v).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError("B1 descriptive input requires 80 finite states and increasing physical time")
    sm=median3(v)
    a=np.gradient(sm,t,edge_order=2) if acceleration is None else np.asarray(acceleration,dtype=float)
    if a.shape!=t.shape or not np.isfinite(a).all():raise ValueError('Invalid existing acceleration vector')
    return t,v,sm,a


def brake_lobe(time, acceleration, start_s, end_s):
    t=np.asarray(time);a=np.asarray(acceleration)
    pool=np.flatnonzero((t>=start_s)&(t<end_s))
    if not len(pool):raise ValueError('No samples in fixed command bracket')
    peak=int(pool[np.argmin(a[pool])])
    return peak,component_bounds(t,a,peak,-.8,True)


def arm_inputs(number,arm,final_row,new_manifest):
    pair=f'B1-TSB-{number:02d}'
    if number<=17:
        trace=ROOT/'outputs/stageR/b1_tsb_qualification_v1'/f'{pair}-{arm}'/'trace/realized_current_ego.jsonl'
        hashes=json.loads(final_row['artifact_hashes'])[arm.lower()]
        expected=hashes[f'{pair}-{arm}/trace/realized_current_ego.jsonl']
    else:
        attempt=2 if number==18 and arm=='TREATMENT' else 1
        trace=ROOT/'outputs/stageR/b1_resume_execution_r2_v1'/f'{pair}-{arm}'/f'ATTEMPT_{attempt}'/'trace/realized_current_ego.jsonl'
        expected=new_manifest['raw_artifact_sha256'][str(trace)]
    if sha(trace)!=expected:raise ValueError(f'Frozen trace hash mismatch: {trace}')
    records=[json.loads(s) for s in trace.read_text().splitlines()]
    if [r['iteration_index'] for r in records]!=list(range(80)):raise ValueError(f'Non-Primary80 trace: {trace}')
    if any(r['primary_measurement_source']!='REALIZED_CURRENT_EGO' for r in records):raise ValueError(f'Wrong source: {trace}')
    states=[r['current_ego'] for r in records]
    t=np.asarray([s['time_us']-states[0]['time_us'] for s in states],dtype=float)/1e6
    v=np.asarray([s['speed_mps'] for s in states],dtype=float)
    provenance={'trace':str(trace.relative_to(ROOT)),'trace_sha256':expected,
                'measurement_source':'FROZEN_TRACE_MEDIAN3_PHYSICAL_GRADIENT_DESCRIPTIVE_ONLY'}
    acc=None
    if number>=18:
        measured=B1/'pair_evaluations'/f'{pair}__measurements.json'
        expected_meas=new_manifest['artifact_sha256'][str(measured.relative_to(ROOT))]
        if sha(measured)!=expected_meas:raise ValueError(f'Frozen measurement hash mismatch: {measured}')
        m=read(measured)[arm];acc=m['acceleration_mps2']
        # Verify the trace is the recorded measurement's source; do not recompute status.
        if np.max(np.abs(np.asarray(m['median3_speed_mps'])-median3(v)))>5.1e-7:
            raise ValueError('Stored measurement speed differs from frozen trace')
        provenance.update(measurement_source='EXISTING_FROZEN_MEASUREMENT_ARRAY',measurement=str(measured.relative_to(ROOT)),measurement_sha256=expected_meas)
    return describe(t,v,acc),provenance


def analyze():
    verify_protected()
    expected={'start_s':1.1,'baseline_brake_mps2':-1.45,'baseline_duration_s':1.8,
        'first_brake_mps2':-2.4,'first_brake_duration_s':.9,'release_mps2':1.4,
        'release_duration_s':1.3,'second_brake_mps2':-2.4,'second_brake_duration_s':.9}
    if read(PARAMS)['parameters']!=expected:raise ValueError('Frozen intervention differs from reviewed schedule')
    rows=list(csv.DictReader((B1/'B1_R2_Final_Pair_Results_v1.csv').open()))
    if [r['pair_id'] for r in rows]!=[f'B1-TSB-{i:02d}' for i in range(1,21)]:raise ValueError('Original20 order differs')
    manifest=read(B1/'B1_R2_Manifest_v1.json');results=[];curves=[];sources={}
    for number,old in enumerate(rows,1):
        baseline,bp=arm_inputs(number,'BASELINE',old,manifest)
        treatment,tp=arm_inputs(number,'TREATMENT',old,manifest)
        bt,bv,bs,ba=baseline;t,v,sm,a=treatment
        # Fixed command boundary brackets, never search alternative windows for better results.
        first,f=brake_lobe(t,a,1.1,2.0)
        second,s=brake_lobe(t,a,3.3,4.2)
        bpeak,b=brake_lobe(bt,ba,1.1,2.9)
        lo,hi=f['last_index']+1,s['first_index']
        if lo>=hi:raise ValueError(f'No descriptive inter-brake interval: {old["pair_id"]}')
        peak=lo+int(np.argmax(a[lo:hi]));positive=component_bounds(t,a,peak,0.,False)
        pstart,pend=positive['crossing_start_s'],positive['crossing_end_s']
        after_peak=lo+int(np.argmax(sm[lo:hi]));trough=lo+int(np.argmin(sm[lo:after_peak+1]))
        row={k:old[k] for k in ['pair_id','session_id','log_id','scenario_token']}
        row.update(frozen_B1_joint_status=old['joint_scientific_status'],frozen_release_fraction=float(old['release_fraction']),
            frozen_second_peak_ratio=float(old['second_peak_ratio']),
            treatment_initial_speed_mps=float(v[0]),baseline_initial_speed_mps=float(bv[0]),
            treatment_preintervention_peak_decel_mps2=float(max(0.,-np.min(a[t<1.1]))),
            baseline_preintervention_peak_decel_mps2=float(max(0.,-np.min(ba[bt<1.1]))),
            treatment_first_peak_decel_mps2=float(-a[first]),treatment_first_peak_time_s=float(t[first]),
            treatment_second_peak_decel_mps2=float(-a[second]),treatment_second_peak_time_s=float(t[second]),
            treatment_positive_peak_mps2=float(a[peak]),treatment_positive_peak_time_s=float(t[peak]),
            treatment_positive_speed_gain_mps=float(np.interp(pend,t,sm)-np.interp(pstart,t,sm)),
            treatment_positive_raw_speed_gain_mps=float(np.interp(pend,t,v)-np.interp(pstart,t,v)),
            treatment_gap_trough_to_peak_gain_mps=float(sm[after_peak]-sm[trough]),
            treatment_gap_gain_from_first_brake_end_mps=float(np.max(sm[lo:hi])-sm[f['last_index']]),
            treatment_gap_trough_time_s=float(t[trough]),treatment_gap_speed_peak_time_s=float(t[after_peak]),
            baseline_peak_decel_mps2=float(-ba[bpeak]),baseline_peak_time_s=float(bt[bpeak]),
            baseline_speed_change_over_treatment_positive_interval_mps=float(np.interp(pend,bt,bs)-np.interp(pstart,bt,bs)),
            treatment_net_speed_change_mps=float(v[-1]-v[0]),baseline_net_speed_change_mps=float(bv[-1]-bv[0]),
            command_first_brake_mps2=-2.4,command_first_duration_s=.9,command_middle_accel_mps2=1.4,
            command_middle_duration_s=1.3,command_second_brake_mps2=-2.4,command_second_duration_s=.9,
            command_baseline_brake_mps2=-1.45,command_baseline_duration_s=1.8,
            baseline_trace=bp['trace'],baseline_trace_sha256=bp['trace_sha256'],
            treatment_trace=tp['trace'],treatment_trace_sha256=tp['trace_sha256'],
            measurement_source=tp['measurement_source'])
        for prefix,support in [('treatment_first_brake',f),('treatment_second_brake',s),('treatment_positive',positive),('baseline_brake',b)]:
            row.update({prefix+'_'+k:val for k,val in support.items() if k not in ['first_index','last_index']})
        results.append(row);curves.append((old['pair_id'],baseline,treatment))
        sources[old['pair_id']]={'BASELINE':bp,'TREATMENT':tp}
    out=OUT/'TSB_SR_20Pair_Acceleration_Characteristics_v1.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
    stats={}
    for key,val in results[0].items():
        if isinstance(val,(int,float)) and not isinstance(val,bool):
            values=np.asarray([r[key] for r in results],dtype=float)
            stats[key]={'n':20,'min':float(values.min()),'median':float(np.median(values)),
                        'max':float(values.max()),'mean':float(np.mean(values)),'sd_ddof1':float(np.std(values,ddof=1))}
    write(OUT/'evidence/TSB_SR_Descriptive_Summary_v1.json',{'statistics':stats,'traces':sources,
        'all_original20_retained':True,'excluded_original_incomplete_18T_attempt1':'RETAINED_HISTORICALLY_NOT_SUBSTITUTED_FOR_COMPLETED_TRACE',
        'scientific_adjudication_invocations':0,'H_feature_invocations':0,'protection':verify_protected()})
    plot(curves)
    return {'pairs':len(results),'positive_peak_range':[stats['treatment_positive_peak_mps2'][k] for k in ['min','max']],
            'protected':verify_protected()}


def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,4,figsize=(16,15),sharex=True,sharey=True)
    for ax,(pair,b,t) in zip(axes.flat,curves):
        ax.plot(b[0],b[3],label='Baseline realized',color='#2874a6',lw=1.5)
        ax.plot(t[0],t[3],label='Treatment realized',color='#c0392b',lw=1.5)
        ax.step([0,1.1,2.,3.3,4.2,7.9],[0,-2.4,1.4,-2.4,0,0],where='post',color='#777777',lw=.9,ls='--',label='Treatment reference schedule')
        ax.axhline(0,color='black',lw=.5);ax.axhline(-.8,color='gray',lw=.5,ls=':')
        ax.axvspan(2.,3.3,alpha=.08,color='orange');ax.set_title(pair,fontsize=10)
        ax.set_xlim(0,7.9);ax.set_ylim(-2.7,1.8);ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.974))
    fig.suptitle('Frozen B1 original20: reference acceleration vs realized speed derivative',fontsize=14,y=.994)
    fig.supxlabel('Physical seconds since first recorded current-ego state')
    fig.supylabel('Acceleration (m/s²); median3 speed + physical-time gradient')
    fig.tight_layout(rect=[.015,.02,1,.95])
    fig.savefig(OUT/'TSB_SR_20Pair_Acceleration_Curves_v1.png',dpi=160)
    plt.close(fig)


def manifest():
    protected=verify_protected();baseline=read(OUT/'evidence/TSB_SR_Protected_Baseline_v1.json')
    sources=[PARAMS,B1/'B1_R2_Final_Pair_Results_v1.csv',B1/'B1_R2_Manifest_v1.json',B1/'B1_R2_Summary_v1.json',
        ROOT/'tools/r1_context_mechanism_core.py',ROOT/'tools/r1_closed_loop_benchmark_v2_1.py',
        ROOT/'tools/r2_b_controller_aware_generator_v1.py',ROOT/'tools/r2_b_controller_aware_planner_v1.py',
        ROOT/'tools/r1_residual_generators.py',ROOT/'tools/r2_b_freeze_controller_aware_development.py',
        ROOT/'tools/s1_protocol_schema.py',ROOT/'tools/stage6l_prepare_context_representation_ablation.py',
        Path(__file__),ROOT/'tests/test_tsb_scientific_semantics_review.py']
    docs=['docs/stageR/r2/R2_A_Controller_Transfer_Identification_Report_v1.md',
          'docs/stageR/r2/R2_A_TSB_Replanning_Transfer_Audit_v1.md',
          'docs/stageR/r2/R2_A_R2B_Generator_Architecture_Decision_v1.md',
          'docs/stageR/r2/R2_B_Controller_Aware_Generator_Development_Report_v1.md',
          'docs/stageR/r2/r2_b_controller_aware_generator_contract_v1.0.json',
          'docs/stageR/r2/r2_a_tsb_excitation_grid_v1.0.json',
          'docs/stageR/r2/r2_b_tsb_calibration_parameter_space_v1.0.json',
          'docs/stageR/b1_tsb_qualification/B1_TSB_Qualification_Protocol_v1.md',
          'docs/stageR/b1_resume_execution_r2/B1_R2_Final_Qualification_Report_v1.md',
          'docs/stageR/b1_resume_execution_r2/B1_R2_Exposure_Ledger_Update_v1.csv',
          'docs/stageR/s1/S1_Scope_and_Claim_Freeze_Draft_v0.1.md',
          'docs/stageR/s1/S1_Handcrafted_Challenger_Contract_Draft_v0.1.md',
          'docs/stageR/s1/S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md',
          'docs/stageR/s1r_owner_amendment/S1R_Owner_Approval_Record_v1.md',
          'docs/stageR/s1r_owner_amendment/S1R_Exposure_Policy_v1.md',
          'RBR-64 博士研究总体方案 v2.4.md']
    sources.extend(ROOT/p for p in docs)
    target=OUT/'TSB_SR_Manifest_v1.json'
    write(target,{'stage':'TSB-SR','base_git_head':baseline['base_git_head'],
        'review_status':'CONSTRUCT_LABEL_REQUIRES_PRECISION_ROUTE_DECISION_PENDING_OWNER',
        'B1_verdict':'B1_TSB_BENCHMARK_QUALIFICATION_PASS','LOW_ORDER_NUISANCE_ELIMINATED':'NOT_ESTABLISHED',
        'TSB_CLEAN_RESIDUAL_TASK':'NOT_ESTABLISHED','scientific_verdicts_recomputed':False,
        'design_parameter_record_commit':'16a931634c2cd7ca7546427728278c2575b6c6df',
        'design_freeze_blob':git('rev-parse','16a931634c2cd7ca7546427728278c2575b6c6df:tools/r2_b_freeze_controller_aware_development.py'),
        'execution_counts':{k:0 for k in ['SIMULATION','RBR_TRAINING','H_FEATURE_EXTRACTION','H_BDD','RBR_BDD','MMD',
                                       'PRIMARY_EVALUATION','V_EXECUTION','C_EXECUTION','MODEL_SELECTION']},
        'protected':protected,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'output_sha256':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=target},
        'self_hash':'bound by final Git tree; excluded from its own hash list',
        'runtime':{'python':sys.version,'numpy':np.__version__}})
    return {'status':'PASS_HASH_BOUND_REVIEW_ONLY','protected':protected}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['snapshot','analyze','manifest'])
    args=p.parse_args();print(json.dumps(globals()[args.action](),ensure_ascii=False))


if __name__=='__main__':main()
