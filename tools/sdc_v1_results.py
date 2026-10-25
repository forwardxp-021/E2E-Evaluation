#!/usr/bin/env python3
"""Read actual SDC records, retain full denominator, export fair benchmark evidence."""
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sdc_v1_execution import ROOT,OUT,STATE,SCIENCE,read,write,sha


def csv_new(path,rows,fields=None):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def export():
    import numpy as np
    ledger=read(STATE);attempts={x['run_id']:x for x in ledger['attempts']};pairs={x['pair_id']:x for x in ledger['pairs']}
    rows=[];arms=[];fm=[];exposure=[];coverage=[];rejections=[]
    for phase,revision in [('DEVELOPMENT',4),('QUALIFICATION',2)]:
        census_path=OUT/'evidence'/(phase+f'_Applicability_v{revision}.json')
        if census_path.exists():
            census=read(census_path)
            for x in census['rows']:
                trials=x.get('anchor_trials',[])
                coverage.append(dict(phase=phase,session_id=x['session_id'],status=x['status'],reasons=';'.join(x['reasons']),predeclared_anchors=x.get('predeclared_anchor_count',0),assessed_anchors=len(trials),selected_scenario=x.get('spec',{}).get('scenario_token','')))
                for trial in trials:
                    for reason in trial['reasons']:rejections.append(dict(phase=phase,session_id=x['session_id'],anchor_rank=trial['anchor_rank'],reason=reason))
                if x.get('sources_opened'):
                    exposure.append(dict(phase=phase,event='OFFLINE_APPLICABILITY_SOURCE_ACCESS',session_id=x['session_id'],run_id='',added_exposure='E2_BENCHMARK_ENGINEERING',primary_comparison=False,E5=False))
        roster_path=OUT/(phase+'_Frozen_Roster_v'+('2' if phase=='QUALIFICATION' else '1')+'.json')
        if not roster_path.exists():continue
        for item in read(roster_path)['roster']:
            pid=item['pair_id'];pair=pairs.get(pid);arm_reviews=[]
            for arm in ['BASELINE','TREATMENT']:
                run=pid+'-'+arm;attempt=attempts.get(run);root=SCIENCE/phase/run
                review=read(root/'arm_science_review.json') if (root/'arm_science_review.json').exists() else None
                safe=read(root/'official_safety_adapter.json') if (root/'official_safety_adapter.json').exists() else None
                clock=read(root/'canonical_clock_mapping.json') if (root/'canonical_clock_mapping.json').exists() else None
                control=[json.loads(x) for x in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()] if (root/'telemetry/actual_lqr_controller_telemetry.jsonl').exists() else []
                trace=[json.loads(x)['current_ego'] for x in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()] if (root/'trace/realized_current_ego.jsonl').exists() else []
                arm_reviews.append(review)
                cr=[x for x in control if 2100000<=x['episode_time_us']<3900000]
                ar=dict(phase=phase,pair_id=pid,session_id=item['session_id'],run_id=run,arm=arm,attempt_status=attempt['status'] if attempt else 'NOT_RUN',
                    technical_status='PASS' if (root/'completion.json').exists() else 'INCOMPLETE' if attempt else 'NOT_RUN',
                    mechanism_pass=review['mechanism']['proposal_diagnostic_pass'] and not review['mechanism']['reasons'] if review else '',
                    scientific_arm_pass=review['arm_pass'] if review else '',mechanism_reasons=json.dumps(review['mechanism']['reasons']) if review else '',
                    qualified_phases=json.dumps(review['mechanism']['brake_phases']) if review else '',
                    coast_zero_count=sum(x['longitudinal_requested_acceleration']==0 for x in cr) if arm=='TREATMENT' and control else '',
                    states=len(trace) if attempt else '',transitions=len(control) if attempt else '',minimum_speed=min(x['speed_mps'] for x in trace) if trace else '',
                    at_fault_collisions=safe['canonical_payload']['collision']['number_of_all_at_fault_collisions_stat_value'] if safe else '',
                    drivable_compliance=safe['canonical_payload']['drivable_area']['drivable_area_compliance_stat_value'] if safe else '',
                    safety_pass=safe['frozen_arm_safety_pass'] if safe else '',clock_skew_min_us=clock['min_raw_minus_sim_us'] if clock else '',clock_skew_max_us=clock['max_raw_minus_sim_us'] if clock else '',
                    cpu_seconds=attempt.get('cpu_seconds','') if attempt else '',wall_seconds=attempt.get('wall_seconds','') if attempt else '',
                    artifact_manifest_sha256=sha(root/'execution_manifest.json') if (root/'execution_manifest.json').exists() else '')
                arms.append(ar)
                if attempt:exposure.append(dict(phase=phase,event='SCIENTIFIC_ATTEMPT',session_id=item['session_id'],run_id=run,added_exposure='E2_BENCHMARK_ENGINEERING',primary_comparison=False,E5=False))
            f=pair.get('F_match',{}) if pair else {};extra=pair.get('retained_P1_balance_review',{}) if pair else {};tm=pair.get('temporal',{}) if pair else {}
            reasons=[]
            if pair:
                if not all(pair['arm_scientific_pass']):reasons.append('ARM_MECHANISM_OR_SAFETY_FAILED')
                if f.get('pass') is False:reasons.append('F_MATCH_FAILED')
                if extra.get('pass') is False:reasons.append('EXTRA_BALANCE_FAILED')
                if not tm.get('pass_gate'):reasons.append('TEMPORAL_DIFFERENCE_FAILED')
            status='PASS' if pair and pair['pair_pass'] else 'SCIENTIFIC_FAIL' if pair else 'INFRASTRUCTURE_OR_NOT_RUN'
            rows.append(dict(phase=phase,pair_id=pid,session_id=item['session_id'],log_id=item['log_id'],scenario_token=item['scenario_token'],status=status,reasons=';'.join(reasons),
                F_match_status='PASS' if f.get('pass') else 'FAIL' if f.get('pass') is False else f.get('status','NOT_RUN'),
                F_absolute_delta=json.dumps(f.get('absolute_delta',{})),extra_balance=json.dumps(extra),temporal_max_speed_difference_mps=tm.get('max_abs_speed_difference_mps',''),temporal_L1_speed_difference_m=tm.get('absolute_speed_difference_integral_m',''),
                actual_traces_distinct=tm.get('actual_trace_hash_distinct',''),requests_distinct=tm.get('request_sequences_distinct',''),pair_review_sha256=sha(SCIENCE/phase/(pid+'-pair_review.json')) if pair else ''))
            for metric,limit in dict(mean_speed=.1,end_minus_start_speed=.15,path_length=.8,mean_abs_accel=.02).items():
                fm.append(dict(phase=phase,pair_id=pid,metric=metric,baseline=f.get('baseline',{}).get(metric,''),treatment=f.get('treatment',{}).get(metric,''),absolute_delta=f.get('absolute_delta',{}).get(metric,''),caliper=limit,status=('PASS' if f['absolute_delta'][metric]<=limit+1e-12 else 'FAIL') if 'absolute_delta' in f else f.get('status','NOT_RUN')))
    csv_new(OUT/'Pair_Results_v1.csv',rows);csv_new(OUT/'Arm_Mechanism_Safety_v1.csv',arms);csv_new(OUT/'FMatch_v1.csv',fm);csv_new(OUT/'Exposure_Ledger_v1.csv',exposure)
    csv_new(OUT/'Applicability_Coverage_v1.csv',coverage);csv_new(OUT/'Anchor_Exclusions_v1.csv',rejections,fields=['phase','session_id','anchor_rank','reason'])
    write(OUT/'Execution_Ledger_v1.json',ledger,exclusive=True)
    raw={str(p.relative_to(ROOT)):sha(p) for p in sorted(SCIENCE.rglob('*')) if p.is_file()};write(OUT/'evidence/Raw_Artifact_Index_v1.json',dict(count=len(raw),sha256=raw),exclusive=True)
    q=[x for x in rows if x['phase']=='QUALIFICATION'];qa=[x for x in arms if x['phase']=='QUALIFICATION'];dev=[x for x in rows if x['phase']=='DEVELOPMENT'];cov=[x for x in coverage if x['phase']=='QUALIFICATION']
    summary=dict(status='E2E_SDC_V1_QUALIFICATION_PASS_20_OF_20' if len(q)==20 and all(x['status']=='PASS' for x in q) else 'E2E_SDC_V1_QUALIFICATION_NOT_PASS' if q else 'E2E_SDC_V1_CAPACITY_BLOCKED',
        development_pairs=len(dev),development_PASS=sum(x['status']=='PASS' for x in dev),qualification_planned=20,qualification_rostered=len(q),qualification_complete=sum(x['status'] in ['PASS','SCIENTIFIC_FAIL'] for x in q),qualification_PASS=sum(x['status']=='PASS' for x in q),qualification_scientific_FAIL=sum(x['status']=='SCIENTIFIC_FAIL' for x in q),
        qualification_unrun_or_infrastructure=sum(x['status']=='INFRASTRUCTURE_OR_NOT_RUN' for x in q),attempts=len(ledger['attempts']),attempts_by_phase=dict(Counter(x['phase'] for x in ledger['attempts'])),total_CPU_seconds=ledger['cpu_seconds'],
        maximum_arm_wall_seconds=max(x['wall_seconds'] for x in ledger['attempts']),qual_coverage_counts=dict(Counter(x['status'] for x in cov)),qual_anchors_assessed=sum(x['assessed_anchors'] for x in cov),
        qualification_min_temporal_speed_delta=min((x['temporal_max_speed_difference_mps'] for x in q if x['temporal_max_speed_difference_mps']!=''),default=None),
        qualification_max_F_deltas={k:max([float(x['absolute_delta']) for x in fm if x['phase']=='QUALIFICATION' and x['metric']==k and x['absolute_delta']!=''],default=None) for k in ['mean_speed','end_minus_start_speed','path_length','mean_abs_accel']},
        qualification_safety_pass_arms=sum(x['safety_pass'] is True for x in qa),qualification_mechanism_pass_arms=sum(x['mechanism_pass'] is True for x in qa),qualification_coast_counts=[x['coast_zero_count'] for x in qa if x['arm']=='TREATMENT'],raw_artifacts=len(raw),
        scientific_failure_retries=0,qualification_technical_retries=0,controller_or_model_changes=0,RBR_H30_BDD_PRIMARY=0)
    write(OUT/'evidence/Actual_Summary_v1.json',summary,exclusive=True)
    # Representative chosen prospectively as first frozen qualification pair,
    # otherwise first development pair; never choose by detection or PASS.
    plot_phase='QUALIFICATION' if q else 'DEVELOPMENT';first=(q if q else dev)[0]['pair_id']
    plot(plot_phase,first)
    print(json.dumps(summary,indent=2))


def plot(phase,pair_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf');matplotlib.font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'svg.hashsalt':'SDC_V1','svg.fonttype':'path'})
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained');data_rows=[]
    for arm,color in [('BASELINE','#21618c'),('TREATMENT','#bd5b35')]:
        root=SCIENCE/phase/(pair_id+'-'+arm)
        if not (root/'trace/realized_current_ego.jsonl').exists():continue
        trace=[json.loads(x)['current_ego'] for x in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()];control=[json.loads(x) for x in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]
        t=[(x['time_us']-trace[0]['time_us'])/1e6 for x in trace];v=[x['speed_mps'] for x in trace];a=[control[0]['model_acceleration_before']]+[x['model_realized_acceleration'] for x in control];u=[x['longitudinal_requested_acceleration'] for x in control]
        axes[0].step(t[:79],u,where='post',color=color,label=arm+'最终纵向请求');axes[1].plot(t,a,color=color,label=arm+'官方模型实际加速度');axes[2].plot(t,v,color=color,label=arm+'真实速度')
        for i in range(80):data_rows.append(dict(phase=phase,pair_id=pair_id,arm=arm,index=i,episode_s=t[i],speed_mps=v[i],model_acceleration_mps2=a[i],outgoing_request_mps2=u[i] if i<79 else ''))
    for ax in axes:
        ax.axvspan(2.1,3.9,color='#dbece4',alpha=.65,label='Treatment释放窗口（请求0）');ax.grid(alpha=.2);ax.legend(fontsize=9);ax.set_xlim(0,7.9)
    axes[0].set_ylabel('请求（m/s²）');axes[1].set_ylabel('实际加速度（m/s²）');axes[2].set_ylabel('实际速度（m/s）');axes[2].set_xlabel('规范episode时间（秒）')
    fig.suptitle('E2E Scoped DirectCoast v1：'+pair_id+'\n固定顺序首对真实记录；中段零请求保留官方模型滤波尾巴',fontsize=13)
    for ext in ['png','svg']:fig.savefig(OUT/('Representative_Actual_Curves_v1.'+ext),dpi=160,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig);p=OUT/'Representative_Actual_Curves_v1.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    csv_new(OUT/'Representative_Actual_Curves_v1.csv',data_rows)


if __name__=='__main__':export()
