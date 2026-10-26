#!/usr/bin/env python3
"""D5 claim/null counterexamples and unit-invariance diagnostics; no scientific runner."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv,json,time,resource
from pathlib import Path
import numpy as np
from tools.rbr_v2_d1_interface import sha,write
from tools.rbr_v2_d1_statistics import fit_baseline,transform
from tools.rbr_v2_d4_randomization import direct_stat,kernel
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_d5_v1';PLAN=ROOT/'configs/rbr_v2_d5_v1/analysis_plan.json'


def read(p):return json.loads(Path(p).read_text())


def check():
    plan=read(PLAN)
    for p,h in plan['protected'].items():
        if sha(ROOT/p)!=h:raise ValueError('Protected source changed '+p)
    return plan


def export_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def source_example(rng,case,standardized):
    if case=='coarse_task_balance_residual_confounding':
        # Both arms have identical coarse S distribution; full C still differs.
        c0=rng.binomial(1,.5,20)+rng.uniform(.05,.15,20)
        c1=rng.binomial(1,.5,20)+rng.uniform(.85,.95,20)
    else:
        p0,p1=(.1,.9) if case=='task_mix_only' and not standardized else (.5,.5)
        c0=rng.binomial(1,p0,20);c1=rng.binomial(1,p1,20)
    x=(3*c0+rng.normal(0,.4,20))[:,None];y=(3*c1+rng.normal(0,.4,20))[:,None]
    if case=='known_behavior_change_balanced_tasks':y+=1.
    return x,y


def main():
    p=check();wall=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF);rng=np.random.RandomState(p['synthetic_seed'])
    # Independent synthetic calibration, not a copied/reoriented B/T pool.
    cal=np.array([direct_stat(*source_example(rng,'matched_task_null',False),p['bandwidth']) for _ in range(p['calibration_datasets'])]);cut=float(np.sort(cal)[1899])
    rows=[];summary=[]
    for case in p['synthetic_cases']:
        item=dict(case=case,replicates=p['replicates'],conditional_behavior_equal=case!='known_behavior_change_balanced_tasks')
        for mode in ['raw','oracle_task_standardized']:
            stats=np.array([direct_stat(*source_example(rng,case,mode!='raw'),p['bandwidth']) for _ in range(p['replicates'])]);item[mode+'_alarm_fraction']=float((stats>cut).mean())
            rows.extend(dict(case=case,mode=mode,replicate=j,statistic=float(v),external_threshold=cut,alarm=bool(v>cut)) for j,v in enumerate(stats))
        summary.append(item);print('synthetic claim distinction',item,flush=True)
    export_csv(OUT/'synthetic_claim_counterexamples.csv',rows)
    # Invertible physical-unit transformations; no change to stored H30 or alarm rules.
    m=read(ROOT/'outputs/rbr_v2_d1_v1/input_manifest.json')['rows'];idx=[i for i,r in enumerate(m) if r['arm']=='BASELINE'];fitH=np.load(ROOT/'outputs/rbr_v2_d1_v1/H30.npy')
    d3=read(ROOT/'outputs/rbr_v2_d3_v1/interface/input_manifest.json')['rows'];H=np.load(ROOT/'outputs/rbr_v2_d3_v1/interface/H30.npy');pairs={}
    for i,r in enumerate(d3):pairs.setdefault(r['session_id'],{})[r['arm']]=i
    bi=[v['BASELINE'] for v in pairs.values()];ti=[v['TREATMENT'] for v in pairs.values()];unit=[];reference=None;reference_bb=None;acc=[4,5,6,13]+list(range(14,22))
    for label,factor in [('SI',1.),('g',1/9.80665),('kmh_per_second',3.6)]:
        data=H.copy();fitdata=fitH.copy();data[:,acc]*=factor;fitdata[:,acc]*=factor
        fit=fit_baseline(fitdata[idx],[m[i]['session_id'] for i in idx],'D_fit');b=transform(data[bi],fit);t=transform(data[ti],fit);K=kernel(b,t,fit['sigma']);diag=np.diag(K);bb=kernel(b,b,fit['sigma'])
        if reference is None:reference=K.copy();reference_bb=bb.copy()
        unit.append(dict(units=label,invertible_conversion_factor=factor,constant_columns=fit['constant_columns'],sigma=fit['sigma'],same_BT_kernel_min=float(diag.min()),same_BT_kernel_median=float(np.median(diag)),same_BT_kernel_max=float(diag.max()),same_BT_underflow_fraction=float((diag==0).mean()),max_BT_kernel_change_from_SI=float(np.max(np.abs(K-reference))),max_BB_kernel_change_from_SI=float(np.max(np.abs(bb-reference_bb))),alarm_not_recomputed=True))
    # Same marginal distribution, nonexchangeable paired joint law.
    joint=np.array([[0,1/3,0],[0,0,1/3],[1/3,0,0.]])
    np.testing.assert_array_equal(joint.sum(0),joint.sum(1));assert not np.array_equal(joint,joint.T)
    result=dict(status='D5_EXPLORATORY_CLAIM_AND_DESIGN_EVIDENCE_ONLY',plan_sha256=sha(PLAN),synthetic_external_threshold=cut,synthetic=summary,H30_unit_diagnostic=unit,
        equal_marginals_nonexchangeable_counterexample=dict(joint=joint.tolist(),joint_vs_swap_total_variation=float(np.abs(joint-joint.T).sum()/2)),
        saturation_bound=[dict(assumed_population_H_alarm=h,maximum_possible_RBR_increment=1-h) for h in [.8,.9,.95,.99,1.]],
        interpretation='Oracle source sampling is a synthetic design contrast, not an implemented conditional two-sample method or validation of real task exchangeability. All unit variants retained; none selected.',
        resources=dict(wall_seconds=time.monotonic()-wall,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime-cpu.ru_utime-cpu.ru_stime),new_vehicle_model_propagation=0,new_model_forward=0,training=0,downloads=0,Primary_V_C=0)
    if (OUT/'evidence.json').exists():
        journal=OUT/'deterministic_replay_journal.json';old=read(journal) if journal.exists() else []
        prior=read(OUT/'evidence.json');old.append(dict(previous_result_sha256=sha(OUT/'evidence.json'),previous_resources=prior['resources'],reason='Add baseline-baseline kernel unit-invariance diagnostic; original synthetic design and all fixed unit variants unchanged'))
        write(journal,old)
    write(OUT/'evidence.json',result);check()

if __name__=='__main__':main()
