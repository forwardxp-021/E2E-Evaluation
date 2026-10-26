#!/usr/bin/env python3
"""Read-only D4 statistics, synthetic experiments, and predeclared exploratory sensitivity."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, csv, json, time, resource
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tools.rbr_v2_d1_interface import sha, write
from tools.rbr_v2_d1_statistics import fit_baseline, transform
from tools.rbr_v2_d4_randomization import kernel, direct_stat, orbit, all_signs, pvalue
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rbr_v2_d4_v1';DOC=ROOT/'docs/stageR/rbr_v2_d4_v1'
D3=ROOT/'outputs/rbr_v2_d3_v1';PLAN=ROOT/'configs/rbr_v2_d4_v1/analysis_plan.json'


def read(p):
    return json.loads(Path(p).read_text())


def check():
    p=read(PLAN)
    for f,h in p['protected_sources'].items():
        if sha(ROOT/f)!=h:raise ValueError('Protected statistical evidence changed: '+f)
    return p


def quant(v):
    return np.quantile(np.asarray(v),[0,.05,.5,.95,1]).tolist()


def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def synthetic():
    p=check();rng=np.random.RandomState(p['synthetic_seed']);results=[];raw=[];start=time.monotonic()
    # Global threshold deliberately trained under the reference null only, before stress cases.
    def generate(case):
        latent=rng.normal(size=(20,8))*(5 if case in ['source_variance_null','known_effect_large_background'] else 1)
        x=latent+.2*rng.normal(size=(20,8));y=latent+.2*rng.normal(size=(20,8))
        if case=='symmetric_mask_null':
            x*=rng.binomial(1,.6,x.shape);y*=rng.binomial(1,.6,y.shape)
        if case=='extreme_scale_null':
            x[:,0]*=1e6;y[:,0]*=1e6
        if case=='identical_null':y=x.copy()
        if case.startswith('known_effect'):
            # Exactly10 affected sources, fixed before results; no effect tuning.
            y[:10,:2]+=p['synthetic_effect']
        if case=='label_dependent_mask_invalid_null':y[:,:4]=0.
        return x,y
    cs=[direct_stat(*generate('exchangeable_null'),2.) for _ in range(1999)]
    cut=float(np.sort(cs)[1899])
    for case in p['synthetic_cases']:
        pv=[];fixed=[];center=[]
        for j in range(p['synthetic_replicates']):
            x,y=generate(case);signs=rng.choice([-1.,1.],(p['synthetic_random_signs'],20))
            v,s,c=pvalue(x,y,2.,signs);pv.append(v);fixed.append(s>cut);center.append(c)
            raw.append(dict(case=case,replicate=j,p=v,original_statistic=s,orbit_mean=c,frozen_reference_threshold=cut))
        k=int(np.count_nonzero(np.array(pv)<=.05));n=len(pv);rate=k/n
        z=1.959963984540054;den=1+z*z/n;mid=(rate+z*z/(2*n))/den;half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
        results.append(dict(case=case,replicates=n,conditional_rejection_rate=rate,synthetic_binomial_Wilson95=[mid-half,mid+half],reference_global_threshold_alarm_rate=float(np.mean(fixed)),orbit_mean_quantiles=quant(center),exchangeability_assumed=case!='label_dependent_mask_invalid_null',population_claim=False))
        print('synthetic',case,rate,flush=True)
    write(OUT/'synthetic_results.json',dict(plan_sha256=sha(PLAN),rows=results,wall_seconds=time.monotonic()-start,scope='SYNTHETIC_SIZE_POWER_ONLY_NOT_RBR_H30_PERFORMANCE',reference_threshold=cut))
    table(OUT/'synthetic_replicates.csv',raw)


def real():
    p=check();started=time.monotonic();m=read(D3/'interface/input_manifest.json');rows=m['rows'];result=read(D3/'D_statistics_results.json')
    fits=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines']
    arrays={k:np.load(D3/'interface'/(k+'.npy'),mmap_mode='r') for k in fits};pairs={}
    for i,r in enumerate(rows):pairs.setdefault(r['session_id'],{})[r['arm']]=i
    meta=[];vectors=[];effects=[];summaries=[];sensitivity=[];channels=[];replay=[];artifacts={}
    def bind(path):
        path=Path(path);artifacts[str(path.relative_to(ROOT))]=sha(path)
    for g,ids in result['groups'].items():
        bi=[pairs[s]['BASELINE'] for s in ids];ti=[pairs[s]['TREATMENT'] for s in ids]
        local=[]
        for s,bidx,tidx in zip(ids,bi,ti):
            r=rows[bidx];ap=ROOT/r['input_path'];bind(ap);a=np.load(ap);bind(ROOT/rows[tidx]['input_path']);t=np.load(ROOT/rows[tidx]['input_path'])
            ep=ROOT/r['root']/'execution_manifest.json';bind(ep);spec=read(ep)['spec']
            dt=datetime.fromtimestamp(r['time_origin_us']/1e6,tz=timezone.utc)
            x=a['x'];v=a['channel_valid'];presence=[8,23,38,53,68]
            city=spec.get('map_name',spec.get('precontext',{}).get('map_identity',{}).get('map_name','UNKNOWN'))
            item=dict(group=g,session_id=s,map_name=city,UTC_hour=dt.hour,UTC_date=dt.date().isoformat(),initial_speed_mps=spec['precontext']['ego_initial_state']['initial_speed_mps'],mask_valid_fraction=float(v.mean()),neighbor_presence_fraction=float(x[:,presence].sum()/(80*5)),presence_known_fraction=float(v[:,presence].mean()),mean_speed_H30=float(arrays['H30'][bidx,0]),BT_mask_disagreement=float(np.mean(a['channel_valid']!=t['channel_valid'])))
            meta.append(item);local.append(item)
            vectors.append(dict(group=g,session_id=s,mask_by_channel=v.mean(0).tolist(),validity_BT_disagreement_by_channel=np.mean(a['channel_valid']!=t['channel_valid'],0).tolist()))
        for name,A in arrays.items():
            b=transform(A[bi],fits[name]);t=transform(A[ti],fits[name]);sigma=fits[name]['sigma'];xx=kernel(b,b,sigma);yy=kernel(t,t,sigma);xy=kernel(b,t,sigma);off=np.triu_indices(20,1)
            same=np.diag(xy);ind=xy[~np.eye(20,dtype=bool)]
            within=np.linalg.norm(b-t,axis=1);between=np.linalg.norm(b[:,None]-b[None,:],axis=-1)[off]
            same_shift=-2/20*(float(same.mean())-float(ind.mean()))
            # A paired-source U-form excludes all i=j terms. Diagnostic only; not selected as candidate.
            paired_U=((xx.sum()-np.trace(xx))+(yy.sum()-np.trace(yy))-(2*(xy.sum()-np.trace(xy))))/(20*19)
            summaries.append(dict(group=g,representation=name,BB_kernel_offdiag=quant(xx[off]),TT_kernel_offdiag=quant(yy[off]),BT_between_kernel=quant(ind),BT_same_kernel=quant(same),within_standardized_distance=quant(within),between_standardized_distance=quant(between),median_within_over_between=float(np.median(within)/np.median(between)),full_BT_original_MMD=float(direct_stat(b,t,sigma)),full_BT_pair_U_diagnostic=float(paired_U),full_BT_crossdiag_dependence_correction=same_shift,full_BT_original_minus_pair_U=float(direct_stat(b,t,sigma)-paired_U),same_kernel_underflow_fraction=float((same==0).mean())))
            for j,s in enumerate(ids):effects.append(dict(group=g,session_id=s,representation=name,raw_L2=float(np.linalg.norm(A[bi[j]]-A[ti[j]])),standardized_L2=float(within[j]),paired_kernel=float(same[j])))
            # Fixed128 subsets and orientation seeds shared by H/RBR. Exact conditional1024 flips.
            rng=np.random.RandomState(p['real_subset_seed']);sign10=all_signs(10);cache=[]
            for j in range(p['real_fixed_subsets']):
                changed=rng.choice(20,10,replace=False);right=b.copy();right[changed]=t[changed]
                signs=np.ones((1024,20));signs[:,changed]=sign10
                scores,c=orbit(b,right,sigma,signs);obs=direct_stat(b,right,sigma)
                permidx=int(rng.randint(1024));aaobs=scores[permidx]
                pAB=float(np.mean(scores>=obs-1e-12));pAA=float(np.mean(scores>=aaobs-1e-12))
                cache.append(dict(group=g,representation=name,release=j,AB_p_exact=pAB,random_orientation_AA_p_exact=pAA,observed_MMD=obs,orbit_mean=c,orbit_sd=float(scores.std()),orbit_shift_vs_original_threshold=c-result['thresholds'][name]['value']))
            replay.extend(cache)
        print('D4 real diagnostics',g,flush=True)
    # All predeclared scaler variants, no winner selected; original remains unchanged.
    fitm=read(ROOT/'outputs/rbr_v2_d1_v1/input_manifest.json')['rows'];fidx=[i for i,r in enumerate(fitm) if r['arm']=='BASELINE']
    allbi=[pairs[s]['BASELINE'] for ids in result['groups'].values() for s in ids];allti=[pairs[s]['TREATMENT'] for ids in result['groups'].values() for s in ids]
    for name,A in arrays.items():
        old=np.load(ROOT/'outputs/rbr_v2_d1_v1'/(name+'.npy'));variants={'frozen3':fits[name]}
        for omit in range(3):
            ix=[i for j,i in enumerate(fidx) if j!=omit];variants['leave_one_out_'+str(omit)]=fit_baseline(old[ix],[fitm[i]['session_id'] for i in ix],'D_fit')
        for g,ids in result['groups'].items():
            ix=[pairs[s]['BASELINE'] for s in ids];variants['EXPLORATORY_'+g+'20']=fit_baseline(A[ix],ids,'D_fit')
        variants['EXPLORATORY_all60']=fit_baseline(A[allbi],sum(result['groups'].values(),[]),'D_fit')
        for label,fit in variants.items():
            b=transform(A[allbi],fit);t=transform(A[allti],fit);K=kernel(b,t,fit['sigma']);sq=(b-t)**2/(2*fit['sigma']**2);contrib=sq.sum(0);total=float(contrib.sum());order=np.argsort(-contrib)
            sensitivity.append(dict(representation=name,variant=label,fit_n=len(fit['fit_ids']),sigma=fit['sigma'],minimum_nonconstant_scale=min([v for i,v in enumerate(fit['scale']) if i not in fit['constant_columns']]),constant_columns=fit['constant_columns'],same_BT_kernel_quantiles=quant(np.diag(K)),BT_underflow_fraction=float((np.diag(K)==0).mean()),top_exponent_columns=[int(i) for i in order[:5]],top_exponent_fraction=float(contrib[order[0]]/total) if total else 0.,all_BT_total_exponent=total))
            for i,(scale,con) in enumerate(zip(fit['scale'],contrib)):
                channels.append(dict(representation=name,variant=label,channel0=i,scale=scale,summed_BT_RBF_exponent=float(con),fraction=float(con/total) if total else 0.,constant=i in fit['constant_columns']))
    # No training arrays reread: production manifest provides complete retained mask/presence counts.
    prod=read(ROOT/'outputs/rbr_v2_prod_v1/data_manifest.json');norm=read(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json');bind(ROOT/'outputs/rbr_v2_prod_v1/data_manifest.json');bind(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json')
    train=dict(rows=prod['rows'],valid_channel_fraction=sum(s['valid_channel_cells'] for s in prod['shards'])/(prod['rows']*80*83),neighbor_presence_fraction=sum(s['neighbor_present_frames'] for s in prod['shards'])/(prod['rows']*80*5),per_channel_valid_fraction=(np.array(norm['input']['count'])/(prod['rows']*80)).tolist())
    write(OUT/'real_diagnostics.json',dict(scope='EXPOSED_D_EXPLORATORY_ONLY_NO_NEW_FIT_ADOPTED',kernel_and_information=summaries,scaler_sensitivity=sensitivity,metadata_summary={g:dict(maps=dict(Counter(r['map_name'] for r in meta if r['group']==g)),dates=dict(Counter(r['UTC_date'] for r in meta if r['group']==g)),UTC_hour=quant([r['UTC_hour'] for r in meta if r['group']==g]),initial_speed=quant([r['initial_speed_mps'] for r in meta if r['group']==g]),mask_fraction=quant([r['mask_valid_fraction'] for r in meta if r['group']==g]),neighbor_presence=quant([r['neighbor_presence_fraction'] for r in meta if r['group']==g])) for g in result['groups']},training_mask_summary=train,nuplan_channel_masks=vectors,replay_summary=[dict(group=g,representation=name,AB_alarm_fraction=float(np.mean([v['AB_p_exact']<=.05 for v in replay if v['group']==g and v['representation']==name])),random_orientation_AA_alarm_fraction=float(np.mean([v['random_orientation_AA_p_exact']<=.05 for v in replay if v['group']==g and v['representation']==name]))) for g in result['groups'] for name in fits],bound_read_only_artifacts=artifacts,wall_seconds=time.monotonic()-started))
    table(OUT/'session_covariates.csv',meta);table(OUT/'paired_representation_response.csv',effects);table(OUT/'scaler_channel_sensitivity.csv',channels);table(OUT/'exploratory_conditional_randomization.csv',replay)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['synthetic','real']);a=p.parse_args();before=resource.getrusage(resource.RUSAGE_SELF);started=time.monotonic()
    {'synthetic':synthetic,'real':real}[a.mode]();after=resource.getrusage(resource.RUSAGE_SELF)
    write(OUT/(a.mode+'_resources.json'),dict(wall_seconds=time.monotonic()-started,CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,vehicle_propagation=0,new_model_inference=0,training=0,downloads=0,Primary_V_C=0))
    check()

if __name__=='__main__':main()
