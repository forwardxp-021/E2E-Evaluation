#!/usr/bin/env python3
"""Independent finite-population proofs and descriptive summaries; existing arrays only."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
import numpy as np
from scipy.stats import beta, binom, ks_2samp, spearmanr
from tools.rbr_v2_d4_analysis import ROOT,OUT,D3,read,write,check,table,quant
from tools.rbr_v2_d4_randomization import direct_stat,all_signs,orbit


def main():
    check();rng=np.random.RandomState(2026102703)
    x,y=rng.normal(size=(2,8,4));signs=all_signs(8);scores,c=orbit(x,y,2.,signs)
    from tools.rbr_v2_d4_randomization import kernel
    Q=kernel(x,x,2.)+kernel(y,y,2.)-kernel(x,y,2.)-kernel(y,x,2.);np.fill_diagonal(Q,0)
    a=15/(2*64*7);var=2*a*a*np.sum(Q*Q)
    assert abs(scores.mean()-c)<1e-14 and abs(scores.var()-var)<1e-14
    # Exhaust all64 datasets from a dependent discrete joint null: equal marginals, P(X=Y)=.8.
    states=[(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)];probs=[.4,.1,.1,.4];vals=[];weights=[]
    for indices in itertools.product(range(4),repeat=3):
        z=np.array([states[i] for i in indices]);weights.append(np.prod([probs[i] for i in indices]));vals.append(direct_stat(z[:,:1],z[:,1:],1.))
    mu=(1+np.exp(-2))/2;c_pair=.8+.2*np.exp(-2);pred=-2/3*(c_pair-mu);mean=float(np.dot(vals,weights))
    assert abs(mean-pred)<1e-14
    # Prospective precision costs, genuinely independent release blocks only, not repeated pools.
    planning=[]
    for n in [48,200,500,1000]:
        allowable=[k for k in range(n) if beta.ppf(.975,k+1,n-k)<=.075]
        cutoff=max(allowable) if allowable else -1
        planning.append(dict(independent_AA_release_blocks=n,AA_sources_at_m20=20*n,largest_FPR_count_passing_one_sided_975_upper=cutoff,probability_gate_pass_at_true_FPR005=float(binom.cdf(cutoff,n,.05)),zero_error_975_upper=float(1-.025**(1/n))))
    real=read(OUT/'real_diagnostics.json');meta=list(__import__('csv').DictReader((OUT/'session_covariates.csv').open()));pool=[]
    for g,h in [('cal','aa'),('cal','ab'),('aa','ab')]:
        v=[r for r in meta if r['group']==g];w=[r for r in meta if r['group']==h];maps=set(r['map_name'] for r in v+w)
        pool.append(dict(groups=[g,h],map_total_variation=.5*sum(abs(sum(r['map_name']==s for r in v)/20-sum(r['map_name']==s for r in w)/20) for s in maps),covariate_descriptive_KS={k:float(ks_2samp([float(r[k]) for r in v],[float(r[k]) for r in w]).statistic) for k in ['initial_speed_mps','mask_valid_fraction','neighbor_presence_fraction','UTC_hour']},no_causal_attribution=True))
    m=read(D3/'interface/input_manifest.json')['rows'];pairs={}
    for i,r in enumerate(m):pairs.setdefault(r['session_id'],{})[r['arm']]=i
    z=np.load(D3/'interface/RBR64.npy');h=np.load(D3/'interface/H30.npy');sids=[r['session_id'] for r in m if r['arm']=='BASELINE'];dH=np.array([h[pairs[s]['TREATMENT']]-h[pairs[s]['BASELINE']] for s in sids]);dz=np.array([z[pairs[s]['TREATMENT']]-z[pairs[s]['BASELINE']] for s in sids])
    names=['mean_speed','speed_SD','speed_q95','end_minus_start_speed','accel_RMS','mean_abs_accel_legacy','abs_accel_q95','jerk_RMS','abs_jerk_q95','yaw_rate_RMS','abs_yaw_rate_mean','total_abs_heading_change','path_length','mean_abs_accel_physical']+['signed_accel_bin_'+str(i) for i in range(8)]+['ACF_lag5','ACF_lag10','ACF_lag20','braking_centroid','braking_spread','ACF_valid','mass_valid','cadence_error']
    raw=[]
    for j,name in enumerate(names):raw.append(dict(channel0=j,name=name,BT_signed_delta_min=float(dH[:,j].min()),BT_signed_delta_median=float(np.median(dH[:,j])),BT_signed_delta_max=float(dH[:,j].max()),BT_abs_delta_median=float(np.median(np.abs(dH[:,j])))))
    table(OUT/'all_H30_physical_changes.csv',raw)
    eff=list(__import__('csv').DictReader((OUT/'paired_representation_response.csv').open()));dist={r['session_id']:float(r['standardized_L2']) for r in eff if r['representation']=='RBR64'};base={r['session_id']:r for r in meta}
    correlation={}
    for name,values in {'initial_speed':[float(base[s]['initial_speed_mps']) for s in sids],'mask_valid_fraction':[float(base[s]['mask_valid_fraction']) for s in sids],'mask_BT_change':[float(base[s]['BT_mask_disagreement']) for s in sids],'braking_spread_change':dH[:,26],'braking_centroid_change':dH[:,25]}.items():
        correlation[name]=float(spearmanr([dist[s] for s in sids],values).statistic)
    coherence=float(np.linalg.norm(dz.mean(0))/np.linalg.norm(dz,axis=1).mean())
    per_channel_sign_coherence=np.maximum((dz>0).mean(0),(dz<0).mean(0))
    write(OUT/'mathematical_checks.json',dict(dependent_equal_marginal_null=dict(m=3,exact64_dataset_mean=mean,predicted_mean=pred,paired_kernel_expectation=c_pair,independent_kernel_expectation=mu),conditional_orientation_moments=dict(exact256_mean=float(scores.mean()),formula_mean=c,exact256_variance=float(scores.var()),formula_variance=var),prospective_binomial_precision_planning=planning,pool_covariate_shifts=pool,descriptive_RBR_raw_direction_coherence=coherence,all64_channel_sign_coherence_quantiles=quant(per_channel_sign_coherence),RBR_distance_correlations_descriptive_only=correlation,no_feature_or_model_selection=True))
    check();print('Independent finite enumeration and orientation moment checks PASS')

if __name__=='__main__':main()
