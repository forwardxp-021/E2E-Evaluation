#!/usr/bin/env python3
"""Fixed physical/scaler diagnostics, no detector refit replacement or new inference."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter
from tools.bdd_p0_data import ROOT,OUT,PLAN,read,write,check
from tools.rbr_v2_d1_statistics import transform
from tools.rbr_v2_d1_interface import sha
from tools.bdd_p0_analysis import kernel_matrix


def refit_diagnostic(x):
    mean=x.mean(0);s=x.std(0);constant=s<=1e-12;s[constant]=1.;z=(x-mean)/s;d=np.sum((z[:,None]-z[None,:])**2,axis=-1);positive=d[np.triu_indices(len(x),1)];positive=positive[positive>0]
    if not len(positive):raise ValueError('DEGENERATE_NO_POSITIVE_DISTANCE')
    return dict(mean=mean.tolist(),scale=s.tolist(),width=x.shape[1],sigma=float(np.sqrt(np.median(positive)/2)),constant_columns=np.flatnonzero(constant).tolist())


def main():
    check();m=read(OUT/'representations.json');rows=m['rows'];fits=read(OUT/'detector_fit.json')['pipelines'];ix=[i for i,r in enumerate(rows) if r['role']=='D_P0_FIT_EXPOSED'];arrays={k:np.load(OUT/(k+'.npy')) for k in fits};rng=np.random.RandomState(2026102708);draws=[rng.choice(len(ix),len(ix),replace=True) for _ in range(20)];diagnostics={}
    for name,a in arrays.items():
        fixed=kernel_matrix(a[ix],fits[name]);diag=[]
        for j,ind in enumerate(draws):
            f=refit_diagnostic(a[np.array(ix)[ind]])
            K=kernel_matrix(a[ix],f);diag.append(dict(rep=j,sigma=f['sigma'],minimum_nonconstant_scale=min(v for k,v in enumerate(f['scale']) if k not in f['constant_columns']),fixed57_kernel_max_abs_change=float(np.max(np.abs(K-fixed))),constant_columns=f['constant_columns']))
        diagnostics[name]=diag
    # Unit transformations are fixed physical/metamorphic diagnostics, never used to choose a fit.
    H=arrays['H30'];acc=[4,5,6,13]+list(range(14,22));unit=[];fixed=kernel_matrix(H[ix],fits['H30'])
    for name,factor in [('SI',1.),('g',1/9.80665),('kmh_per_s',3.6)]:
        x=H[ix].copy();x[:,acc]*=factor;f=refit_diagnostic(x);K=kernel_matrix(x,f)
        unit.append(dict(unit=name,constant_columns=f['constant_columns'],sigma=f['sigma'],kernel_max_abs_change=float(np.max(np.abs(K-fixed)))))
    per_role={}
    for role in sorted({r['role'] for r in rows}):
        rs=[r for r in rows if r['role']==role];inds=[i for i,r in enumerate(rows) if r['role']==role]
        per_role[role]=dict(SESSION=len(rs),blocks=len({r['dependency_block'] for r in rs}),maps=dict(Counter(r['map_name'] for r in rs)),tasks=dict(Counter(r['task'] for r in rs)),mask_fraction_quantiles=np.quantile([r['valid_channel_fraction'] for r in rs],[0,.05,.5,.95,1]).tolist(),neighbor_presence_quantiles=np.quantile([r['neighbor_presence_fraction'] for r in rs],[0,.05,.5,.95,1]).tolist(),speed_quantiles=np.quantile([r['initial_speed_mps'] for r in rs],[0,.05,.5,.95,1]).tolist())
    # Validate input normalization domain without neural updates or selecting any data.
    norm=read(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json')['input'];values=[]
    for r in rows:
        a=np.load(ROOT/r['input_path']);v=(a['x']-np.asarray(norm['mean']))/np.asarray(norm['std']);values.extend(np.abs(v[a['channel_valid']]).tolist())
    q=np.quantile(values,[.5,.95,.99,1]).tolist()
    velocity_oracle=dict(
        body_rotated_world_RMSE_median=float(np.median([r['velocity_position_RMSE_declared_world'] for r in rows])),
        wrong_unrotated_RMSE_median=float(np.median([r['velocity_position_RMSE_wrong_unrotated'] for r in rows])))
    write(OUT/'numerical_diagnostics.json',dict(development_source_bootstrap_fits=diagnostics,
        original_fit_not_replaced=True,all_unit_variants=unit,role_contexts=per_role,
        normalized_valid_absolute_quantiles=q,source_velocity_oracle=velocity_oracle,
        dataset_sha256=sha(OUT/'dataset_manifest.json'),no_alarm_or_quota_or_feature_selection=True))
    # Claim-relevant exposure: all accessed blocks remain development, including this P0 holdout.
    import csv
    with (OUT/'Exposure_Update.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['session_id','dependency_block','role','prior_exposure','new_exposure','future_confirmatory_use_for_this_method'],lineterminator='\n');w.writeheader()
        for r in rows:w.writerow(dict(session_id=r['session_id'],dependency_block=r['dependency_block'],role=r['role'],prior_exposure='E3_EXISTING' if r['role']=='D_P0_FIT_EXPOSED' else 'NO_PRIOR_DETECTOR_OUTCOME_READ',new_exposure='E3_PRIMARY_METHOD_DEVELOPMENT',future_confirmatory_use_for_this_method=False))
    check();print('P0 fixed numerical/source diagnostics complete; no alternative fit adopted')

if __name__=='__main__':main()
