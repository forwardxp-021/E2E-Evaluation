#!/usr/bin/env python3
"""Approved SAP numerical primitives; no runner, model fitting or Primary CLI."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np


def matrix(x):
    a=np.asarray(x,np.float64)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Finite [N,d] representation required')
    return a


def unique_ids(ids,n):
    ids=np.asarray(ids,dtype=str)
    if len(ids)!=n or len(set(ids))!=n or any(not i for i in ids):raise ValueError('Duplicate/missing SESSION; scenario tokens cannot be independent units')
    return ids


def fit_baseline(x,ids,role):
    if role!='D_fit':raise PermissionError('Scaler and bandwidth may fit D_fit baseline only')
    x=matrix(x);unique_ids(ids,len(x))
    if len(x)<2:raise ValueError('At least two distinct D_fit SESSION required')
    mean=x.mean(0);scale=x.std(0);scale[scale<=1e-12]=1
    z=(x-mean)/scale;d=np.sum((z[:,None]-z[None,:])**2,axis=-1)
    positive=d[np.triu_indices(len(x),1)];positive=positive[positive>0]
    if not len(positive):raise ValueError('DEGENERATE_NO_POSITIVE_DISTANCE: no epsilon bandwidth rescue')
    return dict(mean=mean.tolist(),scale=scale.tolist(),sigma=float(np.sqrt(np.median(positive)/2)),
                fit_ids=list(map(str,ids)),fit_role=role,width=x.shape[1],constant_columns=np.flatnonzero(x.std(0)<=1e-12).tolist())


def transform(x,state):
    x=matrix(x)
    if x.shape[1]!=state['width']:raise ValueError('Frozen representation dimension mismatch')
    return (x-np.asarray(state['mean']))/np.asarray(state['scale'])


def mmd2(x,y,sigma):
    x,y=matrix(x),matrix(y)
    if x.shape[1]!=y.shape[1] or len(x)<2 or len(y)<2:raise ValueError('MMD requires matching widths and at least two rows per arm')
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive frozen bandwidth required')
    def k(a,b):return np.exp(-np.sum((a[:,None]-b[None,:])**2,axis=-1)/(2*sigma*sigma))
    xx,yy,xy=k(x,x),k(y,y),k(x,y)
    return float((xx.sum()-np.trace(xx))/(len(x)*(len(x)-1))+(yy.sum()-np.trace(yy))/(len(y)*(len(y)-1))-2*xy.mean())


def release_indices(ids,rng,m=20,pi=.5,weights=None):
    ids=unique_ids(ids,len(ids));w=np.ones(len(ids)) if weights is None else np.asarray(weights,np.float64)
    if w.shape!=(len(ids),) or not np.isfinite(w).all() or (w<0).any():raise ValueError('Invalid source-cluster bootstrap weights')
    if np.count_nonzero(w)>0:w=w/w.sum()
    if np.count_nonzero(w)<2*m:raise ValueError('INSUFFICIENT_UNIQUE_SESSION_SUPPORT_UNPAIRED: need 2m original IDs')
    chosen=rng.choice(len(ids),2*m,replace=False,p=w);ref,target=chosen[:m],chosen[m:]
    n=int(round(m*pi))
    if abs(n-m*pi)>1e-10:raise ValueError('Drift composition must be exact')
    drift=np.zeros(m,bool);drift[rng.choice(m,n,replace=False)]=True
    return ref,target,drift


def paired_swap(x,y,ids,swaps,sigma):
    """Controlled within-source exchange null; never an operational release FPR."""
    x,y=matrix(x),matrix(y);unique_ids(ids,len(x));s=np.asarray(swaps,bool)
    if x.shape!=y.shape or s.shape!=(len(x),):raise ValueError('Whole-pair shape mismatch')
    return mmd2(np.where(s[:,None],y,x),np.where(s[:,None],x,y),sigma)


def threshold(statistics,alpha=.05,expected_draws=1999):
    s=np.asarray(statistics,np.float64)
    if s.shape!=(expected_draws,) or not np.isfinite(s).all():raise ValueError('Complete finite calibration draw denominator required')
    rank=int(math.ceil((1-alpha)*(len(s)+1)))
    if rank>len(s):raise ValueError('Calibration rank exceeds finite support')
    return dict(value=float(np.sort(s)[rank-1]),rank_1based=rank,draws=len(s),alpha=alpha,alarm_comparison='strict_gt')


def alarms(statistics,t):return np.asarray(statistics)>t['value']


def pool(ids,baseline,treatment=None,role='D_diagnostic'):
    ids=unique_ids(ids,len(ids));b={k:matrix(v) for k,v in baseline.items()}
    if set(b)!={'RBR64','H30'} or any(len(v)!=len(ids) for v in b.values()):raise ValueError('Fair dual-representation source denominator required')
    t=b if treatment is None else {k:matrix(v) for k,v in treatment.items()}
    if set(t)!=set(b) or any(t[k].shape!=b[k].shape for k in b):raise ValueError('Complete paired arms and identical feature widths required')
    return dict(ids=ids,baseline=b,treatment=t,role=role)


def draw_statistics(p,fit,draws,seed,ab=False,weights=None):
    if not p['role'].startswith('D_'):raise PermissionError('D-only numerical interface rejects V/C roles')
    rng=np.random.RandomState(seed);out={k:[] for k in fit}
    b={k:transform(p['baseline'][k],s) for k,s in fit.items()};t={k:transform(p['treatment'][k],s) for k,s in fit.items()}
    for _ in range(draws):
        i,j,drift=release_indices(p['ids'],rng,weights=weights)
        for k,s in fit.items():
            target=np.where(drift[:,None],t[k][j],b[k][j]) if ab else b[k][j]
            out[k].append(mmd2(b[k][i],target,s['sigma']))
    return {k:np.array(v) for k,v in out.items()}


def assert_role_disjoint(pools,fit):
    seen=set().union(*(set(s['fit_ids']) for s in fit.values()))
    if set(fit['RBR64']['fit_ids'])!=set(fit['H30']['fit_ids']):raise ValueError('Unequal D_fit SESSION budgets')
    for p in pools:
        ids=set(p['ids'])
        if seen&ids:raise ValueError('SESSION role overlap between D_fit/calibration/evaluation')
        seen|=ids


def conditional_diagnostic(cal,aa,ab,fit,cal_draws=1999,eval_draws=10000,seeds=(2026091101,2026091102,2026091103),weights=None):
    """Finite-pool D-only estimate; MC SE is not scientific uncertainty."""
    assert_role_disjoint([cal,aa,ab],fit);weights=weights or [None]*3
    cs=draw_statistics(cal,fit,cal_draws,seeds[0],weights=weights[0]);cut={k:threshold(v,expected_draws=cal_draws) for k,v in cs.items()}
    stats=[draw_statistics(aa,fit,eval_draws,seeds[1],weights=weights[1]),draw_statistics(ab,fit,eval_draws,seeds[2],ab=True,weights=weights[2])]
    result={}
    for label,s in zip(['AA','AB'],stats):
        a={k:alarms(v,cut[k]) for k,v in s.items()};d=a['RBR64'].astype(float)-a['H30']
        result[label]=dict(alarm_probability={k:float(v.mean()) for k,v in a.items()},delta=float(d.mean()),conditional_MC_SE=float(d.std(ddof=1)/np.sqrt(len(d))))
    result.update(thresholds=cut,scientific_interval='NOT_ESTABLISHED_UNLESS_CLUSTER_COVERAGE_VALIDATED',scope='D_ONLY_FINITE_POOL_CONDITIONAL')
    return result


def cluster_interval(cal,aa,ab,fit,coverage_validated=False,outer=2000,inner=2000):
    """SAP source bootstrap: all arms travel together; unique IDs never duplicated in release."""
    if not coverage_validated:raise ValueError('SOURCE_DOMAIN_INTERVAL_NOT_ESTABLISHED: development coverage validation missing')
    assert_role_disjoint([cal,aa,ab],fit);rng=np.random.RandomState(2026091104);d=[];fpr={k:[] for k in fit};support=[]
    for _ in range(outer):
        weights=[rng.multinomial(len(p['ids']),np.full(len(p['ids']),1/len(p['ids']))) for p in [cal,aa,ab]]
        support.append([int(np.count_nonzero(w)) for w in weights])
        if min(support[-1])<40:raise ValueError('Bootstrap unique-ID support insufficient; do not condition away failed replicates')
        r=conditional_diagnostic(cal,aa,ab,fit,eval_draws=inner,weights=weights)
        d.append(r['AB']['delta'])
        for k in fit:fpr[k].append(r['AA']['alarm_probability'][k])
    return dict(delta_CI95=np.percentile(d,[2.5,97.5]).tolist(),FPR_simultaneous_upper={k:float(np.percentile(v,97.5)) for k,v in fpr.items()},
                outer=outer,inner=inner,minimum_unique_support=np.min(support,axis=0).tolist(),scope='D_ONLY_COVERAGE_VALIDATED_SOURCE_CLUSTER')


def sap_decision(delta,interval,fpr_intervals,qualification_pass,source_interval_established):
    """Pure approved decision rule; callers must provide certified cluster-aware intervals."""
    if not qualification_pass or not source_interval_established:
        return dict(primary='INCONCLUSIVE',reason='Qualification/source-domain interval not established')
    if set(fpr_intervals)!={'RBR64','H30'}:raise ValueError('Both simultaneous FPR intervals required')
    for bound in fpr_intervals.values():
        if len(bound)!=2 or not np.isfinite(bound).all() or not 0<=float(bound[0])<=float(bound[1])<=1:
            raise ValueError('Invalid simultaneous FPR interval')
    lo,hi=map(float,interval)
    if not np.isfinite([delta,lo,hi]).all() or lo>hi:raise ValueError('Invalid scientific interval')
    if any(float(v[0])>.075 for v in fpr_intervals.values()):
        return dict(primary='INCONCLUSIVE',FPR_gate='FAIL',reason='Calibration invalid; superiority prohibited')
    if any(float(v[1])>.075 for v in fpr_intervals.values()):
        return dict(primary='INCONCLUSIVE',FPR_gate='INCONCLUSIVE')
    if lo>0 and delta>=.10:return dict(primary='SUCCESS',FPR_gate='PASS')
    if hi<.10:return dict(primary='FAIL',FPR_gate='PASS')
    return dict(primary='INCONCLUSIVE',FPR_gate='PASS')
