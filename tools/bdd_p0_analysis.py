#!/usr/bin/env python3
"""Frozen RBR encoding and prespecified raw-human-log D-only statistics, never Primary."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,time,math
from pathlib import Path
from collections import Counter
import numpy as np
from tools.bdd_p0_data import ROOT,OUT,PLAN,read,write,check
from tools.rbr_v2_d1_interface import sha
from tools.rbr_v2_d1_statistics import fit_baseline,transform


def encode():
    p=check();manifest=read(OUT/'dataset_manifest.json');rows=[r for r in manifest['rows'] if r['status']=='COMPLETE'];import torch
    from tools.rbr_v2_model import RBRV2,normalize
    cp=ROOT/'outputs/rbr_v2_prod_v1/run/epoch30.pt';norm=read(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json');state=torch.load(cp,map_location='cpu',weights_only=False)
    if state['committed_optimizer_steps']!=27390 or state['next_epoch']!=31:raise ValueError('Only frozen epoch30 permitted')
    model=RBRV2();model.load_state_dict(state['model']);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device);model.eval();zs=[];hs=[];started=time.monotonic()
    for offset in range(0,len(rows),16):
        batch=rows[offset:offset+16];arrays=[]
        for r in batch:
            path=ROOT/r['input_path']
            if sha(path)!=r['input_sha256']:raise ValueError('Certified raw input changed')
            arrays.append(np.load(path));hs.append(arrays[-1]['H30'])
        x=np.stack([normalize(a['x'],a['channel_valid'],norm['input']) for a in arrays]);valid=np.stack([a['channel_valid'] for a in arrays]);ev=np.stack([a['ego_valid'] for a in arrays])
        def forward():
            with torch.no_grad():return model(torch.as_tensor(x,device=device),torch.as_tensor(valid,device=device),torch.as_tensor(ev,device=device))[0].cpu().numpy()
        z=forward();np.testing.assert_array_equal(z,forward());zs.append(z);print('real human-log z64',offset+len(batch),'/',len(rows),flush=True)
    z=np.concatenate(zs);h=np.array(hs)
    if z.shape!=(len(rows),64) or h.shape!=(len(rows),30) or not np.isfinite(z).all() or not np.isfinite(h).all():raise ValueError('Representation finite/shape failure')
    np.save(OUT/'RBR64.npy',z);np.save(OUT/'H30.npy',h)
    fitidx=[i for i,r in enumerate(rows) if r['role']=='D_P0_FIT_EXPOSED'];fitids=[rows[i]['dependency_block'] for i in fitidx]
    fits={k:fit_baseline(a[fitidx],fitids,'D_fit') for k,a in [('RBR64',z),('H30',h)]}
    write(OUT/'detector_fit.json',dict(status='DEVELOPMENT_CANDIDATE_RAW_HUMAN_LOG_FIT_ONLY',pipelines=fits,fit_ids=[rows[i]['session_id'] for i in fitidx],fit_blocks=fitids,normalizer_sha256=sha(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json'),checkpoint_sha256=sha(cp),dataset_manifest_sha256=sha(OUT/'dataset_manifest.json'),no_heldout_fit=True))
    singular=np.linalg.svd(z.astype(float)-z.mean(0),compute_uv=False);mass=singular**2;prob=mass[mass>1e-20]/mass.sum()
    write(OUT/'representations.json',dict(rows=rows,z_shape=list(z.shape),H30_shape=list(h.shape),RBR_variance=z.var(0).tolist(),RBR_centered_rank=int(np.linalg.matrix_rank(z.astype(float)-z.mean(0))),RBR_entropy_rank=float(np.exp(-np.sum(prob*np.log(prob)))),repeated_eval_equal=True,finite=True,device=device,torch=torch.__version__,wall_seconds=time.monotonic()-started,new_optimizer_steps=0,Primary_V_C=0))
    check()


def kernel_matrix(x,fit):
    x=transform(x,fit);return np.exp(-np.sum((x[:,None]-x[None,:])**2,axis=-1)/(2*fit['sigma']**2))


def classify(rows,kind):
    if kind=='ACTIVITY':return ['LOW' if r['task']=='LOW' else 'MOVING' for r in rows]
    return [r['task'] for r in rows]


DESIGNS={'RAW_IID':None,'ACTIVITY_Q':({'LOW':10,'MOVING':10},{'LOW':10,'MOVING':10}),'FULL_TASK_Q':({'LOW':5,'JUNCTION':5,'LANE':10},{'LOW':5,'JUNCTION':5,'LANE':10}),'ACTIVITY_STRESS':({'LOW':10,'MOVING':10},{'LOW':5,'MOVING':15}),'FULL_TASK_STRESS':({'LOW':5,'JUNCTION':5,'LANE':10},{'LOW':5,'JUNCTION':10,'LANE':5})}


def bank(rows,design,n,seed,weights=None):
    if any(not r['role'].startswith('D_P0_') for r in rows):raise PermissionError('Only new P0 D data accepted; no Primary/V/C')
    if len({r['dependency_block'] for r in rows})!=len(rows):raise ValueError('Repeated dependency block')
    if len({r['session_id'] for r in rows})!=len(rows):raise ValueError('Repeated SESSION')
    weights=np.ones(len(rows)) if weights is None else np.asarray(weights,float)
    rng=np.random.RandomState(seed);left=[];right=[];spec=DESIGNS[design]
    if spec is None:
        groups=[(np.arange(len(rows)),20,20)]
    else:
        kinds=classify(rows,'ACTIVITY' if design.startswith('ACTIVITY') else 'FULL')
        groups=[(np.flatnonzero(np.array(kinds)==k),a,spec[1].get(k,0)) for k,a in spec[0].items()]
    for ids,a,b in groups:
        if np.count_nonzero(weights[ids])<a+b:raise ValueError('INSUFFICIENT_UNIQUE_SUPPORT '+design+' required'+str(a+b)+' available'+str(np.count_nonzero(weights[ids])))
    for _ in range(n):
        l=[];r=[]
        for ids,a,b in groups:
            w=weights[ids];w=w/w.sum();chosen=rng.choice(ids,a+b,replace=False,p=w);l.extend(chosen[:a]);r.extend(chosen[a:])
        left.append(l);right.append(r)
    return np.array(left,int),np.array(right,int)


def score(K,indices):
    left,right=indices;n=left.shape[1];out=[]
    for start in range(0,len(left),500):
        l=left[start:start+500];r=right[start:start+500];xx=K[l[:,:,None],l[:,None,:]];yy=K[r[:,:,None],r[:,None,:]];xy=K[l[:,:,None],r[:,None,:]]
        out.append((xx.sum((1,2))-np.trace(xx,axis1=1,axis2=2)+yy.sum((1,2))-np.trace(yy,axis1=1,axis2=2))/(n*(n-1))-2*xy.mean((1,2)))
    return np.concatenate(out)


def cutoff(values):
    rank=int(math.ceil(.95*(len(values)+1)))
    if rank>len(values):raise ValueError('Insufficient finite calibration ranks')
    return float(np.sort(values)[rank-1])


def statistics():
    p=check();binding=read(OUT/'pre_statistics_freeze.json')
    for f,h in binding['source_hashes'].items():
        if sha(ROOT/f)!=h:raise PermissionError('Post-freeze statistical code/fit changed '+f)
    m=read(OUT/'representations.json');rows=m['rows'];fit=read(OUT/'detector_fit.json')['pipelines'];arrays={k:np.load(OUT/(k+'.npy')) for k in fit};byrole={role:[i for i,r in enumerate(rows) if r['role']==role] for role in ['D_P0_CAL','D_P0_HELDOUT']};local={k:[rows[i] for i in ix] for k,ix in byrole.items()};kernels={role:{name:kernel_matrix(a[ix],fit[name]) for name,a in arrays.items()} for role,ix in byrole.items()};started=time.monotonic();results={};thresholds={};failed={};saved={}
    for design in ['RAW_IID','ACTIVITY_Q','FULL_TASK_Q']:
        try:
            cal=bank(local['D_P0_CAL'],design,1999,p['calibration_seed']);ev=bank(local['D_P0_HELDOUT'],design,10000,p['evaluation_seed'])
        except ValueError as e:failed[design]=str(e);continue
        thresholds[design]={};saved[design]={}
        for name in fit:
            cs=score(kernels['D_P0_CAL'][name],cal);s=score(kernels['D_P0_HELDOUT'][name],ev);cut=cutoff(cs);thresholds[design][name]=cut;a=s>cut;saved[design][name]=a
            np.save(OUT/(design+'_'+name+'_cal.npy'),cs);np.save(OUT/(design+'_'+name+'_eval.npy'),s)
        results[design]=dict(alarm_fraction={k:float(v.mean()) for k,v in saved[design].items()},delta_raw_alarm=float((saved[design]['RBR64'].astype(float)-saved[design]['H30']).mean()),interpretation='FINITE_POOL_REAL_LOG_RANDOM_SPLIT_AA_NOT_A_POLICY_VERSION_RESULT')
    for design,caldesign in [('ACTIVITY_STRESS','ACTIVITY_Q'),('FULL_TASK_STRESS','FULL_TASK_Q')]:
        if caldesign not in thresholds:failed[design]='No supported matched reference calibration';continue
        try:ev=bank(local['D_P0_HELDOUT'],design,10000,p['stress_seed'])
        except ValueError as e:failed[design]=str(e);continue
        alarms={}
        for name in fit:
            s=score(kernels['D_P0_HELDOUT'][name],ev);alarms[name]=s>thresholds[caldesign][name];np.save(OUT/(design+'_'+name+'_eval.npy'),s)
        results[design]=dict(alarm_fraction={k:float(v.mean()) for k,v in alarms.items()},delta_raw_alarm=float((alarms['RBR64'].astype(float)-alarms['H30']).mean()),interpretation='ARTIFICIAL_TASK_COMPOSITION_SHIFT_OF_REAL_LOGS_NO_WITHIN_TASK_INTERVENTION_NO_TRUE_POLICY_VERSION')
    bootstrap={};rng=np.random.RandomState(p['source_bootstrap']['seed'])
    for design in [k for k in ['RAW_IID','ACTIVITY_Q','FULL_TASK_Q'] if k in results]:
        rec=[]
        for rep in range(200):
            wc=rng.multinomial(len(local['D_P0_CAL']),np.full(len(local['D_P0_CAL']),1/len(local['D_P0_CAL'])));we=rng.multinomial(len(local['D_P0_HELDOUT']),np.full(len(local['D_P0_HELDOUT']),1/len(local['D_P0_HELDOUT'])))
            try:
                ci=bank(local['D_P0_CAL'],design,500,2026102800+rep,wc);ei=bank(local['D_P0_HELDOUT'],design,500,2026103800+rep,we)
                probs={name:float((score(kernels['D_P0_HELDOUT'][name],ei)>cutoff(score(kernels['D_P0_CAL'][name],ci))).mean()) for name in fit}
                rec.append(dict(rep=rep,status='COMPUTED',alarm_fraction=probs,unique_cal=int(np.count_nonzero(wc)),unique_eval=int(np.count_nonzero(we))))
            except ValueError as e:rec.append(dict(rep=rep,status='INSUFFICIENT_SOURCE_SUPPORT',reason=str(e),unique_cal=int(np.count_nonzero(wc)),unique_eval=int(np.count_nonzero(we))))
        bootstrap[design]=dict(replicates=rec,failed_replicates=sum(v['status']!='COMPUTED' for v in rec),source_domain_CI='NOT_ESTABLISHED: all replicates/support and independent coverage certification required; diagnostic200x500 is not original SAP inference')
        print('P0 source bootstrap',design,'failures',bootstrap[design]['failed_replicates'],flush=True)
    # Same fixed fit diagnostics for raw group kernel geometry, no alarm-driven refit.
    geometry={}
    for role,ks in kernels.items():
        geometry[role]={}
        for name,K in ks.items():
            values=K[np.triu_indices(len(K),1)];geometry[role][name]=dict(offdiag_quantiles=np.quantile(values,[0,.05,.5,.95,1]).tolist(),underflow_fraction=float((values==0).mean()),sigma=fit[name]['sigma'],constant_columns=fit[name]['constant_columns'])
    write(OUT/'statistics_results.json',dict(status='REAL_HUMAN_LOG_DEVELOPMENT_RESULTS_NOT_PRIMARY',results=results,thresholds=thresholds,unsupported_designs=failed,kernel_geometry=geometry,source_bootstrap=bootstrap,source_domain_power_coverage='NOT_ESTABLISHED',policy_version_comparison='NOT_ESTABLISHED_NO_POLICY_VERSION_DATA',m=20,alpha=.05,no_checkpoint_or_H_or_quota_selection=True,wall_seconds=time.monotonic()-started,Primary_V_C=0))
    check();print('P0 real A/A and task-composition experiments complete',results,failed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['encode','statistics']);a=p.parse_args();{'encode':encode,'statistics':statistics}[a.mode]()
