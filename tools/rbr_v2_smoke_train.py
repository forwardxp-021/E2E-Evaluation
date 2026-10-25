#!/usr/bin/env python3
"""One engineering-only real training run; immutable 2 epochs / 32 steps.

The formal rbr_v2_train.entry_gate and all historical certificates stay unchanged.
"""
import time
PROCESS_STARTED=time.monotonic()
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import hashlib
import inspect
import json
import random
import resource
import subprocess
import uuid
from pathlib import Path
import numpy as np
import torch
from tools.rbr_v2_model import RBRV2,objectives
from tools.rbr_v2_canonical83 import ANGLES
from tools.rbr_v2_train import sha,write,fit_normalizers,make_batch,load_npz,normalize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rbr_v2_smoke_v1'
CFG=ROOT/'configs/rbr_v2_smoke_v1/protocol.json'
SOURCES=['tools/rbr_v2_smoke_train.py','tools/rbr_v2_smoke_data.py','tools/rbr_v2_train.py',
         'tools/rbr_v2_model.py','tools/rbr_v2_canonical83.py']


def sync(device):
    if device.type=='mps':torch.mps.synchronize()
    elif device.type=='cuda':torch.cuda.synchronize()


def memory(device):
    result=dict(cpu_process_seconds=time.process_time(),rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if device.type=='mps':
        result.update(mps_current_bytes=int(torch.mps.current_allocated_memory()),mps_driver_bytes=int(torch.mps.driver_allocated_memory()))
    elif device.type=='cuda':
        result.update(cuda_current_bytes=int(torch.cuda.memory_allocated()),cuda_peak_bytes=int(torch.cuda.max_memory_allocated()))
    return result


def seed_everything():
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
    if hasattr(torch,'mps') and torch.backends.mps.is_available():torch.mps.manual_seed(3407)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)


def grad_norm(grads):
    return float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)))


def groups(model):
    return {'shared_encoder':list(model.gru.parameters())+list(model.project.parameters()),
            'semantic_head':list(model.semantic.parameters()),'temporal_head':list(model.temporal.parameters())}


def parameter_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def device_probe():
    if (OUT/'runtime_probe.json').exists():raise FileExistsError('Probe already frozen')
    available=['cuda'] if torch.cuda.is_available() else []
    mps=hasattr(torch.backends,'mps') and torch.backends.mps.is_available()
    if mps:available.append('mps')
    available.append('cpu');records=[];selected=None
    for name in available:
        try:
            seed_everything();device=torch.device(name);model=RBRV2().to(device)
            # Synthetic compatibility only, zero optimizers and zero parameter updates.
            x=torch.zeros(128,80,83,device=device);mask=torch.ones_like(x,dtype=torch.bool)
            ev=torch.ones(128,80,dtype=torch.bool,device=device)
            z,ps,pr=model(x,mask,ev)
            lr,ls=objectives(ps,pr,torch.zeros(128,13,device=device),x,
                             torch.ones(128,13,dtype=torch.bool,device=device),mask)
            params=groups(model)['shared_encoder']
            gr=grad_norm(torch.autograd.grad(lr,params,retain_graph=True))
            gs=grad_norm(torch.autograd.grad(ls,params,retain_graph=True))
            (lr+ls).backward();sync(device)
            if not np.isfinite([gr,gs]).all() or min(gr,gs)<=0:raise ValueError('Invalid probe gradients')
            records.append(dict(device=name,status='PASS',repr_shared_grad=gr,sem_shared_grad=gs,
                                optimizer_steps=0,resources=memory(device)))
            selected=name;break
        except (RuntimeError,NotImplementedError,ValueError) as error:
            records.append(dict(device=name,status='UNSUPPORTED_BEFORE_RUN_CLAIM',error=str(error),optimizer_steps=0))
            print('Compatibility failure before real run:',name,str(error),flush=True)
    if selected is None:raise RuntimeError('No compatible training device')
    result=dict(selected_device=selected,probes=records,torch=torch.__version__,numpy=np.__version__,
        python=sys.version,executable=sys.executable,cuda_available=torch.cuda.is_available(),mps_available=mps,
        state='RBR_V2_ENGINEERING_SMOKE_ONLY',source_sha256={p:sha(ROOT/p) for p in SOURCES},
        preflight_optimizer_steps=0)
    write(OUT/'runtime_probe.json',result)
    (OUT/'runtime_requirements.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(json.dumps(result,indent=2))


def dataset_gate():
    cfg=json.loads(CFG.read_text());m=json.loads((OUT/'data_manifest.json').read_text())
    if (cfg['primary_seed'],cfg['batch_size'],cfg['epochs'],cfg['maximum_optimizer_steps'])!=(3407,128,2,32):
        raise ValueError('Frozen smoke hyperparameters differ')
    if m['state']!='RBR_V2_ENGINEERING_SMOKE_ONLY' or m['training_certified'] or m['primary_encoder_eligible']:
        raise ValueError('Engineering namespace/certification boundary violated')
    if m['raw_segment_disjointness']!='NOT_ESTABLISHED' or m['checkpoint_label']!='SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT':
        raise ValueError('Source risk or checkpoint label changed')
    if m['counts']!={'train':2048,'diagnostic':256}:raise ValueError('Exact sample count required')
    if sha(CFG)!=m['policy_sha256']:raise ValueError('Protocol changed after selection')
    tg=set();dg=set();seen=set()
    for r in m['rows']:
        if r['candidate_split']!='train' or r['legacy_split']=='test':raise ValueError('Protected test source')
        k=(r['scenario_id'],r['actor_id'],r['start'])
        if k in seen:raise ValueError('Duplicate selected identity')
        seen.add(k)
        (tg if r['split']=='train' else dg).add(r['dependency_group'])
        p=(OUT/r['path']).resolve()
        if OUT not in p.parents or sha(p)!=r['sha256']:raise ValueError('Changed/nonlocal materialized input')
        load_npz(OUT,r)
    if tg&dg:raise ValueError('Observed dependency overlap across smoke roles')
    for p,h in m['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Materialization source changed: '+p)
    return cfg,m


def preflight():
    cfg,m=dataset_gate()
    if (OUT/'execution_freeze.json').exists():raise FileExistsError('Execution already frozen')
    probe=json.loads((OUT/'runtime_probe.json').read_text())
    for p,h in probe['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Probed source changed: '+p)
    scaler=fit_normalizers(OUT,m['rows'])
    scaler.update(state='RBR_V2_ENGINEERING_SMOKE_ONLY',fit_rows=2048,diagnostic_fit_rows=0,
        data_manifest_sha256=sha(OUT/'data_manifest.json'),primary_eligible=False)
    write(OUT/'normalizer.json',scaler)
    # Data/loss support check on all planned batches; no model, optimizer, or fitting on diagnostics.
    training=[r for r in m['rows'] if r['split']=='train'];rng=np.random.RandomState(3407)
    for epoch in (1,2):
        order=rng.permutation(2048)
        for start in range(0,2048,128):
            rows=[training[int(i)] for i in order[start:start+128]]
            support=np.zeros(13,dtype=np.int64)
            for row in rows:support+=load_npz(OUT,row)['semantic_valid']
            if not np.all(support>0):raise ValueError('A frozen training batch lacks semantic support')
    freeze=dict(state=cfg['state'],checkpoint_label=cfg['checkpoint_label'],training_certified=False,
        config_sha256=sha(CFG),data_sha256=sha(OUT/'data_manifest.json'),normalizer_sha256=sha(OUT/'normalizer.json'),
        probe_sha256=sha(OUT/'runtime_probe.json'),device=probe['selected_device'],
        source_sha256={p:sha(ROOT/p) for p in SOURCES},planned_optimizer_steps=32,
        checkpoint_selection='LAST_COMPLETE_EPOCH_2_ONLY',diagnostic_selection=False,ready=True)
    write(OUT/'execution_freeze.json',freeze);print(json.dumps(freeze,indent=2))


def check_execution():
    f=json.loads((OUT/'execution_freeze.json').read_text())
    for p,k in [(CFG,'config_sha256'),(OUT/'data_manifest.json','data_sha256'),
                (OUT/'normalizer.json','normalizer_sha256'),(OUT/'runtime_probe.json','probe_sha256')]:
        if sha(p)!=f[k]:raise ValueError('Execution binding changed: '+str(p))
    for p,h in f['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen training source changed: '+p)
    return f


def to_device(batch,device):return tuple(x.to(device) for x in batch)


def diagnostic(model,rows,scaler,data_hash,device):
    model.eval();nr=0.;den=0;sn=np.zeros(13);sd=np.zeros(13)
    with torch.no_grad():
        for offset in range(0,len(rows),128):
            rs=rows[offset:offset+128]
            x,cv,ev,st,truth,sm,hm=to_device(make_batch(OUT,rs,scaler,2026102301,data_hash),device)
            z,ps,pr=model(x,cv,ev);delta=pr-truth
            delta[:,:,ANGLES]=torch.remainder(delta[:,:,ANGLES]+1,2)-1
            loss=torch.nn.functional.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none')
            nr+=float(((loss*hm).sum((1,2))/hm.sum((1,2))).sum());den+=len(rs)
            sl=torch.nn.functional.smooth_l1_loss(ps,st,reduction='none')
            sn+=(sl*sm).sum(0).cpu().numpy();sd+=sm.sum(0).cpu().numpy()
    if not np.all(sd>0):raise ValueError('Diagnostic semantic support absent')
    lr=nr/den;ls=float((sn/sd).mean())
    if not np.isfinite([lr,ls]).all():raise ValueError('Nonfinite diagnostic loss')
    return dict(L_repr=lr,L_sem=ls,total=lr+ls,used_for_selection=False)


def eval_z(model,rows,scaler,device):
    model.eval();result=[]
    with torch.no_grad():
        for offset in range(0,len(rows),128):
            rs=rows[offset:offset+128];a=[load_npz(OUT,r) for r in rs]
            x=np.stack([normalize(v['x'],v['channel_valid'],scaler['input']) for v in a])
            cv=np.stack([v['channel_valid'] for v in a]);ev=np.stack([v['ego_valid'] for v in a])
            z=model(torch.tensor(x,device=device),torch.tensor(cv,device=device),torch.tensor(ev,device=device))[0]
            result.append(z.cpu().numpy());print('z64 eval %d/%d'%(offset+len(rs),len(rows)),flush=True)
    z=np.concatenate(result,axis=0)
    if z.shape!=(256,64) or not np.isfinite(z).all():raise ValueError('Invalid actual z64 output')
    return z


def train_worker():
    f=check_execution();lease=json.loads((OUT/'run_lease.json').read_text())
    if os.environ.get('RBR_SMOKE_LEASE')!=lease['nonce']:raise ValueError('Real run must be launched once via execute')
    cfg,m=dataset_gate();scaler=json.loads((OUT/'normalizer.json').read_text())
    device=torch.device(f['device']);seed_everything();model=RBRV2().to(device)
    initial_hash=parameter_hash(model);pg=groups(model)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    training=[r for r in m['rows'] if r['split']=='train'];diag=[r for r in m['rows'] if r['split']=='diagnostic']
    rng=np.random.RandomState(3407);steps=[];epochs=[];normal_steps=0;run=OUT/'run'
    ledger=dict(state='RUNNING_ENGINEERING_SMOKE',optimizer_steps=0,complete_epochs=0,
        training_certified=False,primary_eligible=False,device=str(device),initial_parameter_sha256=initial_hash)
    write(run/'progress.json',ledger)
    try:
        for epoch in (1,2):
            ep_start=time.monotonic();model.train();order=rng.permutation(2048);totals=np.zeros(3)
            for offset in range(0,2048,128):
                if time.monotonic()-PROCESS_STARTED>=1800:raise TimeoutError('30 minute single-process budget')
                start=time.monotonic();cpu_start=time.process_time()
                rows=[training[int(i)] for i in order[offset:offset+128]]
                batch=make_batch(OUT,rows,scaler,3407+epoch,f['data_sha256'])
                data_seconds=time.monotonic()-start;batch=to_device(batch,device);sync(device)
                dev_start=time.monotonic();optimizer.zero_grad(set_to_none=True)
                x,cv,ev,st,truth,sm,hm=batch
                z,ps,pr=model(x,cv,ev);lr,ls=objectives(ps,pr,st,truth,sm,hm);loss=lr+ls
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                gr=grad_norm(torch.autograd.grad(lr,pg['shared_encoder'],retain_graph=True))
                gs=grad_norm(torch.autograd.grad(ls,pg['shared_encoder'],retain_graph=True))
                before={k:[p.detach().clone() for p in ps] for k,ps in pg.items()}
                loss.backward()
                head_grads={k:grad_norm([p.grad for p in ps]) for k,ps in pg.items()}
                clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                if not np.isfinite([gr,gs,clip,*head_grads.values()]).all():raise ValueError('Nonfinite actual gradient')
                optimizer.step();normal_steps+=1
                deltas={k:grad_norm([p.detach()-old for p,old in zip(ps,before[k])]) for k,ps in pg.items()}
                if not np.isfinite(list(deltas.values())).all():raise ValueError('Nonfinite parameter update')
                sync(device);elapsed=time.monotonic()-start
                record=dict(epoch=epoch,step=normal_steps,windows=128,L_repr=float(lr.detach()),L_sem=float(ls.detach()),
                    total=float(loss.detach()),repr_to_shared_grad_l2=gr,sem_to_shared_grad_l2=gs,
                    head_gradient_l2=head_grads,parameter_delta_l2=deltas,gradient_norm_before_clip=clip,
                    data_seconds=data_seconds,device_phase_seconds=time.monotonic()-dev_start,
                    step_seconds=elapsed,windows_per_second=128/elapsed,cpu_seconds=time.process_time()-cpu_start,
                    batch_identity_sha256=hashlib.sha256(json.dumps([r['identity_sha256'] for r in rows]).encode()).hexdigest(),
                    resources=memory(device),process_elapsed_seconds=time.monotonic()-PROCESS_STARTED)
                steps.append(record);totals+=np.array([record['L_repr'],record['L_sem'],record['total']])*128
                write(run/'steps.json',steps)
                ledger.update(optimizer_steps=normal_steps,process_elapsed_seconds=time.monotonic()-PROCESS_STARTED,
                              resources=memory(device));write(run/'progress.json',ledger)
                print('Smoke epoch%d step%d/32 Lrepr=%.6f Lsem=%.6f %.2f windows/s elapsed=%.1fs'%
                      (epoch,normal_steps,record['L_repr'],record['L_sem'],record['windows_per_second'],ledger['process_elapsed_seconds']),flush=True)
            train_seconds=time.monotonic()-ep_start
            ds=time.monotonic();d=diagnostic(model,diag,scaler,f['data_sha256'],device);sync(device)
            epoch_record=dict(epoch=epoch,L_repr=totals[0]/2048,L_sem=totals[1]/2048,total=totals[2]/2048,
                train_seconds=train_seconds,diagnostic=d,diagnostic_seconds=time.monotonic()-ds,
                total_epoch_seconds=time.monotonic()-ep_start,optimizer_steps=normal_steps)
            epochs.append(epoch_record);write(run/'epochs.json',epochs)
            state=dict(state='RBR_V2_ENGINEERING_SMOKE_ONLY',checkpoint_label='SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT',
                training_certified=False,primary_encoder_eligible=False,raw_segment_disjointness='NOT_ESTABLISHED',
                model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,optimizer_steps=normal_steps,
                seed=3407,data_sha256=f['data_sha256'],normalizer_sha256=f['normalizer_sha256'],
                config_sha256=f['config_sha256'],source_sha256=f['source_sha256'],execution_freeze_sha256=sha(OUT/'execution_freeze.json'),
                initial_parameter_sha256=initial_hash,parameter_sha256=parameter_hash(model),
                torch_rng=torch.get_rng_state(),numpy_rng=rng.get_state(),python_rng=random.getstate(),epochs=epochs)
            torch.save(state,run/('epoch%d_smoke.pt'%epoch))
            ledger['complete_epochs']=epoch;write(run/'progress.json',ledger)
        if normal_steps!=32:raise ValueError('Exactly 32 actual optimizer steps required')
        if any(not all(s[k]>0 for k in ['repr_to_shared_grad_l2','sem_to_shared_grad_l2']) for s in steps):
            raise ValueError('A head failed to affect the shared encoder')
        if any(any(s['parameter_delta_l2'][k]<=0 for k in pg) for s in steps):
            raise ValueError('Missing actual parameter update')
        final_diff={}
        for n,p in model.named_parameters():final_diff[n]=float(torch.linalg.vector_norm(p.detach().cpu()-initial[n]))
        write(run/'parameter_changes.json',dict(initial_sha256=initial_hash,final_sha256=parameter_hash(model),
            per_parameter_l2=final_diff,total_l2=float(np.sqrt(sum(v*v for v in final_diff.values())))))
        reference=eval_z(model,diag,scaler,device);np.save(run/'reference_diagnostic_z64.npy',reference)
        ledger.update(state='TRAINING_COMPLETED_AWAITING_RELOAD',optimizer_steps=32,
            checkpoint='run/epoch2_smoke.pt',checkpoint_sha256=sha(run/'epoch2_smoke.pt'),
            reference_z64_sha256=sha(run/'reference_diagnostic_z64.npy'),resources=memory(device))
    except BaseException as error:
        ledger.update(state='RBR_V2_SMOKE_TRAINING_BLOCKED',error=str(error));raise
    finally:
        ledger.update(process_elapsed_seconds=time.monotonic()-PROCESS_STARTED,stopped=True)
        write(run/'progress.json',ledger)


def execute():
    check_execution()
    # Global one-run lease; output renaming cannot obtain another run.
    nonce=uuid.uuid4().hex
    with (OUT/'run_lease.json').open('x') as f:
        json.dump(dict(state='CLAIMED_ONCE',nonce=nonce,wall_budget_seconds=1800),f,indent=2)
    (OUT/'run').mkdir();started=time.monotonic()
    env={**os.environ,'RBR_SMOKE_LEASE':nonce,'PYTORCH_ENABLE_MPS_FALLBACK':'0'}
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train-worker']
    with (OUT/'training.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            code=process.wait(timeout=1800);status='COMPLETED' if code==0 else 'BLOCKED_PROCESS_ERROR'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            code=process.returncode;status='BLOCKED_WALL_BUDGET_EXHAUSTED'
    result=dict(state=status,returncode=code,child_pid=process.pid,wall_seconds=time.monotonic()-started,
        budget_seconds=1800,real_training_runs=1,execution_freeze_sha256=sha(OUT/'execution_freeze.json'))
    write(OUT/'training_process.json',result);print(json.dumps(result,indent=2))
    if code!=0:raise RuntimeError('Single real run stopped; retain evidence, never launch a second run')


def load_checkpoint(path):
    kwargs=dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=False
    return torch.load(path,**kwargs)


def reload_export():
    f=check_execution();cfg,m=dataset_gate();run=OUT/'run'
    p=run/'epoch2_smoke.pt';progress=json.loads((run/'progress.json').read_text())
    if sha(p)!=progress['checkpoint_sha256']:raise ValueError('Checkpoint SHA mismatch')
    c=load_checkpoint(p)
    if c['checkpoint_label']!='SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT' or c['epoch']!=2 or c['optimizer_steps']!=32:
        raise ValueError('Invalid last-epoch smoke checkpoint')
    if c['data_sha256']!=f['data_sha256'] or c['normalizer_sha256']!=f['normalizer_sha256'] or c['source_sha256']!=f['source_sha256']:
        raise ValueError('Checkpoint/data/normalizer/source binding mismatch')
    device=torch.device(f['device']);seed_everything();model=RBRV2().to(device);model.load_state_dict(c['model'])
    scaler=json.loads((OUT/'normalizer.json').read_text());rows=[r for r in m['rows'] if r['split']=='diagnostic']
    z=eval_z(model,rows,scaler,device);repeat=eval_z(model,rows,scaler,device)
    reference=np.load(run/'reference_diagnostic_z64.npy',mmap_mode='r')
    np.testing.assert_allclose(z,reference,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(z,repeat,atol=1e-6,rtol=1e-5)
    dest=OUT/'export';dest.mkdir();np.save(dest/'diagnostic_z64.npy',z)
    with (dest/'z64_preview_8rows.csv').open('w',newline='') as fcsv:
        w=csv.writer(fcsv,lineterminator='\n');w.writerow(['scenario_id','actor_id','dependency_group']+['z%d'%i for i in range(64)])
        for r,a in zip(rows[:8],z[:8]):w.writerow([r['scenario_id'],r['actor_id'],r['dependency_group'],*a.tolist()])
    result=dict(status='PASS',state='RBR_V2_ENGINEERING_SMOKE_ONLY',checkpoint_label=c['checkpoint_label'],
        training_certified=False,primary_encoder_eligible=False,shape=list(z.shape),finite=bool(np.isfinite(z).all()),
        new_process_pid=os.getpid(),training_process_pid=json.loads((OUT/'training_process.json').read_text())['child_pid'],
        max_abs_reference_difference=float(np.abs(z-reference).max()),max_abs_repeat_difference=float(np.abs(z-repeat).max()),
        bitwise_reference_equal=bool(np.array_equal(z,reference)),bitwise_repeat_equal=bool(np.array_equal(z,repeat)),
        checkpoint_sha256=sha(p),normalizer_sha256=sha(OUT/'normalizer.json'),data_manifest_sha256=sha(OUT/'data_manifest.json'),
        z64_sha256=sha(dest/'diagnostic_z64.npy'),rows=rows,source_sha256=c['source_sha256'])
    write(dest/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','preflight','execute','train-worker','reload-export']);a=p.parse_args()
    if a.action=='probe':device_probe()
    elif a.action=='preflight':preflight()
    elif a.action=='execute':execute()
    elif a.action=='train-worker':train_worker()
    else:reload_export()
