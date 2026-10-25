#!/usr/bin/env python3
"""One formal U-only identity, fixed epoch30, persisted budget and exact resume state."""
import time
STARTED=time.monotonic()
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import json
import random
import resource
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from tools.rbr_v2_model import RBRV2,objectives,mask_plan,normalize
from tools.rbr_v2_prod_runtime import Corpus,fit_normalizer,sha,write,ROOT,OUT,CFG

SOURCES=['tools/rbr_v2_prod_train.py','tools/rbr_v2_prod_runtime.py','tools/rbr_v2_prod_features.py',
         'tools/rbr_v2_model.py','tools/rbr_v2_canonical83.py']


def seed():
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.backends.mps.is_available():torch.mps.manual_seed(3407)


def sync():torch.mps.synchronize()


def norm(gs):return float(torch.sqrt(sum(g.detach().float().square().sum() for g in gs)))


def params_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def resources():
    return dict(cpu_seconds=time.process_time(),rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mps_current_bytes=int(torch.mps.current_allocated_memory()),mps_driver_bytes=int(torch.mps.driver_allocated_memory()))


def move(batch):return tuple(x.to('mps') for x in batch)


def preflight():
    if (OUT/'execution_freeze.json').exists():raise FileExistsError('Execution already frozen')
    cfg=json.loads(CFG.read_text());data=Corpus(verify=True)
    if not data.manifest['owner_authorized_training'] or data.manifest['independent_validation_certified']:
        raise ValueError('Wrong limited-claim/validation route')
    if cfg['model_selection']!='FIXED_EPOCH_30_NO_U_VAL_CHECKPOINT_SELECTION':raise ValueError('Unapproved selection route')
    scaler=fit_normalizer(data);write(OUT/'normalizer.json',scaler)
    # Full actual training roster, including cache coordinates and original record identity.
    with (OUT/'actual_training_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data.metadata[0])+['shard_id'],lineterminator='\n');w.writeheader()
        for shard,s in zip(data.shards,data.maps):
            start=shard['global_start']
            for r in data.metadata[start:start+shard['rows']]:w.writerow({**r,'shard_id':shard['shard_id']})
    # Ensure ALL fixed future batches support every semantic term, without fitting a model.
    sem=np.concatenate([a['semantic_valid'] for a in data.maps],axis=0) # tiny N x13 validity only
    rng=np.random.RandomState(3407);steps=(data.n+127)//128
    for epoch in range(1,31):
        order=rng.permutation(data.n)
        for offset in range(0,data.n,128):
            if not np.all(sem[order[offset:offset+128]].sum(0)>0):raise ValueError('Unsupported target in fixed training batch')
    if not torch.backends.mps.is_available():raise RuntimeError('Frozen MPS device unavailable')
    seed();a=RBRV2().to('mps');b=RBRV2().to('mps');b.load_state_dict(a.state_dict())
    raw=data.gather(np.arange(128));cpu_batch=data.batch(np.arange(128),scaler,3408)
    for i,row in enumerate(data.metadata[:128]):
        identity=json.dumps([data.hash,row['scenario_id'],row['actor_id'],row['start']])
        np.testing.assert_array_equal(cpu_batch[-1][i].numpy(),mask_plan(raw['channel_valid'][i],identity,3408))
        np.testing.assert_array_equal(cpu_batch[4][i].numpy(),normalize(raw['x'][i],raw['channel_valid'][i],scaler['input']))
    batch=move(cpu_batch);x,cv,ev,st,truth,sm,hm=batch
    za,sa,ra=a(x,cv,ev);lr,ls=objectives(sa,ra,st,truth,sm,hm)
    shared=list(a.gru.parameters())+list(a.project.parameters())
    gr=norm(torch.autograd.grad(lr,shared,retain_graph=True));gs=norm(torch.autograd.grad(ls,shared,retain_graph=True))
    (lr+ls).backward();zb,sb,rb=b(x,cv,ev);rr,ss=objectives(sb,rb,st,truth,sm,hm);(rr+ss).backward()
    maxdiff=0.;bitwise=True
    for p,q in zip(a.parameters(),b.parameters()):
        pp=p.grad.detach().cpu().numpy();qq=q.grad.detach().cpu().numpy()
        np.testing.assert_allclose(pp,qq,atol=1e-6,rtol=1e-5);maxdiff=max(maxdiff,float(np.abs(pp-qq).max()));bitwise&=bool(np.array_equal(pp,qq))
    ca=float(torch.nn.utils.clip_grad_norm_(a.parameters(),1.));cb=float(torch.nn.utils.clip_grad_norm_(b.parameters(),1.))
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
    if not np.isfinite([gr,gs,ca,cb]).all() or min(gr,gs)<=0:raise ValueError('Shared-head gradient preflight failed')
    proof=dict(status='PASS',optimized_mask_and_normalization_reference_rows=128,
        audit_vs_single_backward_grad_max_abs=maxdiff,bitwise_gradient_equal=bitwise,
        repr_shared_grad_l2=gr,sem_shared_grad_l2=gs,optimizer_steps=0,clip_norms=[ca,cb])
    write(OUT/'gradient_equivalence.json',proof)
    source={p:sha(ROOT/p) for p in SOURCES}
    freeze=dict(state='OWNER_AUTHORIZED_FORMAL_U_ONLY_WITH_DISCLOSED_SOURCE_RISK',source_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',
        selection='FIXED_EPOCH_30_NO_U_VAL_CHECKPOINT_SELECTION',epochs=30,seed=3407,batch_size=128,
        rows=data.n,steps_per_epoch=steps,expected_committed_steps=30*steps,device='mps',device_budget_seconds=86400,
        cfg_sha256=sha(CFG),data_manifest_sha256=sha(OUT/'data_manifest.json'),normalizer_sha256=sha(OUT/'normalizer.json'),
        actual_training_manifest_sha256=sha(OUT/'actual_training_manifest.csv'),source_sha256=source,
        gradient_proof_sha256=sha(OUT/'gradient_equivalence.json'),torch=torch.__version__,numpy=np.__version__,
        budget_reference_hours=30*steps*.6174367525936759/3600,smoke_checkpoint_loaded=False)
    if freeze['budget_reference_hours']>24:raise RuntimeError('Reference resource forecast exceeds authorized training budget')
    write(OUT/'execution_freeze.json',freeze);print(json.dumps(freeze,indent=2))


def freeze_check():
    f=json.loads((OUT/'execution_freeze.json').read_text())
    for p,key in [(CFG,'cfg_sha256'),(OUT/'data_manifest.json','data_manifest_sha256'),
        (OUT/'normalizer.json','normalizer_sha256'),(OUT/'actual_training_manifest.csv','actual_training_manifest_sha256'),
        (OUT/'gradient_equivalence.json','gradient_proof_sha256')]:
        if sha(p)!=f[key]:raise ValueError('Frozen artifact changed: '+str(p))
    for p,h in f['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen optimization source changed: '+p)
    return f


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def checkpoint(path,model,opt,epoch,offset,order,rng,partial,history,total_steps,f,initial_hash):
    for p in model.parameters():
        if not torch.isfinite(p).all():raise ValueError('NUMERICAL_NONFINITE_PARAMETER')
    state=dict(schema='RBR_V2_PROD_CHECKPOINT_V1',run_id='RBR_V2_PROD_SEED3407_V1',seed=3407,
        source_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',selection=f['selection'],
        model=model.state_dict(),optimizer=opt.state_dict(),next_epoch=epoch,next_offset=offset,
        order=order,numpy_rng=rng.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state(),partial=partial,history=history,committed_optimizer_steps=total_steps,
        bindings=f,initial_parameter_sha256=initial_hash,parameter_sha256=params_hash(model),
        code_commit=json.loads((OUT/'run/identity.json').read_text())['frozen_code_commit'])
    temp=path.with_name(path.name+'.writing');torch.save(state,temp);os.replace(temp,path)


def diagnostics(model,data,scaler):
    model.eval();rn=0.;count=0;sn=np.zeros(13);sc=np.zeros(13)
    with torch.no_grad():
        for offset in range(0,256,128):
            batch=move(data.batch(data.diagnostic[offset:offset+128],scaler,2026102301));x,cv,ev,st,truth,sm,hm=batch
            z,s,r=model(x,cv,ev);lr,ls=objectives(s,r,st,truth,sm,hm)
            rn+=float(lr)*len(x);count+=len(x)
            sl=torch.nn.functional.smooth_l1_loss(s,st,reduction='none')
            sn+=(sl*sm).sum(0).cpu().numpy();sc+=sm.sum(0).cpu().numpy()
    if not np.all(sc>0):raise ValueError('Diagnostic target support absent')
    a=rn/count;b=float((sn/sc).mean())
    if not np.isfinite([a,b]).all():raise ValueError('NUMERICAL_NONFINITE_DIAGNOSTIC')
    return dict(L_repr=a,L_sem=b,total=a+b,role='IN_SAMPLE_NUMERICAL_ONLY',used_for_selection=False)


def encode256(model,data,scaler):
    model.eval();outputs=[]
    with torch.no_grad():
        for offset in range(0,256,128):
            x,cv,ev,*_=move(data.batch(data.diagnostic[offset:offset+128],scaler,0,masked=False))
            a=model(x,cv,ev)[0].cpu().numpy();b=model(x,cv,ev)[0].cpu().numpy()
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5);outputs.append(a)
    z=np.concatenate(outputs,axis=0)
    if z.shape!=(256,64) or not np.isfinite(z).all():raise ValueError('Invalid real z64 export')
    return z


def worker():
    f=freeze_check();active=json.loads((OUT/'run/active_attempt.json').read_text())
    if os.environ.get('RBR_PROD_NONCE')!=active['nonce']:raise ValueError('Missing same-run supervisor lease')
    data=Corpus(verify=True);scaler=json.loads((OUT/'normalizer.json').read_text());seed()
    model=RBRV2().to('mps');opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    rng=np.random.RandomState(3407);epoch=1;offset=0;order=None;partial=dict(sums=[0.,0.,0.],rows=0,seconds=0.)
    history=[];total_steps=0;initial_hash=params_hash(model);run=OUT/'run';latest=run/'latest.pt'
    if latest.exists():
        state=load(latest)
        if state['bindings']!=f:raise ValueError('Resume source/data/config mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for st0 in opt.state.values():
            for key,value in st0.items():
                if isinstance(value,torch.Tensor) and key!='step':st0[key]=value.to('mps')
        rng.set_state(state['numpy_rng']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.mps.set_rng_state(state['mps_rng'])
        epoch=state['next_epoch'];offset=state['next_offset'];order=state['order'];partial=state['partial'];history=state['history']
        total_steps=state['committed_optimizer_steps'];initial_hash=state['initial_parameter_sha256']
        print('RESUME same identity checkpoint step',total_steps,'epoch',epoch,'offset',offset,flush=True)
    elif active['attempt']!=1:
        raise RuntimeError('No recoverable state after interruption; do not reinitialize identity')
    else:
        if (run/'random_initial_state.pt').exists():raise FileExistsError('Never overwrite initial random state')
        torch.save(dict(label='INITIAL_RANDOM_NOT_TRAINED',seed=3407,optimizer_steps=0,
                        model=model.state_dict(),parameter_sha256=initial_hash),run/'random_initial_state.pt')
        checkpoint(latest,model,opt,1,0,None,rng,partial,history,0,f,initial_hash)
    attempt_start_steps=total_steps;status=dict(state='RUNNING',complete_epochs=len(history),optimizer_steps=total_steps,
        initial_parameter_sha256=initial_hash,device='mps',attempt=active['attempt'],initialization='FRESH_RANDOM' if active['attempt']==1 else 'RESTORED_FULL_STATE')
    write(run/'progress.json',status)
    optimization_started=time.monotonic()
    try:
        with (run/('steps_attempt%03d.jsonl'%active['attempt'])).open('a') as logs,ThreadPoolExecutor(max_workers=1) as prefetch:
            while epoch<=30:
                data.unchanged();model.train()
                if order is None:order=rng.permutation(data.n).astype(np.int32)
                epoch_start=time.monotonic();start_offset=offset
                offsets=list(range(offset,data.n,128))
                future=prefetch.submit(data.batch,order[offsets[0]:offsets[0]+128],scaler,3407+epoch) if offsets else None
                audit_values=None
                for oi,pos in enumerate(offsets):
                    if time.monotonic()-STARTED>=active['remaining_seconds']:raise TimeoutError('AUTHORIZED_DEVICE_BUDGET_EXHAUSTED')
                    t=time.monotonic();batch=move(future.result())
                    if oi+1<len(offsets):
                        nx=offsets[oi+1];future=prefetch.submit(data.batch,order[nx:nx+128],scaler,3407+epoch)
                    x,cv,ev,st,truth,sm,hm=batch;opt.zero_grad(set_to_none=True)
                    z,s,r=model(x,cv,ev);lr,ls=objectives(s,r,st,truth,sm,hm);loss=lr+ls
                    if not torch.isfinite(loss):raise ValueError('NUMERICAL_NONFINITE_LOSS')
                    if audit_values is None:
                        shared=list(model.gru.parameters())+list(model.project.parameters())
                        gr=norm(torch.autograd.grad(lr,shared,retain_graph=True));gs=norm(torch.autograd.grad(ls,shared,retain_graph=True))
                        if not np.isfinite([gr,gs]).all() or min(gr,gs)<=0:raise ValueError('NUMERICAL_INVALID_SHARED_GRADIENT')
                        audit_values=dict(repr_to_shared_l2=gr,sem_to_shared_l2=gs)
                    loss.backward();clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                    if not np.isfinite(clip):raise ValueError('NUMERICAL_NONFINITE_GRADIENT')
                    if oi==0:
                        audit_values.update(semantic_head_grad_l2=norm([p.grad for p in model.semantic.parameters()]),
                                            temporal_head_grad_l2=norm([p.grad for p in model.temporal.parameters()]))
                    opt.step();total_steps+=1;offset=pos+len(x)
                    vals=[float(lr.detach()),float(ls.detach()),float(loss.detach())]
                    partial['sums']=(np.array(partial['sums'])+np.array(vals)*len(x)).tolist();partial['rows']+=len(x)
                    record=dict(epoch=epoch,step=total_steps,rows=len(x),L_repr=vals[0],L_sem=vals[1],total=vals[2],
                        grad_norm=clip,batch_seconds=time.monotonic()-t,attempt=active['attempt'],process_seconds=time.monotonic()-STARTED)
                    logs.write(json.dumps(record)+'\n');logs.flush()
                    if total_steps%16==0 or offset==data.n:
                        status.update(optimizer_steps=total_steps,epoch=epoch,next_offset=offset,process_seconds=time.monotonic()-STARTED,resources=resources())
                        write(run/'progress.json',status)
                        if status['resources']['rss_peak_bytes']>8*1024**3:raise OSError('RSS guard exceeded; preserve state and stop')
                    if total_steps%64==0 or offset==data.n:
                        print('PROD epoch%d/30 rows%d/%d step%d L=%.6f elapsed=%.1fs'%(epoch,offset,data.n,total_steps,vals[2],time.monotonic()-STARTED),flush=True)
                    if (pos//128+1)%128==0:
                        partial['seconds']+=time.monotonic()-epoch_start;epoch_start=time.monotonic()
                        checkpoint(latest,model,opt,epoch,offset,order,rng,partial,history,total_steps,f,initial_hash)
                    if (total_steps-attempt_start_steps)>=256 and total_steps%256==0:
                        seconds_per_step=(time.monotonic()-optimization_started)/(total_steps-attempt_start_steps)
                        forecast=(f['expected_committed_steps']-total_steps)*seconds_per_step
                        available=active['remaining_seconds']-(time.monotonic()-STARTED)
                        status['remaining_resource_projection_seconds']=forecast;write(run/'progress.json',status)
                        if forecast>available:raise TimeoutError('PROJECTED_REMAINDER_EXCEEDS_DEVICE_BUDGET')
                train_seconds=partial['seconds']+time.monotonic()-epoch_start
                if partial['rows']!=data.n:raise ValueError('Complete epoch denominator mismatch')
                dt=time.monotonic();diagnostic=diagnostics(model,data,scaler);sync()
                values=(np.array(partial['sums'])/data.n).tolist()
                row=dict(epoch=epoch,L_repr=values[0],L_sem=values[1],total=values[2],rows=data.n,
                    train_seconds=train_seconds,diagnostic_seconds=time.monotonic()-dt,diagnostic=diagnostic,
                    shared_gradient_audit=audit_values,optimizer_steps=total_steps,resources=resources())
                history.append(row);write(run/'epochs.json',history)
                print('EPOCH COMPLETE',json.dumps(row),flush=True)
                epoch+=1;offset=0;order=None;partial=dict(sums=[0.,0.,0.],rows=0,seconds=0.)
                path=run/('epoch%02d.pt'%(epoch-1))
                checkpoint(path,model,opt,epoch,offset,order,rng,partial,history,total_steps,f,initial_hash)
                # Exact saved epoch state also becomes restart anchor; previous epoch files retained.
                import shutil
                tmp=latest.with_name('latest.pt.writing');shutil.copyfile(path,tmp);os.replace(tmp,latest)
                status.update(complete_epochs=len(history),optimizer_steps=total_steps,checkpoint_sha256=sha(path))
                write(run/'progress.json',status)
        if len(history)!=30 or total_steps!=f['expected_committed_steps']:raise ValueError('Fixed epoch30 trajectory incomplete')
        np.save(run/'final_reference_z64.npy',encode256(model,data,scaler))
        status.update(state='TRAINING_COMPLETED_FIXED_EPOCH30',parameter_sha256=params_hash(model),final_checkpoint_sha256=sha(run/'epoch30.pt'))
    except BaseException as error:
        numerical=isinstance(error,ValueError)
        budget=isinstance(error,TimeoutError)
        status.update(state='NUMERICAL_FAILURE' if numerical else ('BUDGET_STOP' if budget else 'INFRASTRUCTURE_INTERRUPTED'),error=str(error));raise
    finally:
        status.update(optimizer_steps=total_steps,executed_steps_this_attempt=total_steps-attempt_start_steps,
            process_seconds=time.monotonic()-STARTED,resources=resources(),stopped=True)
        write(run/'progress.json',status)


def execute():
    f=freeze_check();run=OUT/'run';run.mkdir(exist_ok=True)
    identity=run/'identity.json'
    if not identity.exists():
        paths=list(f['source_sha256'])+[str(CFG.relative_to(ROOT)),'outputs/rbr_v2_prod_v1/execution_freeze.json',
            'outputs/rbr_v2_prod_v1/data_manifest.json','outputs/rbr_v2_prod_v1/normalizer.json','outputs/rbr_v2_prod_v1/actual_training_manifest.csv']
        dirty=subprocess.check_output(['git','status','--porcelain','--',*paths],text=True)
        if dirty.strip():raise ValueError('Mandatory freeze artifacts must be COMMITTED before first optimizer step')
        with identity.open('x') as h:json.dump(dict(run_id='RBR_V2_PROD_SEED3407_V1',seed=3407,
            frozen_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            execution_freeze_sha256=sha(OUT/'execution_freeze.json')),h,indent=2)
    prior=json.loads((run/'attempts.json').read_text()) if (run/'attempts.json').exists() else []
    if (run/'progress.json').exists():
        state=json.loads((run/'progress.json').read_text())['state']
        if state in ['NUMERICAL_FAILURE','BUDGET_STOP','TRAINING_COMPLETED_FIXED_EPOCH30']:raise RuntimeError('Stopped identity cannot restart: '+state)
    for a in prior:
        if 'wall_seconds' not in a:
            pid=a.get('pid')
            if pid:
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise RuntimeError('Prior training process still alive; never start a duplicate identity')
            import calendar
            start_epoch=calendar.timegm(time.strptime(a['start_utc'],'%Y-%m-%dT%H:%M:%SZ'))
            a.update(wall_seconds=max(0.,time.time()-start_epoch),
                accounting='CONSERVATIVE_WALL_UPPER_BOUND_AFTER_SUPERVISOR_LOSS',returncode='UNKNOWN_INFRASTRUCTURE_EXIT')
    if prior:write(run/'attempts.json',prior)
    used=sum(a['wall_seconds'] for a in prior);remaining=86400-used
    if remaining<=0:raise TimeoutError('Cumulative authorized device budget exhausted')
    active=dict(attempt=len(prior)+1,nonce=uuid.uuid4().hex,remaining_seconds=remaining,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(run/'active_attempt.json',active);prior.append(dict(active));write(run/'attempts.json',prior)
    started=time.monotonic();env={**os.environ,'RBR_PROD_NONCE':active['nonce'],'PYTORCH_ENABLE_MPS_FALLBACK':'0'}
    with (run/('attempt%03d.log'%active['attempt'])).open('w') as log:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        prior[-1]['pid']=process.pid;write(run/'attempts.json',prior)
        try:
            code=process.wait(timeout=remaining)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            code=process.returncode
        finally:
            prior[-1].update(wall_seconds=time.monotonic()-started,returncode=process.returncode,pid=process.pid)
            write(run/'attempts.json',prior)
    print(json.dumps(prior[-1],indent=2))
    if code:raise RuntimeError('Attempt retained; no new identity. Infrastructure-only resume requires complete latest state.')


def export():
    f=freeze_check();data=Corpus(verify=True);scaler=json.loads((OUT/'normalizer.json').read_text())
    c=load(OUT/'run/epoch30.pt')
    if c['next_epoch']!=31 or c['committed_optimizer_steps']!=f['expected_committed_steps'] or c['bindings']!=f:
        raise ValueError('Unique fixed epoch30 checkpoint not satisfied')
    seed();model=RBRV2().to('mps');model.load_state_dict(c['model'])
    z=encode256(model,data,scaler);reference=np.load(OUT/'run/final_reference_z64.npy')
    np.testing.assert_allclose(z,reference,atol=1e-6,rtol=1e-5)
    if params_hash(model)!=c['parameter_sha256']:raise ValueError('Loaded trained parameter identity differs')
    initial=load(OUT/'run/random_initial_state.pt')
    if initial['parameter_sha256']!=c['initial_parameter_sha256']:raise ValueError('Random initialization binding differs')
    deltas={}
    for prefix in ['gru.','project.','semantic.','temporal.']:
        total=0.
        for name,p in model.state_dict().items():
            if name.startswith(prefix):total+=float((p.detach().cpu()-initial['model'][name].cpu()).square().sum())
        deltas[prefix]=float(np.sqrt(total))
    if min(deltas.values())<=0:raise ValueError('A required module never changed from random initialization')
    opt_steps=sorted({int(s['step'].item()) for s in c['optimizer']['state'].values()})
    if opt_steps!=[f['expected_committed_steps']]:raise ValueError('Optimizer step state incomplete')
    dest=OUT/'export';dest.mkdir(exist_ok=True)
    path=dest/'z64.npy'
    if path.exists():np.testing.assert_allclose(np.load(path),z,atol=1e-6,rtol=1e-5)
    else:np.save(path,z)
    result=dict(status='PASS',pid=os.getpid(),shape=list(z.shape),finite=True,checkpoint_sha256=sha(OUT/'run/epoch30.pt'),
        reference_max_abs_difference=float(np.abs(z-reference).max()),reference_bitwise_equal=bool(np.array_equal(z,reference)),
        module_parameter_delta_l2=deltas,optimizer_parameter_step_values=opt_steps,
        normalizer_sha256=sha(OUT/'normalizer.json'),data_manifest_sha256=data.hash,z64_sha256=sha(path),
        parameter_sha256=params_hash(model),initial_parameter_sha256=c['initial_parameter_sha256'],
        selection=f['selection'],source_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',
        rows=[data.metadata[int(i)] for i in data.diagnostic],source_sha256=f['source_sha256'])
    previous=dest/'reload_verification.json'
    if previous.exists():
        old=json.loads(previous.read_text());result['independent_previous_pid']=old['pid'];result['independent_process_repeat_pass']=old['pid']!=os.getpid()
    write(previous,result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_sha256']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','execute','worker','export']);a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='execute':execute()
    elif a.action=='worker':worker()
    else:export()
