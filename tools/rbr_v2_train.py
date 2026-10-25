#!/usr/bin/env python3
"""Single seed U-only trainer. Provenance failure occurs before model construction.

No nuPlan/outcome reader, pairwise handcrafted distance objective, seed search,
or U_test selection. Dataset manifest is an immutable hash-bound certificate.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from tools.rbr_v2_model import RBRV2, Normalizer, normalize, mask_plan, objectives


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def entry_gate(manifest_path, config):
    manifest_path=Path(manifest_path).resolve()
    m=json.loads(manifest_path.read_text()); reasons=[]
    fixed=json.loads((Path(__file__).resolve().parents[1]/'configs/rbr_v2_uonly_v1/seed3407.json').read_text())
    if config != fixed:
        reasons.append('Configuration differs from the sole frozen candidate')
    if config['primary_seed'] != 3407 or config['max_epochs'] > 30 or config['device_budget_seconds'] > 86400:
        reasons.append('Unauthorized seed/epoch/device budget')
    if not m.get('training_certified',False):
        reasons.append('U dataset is not training-certified')
    if m.get('raw_segment_disjointness') != 'ESTABLISHED':
        reasons.append('RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED')
    if m.get('role') != 'U_ONLY' or m.get('input_contract') != 'canonical83_v1':
        reasons.append('U-only role/input identity missing')
    if not m.get('source_mapping'):
        reasons.append('Missing upstream segment/object/global-time mapping')
    if not m.get('rows'):
        reasons.append('No full canonical corpus certified')
    if m.get('frozen_config_sha256') != sha(Path(__file__).resolve().parents[1]/'configs/rbr_v2_uonly_v1/seed3407.json'):
        reasons.append('Frozen config binding missing or mismatched')
    if reasons:
        raise ValueError('TRAINING_ENTRY_BLOCKED: '+'; '.join(reasons))
    base=manifest_path.parent
    mapping_path=base/m['source_mapping']['path']
    if sha(mapping_path) != m['source_mapping']['sha256']:
        raise ValueError('Source mapping hash mismatch')
    mapping=list(csv.DictReader(mapping_path.open()))
    upstream={r['scenario_id']:r for r in mapping}
    group_splits={}; segment_splits={}; seen=set(); split_count={'train':0,'val':0,'test':0}
    for r in m['rows']:
        if r['split'] not in split_count or r.get('role') != 'U':
            raise ValueError('Dataset includes non-U identity or unknown split')
        identity=(r['scenario_id'],str(r['actor_id']),r['start'])
        if identity in seen:
            raise ValueError('Duplicate row identity')
        seen.add(identity); split_count[r['split']]+=1
        src=upstream[r['scenario_id']]
        required=['raw_segment_id','global_object_id_namespace','absolute_time_origin_us','upstream_evidence_sha256']
        if any(not src.get(k) or src[k] in ('UNKNOWN','NOT_AVAILABLE') for k in required):
            raise ValueError('Uncertifiable raw source identity')
        for key,d in [(r['dependency_group'],group_splits),(src['raw_segment_id'],segment_splits)]:
            if key in d and d[key] != r['split']:
                raise ValueError('Dependency/source segment crosses splits')
            d[key]=r['split']
        p=(base/r['path']).resolve()
        if base not in p.parents or sha(p) != r['sha256']:
            raise ValueError('Nonlocal or changed canonical data')
    if not all(split_count.values()):
        raise ValueError('All three independent U splits are required')
    # Complete source/config/data hashes must have been frozen, not generated after training.
    if m.get('preflight_status') != 'PASS' or not m.get('frozen_source_sha256'):
        raise ValueError('Pre-training zero-step/source freeze absent')
    for p,h in m['frozen_source_sha256'].items():
        if sha(Path(__file__).resolve().parents[1]/p) != h:
            raise ValueError('Frozen training source changed: '+p)
    return m


def load_npz(base,row):
    p=(base/row['path']).resolve()
    with np.load(p,allow_pickle=False) as f:
        out={k:f[k].copy() for k in ['x','channel_valid','ego_valid','semantic','semantic_valid']}
    if out['x'].shape != (80,83) or not np.isfinite(out['x']).all():
        raise ValueError('Canonical input shape/nonfinite error')
    if out['ego_valid'].sum() < 64 or out['channel_valid'][:,6].sum() < 2:
        raise ValueError('Unaccounted input-invalid row')
    return out


def fit_normalizers(base,rows):
    nx=Normalizer(83); ns=Normalizer(13)
    for r in rows:
        if r['split'] != 'train':
            continue
        a=load_npz(base,r)
        nx.update(a['x'],a['channel_valid'],'train')
        ns.update(a['semantic'],a['semantic_valid'],'train')
    x,s=nx.state(True),ns.state()
    if s['zero_support']:
        raise ValueError('Semantic target has no U_train support')
    return {'input':x,'semantic':s}


def make_batch(base,rows,scaler,seed,dataset_hash):
    data=[load_npz(base,r) for r in rows]
    x=np.stack([normalize(a['x'],a['channel_valid'],scaler['input']) for a in data])
    cv=np.stack([a['channel_valid'] for a in data]); ev=np.stack([a['ego_valid'] for a in data])
    sm=np.stack([a['semantic_valid'] for a in data])
    st=np.stack([normalize(a['semantic'],a['semantic_valid'],scaler['semantic']) for a in data])
    hm=np.stack([mask_plan(a['channel_valid'],json.dumps([dataset_hash,r['scenario_id'],r['actor_id'],r['start']]),seed)
                 for a,r in zip(data,rows)])
    target=torch.from_numpy(x); hidden=torch.from_numpy(hm); visible=torch.from_numpy(cv & ~hm)
    corrupted=target.clone(); corrupted[~visible]=0
    return corrupted,visible,torch.from_numpy(ev),torch.from_numpy(st),target,torch.from_numpy(sm),hidden


def losses(model,batch):
    x,cv,ev,st,truth,sm,hm=batch
    z,ps,pr=model(x,cv,ev)
    return objectives(ps,pr,st,truth,sm,hm)


def selection_improved(value,best):
    return value < best-1e-6


def train(manifest_path,config,out):
    m=entry_gate(manifest_path,config)  # fail BEFORE seed/model/optimizer/device timer.
    sole_run=Path(__file__).resolve().parents[1]/'outputs/rbr_v2_uonly_v1/seed3407'
    if out.resolve() != sole_run:
        raise ValueError('One immutable seed3407 run namespace; no budget reset via alternate output')
    if out.exists():
        raise FileExistsError('Never reset/reuse a seed3407 run ledger')
    out.mkdir(parents=True); started=time.monotonic(); base=Path(manifest_path).resolve().parent
    used=0.; seed=3407; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config['cpu_threads']); torch.use_deterministic_algorithms(True)
    manifest_hash=sha(manifest_path)
    write(out/'frozen_config.json',config); write(out/'frozen_data_manifest.json',m)
    scaler=fit_normalizers(base,m['rows']); write(out/'normalizer.json',scaler)
    model=RBRV2(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    tr=[r for r in m['rows'] if r['split']=='train']; va=[r for r in m['rows'] if r['split']=='val']
    # U_test numerical data are never loaded by the optimization/selection loop.
    best=float('inf'); stale=0; rng=np.random.RandomState(seed); history=[]
    ledger={'seed':seed,'device':'cpu','device_seconds':0,'epochs_completed':0,'stopped':False,
            'data_sha256':manifest_hash,'config_sha256':sha(out/'frozen_config.json'),'attempt':1}
    write(out/'budget.json',ledger)
    try:
        for epoch in range(1,config['max_epochs']+1):
            if time.monotonic()-started >= config['device_budget_seconds']:
                raise TimeoutError('24h cumulative device budget exhausted')
            model.train(); totals=np.zeros(3); n=0
            order=rng.permutation(len(tr))
            for offset in range(0,len(tr),128):
                if time.monotonic()-started >= config['device_budget_seconds']:
                    raise TimeoutError('Cumulative device budget exhausted')
                selected=[tr[int(i)] for i in order[offset:offset+128]]
                batch=make_batch(base,selected,scaler,seed+epoch,manifest_hash)
                optimizer.zero_grad(); lr,ls=losses(model,batch); loss=lr+ls
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient')
                optimizer.step(); totals+=np.array([loss.item(),lr.item(),ls.item()])*len(selected); n+=len(selected)
                ledger['device_seconds']=time.monotonic()-started; write(out/'budget.json',ledger)
                print('seed3407 epoch%d train%d/%d %.1fs'%(epoch,n,len(tr),ledger['device_seconds']),flush=True)
            # Aggregate semantic loss exactly by target across all val rows, not batch means.
            model.eval(); vr_num=0.; val_n=0; sem_num=np.zeros(13); sem_den=np.zeros(13)
            with torch.no_grad():
                for offset in range(0,len(va),128):
                    selected=va[offset:offset+128]; b=make_batch(base,selected,scaler,2026102301,manifest_hash)
                    x,cv,ev,st,truth,sm,hm=b; z,ps,pr=model(x,cv,ev)
                    # Reconstruction function also checks semantic support; calculate masked numerator here.
                    delta=pr-truth
                    from tools.rbr_v2_canonical83 import ANGLES
                    delta[:,:,ANGLES]=torch.remainder(delta[:,:,ANGLES]+1,2)-1
                    rl=torch.nn.functional.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none')
                    vr_num+=float(((rl*hm).sum((1,2))/hm.sum((1,2))).sum()); val_n+=len(selected)
                    sl=torch.nn.functional.smooth_l1_loss(ps,st,reduction='none')
                    sem_num+=(sl*sm).sum(0).numpy(); sem_den+=sm.sum(0).numpy()
            if not np.all(sem_den>0):
                raise ValueError('U_val lacks mandatory semantic support')
            vlr=vr_num/val_n; vls=float(np.mean(sem_num/sem_den)); value=vlr+vls
            if not np.isfinite(value):
                raise ValueError('Nonfinite validation loss')
            improved=selection_improved(value,best)
            if improved:
                best=value; stale=0
            else:
                stale+=1
            record=dict(epoch=epoch,train_total=totals[0]/n,train_repr=totals[1]/n,train_sem=totals[2]/n,
                        val_total=value,val_repr=vlr,val_sem=vls,best=improved,device_seconds=time.monotonic()-started)
            history.append(record); write(out/'history.json',history)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                       torch_rng=torch.get_rng_state(),numpy_rng=rng.get_state(),python_rng=random.getstate(),
                       data_sha256=manifest_hash,config_sha256=ledger['config_sha256'],
                       normalizer_sha256=sha(out/'normalizer.json'),history=history)
            torch.save(state,out/'last.pt')
            if improved:
                torch.save(state,out/'best.pt')
            ledger.update(epochs_completed=epoch,device_seconds=time.monotonic()-started)
            write(out/'budget.json',ledger)
            if stale>=5:
                break
        ledger['result']='COMPLETED'; ledger['checkpoint_sha256']=sha(out/'best.pt')
    except BaseException as e:
        ledger['result']='STOPPED_INCOMPLETE'; ledger['error']=str(e)
        raise
    finally:
        ledger['stopped']=True; ledger['device_seconds']=time.monotonic()-started
        write(out/'budget.json',ledger)


def export(manifest_path,config,run,out,split):
    m=entry_gate(manifest_path,config)
    if sha(manifest_path) != json.loads((run/'budget.json').read_text())['data_sha256']:
        raise ValueError('Checkpoint/data identity mismatch')
    state=torch.load(run/'best.pt',map_location='cpu')
    if state['normalizer_sha256'] != sha(run/'normalizer.json'):
        raise ValueError('Normalizer changed')
    model=RBRV2(); model.load_state_dict(state['model']); model.eval()
    scaler=json.loads((run/'normalizer.json').read_text()); rows=[r for r in m['rows'] if r['split']==split]
    if out.exists():
        raise FileExistsError('Immutable z64 export already exists')
    out.mkdir(parents=True); base=Path(manifest_path).resolve().parent
    z=np.lib.format.open_memmap(out/'z64.npy',mode='w+',dtype='float32',shape=(len(rows),64))
    with torch.no_grad():
        for i,r in enumerate(rows):
            a=load_npz(base,r); x=normalize(a['x'],a['channel_valid'],scaler['input'])
            z[i]=model(torch.tensor(x[None]),torch.tensor(a['channel_valid'][None]),torch.tensor(a['ego_valid'][None]))[0].numpy()[0]
            if not np.isfinite(z[i]).all():
                raise ValueError('Nonfinite z64; no silent exclusion')
            if i%128==0: print('z64 export %d/%d'%(i,len(rows)),flush=True)
    z.flush()
    write(out/'manifest.json',dict(shape=[len(rows),64],rows=rows,data_sha256=sha(manifest_path),
        checkpoint_sha256=sha(run/'best.pt'),normalizer_sha256=sha(run/'normalizer.json'),
        input_contract='canonical83_v1',z64_sha256=sha(out/'z64.npy')))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['gate','train','export'])
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path); p.add_argument('--run',type=Path); p.add_argument('--split',choices=['train','val','test'])
    a=p.parse_args(); c=json.loads(a.config.read_text())
    if a.action=='gate':
        try:
            entry_gate(a.manifest,c); print('TRAINING_ENTRY_PASS')
        except ValueError as error:
            print(json.dumps({'status':'TRAINING_ENTRY_BLOCKED','reason':str(error),
                              'training_runs':0,'optimizer_steps':0},indent=2))
            sys.exit(2)
    elif a.action=='train':
        if a.out is None: p.error('--out required')
        train(a.manifest,c,a.out)
    else:
        if a.out is None or a.run is None or a.split is None: p.error('--out --run --split required')
        export(a.manifest,c,a.run,a.out,a.split)
