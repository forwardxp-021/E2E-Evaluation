#!/usr/bin/env python3
"""Bounded real-Waymo engineering materialization; no source rescan or certification."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from tools.rbr_v2_train import sha,write
from tools.rbr_v2_canonical83 import from_waymo,semantic13
from tools.rbr_v2_waymo_source import Scenario

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rbr_v2_smoke_v1'
OLD=ROOT/'outputs/rbr_v2_uonly_v1'
CFG=ROOT/'configs/rbr_v2_smoke_v1/protocol.json'
SOURCE_FILES=['tools/rbr_v2_canonical83.py','tools/rbr_v2_model.py','tools/rbr_v2_waymo_source.py',
              'tools/rbr_v2_train.py','tools/lane_aware_assignment.py','tools/waymo_lane_utils.py',
              'tools/rbr_v2_smoke_data.py']


def identity(r):
    return ':'.join([r['scenario_id'],r['actor_id'],str(r['start']),str(r['length'])])


def role_for_group(g):
    n=int(hashlib.sha256(('RBR_V2_SMOKE_V1:GROUP:'+g).encode()).hexdigest()[:16],16)/2**64
    return 'train' if n<.9 else 'diagnostic'


def ordered_candidates(rows):
    forbidden={r['dependency_group'] for r in rows if r['candidate_split']=='test' or r['legacy_split']=='test'}
    result={'train':[],'diagnostic':[]}
    for r in rows:
        if r['candidate_split']!='train' or r['dependency_group'] in forbidden:
            continue
        r=dict(r); r['role']=role_for_group(r['dependency_group'])
        r['identity_sha256']=hashlib.sha256(('RBR_V2_SMOKE_V1:WINDOW:'+identity(r)).encode()).hexdigest()
        result[r['role']].append(r)
    for rs in result.values():
        rs.sort(key=lambda r:(r['identity_sha256'],identity(r)))
        for i,r in enumerate(rs):r['rank']=i
    return result,forbidden


def freeze():
    if (OUT/'selection_freeze.json').exists():
        raise FileExistsError('Selection already frozen; do not reset')
    cfg=json.loads(CFG.read_text())
    rows=list(csv.DictReader((OLD/'candidate_quality.csv').open()))
    ordered,forbidden=ordered_candidates(rows)
    write(OUT/'candidate_order.json',ordered)
    targets={'train':cfg['train_windows'],'diagnostic':cfg['diagnostic_windows']}
    capacity={k:sum(r['canonical_eligible']=='True' for r in rs) for k,rs in ordered.items()}
    if any(capacity[k]<targets[k] for k in targets):
        raise ValueError('Insufficient fixed-role physical input capacity: '+str(capacity))
    freeze=dict(state=cfg['state'],training_certified=False,raw_segment_disjointness='NOT_ESTABLISHED',
        policy_sha256=sha(CFG),candidate_order_sha256=sha(OUT/'candidate_order.json'),
        input_metadata_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'candidate_quality.csv',
            OLD/'candidate_split.csv',OLD/'dependency_groups.json',OLD/'source_records.csv',OLD/'data_manifest.json']},
        source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},capacity=capacity,
        forbidden_groups=len(forbidden),targets=targets,
        materialization_started=False,selection_basis='identity/observed dependency groups/physical masks only',
        source_risk='Observed group disjointness is NOT upstream segment disjointness')
    write(OUT/'selection_freeze.json',freeze)
    print(json.dumps(freeze,indent=2))


def worker(job):
    """Exact previous production interface, one immutable new NPZ per identity."""
    row,rec,destination=job; started=time.monotonic()
    with (ROOT.parent/'training'/rec['raw_file']).open('rb') as f:
        f.seek(int(rec['offset'])+12); payload=f.read(int(rec['payload_bytes']))
    if hashlib.sha256(payload).hexdigest()!=rec['payload_sha256']:
        raise ValueError('Raw payload SHA mismatch: '+row['scenario_id'])
    s=Scenario();s.ParseFromString(payload)
    if s.scenario_id!=row['scenario_id']:
        raise ValueError('Raw Scenario identity mismatch')
    v=from_waymo(s,row['actor_id'],int(row['start'])); y,ym=semantic13(v)
    eligible=int(v['ego_valid'].sum())>=64 and int(v['channel_valid'][:,6].sum())>=2
    if not np.isfinite(v['x']).all() or not np.isfinite(y[ym]).all():
        raise ValueError('Nonfinite real canonical data')
    result={**row,'valid_ego_frames':int(v['ego_valid'].sum()),
        'valid_acceleration_cells':int(v['channel_valid'][:,6].sum()),
        'valid_channel_cells':int(v['channel_valid'].sum()),
        'present_neighbor_frames':int(sum(v['x'][:,8+15*j].sum() for j in range(5))),
        'semantic_support':ym.astype(int).tolist(),'materialize_seconds':time.monotonic()-started,
        'raw_file':rec['raw_file'],'raw_record_offset':int(rec['offset']),
        'raw_payload_sha256':rec['payload_sha256'],'eligible':eligible}
    if not eligible:
        result['exclusion_reason']='INSUFFICIENT_VALID_PHYSICAL_EGO_SUPPORT'
        return result
    p=Path(destination)
    if p.exists():raise FileExistsError('Refusing to overwrite '+str(p))
    np.savez_compressed(p,x=v['x'],channel_valid=v['channel_valid'],ego_valid=v['ego_valid'],
        slot_ids=v['slot_ids'],time_us=v['time_us'],semantic=y,semantic_valid=ym)
    result.update(path=str(p.relative_to(OUT)),sha256=sha(p),split=row['role'])
    return result


def validate_frozen():
    f=json.loads((OUT/'selection_freeze.json').read_text())
    if sha(CFG)!=f['policy_sha256'] or sha(OUT/'candidate_order.json')!=f['candidate_order_sha256']:
        raise ValueError('Frozen selection changed')
    for p,h in {**f['source_sha256'],**f['input_metadata_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('Bound input/source changed: '+p)
    return f


def materialize(workers):
    frozen=validate_frozen(); started=time.monotonic()
    if (OUT/'materialization_lease.json').exists():
        raise FileExistsError('Materialization already claimed; preserve partial evidence')
    write(OUT/'materialization_lease.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        workers=workers,state='CLAIMED'))
    ordered=json.loads((OUT/'candidate_order.json').read_text())
    source={r['scenario_id']:r for r in csv.DictReader((OLD/'source_records.csv').open())}
    accepted=[]; rejected=[]; progress=[]
    data_dir=OUT/'data';data_dir.mkdir()
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as pool:
        for role,target in frozen['targets'].items():
            count=0; cursor=0; rows=ordered[role]
            while count<target:
                batch=[]
                while len(batch)<min(32,target-count) and cursor<len(rows):
                    r=rows[cursor];cursor+=1
                    if r['canonical_eligible']!='True':
                        rejected.append({**r,'exclusion_reason':r['exclusion_reason'],
                                         'stage':'prior_complete_raw_physical_quality_check'})
                        continue
                    path=data_dir/('%s_rank%06d.npz'%(role,r['rank']))
                    batch.append((r,source[r['scenario_id']],str(path)))
                if not batch:raise ValueError('Candidate capacity exhausted')
                # Ordered map preserves rank; all physical invalids remain in the denominator.
                for result in pool.map(worker,batch,chunksize=1):
                    if result['eligible']:
                        accepted.append(result);count+=1
                    else:
                        rejected.append({**result,'stage':'new_real_canonical_materialization'})
                elapsed=time.monotonic()-started
                progress.append(dict(role=role,accepted=count,target=target,rank_cursor=cursor,elapsed_seconds=elapsed))
                write(OUT/'materialization_progress.json',progress)
                write(OUT/'materialization_rejections.json',rejected)
                print('%s materialized %d/%d; rejected=%d elapsed=%.1fs'%(role,count,target,len(rejected),elapsed),flush=True)
    train_groups={r['dependency_group'] for r in accepted if r['role']=='train'}
    diag_groups={r['dependency_group'] for r in accepted if r['role']=='diagnostic'}
    if train_groups & diag_groups:raise ValueError('Observed dependency groups overlap')
    manifest=dict(state='RBR_V2_ENGINEERING_SMOKE_ONLY',checkpoint_label='SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT',
        training_certified=False,primary_encoder_eligible=False,raw_segment_disjointness='NOT_ESTABLISHED',
        input_contract='canonical83_v1',rows=accepted,counts={role:sum(r['role']==role for r in accepted) for role in frozen['targets']},
        observed_groups=dict(train=len(train_groups),diagnostic=len(diag_groups),intersection=0),
        excluded_invalid=len(rejected),rejection_sha256=sha(OUT/'materialization_rejections.json'),
        elapsed_seconds=time.monotonic()-started,selection_freeze_sha256=sha(OUT/'selection_freeze.json'),
        policy_sha256=sha(CFG),source_sha256=frozen['source_sha256'],
        use='Engineering numerical checks only; diagnostics are NOT U_val/generalization evidence')
    write(OUT/'data_manifest.json',manifest)
    write(OUT/'materialization_lease.json',dict(state='COMPLETED',workers=workers,elapsed_seconds=manifest['elapsed_seconds']))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('rows','source_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','materialize']);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='freeze':freeze()
    else:materialize(a.workers)
