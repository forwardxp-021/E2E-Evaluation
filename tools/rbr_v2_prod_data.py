#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete versioned canonical83 production, source-file shards and restartable cache."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import hashlib
import json
import multiprocessing
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from tools.rbr_v2_prod_features import PreparedScenario
from tools.rbr_v2_canonical83 import semantic13,GRID_US,from_waymo
from tools.rbr_v2_waymo_source import Scenario

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_prod_v1';OLD=ROOT/'outputs/rbr_v2_uonly_v1'
CFG=ROOT/'configs/rbr_v2_prod_v1/protocol.json'
SOURCE_FILES=['tools/rbr_v2_prod_data.py','tools/rbr_v2_prod_features.py','tools/rbr_v2_canonical83.py',
              'tools/rbr_v2_waymo_source.py','tools/waymo_lane_utils.py','tools/lane_aware_assignment.py']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.writing')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(temp,path)


def ident(r):return ':'.join([r['scenario_id'],r['actor_id'],str(r['start']),str(r['length'])])


def plan():
    if (OUT/'production_plan.json').exists():raise FileExistsError('Plan already frozen')
    rows=list(csv.DictReader((OLD/'candidate_quality.csv').open()))
    forbidden={r['dependency_group'] for r in rows if r['candidate_split']=='test' or r['legacy_split']=='test'}
    records={r['scenario_id']:r for r in csv.DictReader((OLD/'source_records.csv').open())}
    eligible=[r for r in rows if r['candidate_split']=='train' and r['dependency_group'] not in forbidden]
    eligible.sort(key=lambda r:(records[r['scenario_id']]['raw_file'],int(records[r['scenario_id']]['offset']),r['actor_id'],int(r['start'])))
    valid=[r for r in eligible if r['canonical_eligible']=='True']
    diagnostic={ident(r) for r in sorted(valid,key=lambda r:hashlib.sha256(('RBR_V2_PROD_V1:DIAGNOSTIC:'+ident(r)).encode()).digest())[:256]}
    expected=json.loads((OLD/'data_manifest.json').read_text())['raw_files']
    raw_files=list(sorted((ROOT.parent/'training').glob('*tfrecord*')))
    official_train=all(re.fullmatch(r'uncompressed_scenario_training_training\.tfrecord-\d{5}-of-01000',p.name) for p in raw_files)
    if len(raw_files)!=51 or not official_train:raise ValueError('Unexpected raw source/split inventory')
    for item in expected:
        if Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('Raw source size changed')
    files=defaultdict(list)
    for r in eligible:files[records[r['scenario_id']]['raw_file']].append(r)
    jobs=[];base=0
    for i,(name,rs) in enumerate(sorted(files.items())):
        count=sum(r['canonical_eligible']=='True' for r in rs)
        jobs.append(dict(shard_id='file%05d'%i,raw_file=name,rows=rs,expected_rows=count,global_start=base,
                         records={sid:records[sid] for sid in {r['scenario_id'] for r in rs}},
                         raw_source=next(x for x in expected if Path(x['path']).name==name)))
        base+=count
    write(OUT/'jobs.json',jobs)
    with (OUT/'planned_candidates.csv').open('w',newline='') as f:
        fields=list(eligible[0])+['raw_file','raw_record_offset','diagnostic_in_sample']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in eligible:w.writerow({**r,'raw_file':records[r['scenario_id']]['raw_file'],
            'raw_record_offset':records[r['scenario_id']]['offset'],'diagnostic_in_sample':ident(r) in diagnostic})
    obj=dict(state='OWNER_AUTHORIZED_U_ONLY_WITH_DISCLOSED_SOURCE_RISK',selection='FIXED_EPOCH_30_NO_U_VAL_CHECKPOINT_SELECTION',
        raw_segment_disjointness='NOT_ESTABLISHED',official_split='TRAINING_FILENAME_AND_SCHEMA_CONSISTENT',
        official_independent_validation=False,release_download_receipt='NOT_AVAILABLE_IN_LOCAL_ASSETS',
        source_note='51 of1000 named training containers; original acquisition segment/global-time mapping remains absent',
        official_reference='https://waymo.com/intl/it/open/data/motion/',
        denominator=len(rows),candidate_train_before_test_group_isolation=sum(r['candidate_split']=='train' for r in rows),
        candidates_after_isolation=len(eligible),expected_train_rows=len(valid),expected_invalid=len(eligible)-len(valid),
        observed_train_groups=len({r['dependency_group'] for r in valid}),scenarios=len({r['scenario_id'] for r in eligible}),
        quarantined_groups=len(forbidden),diagnostic_identities=sorted(diagnostic),diagnostic_role='IN_SAMPLE_NUMERICAL_ONLY',
        cfg_sha256=sha(CFG),jobs_sha256=sha(OUT/'jobs.json'),candidate_roster_sha256=sha(OUT/'planned_candidates.csv'),
        input_metadata_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'candidate_quality.csv',OLD/'dependency_groups.json',OLD/'source_records.csv',OLD/'data_manifest.json']},
        source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},raw_inventory=expected)
    write(OUT/'production_plan.json',obj);print(json.dumps({k:v for k,v in obj.items() if k not in ('raw_inventory','diagnostic_identities')},indent=2))


def verify_shard(directory):
    m=json.loads((directory/'COMMITTED.json').read_text())
    for name,h in m['sha256'].items():
        if sha(directory/name)!=h:raise ValueError('Certified shard changed: '+str(directory/name))
    return m


def produce_file(job):
    started=time.monotonic();dest=OUT/'cache'/job['shard_id']
    if dest.exists():return verify_shard(dest)
    attempt=1
    while (OUT/'work'/('%s_attempt%d'%(job['shard_id'],attempt))).exists():attempt+=1
    work=OUT/'work'/('%s_attempt%d'%(job['shard_id'],attempt));work.mkdir(parents=True)
    status=dict(shard_id=job['shard_id'],attempt=attempt,state='STARTED',pid=os.getpid())
    write(work/'attempt.json',status)
    try:
        source=Path(job['raw_source']['path'])
        if sha(source)!=job['raw_source']['sha256']:raise ValueError('Raw source SHA changed: '+str(source))
        n=job['expected_rows'];shapes={'x':((n,80,83),'float32'),'channel_valid':((n,80,83),'bool'),
            'ego_valid':((n,80),'bool'),'semantic':((n,13),'float32'),'semantic_valid':((n,13),'bool'),
            'slot_ids':((n,80,5),'int64')}
        arrays={k:np.lib.format.open_memmap(work/(k+'.npy'),mode='w+',dtype=dt,shape=shape) for k,(shape,dt) in shapes.items()}
        np.save(work/'time_us.npy',GRID_US)
        by_sid=defaultdict(list)
        for r in job['rows']:by_sid[r['scenario_id']].append(r)
        index=0;excluded=[];metadata=[];repeat=[]
        with source.open('rb') as f:
            for sid,rs in by_sid.items():
                rec=job['records'][sid];f.seek(int(rec['offset'])+12);payload=f.read(int(rec['payload_bytes']))
                if hashlib.sha256(payload).hexdigest()!=rec['payload_sha256']:raise ValueError('Payload identity mismatch')
                s=Scenario();s.ParseFromString(payload)
                if s.scenario_id!=sid:raise ValueError('Scenario identity mismatch')
                prepared={}
                for r in rs:
                    start=int(r['start'])
                    if start not in prepared:prepared[start]=PreparedScenario(s,start)
                    p=prepared[start];e=p.tracks[r['actor_id']];ev=e[:,5]>.5
                    valid=int(ev.sum())>=64 and int((ev[1:]&ev[:-1]).sum())>=2
                    if valid!=(r['canonical_eligible']=='True'):raise ValueError('Frozen input eligibility disagrees with actual raw data')
                    if not valid:
                        excluded.append({**r,'actual_valid_ego_frames':int(ev.sum()),'reason':'INSUFFICIENT_VALID_PHYSICAL_EGO_SUPPORT'});continue
                    v=p.sample(r['actor_id']);y,ym=semantic13(v)
                    if not np.isfinite(v['x']).all() or not np.isfinite(y[ym]).all():raise ValueError('Nonfinite canonical data')
                    for key in ['x','channel_valid','ego_valid']:arrays[key][index]=v[key]
                    arrays['semantic'][index]=y;arrays['semantic_valid'][index]=ym
                    arrays['slot_ids'][index]=np.where(v['slot_ids']=='','-1',v['slot_ids']).astype(np.int64)
                    metadata.append(dict(global_index=job['global_start']+index,shard_row=index,scenario_id=sid,
                        actor_id=r['actor_id'],start=start,length=80,dependency_group=r['dependency_group'],
                        raw_file=source.name,raw_record_offset=int(rec['offset']),raw_payload_sha256=rec['payload_sha256'],
                        valid_ego_frames=int(ev.sum()),valid_channel_cells=int(v['channel_valid'].sum()),
                        neighbor_present_frames=int(sum(v['x'][:,8+15*j].sum() for j in range(5)))))
                    # Deterministic re-production for the first accepted window in EVERY shard.
                    if index==0:
                        again=PreparedScenario(s,start).sample(r['actor_id'])
                        reference=from_waymo(s,r['actor_id'],start)
                        for key in ['x','channel_valid','ego_valid','slot_ids','time_us']:
                            np.testing.assert_array_equal(v[key],again[key])
                            np.testing.assert_array_equal(v[key],reference[key])
                        repeat.append(dict(identity=ident(r),status='ORIGINAL_REFERENCE_AND_REPEAT_BITWISE_PASS'))
                    index+=1
        if index!=n:raise ValueError('Full training denominator changed')
        for arr in arrays.values():arr.flush()
        del arrays
        with (work/'rows.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metadata[0]),lineterminator='\n');w.writeheader();w.writerows(metadata)
        write(work/'excluded.json',excluded)
        summary=dict(shard_id=job['shard_id'],rows=n,candidates=len(job['rows']),excluded=len(excluded),
            global_start=job['global_start'],raw_file=source.name,raw_source_sha256=job['raw_source']['sha256'],
            elapsed_seconds=time.monotonic()-started,attempt=attempt,repeat_tests=repeat,
            observed_groups=len({r['dependency_group'] for r in metadata}),
            valid_ego_frames=sum(r['valid_ego_frames'] for r in metadata),
            valid_channel_cells=sum(r['valid_channel_cells'] for r in metadata),
            neighbor_present_frames=sum(r['neighbor_present_frames'] for r in metadata),
            producer_source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},
            sha256={p.name:sha(p) for p in sorted(work.iterdir()) if p.is_file() and p.name!='attempt.json'})
        write(work/'COMMITTED.json',summary);dest.parent.mkdir(exist_ok=True);os.replace(work,dest)
        return summary
    except BaseException as error:
        status.update(state='FAILED_PRESERVED',error=str(error),elapsed_seconds=time.monotonic()-started)
        write(work/'attempt.json',status);raise


def produce(workers):
    p=json.loads((OUT/'production_plan.json').read_text())
    if sha(CFG)!=p['cfg_sha256'] or sha(OUT/'jobs.json')!=p['jobs_sha256']:raise ValueError('Plan/config binding changed')
    for path,h in {**p['source_sha256'],**p['input_metadata_sha256']}.items():
        if sha(ROOT/path)!=h:raise ValueError('Bound production input changed: '+path)
    jobs=json.loads((OUT/'jobs.json').read_text());done=[];pending=[];began=time.monotonic()
    if (OUT/'data_manifest.json').exists():
        existing=json.loads((OUT/'data_manifest.json').read_text())
        for shard in existing['shards']:verify_shard(OUT/'cache'/shard['shard_id'])
        print('Production already complete; all immutable shards verified without rewriting manifest',flush=True)
        return
    for j in jobs:
        directory=OUT/'cache'/j['shard_id']
        if directory.exists():done.append(verify_shard(directory))
        else:pending.append(j)
    print('Certified shards reused=%d pending=%d'%(len(done),len(pending)),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(produce_file,j):j['shard_id'] for j in pending}
        for future in as_completed(futures):
            result=future.result();done.append(result)
            write(OUT/'production_progress.json',dict(certified_shards=len(done),total_shards=len(jobs),
                rows=sum(x['rows'] for x in done),this_process_wall_seconds=time.monotonic()-began,
                worker_seconds=sum(x['elapsed_seconds'] for x in done)))
            print('Production %d/%d shards, rows=%d elapsed=%.1fs last=%s %.1fs'%
                (len(done),len(jobs),sum(x['rows'] for x in done),time.monotonic()-began,result['shard_id'],result['elapsed_seconds']),flush=True)
    done.sort(key=lambda x:x['shard_id'])
    if sum(s['rows'] for s in done)!=p['expected_train_rows']:raise ValueError('Final denominator mismatch')
    manifest=dict(schema='RBR_V2_PROD_CANONICAL83_V1',authorization=p['state'],
        raw_segment_disjointness='NOT_ESTABLISHED',U_source_complete_independence=False,
        owner_authorized_training=True,independent_validation_certified=False,
        selection='FIXED_EPOCH_30_NO_U_VAL_CHECKPOINT_SELECTION',rows=sum(s['rows'] for s in done),
        candidates=p['candidates_after_isolation'],excluded=sum(s['excluded'] for s in done),
        observed_train_groups=p['observed_train_groups'],diagnostic_identities=p['diagnostic_identities'],
        diagnostic_role='IN_SAMPLE_NUMERICAL_ONLY',shards=done,input_contract='canonical83_v1',
        plan_sha256=sha(OUT/'production_plan.json'),source_sha256=p['source_sha256'],
        jobs_sha256=p['jobs_sha256'],source_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',
        worker_seconds=sum(s['elapsed_seconds'] for s in done),finalize_process_wall_seconds=time.monotonic()-began)
    write(OUT/'data_manifest.json',manifest);print('COMPLETE: %d real training rows in %d certified shards'%(manifest['rows'],len(done)),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','produce']);a.add_argument('--workers',type=int,default=6)
    args=a.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    if args.action=='plan':plan()
    else:produce(args.workers)
