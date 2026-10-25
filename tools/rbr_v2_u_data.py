#!/usr/bin/env python3
"""New U source/dependency census and canonical83 materializer; legacy assets read-only."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tools.rbr_v2_waymo_source import Scenario, records, load_record
from tools.rbr_v2_canonical83 import waymo_tracks, from_waymo, semantic13, resample

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'docs/stageR/s3_p0_rbr_training_entry'
DEFAULT = ROOT/'outputs/rbr_v2_uonly_v1'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


class Groups:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while x != self.p[x]:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def split_for_group(group):
    q = int(hashlib.sha256(('2026102302:'+group).encode()).hexdigest()[:16], 16)/2**64
    return 'train' if q < .8 else ('val' if q < .9 else 'test')


def census(out):
    if (out/'data_manifest.json').exists():
        raise FileExistsError('Version already exists; do not overwrite a data ledger')
    begin = time.monotonic()
    old = json.loads((OLD/'S3_P0_U_Data_Split_Manifest_v1.json').read_text())
    inventory = old['raw_metadata']['files']
    index = list(csv.DictReader((OLD/'evidence/U_Raw_Record_Index_v1.csv').open()))
    sid_index = {r['scenario_id']: i for i, r in enumerate(index)}
    groups = Groups(len(index)); counts = Counter(); files = []
    # Selection inherited only from U metadata; no learned features or downstream outcomes.
    dynamic = json.loads((ROOT/'outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json').read_text())
    identities = []; wanted = defaultdict(set)
    for sp in dynamic['shard_paths']:
        sd = Path(sp); meta = np.load(sd/'meta.npy', allow_pickle=True)
        for row in meta:
            sid, aid = str(row['scenario_id']), str(row['target_agent_id'])
            identities.append(dict(scenario_id=sid, actor_id=aid, start=int(row['start']),
                                   length=int(row['window_len']), legacy_split=str(row['split'])))
            wanted[sid].add(aid)
    write(out/'inherited_window_identities.json', identities)
    # External-sort streaming keeps RAM bounded. All moving vehicle tracks, not just egos.
    anchors_path = out/'world_overlap_anchors.tsv'
    with anchors_path.open('w') as anchors, (out/'source_records.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['scenario_id','raw_file','offset','payload_bytes','payload_sha256',
            'frame_count','vehicles','camera_frames','lidar_frames','timestamp_start','timestamp_end','segment_id'])
        writer.writeheader()
        for fi, item in enumerate(inventory):
            path = Path(item['path']); h = hashlib.sha256(); nr = 0
            for offset, payload, header, tail in records(path):
                h.update(header); h.update(payload); h.update(tail)
                s = Scenario(); s.ParseFromString(payload); sid = s.scenario_id; si = sid_index[sid]
                times = np.asarray(s.timestamps_seconds, np.float64)
                if len(times) < 80 or not np.all(np.diff(times) > 0):
                    raise ValueError('Invalid source physical time: '+sid)
                tr = waymo_tracks(s)
                writer.writerow(dict(scenario_id=sid, raw_file=path.name, offset=offset,
                    payload_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest(),
                    frame_count=len(times), vehicles=len(tr), camera_frames=len(s.frame_camera_tokens),
                    lidar_frames=len(s.compressed_frame_laser_data), timestamp_start=times[0],
                    timestamp_end=times[-1], segment_id='NOT_AVAILABLE'))
                counts['scenarios'] += 1; counts['vehicle_tracks'] += len(tr)
                counts['camera_frames'] += len(s.frame_camera_tokens); counts['lidar_frames'] += len(s.compressed_frame_laser_data)
                counts['zero_start_scenarios'] += int(times[0] == 0)
                for aid, a in tr.items():
                    counts['retained_actor_tracks'] += int(aid in wanted[sid])
                    # 0.8 s overlapping physical snippets, EVERY source-frame shift.
                    # Quantized variant is conservative, with <=1cm / .01 m/s / .001rad bins;
                    # it merges possible dependencies, never certifies absence of approximate overlap.
                    for j in range(len(a)-7):
                        seg = a[j:j+8]
                        if not np.all(seg[:, 5] > .5):
                            continue
                        if np.linalg.norm(seg[-1, :2]-seg[0, :2]) < .5:
                            continue
                        time_deltas = np.rint(np.diff(times[j:j+8])*1e6).astype('<i8').tobytes()
                        raw = np.asarray(seg[:, :5], dtype='<f8').tobytes()+time_deltas
                        key = hashlib.blake2b(raw, digest_size=16).hexdigest()
                        anchors.write('E'+key+'\t'+str(si)+'\n')
                        q = np.rint(seg[:, :5]/np.array([.01,.01,.01,.01,.001])).astype('<i8')
                        key = hashlib.blake2b(q.tobytes()+time_deltas, digest_size=16).hexdigest()
                        anchors.write('Q'+key+'\t'+str(si)+'\n')
                        counts['moving_overlap_snippets'] += 1
                nr += 1
            if h.hexdigest() != item['sha256']:
                raise ValueError('Raw source changed: '+str(path))
            files.append(dict(path=str(path), sha256=h.hexdigest(), records=nr, bytes=path.stat().st_size))
            print('U source census %d/%d; scenarios=%d tracks=%d elapsed=%.1fs' %
                  (fi+1,len(inventory),counts['scenarios'],counts['vehicle_tracks'],time.monotonic()-begin), flush=True)
    sorted_path = out/'world_overlap_anchors.sorted.tsv'
    with sorted_path.open('w') as f:
        subprocess.run(['sort','-S','256M','-T',str(out),str(anchors_path)], stdout=f,
                       env={**os.environ,'LC_ALL':'C'}, check=True)
    edges = Counter(); prev = None; first = None
    with sorted_path.open() as f, (out/'dependency_edges.csv').open('w', newline='') as ef:
        ew = csv.writer(ef); ew.writerow(['kind','scenario_a','scenario_b','snippet_digest'])
        for line in f:
            key, sn = line.rstrip().split('\t'); sn = int(sn)
            if key != prev:
                first = sn; prev = key
            elif sn != first and groups.find(sn) != groups.find(first):
                ew.writerow([key[0],index[first]['scenario_id'],index[sn]['scenario_id'],key[1:]])
                groups.union(first, sn); edges[key[0]] += 1
    # Raw shards are containers, NOT independent source segments. We do not treat file as source group.
    members = defaultdict(list)
    for sid, i in sid_index.items():
        members[groups.find(i)].append(sid)
    gid = {sid: min(ss) for ss in members.values() for sid in ss}
    with (out/'candidate_split.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(identities[0])+['dependency_group','candidate_split','certified'])
        w.writeheader()
        for row in identities:
            g = gid[row['scenario_id']]; sp = split_for_group(g)
            counts['candidate_'+sp] += 1
            w.writerow({**row, 'dependency_group':g,'candidate_split':sp,'certified':False})
    write(out/'dependency_groups.json', {min(v):sorted(v) for v in members.values()})
    # Exact and quantized matching do not prove disjointness of invisible segments/near-overlaps.
    manifest = dict(schema='rbr_v2_u_candidate_v1', training_certified=False,
        status='BLOCKED_U_SOURCE_INDEPENDENCE', raw_segment_disjointness='NOT_ESTABLISHED',
        source_policy='No upstream segment/global-object/absolute-time map exists in these inputs',
        split_policy='Observed moving-overlap components BEFORE deterministic group hash split',
        overlap_scope='All vehicle tracks; 8 valid adjacent frames, stride1; >=0.5m displacement; exact + quantized',
        residual_risks=['unidentified common raw segment','stationary/short overlaps','near matches straddling quantization bins',
                        'nonoverlapping trajectories from same source segment'],
        counts=dict(counts), observed_groups=len(members), multi_scenario_groups=sum(len(v)>1 for v in members.values()),
        largest_group=max(map(len,members.values())), union_edges=dict(edges), raw_files=files,
        inherited_selection='Legacy U identity/eligibility only; physical arrays never reused as canonical truth',
        source_index_sha256=sha(out/'source_records.csv'), candidate_split_sha256=sha(out/'candidate_split.csv'),
        elapsed_seconds=time.monotonic()-begin, crc='Whole-file SHA256 checked; individual CRC verification not claimed')
    write(out/'data_manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('raw_files',)}, indent=2), flush=True)


def materialize(out, limit):
    """Real production adapter validation data. Never promoted to certified training data."""
    rows = list(csv.DictReader((out/'candidate_split.csv').open()))
    source = {r['scenario_id']:r for r in csv.DictReader((out/'source_records.csv').open())}
    # Deterministic identities selected before processing; no output-dependent replacement.
    rows.sort(key=lambda r: hashlib.sha256((r['scenario_id']+':'+r['actor_id']).encode()).digest())
    if limit:
        rows = rows[:limit]
    dest = out/'canonical_smoke_v1'
    if dest.exists():
        raise FileExistsError('Do not overwrite materialization version')
    dest.mkdir(); results=[]; failed=[]
    for i, r in enumerate(rows):
        p = source[r['scenario_id']]
        s = load_record(ROOT.parent/'training'/p['raw_file'],p['offset'],p['payload_bytes'])
        v = from_waymo(s, r['actor_id'], int(r['start'])); target, tm = semantic13(v)
        destfile = dest/('%06d.npz'%i)
        np.savez_compressed(destfile, x=v['x'], channel_valid=v['channel_valid'], ego_valid=v['ego_valid'],
                            slot_ids=v['slot_ids'], time_us=v['time_us'], semantic=target, semantic_valid=tm)
        eligible = int(v['ego_valid'].sum()) >= 64 and int(v['channel_valid'][:,6].sum()) >= 2
        results.append({**r,'path':str(destfile.relative_to(ROOT)), 'sha256':sha(destfile),
                        'input_eligible':eligible,'valid_ego_frames':int(v['ego_valid'].sum()),
                        'valid_neighbor_cells':int(v['channel_valid'][:,8:].sum())})
        if not eligible:
            failed.append(i)
        print('Canonical real raw interface %d/%d' % (i+1,len(rows)), flush=True)
    write(dest/'manifest.json', dict(role='U_ENGINEERING_ONLY_NOT_TRAINING_CERTIFIED', rows=results,
        source_manifest_sha256=sha(out/'data_manifest.json'), excluded_invalid=len(failed),
        source_population=len(source), materialized=len(rows), training_certified=False))


def quality(out):
    """Full candidate ego eligibility and raw schema scan, not model evaluation."""
    destination=out/'candidate_quality.csv'
    if destination.exists():
        raise FileExistsError('Do not overwrite the quality audit')
    start=time.monotonic(); rows=list(csv.DictReader((out/'candidate_split.csv').open()))
    by_sid=defaultdict(list)
    for r in rows: by_sid[r['scenario_id']].append(r)
    src={r['scenario_id']:r for r in csv.DictReader((out/'source_records.csv').open())}
    counts=Counter(); top_fields=Counter(); exclusions=Counter()
    from tools.s3_p0_training_entry_audit import wire_fields
    with destination.open('w',newline='') as f:
        names=list(rows[0])+['valid_ego_frames','valid_adjacent_differences','canonical_eligible','exclusion_reason']
        w=csv.DictWriter(f,fieldnames=names); w.writeheader()
        for fi,file in enumerate(json.loads((out/'data_manifest.json').read_text())['raw_files']):
            for off,payload,header,tail in records(file['path']):
                s=Scenario(); s.ParseFromString(payload)
                if hashlib.sha256(payload).hexdigest()!=src[s.scenario_id]['payload_sha256']:
                    raise ValueError('Raw record changed during quality scan')
                for field in {k for k,wire,value in wire_fields(payload)}:
                    top_fields[str(field)]+=1
                needed=by_sid.get(s.scenario_id,[])
                if not needed: continue
                wanted={r['actor_id'] for r in needed}; tracks={}
                fields=['center_x','center_y','velocity_x','velocity_y','heading']
                for tr in s.tracks:
                    if str(tr.id) in wanted:
                        tracks[str(tr.id)]=np.array([[getattr(a,k) for k in fields]+[
                            a.valid and all(a.HasField(k) for k in fields)] for a in tr.states],np.float64)
                times=np.rint(np.array(s.timestamps_seconds,np.float64)*1e6).astype(np.int64)
                for r in needed:
                    a=resample(tracks[r['actor_id']],times,times[int(r['start'])])
                    valid=a[:,5]>.5; diffs=valid[1:] & valid[:-1]
                    reason='' if valid.sum()>=64 and diffs.sum()>=2 else 'INSUFFICIENT_VALID_PHYSICAL_EGO_SUPPORT'
                    counts['windows']+=1; counts['eligible']+=int(not reason); counts['excluded']+=int(bool(reason))
                    counts['eligible_'+r['candidate_split']]+=int(not reason)
                    if reason: exclusions[reason]+=1
                    w.writerow({**r,'valid_ego_frames':int(valid.sum()),'valid_adjacent_differences':int(diffs.sum()),
                                'canonical_eligible':not reason,'exclusion_reason':reason})
            print('U raw quality %d/51, windows=%d excluded=%d'%(fi+1,counts['windows'],counts['excluded']),flush=True)
    write(out/'quality_audit.json',dict(counts=dict(counts),exclusion_reasons=dict(exclusions),
        exclusion_rate=counts['excluded']/counts['windows'],top_level_field_record_counts=dict(top_fields),
        unknown_top_level_fields=sorted(set(top_fields)-set(map(str,range(1,14)))),
        scope='Full inherited U ego support only; complete context eligibility awaits full canonical materialization',
        elapsed_seconds=time.monotonic()-start,quality_sha256=sha(destination),training_certified=False))


def verify_materialization(out):
    """Replay fixed input interface examples after source edits; never alter their bytes."""
    begin=time.monotonic()
    src={r['scenario_id']:r for r in csv.DictReader((out/'source_records.csv').open())}
    m=json.loads((out/'canonical_smoke_v1/manifest.json').read_text())
    for row in m['rows']:
        r=src[row['scenario_id']]
        s=load_record(ROOT.parent/'training'/r['raw_file'],r['offset'],r['payload_bytes'])
        a=from_waymo(s,row['actor_id'],int(row['start'])); y,ym=semantic13(a)
        if sha(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Materialized artifact changed')
        with np.load(ROOT/row['path']) as f:
            for k in ['x','channel_valid','ego_valid','slot_ids','time_us']:
                np.testing.assert_array_equal(a[k],f[k])
            np.testing.assert_array_equal(y,f['semantic']); np.testing.assert_array_equal(ym,f['semantic_valid'])
    result=dict(status='PASS',rows_verified=len(m['rows']),elapsed_seconds=time.monotonic()-begin,
        current_canonical_source_sha256=sha(ROOT/'tools/rbr_v2_canonical83.py'),historical_artifacts_modified=False)
    write(out/'current_source_replay_check.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('action', choices=['census','materialize','quality','verify'])
    a.add_argument('--out',type=Path,default=DEFAULT); a.add_argument('--limit',type=int,default=16)
    args=a.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.action == 'census':
        census(args.out)
    elif args.action=='materialize':
        materialize(args.out,args.limit)
    elif args.action=='quality':
        quality(args.out)
    else:
        verify_materialization(args.out)
