#!/usr/bin/env python3
"""Read-only S3-P0 audit: physical fixtures, bounded U metadata, and hashes.

No training, checkpoint loading, simulator, detector, or evaluation entry point.
The two historical Waymo builders and nuPlan builder are called on fixtures only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/s3_p0_rbr_training_entry"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
EXPECTED_CSV = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
DYNAMIC = ROOT / "outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json"
LEGACY = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def protected_paths():
    folders = [p for p in (ROOT / "docs/stageR").iterdir()
               if p.is_dir() and p.name.startswith(("b1_", "s1", "s2"))]
    paths = [p for d in folders for p in d.rglob("*") if p.is_file()]
    paths += list((ROOT / "tools").glob("*.py")) + list((ROOT / "configs").glob("*"))
    return sorted(p for p in paths if p.is_file() and p.name != Path(__file__).name)


def snapshot():
    target = OUT / "evidence/Protected_Baseline_v1.json"
    if target.exists():
        raise FileExistsError(f"Refusing to replace baseline: {target}")
    assert sha(PROTECTED_CSV) == EXPECTED_CSV, "Protected CSV SHA differs"
    untracked = git("ls-files", "--others", "--exclude-standard")
    write(target, {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                   "tracked_dirty": git("status", "--porcelain", "--untracked-files=no"),
                   "preexisting_untracked_count": len(untracked.splitlines()),
                   "preexisting_untracked_listing_sha256": hashlib.sha256(untracked.encode()).hexdigest(),
                   "protected_csv_sha256": EXPECTED_CSV,
                   "file_sha256": {str(p.relative_to(ROOT)): sha(p) for p in protected_paths()}})


def verify():
    baseline = read(OUT / "evidence/Protected_Baseline_v1.json")
    bad = [p for p, h in baseline["file_sha256"].items() if not (ROOT / p).exists() or sha(ROOT / p) != h]
    assert sha(PROTECTED_CSV) == EXPECTED_CSV
    assert not bad, f"Protected files changed: {bad}"
    result = {"status": "PASS", "protected_files_checked": len(baseline["file_sha256"]),
              "changed": bad, "protected_csv_sha256": EXPECTED_CSV}
    write(OUT / "evidence/Protected_Verification_v1.json", result)
    return result


def fixture_args(mode="geometric_only"):
    return argparse.Namespace(window_len=80, stride=20, dt=.1, min_valid_ratio=.8,
        min_speed=1., max_agents_per_scenario=1, assignment_mode=mode,
        lane_max_lateral_distance=3., lane_max_heading_diff_deg=45.,
        adjacent_lane_min_offset=2., adjacent_lane_max_offset=5.5,
        adjacent_lane_max_heading_diff_deg=35., lane_search_radius=20., lane_topk_candidates=32,
        disable_lane_spatial_index=True, front_max_distance=120., side_front_max_distance=80.,
        side_rear_max_distance=120., lane_lateral_tolerance=2., slot_heading_diff_deg=45.,
        static_speed_threshold=.5, drop_if_ego_lane_missing=False, ttc_cap=999., thw_cap=999.)


def make_fixture(name):
    """World states sampled from analytic straight/circular physical motion."""
    t = np.arange(80, dtype=np.float64) * .1
    if name == "irregular_time":
        t[1:] += np.cumsum(np.where(np.arange(1, 80) % 2, .02, -.02))
    omega = .04 if name in {"turn", "angle_wrap", "angular_gap"} else 0.
    initial = 3.1 if name in {"angle_wrap", "angular_gap"} else 0.
    speed = 8. + (.3 * t if name in {"acceleration", "irregular_time", "accelerating_gap"} else 0. * t)
    if omega:
        xy = np.column_stack([8. / omega * (np.sin(initial + omega*t)-np.sin(initial)),
                              8. / omega * (np.cos(initial)-np.cos(initial + omega*t))])
    else:
        xy = np.column_stack([8.*t + (.15*t*t if name in {"acceleration", "irregular_time", "accelerating_gap"} else 0.*t), 0.*t])
    if name == "start_rest":
        speed=t.copy();xy=np.column_stack([.5*t*t,0.*t])
    h = initial + omega*t
    track = np.column_stack([xy, speed*np.cos(h), speed*np.sin(h), (h+np.pi)%(2*np.pi)-np.pi, np.ones(80)]).astype(np.float32)
    if name == "sideslip":
        track[:, 3] += .8  # Same world velocity; nuPlan row drops this degree of freedom.
    tracks = {"ego": track}
    for i, (dx, dy) in enumerate([(25., 0.), (20., 3.5), (-20., 3.5), (30., -3.5), (-30., -3.5)]):
        other = track.copy()
        other[:, 0] += dx*np.cos(h)-dy*np.sin(h)
        other[:, 1] += dx*np.sin(h)+dy*np.cos(h)
        # Velocity derivative of the fixed rotating relative offset.
        other[:, 2] += omega*(-dx*np.sin(h)-dy*np.cos(h))
        other[:, 3] += omega*(dx*np.cos(h)-dy*np.sin(h))
        tracks[f"n{i}"] = other
    if name == "slot_switch":
        tracks["n0"][:, 0] = track[:, 0] + 25. - 2.*t
        other = track.copy(); other[:, 0] += 10. + 2.*t; other[:, 2] += 2.
        tracks["overtaker"] = other
        tracks["n0"][:, 2] -= 2.
    if name in {"missing_neighbor", "accelerating_gap"}:
        tracks["n0"][25:29, :5] = np.nan; tracks["n0"][25:29, 5] = 0.
    if name == "angular_gap":
        tracks["n0"][9:14, :5] = np.nan; tracks["n0"][9:14, 5] = 0.
    if name == "invalid_ego":
        track[25, 5] = 0.  # Finite pose retained to isolate validity handling.
    seq = np.zeros((80, 8), dtype=np.float32)
    seq[:, :2] = track[:, :2]; seq[:, 2] = track[:, 4]
    seq[:, 3] = np.hypot(track[:, 2], track[:, 3]); seq[:, 7] = t
    if name == "epoch_time": seq[:, 7] = t + 1_600_000_000.
    nu_tracks = {k: {j: tuple(float(x) for x in np.r_[v[j, :5], np.hypot(v[j, 2], v[j, 3])])
                      for j in range(80) if v[j, 5] > .5 and np.isfinite(v[j, :5]).all()}
                 for k, v in tracks.items() if k != "ego"}
    return tracks, seq, track[:, 5] > .5, nu_tracks


def fixture_lanes():
    from tools.waymo_lane_utils import LaneInfo, _build_lane_geom
    result = {}
    for key, y, left, right in [("0",0.,["1"],["2"]),("1",3.5,[],["0"]),("2",-3.5,["0"],[])]:
        points = np.asarray([[-100.,y],[200.,y]], dtype=np.float32)
        geom = _build_lane_geom(points)
        result[key] = LaneInfo(key, points, *geom, left_neighbor_lane_ids=left, right_neighbor_lane_ids=right)
    return result


def parity():
    from tools import build_waymo_5neighbor_context_dataset as legacy
    from tools import build_waymo_dynamic_interaction_dataset_v2 as dynamic
    from tools import build_nuplan_5neighbor_context_dataset as nu
    from tools.stage5d_context_core import build_context_traj_from_standard_tracks
    from tools.interaction_context_features import aggregate_interaction_features
    cases = ["straight", "acceleration", "turn", "angle_wrap", "irregular_time", "sideslip",
             "slot_switch", "missing_neighbor", "invalid_ego", "lane_aware", "epoch_time",
             "start_rest", "accelerating_gap", "angular_gap"]
    evidence = {"absolute_tolerance": .0002, "relative_tolerance": .00002, "cases": {},
                "scope": "actual processing APIs; synthetic physical states, no simulation or neural inference"}
    max_err = {"legacy": np.zeros(83), "dynamic": np.zeros(83)}
    semantic_err = {"legacy": np.zeros(83), "dynamic": np.zeros(83)}
    for name in cases:
        tracks, seq, mask, nu_tracks = make_fixture(name)
        args = fixture_args("lane_aware_only" if name == "lane_aware" else "geometric_only")
        lanes = fixture_lanes() if name == "lane_aware" else {}
        ne, nn, nc, ids, _ = nu.build_row_context(seq, mask, nu_tracks, args, lanes)
        row = {"nuplan_slot_ids": ids, "nuplan_estimated_dt": nu.estimate_dt(seq,mask),
               "distinct_serialized_timestamps":len(np.unique(seq[:,7])),
               "input_sha256":hashlib.sha256(b"".join(v.tobytes() for v in tracks.values())+seq.tobytes()+mask.tobytes()).hexdigest(),
               "pipelines": {}}
        for key, module in [("legacy",legacy),("dynamic",dynamic)]:
            b, _, _, _ = module.process_one_scenario("S3_P0_FIXTURE", {k:v.copy() for k,v in tracks.items()}, lanes, args, defaultdict(int), defaultdict(float), 0)
            assert len(b.context_traj) == 1
            e, n, c = b.ego_seq[0], b.neighbor_seq[0], b.context_traj[0]
            delta = np.abs(c.astype(float)-nc)
            semantics = np.concatenate([np.abs(e.astype(float)-ne), np.abs(n.astype(float)-nn).transpose(1,0,2).reshape(80,75)],axis=1)
            max_err[key] = np.maximum(max_err[key],delta.max(axis=0))
            semantic_err[key] = np.maximum(semantic_err[key],semantics.max(axis=0))
            row["pipelines"][key] = {"raw_context_max_error": float(delta.max()),
                "semantic_ego_max_error":float(np.abs(e-ne).max()),
                "semantic_neighbor_max_error":float(np.abs(n-nn).max()),
                "semantic_yaw_rate_max_error":float(np.abs(n[:,:,14]-nn[:,:,14]).max()),
                "semantic_neighbor_accel_max_error":float(np.abs(n[:,:,12]-nn[:,:,12]).max()),
                "slot_ids": b.neighbor_slot_ids[0],
                "raw_context_allclose":bool(np.allclose(c,nc,atol=.0002,rtol=.00002)),
                "semantic_allclose":bool(np.allclose(e,ne,atol=.0002,rtol=.00002) and np.allclose(n,nn,atol=.0002,rtol=.00002))}
        evidence["cases"][name] = row
    # Independent marker oracle: exercise all slots/times/channels, not self-parity.
    neighbor = np.arange(5*80*15,dtype=np.float32).reshape(5,80,15)
    actual = build_context_traj_from_standard_tracks(np.zeros((80,8),np.float32), neighbor)
    correct = neighbor.transpose(1,0,2).reshape(80,75)
    evidence["layout_oracle"] = {"production_matches_time_slot_channel":bool(np.array_equal(actual[:,8:],correct)),
        "production_matches_slot_time_reshape":bool(np.array_equal(actual[:,8:],neighbor.reshape(80,75))),
        "mismatched_cells":int(np.count_nonzero(actual[:,8:]!=correct)),
        "example_actual_t0_columns_8_23_38":actual[0,[8,23,38]].tolist(),
        "example_expected_t0_columns_8_23_38":correct[0,[0,15,30]].tolist()}
    # Physical stationary equal-speed neighbor is not closing, but legacy formula gives v_ego.
    from tools.stage5d_context_core import build_neighbor_features_15d
    v = build_neighbor_features_15d(rel_x=20,rel_y=0,rel_vx=0,rel_vy=0,ego_forward_speed=8,
                                  neighbor_speed=8,neighbor_accel=0,heading_rel=0,neighbor_yaw_rate=0)
    evidence["closing_oracle"] = {"same_speed_gap_constant":True,"expected_closing_mps":0.,
        "actual_closing_mps":float(v[8]),"actual_ttc_s":float(v[9]),"physical_ttc":"undefined/nonclosing"}
    # Legacy Raw33 lateral acceleration omits division by dt; old duration is a transition proxy.
    e = np.zeros((80,8),np.float32); e[:,3] = .5*np.arange(80)*.1; e[:,5] = 8
    f,names = aggregate_interaction_features(e,np.zeros((5,80,15),np.float32),.1)
    evidence["raw33_discretization_oracle"] = {"constant_lateral_accel_mps2":.5,
        "historical_rms_lateral_accel":float(f[names.index("rms_lateral_accel")]),
        "expected_rms_including_initial_zero":float(.5*np.sqrt(79/80))}
    evidence["per_column_max_error"] = {k:v.tolist() for k,v in max_err.items()}
    evidence["per_semantic_column_max_error"] = {k:v.tolist() for k,v in semantic_err.items()}
    write(OUT/"evidence/Input83_Golden_Fixture_Results_v1.json",evidence)
    return evidence


def varint(buf, pos):
    value = 0
    for shift in range(0,70,7):
        if pos >= len(buf): raise ValueError("Truncated protobuf varint")
        byte = buf[pos]; pos += 1; value |= (byte & 127) << shift
        if byte < 128: return value,pos
    raise ValueError("Oversized protobuf varint")


def wire_fields(buf):
    """Minimal bounded protobuf wire reader; rejects unsupported/truncated wire types."""
    p = 0
    while p < len(buf):
        tag,p = varint(buf,p); field,wire = tag>>3,tag&7
        if not field: raise ValueError("Invalid protobuf field zero")
        if wire == 0: value,p = varint(buf,p)
        elif wire in (1,5):
            end = p+(8 if wire==1 else 4); value=buf[p:end]; p=end
        elif wire == 2:
            size,p = varint(buf,p); end = p+size; value=buf[p:end]; p=end
        else: raise ValueError(f"Unsupported protobuf wire type {wire}")
        if p > len(buf): raise ValueError("Truncated protobuf field")
        yield field,wire,value


def raw_index(raw_root, expected_ids):
    """Scan TFRecord bytes and Scenario identity/time metadata, never construct training arrays.

    Scenario fields 5=id, 1=timestamps, 2=tracks follow the hash-bound official
    scenario.proto. No global segment ID or absolute time is invented.
    CRC bytes are bound in file SHA256; CRC validation is explicitly not claimed.
    """
    raw_root=Path(raw_root)
    if not raw_root.is_dir(): raise FileNotFoundError(f"Frozen Waymo raw directory missing: {raw_root}")
    raw_files=sorted(raw_root.glob("*tfrecord*"))
    if len(raw_files)!=51: raise ValueError(f"Frozen full51 requires 51 TFRecord files, got {len(raw_files)} in {raw_root}")
    inventory=[]; matched=set(); scenario_sources=defaultdict(set); time_ranges=Counter()
    index_hash=hashlib.sha256(); count=0
    index_path=OUT/"evidence/U_Raw_Record_Index_v1.csv"
    index_handle=index_path.open("w",newline="",encoding="utf-8")
    index_writer=csv.writer(index_handle,lineterminator="\n")
    index_writer.writerow(["raw_file","offset","payload_bytes","scenario_id","frame_count","time_start_s","time_end_s","timestamps_sha256","raw_segment_id"])
    for path in raw_files:
        h=hashlib.sha256(); n=0; begin=0; local_splits=Counter()
        with path.open("rb") as f:
            while True:
                header=f.read(12)
                if not header: break
                if len(header)!=12: raise ValueError(f"Truncated TFRecord header: {path}:{begin}")
                size=struct.unpack("<Q",header[:8])[0]
                if size > 64*1024*1024: raise ValueError(f"Unexpected TFRecord size {size}")
                payload=f.read(size); crc=f.read(4)
                if len(payload)!=size or len(crc)!=4: raise ValueError(f"Truncated record: {path}:{begin}")
                h.update(header);h.update(payload);h.update(crc)
                sid=None;times=[]
                for field,wire,value in wire_fields(payload):
                    if field==5 and wire==2: sid=value.decode("utf-8")
                    if field==1:
                        if wire==1: times.append(struct.unpack("<d",value)[0])
                        elif wire==2: times.extend(struct.unpack("<"+"d"*(len(value)//8),value))
                        else: raise ValueError("Unexpected timestamp wire type")
                if sid is None: raise ValueError(f"Scenario ID missing: {path}:{begin}")
                scenario_sources[sid].add(path.name)
                if sid in expected_ids: matched.add(sid)
                from tools.build_waymo_5neighbor_context_dataset import split_of_sid
                local_splits[split_of_sid(sid)]+=1
                time_ranges[str((len(times),round(min(times),6) if times else None,round(max(times),6) if times else None))]+=1
                index_hash.update(json.dumps([path.name,begin,size,sid,times],separators=(",",":")).encode()+b"\n")
                index_writer.writerow([path.name,begin,size,sid,len(times),times[0] if times else "",times[-1] if times else "",hashlib.sha256(np.asarray(times,dtype="<f8").tobytes()).hexdigest(),"NOT_AVAILABLE"])
                n+=1;count+=1;begin+=16+size
        inventory.append({"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":h.hexdigest(),"records":n,"scenario_split_counts":dict(local_splits)})
        print(f"raw metadata {len(inventory)}/51: {path.name} records={n}",flush=True)
    index_handle.close()
    if matched!=expected_ids: raise ValueError(f"Retained Scenario raw provenance missing: {len(expected_ids-matched)} identities")
    return {"files":inventory,"records":count,"unique_scenarios":len(scenario_sources),
            "expected_scenarios":len(expected_ids),"expected_scenarios_matched":len(matched),
            "scenario_in_multiple_raw_files":sum(len(v)>1 for v in scenario_sources.values()),
            "record_identity_time_index_sha256":index_hash.hexdigest(),"time_ranges":dict(time_ranges),
            "record_index_csv_sha256":sha(index_path),
            "global_raw_segment_id":"ABSENT_FROM_CURRENT_SCHEMA_AND_DERIVED_META",
            "crc_validation":"NOT_PERFORMED; complete file bytes SHA256 bound"}


def duplicate_raw_check(examples):
    records={r["scenario_id"]:r for r in csv.DictReader((OUT/"evidence/U_Raw_Record_Index_v1.csv").open())}
    results=[]
    for example in examples:
        arrays=[]; provenance=[]
        for side in ("first","second"):
            loc=example[side]; rec=records[loc["scenario_id"]]
            path=ROOT.parent/"training"/rec["raw_file"]
            with path.open("rb") as f:
                f.seek(int(rec["offset"])+12);payload=f.read(int(rec["payload_bytes"]))
            found=None
            for field,wire,value in wire_fields(payload):
                if field!=2 or wire!=2: continue
                tf=list(wire_fields(value)); actor=next((v for k,w,v in tf if k==1 and w==0),None)
                if str(actor)!=loc["actor_id"]: continue
                states=[]
                for k,w,v in tf:
                    if k!=3 or w!=2: continue
                    state={}
                    for sk,sw,sv in wire_fields(v):
                        if sw==1:state[sk]=struct.unpack('<d',sv)[0]
                        elif sw==5:state[sk]=struct.unpack('<f',sv)[0]
                        elif sw==0:state[sk]=sv
                    states.append([state.get(sk,0.) for sk in (2,3,9,10,8,11)])
                found=np.asarray(states,dtype=np.float64)[loc["start"]:loc["start"]+80]
                break
            if found is None or found.shape!=(80,6): raise ValueError(f"Missing raw track: {loc}")
            arrays.append(found);provenance.append({**loc,**rec,"world_track_sha256":hashlib.sha256(found.tobytes()).hexdigest()})
        delta=np.abs(arrays[0]-arrays[1])
        contexts=[np.load(ROOT/loc["shard"]/"context_traj.npy",mmap_mode="r")[loc["row"]]
                  for loc in (example["first"],example["second"])]
        results.append({"local_sequence_sha256":example["sha256"],"sources":provenance,
                        "raw_world_track_exact_equal":bool(np.array_equal(*arrays)),
                        "raw_world_valid_frame_counts":[int(np.sum(a[:,5]>.5)) for a in arrays],
                        "raw_world_path_lengths_m":[float(np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).sum()) for a in arrays],
                        "full_context_exact_equal":bool(np.array_equal(*contexts)),
                        "max_abs_world_field_difference":delta.max(axis=0).tolist(),
                        "origin_xy": [a[0,:2].tolist() for a in arrays],
                        "interpretation":"EXACT_WORLD_TRACK_DUPLICATE" if np.array_equal(*arrays) else "LOCAL_MOTION_DUPLICATE_ONLY; global segment identity remains unknown"})
    write(OUT/"evidence/U_Local_Duplicate_Raw_Check_v1.json",results)
    return results


def lineage():
    obj=read(DYNAMIC); groups=defaultdict(set); counts=Counter(); keys=Counter(); intervals=defaultdict(list)
    shards=[]; ids=set(); exact_hashes=defaultdict(set); hash_examples={}; duplicate_examples=[]
    for sp in obj["shard_paths"]:
        sd=Path(sp); meta=np.load(sd/"meta.npy",allow_pickle=True); split=np.load(sd/"split.npy",allow_pickle=True).astype(str)
        assert np.array_equal(meta["split"].astype(str),split)
        ego=np.load(sd/"ego_seq.npy",mmap_mode="r")
        neighbor=np.load(sd/"neighbor_seq.npy",mmap_mode="r");context=np.load(sd/"context_traj.npy",mmap_mode="r")
        for i,row in enumerate(meta):
            sid,actor,s,start,length=str(row["scenario_id"]),str(row["target_agent_id"]),str(row["split"]),int(row["start"]),int(row["window_len"])
            from tools.build_waymo_5neighbor_context_dataset import split_of_sid
            if s!=split_of_sid(sid): raise ValueError(f"Split differs from production hash rule: {sid} {s}")
            groups[sid].add(s);counts[s]+=1;ids.add(sid);keys[(sid,actor,start,length)]+=1
            intervals[(sid,actor)].append((start,start+length,s))
            eh=hashlib.sha256(np.asarray(ego[i]).tobytes()).hexdigest()
            exact_hashes[eh].add(s)
            loc={"scenario_id":sid,"actor_id":actor,"split":s,"shard":str(sd.relative_to(ROOT)),"row":i,"start":start}
            if eh in hash_examples and hash_examples[eh]["split"]!=s:
                duplicate_examples.append({"sha256":eh,"first":hash_examples[eh],"second":loc})
            else: hash_examples[eh]=loc
        layout=[]
        for i in range(min(2,len(meta))):
            n=np.asarray(neighbor[i]);c=np.asarray(context[i])
            layout.append({"row":i,"legacy_reshape_error":float(np.max(np.abs(c[:,8:]-n.reshape(80,75)))),
                           "canonical_layout_error":float(np.max(np.abs(c[:,8:]-n.transpose(1,0,2).reshape(80,75))))})
        sources={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in [sd/"meta.npy",sd/"split.npy",sd/"shard_summary.json",sd/"ego_seq.npy",sd/"neighbor_seq.npy",sd/"context_traj.npy"]}
        shards.append({"path":str(sd.relative_to(ROOT)),"rows":len(meta),"meta_fields":list(meta.dtype.names),"sources":sources,
                       "shapes":{"ego":list(ego.shape),"neighbor":list(neighbor.shape),"context":list(context.shape)},"real_data_layout_samples":layout})
        print(f"U metadata {len(shards)}/{len(obj['shard_paths'])}: rows={len(meta)}",flush=True)
    overlap=0
    for rows in intervals.values():
        rows.sort()
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if b[0]>=a[1]: break
                if a[2]!=b[2]: overlap+=1
    old=read(LEGACY); missing=[p for p in old["shard_paths"] if not (Path(p)/"meta.npy").exists()]
    result={"schema":"S3_P0_READ_ONLY_U_METADATA_AUDIT_V1","U_training_authorized":False,
            "candidate_dataset":"Dynamic v2 full51; NOT_SELECTED_OR_AUTHORIZED_FOR_RBR_V2",
            "manifest":{"path":str(DYNAMIC.relative_to(ROOT)),"sha256":sha(DYNAMIC)},
            "split_unit":"scenario_id MD5 first 8 hex / 0xFFFFFFFF; .8/.9 thresholds",
            "rows":sum(counts.values()),"scenario_count":len(groups),"split_counts":dict(counts),
            "scenario_counts_by_split":dict(Counter(next(iter(s)) for s in groups.values())),
            "scenario_cross_split":sum(len(v)>1 for v in groups.values()),
            "duplicate_identity_windows":sum(n-1 for n in keys.values() if n>1),
            "same_scenario_actor_overlapping_windows_cross_split":overlap,
            "exact_local_ego_sequence_hashes_cross_split":sum(len(s)>1 for s in exact_hashes.values()),
            "exact_local_ego_cross_split_examples":duplicate_examples,
            "local_hash_interpretation":"Equal normalized motion can be coincidental; not proof of shared raw segment. No fuzzy world-track deduplication performed.",
            "shards":shards,"legacy":{"manifest_sha256":sha(LEGACY),"declared_shards":len(old['shard_paths']),"missing_meta_shards":missing},
            "RAW_SEGMENT_DISJOINTNESS":"RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED"}
    result["raw_metadata"]=raw_index(ROOT.parent/"training",ids)
    result["local_duplicate_raw_check"]=duplicate_raw_check(duplicate_examples)
    write(OUT/"S3_P0_U_Data_Split_Manifest_v1.json",result)
    return {k:result[k] for k in ["rows","scenario_count","split_counts","scenario_cross_split","duplicate_identity_windows","same_scenario_actor_overlapping_windows_cross_split","exact_local_ego_sequence_hashes_cross_split"]}


def manifest():
    """Bind audit outputs and historical sources; never authorize future training."""
    protection=verify()
    gates=list(csv.DictReader((OUT/"S3_P0_Training_Entry_Gates_v1.csv").open()))
    fields=list(csv.DictReader((OUT/"S3_P0_Input83_Parity_Audit_v1.csv").open()))
    assert [r['gate'] for r in gates]==[f'G{i}' for i in range(1,13)]
    assert [int(r['index_zero_based']) for r in fields]==list(range(83))
    assert all(r['status'] in {'PASS','BLOCKED','INCONCLUSIVE','NOT_APPLICABLE_WITH_REASON'} for r in gates)
    required=['Training_Entry_Report','Waymo_NuPlan_Input83_Contract','U_Data_Lineage_And_Overlap_Audit',
              'RBR_V2_Training_Architecture_Proposal','Semantic_Target_Definitions','Representation_Roles_Audit',
              'Primary_Firewall_Check','Owner_Decision_Items']
    assert all((OUT/f'S3_P0_{name}_v1.md').exists() for name in required)
    u=read(OUT/'S3_P0_U_Data_Split_Manifest_v1.json')
    old_ledger_path=ROOT/'outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_full51_sha256_ledger.json'
    old_ledger=read(old_ledger_path); compared=0; mismatches=[]
    for f in u['raw_metadata']['files']:
        compared+=1
        if old_ledger['source_tfrecord_sha256'].get(Path(f['path']).name)!=f['sha256']:mismatches.append(f['path'])
    for shard in u['shards']:
        old=old_ledger['shard_artifact_sha256'][str(ROOT/shard['path'])]
        for name,record in shard['sources'].items():
            if name not in old:continue
            compared+=1
            if record['sha256']!=old[name]:mismatches.append(shard['path']+'/'+name)
    write(OUT/'evidence/Historical_Data_Hash_Comparison_v1.json',{
        'ledger_path':str(old_ledger_path.relative_to(ROOT)),'ledger_sha256':sha(old_ledger_path),
        'compared':compared,'mismatches':mismatches,'status':'PASS' if not mismatches else 'MISMATCH_REQUIRES_REVIEW'})
    assert not mismatches, 'Historical data hashes differ'
    source_paths=[Path(__file__),ROOT/'tests/test_s3_p0_training_entry_audit.py',DYNAMIC,LEGACY,old_ledger_path,
        ROOT/'tools/stage5d_context_core.py',ROOT/'tools/build_waymo_5neighbor_context_dataset.py',
        ROOT/'tools/build_waymo_dynamic_interaction_dataset_v2.py',ROOT/'tools/build_nuplan_5neighbor_context_dataset.py',
        ROOT/'tools/trajectory_context_utils.py',ROOT/'tools/lane_aware_assignment.py',ROOT/'tools/waymo_lane_utils.py',
        ROOT/'tools/train_context_behavior_embedding.py',ROOT/'tools/context_shard_dataset.py',
        ROOT/'tools/interaction_context_features.py',ROOT/'tools/stage7c1_run_nuplan_simulation.py',
        ROOT/'tools/stage7d_extract_neighbors_from_nuplan.py',ROOT/'tools/stage7e_embed_stage6_dataset.py',
        ROOT/'tools/stage6u_unified_abc_trainer.py',ROOT/'tools/s1_protocol_schema.py',
        ROOT/'tools/stage6l_prepare_context_representation_ablation.py',ROOT/'configs/stage6t_training_evaluation_protocol.json',
        ROOT/'configs/stage6u_unified_abc_trainer.json',ROOT/'RBR-64_博士研究总体方案_v2.3.md']
    oldroot=ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged'
    source_paths += [oldroot/p for p in ['interaction_feature_standardization.json','global_feature_standardization_report.json',
        'context_gru_stage5d_balanced_v2/training_config.json','context_gru_stage5d_balanced_v2/training_summary.json',
        'context_gru_stage5d_balanced_v2/best_model.pt']]
    roadmap=ROOT/'RBR-64 博士研究总体方案 v2.4.md'
    baseline=read(OUT/'evidence/Protected_Baseline_v1.json')
    target=OUT/'S3_P0_Manifest_v1.json'
    outputs=sorted(p for p in OUT.rglob('*') if p.is_file() and p!=target and p.name!='parity_stdout.log')
    value={'stage':'S3-P0','date':'2026-10-23','base_git_head':baseline['head'],'branch':baseline['branch'],
        'overall_status':'S3_P0_BLOCKED_INPUT_CONTRACT',
        'blocking_statuses':['S3_P0_BLOCKED_INPUT_CONTRACT','S3_P0_BLOCKED_U_DATA_INDEPENDENCE','S3_P0_BLOCKED_TRAINING_PROTOCOL'],
        'ready_for_training_owner_review':False,'RBR_TRAINING':'NOT_AUTHORIZED','PRIMARY_EVALUATION':'NOT_AUTHORIZED',
        'execution_counts':{k:0 for k in ['NEW_NUPLAN_SIMULATION','NEW_TSB_ROLLOUT','HLC_ROLLOUT','RBR_TRAINING',
            'RBR_CHECKPOINT_SELECTION','NEW_RBR_H_COMPARISON','NEW_BDD_EVALUATION','PRIMARY_EVALUATION','V_EXECUTION','C_EXECUTION']},
        'protected_verification':protection,'gate_counts':dict(Counter(r['status'] for r in gates)),
        'roadmap_v2_4':{'path':str(roadmap.relative_to(ROOT)),'sha256':sha(roadmap),
            'git_tracked':bool(git('ls-files','--',str(roadmap.relative_to(ROOT)))),'status':'ROADMAP_V2_4_REPOSITORY_SYNC_PENDING',
            'normative_status':'local integration draft; does not supersede approved S1R or B1 R2'},
        'sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        'output_sha256':{str(p.relative_to(OUT)):sha(p) for p in outputs},
        'self_hash_policy':'Manifest excludes its own bytes; final Git commit/tree binds it. No future dataset/model/config hashes fabricated.'}
    write(target,value)
    return {'status':value['overall_status'],'blocking_statuses':value['blocking_statuses'],
            'outputs':len(outputs),'gates':value['gate_counts'],'historical_data_hashes_compared':compared}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["snapshot","parity","lineage","verify","manifest"])
    a=p.parse_args(); result=globals()[a.action]()
    if a.action == "parity":
        result={"cases":len(result["cases"]),"layout_oracle":result["layout_oracle"],"closing_oracle":result["closing_oracle"]}
    print(json.dumps(result,ensure_ascii=False,default=str))


if __name__=="__main__":
    main()
