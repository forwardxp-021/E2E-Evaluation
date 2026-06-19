#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.stage7f_idm_diagnostic_common import require_file

EGO = {"x":0,"y":1,"vx":2,"vy":3,"heading":4,"speed":5,"accel":6,"yaw_rate":7}
FRONT = {"valid":0,"distance":5,"ttc":9,"thw":10}


def planner_column(meta):
    for c in ["planner_name", "policy_style", "planner", "planner_id", "nuplan_planner_config"]:
        if c in meta.columns: return c
    raise ValueError(f"metadata.csv lacks planner axis column; columns={list(meta.columns)}")

def scenario_column(meta):
    for c in ["scenario_token", "scenario_id", "scenario_index", "log_scenario_id"]:
        if c in meta.columns: return c
    raise ValueError(f"metadata.csv lacks scenario axis column; columns={list(meta.columns)}")

def load_embedding(embedding_dir):
    p = Path(embedding_dir) / "embedding.npy"
    if p.exists(): return np.load(p, mmap_mode="r")
    manifest = json.loads(require_file(Path(embedding_dir)/"embedding_manifest.json", "embedding_manifest.json").read_text())
    paths = manifest.get("embedding_shard_paths", [])
    if len(paths) != 1: raise ValueError(f"Expected embedding.npy or one embedding shard, got {paths}")
    ep = Path(paths[0]); ep = ep if ep.is_absolute() else Path(embedding_dir)/ep
    return np.load(require_file(ep, "embedding shard"), mmap_mode="r")

def load_metadata(embedding_dir, context_dir):
    for p in [Path(embedding_dir)/"metadata.csv", Path(context_dir)/"metadata.csv", Path(context_dir)/"shards"/"shard_000000"/"metadata.csv", Path(context_dir)/"shards"/"shard_000"/"metadata.csv"]:
        if p.exists(): return pd.read_csv(p), p
    raise FileNotFoundError(f"Missing metadata.csv in embedding_dir={embedding_dir} or context_dataset_dir={context_dir}")

def load_context_arrays(context_dir):
    c = Path(context_dir)
    candidates = [c, c/"shards"/"shard_000000", c/"shards"/"shard_000"]
    for d in candidates:
        ego = d/"ego_seq.npy"
        if ego.exists():
            nei = d/"neighbor_seq.npy"
            feat = d/"interaction_feat_style.npy"
            return np.load(ego, mmap_mode="r"), (np.load(nei, mmap_mode="r") if nei.exists() else None), (np.load(feat, mmap_mode="r") if feat.exists() else None), d
    raise FileNotFoundError(f"Missing ego_seq.npy under {context_dir}")


def _json_loads_maybe(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _planner_name_from_profile(profile):
    for key in ["planner_name", "name", "planner", "policy_style", "planner_id"]:
        val = profile.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _parameters_from_profiles_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data if isinstance(data, list) else data.get("planner_profiles", [])
    out = {}
    if not isinstance(profiles, list):
        return out
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        name = _planner_name_from_profile(profile)
        params = profile.get("parameters_json")
        params = _json_loads_maybe(params) or profile.get("parameters")
        if name and isinstance(params, dict):
            out[name] = params
    return out


def _parameters_from_metadata(meta, planner_col):
    if "parameters_json" not in meta.columns:
        return {}
    out = {}
    for _, row in meta.iterrows():
        name = str(row.get(planner_col, "")).strip()
        params = _json_loads_maybe(row.get("parameters_json"))
        if name and params and name not in out:
            out[name] = params
    return out


def _parameters_from_csv(path):
    frame = pd.read_csv(path)
    pcol = planner_column(frame)
    return _parameters_from_metadata(frame, pcol)


def _candidate_profile_paths(embedding_dir, context_dir, stage7f_dir):
    roots = []
    for raw in [embedding_dir, context_dir, stage7f_dir]:
        if raw:
            p = Path(raw)
            # Python 3.9 pathlib parents does not support slice access;
            # materialize before limiting upward profile search roots.
            roots.extend([p, *list(p.parents)[:3]])
    seen = set()
    for root in roots:
        for name in ["simulation_schema.json", "simulated_planner_metadata.csv"]:
            path = root / name
            if path not in seen:
                seen.add(path)
                yield path


def load_planner_parameters(meta, planner_col, embedding_dir, context_dir, stage7f_dir):
    sources = []
    params = _parameters_from_metadata(meta, planner_col)
    if params:
        sources.append("metadata.csv:parameters_json")
    for path in _candidate_profile_paths(embedding_dir, context_dir, stage7f_dir):
        if not path.exists():
            continue
        try:
            found = _parameters_from_profiles_file(path) if path.suffix == ".json" else _parameters_from_csv(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sources.append(f"warning: could not read {path}: {exc}")
            continue
        for name, values in found.items():
            params.setdefault(name, values)
        if found:
            sources.append(str(path))
    return params, sources


def planner_parameter_markdown(planner_a, planner_b, planner_parameters, sources):
    a = planner_parameters.get(planner_a)
    b = planner_parameters.get(planner_b)
    if not a and not b:
        return ""
    lines = ["## Planner parameter definitions", "", "Parameters are read from `parameters_json` in row metadata or planner profiles; no IDM defaults are hard-coded in this report.", ""]
    if sources:
        lines += ["Sources:", ""] + [f"* `{src}`" for src in sources] + [""]
    for planner, vals in [(planner_a, a), (planner_b, b)]:
        lines += [f"### {planner}", ""]
        if not vals:
            lines += ["* parameters_json unavailable", ""]
            continue
        for key in sorted(vals):
            lines.append(f"* {key} = `{vals[key]}`")
        lines.append("")
    if a and b:
        common = sorted(set(a) & set(b))
        diffs = []
        for key in common:
            try:
                diffs.append((key, float(a[key]) - float(b[key])))
            except (TypeError, ValueError):
                continue
        if diffs:
            lines += [f"### {planner_a} - {planner_b} numeric parameter differences", ""]
            for key, diff in diffs:
                lines.append(f"* {key}: `{diff:+.6g}`")
            lines.append("")
    return "\n".join(lines)

def finite(x):
    a = np.asarray(x, dtype=float); return a[np.isfinite(a)]

def mean(x):
    v=finite(x); return float(np.mean(v)) if v.size else np.nan

def rms(x):
    v=finite(x); return float(np.sqrt(np.mean(v*v))) if v.size else np.nan

def minv(x):
    v=finite(x); return float(np.min(v)) if v.size else np.nan

def maxv(x):
    v=finite(x); return float(np.max(v)) if v.size else np.nan

def row_metrics(i, emb, ego, nei, meta):
    e = np.asarray(ego[i], dtype=float)
    speed=e[:,EGO["speed"]]; accel=e[:,EGO["accel"]]; yaw=e[:,EGO["yaw_rate"]]
    jerk=np.diff(accel)/0.1 if accel.size>1 else np.asarray([])
    m = {"mean_speed":mean(speed), "max_speed":maxv(speed), "rms_accel":rms(accel), "max_abs_accel":maxv(np.abs(accel)), "rms_jerk":rms(jerk), "max_abs_jerk":maxv(np.abs(jerk)), "mean_abs_yaw_rate":mean(np.abs(yaw))}
    if nei is not None and nei.ndim >= 4 and nei.shape[1] > 0:
        front = np.asarray(nei[i,0], dtype=float)
        valid = front[:,FRONT["valid"]] > 0.5 if front.shape[1] > FRONT["valid"] else np.zeros(front.shape[0], dtype=bool)
        dist = np.where(valid, front[:,FRONT["distance"]], np.nan) if front.shape[1] > FRONT["distance"] else np.full(front.shape[0], np.nan)
        thw = np.where(valid, front[:,FRONT["thw"]], np.nan) if front.shape[1] > FRONT["thw"] else np.full(front.shape[0], np.nan)
        m.update({"min_thw":minv(thw), "mean_thw":mean(thw), "min_front_distance":minv(dist), "mean_front_distance":mean(dist), "front_valid_ratio":float(np.mean(valid)) if valid.size else np.nan})
    else:
        m.update({"min_thw":np.nan, "mean_thw":np.nan, "min_front_distance":np.nan, "mean_front_distance":np.nan, "front_valid_ratio":np.nan})
    for c in ["lane_change_count_proxy", "fallback_used", "laneaware_available"]:
        if c in meta.index: m[c] = meta[c]
    return m

def align_pairs(meta, planner_a, planner_b):
    pcol, scol = planner_column(meta), scenario_column(meta)
    tmp = meta.copy(); tmp["_row"] = np.arange(len(tmp)); tmp[pcol]=tmp[pcol].astype(str); tmp[scol]=tmp[scol].astype(str)
    dup = tmp[tmp.duplicated([scol,pcol], keep=False)]
    if len(dup): raise ValueError(f"Duplicate scenario-planner pairs exist: {dup[[scol,pcol]].head().to_dict('records')}")
    a = tmp[tmp[pcol] == planner_a].set_index(scol); b = tmp[tmp[pcol] == planner_b].set_index(scol)
    common = sorted(set(a.index) & set(b.index))
    missing = sorted((set(a.index) | set(b.index)) - set(common))
    if not common: raise ValueError(f"No paired scenarios containing both planner_a={planner_a} and planner_b={planner_b}; missing_examples={missing[:10]}")
    return [(s, int(a.loc[s,"_row"]), int(b.loc[s,"_row"])) for s in common], {"scenario_column":scol,"planner_column":pcol,"paired_scenarios":len(common),"unpaired_scenarios":len(missing),"unpaired_examples":missing[:20]}

def summarize(df):
    out={}
    for c in [c for c in df.columns if c.startswith("delta_") or c in ["embedding_l2_distance","embedding_cosine_distance"]]:
        v=pd.to_numeric(df[c], errors="coerce").to_numpy(float); ok=np.isfinite(v)
        out[c]={k: (float(val) if np.isfinite(val) else None) for k,val in {"mean":np.nanmean(v) if ok.any() else np.nan,"std":np.nanstd(v) if ok.any() else np.nan,"median":np.nanmedian(v) if ok.any() else np.nan,"p25":np.nanpercentile(v,25) if ok.any() else np.nan,"p75":np.nanpercentile(v,75) if ok.any() else np.nan}.items()}
    return out

def conclusion(df):
    fv = pd.to_numeric(df.get("front_valid_ratio_A", pd.Series(dtype=float)), errors="coerce")
    if len(fv) and np.nanmean(fv) < 0.2: return "insufficient_task_exposure"
    speed = abs(np.nanmean(pd.to_numeric(df["delta_mean_speed"], errors="coerce")))
    accel = abs(np.nanmean(pd.to_numeric(df["delta_rms_accel"], errors="coerce")))
    thw = abs(np.nanmean(pd.to_numeric(df["delta_mean_thw"], errors="coerce")))
    dist = abs(np.nanmean(pd.to_numeric(df["delta_mean_front_distance"], errors="coerce")))
    emb = np.nanmean(pd.to_numeric(df["embedding_l2_distance"], errors="coerce"))
    if all(x < t for x,t in [(speed,0.2),(accel,0.1),(thw,0.2),(dist,1.0),(emb,0.5)]): return "realized_difference_small"
    if speed > 0.5 or accel > 0.2 or thw > 0.5 or dist > 2.0: return "realized_difference_present_but_embedding_bdd_small"
    return "inconclusive"

def plots(out, df):
    vals = {c: abs(np.nanmean(pd.to_numeric(df[c], errors="coerce"))) for c in df.columns if c.startswith("delta_")}
    top = sorted(vals.items(), key=lambda x: -np.nan_to_num(x[1], nan=-1))[:12]
    plt.figure(figsize=(10,4)); plt.bar([k.replace("delta_","") for k,_ in top], [v for _,v in top]); plt.xticks(rotation=60, ha="right"); plt.tight_layout(); plt.savefig(out/"paired_delta_bar.png", dpi=150); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(pd.to_numeric(df["embedding_l2_distance"], errors="coerce").dropna(), bins=min(10,max(1,len(df)))); plt.xlabel("embedding L2 distance"); plt.tight_layout(); plt.savefig(out/"embedding_pair_distance_hist.png", dpi=150); plt.close()

def run(args):
    out=Path(args.output_dir)
    if out.exists():
        if not args.overwrite: raise FileExistsError(f"output_dir exists: {out}. Use --overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meta, meta_path = load_metadata(args.embedding_dir, args.context_dataset_dir)
    emb = load_embedding(args.embedding_dir); ego, nei, feat, arr_dir = load_context_arrays(args.context_dataset_dir)
    if len(meta) != emb.shape[0] or ego.shape[0] != emb.shape[0]: raise ValueError(f"row count mismatch: metadata={len(meta)} embeddings={emb.shape[0]} ego={ego.shape[0]}")
    pairs, align = align_pairs(meta, args.planner_a, args.planner_b)
    planner_params, planner_param_sources = load_planner_parameters(meta, align["planner_column"], args.embedding_dir, args.context_dataset_dir, args.stage7f_dir)
    rows=[]
    for scen, ia, ib in pairs:
        ma=row_metrics(ia, emb, ego, nei, meta.iloc[ia]); mb=row_metrics(ib, emb, ego, nei, meta.iloc[ib])
        ea=np.asarray(emb[ia], float); eb=np.asarray(emb[ib], float); denom=np.linalg.norm(ea)*np.linalg.norm(eb)
        r={"scenario":scen,"row_A":ia,"row_B":ib,"embedding_l2_distance":float(np.linalg.norm(ea-eb)),"embedding_cosine_distance":float(1-np.dot(ea,eb)/denom) if denom>1e-12 else np.nan}
        keys=sorted(set(ma)|set(mb))
        for k in keys:
            r[f"{k}_A"]=ma.get(k,np.nan); r[f"{k}_B"]=mb.get(k,np.nan)
            if isinstance(ma.get(k,np.nan),(int,float,np.floating,np.integer)) and isinstance(mb.get(k,np.nan),(int,float,np.floating,np.integer)):
                r[f"delta_{k}"]=float(ma.get(k,np.nan))-float(mb.get(k,np.nan))
        rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(out/"paired_delta_by_scenario.csv", index=False)
    label=conclusion(df)
    summary={"planner_a":args.planner_a,"planner_b":args.planner_b,"num_paired_scenarios":len(df),"alignment":align,"metadata_path":str(meta_path),"context_array_dir":str(arr_dir),"planner_parameter_sources":planner_param_sources,"planner_parameters_available":[p for p in [args.planner_a,args.planner_b] if p in planner_params],"delta_summary":summarize(df),"aggressive_gt_conservative_speed_count":int((df["delta_mean_speed"]>0).sum()),"aggressive_gt_conservative_speed_fraction":float((df["delta_mean_speed"]>0).mean()),"aggressive_gt_conservative_accel_count":int((df["delta_rms_accel"]>0).sum()),"aggressive_gt_conservative_accel_fraction":float((df["delta_rms_accel"]>0).mean()),"aggressive_smaller_thw_count":int((df["delta_mean_thw"]<0).sum()),"aggressive_smaller_thw_fraction":float((df["delta_mean_thw"]<0).mean()),"aggressive_smaller_front_distance_count":int((df["delta_mean_front_distance"]<0).sum()),"aggressive_smaller_front_distance_fraction":float((df["delta_mean_front_distance"]<0).mean()),"conclusion_label":label}
    (out/"paired_delta_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plots(out, df)
    parameter_section = planner_parameter_markdown(args.planner_a, args.planner_b, planner_params, planner_param_sources)
    lines=["# Stage7F same-scenario paired delta report","",f"* A = `{args.planner_a}`",f"* B = `{args.planner_b}`","* Delta convention: A - B.",f"* paired scenarios: `{len(df)}`",f"* conclusion_label: `{label}`"]
    if parameter_section:
        lines += ["", parameter_section]
    lines += ["","## Summary","",f"* aggressive > conservative speed: {summary['aggressive_gt_conservative_speed_count']} / {len(df)}",f"* aggressive > conservative rms accel: {summary['aggressive_gt_conservative_accel_count']} / {len(df)}",f"* aggressive smaller mean THW: {summary['aggressive_smaller_thw_count']} / {len(df)}",f"* aggressive smaller mean front distance: {summary['aggressive_smaller_front_distance_count']} / {len(df)}"]
    (out/"paired_delta_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def parse_args():
    p=argparse.ArgumentParser(description="Stage7F same-scenario paired kinematic and embedding delta for any two planners.")
    p.add_argument("--embedding_dir", required=True); p.add_argument("--context_dataset_dir", required=True); p.add_argument("--stage7f_dir", required=True)
    p.add_argument("--planner_a", required=True); p.add_argument("--planner_b", required=True); p.add_argument("--output_dir", required=True); p.add_argument("--overwrite", action="store_true")
    return p.parse_args()
if __name__ == "__main__": run(parse_args())
