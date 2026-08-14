#!/usr/bin/env python3
"""Conditional Stage6S-v2 representation evaluation after mechanism PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage6k_run_longitudinal_dose_bdd import null_diagnostics  # noqa: E402
from tools.stage6l_prepare_context_representation_ablation import apply_scaler, ego_kinematic_features  # noqa: E402
from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import exact_median_bandwidth, rbf_kernel  # noqa: E402
from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder  # noqa: E402

AUTH = ROOT / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"
PROTOCOL = ROOT / "configs/stage6t_training_evaluation_protocol.json"
LEDGER = ROOT / "outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"
OLD_CKPT = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt"
SCALER = ROOT / "outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired/scalers/handcrafted_reference_scalers.npz"
PLANNERS = ["pdm_closed_interaction_short_headway_v2", "pdm_closed_interaction_long_headway_v2"]
REPS = ["old64", "A", "B", "C", "ego13", "C_neighbor_zero"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def schema_names(context_dir: Path) -> list[str]:
    schema = read_json(context_dir / "feature_schema.json")
    return [str(row["name"]) for row in schema["features"]]


def embed(model: torch.nn.Module, context: np.ndarray, device: torch.device) -> np.ndarray:
    values = []
    with torch.no_grad():
        for start in range(0, len(context), 128):
            batch = torch.from_numpy(np.asarray(context[start:start+128], dtype=np.float32).copy()).to(device)
            values.append(model(batch).detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(values)
    if result.shape != (160, 64) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid representation shape: {result.shape}")
    return result


def build_representations(context_dir: Path, ledger: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    context = np.asarray(np.load(context_dir / "context_traj.npy", mmap_mode="r"), dtype=np.float32)
    if context.shape != (160, 150, 83):
        raise RuntimeError(f"unexpected confirmation context shape: {context.shape}")
    groups = feature_group_indices(schema_names(context_dir))
    old = ContextFlattenGRUEncoder(input_dim=83, hidden_dim=128, embedding_dim=64)
    old.load_state_dict(torch.load(OLD_CKPT, map_location="cpu", weights_only=False)["model"])
    old = old.eval().to(device)
    result = {"old64": embed(old, context, device)}
    candidate_models = {}
    for candidate in "ABC":
        row = next(row for row in ledger["rows"] if row["candidate"] == candidate and int(row["seed"]) == 3407)
        checkpoint = Path(row["best_checkpoint_path"])
        if sha256(checkpoint) != row["best_checkpoint_sha256"]:
            raise RuntimeError(f"locked checkpoint changed: {candidate}")
        model = UnifiedABCModel(candidate, groups)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
        candidate_models[candidate] = model.eval().to(device)
        result[candidate] = embed(candidate_models[candidate], context, device)
    masked = context.copy(); masked[:,:,8:83] = 0.0
    result["C_neighbor_zero"] = embed(candidate_models["C"], masked, device)
    ego = np.asarray(np.load(context_dir / "ego_seq.npy", mmap_mode="r"), dtype=np.float32)
    mask = np.asarray(np.load(context_dir / "ego_seq_mask.npy", mmap_mode="r"), dtype=bool)
    scaler = np.load(SCALER)
    result["ego13"] = np.asarray(apply_scaler(ego_kinematic_features(ego, mask), scaler["ego_median"], scaler["ego_scale"]), dtype=np.float64)
    if result["ego13"].shape != (160,13):
        raise RuntimeError("invalid ego13 confirmation representation")
    return result


def signed_null(contrast: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(contrast)
    samples = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 2000):
        stop = min(start+2000, repetitions)
        signs = rng.integers(0,2,size=(stop-start,n),dtype=np.int8).astype(np.float64)*2-1
        samples[start:stop] = np.einsum("bi,ij,bj->b", signs, contrast, signs, optimize=True)/(n*n)
    return samples


def analyze(values: np.ndarray, pairs: np.ndarray, repetitions: int, seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    a, b = values[pairs[:,0]], values[pairs[:,1]]
    pooled = np.vstack([a,b])
    bandwidth = exact_median_bandwidth(pooled)
    kernel = rbf_kernel(pooled, bandwidth)
    n = len(a)
    contrast = kernel[:n,:n]+kernel[n:,n:]-kernel[:n,n:]-kernel[n:,:n]
    observed = float(contrast.mean())
    samples = signed_null(contrast, repetitions, seed)
    exceedance = int(np.sum(samples >= observed))
    row = {"n_pairs":n,"mmd2":observed,"bandwidth":bandwidth,"permutations":repetitions,"exceedance_count":exceedance,
           "raw_p":float((exceedance+1)/(repetitions+1)),**null_diagnostics(observed,samples)}
    return row, samples, contrast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism_summary", type=Path, required=True)
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"Refusing to overwrite {out}")
    if sha256(AUTH) != AUTH_SHA: raise RuntimeError("blind authorization changed")
    auth = read_json(AUTH); mechanism = read_json(args.mechanism_summary)
    if mechanism.get("status") != "CONFIRMATION_MECHANISM_GATE_PASS_REPRESENTATION_EVALUATION_AUTHORIZED" or mechanism.get("mechanism_gate_passed") is not True:
        raise RuntimeError("mechanism gate did not authorize representation evaluation")
    if mechanism.get("embedding_or_bdd_read") is not False: raise RuntimeError("mechanism summary blind state invalid")
    protocol = read_json(PROTOCOL); scorecard = protocol["stage6s_v2_interaction_scorecard"]
    ledger = read_json(LEDGER)
    meta = pd.read_csv(args.context_dir / "metadata.csv").sort_values("global_row").reset_index(drop=True)
    pairs=[]; logs=[]
    for scenario, frame in meta.groupby("scenario_index", sort=True):
        short=frame[frame.planner_name==PLANNERS[0]]; long=frame[frame.planner_name==PLANNERS[1]]
        if len(short)!=1 or len(long)!=1: raise RuntimeError(f"incomplete confirmation pair {scenario}")
        pairs.append((int(short.iloc[0].global_row),int(long.iloc[0].global_row))); logs.append(str(short.iloc[0].log_name))
    pairs=np.asarray(pairs,dtype=np.int64)
    if pairs.shape!=(80,2): raise RuntimeError(f"expected 80 pairs, got {pairs.shape}")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    representations=build_representations(args.context_dir.resolve(),ledger,device)
    out.mkdir(parents=True); rep_dir=out/"representations"; rep_dir.mkdir()
    rows=[]; nulls={}; contrasts={}
    null_cfg=scorecard["paired_null"]
    for representation in REPS:
        path=rep_dir/f"{representation}.npy"; np.save(path,representations[representation].astype(np.float32))
        row,samples,contrast=analyze(representations[representation],pairs,int(null_cfg["permutations"]),int(null_cfg["seed"]))
        row={"representation":representation,**row}
        row["candidate_detection_gate_pass"]=bool(row["raw_p"]<.05 and row["null_standardized_z_bdd"]>1.645)
        rows.append(row); nulls[representation]=samples.astype(np.float32); contrasts[representation]=contrast
    results=pd.DataFrame(rows); results.to_csv(out/"stage6s_v2_confirmation_representation_results.csv",index=False)
    np.savez_compressed(out/"stage6s_v2_confirmation_null_samples.npz",**nulls)
    by_rep=results.set_index("representation")
    delta=float(by_rep.loc["C","null_standardized_z_bdd"]-by_rep.loc["C_neighbor_zero","null_standardized_z_bdd"])
    log_values=np.asarray(logs,dtype=str); unique=np.unique(log_values); rng=np.random.default_rng(int(scorecard["c_context_increment"]["bootstrap_seed"]))
    bootstrap=np.empty(10_000,dtype=float)
    c_mean=float(np.mean(nulls["C"])); c_sd=float(np.std(nulls["C"],ddof=1)); z_mean=float(np.mean(nulls["C_neighbor_zero"])); z_sd=float(np.std(nulls["C_neighbor_zero"],ddof=1))
    for repetition in range(10_000):
        selected_logs=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([np.flatnonzero(log_values==log) for log in selected_logs])
        c_obs=float(np.mean(contrasts["C"][np.ix_(idx,idx)])); zero_obs=float(np.mean(contrasts["C_neighbor_zero"][np.ix_(idx,idx)]))
        bootstrap[repetition]=(c_obs-c_mean)/c_sd-(zero_obs-z_mean)/z_sd
    lower=float(np.quantile(bootstrap,.025)); upper=float(np.quantile(bootstrap,.975))
    increment={"comparison":"C_full_minus_C_neighbor_zero","delta_z_bdd":delta,"log_cluster_bootstrap95_lower":lower,
               "log_cluster_bootstrap95_upper":upper,"bootstrap_repetitions":10000,"bootstrap_seed":int(scorecard["c_context_increment"]["bootstrap_seed"]),
               "incremental_interaction_information_pass":bool(delta>0 and lower>0),"raw_mmd2_difference_computed":False}
    write_json(out/"stage6s_v2_c_context_increment.json",increment)
    files=["stage6s_v2_confirmation_representation_results.csv","stage6s_v2_confirmation_null_samples.npz","stage6s_v2_c_context_increment.json"]
    manifest={"schema_version":"stage6v_stage6s_v2_representation_v1","status":"FROZEN_STAGE6S_V2_REPRESENTATION_EVALUATION_COMPLETE",
              "immutability_statement":auth["immutability_statement"],"mechanism_summary_sha256":sha256(args.mechanism_summary),"primary_seed":3407,
              "representations":REPS,"common_swap_vectors":True,"permutations":int(null_cfg["permutations"]),"seed":int(null_cfg["seed"]),
              "cross_representation_raw_mmd2_comparison_performed":False,"training_or_protocol_modified":False,
              "result_files":{name:sha256(out/name) for name in files},"representation_sha256":{name:sha256(rep_dir/f"{name}.npy") for name in REPS}}
    write_json(out/"stage6s_v2_representation_result_manifest.json",manifest)
    lines=["# Stage6S-v2 confirmation representation 盲测报告","",f"- 状态：`{manifest['status']}`",
           "- 各 representation 独立 bandwidth/null；共同 pair-swap 随机流；不跨 representation 比 raw MMD²。","","## Null-standardized 结果","",results.to_markdown(index=False),"","## C 上下文增量","",pd.DataFrame([increment]).to_markdown(index=False)]
    (out/"stage6s_v2_representation_report_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"manifest_sha256":sha256(out/"stage6s_v2_representation_result_manifest.json"),"results":rows,"c_context_increment":increment},indent=2,ensure_ascii=False))


if __name__ == "__main__": main()
