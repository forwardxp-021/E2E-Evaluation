#!/usr/bin/env python3
"""Stage6P locked unpaired-release blind evaluation for old64/A/B/C/ego13."""

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
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage6d_unpaired_version_bdd as stage6d  # noqa: E402
from tools.stage6f_unpaired_power_curve import validate_power_config  # noqa: E402
from tools.stage6p_run_representation_unpaired_release import (  # noqa: E402
    IDENTITY, build_trials, higher_quantile, median_bandwidth, rbf_kernel, wilson_interval,
)
from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402

AUTH = ROOT / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"
PROTOCOL = ROOT / "configs/stage6t_training_evaluation_protocol.json"
LEDGER = ROOT / "outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"
STAGE6P_CONFIG = ROOT / "configs/stage6p_representation_unpaired_release.json"
POWER_CONFIG = ROOT / "configs/stage6h_nuplan_power_curve_800.json"
POOL = ROOT / "outputs/stage6h_expanded_800_embedding_pool_v1"
ASSIGNMENTS = ROOT / "outputs/stage6h_nuplan_power_curve_800_v1/power_curve_log_assignments.csv"
CONTEXT_EXISTING = ROOT / "outputs/stage7_m6_5_locked_confirmation_context_v1"
CONTEXT_EXPANDED = ROOT / "outputs/stage6h_expanded_490_context_v1"
OLD_STAGE6P = ROOT / "outputs/stage6p_representation_unpaired_release_v1"
DEFAULT_OUT = ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1"

PRIMARY_REPS = ["old64", "A_3407", "B_3407", "C_3407", "ego13"]
CANDIDATE_REPS = [f"{candidate}_{seed}" for candidate in "ABC" for seed in (3407, 3408, 3409)]
ALL_REPS = ["old64", "ego13", *CANDIDATE_REPS]
METHODS = ["raw_marginal", "context_balanced"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(AUTH) != AUTH_SHA:
        raise RuntimeError("Blind authorization SHA changed")
    auth = read_json(AUTH)
    ledger = read_json(LEDGER)
    pconfig = read_json(STAGE6P_CONFIG)
    if auth["status"] != "AUTHORIZED_STAGE6_ONE_TIME_BLIND_EVALUATION":
        raise RuntimeError("Blind evaluation not authorized")
    if ledger["status"] != "LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK":
        raise RuntimeError("Checkpoints are not locked 9/9")
    if sha256(ASSIGNMENTS) != pconfig["source_contract"]["log_assignments_sha256"]:
        raise RuntimeError("Frozen Stage6P assignments changed")
    if sha256(POOL / "metadata.csv") != pconfig["source_contract"]["pool_metadata_sha256"]:
        raise RuntimeError("Frozen Stage6P pool changed")
    return auth, ledger, pconfig


def schema_names() -> list[str]:
    schema = read_json(CONTEXT_EXISTING / "feature_schema.json")
    return [str(row["name"]) for row in schema["features"]]


def load_models(ledger: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    groups = feature_group_indices(schema_names())
    result: dict[str, torch.nn.Module] = {}
    for row in ledger["rows"]:
        checkpoint = Path(row["best_checkpoint_path"])
        if sha256(checkpoint) != row["best_checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed: {checkpoint}")
        candidate = str(row["candidate"])
        name = f"{candidate}_{row['seed']}"
        model = UnifiedABCModel(candidate, groups)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        result[name] = model.eval().to(device)
    return result


def embed_context(models: dict[str, torch.nn.Module], context: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in models}
    with torch.no_grad():
        for start in range(0, len(context), 128):
            batch = torch.from_numpy(np.asarray(context[start : start + 128], dtype=np.float32).copy()).to(device)
            for name, model in models.items():
                chunks[name].append(model(batch).detach().cpu().numpy().astype(np.float32))
    return {name: np.concatenate(values) for name, values in chunks.items()}


def build_representations(pool: pd.DataFrame, ledger: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    arrays = {
        "old64": np.asarray(np.load(OLD_STAGE6P / "representations/full64.npy"), dtype=np.float32),
        "ego13": np.asarray(np.load(OLD_STAGE6P / "representations/ego13.npy"), dtype=np.float32),
    }
    models = load_models(ledger, device)
    candidates = {name: np.empty((1600, 64), dtype=np.float32) for name in models}
    for source_pool, directory in (("existing_310", CONTEXT_EXISTING), ("stage6g_490", CONTEXT_EXPANDED)):
        positions = np.flatnonzero(pool["source_pool"].astype(str).to_numpy() == source_pool)
        source_rows = pool.iloc[positions]["source_global_row"].astype(int).to_numpy()
        source_meta = pd.read_csv(directory / "metadata.csv").sort_values("global_row").reset_index(drop=True)
        selected = source_meta.iloc[source_rows].reset_index(drop=True)
        expected = pool.iloc[positions].reset_index(drop=True)
        for column in ("scenario_token", "planner_name", "log_name"):
            if selected[column].astype(str).tolist() != expected[column].astype(str).tolist():
                raise RuntimeError(f"Stage6P row alignment failed: {source_pool}/{column}")
        context = np.asarray(np.load(directory / "context_traj.npy", mmap_mode="r")[source_rows], dtype=np.float32)
        if context.shape[1:] != (150, 83):
            raise RuntimeError(f"Unexpected context shape: {context.shape}")
        embedded = embed_context(models, context, device)
        for name, values in embedded.items():
            candidates[name][positions] = values
    arrays.update(candidates)
    for name, value in arrays.items():
        expected_dim = 13 if name == "ego13" else 64
        if value.shape != (1600, expected_dim) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid Stage6P representation {name}: {value.shape}")
    return arrays


def row_indices(pool: pd.DataFrame, planner: str, logs: list[str]) -> np.ndarray:
    mask = (pool.planner_name.astype(str).to_numpy() == str(planner)) & pool.log_name.astype(str).isin(logs).to_numpy()
    return np.flatnonzero(mask)


def prepare_trials(pool: pd.DataFrame, assignments: pd.DataFrame, power: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for trial in build_trials(assignments):
        ia = row_indices(pool, str(trial["planner_A"]), trial["logs_A"])
        ib = row_indices(pool, str(trial["planner_B"]), trial["logs_B"])
        frame_a = pool.iloc[ia].copy(); frame_a["_release_group"] = "A"
        frame_b = pool.iloc[ib].copy(); frame_b["_release_group"] = "B"
        frame = pd.concat([frame_a, frame_b], ignore_index=True)
        frame, _ = stage6d.coarsen_covariates(frame, power["stage6d_design"])
        standardized = stage6d.build_standardization(frame, power["stage6d_design"])
        indices = np.concatenate([ia, ib]).astype(np.int64)
        raw_weights = np.concatenate([
            np.full(len(ia), 1.0 / len(ia)), -np.full(len(ib), 1.0 / len(ib))
        ])
        balanced_weights = np.asarray(standardized["weights"], dtype=np.float64)
        balanced_weights[len(ia):] *= -1.0
        result.append({
            **{column: trial[column] for column in IDENTITY},
            "indices": indices,
            "raw_weights": raw_weights,
            "balanced_weights": balanced_weights,
            "balanced_valid": bool(standardized["passed"]),
            "support_A": float(standardized["group_A"]["support_fraction"]),
            "support_B": float(standardized["group_B"]["support_fraction"]),
        })
    return result


def quadratic(kernel: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> float:
    selected = kernel[np.ix_(indices, indices)]
    value = float(weights @ selected @ weights)
    return max(0.0, value)


def compute_representation(name: str, values: np.ndarray, bandwidth: float, trials: list[dict[str, Any]], primary: bool) -> pd.DataFrame:
    kernel = rbf_kernel(values, bandwidth)
    rows: list[dict[str, Any]] = []
    for trial in trials:
        if not primary and int(trial["target_scenarios_per_release"]) != 400:
            continue
        common = {column: trial[column] for column in IDENTITY}
        rows.append({**common, "representation": name, "method": "raw_marginal", "valid": True,
                     "statistic": quadratic(kernel, trial["indices"], trial["raw_weights"])})
        rows.append({**common, "representation": name, "method": "context_balanced", "valid": trial["balanced_valid"],
                     "statistic": quadratic(kernel, trial["indices"], trial["balanced_weights"]) if trial["balanced_valid"] else math.nan})
    return pd.DataFrame(rows)


def calibrate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    evaluated = []
    for (representation, size, method), group in raw.groupby(["representation", "target_scenarios_per_release", "method"], sort=False):
        calibration = group[(group.family == "AA_CALIBRATION") & group.valid]
        threshold = higher_quantile(calibration.statistic, 0.95)
        threshold_rows.append({"representation": representation, "target_scenarios_per_release": int(size), "method": method,
                               "calibration_trials": len(calibration), "threshold": threshold})
        current = group.copy()
        current["threshold"] = threshold
        current["alert"] = current.valid & (current.statistic > threshold)
        evaluated.append(current)
    return pd.DataFrame(threshold_rows), pd.concat(evaluated, ignore_index=True)


def summarize(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    operating, directions = [], []
    for (rep, size, method), group in evaluated.groupby(["representation", "target_scenarios_per_release", "method"], sort=False):
        aa = group[(group.family == "AA_EVALUATION") & group.valid]
        ab = group[(group.family == "AB_EVALUATION") & group.valid]
        fp, det = int(aa.alert.sum()), int(ab.alert.sum())
        fp_ci, det_ci = wilson_interval(fp, len(aa)), wilson_interval(det, len(ab))
        operating.append({"representation": rep, "target_scenarios_per_release": int(size), "method": method,
                          "aa_trials": len(aa), "aa_false_positive_rate": fp/len(aa), "aa_wilson95_low": fp_ci[0], "aa_wilson95_high": fp_ci[1],
                          "ab_trials": len(ab), "ab_detection_rate": det/len(ab), "ab_wilson95_low": det_ci[0], "ab_wilson95_high": det_ci[1],
                          "detection_minus_false_positive": det/len(ab)-fp/len(aa)})
        for direction, current in ab.groupby("experiment_set", sort=False):
            directions.append({"representation": rep, "target_scenarios_per_release": int(size), "method": method,
                               "experiment_set": direction, "trials": len(current), "detection_rate": float(current.alert.mean())})
    return pd.DataFrame(operating), pd.DataFrame(directions)


def paired_improvement(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = evaluated[(evaluated.family == "AB_EVALUATION") & (evaluated.target_scenarios_per_release == 400) & evaluated.representation.isin(PRIMARY_REPS)]
    pivot = subset.pivot(index=[*IDENTITY, "method"], columns="representation", values="alert").reset_index()
    rng = np.random.default_rng(620273)
    for method, group in pivot.groupby("method"):
        old = group.old64.astype(bool).to_numpy()
        for candidate in ("A_3407", "B_3407", "C_3407"):
            current = group[candidate].astype(bool).to_numpy()
            gained, lost = int((current & ~old).sum()), int((~current & old).sum())
            bootstrap = np.empty(10_000)
            for start in range(0, 10_000, 1000):
                idx = rng.integers(0, len(old), size=(min(1000, 10_000-start), len(old)))
                bootstrap[start:start+len(idx)] = current[idx].mean(axis=1)-old[idx].mean(axis=1)
            rows.append({"method": method, "candidate": candidate, "old64_detection": float(old.mean()),
                         "candidate_detection": float(current.mean()), "delta": float(current.mean()-old.mean()),
                         "paired_bootstrap_ci95_lower": float(np.quantile(bootstrap,.025)), "paired_bootstrap_ci95_upper": float(np.quantile(bootstrap,.975)),
                         "candidate_only_alerts": gained, "old64_only_alerts": lost,
                         "mcnemar_exact_p": float(binomtest(gained, gained+lost, .5).pvalue) if gained+lost else 1.0})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    auth, ledger, pconfig = verify()
    protocol = read_json(PROTOCOL)
    power = validate_power_config(read_json(POWER_CONFIG))
    pool = pd.read_csv(POOL / "metadata.csv").sort_values("global_row").reset_index(drop=True)
    if len(pool) != 1600 or pool.log_name.nunique() != 489:
        raise RuntimeError("Stage6P pool contract failed")
    assignments = pd.read_csv(ASSIGNMENTS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    arrays = build_representations(pool, ledger, device)
    rep_dir = out / "representations"; rep_dir.mkdir()
    trials = prepare_trials(pool, assignments, power)
    raw_parts, bandwidth_rows = [], []
    for name in ALL_REPS:
        path = rep_dir / f"{name}.npy"; np.save(path, arrays[name])
        bandwidth = median_bandwidth(arrays[name], int(pconfig["statistic"]["bandwidth_seed"]), int(pconfig["statistic"]["median_pair_draws"]))
        bandwidth_rows.append({"representation": name, "dimension": arrays[name].shape[1], "bandwidth": bandwidth, "sha256": sha256(path)})
        raw_parts.append(compute_representation(name, arrays[name], bandwidth, trials, name in PRIMARY_REPS))
    raw = pd.concat(raw_parts, ignore_index=True)
    thresholds, evaluated = calibrate(raw)
    operating, directions = summarize(evaluated)
    improvements = paired_improvement(evaluated)
    raw.to_csv(out / "stage6v_stage6p_trial_statistics.csv", index=False)
    thresholds.to_csv(out / "stage6v_stage6p_thresholds.csv", index=False)
    evaluated.to_csv(out / "stage6v_stage6p_evaluated_trials.csv", index=False)
    operating.to_csv(out / "stage6v_stage6p_operating_characteristics.csv", index=False)
    directions.to_csv(out / "stage6v_stage6p_direction_detection.csv", index=False)
    improvements.to_csv(out / "stage6v_stage6p_old64_improvement.csv", index=False)
    pd.DataFrame(bandwidth_rows).to_csv(out / "stage6v_stage6p_bandwidths_do_not_compare_raw_mmd.csv", index=False)

    gates = protocol["stage6p_unpaired_scorecard"]["primary_n400_gates"]
    decisions = []
    for rep in PRIMARY_REPS:
        op = operating[(operating.representation == rep) & (operating.target_scenarios_per_release == 400)].set_index("method")
        dirs = directions[(directions.representation == rep) & (directions.target_scenarios_per_release == 400) & (directions.method == "context_balanced")]
        decision = {"representation": rep,
                    "context_balanced_fpr": float(op.loc["context_balanced", "aa_false_positive_rate"]),
                    "context_balanced_detection": float(op.loc["context_balanced", "ab_detection_rate"]),
                    "context_balanced_direction_min": float(dirs.detection_rate.min()),
                    "raw_detection": float(op.loc["raw_marginal", "ab_detection_rate"])}
        decision["frozen_n400_gate_pass"] = bool(
            decision["context_balanced_fpr"] <= float(gates["maximum_aa_holdout_false_positive_rate"])
            and decision["context_balanced_detection"] >= float(gates["minimum_context_balanced_ab_detection_rate"])
            and decision["context_balanced_direction_min"] >= float(gates["minimum_each_direction_ab_detection_rate"])
            and decision["raw_detection"] >= float(gates["minimum_raw_method_ab_detection_rate"])
        )
        decisions.append(decision)
    decision_df = pd.DataFrame(decisions); decision_df.to_csv(out / "stage6v_stage6p_primary_decisions.csv", index=False)
    seed_rows = []
    for candidate in "ABC":
        reps = [f"{candidate}_{seed}" for seed in (3407,3408,3409)]
        for rep in reps:
            op = operating[(operating.representation == rep) & (operating.target_scenarios_per_release == 400)].set_index("method")
            seed_rows.append({"candidate": candidate, "seed": int(rep.split("_")[1]),
                              "raw_detection": float(op.loc["raw_marginal", "ab_detection_rate"]),
                              "context_balanced_detection": float(op.loc["context_balanced", "ab_detection_rate"]),
                              "context_balanced_fpr": float(op.loc["context_balanced", "aa_false_positive_rate"])})
    seed_df = pd.DataFrame(seed_rows); seed_df.to_csv(out / "stage6v_stage6p_seed_stability_n400.csv", index=False)

    result_names = ["stage6v_stage6p_trial_statistics.csv","stage6v_stage6p_thresholds.csv","stage6v_stage6p_evaluated_trials.csv",
                    "stage6v_stage6p_operating_characteristics.csv","stage6v_stage6p_direction_detection.csv","stage6v_stage6p_old64_improvement.csv",
                    "stage6v_stage6p_bandwidths_do_not_compare_raw_mmd.csv","stage6v_stage6p_primary_decisions.csv","stage6v_stage6p_seed_stability_n400.csv"]
    manifest = {"schema_version":"stage6v_stage6p_unpaired_blind_v1","status":"FROZEN_STAGE6P_UNPAIRED_BLIND_COMPLETE",
                "immutability_statement":auth["immutability_statement"],"pool_pairs":800,"unique_logs":489,"frozen_trials":2400,
                "primary_seed":3407,"secondary_seed_role":"stability_only","methods":METHODS,"cross_representation_raw_mmd2_comparison_performed":False,
                "nuplan_simulation_rerun":False,"training_or_protocol_modified":False,"authorization_sha256":sha256(AUTH),
                "checkpoint_ledger_sha256":sha256(LEDGER),"assignments_sha256":sha256(ASSIGNMENTS),"pool_metadata_sha256":sha256(POOL/"metadata.csv"),
                "result_files":{name:sha256(out/name) for name in result_names}}
    write_json(out / "stage6v_stage6p_result_manifest.json", manifest)
    lines=["# Stage6V Stage6P 非配对发布盲测报告","",f"- 状态：`{manifest['status']}`",
           "- 800 pairs / 489 logs / 2400 frozen splits；各 representation×样本量×方法独立 A/A 标定。",
           "- 禁止并且未执行跨 representation raw MMD² 比较。","- evaluation results cannot trigger retraining or protocol changes","",
           "## n=400 冻结门禁","",decision_df.to_markdown(index=False),"","## Seed 稳定性","",seed_df.to_markdown(index=False)]
    (out / "stage6v_stage6p_report_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"manifest_sha256":sha256(out/"stage6v_stage6p_result_manifest.json"),
                      "decisions":decisions,"seed_stability":seed_rows},indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
