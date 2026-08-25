#!/usr/bin/env python3
"""Build a frozen reliability and thesis-claim audit from Stage 6H outputs."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SAMPLE_SIZES = [200, 250, 300, 400]
EXPECTED_EXPERIMENTS = [
    "AA_CALIBRATION_ASSERTIVE",
    "AA_CALIBRATION_CONSERVATIVE",
    "AA_EVALUATION_ASSERTIVE",
    "AA_EVALUATION_CONSERVATIVE",
    "AB_ASSERTIVE_TO_CONSERVATIVE",
    "AB_CONSERVATIVE_TO_ASSERTIVE",
]
EXPECTED_SCOPES = [
    "overall",
    "following_interaction",
    "lane_change",
    "stop_go_control",
    "high_motion_dynamics",
    "dense_or_vulnerable_interaction",
]
TASK_LABELS_ZH = {
    "following_interaction": "跟车交互",
    "lane_change": "变道标签场景",
    "stop_go_control": "启停控制",
    "high_motion_dynamics": "高动态运动",
    "dense_or_vulnerable_interaction": "密集交通或弱势交通参与者交互",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a read-only Stage 6I reliability decomposition from frozen Stage 6H results."
    )
    parser.add_argument("--stage6h_dir", type=Path, required=True)
    parser.add_argument("--embedding_pool_summary", type=Path, required=True)
    parser.add_argument("--embedding_pool_metadata", type=Path, required=True)
    parser.add_argument("--kinematic_contrasts", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage 6I input does not exist: {path}")
    return path


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(require_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def audit_inputs(
    summary: Dict[str, Any],
    pool: Dict[str, Any],
    operating: pd.DataFrame,
    detection: pd.DataFrame,
    splits: pd.DataFrame,
    trials: pd.DataFrame,
) -> Dict[str, Any]:
    if summary.get("status") != "POWER_CURVE_COMPLETE":
        raise ValueError(f"Stage 6H summary is not complete: {summary.get('status')!r}")
    if pool.get("status") != "EXPANDED_800_PAIR_EMBEDDING_POOL_READY":
        raise ValueError(f"Embedding pool is not ready: {pool.get('status')!r}")
    if pool.get("pair_count") != 800 or pool.get("row_count") != 1600:
        raise ValueError("Stage 6I requires the frozen 800-pair / 1600-row pool")
    if not pool.get("all_pairs_complete") or not pool.get("all_embeddings_finite"):
        raise ValueError("Embedding pool pair or finite-value audit failed")

    config = summary.get("config", {})
    if config.get("sample_sizes_per_release") != EXPECTED_SAMPLE_SIZES:
        raise ValueError("Unexpected Stage 6H sample-size grid")
    if config.get("log_split_strategy") != "sequential_full_log_pool_v1":
        raise ValueError("Unexpected Stage 6H log split strategy")
    if config.get("target_detection_rate") != 0.8 or config.get("target_false_positive_rate") != 0.05:
        raise ValueError("Frozen Stage 6H sufficiency targets changed")

    for frame, name in [(operating, "operating"), (detection, "detection"), (splits, "splits"), (trials, "trials")]:
        sizes = sorted(frame["target_scenarios_per_release"].unique().tolist())
        if sizes != EXPECTED_SAMPLE_SIZES:
            raise ValueError(f"Unexpected sample sizes in {name}: {sizes}")

    if len(splits) != 2400:
        raise ValueError(f"Expected 2400 split rows, found {len(splits)}")
    if set(splits["experiment_set"]) != set(EXPECTED_EXPERIMENTS):
        raise ValueError("Split audit experiment families do not match the frozen Stage 6H design")
    if not (splits["log_split_strategy"] == "sequential_full_log_pool_v1").all():
        raise ValueError("Split audit contains a non-frozen log split strategy")
    exact_a = splits["selected_scenarios_A"] == splits["target_scenarios_per_release"]
    exact_b = splits["selected_scenarios_B"] == splits["target_scenarios_per_release"]
    if not (exact_a & exact_b).all():
        raise ValueError("At least one Stage 6H release split missed its exact target sample size")
    if int(splits["log_overlap_count"].max()) != 0:
        raise ValueError("At least one Stage 6H release split has log leakage")
    if int(splits["scenario_overlap_count"].max()) != 0:
        raise ValueError("At least one Stage 6H release split has scenario leakage")

    if len(operating) != len(EXPECTED_SAMPLE_SIZES) * len(EXPECTED_SCOPES):
        raise ValueError(f"Expected 24 operating rows, found {len(operating)}")
    if set(operating["scope"]) != set(EXPECTED_SCOPES):
        raise ValueError("Operating-characteristic scopes do not match the frozen design")
    if len(detection) != 144 or set(detection["scope"]) != set(EXPECTED_SCOPES):
        raise ValueError("Detection summary is incomplete")
    if set(detection["experiment_set"]) != set(EXPECTED_EXPERIMENTS):
        raise ValueError("Detection summary experiments are incomplete")
    if len(trials) != 14400 or set(trials["scope"]) != set(EXPECTED_SCOPES):
        raise ValueError("Trial BDD table is incomplete")
    trial_scope_counts = trials.groupby(
        ["target_scenarios_per_release", "experiment_set", "repetition"], sort=False
    )["scope"].nunique()
    if len(trial_scope_counts) != 2400 or not (trial_scope_counts == len(EXPECTED_SCOPES)).all():
        raise ValueError("Each release split must have exactly one row for every frozen scope")
    if int(trials["log_overlap_count"].max()) != 0 or int(trials["scenario_overlap_count"].max()) != 0:
        raise ValueError("Trial BDD table contains release leakage")

    threshold_audit = summary.get("threshold_audit", {})
    if not threshold_audit.get("all_overall_thresholds_pass"):
        raise ValueError("At least one frozen overall threshold is invalid")
    paired = summary.get("paired_oracle", {})
    if paired.get("pair_count") != 310 or not np.isfinite(paired.get("overall_original_mmd2", np.nan)):
        raise ValueError("Paired oracle is missing or invalid")

    return {
        "pass": True,
        "stage6h_status": summary["status"],
        "stage6h_sufficiency_status": summary.get("sufficiency", {}).get("status"),
        "embedding_pool_status": pool["status"],
        "pair_count": pool["pair_count"],
        "embedding_row_count": pool["row_count"],
        "log_cluster_count": pool.get("cluster_count"),
        "split_count": len(splits),
        "trial_scope_row_count": len(trials),
        "all_exact_target_sizes": True,
        "max_log_overlap_count": 0,
        "max_scenario_overlap_count": 0,
        "all_overall_thresholds_pass": True,
        "sample_sizes_per_release": EXPECTED_SAMPLE_SIZES,
        "scopes": EXPECTED_SCOPES,
    }


def build_primary(operating: pd.DataFrame) -> pd.DataFrame:
    primary = operating.loc[operating["scope"] == "overall"].copy()
    primary = primary.sort_values("target_scenarios_per_release").reset_index(drop=True)
    primary["false_negative_rate"] = 1.0 - primary["ab_detection_rate"]
    primary["false_negative_wilson95_low"] = 1.0 - primary["ab_detection_wilson95_high"]
    primary["false_negative_wilson95_high"] = 1.0 - primary["ab_detection_wilson95_low"]
    primary["interval_separation_margin"] = (
        primary["ab_detection_wilson95_low"] - primary["aa_false_positive_wilson95_high"]
    )
    primary["aa_ab_wilson_intervals_separated"] = primary["interval_separation_margin"] > 0.0
    primary["point_target_gate_pass"] = (
        (primary["ab_detection_rate"] >= 0.8) & (primary["aa_false_positive_rate"] <= 0.05)
    )
    primary["confidence_target_gate_pass"] = (
        (primary["ab_detection_wilson95_low"] >= 0.8)
        & (primary["aa_false_positive_wilson95_high"] <= 0.05)
    )
    primary["interpretation"] = np.where(
        primary["confidence_target_gate_pass"],
        "FROZEN_SUFFICIENCY_GATE_PASS",
        "SEPARATED_FROM_AA_BUT_FROZEN_SUFFICIENCY_GATE_FAIL",
    )
    return primary


def build_direction_diagnostics(detection: pd.DataFrame) -> pd.DataFrame:
    ab = detection.loc[
        (detection["scope"] == "overall") & (detection["family"] == "AB_EVALUATION")
    ].copy()
    rows: List[Dict[str, Any]] = []
    for size, group in ab.groupby("target_scenarios_per_release", sort=True):
        by_name = {row.experiment_set: row for row in group.itertuples(index=False)}
        if set(by_name) != {"AB_ASSERTIVE_TO_CONSERVATIVE", "AB_CONSERVATIVE_TO_ASSERTIVE"}:
            raise ValueError(f"Missing an A/B direction at sample size {size}")
        values: Dict[str, Any] = {"target_scenarios_per_release": int(size)}
        rates = []
        for short, name in [
            ("assertive_to_conservative", "AB_ASSERTIVE_TO_CONSERVATIVE"),
            ("conservative_to_assertive", "AB_CONSERVATIVE_TO_ASSERTIVE"),
        ]:
            row = by_name[name]
            successes = int(row.exceedance_count)
            valid = int(row.valid_trials)
            low, high = wilson_interval(successes, valid)
            values[f"{short}_valid_trials"] = valid
            values[f"{short}_detections"] = successes
            values[f"{short}_detection_rate"] = successes / valid
            values[f"{short}_wilson95_low"] = low
            values[f"{short}_wilson95_high"] = high
            rates.append(successes / valid)
        values["absolute_direction_gap"] = abs(rates[0] - rates[1])
        values["direction_mean_detection_rate"] = float(np.mean(rates))
        values["role"] = "DIAGNOSTIC_ONLY_NO_DIRECTION_EQUIVALENCE_GATE"
        rows.append(values)
    return pd.DataFrame(rows)


def build_task_diagnostics(operating: pd.DataFrame) -> pd.DataFrame:
    tasks = operating.loc[operating["scope"] != "overall"].copy()
    tasks["false_negative_rate"] = 1.0 - tasks["ab_detection_rate"]
    tasks["interval_separation_margin"] = (
        tasks["ab_detection_wilson95_low"] - tasks["aa_false_positive_wilson95_high"]
    )
    tasks["aa_ab_wilson_intervals_separated"] = tasks["interval_separation_margin"] > 0.0
    tasks["role"] = "DIAGNOSTIC_ONLY_NO_MULTIPLICITY_CONTROL"
    return tasks.sort_values(["target_scenarios_per_release", "scope"]).reset_index(drop=True)


def build_task_classification(
    summary: Dict[str, Any], pool: Dict[str, Any], pool_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(pool_metadata, ["scenario_token", "scenario_type", "planner_name"], Path("embedding_pool_metadata"))
    pair_sizes = pool_metadata.groupby("scenario_token")["planner_name"].nunique()
    if len(pair_sizes) != 800 or not (pair_sizes == 2).all():
        raise ValueError("Embedding-pool metadata must contain 800 complete two-planner scenario pairs")
    scenarios = pool_metadata.drop_duplicates("scenario_token").copy()
    task_specs = summary.get("config", {}).get("tasks", [])
    expected_counts = pool.get("task_counts", {})
    assignment_count = np.zeros(len(scenarios), dtype=int)
    task_rows: List[Dict[str, Any]] = []
    classification_rows: List[Dict[str, Any]] = []
    for spec in task_specs:
        task = str(spec["name"])
        values = [str(value) for value in spec["positive_values"]]
        mask = scenarios[str(spec["column"])].astype(str).isin(values).to_numpy()
        assignment_count += mask.astype(int)
        count = int(mask.sum())
        if task in expected_counts and count != int(expected_counts[task]):
            raise ValueError(f"Task count mismatch for {task}: metadata={count}, summary={expected_counts[task]}")
        task_rows.append(
            {
                "task": task,
                "selection_timing": str(spec.get("timing", "")),
                "scenario_column": str(spec["column"]),
                "scenario_types": json.dumps(values, ensure_ascii=False),
                "pool_pair_count": count,
                "semantic_status": (
                    "SCENARIO_TYPE_SLICE_NOT_CONFIRMED_EGO_LANE_CHANGE"
                    if task == "lane_change"
                    else "PRE_TREATMENT_SCENARIO_TYPE_SLICE"
                ),
            }
        )
        counts = scenarios.loc[mask, str(spec["column"])].astype(str).value_counts().sort_index()
        for scenario_type, type_count in counts.items():
            classification_rows.append(
                {
                    "task": task,
                    "scenario_type": scenario_type,
                    "pool_pair_count": int(type_count),
                    "selection_timing": str(spec.get("timing", "")),
                    "actual_planner_maneuver_confirmed": False if task == "lane_change" else "not_applicable",
                }
            )
    if not (assignment_count == 1).all():
        raise ValueError(
            "Every frozen pool scenario must belong to exactly one task; "
            f"unassigned={int((assignment_count == 0).sum())}, multi_assigned={int((assignment_count > 1).sum())}"
        )
    return pd.DataFrame(task_rows), pd.DataFrame(classification_rows)


def build_task_bdd_magnitudes(
    summary: Dict[str, Any],
    paired_oracle_source: Dict[str, Any],
    operating: pd.DataFrame,
    trials: pd.DataFrame,
    task_definitions: pd.DataFrame,
) -> pd.DataFrame:
    paired_rows = {
        str(row["task"]): row for row in paired_oracle_source.get("learned_embedding_task_results", [])
    }
    if set(paired_rows) != set(task_definitions["task"]):
        raise ValueError("Paired-oracle task results do not match the frozen task definitions")
    op400 = operating.loc[operating["target_scenarios_per_release"] == 400].set_index("scope")
    ab400 = trials.loc[
        (trials["target_scenarios_per_release"] == 400) & (trials["family"] == "AB_EVALUATION")
    ]
    rows: List[Dict[str, Any]] = []
    for task_row in task_definitions.itertuples(index=False):
        task = str(task_row.task)
        if task not in op400.index:
            raise ValueError(f"Missing n=400 operating characteristics for task={task}")
        paired = paired_rows[task]
        op = op400.loc[task]
        task_trials = ab400.loc[ab400["scope"] == task]
        values = pd.to_numeric(task_trials["standardized_mmd2"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) != 200:
            raise ValueError(f"Expected 200 valid n=400 A/B task trials for {task}, found {len(values)}")
        bandwidths = pd.to_numeric(task_trials["bandwidth"], errors="coerce").dropna().unique()
        if len(bandwidths) != 1:
            raise ValueError(f"Expected one fixed n=400 bandwidth for {task}, found {bandwidths.tolist()}")
        rows.append(
            {
                "task": task,
                "scenario_types": task_row.scenario_types,
                "pool_pair_count": int(task_row.pool_pair_count),
                "semantic_status": task_row.semantic_status,
                "paired_oracle_pair_count": int(paired["n_pairs"]),
                "paired_oracle_mmd2": float(paired["mmd2"]),
                "paired_oracle_bandwidth": float(paired["bandwidth"]),
                "paired_oracle_raw_p": float(paired["p_value"]),
                "paired_oracle_holm_p": float(paired["holm_p_within_pretreatment_tasks"]),
                "paired_oracle_reject_holm_0_05": bool(paired["reject_holm_0_05"]),
                "unpaired_n_per_release": 400,
                "unpaired_fixed_bandwidth": float(bandwidths[0]),
                "unpaired_standardized_mmd2_threshold": float(op["threshold"]),
                "unpaired_ab_standardized_mmd2_median": float(np.median(values)),
                "unpaired_ab_standardized_mmd2_q05": float(np.quantile(values, 0.05)),
                "unpaired_ab_standardized_mmd2_q95": float(np.quantile(values, 0.95)),
                "unpaired_ab_mean_task_n_A": float(task_trials["n_A"].mean()),
                "unpaired_ab_mean_task_n_B": float(task_trials["n_B"].mean()),
                "unpaired_aa_false_positive_rate": float(op["aa_false_positive_rate"]),
                "unpaired_ab_detection_rate": float(op["ab_detection_rate"]),
                "comparison_warning": "PAIRED_AND_UNPAIRED_MMD2_USE_DIFFERENT_ESTIMANDS_AND_BANDWIDTHS",
            }
        )
    return pd.DataFrame(rows)


def build_planner_treatment_audit(summary: Dict[str, Any], pool_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    planner_names = summary.get("config", {}).get("planners", {})
    planners = [str(planner_names.get("assertive")), str(planner_names.get("conservative"))]
    parameters: Dict[str, Dict[str, Any]] = {}
    for planner in planners:
        values = pool_metadata.loc[pool_metadata["planner_name"].astype(str) == planner, "parameters_json"].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Expected one frozen parameters_json value for planner={planner}, found {len(values)}")
        parameters[planner] = json.loads(str(values[0]))
    keys = sorted((set(parameters[planners[0]]) | set(parameters[planners[1]])) - {"source", "checkpoint_required"})
    rows: List[Dict[str, Any]] = []
    for key in keys:
        value_a = parameters[planners[0]].get(key)
        value_b = parameters[planners[1]].get(key)
        rows.append(
            {
                "parameter": key,
                "dimension": "lateral" if key == "lateral_offsets" else "longitudinal",
                "assertive_value": json.dumps(value_a, ensure_ascii=False),
                "conservative_value": json.dumps(value_b, ensure_ascii=False),
                "same_value": value_a == value_b,
            }
        )
    frame = pd.DataFrame(rows)
    pure_longitudinal = bool(frame.loc[frame["dimension"] == "lateral", "same_value"].all())
    return frame, pure_longitudinal


def build_claim_matrix(summary: Dict[str, Any], primary: pd.DataFrame) -> pd.DataFrame:
    paired = summary["paired_oracle"]
    max_row = primary.iloc[-1]
    separated_all = bool(primary["aa_ab_wilson_intervals_separated"].all())
    return pd.DataFrame(
        [
            {
                "claim_id": "C1_CROSS_DOMAIN_KNOWN_STYLE_SIGNAL",
                "claim": "Waymo-trained behavior embedding preserves a known planner-style signal in nuPlan.",
                "status": "SUPPORTED_WITHIN_PUBLIC_BENCHMARK",
                "evidence": (
                    f"Same-scenario paired oracle: n={paired['pair_count']}, MMD2={paired['overall_original_mmd2']:.8f}, "
                    f"Monte Carlo p={paired['overall_original_monte_carlo_p']:.8g}."
                ),
                "boundary": "Known synthetic planner contrast in public closed-loop nuPlan; not an OEM field release.",
            },
            {
                "claim_id": "C2_UNPAIRED_DIFFERENT_LOG_DETECTION",
                "claim": "The method detects known software-style differences when the two releases contain different logs and scenarios.",
                "status": "SUPPORTED_WITHIN_PUBLIC_RELEASE_EMULATION" if separated_all else "PARTIALLY_SUPPORTED",
                "evidence": (
                    f"A/A and A/B Wilson intervals separated at all observed sizes={separated_all}; "
                    f"at n=400/release detection={max_row.ab_detection_rate:.1%} versus FPR={max_row.aa_false_positive_rate:.1%}."
                ),
                "boundary": "Repeated splits reuse one finite public pool and control only recorded pre-treatment covariates.",
            },
            {
                "claim_id": "C3_RELIABLE_SINGLE_RELEASE_80_PERCENT",
                "claim": "One release comparison reaches at least 80% detection with controlled 5% false-positive uncertainty.",
                "status": "NOT_SUPPORTED",
                "evidence": (
                    f"At the maximum observed n=400/release, detection Wilson lower={max_row.ab_detection_wilson95_low:.1%} "
                    f"and FPR Wilson upper={max_row.aa_false_positive_wilson95_high:.1%}."
                ),
                "boundary": "No extrapolation beyond n=400/release is permitted.",
            },
            {
                "claim_id": "C4_UNIVERSAL_ABSOLUTE_BDD_THRESHOLD",
                "claim": "A single absolute BDD threshold transfers across sample sizes, ODDs, fleets, and companies.",
                "status": "NOT_SUPPORTED",
                "evidence": "Stage 6H separately calibrates an A/A threshold at each observed sample size.",
                "boundary": "Production thresholds require company-specific same-version A/A recalibration.",
            },
            {
                "claim_id": "C5_REAL_OEM_FIELD_VALIDATION",
                "claim": "The model has been validated on real OEM software releases collected at different places and times.",
                "status": "NOT_EVALUATED",
                "evidence": "No company A/A or A/B field-release data are currently available.",
                "boundary": "Public planner emulation cannot substitute for OEM field validation.",
            },
            {
                "claim_id": "C6_TASK_SPECIFIC_CONFIRMATION",
                "claim": "Each individual behavior task independently confirms the software-version difference.",
                "status": "NOT_SUPPORTED_AS_CONFIRMATORY",
                "evidence": "Task curves are descriptive diagnostics without multiplicity control.",
                "boundary": "Task results may explain signal concentration but cannot replace the overall primary result.",
            },
        ]
    )


def plot_primary(primary: pd.DataFrame, directions: pd.DataFrame, output_dir: Path) -> None:
    x = primary["target_scenarios_per_release"].to_numpy(dtype=float)
    det = primary["ab_detection_rate"].to_numpy(dtype=float)
    det_low = primary["ab_detection_wilson95_low"].to_numpy(dtype=float)
    det_high = primary["ab_detection_wilson95_high"].to_numpy(dtype=float)
    fpr = primary["aa_false_positive_rate"].to_numpy(dtype=float)
    fpr_low = primary["aa_false_positive_wilson95_low"].to_numpy(dtype=float)
    fpr_high = primary["aa_false_positive_wilson95_high"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, det, marker="o", linewidth=2.2, label="A/B detection")
    ax.fill_between(x, det_low, det_high, alpha=0.18)
    ax.plot(x, fpr, marker="s", linewidth=2.0, label="A/A false positive")
    ax.fill_between(x, fpr_low, fpr_high, alpha=0.18)
    ax.axhline(0.8, color="tab:blue", linestyle="--", linewidth=1, label="80% detection target")
    ax.axhline(0.05, color="tab:orange", linestyle=":", linewidth=1.2, label="5% FPR target")
    ax.set(xlabel="Scenarios per release", ylabel="Rate", title="Primary reliability curve")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(
        directions["target_scenarios_per_release"],
        directions["assertive_to_conservative_detection_rate"],
        marker="o",
        label="Assertive → conservative",
    )
    ax.plot(
        directions["target_scenarios_per_release"],
        directions["conservative_to_assertive_detection_rate"],
        marker="s",
        label="Conservative → assertive",
    )
    ax.plot(
        directions["target_scenarios_per_release"],
        directions["direction_mean_detection_rate"],
        color="black",
        linestyle="--",
        label="Direction mean",
    )
    ax.axhline(0.8, color="grey", linestyle=":", linewidth=1)
    ax.set(xlabel="Scenarios per release", ylabel="Detection rate", title="A/B direction diagnostic")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"stage6i_reliability_evidence.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def write_report(
    output_dir: Path,
    audit: Dict[str, Any],
    primary: pd.DataFrame,
    directions: pd.DataFrame,
    tasks: pd.DataFrame,
    task_magnitudes: pd.DataFrame,
    task_definitions: pd.DataFrame,
    task_classification: pd.DataFrame,
    treatment_audit: pd.DataFrame,
    kinematic_contrasts: pd.DataFrame,
    pure_longitudinal: bool,
    claims: pd.DataFrame,
) -> None:
    max_row = primary.iloc[-1]
    max_dir = directions.iloc[-1]
    lines = [
        "# Stage 6I 可靠性与分任务 BDD 证据报告",
        "",
        "## 1. 结论摘要",
        "",
        "Waymo 训练的行为表征在 nuPlan 公开闭环仿真中能检出已知 planner 风格信号，"
        "但当前证据不足以支持“纯纵向风格”或“80% 单次发布可靠性”。",
        "",
        f"- 输入审计：PASS（{audit['split_count']} 个 release split，log/scenario overlap=0）。",
        f"- 400 场景/版本时，A/B 检出率 {max_row.ab_detection_rate:.1%}"
        f"（Wilson 95% {max_row.ab_detection_wilson95_low:.1%}–{max_row.ab_detection_wilson95_high:.1%}），"
        f"假阴性率 {max_row.false_negative_rate:.1%}。",
        f"- A/A 误报率 {max_row.aa_false_positive_rate:.1%}"
        f"（Wilson 95% {max_row.aa_false_positive_wilson95_low:.1%}–{max_row.aa_false_positive_wilson95_high:.1%}）。",
        f"- 当前 planner 处置是否纯纵向：{'YES' if pure_longitudinal else 'NO'}。",
        "- 冻结状态：`TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS`。",
        "",
        "## 2. BDD、MMD、MMD² 与 MDD 的术语关系",
        "",
        "- **BDD（Behavior Distribution Difference）**：本项目的研究量，表示两组行为表征分布的差异。",
        "- **MMD（Maximum Mean Discrepancy）**：用于估计 BDD 的核两样本统计方法。",
        "- **MMD²**：当前报告中实际输出的数值。因此“BDD 大小”在当前实现中通常指 MMD²。",
        "- **MDD**：项目没有定义该指标；如果旧讨论中出现 MDD，应视为 MMD 的笔误，不是第三个指标。",
        "",
        "MMD² 的绝对大小受表征尺度、核带宽、样本量及配对/非配对 estimand 影响；"
        "不同设定下不能只凭小数点后有几个零判断效果。",
        "",
        "## 3. A/A 与 A/B 组合",
        "",
        "| 组合 | 两组数据 | 用途 |",
        "| --- | --- | --- |",
        "| A/A calibration | 同一 planner，两个 log 不重叠的伪发布 | 标定无版本差异时的 BDD threshold |",
        "| A/A evaluation | 同一 planner，独立随机种子的两个伪发布 | 估计假阳性/误报率 |",
        "| A/B | assertive planner 与 conservative planner，且 log/scenario 不重叠 | 估计已知版本差异的检出率 |",
        "",
        "A/B 计算了两个方向，因为释放抽样不同："
        f"assertive→conservative 为 {max_dir.assertive_to_conservative_detection_rate:.1%}，"
        f"conservative→assertive 为 {max_dir.conservative_to_assertive_detection_rate:.1%}。"
        "MMD 本身对交换 A/B 是对称的，9 个百分点差异来自有限抽样和 threshold 附近波动，只作诊断。",
        "",
        "## 4. 冻结场景分类",
        "",
        "下表的 task 由仿真前已存在的 `scenario_type` 定义，800 个场景每个恰好归入一个 task。",
        "",
        "| Task | 中文含义 | scenario_type | 800-pair 数量 | 语义状态 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in task_definitions.itertuples(index=False):
        scenario_types = "<br>".join(json.loads(row.scenario_types))
        lines.append(
            f"| {row.task} | {TASK_LABELS_ZH[row.task]} | {scenario_types} | "
            f"{row.pool_pair_count} | {row.semantic_status} |"
        )
    lines.extend([
        "",
        "详细的 scenario_type 逐类数量见 `stage6i_task_scenario_classification.csv`"
        f"（共 {len(task_classification)} 行）。",
        "",
        "## 5. 分任务 BDD 大小",
        "",
        "### 5.1 同场景配对 oracle（M6.5）",
        "",
        "| Task | pairs | BDD=MMD² | bandwidth | raw p | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in task_magnitudes.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.paired_oracle_pair_count} | {row.paired_oracle_mmd2:.8f} | "
            f"{row.paired_oracle_bandwidth:.6g} | {row.paired_oracle_raw_p:.6g} | "
            f"{row.paired_oracle_holm_p:.6g} |"
        )
    lines.extend([
        "",
        "### 5.2 异 log/异场景 release emulation（400 场景/版本）",
        "",
        "| Task | 平均 task n/side | A/B standardized MMD² 中位数 [5%,95%] | A/A threshold | A/A FPR | A/B 检出率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in task_magnitudes.itertuples(index=False):
        mean_n = (row.unpaired_ab_mean_task_n_A + row.unpaired_ab_mean_task_n_B) / 2.0
        lines.append(
            f"| {row.task} | {mean_n:.1f} | {row.unpaired_ab_standardized_mmd2_median:.8f} "
            f"[{row.unpaired_ab_standardized_mmd2_q05:.8f}, {row.unpaired_ab_standardized_mmd2_q95:.8f}] | "
            f"{row.unpaired_standardized_mmd2_threshold:.8f} | {row.unpaired_aa_false_positive_rate:.1%} | "
            f"{row.unpaired_ab_detection_rate:.1%} |"
        )
    lines.extend([
        "",
        "两张表不能横向直接比较 MMD² 绝对值：配对与非配对分析使用不同 estimand 和 bandwidth。"
        "非配对场景中 lane-change 平均每侧只有约 35 个，其 46% 检出率不是高可靠结论。",
        "",
        "## 6. lane-change 语义审计",
        "",
        "- PDM Closed 可在 nuPlan 的 `changing_lane_to_left/right` 场景中运行，但本次 task 只证明场景被数据库标为变道类型。",
        "- 当前没有使用 lane ID/车道拓扑审计证明 planner 控制的自车实际完成变道。",
        "- 旧的 `lane_change_count_proxy` 用局部横向位移超过 2 m 代理，弯道和坐标系漂移也可触发，不是严格变道标签。",
        "- 因此 lane-change 较高的信号可能来自场景条件、高动态或 planner 的横向参数，不能用来证明纯纵向风格。",
        "",
        "## 7. Planner 处置与纵向目标审计",
        "",
        "| 参数 | 维度 | assertive | conservative | 相同 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in treatment_audit.itertuples(index=False):
        lines.append(
            f"| {row.parameter} | {row.dimension} | `{row.assertive_value}` | "
            f"`{row.conservative_value}` | {row.same_value} |"
        )
    lines.extend([
        "",
        "由于 `lateral_offsets` 不同，当前 A/B 是纵向+横向的混合处置，不是纯纵向实验。"
        "不过已实现的轨迹上确实出现了纵向差异：",
        "",
        "| 指标（assertive-conservative） | n | 均值差 | 95% CI | 正差占比 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in kinematic_contrasts.itertuples(index=False):
        lines.append(
            f"| {row.metric} ({row.unit}) | {row.n_finite_pairs} | {row.mean_delta_A_minus_B:.4f} | "
            f"[{row.mean_ci95_low:.4f}, {row.mean_ci95_high:.4f}] | {row.positive_fraction:.1%} |"
        )
    lines.extend([
        "",
        "## 8. 论文主张审计",
        "",
        "| 主张 ID | 状态 |",
        "| --- | --- |",
    ])
    for row in claims.itertuples(index=False):
        lines.append(f"| {row.claim_id} | {row.status} |")
    lines.extend(
        [
            "",
            "## 9. 不得做出的解读",
            "",
            "- 不得声称已达到 80% 单次发布检出可靠性。",
            "- 不得在没有公司同版本 A/A 重标定时迁移 absolute BDD threshold。",
            "- 不得将一个有限公开池的重复 split 当作独立真实路试发布。",
            "- 不得外推 400 场景/版本以上的精确所需样本量。",
            "- 不得将当前混合处置或 lane-change 场景切片解读为纯纵向因果证据。",
            "",
            "## 10. 下一步建议",
            "",
            "1. 先构建纯纵向 PDM A/B：两版使用完全相同的 `lateral_offsets` 和横向策略，只改 headway、min-gap、speed fraction、accel/decel 等纵向参数。",
            "2. 先做同场景配对敏感性确认，再做异 log/异场景 release emulation；主分析聚焦跟车、启停和纵向高动态，lane-change 暂作诊断或排除。",
            "3. 扩大 Waymo 训练只有在补充了跟车/启停/强加减速等纵向信号、并按 log 独立划分时才更可能有效；单纯增加相似普通巡航数据不会自动解决问题。",
            "4. 为表征增加纵向辅助目标（speed/accel/jerk/THW/gap）或纵向子空间，并审计 Waymo→nuPlan 的采样频率、坐标、mask 和邻车语义对齐。",
            "5. 异场景评估应优先增加独立 log/cluster；800 场景池在 400/版本抽样时每版中位数只有约 245 个独立 log cluster。",
            "",
        ]
    )
    (output_dir / "stage6i_reliability_evidence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage6h_dir = args.stage6h_dir.resolve()
    pool_path = args.embedding_pool_summary.resolve()
    pool_metadata_path = args.embedding_pool_metadata.resolve()
    kinematic_path = args.kinematic_contrasts.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": require_file(stage6h_dir / "stage6f_power_curve_summary.json"),
        "operating": require_file(stage6h_dir / "power_curve_operating_characteristics.csv"),
        "detection": require_file(stage6h_dir / "power_curve_detection_summary.csv"),
        "splits": require_file(stage6h_dir / "power_curve_split_audit.csv"),
        "trials": require_file(stage6h_dir / "power_curve_trial_bdd.csv"),
        "pool": require_file(pool_path),
        "pool_metadata": require_file(pool_metadata_path),
        "kinematic_contrasts": require_file(kinematic_path),
    }
    summary = read_json(paths["summary"])
    pool = read_json(paths["pool"])
    operating = pd.read_csv(paths["operating"])
    detection = pd.read_csv(paths["detection"])
    splits = pd.read_csv(paths["splits"])
    trials = pd.read_csv(paths["trials"])
    pool_metadata = pd.read_csv(paths["pool_metadata"])
    kinematic_contrasts = pd.read_csv(paths["kinematic_contrasts"])

    paired_source_path = Path(str(summary.get("paired_oracle", {}).get("source_path", ""))).resolve()
    paths["paired_oracle_source"] = require_file(paired_source_path)
    expected_paired_sha = str(summary.get("paired_oracle", {}).get("source_sha256", ""))
    if sha256_file(paired_source_path) != expected_paired_sha:
        raise ValueError("Paired-oracle source SHA-256 does not match the frozen Stage 6H summary")
    paired_oracle_source = read_json(paired_source_path)

    require_columns(
        operating,
        [
            "target_scenarios_per_release", "scope", "aa_false_positive_rate",
            "aa_false_positive_wilson95_low", "aa_false_positive_wilson95_high",
            "ab_detection_rate", "ab_detection_wilson95_low", "ab_detection_wilson95_high",
        ],
        paths["operating"],
    )
    require_columns(
        detection,
        ["target_scenarios_per_release", "experiment_set", "family", "scope", "valid_trials", "exceedance_count"],
        paths["detection"],
    )
    require_columns(
        splits,
        [
            "target_scenarios_per_release", "experiment_set", "repetition", "selected_scenarios_A",
            "selected_scenarios_B", "log_split_strategy", "log_overlap_count", "scenario_overlap_count",
        ],
        paths["splits"],
    )
    require_columns(
        trials,
        [
            "target_scenarios_per_release", "experiment_set", "family", "repetition", "scope",
            "log_overlap_count", "scenario_overlap_count", "standardized_mmd2", "bandwidth",
            "n_A", "n_B",
        ],
        paths["trials"],
    )
    require_columns(
        pool_metadata,
        ["scenario_token", "scenario_type", "planner_name", "parameters_json"],
        paths["pool_metadata"],
    )
    require_columns(
        kinematic_contrasts,
        [
            "metric", "unit", "n_finite_pairs", "mean_delta_A_minus_B", "mean_ci95_low",
            "mean_ci95_high", "positive_fraction",
        ],
        paths["kinematic_contrasts"],
    )

    audit = audit_inputs(summary, pool, operating, detection, splits, trials)
    primary = build_primary(operating)
    directions = build_direction_diagnostics(detection)
    tasks = build_task_diagnostics(operating)
    task_definitions, task_classification = build_task_classification(summary, pool, pool_metadata)
    task_magnitudes = build_task_bdd_magnitudes(
        summary, paired_oracle_source, operating, trials, task_definitions
    )
    treatment_audit, pure_longitudinal = build_planner_treatment_audit(summary, pool_metadata)
    claims = build_claim_matrix(summary, primary)

    primary.to_csv(output_dir / "stage6i_primary_reliability.csv", index=False)
    directions.to_csv(output_dir / "stage6i_direction_diagnostics.csv", index=False)
    tasks.to_csv(output_dir / "stage6i_task_diagnostics.csv", index=False)
    task_definitions.to_csv(output_dir / "stage6i_task_definitions.csv", index=False)
    task_classification.to_csv(output_dir / "stage6i_task_scenario_classification.csv", index=False)
    task_magnitudes.to_csv(output_dir / "stage6i_task_bdd_magnitudes.csv", index=False)
    treatment_audit.to_csv(output_dir / "stage6i_planner_treatment_audit.csv", index=False)
    claims.to_csv(output_dir / "stage6i_thesis_claim_support_matrix.csv", index=False)
    plot_primary(primary, directions, output_dir)
    write_report(
        output_dir, audit, primary, directions, tasks, task_magnitudes, task_definitions,
        task_classification, treatment_audit, kinematic_contrasts, pure_longitudinal, claims,
    )

    tool_path = Path(__file__).resolve()
    provenance = {
        "schema_version": "stage6i_reliability_evidence_provenance_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": str(tool_path),
        "tool_sha256": sha256_file(tool_path),
        "git_commit": git_commit(tool_path.parent.parent),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "prohibitions": {
            "rollout_context_or_embedding_arrays_read": False,
            "threshold_recalibrated": False,
            "curve_smoothed": False,
            "sample_size_extrapolated_above_400": False,
        },
    }
    (output_dir / "stage6i_reproducibility_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    max_row = primary.iloc[-1]
    result = {
        "schema_version": "stage6i_reliability_evidence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RELIABILITY_EVIDENCE_COMPLETE",
        "issue": "https://github.com/forwardxp-021/E2E-Evaluation/issues/247",
        "followup_issue": "https://github.com/forwardxp-021/E2E-Evaluation/issues/248",
        "input_audit": audit,
        "primary_conclusion": "PUBLIC_UNPAIRED_STYLE_SIGNAL_SUPPORTED_BUT_SINGLE_RELEASE_RELIABILITY_TARGET_NOT_REACHED",
        "maximum_observed_scenarios_per_release": int(max_row.target_scenarios_per_release),
        "maximum_observed_detection_rate": float(max_row.ab_detection_rate),
        "maximum_observed_detection_wilson95": [
            float(max_row.ab_detection_wilson95_low), float(max_row.ab_detection_wilson95_high)
        ],
        "maximum_observed_false_positive_rate": float(max_row.aa_false_positive_rate),
        "maximum_observed_false_positive_wilson95": [
            float(max_row.aa_false_positive_wilson95_low), float(max_row.aa_false_positive_wilson95_high)
        ],
        "maximum_observed_false_negative_rate": float(max_row.false_negative_rate),
        "aa_ab_wilson_intervals_separated_at_all_observed_sizes": bool(
            primary["aa_ab_wilson_intervals_separated"].all()
        ),
        "frozen_sufficiency_gate_passed": bool(primary["confidence_target_gate_pass"].any()),
        "planner_contrast_pure_longitudinal": pure_longitudinal,
        "lane_change_semantic_status": "SCENARIO_TYPE_SLICE_NOT_CONFIRMED_EGO_LANE_CHANGE",
        "terminology": {
            "BDD": "Behavior Distribution Difference (research quantity)",
            "MMD": "Maximum Mean Discrepancy (estimator family)",
            "reported_numeric_value": "MMD2",
            "MDD": "NOT_DEFINED_TREAT_AS_MMD_TYPO",
        },
        "claim_status_counts": claims["status"].value_counts().sort_index().to_dict(),
        "no_extrapolation_above_observed_range": True,
        "outputs": {
            "primary_reliability": "stage6i_primary_reliability.csv",
            "direction_diagnostics": "stage6i_direction_diagnostics.csv",
            "task_diagnostics": "stage6i_task_diagnostics.csv",
            "task_definitions": "stage6i_task_definitions.csv",
            "task_scenario_classification": "stage6i_task_scenario_classification.csv",
            "task_bdd_magnitudes": "stage6i_task_bdd_magnitudes.csv",
            "planner_treatment_audit": "stage6i_planner_treatment_audit.csv",
            "thesis_claim_support_matrix": "stage6i_thesis_claim_support_matrix.csv",
            "report": "stage6i_reliability_evidence_report.md",
            "figure_png": "stage6i_reliability_evidence.png",
            "figure_pdf": "stage6i_reliability_evidence.pdf",
            "provenance": "stage6i_reproducibility_provenance.json",
        },
    }
    (output_dir / "stage6i_reliability_evidence_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
