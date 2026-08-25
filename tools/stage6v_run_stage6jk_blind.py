#!/usr/bin/env python3
"""Frozen Stage6J/K paired longitudinal blind evaluation for locked representations."""

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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage6j_run_paired_bdd import build_task_masks  # noqa: E402
from tools.stage6k_run_longitudinal_dose_bdd import build_pairs, null_diagnostics  # noqa: E402
from tools.stage6l_run_context_representation_ablation import kernel_analysis  # noqa: E402
from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import holm_adjust  # noqa: E402

AUTH = ROOT / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"
PROTOCOL = ROOT / "configs/stage6t_training_evaluation_protocol.json"
LEDGER = ROOT / "outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"
STAGE6J_CONFIG = ROOT / "configs/stage6j_paired_bdd_analysis.json"
REP_SOURCE = ROOT / "outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired"
CONTEXT_J = ROOT / "outputs/stage6j_pure_longitudinal_context_v1"
CONTEXT_K = ROOT / "outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired"
REALIZED = ROOT / "outputs/stage6k_realized_longitudinal_dose_curve_v2_runtime_repaired/stage6k_realized_dose_summary.json"
DEFAULT_OUT = ROOT / "outputs/stage6v_stage6jk_paired_blind_v1"

DOSES = [
    ("dose25", 0.25, "pdm_closed_assertive_longitudinal_dose25_v1"),
    ("dose50", 0.50, "pdm_closed_assertive_longitudinal_dose50_v1"),
    ("dose75", 0.75, "pdm_closed_assertive_longitudinal_dose75_v1"),
    ("dose100", 1.00, "pdm_closed_assertive_longitudinal_v1"),
]
REPS = ["old64", "A", "B", "C", "ego13"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(AUTH) != AUTH_SHA:
        raise RuntimeError("Blind authorization changed")
    auth = read_json(AUTH)
    if auth["status"] != "AUTHORIZED_STAGE6_ONE_TIME_BLIND_EVALUATION":
        raise RuntimeError("Blind evaluation not authorized")
    ledger = read_json(LEDGER)
    if ledger["status"] != "LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK":
        raise RuntimeError("Checkpoint ledger not locked")
    return auth, ledger


def context_dir(label: str) -> Path:
    return CONTEXT_J if label == "dose100" else CONTEXT_K / label


def schema_names() -> list[str]:
    schema = read_json(CONTEXT_J / "feature_schema.json")
    names = [str(row["name"]) for row in schema["features"]]
    if len(names) != 33:
        raise RuntimeError("Invalid context feature schema")
    return names


def candidate_models(ledger: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    groups = feature_group_indices(schema_names())
    models: dict[str, torch.nn.Module] = {}
    for candidate in ("A", "B", "C"):
        row = next(row for row in ledger["rows"] if row["candidate"] == candidate and int(row["seed"]) == 3407)
        if sha256(Path(row["best_checkpoint_path"])) != row["best_checkpoint_sha256"]:
            raise RuntimeError(f"Locked checkpoint changed: {candidate}")
        model = UnifiedABCModel(candidate, groups)
        payload = torch.load(row["best_checkpoint_path"], map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        models[candidate] = model.eval().to(device)
    return models


def embed_candidates(models: dict[str, torch.nn.Module], values: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {name: [] for name in models}
    with torch.no_grad():
        for start in range(0, len(values), 128):
            x = torch.from_numpy(np.asarray(values[start : start + 128], dtype=np.float32)).to(device)
            for name, model in models.items():
                result[name].append(model(x).detach().cpu().numpy().astype(np.float64))
    return {name: np.concatenate(chunks) for name, chunks in result.items()}


def apply_holm(rows: list[dict[str, Any]]) -> None:
    for representation in REPS:
        overall = [row for row in rows if row["representation"] == representation and row["scope"] == "overall"]
        task = [row for row in rows if row["representation"] == representation and row["scope"] != "overall"]
        for family in (overall, task):
            adjusted = holm_adjust([float(row["raw_p"]) for row in family])
            for row, value in zip(family, adjusted):
                row["holm_p"] = float(value)
                row["reject_holm_0_05"] = bool(value < 0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    auth, ledger = verify()
    protocol = read_json(PROTOCOL)
    scorecard = protocol["stage6jk_paired_scorecard"]
    if scorecard["representations"] != ["old64", "A_primary", "B_primary", "C_primary", "ego13"]:
        raise RuntimeError("Frozen Stage6J/K representation roster changed")
    task_definitions = read_json(STAGE6J_CONFIG)["task_conditioned_secondary"]["tasks"]
    rep_manifest = read_json(REP_SOURCE / "stage6l_representation_manifest.json")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = candidate_models(ledger, device)
    rep_arrays: dict[tuple[str, str], np.ndarray] = {}
    metadata_by_dose: dict[str, pd.DataFrame] = {}
    source_records: dict[str, Any] = {}
    export_dir = out / "representations"
    for representation in REPS:
        (export_dir / representation).mkdir(parents=True, exist_ok=True)
    for label, _, _ in DOSES:
        directory = context_dir(label)
        metadata = pd.read_csv(directory / "metadata.csv").sort_values("global_row").reset_index(drop=True)
        context = np.load(directory / "context_traj.npy", mmap_mode="r")
        if context.shape != (366, 150, 83):
            raise RuntimeError(f"Unexpected Stage6J/K context shape: {label} {context.shape}")
        generated = embed_candidates(models, context, device)
        old_path = Path(rep_manifest["representations"]["learned64_full_context"][label]["path"])
        ego_path = Path(rep_manifest["representations"]["ego_kinematic_13d"][label]["path"])
        rep_arrays[("old64", label)] = np.asarray(np.load(old_path), dtype=np.float64)
        rep_arrays[("ego13", label)] = np.asarray(np.load(ego_path), dtype=np.float64)
        for candidate, values in generated.items():
            rep_arrays[(candidate, label)] = values
        metadata_by_dose[label] = metadata
        source_records[label] = {
            "context": {"path": str(directory / "context_traj.npy"), "sha256": sha256(directory / "context_traj.npy")},
            "metadata": {"path": str(directory / "metadata.csv"), "sha256": sha256(directory / "metadata.csv")},
        }
        for representation in REPS:
            path = export_dir / representation / f"{label}.npy"
            np.save(path, rep_arrays[(representation, label)].astype(np.float32))

    rows: list[dict[str, Any]] = []
    null_samples: dict[str, np.ndarray] = {}
    repetitions = 100_000
    seed_base = 2026081301
    canonical_tokens: list[str] | None = None
    for rep_index, representation in enumerate(REPS):
        for dose_index, (label, dose, planner_a) in enumerate(DOSES):
            values = rep_arrays[(representation, label)]
            metadata = metadata_by_dose[label]
            pairs, tokens, _ = build_pairs(metadata, planner_a)
            if canonical_tokens is None:
                canonical_tokens = tokens
            elif canonical_tokens != tokens:
                raise RuntimeError("Stage6J/K scenario order changed across dose")
            task_masks, _ = build_task_masks(metadata, pairs, task_definitions)
            scopes = [("overall", np.ones(len(pairs), dtype=bool)), *task_masks.items()]
            for scope_index, (scope, mask) in enumerate(scopes):
                selected = pairs[np.asarray(mask, dtype=bool)]
                result, samples, _ = kernel_analysis(
                    values[selected[:, 0]], values[selected[:, 1]],
                    repetitions=repetitions,
                    seed=seed_base + rep_index * 1000 + dose_index * 100 + scope_index,
                )
                rows.append({
                    "representation": representation,
                    "dose_label": label,
                    "nominal_dose": dose,
                    "scope": scope,
                    "role": "overall" if scope == "overall" else "task_conditioned",
                    **result,
                    "holm_p": math.nan,
                    "reject_holm_0_05": False,
                })
                null_samples[f"{representation}__{label}__{scope}"] = samples.astype(np.float32)
    apply_holm(rows)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out / "stage6v_stage6jk_paired_results.csv", index=False)
    np.savez_compressed(out / "stage6v_stage6jk_paired_null_samples.npz", **null_samples)

    realized = read_json(REALIZED)
    realized_pass = bool(
        all(row["kinematic_gate_passed"] for row in realized["gate_decisions"])
        and all(abs(float(row["spearman_rho"])) >= 0.999 for row in realized["ordered_trends_descriptive"] if row["metric"] in {"delta_mean_speed", "delta_rms_accel", "delta_rms_jerk"})
    )
    gates = scorecard["candidate_gates"]
    decisions: list[dict[str, Any]] = []
    for representation in REPS:
        subset = result_df[result_df.representation == representation]
        overall = subset[subset.scope == "overall"].sort_values("nominal_dose")
        task = subset[subset.scope != "overall"]
        detected_doses = overall.loc[overall.reject_holm_0_05, "nominal_dose"].astype(float).tolist()
        z_values = overall["null_standardized_z_bdd"].astype(float).to_numpy()
        rho = float(spearmanr(overall["nominal_dose"].astype(float), z_values).statistic)
        decision = {
            "representation": representation,
            "overall_holm_pass_doses_out_of_4": int(overall.reject_holm_0_05.sum()),
            "task_dose_holm_pass_cells_out_of_12": int(task.reject_holm_0_05.sum()),
            "minimum_detectable_nominal_dose": min(detected_doses) if detected_doses else None,
            "median_overall_z_bdd": float(np.median(z_values)),
            "spearman_nominal_dose_vs_z_bdd": rho,
            "realized_speed_accel_jerk_monotonicity_pass": realized_pass,
        }
        decision["frozen_longitudinal_gate_pass"] = bool(
            decision["overall_holm_pass_doses_out_of_4"] >= int(gates["required_overall_holm_pass_nonzero_doses"])
            and decision["task_dose_holm_pass_cells_out_of_12"] >= int(gates["required_task_dose_holm_pass_cells"])
            and decision["minimum_detectable_nominal_dose"] is not None
            and decision["minimum_detectable_nominal_dose"] <= float(gates["maximum_minimum_detectable_nominal_dose"])
            and decision["median_overall_z_bdd"] >= float(gates["minimum_median_overall_z_bdd"])
            and rho >= float(gates["minimum_spearman_nominal_dose_vs_z_bdd"])
            and realized_pass
        )
        decisions.append(decision)
    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(out / "stage6v_stage6jk_decisions.csv", index=False)

    result_files = {}
    for name in ("stage6v_stage6jk_paired_results.csv", "stage6v_stage6jk_paired_null_samples.npz", "stage6v_stage6jk_decisions.csv"):
        result_files[name] = sha256(out / name)
    manifest = {
        "schema_version": "stage6v_stage6jk_paired_blind_v1",
        "status": "FROZEN_STAGE6J_K_PAIRED_BLIND_COMPLETE",
        "immutability_statement": auth["immutability_statement"],
        "primary_seed": 3407,
        "scenario_count": 183,
        "doses": [dose for _, dose, _ in DOSES],
        "representations": REPS,
        "permutations_per_scope": repetitions,
        "cross_representation_raw_mmd2_comparison_performed": False,
        "nuplan_simulation_rerun": False,
        "source_records": source_records,
        "realized_dose_summary": {"path": str(REALIZED), "sha256": sha256(REALIZED)},
        "authorization_sha256": sha256(AUTH),
        "checkpoint_ledger_sha256": sha256(LEDGER),
        "result_files": result_files,
        "training_or_protocol_modified": False,
    }
    write_json(out / "stage6v_stage6jk_result_manifest.json", manifest)
    lines = [
        "# Stage6V Stage6J/K 纵向 paired 盲测报告",
        "",
        f"- 状态：`{manifest['status']}`",
        "- 183 个同场景 pair；25/50/75/100% 四剂量；每个 representation×dose×scope 独立 bandwidth 和 100000 次 pair 内交换。",
        "- raw MMD² 未跨 representation 比较。",
        "- evaluation results cannot trigger retraining or protocol changes",
        "",
        "## 冻结门禁",
        "",
        decision_df.to_markdown(index=False),
    ]
    (out / "stage6v_stage6jk_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": sha256(out / "stage6v_stage6jk_result_manifest.json"),
        "decisions": decisions,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
