#!/usr/bin/env python3
"""Complete frozen R0 D0 semantic-retention/readout analyses for Wave 1.1.

This command uses only existing Wave-1 inputs/checkpoints.  It does not train
encoders, call a planner, alter a frozen protocol asset, or rewrite a prior
metric table.  The sole prior-result mutation is the required factual entry in
the Wave-1 protocol-deviation log when the earlier D0 conclusion is superseded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold

from tools.stage6l_prepare_context_representation_ablation import ego_kinematic_features
from tools import stageR_execute_r0_wave1 as wave1


ROOT = wave1.ROOT
RESULTS = ROOT / "docs/stageR/r0/results"
OUT = ROOT / "outputs/stageR/r0_wave1_1_d0_retention_v1"
WAVE1_COMMIT = "ed72dbd87d8dc49ff761c95fca460ea4bfafa5c7"
NEW_RESULT_NAMES = {
    "r0_d0_retention_probe_metrics_wave1_1.csv",
    "r0_d0_retention_family_summary_wave1_1.csv",
    "r0_d0_retention_decision_matrix_wave1_1.csv",
    "r0_wave1_1_hypothesis_results.json",
    "R0_Wave1_1_D0_Retention_Completion_Report_v1.md",
    "R0_Wave1_Cross_Module_Diagnosis_v1.1.md",
    "r0_wave1_1_execution_manifest.json",
    "r0_wave1_1_command_ledger.json",
    "r0_wave1_1_execution_completeness_assessment.json",
}
PROTOCOL_LOG = RESULTS / "r0_wave1_protocol_deviation_log.csv"
D0_VIEWS = {
    "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY": ("last", "mean", "max"),
    "D0-D_MASK_PADDING_SENSITIVITY": ("last", "final_valid", "masked_mean"),
}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_wave1_1_freeze() -> dict[str, Any]:
    if git("branch", "--show-current") != wave1.BRANCH:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: wrong branch")
    if git("rev-parse", "HEAD") != WAVE1_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: Wave 1.1 must start at the declared Wave 1 commit")
    if git("rev-parse", "r0-v1.0-protocol-freeze^{}") != wave1.BINDING_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: frozen tag target")
    binding = read_json(ROOT / "docs/stageR/r0/manifests/r0_v1_freeze_binding.json")
    if binding["R0_V1_FREEZE_CONTENT_COMMIT"] != wave1.CONTENT_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: frozen content commit")
    for name, entry in binding["all_frozen_artifact_sha256"].items():
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"PROTOCOL_BINDING_MISMATCH: {name}")
    protected = ROOT / binding["protected_dirty_output_exclusion"]["path"]
    protected_sha = sha256(protected)
    if protected_sha != binding["protected_dirty_output_exclusion"]["sha256"]:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: protected historical CSV")
    return {
        "branch": wave1.BRANCH,
        "wave1_commit": WAVE1_COMMIT,
        "protocol_tag": "r0-v1.0-protocol-freeze",
        "protocol_binding_commit": wave1.BINDING_COMMIT,
        "protocol_content_commit": wave1.CONTENT_COMMIT,
        "frozen_artifact_count": len(binding["all_frozen_artifact_sha256"]),
        "protected_csv_path": str(protected.relative_to(ROOT)),
        "protected_csv_sha256": protected_sha,
        "git_status_sha256_before_execution": hashlib.sha256(git("status", "--porcelain=v1").encode()).hexdigest(),
        "git_status_line_count_before_execution": len(git("status", "--porcelain=v1").splitlines()),
    }


def load_stage7l_d0_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, str]]:
    root = ROOT / "outputs/stage7l_e_prospective_bdd_v1/contexts"
    contexts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ego_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    scenarios: list[str] = []
    logs: list[str] = []
    input_shas: dict[str, str] = {}
    reference_scenarios: list[str] | None = None
    reference_schema: list[str] | None = None
    for dose in ("dose0", "dose25", "dose50", "dose75", "dose100"):
        directory = root / dose
        context = np.asarray(np.load(directory / "context_traj.npy", mmap_mode="r"), dtype=np.float32)
        mask = np.asarray(np.load(directory / "ego_seq_mask.npy", mmap_mode="r"), dtype=bool)
        ego = np.asarray(np.load(directory / "ego_seq.npy", mmap_mode="r"), dtype=np.float64)
        raw = np.asarray(np.load(directory / "interaction_feat_style.npy", mmap_mode="r"), dtype=np.float64)
        metadata = list(csv.DictReader((directory / "metadata.csv").open(encoding="utf-8")))
        schema = read_json(directory / "feature_schema.json")
        names = [str(row["name"]) for row in schema["features"]]
        if context.shape != (80, 150, 83) or mask.shape != (80, 150) or ego.shape != (80, 150, 8) or raw.shape != (80, 33):
            raise RuntimeError(f"Stage7L D0 input contract mismatch at {dose}")
        if not np.isfinite(context).all() or not np.isfinite(ego).all() or not np.isfinite(raw).all():
            raise RuntimeError(f"Stage7L D0 nonfinite input at {dose}")
        local_scenarios = [row["scenario_token"] for row in metadata]
        if len(metadata) != 80 or any(not value for value in local_scenarios) or any(not row["log_name"] for row in metadata):
            raise RuntimeError(f"Stage7L metadata contract mismatch at {dose}")
        if reference_scenarios is None:
            reference_scenarios = local_scenarios
            reference_schema = names
        elif local_scenarios != reference_scenarios or names != reference_schema:
            raise RuntimeError(f"Stage7L scenario/schema contract mismatch at {dose}")
        for name in ("context_traj.npy", "ego_seq_mask.npy", "ego_seq.npy", "interaction_feat_style.npy", "metadata.csv", "feature_schema.json"):
            input_shas[f"{dose}/{name}"] = sha256(directory / name)
        contexts.append(context); masks.append(mask); ego_parts.append(ego); raw_parts.append(raw)
        scenarios.extend(local_scenarios); logs.extend([row["log_name"] for row in metadata])
    ego_all = np.concatenate(ego_parts)
    mask_all = np.concatenate(masks)
    raw_all = np.concatenate(raw_parts)
    valid_lengths = mask_all.sum(axis=1).astype(np.int64)
    if set(valid_lengths.tolist()) != {149, 150}:
        raise RuntimeError("Stage7L D0 valid-length contract mismatch")
    ego13 = ego_kinematic_features(ego_all, mask_all)
    targets = {
        "ego13.mean_speed": ego13[:, 0],
        "ego13.end_minus_start_speed": ego13[:, 3],
        "ego13.rms_accel": ego13[:, 4],
        "ego13.rms_yaw_rate": ego13[:, 9],
        "ego13.heading_change_abs_total": ego13[:, 11],
        "raw33.lane_change_count_proxy": (raw_all[:, 13] > 0).astype(np.int64),
        "raw33.mean_front_distance": raw_all[:, 6],
        "raw33.mean_rel_speed": raw_all[:, 8],
        "raw33.front_pressure_score": raw_all[:, 21],
    }
    if len(contexts) != 5 or any(len(values) != 400 for values in targets.values()):
        raise RuntimeError("Stage7L D0 row contract mismatch")
    if len(np.unique(scenarios)) != 80 or len(np.unique(logs)) != 79:
        raise RuntimeError("Stage7L D0 grouping contract mismatch")
    if any(not np.isfinite(value).all() for value in targets.values()):
        raise RuntimeError("Stage7L D0 target nonfinite")
    return np.concatenate(contexts), valid_lengths, targets, np.asarray(scenarios), np.asarray(logs), input_shas


def grouped_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))


def fit_model(x: np.ndarray, y: np.ndarray, groups: np.ndarray, categorical: bool) -> Any:
    return wave1.selected_logistic(x, y.astype(int), groups) if categorical else wave1.selected_ridge(x, y, groups)


def oof_frozen_and_refit(views: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, categorical: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frozen = {name: np.empty(len(y), dtype=np.float64) for name in views}
    refit = {name: np.empty(len(y), dtype=np.float64) for name in views}
    for train, test in grouped_folds(groups):
        frozen_model = fit_model(views["last"][train], y[train], groups[train], categorical)
        for name, x in views.items():
            frozen[name][test] = frozen_model.predict(x[test])
            refit_model = fit_model(x[train], y[train], groups[train], categorical)
            refit[name][test] = refit_model.predict(x[test])
    return frozen, refit


def heldout_score(y: np.ndarray, prediction: np.ndarray, categorical: bool) -> float:
    return float(balanced_accuracy_score(y.astype(int), prediction.astype(int))) if categorical else float(r2_score(y, prediction))


def row_loss(y: np.ndarray, prediction: np.ndarray, categorical: bool) -> np.ndarray:
    if categorical:
        return (y.astype(int) != prediction.astype(int)).astype(np.float64)
    return np.square(y - prediction) / max(float(np.var(y)), 1e-12)


def paired_loss_effect(reference_loss: np.ndarray, alternate_loss: np.ndarray, log_groups: np.ndarray, seed: int) -> tuple[float, float, float]:
    names, inverse = np.unique(log_groups, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(names))
    diff = alternate_loss - reference_loss
    means = np.bincount(inverse, weights=diff, minlength=len(names)) / np.maximum(counts, 1)
    if not np.isfinite(means).all():
        raise RuntimeError("Nonfinite grouped paired loss contrast")
    scale = float(np.std(means, ddof=1))
    if scale <= 1e-12:
        return 0.0, 0.0, 0.0
    point = float(means.mean() / scale)
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for start in range(0, wave1.BOOTSTRAP_REPS, 100):
        count = min(100, wave1.BOOTSTRAP_REPS - start)
        sample = means[rng.integers(0, len(means), size=(count, len(means)))]
        denom = np.maximum(sample.std(axis=1, ddof=1), 1e-12)
        values.append(sample.mean(axis=1) / denom)
    boot = np.concatenate(values)
    return point, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def target_interpretation(frozen_loss: bool, frozen_change: bool, refit_loss: bool, refit_change: bool, refit_gain: bool) -> str:
    if frozen_loss and refit_loss:
        return "INFORMATION_RETENTION_LOSS_FAVORED"
    if frozen_loss and not refit_change:
        return "COORDINATE_GEOMETRY_SHIFT_FAVORED"
    if not frozen_change and not refit_change:
        return "NO_MATERIAL_RETENTION_LOSS_SUPPORTED"
    if not frozen_change and refit_loss:
        return "REFIT_SPLIT_OR_CALIBRATION_ANOMALY_REVIEW"
    if refit_gain:
        return "REFIT_VIEW_GAIN_NOT_INFORMATION_LOSS"
    return "MIXED_OR_UNRESOLVED"


def family_interpretation(rows: list[dict[str, Any]]) -> str:
    labels = [str(row["interpretation_matrix_result"]) for row in rows]
    if labels.count("INFORMATION_RETENTION_LOSS_FAVORED") >= 2:
        return "INFORMATION_RETENTION_LOSS_FAVORED"
    if labels.count("COORDINATE_GEOMETRY_SHIFT_FAVORED") >= 2 and "INFORMATION_RETENTION_LOSS_FAVORED" not in labels:
        return "COORDINATE_GEOMETRY_SHIFT_FAVORED"
    if labels.count("NO_MATERIAL_RETENTION_LOSS_SUPPORTED") >= 2:
        return "NO_MATERIAL_RETENTION_LOSS_SUPPORTED"
    return "MIXED_OR_UNRESOLVED"


def summarize_families(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study, views in D0_VIEWS.items():
        for rep in sorted({str(row["representation"]) for row in metrics if row["study"] == study}):
            for view in views:
                if view == "last":
                    continue
                for family in ("longitudinal", "lateral", "interaction"):
                    targets = [row for row in metrics if row["study"] == study and row["representation"] == rep and row["view"] == view and row["semantic_family"] == family and row["analysis_mode"] == "FROZEN_PROBE_ACROSS_VIEW"]
                    refit = [row for row in metrics if row["study"] == study and row["representation"] == rep and row["view"] == view and row["semantic_family"] == family and row["analysis_mode"] == "SAME_CAPACITY_REFIT_PROBE_PER_VIEW"]
                    if len(targets) != 3 or len(refit) != 3:
                        raise RuntimeError("D0 family summary lacks a frozen/refit target triplet")
                    # Target-wise matrix outcomes are the same field in both modes; retain the frozen rows.
                    matrix = family_interpretation(targets)
                    candidate, seed = rep.split("_seed")
                    rows.append({
                        "study": study, "representation": rep, "candidate": candidate, "seed": seed, "view": view,
                        "semantic_family": family,
                        "frozen_probe_loss_gate_target_count": sum(bool(row["semantic_loss_gate"]) for row in targets),
                        "refit_probe_loss_gate_target_count": sum(bool(row["semantic_loss_gate"]) for row in refit),
                        "frozen_probe_material_change_target_count": sum(bool(row["semantic_change_gate"]) for row in targets),
                        "refit_probe_material_change_target_count": sum(bool(row["semantic_change_gate"]) for row in refit),
                        "family_interpretation": matrix,
                        "n_core_targets": 3, "evidence_level": wave1.EVIDENCE_LEVEL,
                    })
    return rows


def formal_status(family_rows: list[dict[str, Any]], study: str) -> tuple[str, str]:
    local = [row for row in family_rows if row["study"] == study]
    information_cells: list[tuple[str, str, str]] = []
    geometry_cells: list[tuple[str, str, str]] = []
    mixed = False
    for candidate in ("A", "B", "C"):
        for view in sorted({str(row["view"]) for row in local}):
            for family in ("longitudinal", "lateral", "interaction"):
                seed_rows = [row for row in local if row["candidate"] == candidate and row["view"] == view and row["semantic_family"] == family]
                if len(seed_rows) != 3:
                    raise RuntimeError("Missing fixed A/B/C seed in D0 status evaluation")
                labels = [str(row["family_interpretation"]) for row in seed_rows]
                if labels.count("INFORMATION_RETENTION_LOSS_FAVORED") >= 2:
                    information_cells.append((candidate, view, family))
                if labels.count("COORDINATE_GEOMETRY_SHIFT_FAVORED") >= 2:
                    geometry_cells.append((candidate, view, family))
                if any(label in {"INFORMATION_RETENTION_LOSS_FAVORED", "MIXED_OR_UNRESOLVED", "REFIT_SPLIT_OR_CALIBRATION_ANOMALY_REVIEW"} for label in labels):
                    mixed = True
    if information_cells:
        # The frozen decision table requires MIXED when predeclared views,
        # families, representations, or seeds materially conflict.  A single
        # seed-consistent loss cell cannot upgrade the whole D0 module when
        # other predeclared cells show geometry-only or no-loss outcomes.
        counterexamples = [
            row for row in local
            if row["family_interpretation"] != "INFORMATION_RETENTION_LOSS_FAVORED"
        ]
        if counterexamples:
            return "MIXED", "SEED_CONSISTENT_SEMANTIC_LOSS_CELL_EXISTS_BUT_PREDECLARED_VIEWS_OR_FAMILIES_HAVE_MATERIAL_DIRECTIONAL_CONFLICT"
        return "SUPPORTED", "SEMANTIC_RETENTION_LOSS_GATE_MET_FOR_2_OF_3_FIXED_SEEDS_WITHOUT_PREDECLARED_STRATUM_CONFLICT"
    if mixed:
        return "MIXED", "NO_SEED_CONSISTENT_SEMANTIC_LOSS_CELL_BUT_PREDECLARED_STRATA_ARE_MIXED"
    if geometry_cells:
        return "NOT_SUPPORTED", "POOLING_OR_MASK_GEOMETRY_SENSITIVITY_PRESENT_BUT_SEMANTIC_RETENTION_LOSS_GATE_NOT_MET"
    return "NOT_SUPPORTED", "SEMANTIC_RETENTION_LOSS_GATE_NOT_MET"


def d3_direction_note() -> dict[str, Any]:
    rows = list(csv.DictReader((RESULTS / "r0_measurement_readout_metrics.csv").open(encoding="utf-8")))
    pure = [row for row in rows if row["domain"] == "pure_lateral"]
    note: dict[str, Any] = {"formal_status": "INCONCLUSIVE", "comparison_metric": "ratio_to_null_q95", "representations": {}}
    for representation in sorted({row["representation"] for row in pure}):
        local = {row["readout"]: row for row in pure if row["representation"] == representation}
        full = float(local["R_full64"]["ratio_to_null_q95"])
        note["representations"][representation] = {
            "R_linear_task_higher_than_full64": float(local["R_linear_task"]["ratio_to_null_q95"]) > full,
            "R_fixed_semantic_higher_than_full64": float(local["R_fixed_semantic"]["ratio_to_null_q95"]) > full,
        }
    values = list(note["representations"].values())
    note["linear_task_higher_count"] = sum(bool(row["R_linear_task_higher_than_full64"]) for row in values)
    note["fixed_semantic_higher_count"] = sum(bool(row["R_fixed_semantic_higher_than_full64"]) for row in values)
    note["representation_count"] = len(values)
    return note


def append_protocol_deviation() -> dict[str, str]:
    with PROTOCOL_LOG.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        old_rows = list(reader)
    expected = ["deviation_id", "timestamp_utc", "detected_after_outcome_access", "description", "affected_protocol_section", "affects_primary", "evidence_downgrade", "mitigation", "scientific_owner_disposition", "closed_timestamp_utc"]
    if header != expected or old_rows:
        raise RuntimeError("Wave-1 protocol deviation log is not the expected untouched header-only ledger")
    row = {
        "deviation_id": "R0-WAVE1-D0-PRIMARY-METRIC-OMISSION-001",
        "timestamp_utc": now(), "detected_after_outcome_access": True,
        "description": "PROTOCOL_DEVIATION plus EXECUTION_COMPLETENESS_CORRECTION: Wave 1 used paired standardized embedding displacement alone to set D0_POOLING_EFFECT and D0_MASK_PADDING_SENSITIVITY. Frozen D0 requires frozen-probe-across-view and same-capacity-refit-probe-per-view retention/readout results.",
        "affected_protocol_section": "R0_D0_Temporal_Audit_Policy_v0.1 §§5-6, §8; R0 SAP v1.0 D0 probe_family; r0_decision_table_v1.0 D0 retention/readout contrast",
        "affects_primary": True,
        "evidence_downgrade": "Wave-1 D0 SUPPORTED labels are superseded by Wave 1.1 readout-complete development-diagnostic results; D1/D3 primary statuses unchanged.",
        "mitigation": "Execute frozen D0-C/D paired frozen-probe and refit-probe analyses on the existing Stage7L rows, fixed checkpoints, CORE targets, GroupKFold scenario split, ridge/logistic grid, and 5,000 log-cluster bootstrap.",
        "scientific_owner_disposition": "OPEN_REQUIRES_SCIENTIFIC_OWNER_REVIEW", "closed_timestamp_utc": "",
    }
    write_csv(PROTOCOL_LOG, [row], expected)
    return row


def report(path: Path, title: str, lines: list[str]) -> None:
    path.write_text(f"# {title}\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    allowed_partial_out = {"r0_wave1_1_freeze_verification.json"}
    present_out = {path.name for path in OUT.iterdir()} if OUT.exists() else set()
    if not present_out.issubset(allowed_partial_out) or any((RESULTS / name).exists() for name in NEW_RESULT_NAMES):
        raise RuntimeError("Refusing to overwrite an existing Wave 1.1 D0 retention product")
    freeze = verify_wave1_1_freeze()
    checkpoint_locks = wave1.checkpoint_sha_locks()
    contexts, valid_lengths, targets, scenarios, logs, input_shas = load_stage7l_d0_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "r0_wave1_1_freeze_verification.json").exists():
        write_json(OUT / "r0_wave1_1_freeze_verification.json", freeze)
    models = wave1.load_models()
    models = {name: model for name, model in models.items() if name.split("_")[0] in {"A", "B", "C"}}
    if len(models) != 9:
        raise RuntimeError("Expected exactly A/B/C fixed-seed checkpoints")

    metrics: list[dict[str, Any]] = []
    for representation, model in models.items():
        values = wave1.pooled_embeddings(model, contexts, valid_lengths)
        for study, view_names in D0_VIEWS.items():
            views = {name: values[name] for name in view_names}
            for family, target_id, kind in wave1.CORE_TARGETS:
                y = targets[target_id]
                categorical = kind == "categorical"
                frozen, refit = oof_frozen_and_refit(views, y, scenarios, categorical)
                for mode, predictions in (("FROZEN_PROBE_ACROSS_VIEW", frozen), ("SAME_CAPACITY_REFIT_PROBE_PER_VIEW", refit)):
                    reference_loss = row_loss(y, predictions["last"], categorical)
                    for view, prediction in predictions.items():
                        effect, lo, hi = paired_loss_effect(reference_loss, row_loss(y, prediction, categorical), logs, int(hashlib.sha256((study + representation + target_id + mode + view).encode()).hexdigest()[:8], 16))
                        change = abs(effect) >= 0.10 and not (lo <= 0.0 <= hi)
                        loss = effect >= 0.10 and lo > 0.0
                        gain = effect <= -0.10 and hi < 0.0
                        frozen_effect = None
                        refit_effect = None
                        # The same target/view is inspected after both modes are available below.
                        metrics.append({
                            "study": study, "representation": representation, "candidate": representation.split("_")[0], "seed": representation.split("seed")[1],
                            "analysis_mode": mode, "reference_view": "last", "view": view, "semantic_family": family, "target_id": target_id,
                            "target_kind": kind, "primary_metric": "balanced_accuracy" if categorical else "R2", "heldout_point_estimate": heldout_score(y, prediction, categorical),
                            "paired_standardized_semantic_loss_difference": effect, "ci95_lower": lo, "ci95_upper": hi,
                            "semantic_change_gate": change, "semantic_loss_gate": loss, "semantic_gain_gate": gain,
                            "n_rows": len(y), "n_scenarios": len(np.unique(scenarios)), "n_log_clusters": len(np.unique(logs)),
                            "split_contract": "five-fold GroupKFold grouped by scenario_token; identical rows/folds/targets/ridge-logistic grid across views; paired loss CI uses 5,000 log-cluster bootstrap replicates",
                            "target_source": "ego13 from frozen Stage7L ego_seq plus mask; raw33 from frozen Stage7L interaction_feat_style raw-equivalent report array",
                            "diagnostic_not_historical": view != "last", "evidence_level": wave1.EVIDENCE_LEVEL,
                            "interpretation_matrix_result": "PENDING_MODE_PAIR",
                        })

    index = {(row["study"], row["representation"], row["view"], row["target_id"]): {} for row in metrics}
    for row in metrics:
        index[(row["study"], row["representation"], row["view"], row["target_id"])][row["analysis_mode"]] = row
    for pair in index.values():
        frozen = pair["FROZEN_PROBE_ACROSS_VIEW"]
        refit = pair["SAME_CAPACITY_REFIT_PROBE_PER_VIEW"]
        label = target_interpretation(bool(frozen["semantic_loss_gate"]), bool(frozen["semantic_change_gate"]), bool(refit["semantic_loss_gate"]), bool(refit["semantic_change_gate"]), bool(refit["semantic_gain_gate"]))
        frozen["interpretation_matrix_result"] = label
        refit["interpretation_matrix_result"] = label
    metrics.sort(key=lambda row: (row["study"], row["representation"], row["analysis_mode"], row["view"], row["target_id"]))
    fields = list(metrics[0])
    write_csv(RESULTS / "r0_d0_retention_probe_metrics_wave1_1.csv", metrics, fields)
    family_rows = summarize_families(metrics)
    write_csv(RESULTS / "r0_d0_retention_family_summary_wave1_1.csv", family_rows, list(family_rows[0]))

    pooling_status, pooling_reason = formal_status(family_rows, "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY")
    mask_status, mask_reason = formal_status(family_rows, "D0-D_MASK_PADDING_SENSITIVITY")
    decision_rows = []
    for hypothesis, study, status, reason in (
        ("D0_POOLING_EFFECT", "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY", pooling_status, pooling_reason),
        ("D0_MASK_PADDING_SENSITIVITY", "D0-D_MASK_PADDING_SENSITIVITY", mask_status, mask_reason),
    ):
        local = [row for row in family_rows if row["study"] == study]
        decision_rows.append({
            "hypothesis_id": hypothesis, "study": study, "hypothesis_status": status, "decision_reason": reason,
            "geometry_shift_family_seed_rows": sum(row["family_interpretation"] == "COORDINATE_GEOMETRY_SHIFT_FAVORED" for row in local),
            "information_loss_family_seed_rows": sum(row["family_interpretation"] == "INFORMATION_RETENTION_LOSS_FAVORED" for row in local),
            "no_material_loss_family_seed_rows": sum(row["family_interpretation"] == "NO_MATERIAL_RETENTION_LOSS_SUPPORTED" for row in local),
            "mixed_family_seed_rows": sum(row["family_interpretation"] == "MIXED_OR_UNRESOLVED" for row in local),
            "frozen_primary_rule": "absolute paired standardized retention/readout difference >=0.10 with 95% CI excluding 0 and >=2/3 fixed-seed direction consistency",
            "evidence_level": wave1.EVIDENCE_LEVEL,
        })
    write_csv(RESULTS / "r0_d0_retention_decision_matrix_wave1_1.csv", decision_rows, list(decision_rows[0]))
    d3_note = d3_direction_note()
    case_c = "CASE_C_TEMPORAL_CONTRIBUTION_SUPPORTED" if pooling_status == "SUPPORTED" else ("CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED" if pooling_status == "MIXED" else "CASE_C_TEMPORAL_CONTRIBUTION_NOT_SUPPORTED")
    hypothesis = {
        "execution_status": "COMPLETE_EXECUTION_COMPLETENESS_CORRECTION",
        "evidence_level": wave1.EVIDENCE_LEVEL,
        "supersedes": {"D0_POOLING_EFFECT": "Wave 1 result only", "D0_MASK_PADDING_SENSITIVITY": "Wave 1 result only"},
        "hypothesis_results": {"D0_POOLING_EFFECT": pooling_status, "D0_MASK_PADDING_SENSITIVITY": mask_status},
        "embedding_geometry_effect": "SUPPORTED" if any(row["geometry_shift_family_seed_rows"] > 0 for row in decision_rows) else "NOT_SUPPORTED",
        "semantic_retention_effect": "SUPPORTED" if pooling_status == "SUPPORTED" or mask_status == "SUPPORTED" else ("MIXED" if pooling_status == "MIXED" or mask_status == "MIXED" else "NOT_SUPPORTED"),
        "case_c": case_c,
        "D1_KNOWN_SEMANTIC_INFORMATION_PRESENT": "SUPPORTED_UNCHANGED_FROM_WAVE1",
        "D3_FORMAL_HYPOTHESES": "INCONCLUSIVE_UNCHANGED_FROM_WAVE1",
        "no_training_authorization_change": True,
        "no_d2_d4_or_new_planner_rollout": True,
    }
    write_json(RESULTS / "r0_wave1_1_hypothesis_results.json", hypothesis)
    completion = {
        "classification": "PROTOCOL_DEVIATION_WITH_EXECUTION_COMPLETENESS_CORRECTION",
        "wave1_issue": "D0 primary hypothesis labels were produced from paired standardized embedding displacement without the simultaneously frozen readout-retention probe pair.",
        "frozen_requirement": "D0 must report frozen-probe-across-view plus same-capacity-refit-probe-per-view.",
        "effect_on_primary": "Yes: Wave1 D0 SUPPORTED labels are superseded by Wave1.1 results. D1 and D3 are unchanged.",
        "protocol_assets_modified": False,
    }
    write_json(RESULTS / "r0_wave1_1_execution_completeness_assessment.json", completion)
    append_protocol_deviation()

    by_study = {study: [row for row in family_rows if row["study"] == study] for study in D0_VIEWS}
    report(RESULTS / "R0_Wave1_1_D0_Retention_Completion_Report_v1.md", "R0 Wave 1.1 D0 Retention Completion Report v1", [
        f"Evidence level: `{wave1.EVIDENCE_LEVEL}`.",
        "",
        "Wave 1.1 completes the frozen D0 readout contract without altering Protocol v1.0. It uses only historical Stage7L rows, fixed A/B/C checkpoints and seeds 3407/3408/3409, the frozen nine CORE semantic targets, five-fold scenario-grouped splits, the frozen ridge/logistic grid, and 5,000 log-cluster bootstrap replicates.",
        "",
        f"- D0-C pooling formal status: `{pooling_status}` — {pooling_reason}.",
        f"- D0-D mask/padding formal status: `{mask_status}` — {mask_reason}.",
        f"- D0-C family-seed matrix counts: geometry shift={sum(row['family_interpretation'] == 'COORDINATE_GEOMETRY_SHIFT_FAVORED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}; information loss={sum(row['family_interpretation'] == 'INFORMATION_RETENTION_LOSS_FAVORED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}; no-material-loss={sum(row['family_interpretation'] == 'NO_MATERIAL_RETENTION_LOSS_SUPPORTED' for row in by_study['D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY'])}.",
        f"- D0-D family-seed matrix counts: geometry shift={sum(row['family_interpretation'] == 'COORDINATE_GEOMETRY_SHIFT_FAVORED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}; information loss={sum(row['family_interpretation'] == 'INFORMATION_RETENTION_LOSS_FAVORED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}; no-material-loss={sum(row['family_interpretation'] == 'NO_MATERIAL_RETENTION_LOSS_SUPPORTED' for row in by_study['D0-D_MASK_PADDING_SENSITIVITY'])}.",
        "",
        "`last` is the historical reference. `mean`, `max`, `final_valid`, and `masked_mean` are all `DIAGNOSTIC_NOT_HISTORICAL`; none is used to rewrite historical Stage7L inference. Target-level frozen/refit values and all family summaries are in the accompanying CSV products.",
        "",
        "Wave 1 used embedding displacement alone to label D0 as SUPPORTED. That omission is logged as a protocol deviation affecting the old D0 primary conclusion, with this Wave 1.1 execution as its completeness correction. No training authorization is created.",
    ])
    report(RESULTS / "R0_Wave1_Cross_Module_Diagnosis_v1.1.md", "R0 Wave 1 Cross-Module Diagnosis v1.1", [
        f"Evidence level: `{wave1.EVIDENCE_LEVEL}`.",
        "",
        f"D0 pooling: `{pooling_status}`; D0 mask/padding: `{mask_status}`. Embedding geometry sensitivity is `{hypothesis['embedding_geometry_effect']}`, whereas semantic retention/readout loss is `{hypothesis['semantic_retention_effect']}`. Therefore `{case_c}`.",
        "",
        "D1 is preserved: `KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED` from Wave 1. D3 formal hypotheses remain `INCONCLUSIVE`; no primary D3 result is changed.",
        f"Development-direction note only: in the existing pure-lateral table, R_linear_task has a higher ratio_to_null_q95 than R_full64 in {d3_note['linear_task_higher_count']}/{d3_note['representation_count']} representations and R_fixed_semantic in {d3_note['fixed_semantic_higher_count']}/{d3_note['representation_count']}. This descriptive direction does not upgrade any D3 status.",
        "",
        "Corrected scientific diagnosis: coordinate/geometry sensitivity and semantic information loss are separate claims. Wave 1.1 uses the frozen interpretation matrix rather than embedding L2/cosine displacement alone. No D2/D4 execution, RBR training, or new planner rollout was performed.",
    ])
    manifest = {
        "execution_id": "R0_WAVE1_1_D0_RETENTION_COMPLETION_V1", "completed_at_utc": now(), "freeze_verification": freeze,
        "checkpoint_sha256": checkpoint_locks, "stage7l_input_sha256": input_shas,
        "input_rows": 400, "scenario_groups": 80, "log_clusters": 79, "valid_length_values": sorted(set(valid_lengths.tolist())),
        "hypothesis_results": hypothesis, "protocol_deviation_id": "R0-WAVE1-D0-PRIMARY-METRIC-OMISSION-001",
        "training_authorization_modified": False, "large_embeddings_written": False,
        "result_files": sorted(NEW_RESULT_NAMES),
    }
    write_json(RESULTS / "r0_wave1_1_execution_manifest.json", manifest)
    write_json(RESULTS / "r0_wave1_1_command_ledger.json", {
        "execution_id": "R0_WAVE1_1_D0_RETENTION_COMPLETION_V1", "command_id": "R0_WAVE1_1_D0_001", "timestamp_utc": now(),
        "command": "python -m tools.stageR_execute_r0_wave1_1_d0_retention", "git_commit": WAVE1_COMMIT,
        "input_artifact_sha256": {"checkpoint_locks": checkpoint_locks, "stage7l_inputs": input_shas},
        "output_artifact_sha256": {path.name: sha256(path) for path in RESULTS.iterdir() if path.name in NEW_RESULT_NAMES or path.name == PROTOCOL_LOG.name},
        "exit_code": 0, "seed": 2026082601, "protocol_deviation_id": "R0-WAVE1-D0-PRIMARY-METRIC-OMISSION-001",
    })
    print(json.dumps({"status": "R0_WAVE1_1_COMPLETE", "pooling": pooling_status, "mask_padding": mask_status, "case_c": case_c}, indent=2))


if __name__ == "__main__":
    main()
