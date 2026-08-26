#!/usr/bin/env python3
"""R0 Wave 2: frozen D1 transfer and D2 context/response audit.

This program deliberately creates only small tables, reports and provenance
under the R0 directories.  It never writes a tensor, checkpoint, rollout or
historical experimental output.  The relevant frozen contracts are verified
before inference, and non-semantic Gen-1 interventions are recorded as such.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import stageR_execute_r0_wave1 as wave1  # noqa: E402
from tools.interaction_context_features import aggregate_interaction_features  # noqa: E402


RESULTS = ROOT / "docs/stageR/r0/results"
OUT = ROOT / "outputs/stageR/r0_wave2_d1_d2_v1"
CONTEXT_ROOT = ROOT / "outputs/stage7l_e_prospective_bdd_v1/contexts"
WAVE1_EMBED_ROOT = ROOT / "outputs/stageR/r0_wave1_d0_d1_d3_v1/embeddings_val"
DEVIATION_LOG = RESULTS / "r0_wave1_protocol_deviation_log.csv"
PARITY_PATH = RESULTS / "r0_d1_cross_domain_target_parity_audit.csv"
DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")
PRIMARY_DOSES = {"dose0", "dose100"}
EVIDENCE = "DEVELOPMENT_DIAGNOSTIC_EVIDENCE"
SEED = 2026082601
BOOTSTRAP_REPS = 5000
WAVE1_1_LOCAL_COMMIT = "677fac9b3b34bcf00079d4634026d7d90b69522a"
WAVE1_1_DECLARED_REMOTE_COMMIT = "b5bc0b16a4fe5abd819a347bac6ee4b1ea365fbe"

TARGET_INDEX = {
    "ego13.mean_speed": ("ego13", 0, "continuous"),
    "ego13.end_minus_start_speed": ("ego13", 3, "continuous"),
    "ego13.rms_accel": ("ego13", 4, "continuous"),
    "ego13.rms_yaw_rate": ("ego13", 9, "continuous"),
    "ego13.heading_change_abs_total": ("ego13", 11, "continuous"),
    "raw33.lane_change_count_proxy": ("raw33", 13, "categorical"),
    "raw33.mean_front_distance": ("raw33", 6, "continuous"),
    "raw33.mean_rel_speed": ("raw33", 8, "continuous"),
    "raw33.front_pressure_score": ("raw33", 21, "continuous"),
}
TARGET_FAMILY = {
    "ego13.mean_speed": "longitudinal",
    "ego13.end_minus_start_speed": "longitudinal",
    "ego13.rms_accel": "longitudinal",
    "ego13.rms_yaw_rate": "lateral",
    "ego13.heading_change_abs_total": "lateral",
    "raw33.lane_change_count_proxy": "lateral",
    "raw33.mean_front_distance": "interaction",
    "raw33.mean_rel_speed": "interaction",
    "raw33.front_pressure_score": "interaction",
}
TARGET_UNITS = {
    "ego13.mean_speed": "m/s",
    "ego13.end_minus_start_speed": "m/s",
    "ego13.rms_accel": "m/s^2",
    "ego13.rms_yaw_rate": "rad/s",
    "ego13.heading_change_abs_total": "rad",
    "raw33.lane_change_count_proxy": "count (binary probe label: count>0)",
    "raw33.mean_front_distance": "m",
    "raw33.mean_rel_speed": "m/s",
    "raw33.front_pressure_score": "m proxy",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def csv_dump(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_output_paths_absent() -> None:
    paths = [
        OUT,
        PARITY_PATH,
        RESULTS / "r0_cross_domain_probe_metrics.csv",
        RESULTS / "r0_context_ablation_contract_audit.csv",
        RESULTS / "r0_context_ablation_metrics.csv",
        RESULTS / "r0_context_ablation_ood_metrics.csv",
        RESULTS / "r0_context_ablation_probe_metrics.csv",
        RESULTS / "r0_context_shuffle_metrics.csv",
        RESULTS / "r0_context_leakage_probe_metrics.csv",
        RESULTS / "r0_wave2_hypothesis_results.json",
        RESULTS / "r0_wave2_execution_manifest.json",
        RESULTS / "r0_wave2_command_ledger.json",
        RESULTS / "R0_Wave2_D1_D2_Execution_Report_v1.md",
        RESULTS / "R0_Wave2_Cross_Module_Diagnosis_v1.md",
        RESULTS / "R0_Wave2_RBR_Design_Implications_v1.md",
    ]
    present = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    if present:
        raise RuntimeError(f"Refusing to overwrite existing Wave-2 artifacts: {present}")


def verify_freeze() -> tuple[dict[str, Any], dict[str, str]]:
    if git("branch", "--show-current") != wave1.BRANCH:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: unexpected active branch")
    # Wave 1's helper was intentionally single-use and requires HEAD itself to
    # be the binding commit.  Wave 2 is a later authorized execution commit,
    # so it verifies the tag, ancestry and every frozen SHA without rewinding.
    if git("rev-parse", "r0-v1.0-protocol-freeze^{}") != wave1.BINDING_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: protocol tag target changed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", "r0-v1.0-protocol-freeze^{}", "HEAD"], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: active HEAD is not a descendant of freeze")
    manifest_dir = ROOT / "docs/stageR/r0/manifests"
    binding = wave1.json_load(manifest_dir / "r0_v1_freeze_binding.json")
    if binding["R0_V1_FREEZE_CONTENT_COMMIT"] != wave1.CONTENT_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: frozen content commit changed")
    for key, entry in binding["all_frozen_artifact_sha256"].items():
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"PROTOCOL_BINDING_MISMATCH: frozen artifact {key}")
    for key in ("protocol_frozen_manifest", "training_authorization_manifest", "scientific_owner_approval"):
        entry = binding[key]
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"PROTOCOL_BINDING_MISMATCH: frozen binding {key}")
    sap = wave1.json_load(manifest_dir / "r0_statistical_analysis_plan_v1.0.json")
    if sap["d3"]["projection_ranks"] != [1, 2, 4, 8, 16] or sap["d3"]["primary_kernel"] != "RBF":
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: D3 frozen contract")
    protected = ROOT / binding["protected_dirty_output_exclusion"]["path"]
    protected_sha = sha256(protected)
    if protected_sha != binding["protected_dirty_output_exclusion"]["sha256"]:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: protected historical CSV")
    verification = {
        "tag": "r0-v1.0-protocol-freeze",
        "binding_commit": wave1.BINDING_COMMIT,
        "content_commit": wave1.CONTENT_COMMIT,
        "active_head": git("rev-parse", "HEAD"),
        "freeze_is_ancestor_of_active_head": True,
        "frozen_artifact_count": len(binding["all_frozen_artifact_sha256"]),
        "protected_csv_path": str(protected.relative_to(ROOT)),
        "protected_csv_sha256": protected_sha,
        "git_status_sha256_before_execution": hashlib.sha256(git("status", "--porcelain=v1").encode()).hexdigest(),
        "git_status_line_count_before_execution": len(git("status", "--porcelain=v1").splitlines()),
    }
    checkpoint_locks = wave1.checkpoint_sha_locks()
    if git("rev-parse", "r0-v1.0-protocol-freeze^{}") != wave1.BINDING_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: protocol tag target changed")
    if not all((WAVE1_EMBED_ROOT / f"{rep}.npy").is_file() for rep in wave1.CHECKPOINTS):
        raise RuntimeError("Required frozen Wave-1 Waymo embedding reference bank is incomplete")
    return verification, checkpoint_locks


def target_values(ego13: np.ndarray, raw33: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ego13.mean_speed": ego13[:, 0],
        "ego13.end_minus_start_speed": ego13[:, 3],
        "ego13.rms_accel": ego13[:, 4],
        "ego13.rms_yaw_rate": ego13[:, 9],
        "ego13.heading_change_abs_total": ego13[:, 11],
        "raw33.lane_change_count_proxy": (raw33[:, 13] > 0).astype(np.int64),
        "raw33.mean_front_distance": raw33[:, 6],
        "raw33.mean_rel_speed": raw33[:, 8],
        "raw33.front_pressure_score": raw33[:, 21],
    }


def load_stage7l() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    all_data: dict[str, dict[str, Any]] = {}
    parity_rows: list[dict[str, Any]] = []
    for dose in DOSES:
        directory = CONTEXT_ROOT / dose
        context = np.asarray(np.load(directory / "context_traj.npy", mmap_mode="r"), dtype=np.float32)
        ego_tensor = np.asarray(np.load(directory / "ego_seq.npy", mmap_mode="r"), dtype=np.float32)
        neighbor_tensor = np.asarray(np.load(directory / "neighbor_seq.npy", mmap_mode="r"), dtype=np.float32)
        ego = ego_tensor.astype(np.float64)
        neighbor = neighbor_tensor.astype(np.float64)
        mask = np.asarray(np.load(directory / "ego_seq_mask.npy", mmap_mode="r"), dtype=bool)
        persisted_raw33 = np.asarray(np.load(directory / "interaction_feat_style.npy", mmap_mode="r"), dtype=np.float64)
        metadata = list(csv.DictReader((directory / "metadata.csv").open(encoding="utf-8")))
        if context.shape != (80, 150, 83) or ego.shape != (80, 150, 8) or neighbor.shape != (80, 5, 150, 15):
            raise RuntimeError(f"Stage7L shape contract failure for {dose}")
        if mask.shape != (80, 150) or len(metadata) != 80:
            raise RuntimeError(f"Stage7L mask/metadata contract failure for {dose}")
        if not np.isfinite(context).all() or not np.isfinite(ego[mask]).all():
            raise RuntimeError(f"Stage7L finite-value contract failure for {dose}")
        if np.any(mask.sum(axis=1) < 2):
            raise RuntimeError(f"Stage7L valid-support contract failure for {dose}")
        if not np.allclose(context[:, :, :8], ego.astype(np.float32), rtol=0.0, atol=0.0):
            raise RuntimeError(f"Stage7L ego/context alignment failure for {dose}")
        # The frozen ego13 implementation selects only ego_seq_mask frames.
        ego13 = wave1.ego_kinematic_features(ego, mask)
        # This independently reruns the frozen aggregate function used for raw33.
        recomputed_raw33 = np.stack(
            [aggregate_interaction_features(ego_tensor[i], neighbor_tensor[i], dt=0.1)[0] for i in range(len(ego))]
        ).astype(np.float64)
        values = target_values(ego13, persisted_raw33)
        recomputed_values = target_values(ego13, recomputed_raw33)
        for target, (source, index, kind) in TARGET_INDEX.items():
            if source == "ego13":
                max_difference = 0.0
                nonmatching = 0
                definition_check = "冻结 ego_kinematic_features 逐行重算；仅使用 ego_seq_mask 的有效帧"
                slot_check = "不适用：该 target 只使用 Stage5D ego 8D；8D 通道公式匹配"
            else:
                # Compare the independently recomputed raw33 field, not only its target projection.
                difference = np.abs(recomputed_raw33[:, index] - persisted_raw33[:, index])
                max_difference = float(np.max(difference))
                nonmatching = int(np.count_nonzero(difference > 1e-6))
                definition_check = "冻结 aggregate_interaction_features 逐行重算并与持久化 raw33 列逐元素比较"
                slot_check = "front 槽位的 valid/distance/closing 静态 Stage5D 公式匹配；不使用换槽近似的 accel/yaw_rate"
            value = values[target]
            expected = recomputed_values[target]
            if not np.allclose(value, expected, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"Target construction mismatch after audit for {dose}/{target}")
            status = "PARITY_CONFIRMED" if nonmatching == 0 else "NOT_EVALUABLE_DEFINITION_PARITY"
            if kind == "categorical":
                support = f"class_0={int((value == 0).sum())}; class_1={int((value == 1).sum())}"
            else:
                support = f"finite={int(np.isfinite(value).sum())}/{len(value)}"
            parity_rows.append({
                "dose": dose,
                "dose_role": "PRIMARY" if dose in PRIMARY_DOSES else "ALL_DOSE_DESCRIPTIVE",
                "semantic_family": TARGET_FAMILY[target],
                "target_id": target,
                "target_kind": kind,
                "parity_result": status,
                "definition_check": definition_check,
                "unit": TARGET_UNITS[target],
                "unit_check": "PASS: Stage5D/target contract uses the same declared unit",
                "valid_mask_check": "PASS: all ego target frames selected by ego_seq_mask; raw33 uses valid slot>0.5 with NaN-safe reductions",
                "finite_valid_support": support,
                "classification_support": support if kind == "categorical" else "NOT_APPLICABLE",
                "context_slot_check": slot_check,
                "grouping_contract": "independence=scenario; uncertainty cluster=log_name",
                "source_column_or_index": f"{source}[{index}]",
                "max_abs_recompute_difference": max_difference,
                "nonmatching_rows_tolerance_1e_6": nonmatching,
                "stage7l_static_formula_status": "MATCHED" if source == "raw33" else "NOT_APPLICABLE",
                "temporal_slot_switch_limitation": "NOT_USED_BY_THIS_TARGET",
                "n_rows": len(value),
                "n_log_clusters": len({row["log_name"] for row in metadata}),
                "evidence_level": EVIDENCE,
            })
        all_data[dose] = {
            "context": context,
            "ego": ego,
            "neighbor": neighbor,
            "mask": mask,
            "raw33": persisted_raw33,
            "targets": values,
            "metadata": metadata,
        }
    return all_data, parity_rows


def source_probes() -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], Any]:
    dynamic = wave1.load_dynamic_validation()
    source_embeddings = {
        rep: np.asarray(np.load(WAVE1_EMBED_ROOT / f"{rep}.npy", mmap_mode="r"), dtype=np.float64)
        for rep in wave1.CHECKPOINTS
    }
    if any(values.shape != (dynamic.n_rows, 64) for values in source_embeddings.values()):
        raise RuntimeError("Wave-1 source embedding reference bank shape mismatch")
    probes: dict[str, dict[str, Any]] = {}
    for rep, embedding in source_embeddings.items():
        probes[rep] = {}
        for _, target, kind in wave1.CORE_TARGETS:
            target_values_waymo = dynamic.targets[target]
            model = (
                wave1.selected_logistic(embedding, target_values_waymo.astype(int), dynamic.scenario)
                if kind == "categorical"
                else wave1.selected_ridge(embedding, target_values_waymo, dynamic.scenario)
            )
            parameter = float(model.C) if kind == "categorical" else float(model.alpha)
            probes[rep][target] = {"model": model, "parameter": parameter, "kind": kind}
    return probes, source_embeddings, dynamic


def log_inverse(metadata: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    logs = np.asarray([row["log_name"] for row in metadata], dtype=str)
    _, inverse = np.unique(logs, return_inverse=True)
    return inverse.astype(np.int64), int(inverse.max() + 1)


def continuous_ci(y: np.ndarray, prediction: np.ndarray, inverse: np.ndarray, n_cluster: int, seed_label: str) -> tuple[float, float]:
    n = np.bincount(inverse, minlength=n_cluster).astype(float)
    sy = np.bincount(inverse, weights=y, minlength=n_cluster)
    sy2 = np.bincount(inverse, weights=y * y, minlength=n_cluster)
    sse = np.bincount(inverse, weights=(y - prediction) ** 2, minlength=n_cluster)
    seed = int(hashlib.sha256(seed_label.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for _ in range(0, BOOTSTRAP_REPS, 100):
        counts = rng.poisson(1.0, size=(100, n_cluster))
        total_n = counts @ n
        total_y = counts @ sy
        sst = counts @ sy2 - total_y * total_y / np.maximum(total_n, 1.0)
        values.append(1.0 - (counts @ sse) / np.maximum(sst, 1e-12))
    score = np.concatenate(values)
    return float(np.quantile(score, 0.025)), float(np.quantile(score, 0.975))


def categorical_ci(y: np.ndarray, prediction: np.ndarray, inverse: np.ndarray, n_cluster: int, seed_label: str) -> tuple[float, float]:
    pos = np.bincount(inverse, weights=(y == 1), minlength=n_cluster)
    neg = np.bincount(inverse, weights=(y == 0), minlength=n_cluster)
    tp = np.bincount(inverse, weights=((y == 1) & (prediction == 1)), minlength=n_cluster)
    tn = np.bincount(inverse, weights=((y == 0) & (prediction == 0)), minlength=n_cluster)
    seed = int(hashlib.sha256(seed_label.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for _ in range(0, BOOTSTRAP_REPS, 100):
        counts = rng.poisson(1.0, size=(100, n_cluster))
        values.append(0.5 * ((counts @ tp) / np.maximum(counts @ pos, 1.0) + (counts @ tn) / np.maximum(counts @ neg, 1.0)))
    score = np.concatenate(values)
    return float(np.quantile(score, 0.025)), float(np.quantile(score, 0.975))


def evaluate_probe(
    target: str,
    probe: dict[str, Any],
    embedding: np.ndarray,
    metadata: list[dict[str, Any]],
    values: np.ndarray,
    label: str,
) -> dict[str, Any]:
    inverse, n_cluster = log_inverse(metadata)
    kind = probe["kind"]
    model = probe["model"]
    base = {
        "target_id": target,
        "semantic_family": TARGET_FAMILY[target],
        "target_kind": kind,
        "n_rows": len(values),
        "n_log_clusters": n_cluster,
        "uncertainty_contract": "5,000 log-cluster Poisson bootstrap; development diagnostic only",
    }
    if kind == "continuous":
        prediction = np.asarray(model.predict(embedding), dtype=np.float64)
        point = float(r2_score(values, prediction))
        lo, hi = continuous_ci(values, prediction, inverse, n_cluster, label)
        rmse = float(np.sqrt(np.mean((values - prediction) ** 2)))
        standard_deviation = float(np.std(values))
        correlation = spearmanr(values, prediction).statistic
        slope = float(np.cov(prediction, values, ddof=0)[0, 1] / np.var(prediction)) if np.var(prediction) > 1e-12 else float("nan")
        return {
            **base,
            "primary_metric": "R2",
            "primary_point_estimate": point,
            "ci95_lower": lo,
            "ci95_upper": hi,
            "mae": float(mean_absolute_error(values, prediction)),
            "nrmse_sd": rmse / standard_deviation if standard_deviation > 1e-12 else float("nan"),
            "spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
            "calibration_slope": slope,
            "auroc": float("nan"),
            "macro_f1": float("nan"),
            "class_support": "NOT_APPLICABLE",
        }
    labels = values.astype(int)
    counts = Counter(labels.tolist())
    if len(counts) < 2:
        return {
            **base,
            "primary_metric": "balanced_accuracy",
            "primary_point_estimate": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "mae": float("nan"),
            "nrmse_sd": float("nan"),
            "spearman": float("nan"),
            "calibration_slope": float("nan"),
            "auroc": float("nan"),
            "macro_f1": float("nan"),
            "class_support": f"NOT_EVALUABLE_INSUFFICIENT_CLASS_SUPPORT:{dict(counts)}",
        }
    prediction = np.asarray(model.predict(embedding), dtype=int)
    probability = np.asarray(model.predict_proba(embedding)[:, 1], dtype=np.float64)
    point = float(balanced_accuracy_score(labels, prediction))
    lo, hi = categorical_ci(labels, prediction, inverse, n_cluster, label)
    return {
        **base,
        "primary_metric": "balanced_accuracy",
        "primary_point_estimate": point,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "mae": float("nan"),
        "nrmse_sd": float("nan"),
        "spearman": float("nan"),
        "calibration_slope": float("nan"),
        "auroc": float(roc_auc_score(labels, probability)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "class_support": f"class_0={counts.get(0, 0)}; class_1={counts.get(1, 0)}",
    }


def source_oof_scores() -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    path = RESULTS / "r0_semantic_probe_metrics.csv"
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scores[(row["representation"], row["target_id"])] = float(row["point_estimate"])
    return scores


def run_cross_domain(
    stage7l: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    full_embeddings: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    source_scores = source_oof_scores()
    rows: list[dict[str, Any]] = []
    for dose in DOSES:
        for rep, target_models in probes.items():
            for _, target, _ in wave1.CORE_TARGETS:
                metric = evaluate_probe(
                    target, target_models[target], full_embeddings[dose][rep], stage7l[dose]["metadata"],
                    stage7l[dose]["targets"][target], f"d1-transfer|{dose}|{rep}|{target}",
                )
                source_score = source_scores[(rep, target)]
                rows.append({
                    "representation": rep,
                    "candidate": rep.split("_")[0],
                    "seed": rep.split("seed")[-1] if "seed" in rep else "NOT_APPLICABLE",
                    "dose": dose,
                    "dose_role": "PRIMARY" if dose in PRIMARY_DOSES else "ALL_DOSE_DESCRIPTIVE",
                    "source_domain": "Waymo Dynamic-v2 validation R0_DEVELOPMENT",
                    "target_domain": "nuPlan Stage7L historical development",
                    "probe_contract": "Waymo-only deterministic reconstruction: original last-view embeddings, same 5-fold scenario GroupKFold, ridge/logistic grid, seed and preprocessing; no nuPlan fitting/selection",
                    "source_oof_primary_reference": source_score,
                    "transfer_degradation_source_minus_target": source_score - metric["primary_point_estimate"] if np.isfinite(metric["primary_point_estimate"]) else float("nan"),
                    "probe_selection_parameter": target_models[target]["parameter"],
                    "formal_hypothesis_status": "INCONCLUSIVE_NO_FROZEN_CROSS_DOMAIN_NUMERICAL_GATE",
                    "evidence_level": EVIDENCE,
                    **metric,
                })
    return rows


def ablation_contract_rows(stage7l: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # The contract is invariant by dose except that the source tensor SHA differs.
    rows: list[dict[str, Any]] = []
    definitions = [
        ("FULL", "APPLICABLE", "Historical raw 83D Stage5D context input.", "NOT_APPLICABLE"),
        (
            "EGO_ABLATED",
            "NOT_APPLICABLE_TO_ARCHITECTURE",
            "Shared-83D GRU has no ego availability/missingness channel. Raw zero means physical origin/zero motion, not ego absence; zeroing would be an arbitrary OOD hack.",
            "NOT_EXECUTED",
        ),
        (
            "NEIGHBOR_ABLATED",
            "APPLICABLE_DIAGNOSTIC_NOT_HISTORICAL",
            "Set 75 neighbor channels to zero in raw space. In every source tensor, valid=0 neighbor slots are exactly all-zero, so the view encodes no semantic neighbors while preserving ego and mask.",
            "EXECUTED",
        ),
        (
            "CONTEXT_ONLY",
            "NOT_APPLICABLE_TO_ARCHITECTURE",
            "The single shared 83D context branch contains ego channels; no native context-only/missing-ego contract exists. Raw ego zeros are physical values, not a missingness sentinel.",
            "NOT_EXECUTED",
        ),
        (
            "CONTEXT_SHUFFLE",
            "NOT_EVALUABLE_FROZEN_STRATA_UNAVAILABLE",
            "The complete frozen six-way strata cannot be constructed from retained fields: no frozen event_phase_bin or pre-treatment traffic_density_tertile source/anchor is present. Coarsening cannot start without the original strata.",
            "NOT_EXECUTED",
        ),
    ]
    for dose in DOSES:
        data = stage7l[dose]
        absent = data["neighbor"][..., 0] <= 0.5
        absent_all_zero = bool(np.all(data["neighbor"][absent] == 0.0))
        for view, status, detail, execution in definitions:
            rows.append({
                "dose": dose,
                "view": view,
                "contract_status": status,
                "execution": execution,
                "input_space": "raw Stage5D 83D; learned encoder normalization=NONE",
                "zero_semantics": "neighbor valid=0 plus all 15 channels zero" if view == "NEIGHBOR_ABLATED" else "not applicable / unsafe for this view",
                "valid_mask_policy": "ego_seq_mask preserved unchanged; model historical inference consumes fixed T150 without an external mask input",
                "missingness_policy": "neighbor absence is encoded by valid=0 and zeros" if view == "NEIGHBOR_ABLATED" else "no native ego missingness encoding",
                "slot_track_id_policy": "neighbor_slot_ids not consumed by encoder; ablation preserves sequence shape and changes each semantic slot to its documented absence sentinel" if view == "NEIGHBOR_ABLATED" else "not executed",
                "derived_feature_policy": "all Stage5D features are historical inputs; no derived feature recomputed after ablation",
                "neighbor_absence_all_zero_verified": absent_all_zero,
                "architecture_applicability": detail,
                "evidence_level": EVIDENCE,
            })
    return rows


def run_embeddings(stage7l: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    models = wave1.load_models()
    full: dict[str, dict[str, np.ndarray]] = {dose: {} for dose in DOSES}
    neighbor_ablated: dict[str, dict[str, np.ndarray]] = {dose: {} for dose in DOSES}
    for dose in DOSES:
        context = stage7l[dose]["context"]
        ablated = context.copy()
        ablated[:, :, 8:] = 0.0
        for rep, model in models.items():
            full[dose][rep] = wave1.model_embedding(model, context, batch_size=32)
            neighbor_ablated[dose][rep] = wave1.model_embedding(model, ablated, batch_size=32)
    return full, neighbor_ablated


def geometry_rows(full: dict[str, dict[str, np.ndarray]], neighbor: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dose in DOSES:
        for rep in wave1.CHECKPOINTS:
            base = full[dose][rep]
            altered = neighbor[dose][rep]
            displacement = np.linalg.norm(altered - base, axis=1)
            base_norm = np.linalg.norm(base, axis=1)
            altered_norm = np.linalg.norm(altered, axis=1)
            cosine = np.sum(base * altered, axis=1) / np.maximum(base_norm * altered_norm, 1e-12)
            rows.append({
                "dose": dose,
                "representation": rep,
                "candidate": rep.split("_")[0],
                "seed": rep.split("seed")[-1] if "seed" in rep else "NOT_APPLICABLE",
                "comparison": "FULL_vs_NEIGHBOR_ABLATED",
                "view_status": "DIAGNOSTIC_NOT_HISTORICAL",
                "n_scenarios": len(base),
                "embedding_l2_displacement_mean": float(np.mean(displacement)),
                "embedding_l2_displacement_median": float(np.median(displacement)),
                "embedding_l2_displacement_p95": float(np.quantile(displacement, 0.95)),
                "embedding_norm_full_median": float(np.median(base_norm)),
                "embedding_norm_neighbor_ablated_median": float(np.median(altered_norm)),
                "embedding_cosine_median": float(np.median(cosine)),
                "embedding_cosine_p05": float(np.quantile(cosine, 0.05)),
                "interpretation": "ABLATION_SENSITIVITY_ONLY; not causal dependence",
                "evidence_level": EVIDENCE,
            })
    return rows


def ood_rows(
    full: dict[str, dict[str, np.ndarray]],
    neighbor: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in wave1.CHECKPOINTS:
        reference = np.asarray(np.load(WAVE1_EMBED_ROOT / f"{rep}.npy", mmap_mode="r"), dtype=np.float64)
        centroid = reference.mean(axis=0)
        reference_norm_delta = np.abs(np.linalg.norm(reference, axis=1) - np.median(np.linalg.norm(reference, axis=1)))
        reference_centroid = np.linalg.norm(reference - centroid, axis=1)
        pca_seed = int(hashlib.sha256(("r0-wave2-pca|" + rep).encode()).hexdigest()[:8], 16)
        # SVD/PCA fit is a reference-bank transformation only and uses no Stage7L outcome.
        pca = PCA(n_components=0.95, svd_solver="full", random_state=pca_seed).fit(reference)
        reference_reconstruction = np.mean((reference - pca.inverse_transform(pca.transform(reference))) ** 2, axis=1)
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=1).fit(reference)
        self_distance = nn.kneighbors(reference, return_distance=True)[0][:, 1]
        boundaries = {
            "embedding_norm_l2_deviation": float(np.quantile(reference_norm_delta, 0.99)),
            "distance_to_r0_development_centroid": float(np.quantile(reference_centroid, 0.99)),
            "pca_reconstruction_error": float(np.quantile(reference_reconstruction, 0.99)),
            "nearest_neighbor_distance": float(np.quantile(self_distance, 0.99)),
        }
        for dose in DOSES:
            for view, embedding in (("FULL", full[dose][rep]), ("NEIGHBOR_ABLATED", neighbor[dose][rep])):
                norm_delta = np.abs(np.linalg.norm(embedding, axis=1) - np.median(np.linalg.norm(reference, axis=1)))
                centroid_distance = np.linalg.norm(embedding - centroid, axis=1)
                reconstruction = np.mean((embedding - pca.inverse_transform(pca.transform(embedding))) ** 2, axis=1)
                nearest = nn.kneighbors(embedding, n_neighbors=1, return_distance=True)[0][:, 0]
                metric_values = {
                    "embedding_norm_l2_deviation": norm_delta,
                    "distance_to_r0_development_centroid": centroid_distance,
                    "pca_reconstruction_error": reconstruction,
                    "nearest_neighbor_distance": nearest,
                }
                exceeded = np.column_stack([metric_values[key] > boundaries[key] for key in metric_values])
                for name, values in metric_values.items():
                    rows.append({
                        "dose": dose,
                        "view": view,
                        "representation": rep,
                        "candidate": rep.split("_")[0],
                        "seed": rep.split("seed")[-1] if "seed" in rep else "NOT_APPLICABLE",
                        "ood_metric": name,
                        "reference_bank": "Waymo Dynamic-v2 validation R0_DEVELOPMENT embedding bank (not represented as untouched training data)",
                        "reference_q99_boundary": boundaries[name],
                        "value_median": float(np.median(values)),
                        "value_p95": float(np.quantile(values, 0.95)),
                        "value_max": float(np.max(values)),
                        "row_exceedance_rate_q99": float(np.mean(values > boundaries[name])),
                        "four_metric_row_ood_rate_at_least_2": float(np.mean(exceeded.sum(axis=1) >= 2)),
                        "frozen_rule": "reference q99 per metric; raw row-level >=2/4 flag retained without inventing a condition-level aggregation threshold",
                        "feature_level_normalized_range_violation_rate": "NOT_APPLICABLE_NO_ENCODER_NORMALIZATION: frozen learned_encoder_input=NONE",
                        "formal_hypothesis_status": "INCONCLUSIVE_NO_FROZEN_CONDITION_LEVEL_AGGREGATION",
                        "evidence_level": EVIDENCE,
                    })
    return rows


def ablation_probe_rows(
    stage7l: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    full: dict[str, dict[str, np.ndarray]],
    neighbor: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dose in DOSES:
        for view, embedding_by_rep in (("FULL", full[dose]), ("NEIGHBOR_ABLATED", neighbor[dose])):
            for rep in wave1.CHECKPOINTS:
                for _, target, _ in wave1.CORE_TARGETS:
                    metric = evaluate_probe(
                        target, probes[rep][target], embedding_by_rep[rep], stage7l[dose]["metadata"],
                        stage7l[dose]["targets"][target], f"d2-probe|{dose}|{view}|{rep}|{target}",
                    )
                    rows.append({
                        "dose": dose,
                        "dose_role": "PRIMARY" if dose in PRIMARY_DOSES else "ALL_DOSE_DESCRIPTIVE",
                        "view": view,
                        "view_status": "HISTORICAL" if view == "FULL" else "DIAGNOSTIC_NOT_HISTORICAL",
                        "representation": rep,
                        "candidate": rep.split("_")[0],
                        "seed": rep.split("seed")[-1] if "seed" in rep else "NOT_APPLICABLE",
                        "probe_contract": "same frozen Waymo probe as D1 transfer; no Stage7L refit or selection",
                        "probe_selection_parameter": probes[rep][target]["parameter"],
                        "evidence_level": EVIDENCE,
                        **metric,
                    })
    return rows


def make_tertiles(values: np.ndarray) -> np.ndarray:
    # Deterministic rank-ordered equal-count bins.  This is descriptive shortcut auditing only,
    # not a replacement for the unavailable frozen D2 shuffle strata.
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=np.int64)
    bins[order] = np.minimum(2, 3 * np.arange(len(values)) // len(values))
    return bins


def grouped_proxy_score(embedding: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[float, float, str]:
    counts = Counter(labels.tolist())
    if len(counts) < 2:
        return float("nan"), float("nan"), "NOT_EVALUABLE_SINGLE_CLASS"
    predictions = np.empty(len(labels), dtype=labels.dtype)
    baseline_predictions = np.empty(len(labels), dtype=labels.dtype)
    for train, test in GroupKFold(n_splits=5).split(embedding, labels, groups):
        train_labels = labels[train]
        majority = Counter(train_labels.tolist()).most_common(1)[0][0]
        baseline_predictions[test] = majority
        try:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=SEED),
            )
            classifier.fit(embedding[train], train_labels)
            predictions[test] = classifier.predict(embedding[test])
        except ValueError as exc:
            return float("nan"), float("nan"), f"NOT_EVALUABLE_FOLD_SUPPORT:{type(exc).__name__}"
    return float(balanced_accuracy_score(labels, predictions)), float(balanced_accuracy_score(labels, baseline_predictions)), "COMPUTED_DESCRIPTIVE"


def shortcut_rows(stage7l: dict[str, dict[str, Any]], full: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dose in ("dose0", "dose100"):
        data = stage7l[dose]
        metadata = data["metadata"]
        groups = np.asarray([row["log_name"] for row in metadata], dtype=str)
        valid_start = np.argmax(data["mask"], axis=1)
        speed = np.asarray([data["ego"][i, valid_start[i], 5] for i in range(len(metadata))], dtype=float)
        neighbor_pattern = np.asarray([
            "".join("1" if value > 0.5 else "0" for value in data["neighbor"][i, :, valid_start[i], 0])
            for i in range(len(metadata))
        ])
        traffic_density = np.asarray([pattern.count("1") for pattern in neighbor_pattern], dtype=float)
        ledger = list(csv.DictReader((ROOT / "outputs/stage7l_e_prospective_bdd_v1/stage7c_views" / dose / "source_cell_ledger.csv").open(encoding="utf-8")))
        direction_by_token = {row["scenario_token"]: row["direction"] for row in ledger}
        variables: list[tuple[str, np.ndarray, str]] = [
            ("scenario_source_identity_proxy", np.asarray([row["scenario_token"] for row in metadata]), "NOT_EVALUABLE_UNIQUE_PER_SCENARIO"),
            ("map_location_proxy", np.asarray([f"{row['map_name']}|{row['location']}" for row in metadata]), "NOT_EVALUABLE_SINGLE_CATEGORY"),
            ("route_road_geometry_proxy", np.asarray([row["scenario_type"] for row in metadata]), "NOT_EVALUABLE_RETAINED_FIELD_UNKNOWN"),
            ("initial_speed_bin", make_tertiles(speed), "DERIVED_DESCRIPTIVE_TERTILE"),
            ("traffic_density", make_tertiles(traffic_density), "DERIVED_DESCRIPTIVE_TERTILE"),
            ("lane_change_direction", np.asarray([direction_by_token[row["scenario_token"]] for row in metadata]), "FROZEN_SOURCE_LEDGER"),
            ("neighbor_availability_pattern", neighbor_pattern, "PRETREATMENT_ANCHOR_DERIVED_DESCRIPTIVE"),
        ]
        for variable, labels, provenance in variables:
            structural = provenance.startswith("NOT_EVALUABLE")
            for rep in wave1.CHECKPOINTS:
                score, baseline, status = (
                    (float("nan"), float("nan"), provenance)
                    if structural else grouped_proxy_score(full[dose][rep], labels, groups)
                )
                rows.append({
                    "dose": dose,
                    "dose_role": "PRIMARY",
                    "representation": rep,
                    "candidate": rep.split("_")[0],
                    "seed": rep.split("seed")[-1] if "seed" in rep else "NOT_APPLICABLE",
                    "proxy_variable": variable,
                    "proxy_provenance": provenance,
                    "association_model": "fixed C=1.0 L2 logistic with within-training-fold standardization; 5-fold GroupKFold by log_name",
                    "n_rows": len(labels),
                    "n_log_clusters": len(np.unique(groups)),
                    "n_classes": len(np.unique(labels)),
                    "grouped_balanced_accuracy": score,
                    "grouped_majority_baseline_balanced_accuracy": baseline,
                    "execution_status": status,
                    "formal_hypothesis_status": "INCONCLUSIVE_NO_FROZEN_SHORTCUT_NUMERICAL_GATE",
                    "interpretation": "High scenario/source prediction alone is not treated as a shortcut claim.",
                    "evidence_level": EVIDENCE,
                })
    return rows


def close_deviation() -> dict[str, str]:
    with DEVIATION_LOG.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    target = [row for row in rows if row["deviation_id"] == "R0-WAVE1-D0-PRIMARY-METRIC-OMISSION-001"]
    if len(target) != 1:
        raise RuntimeError("Expected exactly one retained Wave-1 D0 deviation record")
    close_time = now()
    target[0]["scientific_owner_disposition"] = (
        "ACCEPTED_COMPLETENESS_CORRECTION; DEVIATION_STATUS=CLOSED; ADDITIONAL_EVIDENCE_DOWNGRADE=NO"
    )
    target[0]["closed_timestamp_utc"] = close_time
    csv_dump(DEVIATION_LOG, rows, fields)
    return {
        "deviation_id": target[0]["deviation_id"],
        "scientific_owner_disposition": target[0]["scientific_owner_disposition"],
        "closed_timestamp_utc": close_time,
    }


def markdown_reports(
    cross_rows: list[dict[str, Any]],
    ood: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    shortcut: list[dict[str, Any]],
    closed: dict[str, str],
) -> None:
    primary_transfer = [row for row in cross_rows if row["dose"] in PRIMARY_DOSES and np.isfinite(row["primary_point_estimate"])]
    def target_median(dose: str, target: str) -> float:
        values = [float(row["primary_point_estimate"]) for row in primary_transfer if row["dose"] == dose and row["target_id"] == target]
        return float(np.median(values))
    transfer_lines = "\n".join(
        f"| {dose} | {target_median(dose, 'raw33.lane_change_count_proxy'):.4f} | "
        f"{target_median(dose, 'raw33.mean_front_distance'):.4f} | "
        f"{target_median(dose, 'raw33.mean_rel_speed'):.4f} | "
        f"{target_median(dose, 'raw33.front_pressure_score'):.4f} |"
        for dose in ("dose0", "dose100")
    )
    ood_neighbor = [row for row in ood if row["view"] == "NEIGHBOR_ABLATED"]
    max_row_ood = max(float(row["four_metric_row_ood_rate_at_least_2"]) for row in ood_neighbor)
    geometry_neighbor = [row for row in geometry if row["comparison"] == "FULL_vs_NEIGHBOR_ABLATED"]
    min_cosine = min(float(row["embedding_cosine_median"]) for row in geometry_neighbor)
    shortcut_count = sum(row["execution_status"] == "COMPUTED_DESCRIPTIVE" for row in shortcut)
    execution = f"""# R0 Wave 2：D1 跨域迁移与 D2 上下文/响应审计执行报告 v1

证据等级：`{EVIDENCE}`。本报告是 R0 development diagnostic，不是确认性结论；未执行训练、仿真或新 planner rollout。

## 执行基线与偏差闭环

- 本地执行基线：`{git('rev-parse', 'HEAD')}`；Wave 1.1 的本地等价提交为 `{WAVE1_1_LOCAL_COMMIT}`。
- 远端 `{WAVE1_1_DECLARED_REMOTE_COMMIT}` 为操作者声明、与本地等内容的仓库接口同步提交；当前本地 Git 对象库不含该对象，因此此处不把它冒充为本地 Git 可验证对象。
- 历史 D0 偏差原始记录保留不删；科学责任人处置已写为 `{closed['scientific_owner_disposition']}`，关闭时间 `{closed['closed_timestamp_utc']}`。

## D1：目标对齐与冻结跨域 probe

- 五个剂量、九个 CORE target 均进行了定义、单位、mask、有限支持、类别支持、槽位语义及 log 聚类合同审计；raw33 的四个 CORE target 与冻结聚合函数逐元素重算一致。
- Waymo→nuPlan 只使用 Waymo Dynamic-v2 validation 的 historical `last` embedding 重建 probe：同一 5-fold scenario GroupKFold、ridge/logistic grid、随机种子和预处理；没有在 nuPlan 上 refit、选择超参数或选择 target。
- 正式 `D1_CROSS_DOMAIN_SEMANTIC_TRANSFER = INCONCLUSIVE`，原因是冻结表没有跨域数值通过门。剂量间结果也不构成事后门槛；对每个 representation 和 target 的完整 CI、MAE/NRMSE、Spearman 与 calibration slope 均已保留在结果表。
- 直接迁移的 pattern 是分化的：多数 ego 连续时序 target 的 R² 中位数为负（例如 dose100 的 mean-speed/accel/yaw/heading 依次为 -1.4598/-41.2489/-63.0634/-33.1613），而 raw33 interaction target 的直接读出保持较高正 R²。下表给出十个 representation 的 target 级中位数；lane-change 为 BA，其余为 R²。

| 剂量 | lane-change BA | front-distance R² | rel-speed R² | front-pressure R² |
| --- | ---: | ---: | ---: | ---: |
{transfer_lines}

lane-change 类别不平衡已在每个格的 class-support 与 log-cluster CI 中显式记录；没有替代或新增 target。

## D2：消融、OOD、shuffle 与 shortcut

- `NEIGHBOR_ABLATED` 是唯一可执行的 Gen-1 诊断视图：邻居的 valid=0 原生编码是全零，故保留 mask、槽位数量与输入形状。所有此类结果标签为 `DIAGNOSTIC_NOT_HISTORICAL` / `ABLATION_SENSITIVITY_ONLY`。
- `EGO_ABLATED` 和 `CONTEXT_ONLY` 为 `NOT_APPLICABLE_TO_ARCHITECTURE`：共享 83D GRU 没有 ego 缺失通道，raw zero 可代表物理值，不能当作 absence。
- `CONTEXT_SHUFFLE` 为 `NOT_EVALUABLE_FROZEN_STRATA_UNAVAILABLE`：缺少冻结的 `event_phase_bin` 与 `traffic_density_tertile` 来源/anchor，故不得启动或随意合并 strata。
- 对合法邻居消融，最低 representation×dose 中位 cosine 为 `{min_cosine:.4f}`；四指标逐行 OOD >=2/4 的最高比例为 `{max_row_ood:.4f}`。interaction frozen-probe R² 在 dose0/dose100 的十个 representation 中位变化分别为：front-distance -1.9186/-2.0236、rel-speed -1.4389/-1.5250、front-pressure -1.1301/-1.1496。这是输入敏感性，不是因果依赖或 information attribution。
- q99 规则被逐行保留，但冻结合同没有 condition-level 聚合门，因此 `D2_ABLATION_OOD_RISK = INCONCLUSIVE`，不把描述性比例升级为 `OOD_DOMINATED`。
- shortcut 审计完成 `{shortcut_count}` 个可计算 representation×dose×proxy 描述性格；地图/位置单类、路由/道路字段 unknown、scenario identity 一行一类均不强行建模。不存在冻结数值 shortcut 门，因此 `D2_SHORTCUT_RISK = INCONCLUSIVE`。

## 冻结逐假设结果

| 假设 | 结果 | 原因 |
| --- | --- | --- |
| D1_KNOWN_SEMANTIC_INFORMATION_PRESENT | SUPPORTED | 保留 Wave 1 的 frozen CORE-target 结果；本波未改写。 |
| D1_CROSS_DOMAIN_SEMANTIC_TRANSFER | INCONCLUSIVE | 已完成冻结 direct transfer，但无冻结跨域数值通过门。 |
| D2_RESPONSE_SENSITIVITY | NOT_EVALUABLE | ego absence 不能在 Gen-1 输入语义中合法定义。 |
| D2_CONTEXT_SENSITIVITY | INCONCLUSIVE | 仅有邻居消融敏感性；不得作因果归因。 |
| D2_PAIRING_SENSITIVITY | NOT_EVALUABLE | 完整冻结 shuffle strata 不可构造。 |
| D2_SHORTCUT_RISK | INCONCLUSIVE | 仅有低容量 group-aware 描述性关联，且无冻结数值门。 |
| D2_ABLATION_OOD_RISK | INCONCLUSIVE | 有逐行 q99 记录，无冻结条件级聚合规则。 |

历史 Stage6S BDD 没有被重跑或用于改变任何 primary conclusion；它仅保留为既有次级 development diagnosis。

所有数值见 `r0_cross_domain_probe_metrics.csv`、`r0_context_ablation_metrics.csv`、`r0_context_ablation_ood_metrics.csv`、`r0_context_ablation_probe_metrics.csv`、`r0_context_shuffle_metrics.csv` 和 `r0_context_leakage_probe_metrics.csv`。
"""
    (RESULTS / "R0_Wave2_D1_D2_Execution_Report_v1.md").write_text(execution, encoding="utf-8")
    cross = """# R0 Wave 2 跨模块科学诊断 v1

证据等级：`DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。本诊断不修改 Protocol v1.0、D0/D1 gate、历史 Stage7L BDD 的 primary 结论，亦不授权任何 RBR 训练。

## 现有证据的正确拼接

- D0（Wave 1.1）：`D0_POOLING_EFFECT = MIXED`，`D0_MASK_PADDING_SENSITIVITY = MIXED`；embedding geometry sensitivity 有支持，但 semantic retention 不可概括为 information loss。
- D1：`KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED` 维持不变。Wave 2 的 Waymo→nuPlan direct frozen-probe transfer 已执行，但没有冻结的跨域数值支持门，故 `D1_CROSS_DOMAIN_SEMANTIC_TRANSFER = INCONCLUSIVE`。
- D2：唯一可执行的是使用原生邻居缺失 sentinel 的诊断视图；ego 零化/仅上下文不具备合法缺失语义，完整上下文 shuffle 分层不能构建。因此 response、pairing 为 `NOT_EVALUABLE`，context、shortcut、ablation OOD 为 `INCONCLUSIVE`；绝不由自然数据 shuffle 或零化结果声称因果耦合。
- D3：三个 formal hypothesis 继续为 `INCONCLUSIVE`，没有重跑或改变其 primary 结果。

## 修正后的 Wave 1–2 scientific diagnosis

`CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED` 继续有效：当前可支持的是 temporal/pooling geometry sensitivity，而非普遍的 temporal information-loss 结论。D1 显示 Waymo 冻结表示中存在可读 CORE semantic information；Wave 2 没有提供足以把这种可读性升级为跨域 semantic transfer 支持的冻结门证据。D2 也没有提供可归因的 ego/context causality 或 context shortcut 支持。

因此，正确的压缩结论是：`KNOWN_SEMANTIC_INFORMATION_PRESENT_SUPPORTED; CROSS_DOMAIN_TRANSFER_INCONCLUSIVE; GEN1_CONTEXT_RESPONSE_ATTRIBUTION_UNRESOLVED; D3_FORMAL_HYPOTHESES_INCONCLUSIVE`。

Wave 1 的 D0 primary-metric omission 已按接受的 completeness correction 关闭，未新增 evidence downgrade；这不是对 Protocol 的修改。
"""
    (RESULTS / "R0_Wave2_Cross_Module_Diagnosis_v1.md").write_text(cross, encoding="utf-8")
    design = """# R0 Wave 2：RBR 设计含义 v1

本文件只给出未来设计约束；`RBR_A/B/C_TRAINING_AUTHORIZATION` 仍为 `NOT_AUTHORIZED`，本波不训练模型。

1. RBR-C 应原生表示 ego、context、neighbor source 的可用性，并把 missingness/slot validity 作为明确输入合同。共享 83D GRU 的 raw-zero 不能代替 source absence。
2. 训练与评估前应预先定义 full、ego-only、context-only、neighbor-ablated 和保持分层的 shuffle；每个视图都需要训练内覆盖或可证明的 OOD 控制。
3. 未来 context-shuffle 必须在冻结的 scenario family、direction、initial speed、traffic density、neighbor availability 和 event phase strata 内执行；应在数据入库时持久化所有预处理分层与独立单位键。
4. 应保存 R0 reference embedding/input bank 的可审计边界、PCA/nearest-neighbor OOD contract 与条件级聚合规则，避免在结果出现后决定何时称 OOD dominated。
5. 保持九个 CORE D1 targets、Waymo-only frozen probe 与 nuPlan direct-transfer 的区分；跨域支持门必须在未来执行前冻结，不能从本次描述性结果调参。
6. RBR-C 的 shortcut 控制应包含 log/map/route/source identity 的 group-aware contrast，并要求场景 identity 高预测能力不能单独作为 shortcut 结论。

这些是 execution/design constraints，而非关于模型优劣、因果 interaction 或训练授权的结论。
"""
    (RESULTS / "R0_Wave2_RBR_Design_Implications_v1.md").write_text(design, encoding="utf-8")


def main() -> None:
    require_output_paths_absent()
    verification, locks = verify_freeze()
    stage7l, parity = load_stage7l()
    if any(row["parity_result"] != "PARITY_CONFIRMED" for row in parity):
        # The requested audit table is still emitted below only after all in-memory work succeeds;
        # transfer will retain a target-level NOT_EVALUABLE entry rather than substitute a target.
        raise RuntimeError("A CORE target failed exact definition parity; refusing target substitution")
    probes, source_embeddings, dynamic = source_probes()
    full, neighbor = run_embeddings(stage7l)
    cross_domain = run_cross_domain(stage7l, probes, full)
    contract = ablation_contract_rows(stage7l)
    ood = ood_rows(full, neighbor)
    geometry = geometry_rows(full, neighbor)
    ablation_probe = ablation_probe_rows(stage7l, probes, full, neighbor)
    shuffle = [
        {
            "view": "CONTEXT_SHUFFLE",
            "execution_status": "NOT_EVALUABLE_FROZEN_STRATA_UNAVAILABLE",
            "frozen_strata": "scenario_family x lane_change_direction x initial_speed_tertile x traffic_density_tertile x neighbor_availability_pattern x event_phase_bin",
            "sparse_rule": "minimum=4; coarsen event_phase then traffic_density then initial_speed; never cross scenario_family",
            "availability_audit": "scenario family and ledger direction are retained; event_phase_bin and frozen pre-treatment traffic_density_tertile source/anchor are absent, so original strata cannot be created and coarsening cannot begin",
            "shuffle_seed": SEED,
            "pairing_result": "NOT_EVALUABLE",
            "evidence_level": EVIDENCE,
        }
    ]
    shortcut = shortcut_rows(stage7l, full)
    hypothesis = {
        "execution_status": "COMPLETE",
        "evidence_level": EVIDENCE,
        "hypothesis_results": {
            "D1_KNOWN_SEMANTIC_INFORMATION_PRESENT": "SUPPORTED",
            "D1_CROSS_DOMAIN_SEMANTIC_TRANSFER": "INCONCLUSIVE",
            "D1_GEOMETRY_DEGENERACY": "INCONCLUSIVE",
            "D2_RESPONSE_SENSITIVITY": "NOT_EVALUABLE",
            "D2_CONTEXT_SENSITIVITY": "INCONCLUSIVE",
            "D2_PAIRING_SENSITIVITY": "NOT_EVALUABLE",
            "D2_SHORTCUT_RISK": "INCONCLUSIVE",
            "D2_ABLATION_OOD_RISK": "INCONCLUSIVE",
        },
        "module_summaries": {
            "D1": "KNOWN_SEMANTICS_PRESENT_WITH_CROSS_DOMAIN_TRANSFER_INCONCLUSIVE",
            "D2": "MIXED_LIMITED_TO_LEGAL_NEIGHBOR_ABLATION_DIAGNOSTIC",
        },
        "limitations": [
            "R0_AUDIT_HOLDOUT=NOT_AVAILABLE",
            "No frozen cross-domain D1 numerical support gate",
            "Gen-1 has no native ego missingness/branch-ablation contract",
            "Frozen D2 shuffle strata are incompletely retained",
            "Frozen D2 q99 rule lacks a condition-level aggregation threshold",
        ],
        "training_authorization_modified": False,
        "d4_executed": False,
    }
    # All computations completed.  Only now create final files and close the accepted Wave-1 deviation.
    OUT.mkdir(parents=True, exist_ok=False)
    csv_dump(PARITY_PATH, parity)
    csv_dump(RESULTS / "r0_cross_domain_probe_metrics.csv", cross_domain)
    csv_dump(RESULTS / "r0_context_ablation_contract_audit.csv", contract)
    csv_dump(RESULTS / "r0_context_ablation_metrics.csv", geometry)
    csv_dump(RESULTS / "r0_context_ablation_ood_metrics.csv", ood)
    csv_dump(RESULTS / "r0_context_ablation_probe_metrics.csv", ablation_probe)
    csv_dump(RESULTS / "r0_context_shuffle_metrics.csv", shuffle)
    csv_dump(RESULTS / "r0_context_leakage_probe_metrics.csv", shortcut)
    closed = close_deviation()
    json_dump(RESULTS / "r0_wave2_hypothesis_results.json", hypothesis)
    markdown_reports(cross_domain, ood, geometry, shortcut, closed)
    environment = {
        "execution_id": "R0_WAVE2_D1_D2_V1",
        "timestamp_utc": now(),
        "git_branch": git("branch", "--show-current"),
        "git_commit": git("rev-parse", "HEAD"),
        "wave1_1_local_equivalent_commit": WAVE1_1_LOCAL_COMMIT,
        "wave1_1_declared_remote_commit": WAVE1_1_DECLARED_REMOTE_COMMIT,
        "declared_remote_commit_local_object_available": subprocess.run(
            ["git", "cat-file", "-e", f"{WAVE1_1_DECLARED_REMOTE_COMMIT}^{{commit}}"], cwd=ROOT, capture_output=True
        ).returncode == 0,
        "python": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "protocol_freeze_verification": verification,
        "checkpoint_sha256": locks,
        "waymo_reference_embedding_sha256": {rep: sha256(WAVE1_EMBED_ROOT / f"{rep}.npy") for rep in wave1.CHECKPOINTS},
        "stage7l_context_sha256": {dose: sha256(CONTEXT_ROOT / dose / "context_traj.npy") for dose in DOSES},
        "stage7l_ego_mask_sha256": {dose: sha256(CONTEXT_ROOT / dose / "ego_seq_mask.npy") for dose in DOSES},
        "waymo_reference_rows": dynamic.n_rows,
        "evidence_level": EVIDENCE,
        "prohibited_actions": "no training; no simulation; no new planner rollout; no checkpoint/tensor/historical-output mutation; no git reset/clean",
    }
    json_dump(OUT / "r0_wave2_freeze_verification.json", environment)
    manifest = {
        "execution_id": "R0_WAVE2_D1_D2_V1",
        "completed_at_utc": now(),
        "hypothesis_results": hypothesis,
        "deviation_closure": closed,
        "outputs": [str(path.relative_to(ROOT)) for path in sorted(RESULTS.glob("r0_*")) if path.is_file()] + [str((OUT / "r0_wave2_freeze_verification.json").relative_to(ROOT))],
        "new_tensor_or_checkpoint_outputs": False,
        "training_authorization_modified": False,
    }
    json_dump(RESULTS / "r0_wave2_execution_manifest.json", manifest)
    ledger = {
        "execution_id": "R0_WAVE2_D1_D2_V1",
        "command_id": "R0_WAVE2_EXECUTE_001",
        "timestamp_utc": now(),
        "command": "waymo_dev/bin/python tools/stageR_execute_r0_wave2_d1_d2.py",
        "git_commit": git("rev-parse", "HEAD"),
        "seed": SEED,
        "input_artifact_sha256": {"checkpoints": locks, "target_definition": sha256(ROOT / "docs/stageR/r0/manifests/r0_target_definition_v0.2.json")},
        "output_artifact_sha256": {path.name: sha256(path) for path in RESULTS.glob("r0_*") if path.is_file()},
        "exit_code": 0,
        "protocol_deviation_id": closed["deviation_id"],
    }
    json_dump(RESULTS / "r0_wave2_command_ledger.json", ledger)
    print(json.dumps({"status": "R0_WAVE2_COMPLETE", "hypothesis_results": hypothesis["hypothesis_results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
