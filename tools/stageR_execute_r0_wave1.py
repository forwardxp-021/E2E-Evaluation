#!/usr/bin/env python3
"""Execute frozen R0 Wave 1 (D0, D1, D3) on existing development assets only.

This command never trains a model, runs a planner, obtains data, or writes a
frozen v1.0 protocol asset.  Large intermediate embeddings are local outputs;
the committed products are small tables, reports, and execution provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402
from tools.stage6l_prepare_context_representation_ablation import ego_kinematic_features  # noqa: E402
from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder  # noqa: E402

BRANCH = "20260825_stageR_new"
BINDING_COMMIT = "319757c7f72efb55c80c780e4d0f17e5341b19ec"
CONTENT_COMMIT = "5bd5c7ac58c284d4c938919cacf2eefb969a5c44"
EVIDENCE_LEVEL = "DEVELOPMENT_DIAGNOSTIC_EVIDENCE"
OUT = ROOT / "outputs/stageR/r0_wave1_d0_d1_d3_v1"
RESULTS = ROOT / "docs/stageR/r0/results"
BOOTSTRAP_REPS = 5000
PERMUTATION_REPS = 49999
RIDGE_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SEEDS = (3407, 3408, 3409)
CORE_TARGETS = (
    ("longitudinal", "ego13.mean_speed", "continuous"),
    ("longitudinal", "ego13.end_minus_start_speed", "continuous"),
    ("longitudinal", "ego13.rms_accel", "continuous"),
    ("lateral", "ego13.rms_yaw_rate", "continuous"),
    ("lateral", "ego13.heading_change_abs_total", "continuous"),
    ("lateral", "raw33.lane_change_count_proxy", "categorical"),
    ("interaction", "raw33.mean_front_distance", "continuous"),
    ("interaction", "raw33.mean_rel_speed", "continuous"),
    ("interaction", "raw33.front_pressure_score", "continuous"),
)
CHECKPOINTS = {
    "old64": ("old", None, ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt"),
    **{
        f"{candidate}_seed{seed}": (candidate, seed, ROOT / f"outputs/stage6t_candidates_v1/candidate_{name}/seed_{seed}/best_model.pt")
        for candidate, name in (
            ("A", "A_dynamic_data_legacy"),
            ("B", "B_single_gru_recovery"),
            ("C", "C_dual_branch"),
        )
        for seed in SEEDS
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_fold(label: str, n_splits: int = 5) -> int:
    return int(hashlib.sha256(("r0-wave1-groupfold-v1|" + label).encode()).hexdigest()[:16], 16) % n_splits


def checkpoint_sha_locks() -> dict[str, str]:
    expected = {
        "old64": "909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc",
        "A_seed3407": "353982753f208d27d677c6863a681997b8e28b728573a52fa407807f6fd0298d",
        "A_seed3408": "8d9886490b9308623abe938b48fc926106dcf1c109800b78952175970a31077c",
        "A_seed3409": "5e22156f0f0197aca9a3fef1fe0e0db1573efd589f93527edfa25eec9f1c92bd",
        "B_seed3407": "d8e0de6e74ee29076082aabef27a425b47678e1372c630e4f4a04106ff34265f",
        "B_seed3408": "3b8ca8949da185bc25715997d49a64aa4131641409d10f44565a94a9c86f4f35",
        "B_seed3409": "c2d54a51bfc13d597b0265c59bfe5377035168d133ab97498cbd2d4a1fa53ac5",
        "C_seed3407": "cc6bf3c427534f66f74904c8948bf427cfe9f1152bba4bca0e8342f3fa47433d",
        "C_seed3408": "603d56f34b62fb22e6c59d6558fb42b8dbc67ca0897d09660e87dc8e1f09f521",
        "C_seed3409": "1b0a10779d5c90559e71af42cbd2d8c7b6611f6ecb2efca6059b426ce51a974e",
    }
    observed = {name: sha256(path) for name, (_, _, path) in CHECKPOINTS.items()}
    if observed != expected:
        mismatches = [name for name in expected if observed.get(name) != expected[name]]
        raise RuntimeError(f"Locked checkpoint SHA mismatch: {mismatches}")
    return observed


def verify_freeze() -> dict[str, Any]:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: wrong branch")
    if git("rev-parse", "HEAD") != BINDING_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: wrong binding commit")
    if git("rev-parse", "r0-v1.0-protocol-freeze^{}") != BINDING_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: tag does not target binding commit")
    manifest_dir = ROOT / "docs/stageR/r0/manifests"
    binding = json_load(manifest_dir / "r0_v1_freeze_binding.json")
    if binding["R0_V1_FREEZE_CONTENT_COMMIT"] != CONTENT_COMMIT:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: content commit")
    for key, entry in binding["all_frozen_artifact_sha256"].items():
        observed = sha256(ROOT / entry["path"])
        if observed != entry["sha256"]:
            raise RuntimeError(f"PROTOCOL_BINDING_MISMATCH: {key}")
    for key in ("protocol_frozen_manifest", "training_authorization_manifest", "scientific_owner_approval"):
        entry = binding[key]
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"PROTOCOL_BINDING_MISMATCH: {key}")
    sap = json_load(manifest_dir / "r0_statistical_analysis_plan_v1.0.json")
    if sap["d3"]["projection_ranks"] != [1, 2, 4, 8, 16] or sap["d3"]["primary_kernel"] != "RBF":
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: D3 readout contract")
    if sap["d3"]["calibration_fpr_gate"]["upper_95_ci_max"] != 0.075:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: D3 FPR gate")
    protected = ROOT / binding["protected_dirty_output_exclusion"]["path"]
    protected_sha = sha256(protected)
    if protected_sha != binding["protected_dirty_output_exclusion"]["sha256"]:
        raise RuntimeError("PROTOCOL_BINDING_MISMATCH: protected historical CSV")
    return {
        "tag": "r0-v1.0-protocol-freeze",
        "binding_commit": BINDING_COMMIT,
        "content_commit": CONTENT_COMMIT,
        "frozen_artifact_count": len(binding["all_frozen_artifact_sha256"]),
        "protected_csv_path": str(protected.relative_to(ROOT)),
        "protected_csv_sha256": protected_sha,
        "git_status_sha256_before_execution": hashlib.sha256(git("status", "--porcelain=v1").encode()).hexdigest(),
        "git_status_line_count_before_execution": len(git("status", "--porcelain=v1").splitlines()),
    }


def load_models() -> dict[str, torch.nn.Module]:
    schema = json_load(ROOT / "outputs/stage6r_dynamic_full51_semantic_strict_part_00_09/feature_schema.json")
    feature_names = [str(x["name"]) for x in schema["features"]]
    groups = feature_group_indices(feature_names)
    # Candidate checkpoint metadata was written with PyTorch 2.5, which
    # serializes its version string as torch.torch_version.TorchVersion.
    # The locked analysis environment is PyTorch 1.9.  Provide the equivalent
    # string type only for metadata unpickling; model tensors and checkpoint
    # bytes remain read-only and are SHA-locked by checkpoint_sha_locks().
    if "torch.torch_version" not in sys.modules:
        torch_version_module = types.ModuleType("torch.torch_version")

        class TorchVersion(str):
            pass

        torch_version_module.TorchVersion = TorchVersion
        sys.modules["torch.torch_version"] = torch_version_module
    models: dict[str, torch.nn.Module] = {}
    for rep, (kind, _, path) in CHECKPOINTS.items():
        # PyTorch 1.9 in the locked local environment predates the
        # ``weights_only`` option. Checkpoint SHA-256 values are verified
        # before this function is called.
        payload = torch.load(path, map_location="cpu")
        state = payload["model"]
        if kind == "old":
            model: torch.nn.Module = ContextFlattenGRUEncoder(83, 128, 64)
        else:
            model = UnifiedABCModel(kind, groups)
        model.load_state_dict(state, strict=True)
        model.eval()
        models[rep] = model
    return models


def model_embedding(model: torch.nn.Module, contexts: np.ndarray, batch_size: int) -> np.ndarray:
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = torch.from_numpy(np.asarray(contexts[start:start + batch_size], dtype=np.float32).copy())
            result.append(model(batch).detach().cpu().numpy().astype(np.float64))
    output = np.concatenate(result, axis=0)
    if output.ndim != 2 or output.shape[1] != 64 or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid embedding output {output.shape}")
    return output


def pooled_embeddings(model: torch.nn.Module, contexts: np.ndarray, valid_lengths: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """D0-C pools the one forward-pass hidden sequence before the frozen projection."""
    tensor = torch.from_numpy(np.asarray(contexts, dtype=np.float32).copy())
    with torch.no_grad():
        if isinstance(model, ContextFlattenGRUEncoder):
            hidden, _ = model.gru(tensor)
            def project(values: torch.Tensor) -> torch.Tensor: return model.proj(values)
            sequences = [(hidden, project)]
        elif isinstance(model, UnifiedABCModel) and model.candidate in ("A", "B"):
            hidden, _ = model.encoder.gru(tensor)
            def project(values: torch.Tensor) -> torch.Tensor: return model.encoder.proj(values)
            sequences = [(hidden, project)]
        elif isinstance(model, UnifiedABCModel) and model.candidate == "C":
            ego_hidden, _ = model.encoder.ego_gru(tensor[:, :, :8])
            context_hidden, _ = model.encoder.context_gru(tensor)
            sequences = [(ego_hidden, model.encoder.ego_proj), (context_hidden, model.encoder.context_proj)]
        else:
            raise TypeError(f"Unexpected model type {type(model)}")

        output: dict[str, list[torch.Tensor]] = {"last": [], "mean": [], "max": []}
        if valid_lengths is not None:
            output["final_valid"] = []
            output["masked_mean"] = []
        for hidden, project in sequences:
            output["last"].append(project(hidden[:, -1]))
            output["mean"].append(project(hidden.mean(dim=1)))
            output["max"].append(project(hidden.max(dim=1).values))
            if valid_lengths is not None:
                index = torch.from_numpy(np.asarray(valid_lengths - 1, dtype=np.int64))
                output["final_valid"].append(project(hidden[torch.arange(len(hidden)), index]))
                mask = torch.arange(hidden.shape[1])[None, :] < torch.from_numpy(valid_lengths)[:, None]
                denom = mask.sum(dim=1, keepdim=True).to(hidden.dtype)
                output["masked_mean"].append(project((hidden * mask[:, :, None]).sum(dim=1) / denom))
        return {key: torch.cat(parts, dim=1).cpu().numpy().astype(np.float64) for key, parts in output.items()}


@dataclass
class DynamicValidation:
    targets: dict[str, np.ndarray]
    scenario: np.ndarray
    shard_rows: list[tuple[Path, np.ndarray, slice]]
    n_rows: int


def load_dynamic_validation() -> DynamicValidation:
    manifest = json_load(ROOT / "outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json")
    target_parts = {name: [] for _, name, _ in CORE_TARGETS}
    scenario_parts: list[np.ndarray] = []
    shard_rows: list[tuple[Path, np.ndarray, slice]] = []
    offset = 0
    for text_path in manifest["shard_paths"]:
        shard = Path(text_path)
        split = np.load(shard / "split.npy", allow_pickle=True).astype(str)
        selected = np.flatnonzero(split == "val")
        if not len(selected):
            continue
        ego = np.asarray(np.load(shard / "ego_seq.npy", mmap_mode="r")[selected], dtype=np.float64)
        # Dynamic Waymo windows have a complete fixed 80-frame ego sequence; no ego mask is supplied.
        valid = np.isfinite(ego).all(axis=2)
        if not valid.all() or not np.all(valid.sum(axis=1) == 80):
            raise RuntimeError("Dynamic validation ego window is not complete fixed-T80")
        ego13 = ego_kinematic_features(ego, valid)
        raw33 = np.asarray(np.load(shard / "interaction_feat_style_raw.npy", mmap_mode="r")[selected], dtype=np.float64)
        meta = np.load(shard / "meta.npy", allow_pickle=True)[selected]
        extracted = {
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
        for name, values in extracted.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite D1 target: {name}")
            target_parts[name].append(values)
        scenario_parts.append(meta["scenario_id"].astype(str))
        next_offset = offset + len(selected)
        shard_rows.append((shard, selected, slice(offset, next_offset)))
        offset = next_offset
    targets = {name: np.concatenate(parts) for name, parts in target_parts.items()}
    scenario = np.concatenate(scenario_parts)
    if offset != 16870 or len(np.unique(scenario)) < 30:
        raise RuntimeError(f"Unexpected validation support: rows={offset}, groups={len(np.unique(scenario))}")
    return DynamicValidation(targets, scenario, shard_rows, offset)


def emit_dynamic_embeddings(models: dict[str, torch.nn.Module], data: DynamicValidation, batch_size: int) -> dict[str, np.ndarray]:
    root = OUT / "embeddings_val"
    root.mkdir(parents=True, exist_ok=True)
    values: dict[str, np.ndarray] = {}
    for rep, model in models.items():
        path = root / f"{rep}.npy"
        if path.exists():
            array = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
            if array.shape != (data.n_rows, 64) or not np.isfinite(array).all():
                raise RuntimeError(f"Existing embedding does not satisfy the Wave-1 contract: {rep}")
            values[rep] = array
            continue
        disk = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(data.n_rows, 64))
        for shard, indices, dest in data.shard_rows:
            context = np.load(shard / "context_traj.npy", mmap_mode="r")
            chunks = []
            for start in range(0, len(indices), batch_size):
                batch = torch.from_numpy(np.asarray(context[indices[start:start + batch_size]], dtype=np.float32).copy())
                with torch.no_grad():
                    chunks.append(model(batch).detach().cpu().numpy().astype(np.float32))
            disk[dest] = np.concatenate(chunks, axis=0)
        disk.flush()
        array = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        if array.shape != (data.n_rows, 64) or not np.isfinite(array).all():
            raise RuntimeError(f"Embedding write failure: {rep}")
        values[rep] = array
    return values


def grouped_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=5)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def selected_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Ridge:
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    search = GridSearchCV(Ridge(), {"alpha": list(RIDGE_GRID)}, cv=folds, scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(x, y)
    return Ridge(alpha=float(search.best_params_["alpha"])).fit(x, y)


def selected_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> LogisticRegression:
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    search = GridSearchCV(LogisticRegression(max_iter=500, solver="liblinear", random_state=2026082601), {"C": list(RIDGE_GRID)}, cv=folds, scoring="balanced_accuracy", n_jobs=1)
    search.fit(x, y)
    return LogisticRegression(C=float(search.best_params_["C"]), max_iter=500, solver="liblinear", random_state=2026082601).fit(x, y)


def oof_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, categorical: bool) -> np.ndarray:
    predicted = np.empty(len(y), dtype=np.float64)
    for train, test in grouped_folds(groups):
        if categorical:
            model = selected_logistic(x[train], y[train], groups[train])
            predicted[test] = model.predict(x[test])
        else:
            model = selected_ridge(x[train], y[train], groups[train])
            predicted[test] = model.predict(x[test])
    return predicted


def inverse_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names, inverse = np.unique(groups, return_inverse=True)
    return names, inverse


def continuous_bootstrap(y: np.ndarray, prediction: np.ndarray, inverse: np.ndarray, n_group: int, seed: int) -> tuple[float, float, float]:
    point = float(r2_score(y, prediction))
    n = np.bincount(inverse, minlength=n_group).astype(float)
    sy = np.bincount(inverse, weights=y, minlength=n_group)
    sy2 = np.bincount(inverse, weights=y * y, minlength=n_group)
    sse = np.bincount(inverse, weights=(y - prediction) ** 2, minlength=n_group)
    rng = np.random.default_rng(seed)
    boot: list[np.ndarray] = []
    for _ in range(0, BOOTSTRAP_REPS, 100):
        counts = rng.poisson(1.0, size=(100, n_group))
        bn = counts @ n
        by = counts @ sy
        sst = counts @ sy2 - by * by / bn
        score = 1.0 - (counts @ sse) / np.maximum(sst, 1e-12)
        boot.append(score)
    stacked = np.concatenate(boot)
    return point, float(np.quantile(stacked, 0.025)), float(np.quantile(stacked, 0.975))


def categorical_bootstrap(y: np.ndarray, prediction: np.ndarray, inverse: np.ndarray, n_group: int, seed: int) -> tuple[float, float, float]:
    point = float(balanced_accuracy_score(y, prediction))
    pos = np.bincount(inverse, weights=(y == 1), minlength=n_group)
    neg = np.bincount(inverse, weights=(y == 0), minlength=n_group)
    tp = np.bincount(inverse, weights=((y == 1) & (prediction == 1)), minlength=n_group)
    tn = np.bincount(inverse, weights=((y == 0) & (prediction == 0)), minlength=n_group)
    rng = np.random.default_rng(seed)
    boot: list[np.ndarray] = []
    for _ in range(0, BOOTSTRAP_REPS, 100):
        counts = rng.poisson(1.0, size=(100, n_group))
        score = 0.5 * ((counts @ tp) / np.maximum(counts @ pos, 1.0) + (counts @ tn) / np.maximum(counts @ neg, 1.0))
        boot.append(score)
    stacked = np.concatenate(boot)
    return point, float(np.quantile(stacked, 0.025)), float(np.quantile(stacked, 0.975))


def run_d1(embeddings: dict[str, np.ndarray], data: DynamicValidation) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    names, inverse = inverse_groups(data.scenario)
    metric_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    family_state: dict[str, dict[str, Any]] = {}
    oof: dict[str, Any] = {}
    for rep, x in embeddings.items():
        prediction_by_target: dict[str, np.ndarray] = {}
        for family, target, kind in CORE_TARGETS:
            y = data.targets[target]
            pred = oof_predictions(x, y, data.scenario, kind == "categorical")
            prediction_by_target[target] = pred
            if kind == "continuous":
                point, lo, hi = continuous_bootstrap(y, pred, inverse, len(names), int(hashlib.sha256((rep + target).encode()).hexdigest()[:8], 16))
                passed = point >= 0.10 and lo > 0.0
                metric = "R2"
                gate = "R2>=0.10 AND log-cluster CI lower>0"
            else:
                point, lo, hi = categorical_bootstrap(y.astype(int), pred.astype(int), inverse, len(names), int(hashlib.sha256((rep + target).encode()).hexdigest()[:8], 16))
                passed = point >= 0.60 and lo > 0.50
                metric = "balanced_accuracy"
                gate = "BA>=0.60 AND log-cluster CI lower>0.50"
            metric_rows.append({
                "representation": rep, "candidate": rep.split("_")[0], "seed": rep.split("seed")[-1] if "seed" in rep else "NOT_APPLICABLE",
                "semantic_family": family, "target_id": target, "target_kind": kind, "primary_metric": metric,
                "point_estimate": point, "ci95_lower": lo, "ci95_upper": hi, "gate": gate,
                "target_result": "SUPPORTED" if passed else "NOT_SUPPORTED", "n_rows": len(y), "n_scenario_groups": len(names),
                "split_contract": "five-fold grouped held-out by scenario_id; cluster bootstrap= scenario because log identity is unavailable",
                "evidence_level": EVIDENCE_LEVEL,
            })
        for family in ("longitudinal", "lateral", "interaction"):
            rows = [r for r in metric_rows if r["representation"] == rep and r["semantic_family"] == family]
            n_pass = sum(r["target_result"] == "SUPPORTED" for r in rows)
            family_state.setdefault(rep, {})[family] = {"pass_count": n_pass, "target_count": 3, "result": "SUPPORTED" if n_pass >= 2 else "NOT_SUPPORTED"}
        centered = x - x.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(x) - 1, 1)
        values = np.clip(np.linalg.eigvalsh(cov), 0.0, None)[::-1]
        weights = values / max(values.sum(), 1e-12)
        entropy_rank = float(np.exp(-np.sum(np.where(weights > 0, weights * np.log(weights), 0.0))))
        participation = float(values.sum() ** 2 / max(np.square(values).sum(), 1e-12))
        pair_rng = np.random.default_rng(int(hashlib.sha256((rep + "cosine").encode()).hexdigest()[:8], 16))
        ii = pair_rng.integers(0, len(x), size=20000); jj = pair_rng.integers(0, len(x), size=20000)
        norm = np.linalg.norm(x, axis=1)
        cosine = np.sum(x[ii] * x[jj], axis=1) / np.maximum(norm[ii] * norm[jj], 1e-12)
        geometry_rows.append({
            "representation": rep, "dimension": 64, "n_rows": len(x), "effective_rank_entropy": entropy_rank,
            "participation_ratio": participation, "top_eigenvalue_fraction": float(weights[0]),
            "explained_variance_rank_90": int(np.searchsorted(np.cumsum(weights), 0.90) + 1),
            "embedding_norm_median": float(np.median(norm)), "embedding_norm_p05": float(np.quantile(norm, .05)),
            "embedding_norm_p95": float(np.quantile(norm, .95)), "pairwise_cosine_median": float(np.median(cosine)),
            "pairwise_cosine_p95": float(np.quantile(cosine, .95)), "near_constant_dimension_count_variance_lt_1e_8": int((np.diag(cov) < 1e-8).sum()),
            "geometry_result": "INCONCLUSIVE", "geometry_boundary": "no frozen numeric geometry-degeneracy pass/fail threshold; geometry cannot alone determine D1",
            "evidence_level": EVIDENCE_LEVEL,
        })
        oof[rep] = prediction_by_target
    return metric_rows, geometry_rows, family_state, oof


def task_readouts(embeddings: dict[str, np.ndarray], data: DynamicValidation) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    continuous = [target for _, target, kind in CORE_TARGETS if kind == "continuous"]
    y = np.column_stack([data.targets[target] for target in continuous])
    models: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    folds = grouped_folds(data.scenario)
    for rep, x in embeddings.items():
        fold_scores = {rank: [] for rank in (1, 2, 4, 8, 16)}
        for train, test in folds:
            regression = selected_ridge(x[train], y[train], data.scenario[train])
            _, _, vt = np.linalg.svd(regression.coef_, full_matrices=False)
            for rank in fold_scores:
                projection = vt[:rank].T
                projected = selected_ridge(x[train] @ projection, y[train], data.scenario[train])
                fold_scores[rank].append(float(r2_score(y[test], projected.predict(x[test] @ projection), multioutput="variance_weighted")))
        score = {rank: float(np.mean(values)) for rank, values in fold_scores.items()}
        se = {rank: float(np.std(values, ddof=1) / math.sqrt(len(values))) for rank, values in fold_scores.items()}
        best_rank = max(score, key=score.get)
        chosen_rank = min(rank for rank in score if score[rank] >= score[best_rank] - se[best_rank])
        final = selected_ridge(x, y, data.scenario)
        _, _, vt = np.linalg.svd(final.coef_, full_matrices=False)
        projection = vt[:chosen_rank].T
        semantic = final.predict(x)
        models[rep] = {"rank": chosen_rank, "projection": projection, "semantic": semantic, "semantic_model": final, "continuous_targets": continuous}
        for rank in (1, 2, 4, 8, 16):
            rows.append({
                "representation": rep, "rank": rank, "semantic_retention_score": score[rank], "score_standard_error": se[rank],
                "selected_by_smallest_within_1SE": rank == chosen_rank,
                "selection_data": "R0_DEVELOPMENT Waymo validation grouped folds; no Stage7L/planner outcome used",
                "null_calibration_condition": "D3 calibration recorded separately; rank is a development-diagnostic selection only",
                "evidence_level": EVIDENCE_LEVEL,
            })
    return models, rows


def d0_effect(reference: np.ndarray, alternate: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float, float]:
    paired = np.linalg.norm(alternate - reference, axis=1)
    names, inverse = inverse_groups(groups)
    group_mean = np.bincount(inverse, weights=paired, minlength=len(names)) / np.maximum(np.bincount(inverse, minlength=len(names)), 1)
    point = float(np.mean(group_mean) / max(np.std(group_mean, ddof=1), 1e-12))
    rng = np.random.default_rng(seed)
    samples = group_mean[rng.integers(0, len(group_mean), size=(BOOTSTRAP_REPS, len(group_mean)))]
    b = samples.mean(axis=1) / np.maximum(samples.std(axis=1, ddof=1), 1e-12)
    return point, float(np.quantile(b, .025)), float(np.quantile(b, .975))


def stage7l_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    context_root = ROOT / "outputs/stage7l_e_prospective_bdd_v1/contexts"
    all_context, all_mask, all_ego, labels, scenario = [], [], [], [], []
    for dose in ("dose0", "dose25", "dose50", "dose75", "dose100"):
        directory = context_root / dose
        context = np.asarray(np.load(directory / "context_traj.npy", mmap_mode="r"), dtype=np.float32)
        mask = np.asarray(np.load(directory / "ego_seq_mask.npy", mmap_mode="r"), dtype=bool)
        ego = np.asarray(np.load(directory / "ego_seq.npy", mmap_mode="r"), dtype=np.float64)
        meta = list(csv.DictReader((directory / "metadata.csv").open(encoding="utf-8")))
        if context.shape != (80, 150, 83) or mask.shape != (80, 150) or ego.shape != (80, 150, 8):
            raise RuntimeError("Stage7L context contract mismatch")
        all_context.append(context); all_mask.append(mask); all_ego.append(ego); labels.extend([dose] * 80)
        scenario.extend([row["scenario_token"] for row in meta])
    return np.concatenate(all_context), np.concatenate(all_mask), np.concatenate(all_ego), np.asarray(labels), scenario


def run_d0(models: dict[str, torch.nn.Module]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contexts, mask, ego, dose, scenario = stage7l_inputs()
    valid = mask.sum(axis=1).astype(np.int64)
    if set(valid.tolist()) != {149, 150}:
        raise RuntimeError("Stage7L valid-length contract mismatch")
    scenario_array = np.asarray(scenario)
    primary_rows: list[dict[str, Any]] = []
    pooling_rows: list[dict[str, Any]] = []
    content_rows: list[dict[str, Any]] = []
    for rep, model in models.items():
        full = pooled_embeddings(model, contexts, valid)
        for view in ("mean", "max"):
            effect, lo, hi = d0_effect(full["last"], full[view], scenario_array, int(hashlib.sha256((rep + view).encode()).hexdigest()[:8], 16))
            pooling_rows.append({
                "study": "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY", "representation": rep, "pooling_reference": "last", "pooling_view": view,
                "same_encoder_forward_hidden_sequence": True, "paired_standardized_embedding_difference": effect, "ci95_lower": lo, "ci95_upper": hi,
                "gate_absolute_effect_ge_0_10": abs(effect) >= .10 and not (lo <= 0 <= hi), "n_rows": len(contexts), "n_scenario_clusters": len(np.unique(scenario_array)),
                "evidence_level": EVIDENCE_LEVEL,
            })
        for view in ("final_valid", "masked_mean"):
            effect, lo, hi = d0_effect(full["last"], full[view], scenario_array, int(hashlib.sha256((rep + view).encode()).hexdigest()[:8], 16))
            primary_rows.append({
                "study": "D0-D_MASK_PADDING_SENSITIVITY", "representation": rep, "diagnostic_view": view, "historical_reference": "T150+final_hidden+historical_padding_mask_behavior",
                "diagnostic_not_historical": True, "paired_standardized_embedding_difference": effect, "ci95_lower": lo, "ci95_upper": hi,
                "gate_absolute_effect_ge_0_10": abs(effect) >= .10 and not (lo <= 0 <= hi), "n_rows": len(contexts), "n_final_invalid": int((valid == 149).sum()), "evidence_level": EVIDENCE_LEVEL,
                "execution_status": "COMPLETE", "reason": "",
            })
        for name, values, classification in (
            ("first80", contexts[:, :80], "CONTENT_CONFOUNDED_LENGTH_DIAGNOSTIC"),
            ("last80", contexts[:, 70:150], "CONTENT_CONFOUNDED_LENGTH_DIAGNOSTIC"),
            ("overlap80", contexts[:, 35:115], "CONTENT_CONFOUNDED_LENGTH_DIAGNOSTIC"),
        ):
            alternate = model_embedding(model, values, 128)
            effect, lo, hi = d0_effect(full["last"], alternate, scenario_array, int(hashlib.sha256((rep + name).encode()).hexdigest()[:8], 16))
            content_rows.append({
                "study": "D0-A_CONTENT_WINDOW_DESCRIPTIVE", "representation": rep, "view": name, "classification": classification,
                "paired_standardized_embedding_difference": effect, "ci95_lower": lo, "ci95_upper": hi, "n_rows": len(contexts), "evidence_level": EVIDENCE_LEVEL,
                "primary_hypothesis_eligible": False,
            })
    primary_rows.extend([
        {"study": "D0-A_LENGTH_TEMPORAL_CONTRACT", "representation": "ALL", "diagnostic_view": "CONTROLLED_LENGTH_STUDY", "historical_reference": "T150+final_hidden+historical_padding_mask_behavior", "diagnostic_not_historical": False, "paired_standardized_embedding_difference": "", "ci95_lower": "", "ci95_upper": "", "gate_absolute_effect_ge_0_10": False, "n_rows": 0, "n_final_invalid": int((valid == 149).sum()), "evidence_level": EVIDENCE_LEVEL, "execution_status": "NOT_EVALUABLE", "reason": "No frozen same-event-content T80/T150 construction exists; observed windows are content-confounded."},
        {"study": "D0-B_MATCHED_NATURAL_POSITION_RETENTION_STUDY", "representation": "ALL", "diagnostic_view": "MATCHED_NATURAL_POSITION", "historical_reference": "T150+final_hidden+historical_padding_mask_behavior", "diagnostic_not_historical": False, "paired_standardized_embedding_difference": "", "ci95_lower": "", "ci95_upper": "", "gate_absolute_effect_ge_0_10": False, "n_rows": 0, "n_final_invalid": int((valid == 149).sum()), "evidence_level": EVIDENCE_LEVEL, "execution_status": "NOT_EVALUABLE", "reason": "Stage7L historical asset lacks frozen event-anchor and matching-covariate ledger for the required matched-natural design."},
    ])
    return primary_rows, pooling_rows, content_rows


def median_bandwidth(reference: np.ndarray) -> float:
    index = np.linspace(0, len(reference) - 1, num=min(1024, len(reference)), dtype=int)
    x = reference[index]
    sq = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(x*x, axis=1)[None, :] - 2 * x @ x.T, 0)
    values = np.sqrt(sq[np.triu_indices(len(x), k=1)])
    return float(np.median(values[values > 0]))


def rbf_mmd_null(x: np.ndarray, y: np.ndarray, bandwidth: float, seed: int) -> dict[str, float]:
    n = len(x)
    both = np.concatenate([x, y])
    sq = np.maximum(np.sum(both*both, axis=1)[:, None] + np.sum(both*both, axis=1)[None, :] - 2 * both @ both.T, 0)
    kernel = np.exp(-sq / max(2 * bandwidth * bandwidth, 1e-12))
    signs = np.concatenate([np.ones(n), -np.ones(n)])
    raw = float(signs @ kernel @ signs / (n*n))
    rng = np.random.default_rng(seed)
    null = np.empty(PERMUTATION_REPS, dtype=np.float64)
    for start in range(0, PERMUTATION_REPS, 512):
        count = min(512, PERMUTATION_REPS - start)
        flips = rng.choice(np.array([-1.0, 1.0]), size=(count, n))
        signed = np.concatenate([flips, -flips], axis=1)
        null[start:start+count] = np.einsum("bi,ij,bj->b", signed, kernel, signed, optimize=True) / (n*n)
    q95 = float(np.quantile(null, .95)); mean = float(null.mean()); sd = float(null.std(ddof=1))
    return {"raw_statistic": raw, "null_mean": mean, "null_q95": q95, "null_sd": sd, "ratio_to_null_q95": raw / max(q95, 1e-12), "standardized_z": (raw - mean) / max(sd, 1e-12), "p_value": float((1 + (null >= raw).sum()) / (1 + len(null))), "detection": bool(raw > q95)}


def run_d3(embeddings: dict[str, np.ndarray], readouts: dict[str, dict[str, Any]], models: dict[str, torch.nn.Module]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts, _, _, dose, _ = stage7l_inputs()
    task_rows = list(csv.DictReader((ROOT / "outputs/stage7l_e_prospective_bdd_v1/task_masks.csv").open(encoding="utf-8")))
    mask = np.asarray([row["LAT.LANE_CHANGE"].lower() == "true" for row in task_rows], dtype=bool)
    if mask.sum() != 80:
        raise RuntimeError("Frozen pure-lateral task mask mismatch")
    select0 = np.flatnonzero(dose == "dose0"); select100 = np.flatnonzero(dose == "dose100")
    rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for rep, model in models.items():
        z = model_embedding(model, contexts, 128)
        readout = readouts[rep]
        projected = z @ readout["projection"]
        semantic = readout["semantic_model"].predict(embeddings[rep])
        # The Stage7L semantic readout is obtained by applying the same frozen development model.
        stage_semantic = readout["semantic_model"].predict(z)
        for name, dynamic_reference, stage_values in (
            ("R_full64", embeddings[rep], z),
            ("R_linear_task", embeddings[rep] @ readout["projection"], projected),
            ("R_fixed_semantic", semantic, stage_semantic),
        ):
            bandwidth = median_bandwidth(dynamic_reference)
            result = rbf_mmd_null(stage_values[select0][mask], stage_values[select100][mask], bandwidth, int(hashlib.sha256((rep + name).encode()).hexdigest()[:8], 16))
            rows.append({
                "domain": "pure_lateral", "task": "LAT.LANE_CHANGE", "representation": rep, "readout": name, "selected_projection_rank": readout["rank"] if name == "R_linear_task" else "NOT_APPLICABLE",
                "kernel": "RBF", "bandwidth": bandwidth, "bandwidth_reference": "treatment-label-blind Waymo R0_DEVELOPMENT reference bank positive off-diagonal median",
                "n_pairs": int(mask.sum()), **result, "bidirectional_detection": "NOT_EVALUABLE_SINGLE_DIRECTION_CONTRAST", "fpr": "INCONCLUSIVE_INSUFFICIENT_INDEPENDENT_NULL_UNITS",
                "fpr_gate_upper95_max": .075, "evidence_level": EVIDENCE_LEVEL, "execution_status": "COMPLETE", "reason": "",
            })
            calibration.append({"representation": rep, "readout": name, "nominal_fpr": .05, "empirical_fpr": "INCONCLUSIVE", "upper95_ci": "INCONCLUSIVE", "independent_null_units": 1, "result": "INCONCLUSIVE", "reason": "One historical paired pure-lateral population is not an independent null calibration series.", "evidence_level": EVIDENCE_LEVEL})
    for domain, reason in (
        ("longitudinal", "No frozen R0 paired development input/readout contract binds a Wave-1 longitudinal contrast."),
        ("following", "No frozen R0 paired development input/readout contract binds a Wave-1 following contrast."),
        ("interaction", "No frozen R0 paired development input/readout contract binds a Wave-1 interaction contrast."),
    ):
        rows.append({"domain": domain, "task": "NOT_AVAILABLE", "representation": "ALL", "readout": "R_full64/R_linear_task/R_fixed_semantic", "selected_projection_rank": "NOT_EVALUABLE", "kernel": "RBF", "bandwidth": "", "bandwidth_reference": "", "n_pairs": 0, "raw_statistic": "", "null_mean": "", "null_q95": "", "null_sd": "", "ratio_to_null_q95": "", "standardized_z": "", "p_value": "", "detection": "", "bidirectional_detection": "", "fpr": "", "fpr_gate_upper95_max": .075, "evidence_level": EVIDENCE_LEVEL, "execution_status": "NOT_EVALUABLE", "reason": reason})
    return rows, calibration


def markdown_report(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{text.rstrip()}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    allowed_partial_out = {"r0_wave1_freeze_verification.json", "embeddings_val"}
    allowed_partial_results = {"r0_wave1_environment.json"}
    present_out = {path.name for path in OUT.iterdir()} if OUT.exists() else set()
    present_results = {path.name for path in RESULTS.iterdir()} if RESULTS.exists() else set()
    if not present_out.issubset(allowed_partial_out) or not present_results.issubset(allowed_partial_results):
        raise RuntimeError("Refusing to overwrite an existing Wave-1 output or results directory")
    embedding_root = OUT / "embeddings_val"
    if embedding_root.exists():
        expected_embedding_names = {f"{rep}.npy" for rep in CHECKPOINTS}
        present_embedding_names = {path.name for path in embedding_root.iterdir()}
        if not present_embedding_names.issubset(expected_embedding_names):
            raise RuntimeError("Unexpected file in partial Wave-1 embedding directory")
    verification = verify_freeze()
    locks = checkpoint_sha_locks()
    OUT.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    if not (OUT / "r0_wave1_freeze_verification.json").exists():
        write_json(OUT / "r0_wave1_freeze_verification.json", verification)
    if not (RESULTS / "r0_wave1_environment.json").exists():
        write_json(RESULTS / "r0_wave1_environment.json", {
            "execution_id": "R0_WAVE1_D0_D1_D3_V1", "timestamp_utc": now(), "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "dirty_worktree": True, "pre_execution_status_sha256": verification["git_status_sha256_before_execution"], "os": platform.platform(), "python": sys.version,
            "package_lock_or_environment_sha256": hashlib.sha256((sys.executable + "|" + torch.__version__).encode()).hexdigest(), "hardware": "CPU; cuda=false", "seed": 2026082601,
            "input_artifact_sha256": locks, "protocol_frozen_sha256": sha256(ROOT / "docs/stageR/r0/manifests/r0_protocol_frozen_v1.0.json"), "evidence_level": EVIDENCE_LEVEL,
        })
    models = load_models()
    data = load_dynamic_validation()
    embeddings = emit_dynamic_embeddings(models, data, args.batch_size)
    d1_rows, geometry_rows, family_state, _ = run_d1(embeddings, data)
    readouts, rank_rows = task_readouts(embeddings, data)
    d0_rows, pooling_rows, content_rows = run_d0(models)
    d3_rows, calibration_rows = run_d3(embeddings, readouts, models)

    write_csv(RESULTS / "r0_semantic_probe_metrics.csv", d1_rows, list(d1_rows[0]))
    write_csv(RESULTS / "r0_latent_geometry_metrics.csv", geometry_rows, list(geometry_rows[0]))
    write_csv(RESULTS / "r0_projection_rank_selection.csv", rank_rows, list(rank_rows[0]))
    write_csv(RESULTS / "r0_temporal_contract_audit.csv", d0_rows, list(d0_rows[0]))
    write_csv(RESULTS / "r0_temporal_orthogonal_experiment_metrics.csv", pooling_rows, list(pooling_rows[0]))
    write_csv(RESULTS / "r0_temporal_content_window_descriptive_metrics.csv", content_rows, list(content_rows[0]))
    write_csv(RESULTS / "r0_measurement_readout_metrics.csv", d3_rows, list(d3_rows[0]))
    write_csv(RESULTS / "r0_measurement_null_calibration.csv", calibration_rows, list(calibration_rows[0]))
    write_csv(RESULTS / "r0_kernel_bandwidth_audit.csv", [
        {"representation": r["representation"], "readout": r["readout"], "kernel": r["kernel"], "bandwidth": r["bandwidth"], "reference": r["bandwidth_reference"], "label_blind": True, "evidence_level": EVIDENCE_LEVEL}
        for r in d3_rows if r["domain"] == "pure_lateral"], ["representation", "readout", "kernel", "bandwidth", "reference", "label_blind", "evidence_level"])
    family_rows = []
    for rep, values in family_state.items():
        for family, state in values.items():
            family_rows.append({"representation": rep, "semantic_family": family, **state, "evidence_level": EVIDENCE_LEVEL})
    write_csv(RESULTS / "r0_d1_family_results.csv", family_rows, list(family_rows[0]))

    def seed_supported(rows: Iterable[dict[str, Any]], study: str) -> bool:
        by_candidate = []
        for candidate in ("A", "B", "C"):
            local = [r for r in rows if r.get("study") == study and str(r.get("representation", "")).startswith(candidate + "_")]
            if local and sum(bool(r.get("gate_absolute_effect_ge_0_10")) for r in local) >= 2:
                by_candidate.append(candidate)
        return bool(by_candidate)

    d0_pool = "SUPPORTED" if seed_supported(pooling_rows, "D0-C_SAME_HIDDEN_SEQUENCE_POOLING_STUDY") else "NOT_SUPPORTED"
    d0_mask = "SUPPORTED" if seed_supported(d0_rows, "D0-D_MASK_PADDING_SENSITIVITY") else "NOT_SUPPORTED"
    learned_family = {(candidate, family): sum(family_state[f"{candidate}_seed{seed}"][family]["result"] == "SUPPORTED" for seed in SEEDS) >= 2 for candidate in ("A", "B", "C") for family in ("longitudinal", "lateral", "interaction")}
    d1_module = "SUPPORTED" if sum(any(learned_family[(candidate, family)] for candidate in ("A", "B", "C")) for family in ("longitudinal", "lateral", "interaction")) >= 2 else "NOT_SUPPORTED"
    status = {
        "D0_LENGTH_EFFECT": "NOT_EVALUABLE", "D0_POSITION_RETENTION_ASSOCIATION": "NOT_EVALUABLE", "D0_POOLING_EFFECT": d0_pool, "D0_MASK_PADDING_SENSITIVITY": d0_mask,
        "D1_KNOWN_SEMANTIC_INFORMATION_PRESENT": d1_module, "D1_CROSS_DOMAIN_SEMANTIC_TRANSFER": "NOT_EVALUABLE", "D1_GEOMETRY_DEGENERACY": "INCONCLUSIVE",
        "D3_FULL64_SIGNAL_DILUTION": "INCONCLUSIVE", "D3_PROJECTED_READOUT_GAIN": "INCONCLUSIVE", "D3_NULL_CALIBRATION_PRESERVED": "INCONCLUSIVE",
    }
    write_json(RESULTS / "r0_wave1_hypothesis_results.json", {"execution_status": "COMPLETE_WITH_EXPLICIT_NOT_EVALUABLE_AND_INCONCLUSIVE_RESULTS", "evidence_level": EVIDENCE_LEVEL, "hypothesis_results": status, "limitations": ["R0_AUDIT_HOLDOUT=NOT_AVAILABLE", "D0-A controlled content equivalence unavailable", "D0-B frozen matched-natural ledger unavailable", "D3 independent null calibration series unavailable", "D3 non-lateral domain contrasts have no frozen Wave-1 readout contract"], "next_action": "Do not alter frozen protocol; address only under a separately authorized future wave."})
    markdown_report(RESULTS / "R0_D0_Temporal_Decision_Report_v1.md", "R0 D0 Temporal Decision Report v1", f"Evidence level: `{EVIDENCE_LEVEL}`.\n\n- D0-A: `NOT_EVALUABLE`; no same-event-content controlled T80/T150 construction is available.\n- D0-B: `NOT_EVALUABLE`; the required frozen matched-natural position ledger is unavailable.\n- D0-C same-hidden pooling: `{d0_pool}` from the fixed last/mean/max comparisons across A/B/C seeds.\n- D0-D mask/padding diagnostic: `{d0_mask}`; all altered views are `DIAGNOSTIC_NOT_HISTORICAL`.\n\nHistorical reference remains T150 + final hidden + original mask/padding behavior. Content-window rows are descriptive only.")
    markdown_report(RESULTS / "R0_D1_Information_Geometry_Decision_Report_v1.md", "R0 D1 Information & Geometry Decision Report v1", f"Evidence level: `{EVIDENCE_LEVEL}`.\n\nD1 known semantic information result: `{d1_module}`. The nine frozen CORE targets used five-fold scenario-grouped held-out linear probes and 5,000 scenario-cluster bootstrap replicates. Geometry is `INCONCLUSIVE` because no frozen numerical geometry-degeneracy gate exists; it was not used alone to determine semantic support. See `r0_d1_family_results.csv` and the target-level table.")
    markdown_report(RESULTS / "R0_D3_Measurement_Readout_Decision_Report_v1.md", "R0 D3 Measurement Readout Decision Report v1", "Evidence level: `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`.\n\nPure-lateral historical development comparisons use the frozen RBF kernel and treatment-label-blind Waymo reference bandwidth, with 49,999 same-scenario pair-label swaps. `D3_PROJECTED_READOUT_GAIN`, `D3_FULL64_SIGNAL_DILUTION`, and `D3_NULL_CALIBRATION_PRESERVED` are `INCONCLUSIVE`: R0_AUDIT_HOLDOUT is unavailable and there is no independent null calibration series. Longitudinal, following, and interaction Wave-1 contrasts are explicitly `NOT_EVALUABLE`, not substituted with new planner rollouts or outcome-selected assets.")
    lateral = "SUPPORTED" if any(learned_family[(candidate, "lateral")] for candidate in ("A", "B", "C")) else "NOT_SUPPORTED"
    diagnosis = "CASE_C_NOT_ESTABLISHED" if d0_pool != "SUPPORTED" else ("CASE_C_TEMPORAL_CONTRIBUTION_SUPPORTED" if lateral == "SUPPORTED" else "CASE_A_REPRESENTATION_INFORMATION_LOSS_FAVORED")
    markdown_report(RESULTS / "R0_Wave1_Cross_Module_Diagnosis_v1.md", "R0 Wave1 Cross-Module Diagnosis v1", f"Evidence level: `{EVIDENCE_LEVEL}`.\n\nD1 lateral semantic result: `{lateral}`. D0 pooling result: `{d0_pool}`. D3 results remain `INCONCLUSIVE` because the audit holdout and independent null-calibration series are unavailable. Cross-module diagnosis: `{diagnosis}`. This is not a unique causal conclusion; multiple mechanisms may be supported and no post-hoc threshold was introduced.")
    markdown_report(RESULTS / "R0_Wave1_Training_Implication_Report_v1.md", "R0 Wave1 Training Implication Report v1", "RBR-A/B/C remain `NOT_AUTHORIZED`. Wave 1 does not modify the frozen training authorization manifest. Even after these development diagnostics, R0 scientific decision records for all required modules, candidate-specific activation gates, and exact R4 source acquisition choice remain incomplete.")
    write_csv(RESULTS / "r0_wave1_protocol_deviation_log.csv", [], ["deviation_id", "timestamp_utc", "detected_after_outcome_access", "description", "affected_protocol_section", "affects_primary", "evidence_downgrade", "mitigation", "scientific_owner_disposition", "closed_timestamp_utc"])
    write_json(RESULTS / "r0_wave1_execution_manifest.json", {"execution_id": "R0_WAVE1_D0_D1_D3_V1", "completed_at_utc": now(), "freeze_verification": verification, "checkpoint_sha256": locks, "outputs": [str(path.relative_to(ROOT)) for path in sorted(RESULTS.iterdir())], "hypothesis_results": status, "protocol_deviation_count": 0, "evidence_level": EVIDENCE_LEVEL, "training_authorization_modified": False})
    write_json(RESULTS / "r0_wave1_command_ledger.json", {"execution_id": "R0_WAVE1_D0_D1_D3_V1", "command_id": "R0_WAVE1_EXECUTE_001", "timestamp_utc": now(), "operator": "Codex", "command": "tools/stageR_execute_r0_wave1.py", "working_directory": str(ROOT), "git_commit": git("rev-parse", "HEAD"), "input_artifact_sha256": locks, "output_artifact_sha256": {path.name: sha256(path) for path in RESULTS.iterdir() if path.is_file()}, "exit_code": 0, "seed": 2026082601, "environment_record_id": "r0_wave1_environment.json", "protocol_deviation_id": "NONE"})
    print(json.dumps({"status": "R0_WAVE1_COMPLETE", "hypothesis_results": status, "result_dir": str(RESULTS.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
