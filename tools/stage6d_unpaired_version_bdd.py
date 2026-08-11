#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMA_VERSION = "stage6d_unpaired_version_bdd_v1"
PASS_STATUS = "PASS_DESCRIPTIVE_STANDARDIZED_VERSION_DRIFT"
NOT_COMPARABLE_STATUS = "NOT_COMPARABLE_INSUFFICIENT_COMMON_SUPPORT"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def validate_design(design: Mapping[str, Any]) -> Dict[str, Any]:
    required = ["group_column", "groups", "row_id_column", "cluster_column", "covariates"]
    missing = [name for name in required if name not in design]
    if missing:
        raise ValueError(f"design is missing required fields: {missing}")
    groups = design["groups"]
    if not isinstance(groups, dict) or set(groups) != {"A", "B"}:
        raise ValueError("design.groups must contain exactly A and B")
    if str(groups["A"]) == str(groups["B"]):
        raise ValueError("design group labels A and B must differ")
    covariates = design["covariates"]
    if not isinstance(covariates, list) or not covariates:
        raise ValueError("design.covariates must be a non-empty list")
    seen: set = set()
    for item in covariates:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("every covariate must be an object with name")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"duplicate covariate: {name}")
        seen.add(name)
        if item.get("timing") != "pre_treatment":
            raise ValueError(f"matching covariate must be pre_treatment: {name}")
        if item.get("kind") not in {"categorical", "continuous"}:
            raise ValueError(f"covariate kind must be categorical or continuous: {name}")
        if item.get("kind") == "continuous" and int(item.get("bins", 4)) < 2:
            raise ValueError(f"continuous covariate bins must be >=2: {name}")
    forbidden = {str(x) for x in design.get("post_treatment_columns", [])}
    overlap = sorted(seen & forbidden)
    if overlap:
        raise ValueError(f"post-treatment columns cannot be matching covariates: {overlap}")
    tasks = design.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("design.tasks must be a list")
    task_names: set = set()
    for task in tasks:
        if not isinstance(task, dict) or not task.get("name") or not task.get("column"):
            raise ValueError("every task must provide name and column")
        name = str(task["name"])
        if name in task_names:
            raise ValueError(f"duplicate task name: {name}")
        task_names.add(name)
        if task.get("timing") != "pre_treatment":
            raise ValueError(f"task selection must be pre_treatment: {name}")
        if str(task["column"]) in forbidden:
            raise ValueError(f"post-treatment column cannot define task: {task['column']}")
    thresholds = {
        "min_support_fraction_per_group": 0.5,
        "min_ess_ratio_per_group": 0.2,
        "max_weight_ratio": 20.0,
        "min_clusters_per_group": 2,
        **dict(design.get("thresholds", {})),
    }
    if not 0 < float(thresholds["min_support_fraction_per_group"]) <= 1:
        raise ValueError("min_support_fraction_per_group must be in (0,1]")
    if not 0 < float(thresholds["min_ess_ratio_per_group"]) <= 1:
        raise ValueError("min_ess_ratio_per_group must be in (0,1]")
    if float(thresholds["max_weight_ratio"]) < 1:
        raise ValueError("max_weight_ratio must be >=1")
    if int(thresholds["min_clusters_per_group"]) < 2:
        raise ValueError("min_clusters_per_group must be >=2")
    result = dict(design)
    result["thresholds"] = thresholds
    result.setdefault("reference_distribution", "equal_group_pooled_common_support")
    if result["reference_distribution"] != "equal_group_pooled_common_support":
        raise ValueError("only equal_group_pooled_common_support is supported in v1")
    return result


def required_columns(design: Mapping[str, Any]) -> List[str]:
    columns = {
        str(design["group_column"]),
        str(design["row_id_column"]),
        str(design["cluster_column"]),
    }
    columns.update(str(item["name"]) for item in design["covariates"])
    columns.update(str(item["column"]) for item in design.get("tasks", []))
    return sorted(columns)


def validate_metadata(metadata: pd.DataFrame, design: Mapping[str, Any], embedding_rows: int) -> pd.DataFrame:
    missing = sorted(set(required_columns(design)) - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    row_col = str(design["row_id_column"])
    group_col = str(design["group_column"])
    cluster_col = str(design["cluster_column"])
    if metadata[row_col].isna().any():
        raise ValueError(f"row id column contains missing values: {row_col}")
    rows = pd.to_numeric(metadata[row_col], errors="raise").astype(np.int64)
    if rows.duplicated().any():
        examples = rows[rows.duplicated()].head().tolist()
        raise ValueError(f"duplicate row ids in metadata: {examples}")
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= embedding_rows):
        raise ValueError(
            f"metadata row ids exceed embedding shape: min={int(rows.min())}, "
            f"max={int(rows.max())}, embedding_rows={embedding_rows}"
        )
    labels = {str(design["groups"]["A"]), str(design["groups"]["B"])}
    observed = metadata[group_col].astype(str)
    selected = metadata.loc[observed.isin(labels)].copy()
    if selected.empty or set(selected[group_col].astype(str)) != labels:
        raise ValueError(f"metadata must contain both configured groups: {sorted(labels)}")
    if selected[cluster_col].isna().any() or (selected[cluster_col].astype(str).str.strip() == "").any():
        raise ValueError(f"cluster column contains missing/blank values: {cluster_col}")
    selected[group_col] = selected[group_col].astype(str)
    selected[cluster_col] = selected[cluster_col].astype(str)
    selected[row_col] = pd.to_numeric(selected[row_col], errors="raise").astype(np.int64)
    for item in design["covariates"]:
        name = str(item["name"])
        if item["kind"] == "continuous":
            values = pd.to_numeric(selected[name], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"continuous covariate contains missing/non-finite values: {name}")
            selected[name] = values.astype(float)
        elif selected[name].isna().any() or (selected[name].astype(str).str.strip() == "").any():
            raise ValueError(f"categorical covariate contains missing/blank values: {name}")
    return selected.reset_index(drop=True)


def load_selected_embeddings(path: Path, metadata: pd.DataFrame, row_column: str) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2:
        raise ValueError(f"embedding must be a 2D array, observed shape={values.shape}")
    rows = metadata[row_column].to_numpy(dtype=np.int64)
    selected = np.asarray(values[rows], dtype=np.float64)
    if not np.isfinite(selected).all():
        bad = int(np.size(selected) - np.isfinite(selected).sum())
        raise ValueError(f"selected embeddings contain {bad} non-finite values")
    return selected


def coarsen_covariates(
    frame: pd.DataFrame, design: Mapping[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    result = frame.copy()
    components: List[str] = []
    schema: Dict[str, Any] = {}
    for item in design["covariates"]:
        name = str(item["name"])
        output = f"_cell__{name}"
        if item["kind"] == "categorical":
            result[output] = result[name].astype(str)
            schema[name] = {"kind": "categorical", "levels": sorted(result[output].unique().tolist())}
        else:
            requested = int(item.get("bins", 4))
            values = result[name].to_numpy(dtype=float)
            quantiles = np.linspace(0, 1, requested + 1)[1:-1]
            edges = np.unique(np.quantile(values, quantiles))
            if len(edges) < 1:
                raise ValueError(f"continuous covariate cannot form at least two pooled bins: {name}")
            result[output] = np.digitize(values, edges, right=False).astype(str)
            schema[name] = {
                "kind": "continuous",
                "requested_bins": requested,
                "actual_bins": int(len(edges) + 1),
                "pooled_edges": [float(x) for x in edges],
            }
        components.append(output)
    result["_support_cell"] = result[components].astype(str).agg("|".join, axis=1)
    return result, schema


def _group_mask(frame: pd.DataFrame, design: Mapping[str, Any], label: str) -> np.ndarray:
    return frame[str(design["group_column"])].astype(str).to_numpy() == str(design["groups"][label])


def build_standardization(
    frame: pd.DataFrame,
    design: Mapping[str, Any],
    *,
    apply_thresholds: bool = True,
) -> Dict[str, Any]:
    group_col = str(design["group_column"])
    cluster_col = str(design["cluster_column"])
    group_a = str(design["groups"]["A"])
    group_b = str(design["groups"]["B"])
    counts = (
        frame.groupby(["_support_cell", group_col], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[group_a, group_b], fill_value=0)
    )
    counts.columns = ["n_A", "n_B"]
    counts["common_support"] = (counts["n_A"] > 0) & (counts["n_B"] > 0)
    support_cells = set(counts.index[counts["common_support"]].astype(str))
    support = frame["_support_cell"].astype(str).isin(support_cells).to_numpy()
    mask_a = _group_mask(frame, design, "A")
    mask_b = _group_mask(frame, design, "B")
    n_total_a, n_total_b = int(mask_a.sum()), int(mask_b.sum())
    n_support_a = int((support & mask_a).sum())
    n_support_b = int((support & mask_b).sum())
    p_a = counts.loc[counts["common_support"], "n_A"] / max(n_support_a, 1)
    p_b = counts.loc[counts["common_support"], "n_B"] / max(n_support_b, 1)
    target = 0.5 * (p_a + p_b)
    counts["target_mass"] = 0.0
    counts.loc[target.index, "target_mass"] = target
    weights = np.zeros(len(frame), dtype=float)
    for cell, target_mass in target.items():
        cell_mask = frame["_support_cell"].astype(str).to_numpy() == str(cell)
        a_rows = cell_mask & mask_a
        b_rows = cell_mask & mask_b
        weights[a_rows] = float(target_mass / max(int(a_rows.sum()), 1))
        weights[b_rows] = float(target_mass / max(int(b_rows.sum()), 1))
    for mask in (mask_a, mask_b):
        total = float(weights[mask].sum())
        if total > 0:
            weights[mask] /= total

    def diagnostics(mask: np.ndarray, n_total: int, n_support: int) -> Dict[str, Any]:
        w = weights[mask]
        positive = w[w > 0]
        ess = float(1.0 / np.sum(positive**2)) if len(positive) else 0.0
        ess_ratio = float(ess / n_support) if n_support else 0.0
        max_ratio = float(positive.max() * n_support) if len(positive) else math.inf
        return {
            "n_total": n_total,
            "n_common_support": n_support,
            "support_fraction": float(n_support / n_total) if n_total else 0.0,
            "ess": ess,
            "ess_ratio": ess_ratio,
            "max_weight_ratio": max_ratio,
            "clusters": int(frame.loc[mask, cluster_col].astype(str).nunique()),
        }

    diag_a = diagnostics(mask_a, n_total_a, n_support_a)
    diag_b = diagnostics(mask_b, n_total_b, n_support_b)
    thresholds = design["thresholds"]
    checks = {
        "common_cells_nonempty": len(support_cells) > 0,
        "support_fraction_A": diag_a["support_fraction"] >= float(thresholds["min_support_fraction_per_group"]),
        "support_fraction_B": diag_b["support_fraction"] >= float(thresholds["min_support_fraction_per_group"]),
        "ess_ratio_A": diag_a["ess_ratio"] >= float(thresholds["min_ess_ratio_per_group"]),
        "ess_ratio_B": diag_b["ess_ratio"] >= float(thresholds["min_ess_ratio_per_group"]),
        "max_weight_A": diag_a["max_weight_ratio"] <= float(thresholds["max_weight_ratio"]),
        "max_weight_B": diag_b["max_weight_ratio"] <= float(thresholds["max_weight_ratio"]),
        "clusters_A": diag_a["clusters"] >= int(thresholds["min_clusters_per_group"]),
        "clusters_B": diag_b["clusters"] >= int(thresholds["min_clusters_per_group"]),
    }
    passed = all(checks.values()) if apply_thresholds else len(support_cells) > 0
    table = counts.reset_index().rename(columns={"_support_cell": "support_cell"})
    return {
        "passed": bool(passed),
        "status": PASS_STATUS if passed else NOT_COMPARABLE_STATUS,
        "weights": weights,
        "support_mask": support,
        "cell_table": table,
        "group_A": diag_a,
        "group_B": diag_b,
        "checks": checks,
    }


def _squared_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    value = (
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1, keepdims=True).T
        - 2.0 * (a @ b.T)
    )
    return np.maximum(value, 0.0)


def median_bandwidth(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_pairs: int = 20000) -> float:
    pooled = np.vstack([x, y])
    if len(pooled) < 2:
        raise ValueError("at least two embedding rows are required for bandwidth")
    pairs = min(max_pairs, max(1, len(pooled) * 10))
    left = rng.integers(0, len(pooled), size=pairs)
    right = rng.integers(0, len(pooled), size=pairs)
    keep = left != right
    distances = np.sqrt(np.sum((pooled[left[keep]] - pooled[right[keep]]) ** 2, axis=1))
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if not len(distances):
        raise ValueError("pooled embeddings have no positive finite pair distance")
    return float(np.median(distances))


def weighted_kernel_mean(
    x: np.ndarray,
    y: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    gamma: float,
    block_size: int = 512,
) -> float:
    total = 0.0
    for start in range(0, len(x), block_size):
        xb = x[start : start + block_size]
        kernel = np.exp(-gamma * _squared_l2(xb, y))
        total += float(np.sum(kernel * wx[start : start + block_size, None] * wy[None, :]))
    return total


def weighted_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    bandwidth: float,
) -> float:
    wx = np.asarray(wx, dtype=float)
    wy = np.asarray(wy, dtype=float)
    wx = wx / wx.sum()
    wy = wy / wy.sum()
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    value = (
        weighted_kernel_mean(x, x, wx, wx, gamma)
        + weighted_kernel_mean(y, y, wy, wy, gamma)
        - 2.0 * weighted_kernel_mean(x, y, wx, wy, gamma)
    )
    return float(max(value, 0.0))


def evaluation_distribution(
    embeddings: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.int64)
    local_weights = np.asarray(weights[positions], dtype=float)
    local_weights /= local_weights.sum()
    if max_samples > 0 and len(positions) > max_samples:
        chosen = rng.choice(positions, size=max_samples, replace=True, p=local_weights)
        return embeddings[chosen], np.full(max_samples, 1.0 / max_samples)
    return embeddings[positions], local_weights


def evaluate_mmd(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    design: Mapping[str, Any],
    weights: np.ndarray,
    bandwidth: float,
    rng: np.random.Generator,
    max_samples: int,
) -> float:
    mask_a = _group_mask(frame, design, "A") & (weights > 0)
    mask_b = _group_mask(frame, design, "B") & (weights > 0)
    pos_a = np.flatnonzero(mask_a)
    pos_b = np.flatnonzero(mask_b)
    if not len(pos_a) or not len(pos_b):
        return math.nan
    xa, wa = evaluation_distribution(embeddings, pos_a, weights, rng, max_samples)
    xb, wb = evaluation_distribution(embeddings, pos_b, weights, rng, max_samples)
    return weighted_mmd2(xa, xb, wa, wb, bandwidth)


def uniform_group_weights(frame: pd.DataFrame, design: Mapping[str, Any]) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    for label in ("A", "B"):
        mask = _group_mask(frame, design, label)
        weights[mask] = 1.0 / max(int(mask.sum()), 1)
    return weights


def resample_clusters(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    design: Mapping[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray]:
    cluster_col = str(design["cluster_column"])
    pieces: List[pd.DataFrame] = []
    embedding_pieces: List[np.ndarray] = []
    for label in ("A", "B"):
        subset = frame.loc[_group_mask(frame, design, label)]
        clusters = subset[cluster_col].astype(str).unique()
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        for draw, cluster in enumerate(sampled):
            rows = subset.loc[subset[cluster_col].astype(str) == cluster].copy()
            positions = rows.index.to_numpy(dtype=np.int64)
            rows[cluster_col] = rows[cluster_col].astype(str) + f"__bootstrap_{draw}"
            pieces.append(rows)
            embedding_pieces.append(embeddings[positions])
    boot = pd.concat(pieces, ignore_index=True)
    boot_embeddings = np.vstack(embedding_pieces)
    return boot, boot_embeddings


def centered_standard_error_interval(observed: float, values: Sequence[float]) -> Tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(finite))
    standard_error = float(np.std(finite, ddof=1))
    return max(0.0, float(observed - 1.96 * standard_error)), float(observed + 1.96 * standard_error), mean, standard_error


def analyze_scope(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    design: Mapping[str, Any],
    *,
    scope: str,
    repetitions: int,
    seed: int,
    max_samples: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw_weights = uniform_group_weights(frame, design)
    raw_a = embeddings[_group_mask(frame, design, "A")]
    raw_b = embeddings[_group_mask(frame, design, "B")]
    bandwidth = median_bandwidth(raw_a, raw_b, rng)
    raw_mmd = evaluate_mmd(frame, embeddings, design, raw_weights, bandwidth, rng, max_samples)
    standardization = build_standardization(frame, design)
    standardized_mmd = math.nan
    if standardization["passed"]:
        standardized_mmd = evaluate_mmd(
            frame,
            embeddings,
            design,
            standardization["weights"],
            bandwidth,
            rng,
            max_samples,
        )
    bootstrap_rows: List[Dict[str, Any]] = []
    raw_samples: List[float] = []
    standardized_samples: List[float] = []
    if repetitions > 0:
        try:
            from tqdm import trange

            iterator = trange(repetitions, desc=f"cluster bootstrap {scope}", leave=False)
        except Exception:
            iterator = range(repetitions)
        for repetition in iterator:
            boot_frame, boot_embeddings = resample_clusters(frame, embeddings, design, rng)
            boot_raw = evaluate_mmd(
                boot_frame,
                boot_embeddings,
                design,
                uniform_group_weights(boot_frame, design),
                bandwidth,
                rng,
                max_samples,
            )
            boot_standardization = build_standardization(boot_frame, design, apply_thresholds=False)
            boot_standardized = math.nan
            if boot_standardization["passed"]:
                boot_standardized = evaluate_mmd(
                    boot_frame,
                    boot_embeddings,
                    design,
                    boot_standardization["weights"],
                    bandwidth,
                    rng,
                    max_samples,
                )
            raw_samples.append(boot_raw)
            standardized_samples.append(boot_standardized)
            bootstrap_rows.extend([
                {"scope": scope, "repetition": repetition, "estimand": "raw_observed_mixture", "mmd2": boot_raw},
                {"scope": scope, "repetition": repetition, "estimand": "common_support_standardized", "mmd2": boot_standardized},
            ])
    raw_low, raw_high, raw_bootstrap_mean, raw_bootstrap_se = centered_standard_error_interval(raw_mmd, raw_samples)
    std_low, std_high, std_bootstrap_mean, std_bootstrap_se = centered_standard_error_interval(standardized_mmd, standardized_samples)
    return {
        "scope": scope,
        "status": standardization["status"],
        "n_A": int(_group_mask(frame, design, "A").sum()),
        "n_B": int(_group_mask(frame, design, "B").sum()),
        "bandwidth": bandwidth,
        "raw_mmd2": raw_mmd,
        "raw_cluster_bootstrap_ci95_low": raw_low,
        "raw_cluster_bootstrap_ci95_high": raw_high,
        "raw_cluster_bootstrap_mean": raw_bootstrap_mean,
        "raw_cluster_bootstrap_standard_error": raw_bootstrap_se,
        "standardized_mmd2": standardized_mmd,
        "standardized_cluster_bootstrap_ci95_low": std_low,
        "standardized_cluster_bootstrap_ci95_high": std_high,
        "standardized_cluster_bootstrap_mean": std_bootstrap_mean,
        "standardized_cluster_bootstrap_standard_error": std_bootstrap_se,
        "cluster_bootstrap_ci_method": "observed_plus_or_minus_1.96_times_cluster_bootstrap_standard_error_clipped_at_zero",
        "bootstrap_valid_standardized": int(np.isfinite(np.asarray(standardized_samples, dtype=float)).sum()),
        "bootstrap_repetitions": repetitions,
        "standardization": standardization,
        "bootstrap_rows": bootstrap_rows,
    }


def task_mask(frame: pd.DataFrame, task: Mapping[str, Any]) -> np.ndarray:
    column = str(task["column"])
    if "positive_values" in task:
        allowed = {str(value) for value in task["positive_values"]}
        return frame[column].astype(str).isin(allowed).to_numpy()
    return frame[column].map(parse_bool).to_numpy(dtype=bool)


def balance_table(
    frame: pd.DataFrame,
    design: Mapping[str, Any],
    weights: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mask_a = _group_mask(frame, design, "A")
    mask_b = _group_mask(frame, design, "B")
    support = weights > 0

    def weighted_average(values: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any() or float(weights[mask].sum()) <= 0:
            return math.nan
        return float(np.average(values[mask], weights=weights[mask]))

    for item in design["covariates"]:
        name = str(item["name"])
        if item["kind"] == "continuous":
            values = frame[name].to_numpy(dtype=float)
            pooled_sd = float(np.std(values, ddof=1))
            raw_a = float(np.mean(values[mask_a]))
            raw_b = float(np.mean(values[mask_b]))
            weighted_a = weighted_average(values, mask_a & support)
            weighted_b = weighted_average(values, mask_b & support)
            rows.append({
                "covariate": name,
                "kind": "continuous",
                "level": "",
                "raw_A": raw_a,
                "raw_B": raw_b,
                "raw_difference": raw_b - raw_a,
                "standardized_A": weighted_a,
                "standardized_B": weighted_b,
                "standardized_difference": weighted_b - weighted_a,
                "raw_smd": (raw_b - raw_a) / pooled_sd if pooled_sd > 0 else 0.0,
                "standardized_smd": (weighted_b - weighted_a) / pooled_sd if pooled_sd > 0 else 0.0,
            })
        else:
            values = frame[name].astype(str).to_numpy()
            for level in sorted(set(values)):
                indicator = values == level
                raw_a = float(indicator[mask_a].mean())
                raw_b = float(indicator[mask_b].mean())
                weighted_a = float(np.sum(weights[mask_a & support] * indicator[mask_a & support])) if (mask_a & support).any() else math.nan
                weighted_b = float(np.sum(weights[mask_b & support] * indicator[mask_b & support])) if (mask_b & support).any() else math.nan
                rows.append({
                    "covariate": name,
                    "kind": "categorical",
                    "level": level,
                    "raw_A": raw_a,
                    "raw_B": raw_b,
                    "raw_difference": raw_b - raw_a,
                    "standardized_A": weighted_a,
                    "standardized_B": weighted_b,
                    "standardized_difference": weighted_b - weighted_a,
                    "raw_smd": math.nan,
                    "standardized_smd": math.nan,
                })
    return rows


def task_frequency_rows(
    frame: pd.DataFrame,
    design: Mapping[str, Any],
    weights: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    mask_a = _group_mask(frame, design, "A")
    mask_b = _group_mask(frame, design, "B")
    support = weights > 0
    for task in design.get("tasks", []):
        selected = task_mask(frame, task)
        raw_a = float(selected[mask_a].mean())
        raw_b = float(selected[mask_b].mean())
        weighted_a = float(np.sum(weights[mask_a & support] * selected[mask_a & support])) if (mask_a & support).any() else math.nan
        weighted_b = float(np.sum(weights[mask_b & support] * selected[mask_b & support])) if (mask_b & support).any() else math.nan
        rows.append({
            "task": str(task["name"]),
            "timing": str(task["timing"]),
            "raw_frequency_A": raw_a,
            "raw_frequency_B": raw_b,
            "raw_B_minus_A": raw_b - raw_a,
            "standardized_frequency_A": weighted_a,
            "standardized_frequency_B": weighted_b,
            "standardized_B_minus_A": weighted_b - weighted_a,
        })
    return rows


def compact_scope(scope: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in scope.items() if key not in {"standardization", "bootstrap_rows"}}


def write_report(output_dir: Path, summary: Mapping[str, Any], task_rows: Sequence[Mapping[str, Any]]) -> None:
    overall = summary["overall"]
    lines = [
        "# Stage 6D unpaired software-version BDD report",
        "",
        f"- status: `{summary['status']}`",
        f"- raw observed-mixture MMD²: `{overall['raw_mmd2']:.8g}`",
        f"- common-support standardized MMD²: `{overall['standardized_mmd2']:.8g}`" if np.isfinite(overall["standardized_mmd2"]) else "- common-support standardized MMD²: `not estimated`",
        f"- group A support fraction: `{summary['support']['group_A']['support_fraction']:.4f}`",
        f"- group B support fraction: `{summary['support']['group_B']['support_fraction']:.4f}`",
        "",
        "## Interpretation boundary",
        "",
        "Raw BDD measures the behavior distribution actually observed in both road-test collections and includes ODD/exposure composition shift. Standardized BDD compares both versions under the same equal-group pooled common-support cell distribution. It remains observational and may contain residual or unmeasured confounding; it is not a safety, performance, or causal certification result.",
        "",
        "## Task-conditioned results",
        "",
        "| task | status | n_A | n_B | raw MMD² | standardized MMD² | standardized cluster-bootstrap 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in task_rows:
        std = row["standardized_mmd2"]
        std_text = f"{std:.6g}" if np.isfinite(std) else "NA"
        ci = (
            f"[{row['standardized_cluster_bootstrap_ci95_low']:.6g}, {row['standardized_cluster_bootstrap_ci95_high']:.6g}]"
            if np.isfinite(row["standardized_cluster_bootstrap_ci95_low"])
            else "NA"
        )
        lines.append(f"| {row['scope']} | {row['status']} | {row['n_A']} | {row['n_B']} | {row['raw_mmd2']:.6g} | {std_text} | {ci} |")
    if not task_rows:
        lines.append("| none configured | — | — | — | — | — | — |")
    lines.extend([
        "",
        "## Required operational use",
        "",
        "Use same-version historical A/A runs to calibrate an alert threshold. Do not interpret an absolute MMD² cutoff as universal. Report task-frequency shift separately from within-task behavior shift, and return NOT_COMPARABLE for ODD cells without common support.",
    ])
    (output_dir / "stage6d_unpaired_version_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.bootstrap_repetitions < 0:
        raise ValueError("--bootstrap_repetitions must be >=0")
    if args.max_mmd_samples < 1:
        raise ValueError("--max_mmd_samples must be >=1")
    design = validate_design(read_json(args.design_json))
    raw_embedding = np.load(args.embedding_path, mmap_mode="r")
    if raw_embedding.ndim != 2:
        raise ValueError(f"embedding must be 2D: {raw_embedding.shape}")
    metadata_raw = pd.read_csv(args.metadata_csv)
    metadata = validate_metadata(metadata_raw, design, int(raw_embedding.shape[0]))
    metadata, coarsening_schema = coarsen_covariates(metadata, design)
    row_col = str(design["row_id_column"])
    embeddings = load_selected_embeddings(args.embedding_path, metadata, row_col)
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    overall = analyze_scope(
        metadata,
        embeddings,
        design,
        scope="overall",
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        max_samples=args.max_mmd_samples,
    )
    standardization = overall["standardization"]
    cell_table = standardization["cell_table"]
    cell_table.to_csv(output_dir / "common_support_cells.csv", index=False)
    row_weights = metadata[[row_col, str(design["group_column"]), str(design["cluster_column"]), "_support_cell"]].copy()
    row_weights["in_common_support"] = standardization["support_mask"]
    row_weights["standardization_weight"] = standardization["weights"]
    row_weights.to_csv(output_dir / "standardization_row_weights.csv", index=False)

    balance = balance_table(metadata, design, standardization["weights"])
    pd.DataFrame(balance).to_csv(output_dir / "covariate_balance.csv", index=False)
    frequencies = task_frequency_rows(metadata, design, standardization["weights"])
    pd.DataFrame(frequencies).to_csv(output_dir / "task_frequency_shift.csv", index=False)

    task_scopes = []
    all_bootstrap = list(overall["bootstrap_rows"])
    for position, task in enumerate(design.get("tasks", [])):
        mask = task_mask(metadata, task)
        subset = metadata.loc[mask].copy().reset_index(drop=True)
        subset_embeddings = embeddings[mask]
        if not len(subset) or set(subset[str(design["group_column"])].astype(str)) != {
            str(design["groups"]["A"]), str(design["groups"]["B"])
        }:
            task_scopes.append({
                "scope": str(task["name"]),
                "status": NOT_COMPARABLE_STATUS,
                "n_A": int((_group_mask(subset, design, "A")).sum()) if len(subset) else 0,
                "n_B": int((_group_mask(subset, design, "B")).sum()) if len(subset) else 0,
                "bandwidth": math.nan,
                "raw_mmd2": math.nan,
                "raw_cluster_bootstrap_ci95_low": math.nan,
                "raw_cluster_bootstrap_ci95_high": math.nan,
                "raw_cluster_bootstrap_mean": math.nan,
                "raw_cluster_bootstrap_standard_error": math.nan,
                "standardized_mmd2": math.nan,
                "standardized_cluster_bootstrap_ci95_low": math.nan,
                "standardized_cluster_bootstrap_ci95_high": math.nan,
                "standardized_cluster_bootstrap_mean": math.nan,
                "standardized_cluster_bootstrap_standard_error": math.nan,
                "cluster_bootstrap_ci_method": "observed_plus_or_minus_1.96_times_cluster_bootstrap_standard_error_clipped_at_zero",
                "bootstrap_valid_standardized": 0,
                "bootstrap_repetitions": args.bootstrap_repetitions,
            })
            continue
        task_scope = analyze_scope(
            subset,
            subset_embeddings,
            design,
            scope=str(task["name"]),
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 1000 + position,
            max_samples=args.max_mmd_samples,
        )
        all_bootstrap.extend(task_scope["bootstrap_rows"])
        task_scopes.append(compact_scope(task_scope))

    overall_compact = compact_scope(overall)
    pd.DataFrame([overall_compact]).to_csv(output_dir / "overall_bdd_summary.csv", index=False)
    pd.DataFrame(task_scopes).to_csv(output_dir / "task_bdd_summary.csv", index=False)
    pd.DataFrame(all_bootstrap).to_csv(output_dir / "cluster_bootstrap_mmd_samples.csv", index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": standardization["status"],
        "dataset_role": "UNPAIRED_OBSERVATIONAL_SOFTWARE_VERSION_COMPARISON",
        "interpretation_role": "DESCRIPTIVE_ODD_STANDARDIZED_VERSION_DRIFT_NOT_CAUSAL_OR_SAFETY_CERTIFICATION",
        "design": design,
        "coarsening_schema": coarsening_schema,
        "overall": overall_compact,
        "support": {
            "group_A": standardization["group_A"],
            "group_B": standardization["group_B"],
            "checks": standardization["checks"],
            "common_support_cells": int(cell_table["common_support"].map(parse_bool).sum()),
            "total_cells": int(len(cell_table)),
        },
        "tasks": task_scopes,
        "task_frequency_shift": frequencies,
        "limitations": [
            "Standardization controls only measured pre-treatment covariates in the frozen design.",
            "Residual and unmeasured confounding may remain; standardized BDD is not a causal effect.",
            "Cluster bootstrap intervals quantify collection-cluster uncertainty, not a universal alert threshold.",
            "Calibrate operational alerts against independent same-version A/A historical baselines.",
            "BDD is not a safety, performance, or release certification metric.",
        ],
    }
    write_json(output_dir / "stage6d_unpaired_version_summary.json", summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "embedding": {"path": str(args.embedding_path.resolve()), "sha256": sha256_file(args.embedding_path)},
            "metadata": {"path": str(args.metadata_csv.resolve()), "sha256": sha256_file(args.metadata_csv)},
            "design": {"path": str(args.design_json.resolve()), "sha256": sha256_file(args.design_json)},
        },
        "parameters": {
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "max_mmd_samples": args.max_mmd_samples,
        },
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    }
    write_json(output_dir / "stage6d_reproducibility_provenance.json", provenance)
    write_report(output_dir, summary, task_scopes)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Common-support-standardized unpaired software-version BDD with cluster bootstrap."
    )
    parser.add_argument("--embedding_path", type=Path, required=True, help="Aligned 2D embedding .npy; read with mmap.")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Row-aligned version/ODD/cluster metadata CSV.")
    parser.add_argument("--design_json", type=Path, required=True, help="Frozen pre-treatment design and support thresholds.")
    parser.add_argument("--output_dir", type=Path, required=True, help="New output directory; overwrite is forbidden.")
    parser.add_argument("--bootstrap_repetitions", type=int, default=1000, help="Cluster bootstrap repetitions.")
    parser.add_argument("--max_mmd_samples", type=int, default=2000, help="Maximum evaluated rows per group and replicate.")
    parser.add_argument("--seed", type=int, default=20260809, help="Fixed random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
