#!/usr/bin/env python3
"""One-time frozen Waymo Dynamic-v2 test for old64 and Stage6U A/B/C.

This script intentionally has no training code.  It verifies the blind-evaluation
authorization, evaluates every frozen test row once, and freezes the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices  # noqa: E402
from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder  # noqa: E402


AUTH = ROOT / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"
DATA_MANIFEST = ROOT / "outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json"
GLOBAL33 = ROOT / "outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_global_interaction_target_standardization.json"
PROTOCOL = ROOT / "configs/stage6t_training_evaluation_protocol.json"
LEDGER = ROOT / "outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"
OLD_CKPT = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt"
DEFAULT_OUT = ROOT / "outputs/stage6v_waymo_dynamic_v2_test_v1"

CATEGORIES = {
    "longitudinal_comfort": ["rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk"],
    "following_interaction": [
        "mean_thw", "min_thw", "mean_front_distance", "min_front_distance",
        "mean_rel_speed", "p95_rel_speed", "front_pressure_score", "rear_vehicle_pressure_proxy",
    ],
    "lateral_lane_dynamics": [
        "rms_yaw_rate", "rms_curvature", "heading_change_total", "lane_change_count_proxy",
        "lane_change_rate_proxy", "max_lateral_speed", "rms_lateral_accel",
        "lane_change_oscillation_score_proxy", "left_front_min_gap", "left_rear_min_gap",
        "right_front_min_gap", "right_rear_min_gap", "left_gap_min", "right_gap_min",
        "left_gap_acceptance_proxy", "right_gap_acceptance_proxy",
    ],
    "behavior_proxy": ["yielding_score_proxy", "assertiveness_score_proxy"],
}
PSEUDO = [
    "rms_jerk", "mean_thw", "min_thw", "mean_front_distance", "min_front_distance",
    "mean_rel_speed", "p95_rel_speed", "rms_yaw_rate", "rms_curvature",
    "front_pressure_score", "yielding_score_proxy", "assertiveness_score_proxy",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def feature_names(manifest: dict[str, Any]) -> list[str]:
    schema = read_json(Path(manifest["part_roots"][0]) / "feature_schema.json")
    names = [str(row["name"]) for row in sorted(schema["features"], key=lambda row: int(row["index"]))]
    if len(names) != 33:
        raise RuntimeError(f"Expected 33 features, got {len(names)}")
    return names


def verify_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(AUTH) != AUTH_SHA:
        raise RuntimeError("Blind evaluation authorization SHA mismatch")
    auth = read_json(AUTH)
    if auth.get("status") != "AUTHORIZED_STAGE6_ONE_TIME_BLIND_EVALUATION":
        raise RuntimeError("Blind evaluation is not authorized")
    if auth.get("immutability_statement") != "evaluation results cannot trigger retraining or protocol changes":
        raise RuntimeError("Required immutability statement is absent")
    ledger = read_json(LEDGER)
    if ledger.get("status") != "LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK" or len(ledger.get("rows", [])) != 9:
        raise RuntimeError("Checkpoint ledger is not locked 9/9")
    for row in ledger["rows"]:
        checkpoint = Path(row["best_checkpoint_path"])
        if sha256(checkpoint) != row["best_checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint}")
    protocol = read_json(PROTOCOL)
    if protocol["checkpoint_selection"]["test_split_role"] != "one_time_post_lock_noninferiority_only":
        raise RuntimeError("Stage6T Waymo-test rule mismatch")
    return auth, ledger


def load_test_inputs(manifest: dict[str, Any], stats: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    scenarios: list[np.ndarray] = []
    shard_sizes: list[int] = []
    for shard_path in manifest["shard_paths"]:
        shard = Path(shard_path)
        split = np.load(shard / "split.npy", allow_pickle=True).astype(str)
        ids = np.flatnonzero(split == "test")
        if not len(ids):
            continue
        meta = np.load(shard / "meta.npy", allow_pickle=True)
        context = np.asarray(np.load(shard / "context_traj.npy", mmap_mode="r")[ids], dtype=np.float32)
        raw = np.asarray(np.load(shard / "interaction_feat_style_raw.npy", mmap_mode="r")[ids], dtype=np.float32)
        if context.shape[1:] != (80, 83) or raw.shape[1:] != (33,):
            raise RuntimeError(f"Invalid Dynamic-v2 test shape in {shard}")
        if not np.isfinite(context).all() or not np.isfinite(raw).all():
            raise RuntimeError(f"Non-finite Dynamic-v2 test input in {shard}")
        contexts.append(context)
        raw_rows.append(raw)
        scenarios.append(meta[ids]["scenario_id"].astype(str))
        shard_sizes.append(len(ids))
    raw_all = np.concatenate(raw_rows)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(stats["std"], dtype=np.float32), float(stats["epsilon_floor"]))
    standardized = (raw_all - mean) / std
    scenario_all = np.concatenate(scenarios).astype(str)
    return contexts, standardized, scenario_all, np.asarray(shard_sizes, dtype=np.int64)


def build_models(ledger: dict[str, Any], names: list[str], device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    old = ContextFlattenGRUEncoder(input_dim=83, hidden_dim=128, embedding_dim=64)
    old_payload = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    old.load_state_dict(old_payload["model"])
    models["old64"] = old.eval().to(device)
    groups = feature_group_indices(names)
    for row in ledger["rows"]:
        label = f"{row['candidate']}_{row['seed']}"
        model = UnifiedABCModel(str(row["candidate"]), groups)
        payload = torch.load(Path(row["best_checkpoint_path"]), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        models[label] = model.eval().to(device)
    return models


def embed_all(models: dict[str, torch.nn.Module], contexts: list[np.ndarray], device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in models}
    with torch.no_grad():
        for shard_id, context in enumerate(contexts):
            for start in range(0, len(context), batch_size):
                x = torch.from_numpy(context[start : start + batch_size]).to(device)
                for name, model in models.items():
                    z = model(x).detach().cpu().numpy().astype(np.float32)
                    if z.shape[1:] != (64,) or not np.isfinite(z).all():
                        raise RuntimeError(f"Invalid embedding from {name} at test shard {shard_id}")
                    chunks[name].append(z)
    return {name: np.concatenate(values) for name, values in chunks.items()}


def standardized_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average").astype(np.float64)
    return (ranks - ranks.mean()) / max(ranks.std(ddof=0), 1e-12)


def cluster_sums(values: np.ndarray, cluster_index: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.bincount(cluster_index, weights=values, minlength=n_clusters).astype(np.float64)


def nearest_neighbor_rows(embedding: np.ndarray, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean", n_jobs=-1).fit(embedding)
    neighbors = nn.kneighbors(return_distance=False)[:, 1:6]
    hit5 = np.any(labels[neighbors] == labels[:, None], axis=1).astype(np.float64)
    neighbor_distance = np.linalg.norm(features[:, None, :] - features[neighbors], axis=2).mean(axis=1)
    return hit5, neighbor_distance.astype(np.float64)


def bootstrap_intervals(
    cluster_metric_sums: dict[str, np.ndarray], cluster_counts: np.ndarray, repetitions: int, seed: int
) -> dict[str, tuple[float, float]]:
    n_clusters = len(cluster_counts)
    keys = list(cluster_metric_sums)
    matrix = np.stack([cluster_metric_sums[key] for key in keys], axis=1)
    samples: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    batch = 250
    probability = np.full(n_clusters, 1.0 / n_clusters)
    for start in range(0, repetitions, batch):
        size = min(batch, repetitions - start)
        counts = rng.multinomial(n_clusters, probability, size=size).astype(np.float64)
        denominator = np.maximum(counts @ cluster_counts, 1.0)
        estimates = (counts @ matrix) / denominator[:, None]
        for column, key in enumerate(keys):
            samples[key].append(estimates[:, column])
    result: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = np.concatenate(samples[key])
        result[key] = (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty frozen output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    auth, ledger = verify_authorization()
    protocol = read_json(PROTOCOL)
    manifest = read_json(DATA_MANIFEST)
    stats = read_json(GLOBAL33)
    names = feature_names(manifest)
    fmap = {name: index for index, name in enumerate(names)}
    contexts, features, scenario_ids, shard_sizes = load_test_inputs(manifest, stats)
    if len(features) != int(manifest["split_counts"]["test"]):
        raise RuntimeError("Dynamic-v2 test row count mismatch")
    unique_scenarios, cluster_index = np.unique(scenario_ids, return_inverse=True)
    n_clusters = len(unique_scenarios)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = build_models(ledger, names, device)
    embeddings = embed_all(models, contexts, device, args.batch_size)
    del contexts
    emb_dir = out / "embeddings"
    emb_dir.mkdir()
    embedding_shas: dict[str, str] = {}
    for name, value in embeddings.items():
        path = emb_dir / f"{name}.npy"
        np.save(path, value)
        embedding_shas[name] = sha256(path)

    # Common deterministic pair stream.  Cluster membership is the first-row
    # scenario; cluster bootstrap therefore preserves the full query-side block.
    bootstrap_spec = protocol["waymo_test_scorecard"]
    seed = int(bootstrap_spec["bootstrap_seed"])
    rng = np.random.default_rng(seed)
    n_pairs = 100_000
    pair_i = rng.integers(0, len(features), size=n_pairs)
    pair_j = rng.integers(0, len(features), size=n_pairs)
    valid = pair_i != pair_j
    pair_i, pair_j = pair_i[valid], pair_j[valid]
    pair_cluster = cluster_index[pair_i]
    pair_cluster_counts = np.bincount(pair_cluster, minlength=n_clusters).astype(np.float64)

    target_rank: dict[str, np.ndarray] = {}
    for feature in sorted({feature for values in CATEGORIES.values() for feature in values}):
        delta = np.abs(features[pair_i, fmap[feature]] - features[pair_j, fmap[feature]])
        target_rank[feature] = standardized_rank(delta)

    corr_rows: list[dict[str, Any]] = []
    category_pair_contrib: dict[tuple[str, str], np.ndarray] = {}
    for representation, embedding in embeddings.items():
        distance = np.linalg.norm(embedding[pair_i] - embedding[pair_j], axis=1)
        distance_rank = standardized_rank(distance)
        for category, feature_list in CATEGORIES.items():
            contribution = np.mean([distance_rank * target_rank[feature] for feature in feature_list], axis=0)
            category_pair_contrib[(representation, category)] = contribution
            corr_rows.append({
                "representation": representation,
                "category": category,
                "mean_spearman": float(contribution.mean()),
                "feature_count": len(feature_list),
                "pair_count": len(pair_i),
            })

    pseudo_idx = [fmap[name] for name in PSEUDO]
    pseudo = features[:, pseudo_idx]
    quantiles = np.quantile(pseudo, [0.33, 0.66], axis=0)
    bins = (pseudo > quantiles[0]).astype(np.int8) + (pseudo > quantiles[1]).astype(np.int8)
    labels = np.asarray(["_".join(map(str, row)) for row in bins], dtype=str)
    retrieval_rows: list[dict[str, Any]] = []
    retrieval_values: dict[tuple[str, str], np.ndarray] = {}
    for representation, embedding in embeddings.items():
        hit5, neighbor_distance = nearest_neighbor_rows(embedding, features, labels)
        retrieval_values[(representation, "hit_at_5")] = hit5
        retrieval_values[(representation, "neighbor_feature_distance")] = neighbor_distance
        retrieval_rows.append({
            "representation": representation,
            "hit_at_5": float(hit5.mean()),
            "mean_neighbor_feature_distance": float(neighbor_distance.mean()),
        })

    # Build candidate-minus-old cluster sums.  The point estimate is identical
    # to the common-pair/global-row statistic; bootstrap only changes cluster weights.
    cluster_metrics: dict[str, np.ndarray] = {}
    for representation in embeddings:
        if representation == "old64":
            continue
        for category in CATEGORIES:
            delta = category_pair_contrib[(representation, category)] - category_pair_contrib[("old64", category)]
            key = f"{representation}|{category}|delta"
            cluster_metrics[key] = cluster_sums(delta, pair_cluster, n_clusters)
        hit_delta = retrieval_values[(representation, "hit_at_5")] - retrieval_values[("old64", "hit_at_5")]
        cluster_metrics[f"{representation}|hit_at_5|delta"] = cluster_sums(hit_delta, cluster_index, n_clusters)
        # Ratio gate is represented as candidate-minus-1.05*old. <=0 passes.
        distance_margin = retrieval_values[(representation, "neighbor_feature_distance")] - 1.05 * retrieval_values[("old64", "neighbor_feature_distance")]
        cluster_metrics[f"{representation}|neighbor_feature_distance|margin"] = cluster_sums(distance_margin, cluster_index, n_clusters)

    # Correlation metrics use pair counts; retrieval metrics use row counts.
    row_cluster_counts = np.bincount(cluster_index, minlength=n_clusters).astype(np.float64)
    corr_metrics = {key: value for key, value in cluster_metrics.items() if "|delta" in key and "hit_at_5" not in key}
    retrieval_metrics = {key: value for key, value in cluster_metrics.items() if key not in corr_metrics}
    ci = bootstrap_intervals(corr_metrics, pair_cluster_counts, int(bootstrap_spec["bootstrap_repetitions"]), seed)
    ci.update(bootstrap_intervals(retrieval_metrics, row_cluster_counts, int(bootstrap_spec["bootstrap_repetitions"]), seed + 1))

    corr_df = pd.DataFrame(corr_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    corr_df.to_csv(out / "category_spearman.csv", index=False)
    retrieval_df.to_csv(out / "retrieval_metrics.csv", index=False)
    old_corr = corr_df[corr_df.representation == "old64"].set_index("category")["mean_spearman"]
    old_retrieval = retrieval_df[retrieval_df.representation == "old64"].iloc[0]
    criteria = bootstrap_spec["relative_to_same_dynamic_test_old64"]
    decision_rows: list[dict[str, Any]] = []
    for representation in [name for name in embeddings if name != "old64"]:
        row: dict[str, Any] = {"representation": representation}
        candidate_corr = corr_df[corr_df.representation == representation].set_index("category")["mean_spearman"]
        all_noninferior = True
        for category, margin_key in [
            ("following_interaction", "following_spearman_noninferiority_margin"),
            ("lateral_lane_dynamics", "lateral_spearman_noninferiority_margin"),
            ("behavior_proxy", "behavior_proxy_spearman_noninferiority_margin"),
        ]:
            delta = float(candidate_corr[category] - old_corr[category])
            row[f"{category}_delta"] = delta
            row[f"{category}_pass"] = bool(delta >= float(criteria[margin_key]))
            all_noninferior &= row[f"{category}_pass"]
        long_delta = float(candidate_corr["longitudinal_comfort"] - old_corr["longitudinal_comfort"])
        long_ci = ci[f"{representation}|longitudinal_comfort|delta"]
        row["longitudinal_delta"] = long_delta
        row["longitudinal_delta_ci95_lower"] = long_ci[0]
        row["longitudinal_delta_ci95_upper"] = long_ci[1]
        row["longitudinal_improvement_pass"] = bool(
            long_delta >= float(criteria["minimum_longitudinal_comfort_spearman_delta"])
            and long_ci[0] > float(criteria["longitudinal_delta_ci_lower_must_exceed"])
        )
        candidate_retrieval = retrieval_df[retrieval_df.representation == representation].iloc[0]
        hit_delta = float(candidate_retrieval.hit_at_5 - old_retrieval.hit_at_5)
        ratio = float(candidate_retrieval.mean_neighbor_feature_distance / old_retrieval.mean_neighbor_feature_distance)
        row["retrieval_hit_at_5_delta"] = hit_delta
        row["retrieval_hit_at_5_pass"] = bool(hit_delta >= float(criteria["retrieval_hit_at_5_noninferiority_margin"]))
        row["neighbor_feature_distance_ratio"] = ratio
        row["neighbor_feature_distance_pass"] = bool(ratio <= float(criteria["mean_neighbor_feature_distance_max_ratio"]))
        all_noninferior &= row["retrieval_hit_at_5_pass"] and row["neighbor_feature_distance_pass"]
        row["noninferiority_pass"] = bool(all_noninferior)
        row["all_waymo_gates_pass"] = bool(all_noninferior and row["longitudinal_improvement_pass"])
        decision_rows.append(row)
    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(out / "waymo_test_decisions.csv", index=False)

    seed_summary: list[dict[str, Any]] = []
    for candidate in ("A", "B", "C"):
        subset = decision_df[decision_df.representation.str.startswith(candidate + "_")]
        primary = subset[subset.representation == f"{candidate}_3407"].iloc[0]
        seed_summary.append({
            "candidate": candidate,
            "primary_seed": 3407,
            "primary_all_waymo_gates_pass": bool(primary.all_waymo_gates_pass),
            "seed_noninferiority_pass_count": int(subset.noninferiority_pass.sum()),
            "seed_all_waymo_gates_pass_count": int(subset.all_waymo_gates_pass.sum()),
            "stage6t_seed_stability_pass": bool(primary.all_waymo_gates_pass and subset.noninferiority_pass.sum() >= 2),
        })
    seed_df = pd.DataFrame(seed_summary)
    seed_df.to_csv(out / "candidate_seed_stability.csv", index=False)

    source_files = {
        "blind_authorization": {"path": str(AUTH), "sha256": sha256(AUTH)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "data_manifest": {"path": str(DATA_MANIFEST), "sha256": sha256(DATA_MANIFEST)},
        "global33": {"path": str(GLOBAL33), "sha256": sha256(GLOBAL33)},
        "checkpoint_ledger": {"path": str(LEDGER), "sha256": sha256(LEDGER)},
        "old64_checkpoint": {"path": str(OLD_CKPT), "sha256": sha256(OLD_CKPT)},
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    result_manifest = {
        "schema_version": "stage6v_waymo_test_v1",
        "status": "FROZEN_WAYMO_DYNAMIC_V2_TEST_COMPLETE",
        "immutability_statement": auth["immutability_statement"],
        "test_row_count": int(len(features)),
        "scenario_cluster_count": int(n_clusters),
        "test_shard_count": int(len(shard_sizes)),
        "pair_count": int(len(pair_i)),
        "bootstrap_repetitions": int(bootstrap_spec["bootstrap_repetitions"]),
        "bootstrap_seed": seed,
        "primary_seed": 3407,
        "representations": list(embeddings),
        "embedding_sha256": embedding_shas,
        "source_files": source_files,
        "result_files": {},
        "training_or_protocol_modified": False,
        "waymo_test_used_for_model_selection": False,
        "nuplan_read_or_run": False,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
    }
    for name in ("category_spearman.csv", "retrieval_metrics.csv", "waymo_test_decisions.csv", "candidate_seed_stability.csv"):
        result_manifest["result_files"][name] = sha256(out / name)
    write_json(out / "stage6v_waymo_test_result_manifest.json", result_manifest)

    report_lines = [
        "# Stage6V Waymo Dynamic v2 test 盲测报告",
        "",
        f"- 状态：`{result_manifest['status']}`",
        f"- test：{len(features)} 行，{n_clusters} 个 scenario cluster",
        f"- primary seed：3407；其余 seed 仅用于稳定性",
        f"- bootstrap：{bootstrap_spec['bootstrap_repetitions']} 次，seed={seed}",
        "- 冻结约束：evaluation results cannot trigger retraining or protocol changes",
        "",
        "## 冻结门禁结果",
        "",
        decision_df.to_markdown(index=False),
        "",
        "## Seed 稳定性",
        "",
        seed_df.to_markdown(index=False),
        "",
        "说明：各 candidate 的 total loss 不作横向比较；所有差值均相对同一 Dynamic-v2 test 上重新评估的 old64。",
    ]
    (out / "stage6v_waymo_test_report_zh.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result_manifest["status"],
        "output_dir": str(out),
        "result_manifest_sha256": sha256(out / "stage6v_waymo_test_result_manifest.json"),
        "candidate_seed_stability": seed_summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
