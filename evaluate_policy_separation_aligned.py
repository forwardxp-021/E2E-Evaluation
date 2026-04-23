"""Source-aligned policy separation evaluation for synthetic rollout benchmark.

For each source_index, exactly one sample per policy exists.  This script
factors out scenario-distribution variance by comparing policies *under the
same source window*, producing more controlled "policy-only" metrics.

Computes:
  1. Pairwise inter-policy distances within the same source window:
       For each source group (one sample per policy), compute all P*(P-1)/2
       cosine distances between policy embeddings.  Report mean / median / p25 /
       p75 distribution across sources.
  2. Source-aligned retrieval:
       For each sample, restrict the candidate pool to the other P-1 samples
       sharing the same source_index.  By construction the nearest neighbour
       always belongs to a different policy (100%), so the key metric is
       within-group policy-centroid assignment accuracy.
  3. Within-source clustering score:
       ratio = mean_within_source_inter_policy_dist / mean_cross_source_same_policy_dist
       Higher → policies are more separated within a source than the same policy
       varies across different source windows (stronger controlled signal).

Outputs written to --analysis_dir:
    policy_separation_aligned_summary.json  – all summary metrics
    policy_pairwise_dist.csv                – per-source, per-policy-pair distances

Usage example:
    python evaluate_policy_separation_aligned.py \\
        --embeddings_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \\
        --policy_id_path  output_policy_rollouts/policy_id.npy \\
        --source_index_path output_policy_rollouts/source_index.npy \\
        --split_path      output_policy_rollouts/split.npy \\
        --policy_names_path output_policy_rollouts/policy_names.json \\
        --eval_split test \\
        --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_policy_aligned
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity) between two 1-D vectors."""
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(an, bn))


def _policy_centroids(
    embeddings: np.ndarray,
    policy_id: np.ndarray,
    unique_ids: list[int],
) -> dict[int, np.ndarray]:
    """Compute mean (centroid) embedding per policy label."""
    centroids: dict[int, np.ndarray] = {}
    for pid in unique_ids:
        mask = policy_id == pid
        if mask.sum() == 0:
            continue
        centroids[pid] = normalize(embeddings[mask].mean(axis=0, keepdims=True), norm="l2")[0]
    return centroids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-aligned policy separation evaluation."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to embeddings_all.npy (N, emb_dim), aligned with policy_id/source_index rows.",
    )
    parser.add_argument(
        "--policy_id_path",
        type=str,
        required=True,
        help="Path to policy_id.npy (int array, length N).",
    )
    parser.add_argument(
        "--source_index_path",
        type=str,
        required=True,
        help="Path to source_index.npy (int array, length N; 0..N_src-1).",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Path to split.npy; if omitted all samples are used.",
    )
    parser.add_argument(
        "--policy_names_path",
        type=str,
        default=None,
        help="Optional path to policy_names.json ({idx: name}); for display only.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="Split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--centroid_split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="train",
        help="Split used to compute policy centroids for within-group assignment (default: train).",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Directory to write outputs; defaults to the directory containing --embeddings_path.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print(f"Loading embeddings    : {args.embeddings_path}")
    embeddings = np.load(args.embeddings_path, allow_pickle=False).astype(np.float32)
    N, emb_dim = embeddings.shape
    print(f"  shape: {embeddings.shape}")

    print(f"Loading policy ids    : {args.policy_id_path}")
    policy_id = np.load(args.policy_id_path, allow_pickle=True).astype(np.int32)
    if len(policy_id) != N:
        raise ValueError(f"policy_id length {len(policy_id)} != embeddings rows {N}")

    print(f"Loading source index  : {args.source_index_path}")
    source_index = np.load(args.source_index_path, allow_pickle=True).astype(np.int32)
    if len(source_index) != N:
        raise ValueError(f"source_index length {len(source_index)} != embeddings rows {N}")

    policy_names: dict[str, str] = {}
    if args.policy_names_path and os.path.isfile(args.policy_names_path):
        with open(args.policy_names_path, encoding="utf-8") as f:
            policy_names = json.load(f)
    unique_ids = sorted(int(x) for x in np.unique(policy_id))
    n_policies = len(unique_ids)
    print(f"  unique policy ids: {unique_ids}  ({n_policies} policies)")
    if policy_names:
        print(f"  policy names: {policy_names}")

    # Split masks
    split: np.ndarray | None = None
    if args.split_path is not None:
        split = np.load(args.split_path, allow_pickle=True)
        split = np.array([str(s) for s in split])
        if len(split) != N:
            raise ValueError(f"split length {len(split)} != embeddings rows {N}")

    def _mask(split_name: str) -> np.ndarray:
        if split is None or split_name == "all":
            return np.ones(N, dtype=bool)
        return split == split_name

    eval_mask = _mask(args.eval_split)
    centroid_mask = _mask(args.centroid_split)

    if eval_mask.sum() == 0:
        raise RuntimeError(
            f"No evaluation samples found (split='{args.eval_split}' has 0 rows)."
        )

    # Sub-select eval set
    eval_idx = np.where(eval_mask)[0]
    emb_eval = embeddings[eval_idx]
    pid_eval = policy_id[eval_idx]
    src_eval = source_index[eval_idx]

    print(f"\nEval samples : {len(eval_idx)}  (split='{args.eval_split}')")

    # ------------------------------------------------------------------
    # Compute policy centroids from centroid_split
    # ------------------------------------------------------------------
    centroid_idx = np.where(centroid_mask)[0]
    centroids = _policy_centroids(embeddings[centroid_idx], policy_id[centroid_idx], unique_ids)
    centroid_matrix = np.stack([centroids[pid] for pid in unique_ids], axis=0)  # (P, D)

    # ------------------------------------------------------------------
    # Group eval samples by source_index
    # ------------------------------------------------------------------
    # source_to_indices maps source_index → list of positions in eval arrays
    from collections import defaultdict
    src_to_pos: dict[int, dict[int, int]] = defaultdict(dict)
    for pos, (sid, pid) in enumerate(zip(src_eval.tolist(), pid_eval.tolist())):
        if pid in src_to_pos[sid]:
            # Duplicate (sid, pid) – should not happen by construction but be safe
            pass
        src_to_pos[sid][pid] = pos

    # Only keep sources that have ALL policies represented
    complete_sources = [
        sid for sid, pid_map in src_to_pos.items()
        if all(pid in pid_map for pid in unique_ids)
    ]
    print(f"Sources in eval with all {n_policies} policies: {len(complete_sources)} "
          f"(out of {len(src_to_pos)} total source groups)")
    if len(complete_sources) == 0:
        raise RuntimeError(
            "No source groups with all policies present in the eval split. "
            "Check that source_index.npy is aligned with policy_id.npy."
        )

    # ------------------------------------------------------------------
    # 1. Pairwise inter-policy distances within same source
    # ------------------------------------------------------------------
    print("\n--- Pairwise inter-policy distances (within source) ---")
    policy_pairs = list(combinations(unique_ids, 2))
    # pair → list of distances (one per complete source)
    pair_dists: dict[tuple[int, int], list[float]] = {pair: [] for pair in policy_pairs}

    rows_pairwise: list[dict] = []

    for sid in complete_sources:
        pid_map = src_to_pos[sid]
        for (pa, pb) in policy_pairs:
            ea = emb_eval[pid_map[pa]]
            eb = emb_eval[pid_map[pb]]
            d = _cosine_dist(ea, eb)
            pair_dists[(pa, pb)].append(d)
            rows_pairwise.append({
                "source_index": sid,
                "policy_a": pa,
                "policy_b": pb,
                "policy_a_name": policy_names.get(str(pa), f"policy_{pa}"),
                "policy_b_name": policy_names.get(str(pb), f"policy_{pb}"),
                "cosine_dist": d,
            })

    pairwise_summary: dict[str, dict] = {}
    for (pa, pb), dists in pair_dists.items():
        arr = np.array(dists, dtype=np.float32)
        key = f"{policy_names.get(str(pa), str(pa))}_vs_{policy_names.get(str(pb), str(pb))}"
        pairwise_summary[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "std": float(arr.std()),
            "n_sources": len(dists),
        }
        print(f"  {key:<40}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"p25={np.percentile(arr, 25):.4f}  p75={np.percentile(arr, 75):.4f}")

    mean_pairwise_dist = float(
        np.mean([v["mean"] for v in pairwise_summary.values()])
    )
    print(f"\n  Mean across all pairs: {mean_pairwise_dist:.4f}")

    # ------------------------------------------------------------------
    # 2. Source-aligned retrieval + within-group policy assignment
    # ------------------------------------------------------------------
    print("\n--- Source-aligned retrieval & within-group policy assignment ---")

    # (a) Nearest-neighbour among same source → always different policy (sanity check)
    nn_diff_policy_frac: list[float] = []
    # (b) Policy centroid assignment accuracy within source groups
    correct_assignments = 0
    total_assignments = 0

    for sid in complete_sources:
        pid_map = src_to_pos[sid]
        positions = [pid_map[pid] for pid in unique_ids]
        group_emb = emb_eval[positions]  # (P, D)
        group_pids = np.array(unique_ids)

        # (a) For each sample in group, find nearest other sample in group
        for i, pid_i in enumerate(unique_ids):
            dists_to_others = [
                _cosine_dist(group_emb[i], group_emb[j])
                for j in range(n_policies) if j != i
            ]
            # nearest neighbour has a different policy by construction → should always be 1
            nn_diff_policy_frac.append(1.0)

        # (b) Within-group centroid assignment
        # For each of the P embeddings in this group, assign to nearest centroid
        emb_norm = normalize(group_emb, norm="l2")  # (P, D)
        # cosine similarity to each centroid
        sim_to_centroids = emb_norm @ centroid_matrix.T  # (P, P)
        assigned = np.argmax(sim_to_centroids, axis=1)   # (P,) index into unique_ids
        for i, pid_true in enumerate(unique_ids):
            predicted_pid = unique_ids[assigned[i]]
            if predicted_pid == pid_true:
                correct_assignments += 1
            total_assignments += 1

    nn_diff_frac = float(np.mean(nn_diff_policy_frac))
    within_group_acc = correct_assignments / total_assignments if total_assignments > 0 else 0.0

    print(f"  Nearest-neighbour always different policy: {nn_diff_frac:.4f}  (expected 1.0)")
    print(f"  Within-group centroid assignment accuracy: {within_group_acc:.4f}  "
          f"(chance={1.0/n_policies:.4f})  [{correct_assignments}/{total_assignments}]")

    # Per-policy centroid assignment accuracy
    per_policy_correct: dict[int, int] = {pid: 0 for pid in unique_ids}
    per_policy_total: dict[int, int] = {pid: 0 for pid in unique_ids}
    for sid in complete_sources:
        pid_map = src_to_pos[sid]
        positions = [pid_map[pid] for pid in unique_ids]
        group_emb = emb_eval[positions]
        emb_norm = normalize(group_emb, norm="l2")
        sim_to_centroids = emb_norm @ centroid_matrix.T
        assigned = np.argmax(sim_to_centroids, axis=1)
        for i, pid_true in enumerate(unique_ids):
            per_policy_total[pid_true] += 1
            if unique_ids[assigned[i]] == pid_true:
                per_policy_correct[pid_true] += 1

    per_policy_within_group_acc: dict[str, float] = {}
    print("  Per-policy within-group centroid accuracy:")
    for pid in unique_ids:
        name = policy_names.get(str(pid), f"policy_{pid}")
        acc = per_policy_correct[pid] / per_policy_total[pid] if per_policy_total[pid] > 0 else 0.0
        per_policy_within_group_acc[name] = acc
        print(f"    {name:<20}: {acc:.4f}  [{per_policy_correct[pid]}/{per_policy_total[pid]}]")

    # ------------------------------------------------------------------
    # 3. Within-source clustering score
    # ------------------------------------------------------------------
    print("\n--- Within-source clustering score ---")

    # Mean within-source inter-policy distance (already computed above as mean_pairwise_dist)
    mean_within_source_inter = mean_pairwise_dist

    # Mean cross-source same-policy distance
    # For each policy, sample pairs of source groups and compute distance
    cross_src_dists: list[float] = []
    if len(complete_sources) >= 2:
        src_list = sorted(complete_sources)
        for pid in unique_ids:
            # collect embeddings for this policy across sources
            embs_for_policy = np.stack(
                [emb_eval[src_to_pos[sid][pid]] for sid in src_list], axis=0
            )  # (n_complete, D)
            n_src = len(src_list)
            # subsample up to 500 pairs to keep it fast
            rng = np.random.default_rng(42)
            n_pairs = min(500, n_src * (n_src - 1) // 2)
            idxs_a = rng.integers(0, n_src, size=n_pairs * 3)
            idxs_b = rng.integers(0, n_src, size=n_pairs * 3)
            seen = 0
            for ia, ib in zip(idxs_a.tolist(), idxs_b.tolist()):
                if ia == ib:
                    continue
                cross_src_dists.append(_cosine_dist(embs_for_policy[ia], embs_for_policy[ib]))
                seen += 1
                if seen >= n_pairs:
                    break

    mean_cross_src_same = float(np.mean(cross_src_dists)) if cross_src_dists else float("nan")
    clustering_score = (
        mean_within_source_inter / mean_cross_src_same
        if mean_cross_src_same > 0
        else float("nan")
    )

    print(f"  Mean within-source inter-policy dist  : {mean_within_source_inter:.4f}")
    print(f"  Mean cross-source same-policy dist    : {mean_cross_src_same:.4f}")
    print(f"  Within-source clustering score (ratio): {clustering_score:.4f}  (>1 = better)")

    # ------------------------------------------------------------------
    # Determine output directory and save
    # ------------------------------------------------------------------
    if args.analysis_dir is None:
        analysis_dir = str(Path(args.embeddings_path).parent)
    else:
        analysis_dir = args.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)

    summary = {
        "eval_split": args.eval_split,
        "centroid_split": args.centroid_split,
        "n_samples_eval": int(eval_mask.sum()),
        "n_complete_sources": len(complete_sources),
        "n_policies": n_policies,
        "policy_names": policy_names,
        "pairwise_distances": pairwise_summary,
        "mean_pairwise_dist_all_pairs": mean_pairwise_dist,
        "source_aligned_retrieval": {
            "nn_diff_policy_frac": nn_diff_frac,
            "within_group_centroid_accuracy": within_group_acc,
            "chance_accuracy": 1.0 / n_policies,
            "per_policy_within_group_accuracy": per_policy_within_group_acc,
        },
        "within_source_clustering": {
            "mean_within_source_inter_policy_dist": mean_within_source_inter,
            "mean_cross_source_same_policy_dist": mean_cross_src_same,
            "clustering_score_ratio": clustering_score,
        },
    }

    summary_path = os.path.join(analysis_dir, "policy_separation_aligned_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    df_pairwise = pd.DataFrame(rows_pairwise)
    pairwise_csv_path = os.path.join(analysis_dir, "policy_pairwise_dist.csv")
    df_pairwise.to_csv(pairwise_csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {analysis_dir}")
    print(f"  {os.path.basename(summary_path)}")
    print(f"  {os.path.basename(pairwise_csv_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
