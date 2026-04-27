"""Source-aligned policy separation evaluation.

Controls for scenario distribution by grouping samples by ``source_index``
(each source window was rolled out under every policy exactly once).  Within
each source group the distances / centroids are computed so that any observed
difference reflects policy style rather than scenario difficulty.

Computations (applied to --eval_split samples only, centroids from train):
  (a) Validate that each source_index contains every policy exactly once;
      report counts of missing / duplicate (policy, source) pairs.
  (b) Pairwise embedding distances within each source group for every policy
      pair — both Euclidean and cosine distance;
      saved to policy_pairwise_dist.csv.
  (c) Within-source classification accuracy: for each source group in the
      eval split, assign the policy label of the nearest train-split centroid
      and aggregate accuracy.
  (d) Within-source retrieval (centroid-agreement): for each eval sample rank the
      other policies in the *same* source by distance; report whether the nearest
      neighbour is the policy whose train-split centroid is globally nearest to the
      query policy's centroid.  Chance = 1/(n_policies−1).  The old "same-policy
      nearest neighbour" definition is inapplicable here because each source has
      exactly one sample per policy.
  (e) Save policy_separation_aligned_summary.json.

Outputs written to --analysis_dir (default: directory containing embeddings):
    policy_separation_aligned_summary.json
    policy_pairwise_dist.csv

Usage example:
    python evaluate_policy_separation_aligned.py \\
        --embeddings_path  output_policy_rollouts/run_relkin_knn/embeddings_all.npy \\
        --policy_id_path   output_policy_rollouts/policy_id.npy \\
        --source_index_path output_policy_rollouts/source_index.npy \\
        --split_path       output_policy_rollouts/split.npy \\
        --eval_split test \\
        --analysis_dir     output_policy_rollouts/run_relkin_knn/analysis_aligned \\
        --seed 42
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1-D vectors."""
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 – cosine similarity) between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _centroid(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of embeddings selected by boolean mask."""
    return embeddings[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def validate_source_policy_coverage(
    source_index: np.ndarray,
    policy_id: np.ndarray,
    unique_policies: List[int],
) -> Dict:
    """Check that every (source, policy) pair appears exactly once.

    Returns a dict with keys:
        n_sources, n_expected, n_found,
        n_complete_sources, n_missing_pairs, n_duplicate_pairs,
        missing_pairs (list of (src, pid)),
        duplicate_pairs (list of (src, pid, count))
    """
    unique_sources = sorted(set(source_index.tolist()))
    n_sources = len(unique_sources)
    expected_per_src = len(unique_policies)
    n_expected = n_sources * expected_per_src

    # Count (source, policy) occurrences
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for src, pid in zip(source_index.tolist(), policy_id.tolist()):
        pair_counts[(int(src), int(pid))] += 1

    missing_pairs = []
    duplicate_pairs = []
    for src in unique_sources:
        for pid in unique_policies:
            cnt = pair_counts.get((src, pid), 0)
            if cnt == 0:
                missing_pairs.append((src, pid))
            elif cnt > 1:
                duplicate_pairs.append((src, pid, cnt))

    n_complete = sum(
        1 for src in unique_sources
        if all(pair_counts.get((src, pid), 0) == 1 for pid in unique_policies)
    )

    return {
        "n_sources": n_sources,
        "n_expected_pairs": n_expected,
        "n_found_pairs": int(len(source_index)),
        "n_complete_sources": n_complete,
        "n_missing_pairs": len(missing_pairs),
        "n_duplicate_pairs": len(duplicate_pairs),
        "missing_pairs": missing_pairs[:50],   # cap for JSON size
        "duplicate_pairs": duplicate_pairs[:50],
    }


def compute_pairwise_distances(
    embeddings: np.ndarray,
    source_index: np.ndarray,
    policy_id: np.ndarray,
    unique_policies: List[int],
) -> pd.DataFrame:
    """Compute per-source pairwise (Euclidean + cosine) distances for each policy pair.

    Only sources that have exactly one sample per policy are included.
    Returns a DataFrame with columns:
        source_index, policy_a, policy_b, euclidean_dist, cosine_dist
    """
    rows = []
    unique_sources = sorted(set(source_index.tolist()))

    # Build lookup: (src, pid) -> embedding index
    src_pid_to_idx: Dict[Tuple[int, int], int] = {}
    for i, (src, pid) in enumerate(zip(source_index.tolist(), policy_id.tolist())):
        key = (int(src), int(pid))
        if key not in src_pid_to_idx:
            src_pid_to_idx[key] = i
        # If duplicate, skip (keep first)

    for src in unique_sources:
        # Check all policies present exactly once
        indices = {}
        ok = True
        for pid in unique_policies:
            key = (src, pid)
            if key not in src_pid_to_idx:
                ok = False
                break
            indices[pid] = src_pid_to_idx[key]
        if not ok:
            continue

        for ai, pa in enumerate(unique_policies):
            for pb in unique_policies[ai + 1:]:
                ea = embeddings[indices[pa]]
                eb = embeddings[indices[pb]]
                rows.append({
                    "source_index": src,
                    "policy_a": pa,
                    "policy_b": pb,
                    "euclidean_dist": _l2(ea, eb),
                    "cosine_dist": _cosine_dist(ea, eb),
                })

    return pd.DataFrame(rows)


def compute_centroid_accuracy(
    embeddings: np.ndarray,
    source_index: np.ndarray,
    policy_id: np.ndarray,
    unique_policies: List[int],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> Tuple[float, Dict[int, float]]:
    """Nearest-centroid classification accuracy within source groups.

    Centroids are computed from train-split samples (one per policy).
    For each eval sample the policy is predicted as the nearest centroid
    (Euclidean distance).  Only eval sources that have at least one sample
    per policy in the train split are used.

    Returns:
        overall_accuracy: float
        per_policy_accuracy: {pid: float}
    """
    # Build train centroids
    centroids: Dict[int, np.ndarray] = {}
    for pid in unique_policies:
        mask = train_mask & (policy_id == pid)
        if mask.sum() > 0:
            centroids[pid] = embeddings[mask].mean(axis=0)

    if len(centroids) < len(unique_policies):
        missing = [p for p in unique_policies if p not in centroids]
        print(f"  [warn] Centroids missing for policies: {missing} (no train samples)")

    if not centroids:
        return 0.0, {}

    centroid_matrix = np.stack([centroids[pid] for pid in sorted(centroids)], axis=0)
    centroid_pids = sorted(centroids.keys())

    # Predict for eval samples
    eval_indices = np.where(eval_mask)[0]
    correct = 0
    total = 0
    per_policy_correct: Dict[int, int] = defaultdict(int)
    per_policy_total: Dict[int, int] = defaultdict(int)

    for i in eval_indices:
        emb = embeddings[i]
        true_pid = int(policy_id[i])
        dists = np.linalg.norm(centroid_matrix - emb, axis=1)
        pred_pid = centroid_pids[int(np.argmin(dists))]
        per_policy_total[true_pid] += 1
        if pred_pid == true_pid:
            correct += 1
            per_policy_correct[true_pid] += 1
        total += 1

    overall_acc = correct / total if total > 0 else 0.0
    per_policy_acc = {
        pid: per_policy_correct[pid] / per_policy_total[pid]
        for pid in unique_policies
        if per_policy_total[pid] > 0
    }
    return overall_acc, per_policy_acc


def compute_within_source_retrieval(
    embeddings: np.ndarray,
    source_index: np.ndarray,
    policy_id: np.ndarray,
    unique_policies: List[int],
    eval_mask: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Tuple[float, float, float]:
    """Within-source centroid-agreement retrieval metrics for eval samples.

    In the within-source setting each source has exactly **one** sample per
    policy, so a "same-policy nearest neighbour" metric is undefined (there are
    no other samples with the same policy label in the same source).

    Instead, this function measures **centroid-agreement**: for each eval sample
    with policy p_i, the *expected* nearest within-source neighbour is the
    policy whose **train-split centroid** is closest to the centroid of p_i.
    A "hit" occurs when the actual nearest within-source neighbour matches this
    centroid-derived expectation.

    Formally:
      1. Compute centroid-nearest policy t(p_i) = argmin_{p_j ≠ p_i} ‖c_{p_i} − c_{p_j}‖
         for each policy p_i (using train centroids).
      2. For each eval sample i (policy p_i) in source s:
           others = all eval samples in source s with policy ≠ p_i
           nearest = argmin_{j ∈ others} ‖emb_i − emb_j‖
           hit = 1 if policy_id[nearest] == t(p_i) else 0
      3. Chance level = 1 / (n_policies − 1)  (random pick among other policies).

    Margin (unchanged): distance(farthest within-source) − distance(nearest
    within-source).  This is always well-defined when there are ≥ 2 policies per
    source.

    Returns:
        hit_rate:      fraction of eval samples where nearest within-source
                       neighbour agrees with the centroid-order expectation.
        mean_margin:   mean of per-sample distance margins.
        median_margin: median of per-sample distance margins.
    """
    # Pre-compute centroid-nearest policy for each policy using train centroids.
    centroid_target: Dict[int, int] = {}
    for p_i in unique_policies:
        if p_i not in centroids:
            continue
        best_p = -1
        best_d = float("inf")
        for p_j in unique_policies:
            if p_j == p_i or p_j not in centroids:
                continue
            d = float(np.linalg.norm(centroids[p_i] - centroids[p_j]))
            if d < best_d:
                best_d = d
                best_p = p_j
        if best_p >= 0:
            centroid_target[p_i] = best_p

    # Group eval indices by source
    eval_indices = np.where(eval_mask)[0]
    src_to_eval_indices: Dict[int, List[int]] = defaultdict(list)
    for i in eval_indices:
        src_to_eval_indices[int(source_index[i])].append(i)

    hits: List[int] = []
    margins: List[float] = []

    for src, idxs in src_to_eval_indices.items():
        if len(idxs) < 2:
            continue  # need at least 2 samples to do within-source retrieval
        for i in idxs:
            true_pid = int(policy_id[i])
            target_pid = centroid_target.get(true_pid, -1)
            if target_pid < 0:
                continue  # no centroid available for this policy – skip

            others = [j for j in idxs if j != i]
            emb_i = embeddings[i]
            dists = np.array([np.linalg.norm(embeddings[j] - emb_i) for j in others])
            nearest_j = others[int(np.argmin(dists))]
            nearest_pid = int(policy_id[nearest_j])

            # Hit: nearest within-source neighbour is the centroid-nearest policy
            hits.append(int(nearest_pid == target_pid))
            margins.append(float(np.max(dists) - np.min(dists)))

    hit_rate = float(np.mean(hits)) if hits else float("nan")
    mean_margin = float(np.mean(margins)) if margins else float("nan")
    median_margin = float(np.median(margins)) if margins else float("nan")
    return hit_rate, mean_margin, median_margin


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-aligned policy separation evaluation: controls for scenario "
            "distribution by grouping samples by source_index."
        )
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to embeddings_all.npy (N, emb_dim).",
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
        help="Path to source_index.npy (int array, length N) mapping each "
             "sample back to its original source window index.",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Path to split.npy (train/val/test strings). "
             "If omitted, all samples are used for both train and eval.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="Split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Directory to write outputs; defaults to the directory containing "
             "--embeddings_path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (currently unused but reserved for future use).",
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
    print(f"Loading embeddings      : {args.embeddings_path}")
    embeddings = np.load(args.embeddings_path, allow_pickle=False).astype(np.float32)
    N, emb_dim = embeddings.shape
    print(f"  shape: {embeddings.shape}")

    print(f"Loading policy ids      : {args.policy_id_path}")
    policy_id = np.load(args.policy_id_path, allow_pickle=True).astype(np.int32)
    if len(policy_id) != N:
        raise ValueError(f"policy_id length {len(policy_id)} != embeddings rows {N}")

    print(f"Loading source indices  : {args.source_index_path}")
    source_index = np.load(args.source_index_path, allow_pickle=True).astype(np.int32)
    if len(source_index) != N:
        raise ValueError(
            f"source_index length {len(source_index)} != embeddings rows {N}"
        )

    unique_ids = sorted(int(x) for x in np.unique(policy_id))
    n_policies = len(unique_ids)
    print(f"  unique policy ids: {unique_ids}  ({n_policies} policies)")

    # Split masks
    split: Optional[np.ndarray] = None
    if args.split_path is not None:
        split = np.load(args.split_path, allow_pickle=True)
        split = np.array([str(s) for s in split])
        if len(split) != N:
            raise ValueError(f"split length {len(split)} != embeddings rows {N}")

    def _mask(split_name: str) -> np.ndarray:
        if split is None or split_name == "all":
            return np.ones(N, dtype=bool)
        return split == split_name

    train_mask = _mask("train")
    eval_mask = _mask(args.eval_split)

    if train_mask.sum() == 0:
        raise RuntimeError("No training samples found (split='train' has 0 rows).")
    if eval_mask.sum() == 0:
        raise RuntimeError(
            f"No evaluation samples found (split='{args.eval_split}' has 0 rows)."
        )

    print(f"\nTrain samples : {int(train_mask.sum())}")
    print(f"Eval  samples : {int(eval_mask.sum())}  (split='{args.eval_split}')")

    # ------------------------------------------------------------------
    # (a) Validate source / policy coverage (over eval split)
    # ------------------------------------------------------------------
    print("\n--- (a) Source-policy coverage validation (eval split) ---")
    coverage = validate_source_policy_coverage(
        source_index[eval_mask],
        policy_id[eval_mask],
        unique_ids,
    )
    print(f"  Sources in eval         : {coverage['n_sources']}")
    print(f"  Expected (src×pol) pairs: {coverage['n_expected_pairs']}")
    print(f"  Found pairs             : {coverage['n_found_pairs']}")
    print(f"  Complete sources        : {coverage['n_complete_sources']}")
    print(f"  Missing (src,pol) pairs : {coverage['n_missing_pairs']}")
    print(f"  Duplicate (src,pol) pairs: {coverage['n_duplicate_pairs']}")
    if coverage["missing_pairs"]:
        print(f"  First missing pairs     : {coverage['missing_pairs'][:5]}")
    if coverage["duplicate_pairs"]:
        print(f"  First duplicate pairs   : {coverage['duplicate_pairs'][:5]}")

    # ------------------------------------------------------------------
    # (b) Pairwise distances within each source (eval split)
    # ------------------------------------------------------------------
    print("\n--- (b) Within-source pairwise embedding distances (eval split) ---")
    dist_df = compute_pairwise_distances(
        embeddings[eval_mask],
        source_index[eval_mask],
        policy_id[eval_mask],
        unique_ids,
    )

    pairwise_stats: Dict = {}
    if dist_df.empty:
        print("  [warn] No complete source groups found; pairwise distances unavailable.")
    else:
        pair_labels = dist_df.apply(
            lambda r: f"p{int(r.policy_a)}_vs_p{int(r.policy_b)}", axis=1
        )
        for pair in pair_labels.unique():
            sub = dist_df[pair_labels == pair]
            print(f"  {pair}  n={len(sub)}")
            for col in ("euclidean_dist", "cosine_dist"):
                m, med = float(sub[col].mean()), float(sub[col].median())
                print(f"    {col:<20}: mean={m:.4f}  median={med:.4f}")
            pairwise_stats[pair] = {
                "n": int(len(sub)),
                "euclidean_mean": float(sub["euclidean_dist"].mean()),
                "euclidean_median": float(sub["euclidean_dist"].median()),
                "cosine_mean": float(sub["cosine_dist"].mean()),
                "cosine_median": float(sub["cosine_dist"].median()),
            }

    # ------------------------------------------------------------------
    # (c) Within-source centroid classification accuracy
    # ------------------------------------------------------------------
    print("\n--- (c) Within-source centroid classification (train centroids → eval) ---")
    centroid_acc, per_policy_acc = compute_centroid_accuracy(
        embeddings,
        source_index,
        policy_id,
        unique_ids,
        train_mask,
        eval_mask,
    )
    chance = 1.0 / n_policies
    print(f"  Centroid accuracy : {centroid_acc:.4f}  (chance={chance:.4f})")
    for pid, acc in per_policy_acc.items():
        print(f"    policy_{pid}: {acc:.4f}")

    # Pre-build centroids dict for retrieval step
    centroids: Dict[int, np.ndarray] = {}
    for pid in unique_ids:
        mask = train_mask & (policy_id == pid)
        if mask.sum() > 0:
            centroids[pid] = embeddings[mask].mean(axis=0)

    # ------------------------------------------------------------------
    # (d) Within-source retrieval metric (eval split)
    # ------------------------------------------------------------------
    print("\n--- (d) Within-source retrieval (eval split) ---")
    hit_rate, mean_margin, median_margin = compute_within_source_retrieval(
        embeddings,
        source_index,
        policy_id,
        unique_ids,
        eval_mask,
        centroids,
    )
    retrieval_chance = 1.0 / (n_policies - 1) if n_policies > 1 else float("nan")
    print(f"  Centroid-agreement hit rate: {hit_rate:.4f}  (chance={retrieval_chance:.4f})")
    print(f"  [metric] within-source; hit = nearest neighbour matches centroid-nearest policy")
    print(f"  Mean within-source margin  : {mean_margin:.4f}")
    print(f"  Median within-source margin: {median_margin:.4f}")

    # ------------------------------------------------------------------
    # Determine analysis_dir and save outputs
    # ------------------------------------------------------------------
    if args.analysis_dir is None:
        analysis_dir = str(Path(args.embeddings_path).parent)
    else:
        analysis_dir = args.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)

    # (b) Save pairwise distance CSV
    pairwise_csv_path = os.path.join(analysis_dir, "policy_pairwise_dist.csv")
    if not dist_df.empty:
        dist_df.to_csv(pairwise_csv_path, index=False)
        print(f"\nSaved pairwise distances → {pairwise_csv_path}")

    # (e) Save summary JSON
    summary = {
        "eval_split": args.eval_split,
        "n_samples_train": int(train_mask.sum()),
        "n_samples_eval": int(eval_mask.sum()),
        "n_policies": n_policies,
        "unique_policy_ids": unique_ids,
        # (a) Coverage
        "coverage": {
            k: v for k, v in coverage.items()
            if k not in ("missing_pairs", "duplicate_pairs")
        },
        # (b) Pairwise distances
        "pairwise_distances": pairwise_stats,
        # (c) Centroid classification
        "centroid_classification": {
            "accuracy": centroid_acc,
            "chance_accuracy": chance,
            "per_policy_accuracy": {str(k): v for k, v in per_policy_acc.items()},
        },
        # (d) Within-source retrieval
        "within_source_retrieval": {
            "metric_mode": "within_source_centroid_agreement",
            "metric_definition": (
                "For each eval sample with policy p, among the other policies in the same "
                "source, hit = the nearest within-source neighbour is the policy whose "
                "train-split centroid is globally nearest to centroid[p]. "
                "Chance = 1 / (n_policies - 1)."
            ),
            "centroid_agreement_hit_rate": hit_rate,
            "chance_hit_rate": retrieval_chance,
            "mean_within_source_margin": mean_margin,
            "median_within_source_margin": median_margin,
        },
    }

    summary_path = os.path.join(analysis_dir, "policy_separation_aligned_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {analysis_dir}")
    print(f"  {os.path.basename(summary_path)}")
    if not dist_df.empty:
        print(f"  {os.path.basename(pairwise_csv_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
