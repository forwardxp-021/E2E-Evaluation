"""Evaluate whether learned style embeddings separate synthetic policy labels.

Computes:
  1. Linear policy classification (LogisticRegression) trained on the train split
     and evaluated on --eval_split: accuracy + macro-F1.
  2. Retrieval metric: Recall@K of same-policy neighbours in embedding space
     (cosine similarity) on --eval_split.

Outputs written to --analysis_dir (default: directory of --embeddings_path):
    policy_separation_summary.json  – classification and retrieval metrics
    policy_retrieval.csv            – per-sample recall values

Usage example:
    python evaluate_policy_separation.py \\
        --embeddings_path output_policy_rollouts/run_relkin/embeddings_all.npy \\
        --policy_id_path  output_policy_rollouts/policy_id.npy \\
        --split_path      output_policy_rollouts/split.npy \\
        --eval_split test \\
        --k_neighbors 10
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine-similarity matrix for row-normalised X."""
    Xn = normalize(X, norm="l2", axis=1)
    return Xn @ Xn.T


def recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[float, np.ndarray]:
    """Compute Recall@K using cosine similarity.

    For each query, Recall@K = (same-label neighbours in top-K) / (true positives - 1).
    Excludes self from neighbours (top K+1 then drop self).

    Returns:
        mean_recall: float, mean over all queries
        per_sample:  np.ndarray of per-query recall values
    """
    N = len(labels)
    sim = _cosine_sim_matrix(embeddings)
    per_sample = np.empty(N, dtype=np.float32)

    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        top_k_idx = np.argpartition(row, -k)[-k:]  # top-K indices (unordered)
        same_policy = int(np.sum(labels[top_k_idx] == labels[i]))
        n_positives = int(np.sum(labels == labels[i])) - 1  # exclude self
        if n_positives == 0:
            per_sample[i] = 1.0
        else:
            per_sample[i] = same_policy / min(k, n_positives)

    return float(np.mean(per_sample)), per_sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate policy separation of style embeddings."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to embeddings_all.npy (N, emb_dim), aligned with policy_id/split rows.",
    )
    parser.add_argument(
        "--policy_id_path",
        type=str,
        required=True,
        help="Path to policy_id.npy (int array, length N).",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Path to split.npy; if omitted all samples are used for both train and eval.",
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
        "--k_neighbors",
        type=int,
        default=10,
        help="K for Recall@K retrieval metric (default: 10).",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Directory to write outputs; defaults to the directory containing --embeddings_path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LogisticRegression.",
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
    print(f"Loading embeddings: {args.embeddings_path}")
    embeddings = np.load(args.embeddings_path, allow_pickle=False).astype(np.float32)
    N, emb_dim = embeddings.shape
    print(f"  shape: {embeddings.shape}")

    print(f"Loading policy ids: {args.policy_id_path}")
    policy_id = np.load(args.policy_id_path, allow_pickle=True).astype(np.int32)
    if len(policy_id) != N:
        raise ValueError(
            f"policy_id length {len(policy_id)} != embeddings rows {N}"
        )

    policy_names: dict[str, str] = {}
    if args.policy_names_path and os.path.isfile(args.policy_names_path):
        with open(args.policy_names_path, encoding="utf-8") as f:
            policy_names = json.load(f)
    unique_ids = sorted(int(x) for x in np.unique(policy_id))
    n_policies = len(unique_ids)
    print(f"  unique policy ids: {unique_ids}  ({n_policies} policies)")
    if policy_names:
        print(f"  policy names: {policy_names}")

    # Split mask
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

    train_mask = _mask("train")
    eval_mask = _mask(args.eval_split)

    if train_mask.sum() == 0:
        raise RuntimeError("No training samples found (split='train' has 0 rows).")
    if eval_mask.sum() == 0:
        raise RuntimeError(
            f"No evaluation samples found (split='{args.eval_split}' has 0 rows)."
        )

    X_train = embeddings[train_mask]
    y_train = policy_id[train_mask]
    X_eval = embeddings[eval_mask]
    y_eval = policy_id[eval_mask]

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Eval  samples : {len(X_eval)}  (split='{args.eval_split}')")

    # ------------------------------------------------------------------
    # 1. Linear classification
    # ------------------------------------------------------------------
    print("\n--- Linear classification (LogisticRegression) ---")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)

    acc = float(accuracy_score(y_eval, y_pred))
    f1 = float(f1_score(y_eval, y_pred, average="macro", zero_division=0))
    chance = 1.0 / n_policies

    print(f"  Accuracy    : {acc:.4f}  (chance={chance:.4f})")
    print(f"  Macro-F1    : {f1:.4f}")

    # ------------------------------------------------------------------
    # 2. Recall@K retrieval
    # ------------------------------------------------------------------
    k = args.k_neighbors
    print(f"\n--- Retrieval Recall@{k} (cosine) ---")
    mean_recall, per_sample_recall = recall_at_k(X_eval, y_eval, k=k)
    print(f"  Recall@{k:<3} : {mean_recall:.4f}")

    # Per-policy breakdown
    policy_recall: dict[str, float] = {}
    for pid in unique_ids:
        mask_pid = y_eval == pid
        if mask_pid.sum() == 0:
            continue
        r = float(np.mean(per_sample_recall[mask_pid]))
        name = policy_names.get(str(pid), f"policy_{pid}")
        policy_recall[name] = r
        print(f"    {name:<20}: {r:.4f}")

    # ------------------------------------------------------------------
    # Determine analysis_dir and save outputs
    # ------------------------------------------------------------------
    if args.analysis_dir is None:
        analysis_dir = str(Path(args.embeddings_path).parent)
    else:
        analysis_dir = args.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)

    summary = {
        "n_samples_train": int(train_mask.sum()),
        "n_samples_eval": int(eval_mask.sum()),
        "eval_split": args.eval_split,
        "n_policies": n_policies,
        "policy_names": policy_names,
        "classification": {
            "accuracy": acc,
            "macro_f1": f1,
            "chance_accuracy": chance,
        },
        "retrieval": {
            "k": k,
            "mean_recall_at_k": mean_recall,
            "per_policy_recall_at_k": policy_recall,
        },
    }

    summary_path = os.path.join(analysis_dir, "policy_separation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Per-sample retrieval CSV
    eval_indices = np.where(eval_mask)[0]
    df = pd.DataFrame(
        {
            "sample_index": eval_indices,
            "policy_id": y_eval,
            "policy_name": [
                policy_names.get(str(int(pid)), f"policy_{pid}") for pid in y_eval
            ],
            f"recall_at_{k}": per_sample_recall,
        }
    )
    retrieval_csv_path = os.path.join(analysis_dir, "policy_retrieval.csv")
    df.to_csv(retrieval_csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {analysis_dir}")
    print(f"  {os.path.basename(summary_path)}")
    print(f"  {os.path.basename(retrieval_csv_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
