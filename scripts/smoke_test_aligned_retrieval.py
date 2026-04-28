"""Regression tests for aligned evaluator coverage / retrieval applicability.

Run with:
    python scripts/smoke_test_aligned_retrieval.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_policy_separation_aligned import (
    compute_within_source_margin,
    evaluate_within_source_retrieval_applicability,
    validate_source_policy_coverage,
)


def _make_structured_data(
    n_sources: int = 30,
    n_policies: int = 3,
    emb_dim: int = 8,
    seed: int = 0,
):
    """Create one-sample-per-policy aligned data.

    Returns:
        embeddings   (N, emb_dim)
        source_index (N,)  int
        policy_id    (N,)  int
        eval_mask    (N,)  bool   – all True
    """
    rng = np.random.default_rng(seed)
    N = n_sources * n_policies
    embeddings = np.zeros((N, emb_dim), dtype=np.float32)
    source_index = np.zeros(N, dtype=np.int32)
    policy_id = np.zeros(N, dtype=np.int32)

    # Each policy has a well-separated cluster centre
    cluster_centres = np.zeros((n_policies, emb_dim), dtype=np.float32)
    for p in range(n_policies):
        cluster_centres[p, p % emb_dim] = 10.0 * (p + 1)

    idx = 0
    for src in range(n_sources):
        for pid in range(n_policies):
            # Small noise around cluster centre
            embeddings[idx] = cluster_centres[pid] + rng.standard_normal(emb_dim).astype(np.float32) * 0.01
            source_index[idx] = src
            policy_id[idx] = pid
            idx += 1

    eval_mask = np.ones(N, dtype=bool)

    return embeddings, source_index, policy_id, eval_mask


def test_retrieval_marked_not_applicable_for_one_sample_per_policy():
    """Within-source same-policy NN must be N/A when no same-policy positives exist."""
    embeddings, source_index, policy_id, eval_mask = _make_structured_data()
    unique_policies = sorted(int(p) for p in np.unique(policy_id))
    coverage = validate_source_policy_coverage(source_index, policy_id, unique_policies)
    assert coverage["n_missing_pairs"] == 0
    assert coverage["n_duplicate_pairs"] == 0

    applicable, reason = evaluate_within_source_retrieval_applicability(
        source_index, policy_id, eval_mask
    )
    assert applicable is False, "Retrieval should be N/A for one-sample-per-policy groups."
    assert "not well-defined" in reason
    mean_margin, median_margin = compute_within_source_margin(
        embeddings, source_index, eval_mask
    )
    assert mean_margin > 0.0
    assert median_margin > 0.0
    print("  [PASS] retrieval marked N/A and margins remain defined")


def test_coverage_detects_missing_and_duplicate_pairs():
    """Coverage diagnostics should detect missing and duplicate source-policy pairs."""
    rng = np.random.default_rng(99)
    n_sources, n_policies, emb_dim = 10, 3, 4
    N = n_sources * n_policies
    embeddings = rng.standard_normal((N, emb_dim)).astype(np.float32)
    source_index = np.repeat(np.arange(n_sources), n_policies).astype(np.int32)
    policy_id = np.tile(np.arange(n_policies), n_sources).astype(np.int32)

    # Create one missing pair: remove (src=0, pid=2)
    keep = ~((source_index == 0) & (policy_id == 2))
    source_index = source_index[keep]
    policy_id = policy_id[keep]
    embeddings = embeddings[keep]
    # Create one duplicate pair: duplicate first row (src=0,pid=0)
    source_index = np.concatenate([source_index, source_index[:1]])
    policy_id = np.concatenate([policy_id, policy_id[:1]])
    embeddings = np.concatenate([embeddings, embeddings[:1]], axis=0)
    unique_policies = sorted(int(p) for p in np.unique(policy_id))
    coverage = validate_source_policy_coverage(source_index, policy_id, unique_policies)

    assert coverage["n_missing_pairs"] >= 1, "Expected at least one missing pair."
    assert coverage["n_duplicate_pairs"] >= 1, "Expected at least one duplicate pair."
    print("  [PASS] missing/duplicate coverage checks work")


def main():
    print("Running aligned evaluator regression tests ...")

    print("\n[1] test_retrieval_marked_not_applicable_for_one_sample_per_policy")
    test_retrieval_marked_not_applicable_for_one_sample_per_policy()

    print("\n[2] test_coverage_detects_missing_and_duplicate_pairs")
    test_coverage_detects_missing_and_duplicate_pairs()

    print("\nAll regression tests passed.")


if __name__ == "__main__":
    main()
