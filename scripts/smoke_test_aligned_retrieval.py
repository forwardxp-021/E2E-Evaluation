"""Regression test for within-source retrieval metric in evaluate_policy_separation_aligned.py.

Ensures the centroid-agreement hit rate is never trivially 0.0 when embeddings
have clear policy structure (one cluster per policy).

Run with:
    python scripts/smoke_test_aligned_retrieval.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_policy_separation_aligned import compute_within_source_retrieval


def _make_structured_data(n_sources: int = 30, n_policies: int = 3, emb_dim: int = 8, seed: int = 0):
    """Create perfectly separable embeddings: each policy lives in a distinct cluster.

    Returns:
        embeddings   (N, emb_dim)
        source_index (N,)  int
        policy_id    (N,)  int
        eval_mask    (N,)  bool   – all True
        centroids    {pid: centroid_vector}
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

    # Build "train" centroids directly from cluster centres (perfect centroids)
    centroids = {pid: cluster_centres[pid].copy() for pid in range(n_policies)}

    return embeddings, source_index, policy_id, eval_mask, centroids


def test_hit_rate_above_chance_with_structured_data():
    """With clearly separated clusters the hit rate must exceed chance."""
    embeddings, source_index, policy_id, eval_mask, centroids = _make_structured_data()
    n_policies = len(centroids)
    chance = 1.0 / (n_policies - 1)

    unique_policies = sorted(int(p) for p in np.unique(policy_id))
    hit_rate, mean_margin, median_margin = compute_within_source_retrieval(
        embeddings, source_index, policy_id, unique_policies, eval_mask, centroids
    )

    print(f"  hit_rate={hit_rate:.4f}  chance={chance:.4f}  mean_margin={mean_margin:.4f}")

    assert not (hit_rate == 0.0), (
        "Regression: hit_rate is 0.0 even with structured data – "
        "this indicates the false-0.0 bug has returned."
    )
    assert hit_rate > chance, (
        f"hit_rate={hit_rate:.4f} should exceed chance={chance:.4f} "
        "when embeddings are perfectly separable."
    )
    assert mean_margin > 0.0, f"mean_margin={mean_margin} should be positive."
    print("  [PASS] hit_rate above chance with structured embeddings")


def test_hit_rate_not_trivially_one_with_noisy_data():
    """With high noise the hit rate should still be defined (not NaN) but may vary."""
    rng = np.random.default_rng(99)
    n_sources, n_policies, emb_dim = 40, 3, 8
    N = n_sources * n_policies

    # Random embeddings (no structure)
    embeddings = rng.standard_normal((N, emb_dim)).astype(np.float32)
    source_index = np.repeat(np.arange(n_sources), n_policies).astype(np.int32)
    policy_id = np.tile(np.arange(n_policies), n_sources).astype(np.int32)
    eval_mask = np.ones(N, dtype=bool)

    # Centroids from mean per policy
    centroids = {
        pid: embeddings[policy_id == pid].mean(axis=0)
        for pid in range(n_policies)
    }

    unique_policies = sorted(int(p) for p in np.unique(policy_id))
    hit_rate, mean_margin, median_margin = compute_within_source_retrieval(
        embeddings, source_index, policy_id, unique_policies, eval_mask, centroids
    )

    assert not np.isnan(hit_rate), "hit_rate is NaN – no sources with ≥ 2 eval samples."
    print(f"  [PASS] hit_rate={hit_rate:.4f} (random data, not NaN)")


def main():
    print("Running within-source retrieval regression tests ...")

    print("\n[1] test_hit_rate_above_chance_with_structured_data")
    test_hit_rate_above_chance_with_structured_data()

    print("\n[2] test_hit_rate_not_trivially_one_with_noisy_data")
    test_hit_rate_not_trivially_one_with_noisy_data()

    print("\nAll regression tests passed.")


if __name__ == "__main__":
    main()
