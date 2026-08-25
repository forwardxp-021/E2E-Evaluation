import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.stage6u_smoke_unified_abc_trainer import synthetic_datasets
from tools.stage6u_create_formal_authorization import ordered_candidate_seed_pairs
from tools.stage6u_unified_abc_trainer import (
    UnifiedABCModel,
    assert_bc_fairness,
    assert_blind_path,
    build_encoder,
    build_random_plan,
    encoder_parameter_count,
    feature_group_indices,
    pairwise_euclidean,
    random_plan_ledger,
    update_validation_selection,
    validate_resume_plan,
    validate_formal_authorization,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _groups():
    schema = json.loads(
        (
            REPO_ROOT
            / "outputs/stage6r_dynamic_full51_semantic_strict_part_00_09/feature_schema.json"
        ).read_text(encoding="utf-8")
    )
    return feature_group_indices([row["name"] for row in schema["features"]])


def _plan(dataset, candidate):
    return build_random_plan(
        dataset,
        seed=3407,
        pair_seed=93407,
        epoch=0,
        epoch_samples=24,
        batch_size=8,
        candidate=candidate,
        sampling_package="dynamic_longitudinal_v2",
        dropout_package="dynamic_mask_aware_v2",
        slot_dropout_probability=0.15,
        all_neighbor_dropout_probability=0.05,
        ranking_margin=0.2,
    )


def test_all_encoders_export_64d_and_match_frozen_parameter_counts():
    expected = {"A": 106560, "B": 106560, "C": 105616}
    values = torch.randn(4, 80, 83)

    for candidate in "ABC":
        encoder = build_encoder(candidate)
        assert encoder(values).shape == (4, 64)
        assert encoder_parameter_count(encoder) == expected[candidate]


def test_bc_candidate_independent_random_streams_match():
    train, _ = synthetic_datasets(48, 40, 3407, _groups())

    ledger_b = random_plan_ledger(_plan(train, "B"))
    ledger_c = random_plan_ledger(_plan(train, "C"))

    audit = assert_bc_fairness(ledger_b, ledger_c)
    assert audit["all_streams_identical"] is True
    assert all(audit["stream_comparisons"].values())
    pair_types = np.asarray(_plan(train, "B")["pair_types"])
    assert np.bincount(pair_types, minlength=3).tolist() == [12, 6, 6]


def test_bc_fairness_detects_mutated_dropout_stream():
    train, _ = synthetic_datasets(48, 40, 3407, _groups())
    ledger_b = random_plan_ledger(_plan(train, "B"))
    ledger_c = copy.deepcopy(random_plan_ledger(_plan(train, "C")))
    ledger_c["field_sha256"]["slot_dropout_masks"] = "mutated"

    with pytest.raises(ValueError, match="fairness ledger mismatch"):
        assert_bc_fairness(ledger_b, ledger_c)


def test_mps_safe_pairwise_distance_matches_torch_cdist_on_cpu():
    generator = torch.Generator().manual_seed(3407)
    values = torch.randn(12, 7, generator=generator)

    observed = pairwise_euclidean(values)
    expected = torch.cdist(values, values, p=2)

    off_diagonal = ~torch.eye(len(values), dtype=torch.bool)
    assert torch.allclose(observed[off_diagonal], expected[off_diagonal], atol=1e-5, rtol=1e-5)
    assert torch.equal(torch.diagonal(observed), torch.zeros(len(values)))


def test_blind_path_rejects_waymo_test_and_nuplan():
    with pytest.raises(ValueError, match="Forbidden"):
        assert_blind_path(Path("outputs/nuplan_formal_eval"))
    with pytest.raises(ValueError, match="split must be train or val"):
        from tools.stage6u_unified_abc_trainer import DynamicTrainValDataset

        DynamicTrainValDataset(Path("manifest.json"), "test", Path("stats.json"))


def test_formal_training_fails_without_separate_authorization(tmp_path):
    config = json.loads(
        (REPO_ROOT / "configs/stage6u_unified_abc_trainer.json").read_text(encoding="utf-8")
    )

    with pytest.raises(PermissionError, match="separate authorization"):
        validate_formal_authorization(
            config, None, None, tmp_path / "formal", candidate="A", seed=3407
        )


def test_unified_model_uses_one_candidate_switch():
    groups = _groups()

    models = {candidate: UnifiedABCModel(candidate, groups) for candidate in "ABC"}

    assert set(models) == {"A", "B", "C"}
    assert all(model(torch.zeros(2, 80, 83)).shape == (2, 64) for model in models.values())


def test_epoch_boundary_resume_does_not_compare_previous_epoch_plan():
    payload = {"next_batch_index": 0, "plan_ledger": None}
    current = {"candidate_independent_fingerprint_sha256": "next-epoch"}

    assert validate_resume_plan(payload, current) == "epoch_boundary_no_plan_check"


def test_mid_epoch_resume_requires_matching_plan():
    payload = {
        "next_batch_index": 100,
        "plan_ledger": {"candidate_independent_fingerprint_sha256": "frozen-plan"},
    }
    current = {"candidate_independent_fingerprint_sha256": "frozen-plan"}
    assert validate_resume_plan(payload, current) == "mid_epoch_plan_match"

    mutated = {"candidate_independent_fingerprint_sha256": "mutated-plan"}
    with pytest.raises(ValueError, match="Resume random plan differs"):
        validate_resume_plan(payload, mutated)


def test_formal_authorization_uses_frozen_candidate_major_order():
    assert ordered_candidate_seed_pairs(list("ABC"), [3407, 3408, 3409]) == [
        (candidate, seed) for candidate in "ABC" for seed in (3407, 3408, 3409)
    ]


def test_best_val_selection_is_exact_but_patience_uses_min_delta():
    result = update_validation_selection(
        val_loss=0.99995,
        epoch=2,
        best_val_loss=1.0,
        best_epoch=1,
        early_stopping_reference=1.0,
        patience_count=0,
        min_delta=0.0001,
    )

    assert result["best_improved"] is True
    assert result["best_val_loss"] == 0.99995
    assert result["best_epoch"] == 2
    assert result["patience_improved"] is False
    assert result["early_stopping_reference"] == 1.0
    assert result["patience_count"] == 1
