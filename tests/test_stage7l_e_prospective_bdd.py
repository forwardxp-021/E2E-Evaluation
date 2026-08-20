import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from tools.interaction_context_features import get_feature_schema
from tools.stage6l_run_context_representation_ablation import (
    kernel_analysis,
    signed_quadratic_null,
)
from tools.stage6u_unified_abc_trainer import UnifiedABCModel, feature_group_indices
from tools.stage7l_e_prepare_input_contract import build_dose_view, validate_unlock
from tools.stage7l_e_run_prospective_bdd import (
    NULL_SEED_BASE,
    PRIMARY_KEY,
    apply_fixed_holm,
    cell_seed,
)
from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_unlock_requires_every_frozen_stage7l_d_gate() -> None:
    manifest = {
        "status": "STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED",
        "representation_status": "STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED",
        "execution": {"successful_cells": 400, "complete_all_five_doses": 80},
        "gates": {
            "execution": True,
            "canonical_identity": True,
            "mechanism": True,
            "longitudinal_nuisance": True,
            "safety_validity": True,
            "representation_unlock": True,
        },
    }
    validate_unlock(manifest)
    manifest["gates"]["mechanism"] = False
    try:
        validate_unlock(manifest)
    except RuntimeError as exc:
        assert "mechanism" in str(exc)
    else:
        raise AssertionError("failed Stage7L-D gate must block Stage7L-E")


def test_frozen_149_step_input_is_zero_false_padded_to_150(tmp_path: Path) -> None:
    source = tmp_path / "cell" / "stage7c_output"
    source.mkdir(parents=True)
    seq = np.ones((1, 1, 149, 8), dtype=np.float32)
    mask = np.ones((1, 1, 149), dtype=np.uint8)
    np.save(source / "simulated_ego_seq.npy", seq)
    np.save(source / "simulated_ego_seq_mask.npy", mask)
    write_csv(
        source / "scenario_planner_index.csv",
        [{
            "scenario_index": 0,
            "planner_id": 0,
            "planner_name": "stage7l_b2_pure_lateral_dose0",
            "status": "succeeded",
            "num_timesteps": 149,
            "log_name": "log-1",
            "scenario_token": "token-1",
        }],
    )
    write_csv(
        source / "simulated_planner_metadata.csv",
        [{"planner_name": "stage7l_b2_pure_lateral_dose0", "planner_id": 0}],
    )
    msgpack = source / "official_nuplan_runs/scenario_0/stage7l_b2_pure_lateral_dose0/run/token-1.msgpack.xz"
    msgpack.parent.mkdir(parents=True)
    msgpack.write_bytes(b"frozen-msgpack")
    output = tmp_path / "view"
    audit = build_dose_view(
        dose="dose0",
        roster=[{"collection_order": "1", "scenario_token": "token-1", "log_name": "log-1", "direction": "left"}],
        sources={("token-1", "dose0"): {"attempt_dir": str(source.parent), "attempt_id": "A01"}},
        output_dir=output,
    )
    padded = np.load(output / "simulated_ego_seq.npy")
    padded_mask = np.load(output / "simulated_ego_seq_mask.npy")
    assert padded.shape == (1, 1, 150, 8)
    assert padded_mask.shape == (1, 1, 150)
    assert np.all(padded[0, 0, :149] == 1.0)
    assert np.all(padded[0, 0, 149] == 0.0)
    assert padded_mask[0, 0, :149].all()
    assert not padded_mask[0, 0, 149]
    assert audit["padding_policy"] == "zero_values_false_mask_right_pad_to_150"


def test_vectorized_pair_swap_matches_exhaustive_small_n() -> None:
    contrast = np.asarray(
        [[0.3, -0.1, 0.05], [-0.1, 0.4, 0.02], [0.05, 0.02, 0.2]],
        dtype=np.float64,
    )
    exhaustive = sorted(
        float(np.asarray(signs) @ contrast @ np.asarray(signs) / 9.0)
        for signs in product((-1.0, 1.0), repeat=3)
    )
    # Seeded vectorization draws from the same mathematical sign-swap support.
    samples = signed_quadratic_null(contrast, repetitions=8000, seed=17)
    assert set(np.round(samples, 12)).issubset(set(np.round(exhaustive, 12)))
    assert np.isclose(np.mean(samples), np.mean(exhaustive), atol=0.01)


def test_plus_one_null_diagnostics_and_determinism() -> None:
    reference = np.zeros((8, 3), dtype=np.float64)
    target = np.ones((8, 3), dtype=np.float64)
    first, samples1, _ = kernel_analysis(reference, target, repetitions=1000, seed=123)
    second, samples2, _ = kernel_analysis(reference, target, repetitions=1000, seed=123)
    assert np.array_equal(samples1, samples2)
    assert first == second
    assert first["raw_p"] == (first["exceedance_count"] + 1) / 1001
    assert first["paired_null_q95"] == float(np.quantile(samples1, 0.95))
    assert first["null_standardized_z_bdd"] == (
        first["mmd2"] - first["paired_null_mean"]
    ) / first["paired_null_sd"]


def test_frozen_seed_policy_is_deterministic_stage6v_inheritance() -> None:
    assert cell_seed(0, 0, 0) == NULL_SEED_BASE
    assert cell_seed(2, 3, 0) == NULL_SEED_BASE + 2300
    assert cell_seed(2, 3, 0) == cell_seed(2, 3, 0)


def test_primary_excluded_once_and_holm_family_stays_39_with_p1_cell() -> None:
    cells = []
    for representation in ("old64", "A_seed3407", "B_seed3407", "C_seed3407", "ego13"):
        for dose in ("dose25", "dose50", "dose75", "dose100"):
            for task in ("LAT.LANE_CHANGE", "LAT.DYNAMICS"):
                key = (representation, dose, task)
                primary = key == PRIMARY_KEY
                cells.append(
                    {
                        "representation": representation,
                        "dose": dose,
                        "task": task,
                        "raw_p_for_multiplicity": 1.0 if len(cells) == 0 else 0.01,
                        "multiplicity_role": (
                            "PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY"
                            if primary
                            else "SECONDARY_HOLM_39"
                        ),
                    }
                )
    apply_fixed_holm(cells)
    secondary = [row for row in cells if row["multiplicity_role"] == "SECONDARY_HOLM_39"]
    primary = [row for row in cells if row["multiplicity_role"].startswith("PRIMARY_")]
    assert len(cells) == 40
    assert len(secondary) == 39
    assert len(primary) == 1
    assert primary[0]["holm_p"] is None
    assert next(row for row in secondary if row["raw_p_for_multiplicity"] == 1.0)["holm_p"] == 1.0


def test_all_frozen_encoder_topologies_forward_83d_to_64d() -> None:
    names = [row["name"] for row in get_feature_schema()["features"]]
    groups = feature_group_indices(names)
    batch = torch.zeros((2, 150, 83), dtype=torch.float32)
    models = [
        ContextFlattenGRUEncoder(input_dim=83, hidden_dim=128, embedding_dim=64),
        UnifiedABCModel("A", groups),
        UnifiedABCModel("B", groups),
        UnifiedABCModel("C", groups),
    ]
    for model in models:
        model.eval()
        output = model(batch)
        assert output.shape == (2, 64)
        assert torch.isfinite(output).all()
        assert not model.training


def test_tool_hard_codes_no_raw_mmd_cross_representation_ranking() -> None:
    source = Path("tools/stage7l_e_run_prospective_bdd.py").read_text(encoding="utf-8")
    assert '"cross_representation_raw_mmd2_comparison_performed": False' in source
    assert '"stage6v_qualification_changed": False' in source
