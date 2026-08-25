import copy
import json
from pathlib import Path

import pytest

from tools.stage6t_freeze_training_evaluation_protocol import (
    calculated_architecture_parameter_counts,
    validate_protocol_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "stage6t_training_evaluation_protocol.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage6t_config_freezes_attributable_abc_design():
    config = _config()

    summary = validate_protocol_config(config)

    assert summary["calculated_parameter_counts"] == {
        "legacy_single_gru_83_to_64": 106560,
        "single_gru_partitioned_16_48": 106560,
        "dual_branch_ego16_context48": 105616,
    }
    assert 0.95 <= summary["parameter_ratio_C_vs_B"] <= 1.05
    assert config["candidates"]["B"]["sampling_package"] == config["candidates"]["C"]["sampling_package"]
    assert config["candidates"]["B"]["objective_package"] == config["candidates"]["C"]["objective_package"]
    assert config["candidates"]["B"]["architecture"] != config["candidates"]["C"]["architecture"]


def test_stage6t_freeze_rejects_training_authorization():
    config = _config()
    config["authorization"]["training_authorized"] = True

    with pytest.raises(ValueError, match="training_authorized must be false"):
        validate_protocol_config(config)


def test_stage6t_freeze_rejects_B_C_nonarchitecture_confound():
    config = _config()
    config["candidates"]["C"]["sampling_package"] = "legacy_uniform_v1"

    with pytest.raises(ValueError, match="Candidate C does not match|differ only in encoder topology"):
        validate_protocol_config(config)


def test_stage6t_freeze_rejects_part_local_33d_target():
    config = _config()
    config["dataset_contract"]["legacy_33d_target_policy"]["authoritative_array"] = "interaction_feat_style.npy"

    with pytest.raises(ValueError, match="interaction_feat_style_raw"):
        validate_protocol_config(config)


def test_stage6t_freeze_rejects_cross_representation_raw_mmd_comparison():
    config = _config()
    config["stage6s_v2_interaction_scorecard"]["paired_null"][
        "cross_representation_raw_mmd2_comparison_forbidden"
    ] = False

    with pytest.raises(ValueError, match="raw MMD"):
        validate_protocol_config(config)


def test_stage6t_freeze_rejects_automatic_C_preference():
    config = copy.deepcopy(_config())
    config["architecture_decision_rule"]["C_is_not_automatically_preferred"] = False

    with pytest.raises(ValueError, match="must not be automatically preferred"):
        validate_protocol_config(config)


def test_parameter_count_formula_matches_frozen_config():
    counts = calculated_architecture_parameter_counts()

    assert counts["legacy_single_gru_83_to_64"] == 106560
    assert counts["single_gru_partitioned_16_48"] == 106560
    assert counts["dual_branch_ego16_context48"] == 105616
