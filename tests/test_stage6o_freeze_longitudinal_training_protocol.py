import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.stage6o_freeze_longitudinal_training_protocol import run_freeze, split_from_scenario_id


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "stage6o_longitudinal_representation_training_protocol.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_for_split(split: str, count: int):
    result = []
    index = 0
    while len(result) < count:
        candidate = f"synthetic_{split}_{index}"
        if split_from_scenario_id(candidate) == split:
            result.append(candidate)
        index += 1
    return result


def _record(path: Path):
    return {"path": str(path), "sha256": _sha(path)}


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _make_fixture(tmp_path: Path):
    shard = tmp_path / "shard_000000"
    shard.mkdir()
    n_rows = 10
    splits = np.asarray(["train"] * 8 + ["val", "test"], dtype=object)
    scenario_ids = _ids_for_split("train", 8) + _ids_for_split("val", 1) + _ids_for_split("test", 1)

    context = np.zeros((n_rows, 80, 83), dtype=np.float32)
    speeds = np.asarray([2.0, 7.0, 20.0, 2.0, 7.0, 20.0, 7.0, 2.0, 7.0, 20.0])
    context[:, :, 5] = speeds[:, None]
    context[0, :, 5] = np.linspace(0.0, 6.0, 80)
    context[:, :, 6] = np.linspace(0.05, 0.8, n_rows)[:, None]
    mask = np.zeros((n_rows, 80, 5), dtype=np.float32)
    mask[1, :20, 0] = 1.0
    mask[2, :60, 0] = 1.0
    mask[4, :40, 0] = 1.0
    mask[5, :80, 0] = 1.0
    mask[7, :10, 0] = 1.0
    feat_raw = np.ones((n_rows, 33), dtype=np.float32)
    feat_raw[:, 0] = np.linspace(0.1, 1.0, n_rows)
    feat_raw[:, 1] = np.linspace(0.2, 1.1, n_rows)
    feat_raw[:, 2] = np.linspace(0.3, 1.2, n_rows)
    feat_raw[:, 3] = np.linspace(0.4, 1.3, n_rows)
    feat_raw[:, 4] = np.linspace(1.0, 2.0, n_rows)
    feat_raw[:, 5] = np.linspace(0.5, 1.5, n_rows)
    feat_raw[::2, 13] = 0.0
    feat = feat_raw.copy()
    meta_dtype = np.dtype(
        [
            ("row_index", "i4"),
            ("scenario_id", "O"),
            ("target_agent_id", "O"),
            ("start", "i4"),
            ("window_len", "i4"),
            ("split", "O"),
        ]
    )
    meta = np.asarray(
        [(i, scenario_ids[i], f"agent_{i}", 0, 80, splits[i]) for i in range(n_rows)], dtype=meta_dtype
    )
    np.save(shard / "context_traj.npy", context)
    np.save(shard / "context_mask.npy", mask)
    np.save(shard / "interaction_feat_style.npy", feat)
    np.save(shard / "interaction_feat_style_raw.npy", feat_raw)
    np.save(shard / "split.npy", splits)
    np.save(shard / "meta.npy", meta)

    manifest_path = tmp_path / "shard_manifest.json"
    _write_json(manifest_path, {"shard_paths": [str(shard)], "total_windows": n_rows})
    build_summary_path = tmp_path / "build_summary.json"
    _write_json(
        build_summary_path,
        {
            "total_shards": 1,
            "total_windows": n_rows,
            "good_lane_context_rate": 1.0,
            "lane_assignment_success_rate": 1.0,
            "fallback_assignment_rate": 0.0,
            "slot_occupied_window_ratio": {
                "front": 0.5,
                "left_front": 0.2,
                "left_rear": 0.2,
                "right_front": 0.2,
                "right_rear": 0.2,
            },
            "nonfinite_output_detected": 0,
            "warnings": [],
        },
    )
    feature_schema_path = tmp_path / "feature_schema.json"
    _write_json(feature_schema_path, {"feature_dim": 33, "features": []})
    standardization_path = tmp_path / "standardization.json"
    _write_json(standardization_path, {"mean": [0.0] * 33, "std": [1.0] * 33, "train_count": 8})
    standardization_report_path = tmp_path / "standardization_report.json"
    _write_json(standardization_report_path, {"train_count": 8, "feature_dim": 33})
    checkpoint_path = tmp_path / "baseline.pt"
    checkpoint_path.write_bytes(b"frozen-baseline")
    evaluation_path = tmp_path / "evaluation.json"
    _write_json(evaluation_path, {"paper_grade_valid": True})
    category_path = tmp_path / "category.csv"
    category_path.write_text("category,value\nlongitudinal,1\n", encoding="utf-8")
    retrieval_path = tmp_path / "retrieval.csv"
    retrieval_path.write_text("representation,hit_at_5\nbaseline,0.5\n", encoding="utf-8")
    evidence_paths = []
    for name in ("stage6l.json", "stage6m.json", "dose.json"):
        path = tmp_path / name
        _write_json(path, {"status": "frozen"})
        evidence_paths.append(path)

    config = copy.deepcopy(json.loads(BASE_CONFIG.read_text(encoding="utf-8")))
    config["source_dataset"]["manifest"] = _record(manifest_path)
    config["source_dataset"]["build_summary"] = _record(build_summary_path)
    config["source_dataset"]["feature_schema"] = _record(feature_schema_path)
    config["source_dataset"]["standardization"] = {**_record(standardization_path), "fit_split": "train"}
    config["source_dataset"]["standardization_report"] = _record(standardization_report_path)
    config["baseline"]["checkpoint"] = _record(checkpoint_path)
    config["baseline"]["evaluation_summary"] = _record(evaluation_path)
    config["baseline"]["category_correlation"] = _record(category_path)
    config["baseline"]["retrieval_metrics"] = _record(retrieval_path)
    for key, path in zip(config["nuplan_acceptance"]["authoritative_evidence"], evidence_paths):
        config["nuplan_acceptance"]["authoritative_evidence"][key] = _record(path)
    config["expected_dataset"].update(
        {"total_shards": 1, "total_windows": n_rows, "split_counts": {"train": 8, "val": 1, "test": 1}}
    )
    config["coverage_audit"]["min_train_windows_per_speed_bin"] = 1
    config["coverage_audit"]["min_train_windows_per_front_regime"] = 1
    config["coverage_audit"]["min_train_windows_per_nonempty_speed_front_cell"] = 1
    config["coverage_audit"]["min_train_stop_go_windows"] = 1
    config["coverage_audit"]["min_train_steady_speed_windows"] = 1
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    return config_path, shard


def test_freeze_writes_audit_but_never_authorizes_training(tmp_path):
    config_path, _ = _make_fixture(tmp_path)
    out_dir = tmp_path / "freeze"

    manifest = run_freeze(config_path, out_dir)

    assert manifest["status"] == "FROZEN_READY_FOR_IMPLEMENTATION_NOT_TRAINING"
    assert manifest["training_authorized"] is False
    assert manifest["checkpoint_write_authorized"] is False
    assert manifest["baseline_overwrite_authorized"] is False
    assert manifest["validation"]["pass"] is True
    assert manifest["waymo_data_audit"]["split_counts"] == {"train": 8, "val": 1, "test": 1}
    assert manifest["waymo_data_audit"]["scenario_cross_split_overlap_count"] == 0
    assert (out_dir / "stage6o_waymo_data_audit.json").is_file()
    assert (out_dir / "stage6o_training_protocol_report_zh.md").is_file()
    assert not list(out_dir.glob("*.pt"))


def test_freeze_fails_closed_when_scenario_split_is_mutated(tmp_path):
    config_path, shard = _make_fixture(tmp_path)
    meta = np.load(shard / "meta.npy", allow_pickle=True)
    split = np.load(shard / "split.npy", allow_pickle=True)
    meta[1]["scenario_id"] = meta[0]["scenario_id"]
    meta[1]["split"] = "val"
    split[1] = "val"
    np.save(shard / "meta.npy", meta)
    np.save(shard / "split.npy", split)

    with pytest.raises(ValueError, match="split algorithm mismatch|Scenario leakage"):
        run_freeze(config_path, tmp_path / "freeze")


def test_freeze_rejects_training_authorization_in_freeze_config(tmp_path):
    config_path, _ = _make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["authorization"]["training_authorized"] = True
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="training_authorized must be false"):
        run_freeze(config_path, tmp_path / "freeze")


def test_freeze_writes_blocked_manifest_when_coverage_is_insufficient(tmp_path):
    config_path, _ = _make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["coverage_audit"]["min_train_windows_per_speed_bin"] = 9
    _write_json(config_path, config)

    manifest = run_freeze(config_path, tmp_path / "freeze")

    assert manifest["status"] == "FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING"
    assert manifest["training_authorized"] is False
    assert manifest["validation"]["coverage_pass"] is False
    assert manifest["validation"]["pass"] is False
