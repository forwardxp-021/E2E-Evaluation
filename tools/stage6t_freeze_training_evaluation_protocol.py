#!/usr/bin/env python3
"""Freeze the Stage 6T A/B/C training and blinded-evaluation protocol.

This command audits immutable inputs and writes protocol evidence only.  It
does not import a trainer, write a checkpoint, run Waymo test, run nuPlan, or
read any embedding/BDD/MMD result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def _gru_parameter_count(input_dim: int, hidden_dim: int) -> int:
    return 3 * hidden_dim * (input_dim + hidden_dim) + 6 * hidden_dim


def _linear_parameter_count(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


def calculated_architecture_parameter_counts() -> dict[str, int]:
    single = (
        _gru_parameter_count(83, 128)
        + _linear_parameter_count(128, 128)
        + _linear_parameter_count(128, 64)
    )
    dual = (
        _gru_parameter_count(8, 48)
        + _linear_parameter_count(48, 48)
        + _linear_parameter_count(48, 16)
        + _gru_parameter_count(83, 120)
        + _linear_parameter_count(120, 120)
        + _linear_parameter_count(120, 48)
    )
    return {
        "legacy_single_gru_83_to_64": single,
        "single_gru_partitioned_16_48": single,
        "dual_branch_ego16_context48": dual,
    }


def validate_protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(
        config,
        [
            "schema_version", "protocol_id", "issue", "status", "authorization", "source_records",
            "dataset_contract", "frozen_baseline", "common_optimization", "architecture_definitions",
            "sampling_packages", "dropout_packages", "objective_packages", "candidates",
            "checkpoint_selection", "waymo_test_scorecard", "blinded_evaluation_sequence",
            "stage6jk_paired_scorecard", "stage6p_unpaired_scorecard",
            "stage6s_v2_interaction_scorecard", "candidate_C_success_rule",
            "architecture_decision_rule", "forbidden_information_and_actions",
        ],
        "config",
    )
    if config["schema_version"] != "stage6t_training_evaluation_protocol_v1":
        raise ValueError("Unexpected Stage6T schema_version")
    if int(config["issue"]) != 262:
        raise ValueError("Stage6T protocol must reference GitHub Issue #262")

    auth = config["authorization"]
    for key in (
        "training_authorized", "checkpoint_write_authorized", "waymo_test_authorized",
        "nuplan_evaluation_authorized", "confirmation_rollout_authorized",
    ):
        if auth.get(key) is not False:
            raise ValueError(f"authorization.{key} must be false during Stage6T freeze")

    data = config["dataset_contract"]
    if int(data.get("output_dim", -1)) != 64 or int(data.get("context_dim", -1)) != 83:
        raise ValueError("Stage6T must preserve the 83D input and 64D exported embedding interface")
    if data.get("split_unit") != "scenario_id" or data.get("normalization_fit_split") != "train_only":
        raise ValueError("Stage6T requires scenario-level splits and train-only fitted statistics")
    target_policy = data.get("legacy_33d_target_policy", {})
    if target_policy.get("authoritative_array") != "interaction_feat_style_raw.npy":
        raise ValueError("Stage6T 33D target must be derived from interaction_feat_style_raw.npy")
    if target_policy.get("forbidden_training_array") != "interaction_feat_style.npy":
        raise ValueError("Part-locally standardized interaction_feat_style.npy must be forbidden for training")
    if target_policy.get("rewrite_frozen_dynamic_shards") is not False:
        raise ValueError("Stage6T must not rewrite Dynamic v2 shards")

    candidates = config["candidates"]
    if set(candidates) != {"A", "B", "C"}:
        raise ValueError("Stage6T requires exactly candidates A, B and C")
    expected = {
        "A": ("legacy_single_gru_83_to_64", "legacy_uniform_v1", "none", "legacy_stage5d_balanced_v2_exact", False),
        "B": ("single_gru_partitioned_16_48", "dynamic_longitudinal_v2", "dynamic_mask_aware_v2", "longitudinal_recovery_v2", True),
        "C": ("dual_branch_ego16_context48", "dynamic_longitudinal_v2", "dynamic_mask_aware_v2", "longitudinal_recovery_v2", True),
    }
    output_roots: set[str] = set()
    for candidate_id, (architecture, sampling, dropout, objective, clean_target) in expected.items():
        row = candidates[candidate_id]
        observed = (
            row.get("architecture"), row.get("sampling_package"), row.get("dropout_package"),
            row.get("objective_package"), row.get("uses_clean_longitudinal_v2"),
        )
        if observed != (architecture, sampling, dropout, objective, clean_target):
            raise ValueError(f"Candidate {candidate_id} does not match the frozen attribution matrix: {observed}")
        if row.get("uses_dynamic_builder_v2") is not True:
            raise ValueError(f"Candidate {candidate_id} must use Dynamic Builder v2")
        root = str(row.get("output_root", ""))
        if not root or root in output_roots:
            raise ValueError("Candidate output roots must be non-empty and unique")
        output_roots.add(root)
    if candidates["A"].get("causal_wording_limit") is None:
        raise ValueError("Candidate A must retain the no-A0 causal-wording limitation")

    architectures = config["architecture_definitions"]
    calculated = calculated_architecture_parameter_counts()
    for name, expected_count in calculated.items():
        observed_count = int(architectures[name].get("encoder_projection_parameter_count", -1))
        if observed_count != expected_count:
            raise ValueError(f"{name} parameter count mismatch: expected={expected_count}, observed={observed_count}")
    ratio = calculated["dual_branch_ego16_context48"] / calculated["legacy_single_gru_83_to_64"]
    if not 0.95 <= ratio <= 1.05:
        raise ValueError(f"Candidate C is not parameter-matched to B: ratio={ratio}")
    if architectures["dual_branch_ego16_context48"].get("context_branch_keeps_full_ego_and_neighbor_context") is not True:
        raise ValueError("Candidate C must retain traffic/context information")

    b = candidates["B"]
    c = candidates["C"]
    for key in ("sampling_package", "dropout_package", "objective_package"):
        if b[key] != c[key]:
            raise ValueError(f"Candidates B and C must differ only in encoder topology; mismatch in {key}")
    if config["dropout_packages"]["dynamic_mask_aware_v2"].get("same_dropout_draws_required_for_B_and_C") is not True:
        raise ValueError("Candidates B and C require identical dropout draws")

    objectives = config["objective_packages"]
    recovery = objectives["longitudinal_recovery_v2"]
    if float(recovery.get("raw_bdd_weight", -1.0)) != 0.0 or float(recovery.get("nuplan_metric_weight", -1.0)) != 0.0:
        raise ValueError("BDD/MMD and nuPlan outcomes must have zero training weight")
    if recovery.get("global_style_and_legacy_group_target_source") != "stage6t_global_standardized_interaction_feat_style_raw_33d":
        raise ValueError("B/C must share the Stage6T global 33D training target")
    if objectives["legacy_stage5d_balanced_v2_exact"].get("feature_target_source") != "stage6t_global_standardized_interaction_feat_style_raw_33d":
        raise ValueError("A must use the same globally standardized raw 33D target source")

    optimization = config["common_optimization"]
    seeds = [int(seed) for seed in optimization.get("seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3 or int(optimization.get("primary_seed", -1)) not in seeds:
        raise ValueError("Exactly three unique seeds and one predeclared primary seed are required")
    expected_steps = math.ceil(int(data["split_counts"]["train"]) / int(optimization["batch_size"])) * int(
        optimization["max_epochs"]
    )
    if int(optimization.get("max_total_optimizer_steps_per_seed", -1)) != expected_steps:
        raise ValueError(f"Optimizer-step budget must equal the frozen full-epoch cap {expected_steps}")
    if optimization.get("same_budget_for_candidates") != ["A", "B", "C"]:
        raise ValueError("A/B/C must use the same optimization budget")

    checkpoint = config["checkpoint_selection"]
    if checkpoint.get("best_of_three_seed_selection_for_nuplan_forbidden") is not True:
        raise ValueError("Best-of-three seed selection using downstream results must be forbidden")
    if checkpoint.get("all_nine_candidate_seed_checkpoints_locked_before_waymo_test") is not True:
        raise ValueError("All nine A/B/C checkpoints must lock before Waymo test")
    if checkpoint.get("overwrite_existing_seed_directory") is not False:
        raise ValueError("Existing candidate seed directories must never be overwritten")

    waymo = config["waymo_test_scorecard"]
    if waymo.get("old64_reevaluated_on_same_dynamic_v2_test") is not True:
        raise ValueError("old64 must be re-evaluated once on the same Dynamic v2 test rows")
    if waymo.get("historical_old_builder_test_metrics_used_as_gate") is not False:
        raise ValueError("Historical old-builder metrics cannot serve as the Dynamic v2 gate")
    if waymo.get("test_result_driven_retraining_allowed") is not False:
        raise ValueError("Waymo test-driven retraining must be forbidden")

    for label in ("stage6jk_paired_scorecard", "stage6p_unpaired_scorecard"):
        if config[label].get("cross_representation_raw_mmd2_comparison_forbidden") is not True:
            raise ValueError(f"{label} must forbid cross-representation raw MMD² comparison")
    interaction = config["stage6s_v2_interaction_scorecard"]
    if interaction.get("confirmation_pair_count") != 80 or interaction.get("confirmation_roster_immutable") is not True:
        raise ValueError("Stage6S-v2 must retain the immutable 80-pair confirmation roster")
    if interaction["paired_null"].get("cross_representation_raw_mmd2_comparison_forbidden") is not True:
        raise ValueError("Stage6S-v2 must forbid cross-representation raw MMD² comparison")
    increment = interaction["c_context_increment"]
    if increment.get("raw_mmd2_difference_forbidden") is not True or increment.get("primary_statistic") != (
        "difference_in_representation_specific_null_standardized_z_bdd"
    ):
        raise ValueError("C context increment must use null-standardized delta-Z, never raw MMD² difference")

    decision = config["architecture_decision_rule"]
    if decision.get("C_is_not_automatically_preferred") is not True:
        raise ValueError("Candidate C must not be automatically preferred")
    if decision.get("post_blind_result_retraining_or_threshold_change_allowed") is not False:
        raise ValueError("Post-blind retraining or threshold changes must be forbidden")
    forbidden = set(config["forbidden_information_and_actions"])
    required_forbidden = {
        "cross_representation_raw_mmd2_comparison",
        "training_ego_only_as_final_candidate",
        "overwriting_or_deleting_old64",
        "running_training_waymo_test_nuplan_evaluation_or_confirmation_during_stage6t_freeze",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError(f"Missing forbidden actions: {sorted(required_forbidden - forbidden)}")
    return {"calculated_parameter_counts": calculated, "parameter_ratio_C_vs_B": ratio, "expected_steps": expected_steps}


def verify_source_records(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for label, record in config["source_records"].items():
        _require_keys(record, ["path", "sha256"], f"source_records.{label}")
        path = resolve_repo_path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"source_records.{label} missing: {path}")
        actual = sha256_file(path)
        if actual != str(record["sha256"]):
            raise ValueError(
                f"source_records.{label} SHA-256 mismatch: expected={record['sha256']}, actual={actual}, path={path}"
            )
        records[label] = {"path": str(path), "sha256": actual, "size_bytes": int(path.stat().st_size)}
    return records


def _write_candidate_matrix(config: Mapping[str, Any], path: Path) -> None:
    columns = [
        "candidate", "role", "dataset", "architecture", "sampling", "dropout", "objective",
        "clean_longitudinal_v2", "exported_embedding_dim", "attribution_interpretation",
    ]
    rows = []
    for candidate_id, candidate in config["candidates"].items():
        rows.append(
            {
                "candidate": candidate_id,
                "role": candidate["role"],
                "dataset": config["dataset_contract"]["dataset_id"],
                "architecture": candidate["architecture"],
                "sampling": candidate["sampling_package"],
                "dropout": candidate["dropout_package"],
                "objective": candidate["objective_package"],
                "clean_longitudinal_v2": candidate["uses_clean_longitudinal_v2"],
                "exported_embedding_dim": 64,
                "attribution_interpretation": (
                    "dynamic-data-dominant vs historical old64; pure data causality requires optional A0"
                    if candidate_id == "A"
                    else ("longitudinal objective/sampling increment without encoder topology change" if candidate_id == "B" else "encoder topology increment relative to B")
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_npy(path: Path, allow_pickle: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode=None if allow_pickle else "r", allow_pickle=allow_pickle)


def audit_dynamic_dataset(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ledger: Mapping[str, Any],
    verify_shard_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = config["dataset_contract"]
    expected_counts = {key: int(value) for key, value in contract["split_counts"].items()}
    if int(manifest.get("shard_count", -1)) != int(contract["shard_count"]):
        raise ValueError("Dynamic manifest shard_count does not match Stage6T")
    if int(manifest.get("row_count", -1)) != int(contract["row_count"]):
        raise ValueError("Dynamic manifest row_count does not match Stage6T")
    if {key: int(value) for key, value in manifest.get("split_counts", {}).items()} != expected_counts:
        raise ValueError("Dynamic manifest split_counts do not match Stage6T")
    canonical = {key: value for key, value in manifest.items() if key != "content_signature_sha256"}
    if sha256_json(canonical) != manifest.get("content_signature_sha256"):
        raise ValueError("Dynamic manifest content_signature_sha256 is invalid")
    if len(manifest.get("part_roots", [])) != 6:
        raise ValueError("Dynamic full51 must contain six frozen part roots")

    required = list(contract["required_files_per_shard"])
    ledger_rows = ledger.get("shard_artifact_sha256", {})
    split_counts: Counter[str] = Counter()
    scenario_split: dict[str, str] = {}
    overlaps: set[str] = set()
    shape_failures: list[str] = []
    source_hash_mismatches: list[str] = []
    rows = 0
    raw_sum = np.zeros(33, dtype=np.float64)
    raw_sumsq = np.zeros(33, dtype=np.float64)
    raw_train_count = 0
    raw_nonfinite = 0
    training_input_hashes: dict[str, dict[str, str]] = {}

    shard_paths = [Path(path) for path in manifest["shard_paths"]]
    for shard in tqdm(shard_paths, desc="Stage6T dataset audit", unit="shard"):
        if not shard.is_dir():
            raise FileNotFoundError(f"Dynamic shard missing: {shard}")
        missing = [name for name in required if not (shard / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{shard} missing required Stage6T inputs: {missing}")
        context = _load_npy(shard / "context_traj.npy")
        n_rows = int(context.shape[0])
        expected_shapes = {
            "context_traj.npy": (n_rows, 80, 83),
            "context_mask.npy": (n_rows, 80, 5),
            "interaction_feat_style.npy": (n_rows, 33),
            "interaction_feat_style_raw.npy": (n_rows, 33),
            "longitudinal_supervision_v2.npy": (n_rows, 80, 3),
            "longitudinal_supervision_v2_raw.npy": (n_rows, 80, 3),
            "slot_valid_mask.npy": (n_rows, 5, 80),
            "slot_identity_switch_mask.npy": (n_rows, 5, 80),
            "slot_derivative_valid_mask.npy": (n_rows, 5, 80),
            "meta.npy": (n_rows,),
            "split.npy": (n_rows,),
        }
        arrays: dict[str, np.ndarray] = {"context_traj.npy": context}
        for name in required:
            if name not in arrays:
                arrays[name] = _load_npy(shard / name, allow_pickle=name in {"meta.npy", "split.npy"})
            if tuple(arrays[name].shape) != expected_shapes[name]:
                shape_failures.append(f"{shard}/{name}: expected={expected_shapes[name]}, observed={arrays[name].shape}")
        split = arrays["split.npy"].astype(str)
        meta = arrays["meta.npy"]
        if meta.dtype.names is None or not {"scenario_id", "split"}.issubset(meta.dtype.names):
            raise ValueError(f"{shard}/meta.npy lacks scenario_id/split fields")
        if not np.array_equal(split, meta["split"].astype(str)):
            raise ValueError(f"{shard} split.npy and meta.npy split fields differ")
        if not set(np.unique(split)).issubset(SPLITS):
            raise ValueError(f"{shard} contains an unexpected split label: {np.unique(split).tolist()}")
        for name, count in zip(*np.unique(split, return_counts=True)):
            split_counts[str(name)] += int(count)
        for scenario_id, split_name in zip(meta["scenario_id"], split):
            sid = str(scenario_id)
            previous = scenario_split.setdefault(sid, str(split_name))
            if previous != str(split_name):
                overlaps.add(sid)

        raw = np.asarray(arrays["interaction_feat_style_raw.npy"], dtype=np.float64)
        raw_nonfinite += int(raw.size - np.isfinite(raw).sum())
        train_raw = raw[split == "train"]
        raw_sum += train_raw.sum(axis=0)
        raw_sumsq += np.square(train_raw).sum(axis=0)
        raw_train_count += int(len(train_raw))
        rows += n_rows

        frozen_hashes = ledger_rows.get(str(shard))
        if not isinstance(frozen_hashes, Mapping):
            raise ValueError(f"SHA ledger has no row for {shard}")
        current_hashes: dict[str, str] = {}
        hash_names = sorted(set(frozen_hashes) | {"interaction_feat_style_raw.npy"})
        for name in hash_names:
            actual = sha256_file(shard / name)
            current_hashes[name] = actual
            if verify_shard_hashes and name in frozen_hashes and actual != str(frozen_hashes[name]):
                source_hash_mismatches.append(f"{shard}/{name}")
        training_input_hashes[str(shard)] = current_hashes

    if shape_failures:
        raise ValueError(f"Dynamic shard shape failures: {shape_failures[:5]}")
    if source_hash_mismatches:
        raise ValueError(f"Dynamic shard SHA-256 mismatches: {source_hash_mismatches[:5]}")
    if rows != int(contract["row_count"]) or dict(split_counts) != expected_counts:
        raise ValueError(f"Observed dataset counts differ: rows={rows}, split_counts={dict(split_counts)}")
    if overlaps:
        raise ValueError(f"Scenario leakage detected across splits: {sorted(overlaps)[:5]}")
    if raw_nonfinite:
        raise ValueError(f"Raw 33D supervision contains {raw_nonfinite} non-finite values")
    if raw_train_count != expected_counts["train"]:
        raise ValueError(f"Raw 33D train_count mismatch: {raw_train_count}")
    mean = raw_sum / raw_train_count
    variance = np.maximum(raw_sumsq / raw_train_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    global_standardization = {
        "schema_version": "stage6t_global_interaction_target_standardization_v1",
        "source_array": "interaction_feat_style_raw.npy",
        "fit_split": "train",
        "train_count": raw_train_count,
        "feature_dim": 33,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "variance_convention": "population_ddof_0",
        "epsilon_floor": 1e-6,
        "apply_unchanged_to": ["train", "val", "test"],
        "frozen_shards_rewritten": False,
        "part_local_interaction_feat_style_npy_allowed_for_stage6t_training": False,
    }
    audit = {
        "row_count": rows,
        "shard_count": len(shard_paths),
        "split_counts": {key: int(split_counts[key]) for key in SPLITS},
        "scenario_count": len(scenario_split),
        "scenario_cross_split_overlap_count": len(overlaps),
        "shape_failure_count": len(shape_failures),
        "raw_33d_nonfinite_count": raw_nonfinite,
        "raw_33d_global_train_standardization_required": True,
        "verified_existing_stage6r_ledger": bool(verify_shard_hashes),
        "existing_ledger_hash_mismatch_count": len(source_hash_mismatches),
    }
    return audit, global_standardization, training_input_hashes


def audit_locked_state(config: Mapping[str, Any], source_records: Mapping[str, Any]) -> dict[str, Any]:
    dynamic = read_json(Path(source_records["dynamic_full51_manifest"]["path"]))
    readiness = read_json(Path(source_records["stage6o_v2_readiness"]["path"]))
    blocked_v1 = read_json(Path(source_records["stage6o_v1_blocked_freeze"]["path"]))
    confirmation = read_json(Path(source_records["stage6s_v2_confirmation_manifest"]["path"]))
    if dynamic.get("status") != "DYNAMIC_FULL51_FINALIZED_PENDING_STAGE6O_V2":
        raise ValueError("Dynamic full51 is not the frozen finalized dataset")
    if dynamic.get("embedding_or_checkpoint_read") is not False or dynamic.get("old_full51_overwritten") is not False:
        raise ValueError("Dynamic full51 provenance violates the Stage6T boundary")
    if readiness.get("status") != "FROZEN_READY_FOR_INTERACTION_AWARE_V2_PREPARATION":
        raise ValueError("Stage6O-v2 data readiness is not PASS")
    if not readiness.get("checks") or not all(readiness["checks"].values()):
        raise ValueError("One or more Stage6O-v2 data gates failed")
    if readiness.get("checkpoint_training_launched") is not False:
        raise ValueError("Stage6O-v2 reports unexpected checkpoint training")
    if blocked_v1.get("status") != "FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING":
        raise ValueError("Stage6O-v1 must remain permanently blocked")
    if blocked_v1.get("training_authorized") is not False or blocked_v1.get("checkpoint_write_authorized") is not False:
        raise ValueError("Stage6O-v1 authorization changed")
    if confirmation.get("status") != "CONFIRMATION_ROSTER_FROZEN_NOT_RUN":
        raise ValueError("Stage6S-v2 confirmation is no longer in the frozen-not-run state")
    for key in (
        "confirmation_rollouts_launched", "embedding_or_bdd_read", "checkpoint_training_launched",
        "new_model_evaluation_launched",
    ):
        if confirmation.get(key) is not False:
            raise ValueError(f"Stage6S-v2 confirmation.{key} must remain false")
    if confirmation.get("immutable_after_freeze") is not True:
        raise ValueError("Stage6S-v2 confirmation roster must remain immutable")
    roster_path = Path(source_records["stage6s_v2_confirmation_roster"]["path"])
    with roster_path.open(newline="", encoding="utf-8") as handle:
        roster_rows = list(csv.DictReader(handle))
    if len(roster_rows) != 80 or int(confirmation.get("scenario_count", -1)) != 80:
        raise ValueError("Stage6S-v2 confirmation roster must contain exactly 80 pairs")
    if confirmation.get("confirmation_roster_sha256") != source_records["stage6s_v2_confirmation_roster"]["sha256"]:
        raise ValueError("Stage6S-v2 internal roster SHA does not match the Stage6T source record")
    candidate_dirs = {candidate_id: resolve_repo_path(row["output_root"]) for candidate_id, row in config["candidates"].items()}
    nonempty_candidate_dirs = [str(path) for path in candidate_dirs.values() if path.exists() and any(path.iterdir())]
    if nonempty_candidate_dirs:
        raise ValueError(f"Stage6T candidate outputs already contain files before training authorization: {nonempty_candidate_dirs}")
    return {
        "dynamic_full51_status": dynamic["status"],
        "stage6o_v2_status": readiness["status"],
        "stage6o_v2_all_checks_pass": all(readiness["checks"].values()),
        "stage6o_v1_status": blocked_v1["status"],
        "stage6s_v2_status": confirmation["status"],
        "stage6s_v2_roster_rows": len(roster_rows),
        "stage6s_v2_development_log_overlap_count": confirmation["development_log_overlap_count"],
        "stage6s_v2_development_scenario_overlap_count": confirmation["development_scenario_overlap_count"],
        "candidate_output_nonempty_count": len(nonempty_candidate_dirs),
        "blind_state_intact": True,
    }


def environment_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
                "selected_device_under_protocol": "mps" if torch.backends.mps.is_available() else "cpu",
                "torch_import_pass": True,
            }
        )
    except ImportError as error:
        result.update({"torch_import_pass": False, "torch_import_error": str(error)})
    return result


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    config = manifest["frozen_protocol_summary"]
    audit = manifest["dynamic_dataset_audit"]
    state = manifest["locked_state_audit"]
    lines = [
        "# Stage 6T A/B/C训练与盲测协议冻结报告",
        "",
        f"## 结论：{manifest['status']}",
        "",
        "本阶段只完成训练前协议冻结与输入审计。未训练checkpoint，未读取Waymo test，未运行或读取nuPlan正式盲测，未运行Stage6S-v2 confirmation rollout。",
        "",
        "## A/B/C具体差异",
        "",
        "- Candidate A：Dynamic Builder v2数据 + 旧single-GRU 83D→64D架构 + 旧Stage5D objective。它估计数据修复的主导贡献；没有A0时不得声称是严格纯数据因果效应。",
        "- Candidate B：同一Dynamic v2数据 + 相同single-GRU拓扑 + clean longitudinal supervision、纵向ranking/采样与mask-aware dropout。它估计训练目标和采样的增量。",
        "- Candidate C：与B使用相同数据、采样、dropout、loss和随机数流，仅将encoder改为参数量匹配的ego16 + context48双分支。它估计架构拓扑的额外贡献。",
        "- 三者均导出64D，不训练ego-only最终模型；ego13只作外部参考上界。",
        "",
        "## 数据审计与新增fail-closed修正",
        "",
        f"- Dynamic v2：{audit['shard_count']} shards、{audit['row_count']} rows，train/val/test={audit['split_counts']}，scenario跨split重叠={audit['scenario_cross_split_overlap_count']}。",
        "- 发现六个part的33D `interaction_feat_style.npy`使用各part局部train统计，不能直接混合用于A/B/C监督。",
        "- Stage6T因此明确禁止读取该数组训练，统一从`interaction_feat_style_raw.npy`以全体Dynamic v2 train rows重算一次global mean/std；冻结shard不被改写。",
        f"- Stage6R原SHA ledger校验={audit['verified_existing_stage6r_ledger']}，hash mismatch={audit['existing_ledger_hash_mismatch_count']}；raw33 nonfinite={audit['raw_33d_nonfinite_count']}。",
        "",
        "## 如何保证不使用nuPlan选模型",
        "",
        "- 所有9个A/B/C×3 seed checkpoint只能用Waymo train优化、Waymo val早停与选epoch；primary seed预先固定为3407。",
        "- 9个checkpoint全部锁定后，才允许一次性读取同一Dynamic v2 Waymo test；test和nuPlan均禁止返工训练。",
        "- 之后依次运行既有Stage6J/K、Stage6P，最后先运行Stage6S-v2 trajectory mechanism gate；机制未通过时不得读取interaction embedding。",
        "- raw MMD²不能跨representation比较；C full-context相对C neighbor-zero使用各自null标准化Z差及log-cluster bootstrap。",
        "",
        "## Candidate C成功标准",
        "",
        "- Waymo：纵向指标显著改善，同时following/lateral/behavior/retrieval保持预冻结非劣性。",
        "- Stage6J/K：paired dose和task coverage门禁通过；Stage6P：n=400独立A/A标定后FPR、整体及双方向A/B检出率通过。",
        "- Stage6S-v2：mechanism gate先通过，C能检出差异，且C full-context相对C neighbor-zero的delta-Z bootstrap下界>0。",
        "- C不必击败ego13，也不因架构更复杂而自动优先；若B与C相当且C没有增量context证据，应优先更简单的B。",
        "",
        "## 当前授权边界",
        "",
        f"- Stage6O-v1：{state['stage6o_v1_status']}，保持不变。",
        f"- Stage6O-v2：{state['stage6o_v2_status']}。",
        f"- Stage6S-v2：{state['stage6s_v2_status']}，{state['stage6s_v2_roster_rows']} pair，盲态完整。",
        f"- training_authorized={manifest['training_authorized']}；checkpoint_write_authorized={manifest['checkpoint_write_authorized']}。",
        "- 当前只具备实现并审查统一A/B/C trainer的条件；尚未授权训练，更未达到Waymo test或nuPlan正式盲测阶段。",
        "",
        "## 冻结指纹",
        "",
        f"- protocol config SHA-256：`{manifest['protocol_config_sha256']}`",
        f"- protocol content fingerprint：`{manifest['protocol_content_fingerprint_sha256']}`",
        f"- encoder parameter counts：{config['calculated_parameter_counts']}；C/B={config['parameter_ratio_C_vs_B']:.6f}。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_freeze(
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    protocol_summary = validate_protocol_config(config)
    source_records = verify_source_records(config)
    locked_state = audit_locked_state(config, source_records)
    dynamic_manifest = read_json(Path(source_records["dynamic_full51_manifest"]["path"]))
    ledger = read_json(Path(source_records["dynamic_full51_sha256_ledger"]["path"]))
    dataset_audit, global_standardization, training_input_hashes = audit_dynamic_dataset(
        config, dynamic_manifest, ledger, verify_shard_hashes
    )
    part_standardization = []
    for root_text in dynamic_manifest["part_roots"]:
        path = Path(root_text) / "interaction_feature_standardization.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing part-local interaction standardization: {path}")
        stats = read_json(path)
        part_standardization.append(
            {"path": str(path), "sha256": sha256_file(path), "train_count": int(stats.get("train_count", -1))}
        )
    unique_part_standardization_hashes = len({row["sha256"] for row in part_standardization})
    if unique_part_standardization_hashes <= 1:
        raise ValueError("Expected six independently fitted part-local 33D standardizations; protocol diagnosis changed")

    env = environment_snapshot()
    if env.get("torch_import_pass") is not True:
        raise RuntimeError(f"Stage6T Waymo environment cannot import torch: {env.get('torch_import_error')}")

    output = output_dir.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace only this freeze directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _write_candidate_matrix(config, output / "stage6t_candidate_difference_matrix.csv")
    (output / "stage6t_global_interaction_target_standardization.json").write_text(
        json.dumps(global_standardization, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "stage6t_training_input_sha256.json").write_text(
        json.dumps(training_input_hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    content_fingerprint = sha256_json(
        {
            "protocol_config_sha256": sha256_file(config_path),
            "source_records": source_records,
            "dataset_content_signature_sha256": dynamic_manifest["content_signature_sha256"],
            "global_33d_standardization": global_standardization,
            "training_input_sha256": training_input_hashes,
        }
    )
    manifest = {
        "schema_version": "stage6t_training_evaluation_protocol_freeze_v1",
        "status": "FROZEN_READY_FOR_ABC_TRAINER_IMPLEMENTATION_NOT_TRAINING",
        "issue": int(config["issue"]),
        "protocol_id": config["protocol_id"],
        "protocol_config_path": str(config_path),
        "protocol_config_sha256": sha256_file(config_path),
        "protocol_content_fingerprint_sha256": content_fingerprint,
        "source_records": source_records,
        "frozen_protocol_summary": protocol_summary,
        "dynamic_dataset_audit": dataset_audit,
        "global_interaction_target_standardization": global_standardization,
        "part_local_interaction_standardization_diagnostic": {
            "unique_sha256_count": unique_part_standardization_hashes,
            "records": part_standardization,
            "stage6t_training_use_forbidden": True,
        },
        "locked_state_audit": locked_state,
        "environment_snapshot": env,
        "candidate_count": 3,
        "seed_count_per_candidate": 3,
        "planned_checkpoint_count": 9,
        "training_authorized": False,
        "checkpoint_write_authorized": False,
        "waymo_test_authorized": False,
        "nuplan_evaluation_authorized": False,
        "confirmation_rollout_authorized": False,
        "checkpoint_training_launched": False,
        "waymo_test_read": False,
        "nuplan_embedding_bdd_or_mmd_read": False,
        "confirmation_rollout_launched": False,
        "old64_overwritten_or_deleted": False,
        "stage6o_v1_modified": False,
        "stage6s_v2_roster_modified": False,
        "next_authorized_action": config["authorization"]["next_authorized_action"],
        "validation": {
            "config_contract_pass": True,
            "source_hashes_pass": True,
            "dynamic_dataset_pass": True,
            "stage6o_v2_pass": True,
            "stage6o_v1_immutable_blocked": True,
            "stage6s_v2_blind_freeze_pass": True,
            "candidate_output_absence_pass": True,
            "environment_import_pass": True,
            "pass": True,
        },
    }
    manifest_path = output / "stage6t_training_evaluation_protocol_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(output / "stage6t_training_evaluation_protocol_report_zh.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_full_shard_hash_audit",
        action="store_true",
        help="Development-only speed option. Do not use for the authoritative freeze.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_freeze(
        args.config,
        args.output_dir,
        overwrite=args.overwrite,
        verify_shard_hashes=not args.skip_full_shard_hash_audit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
