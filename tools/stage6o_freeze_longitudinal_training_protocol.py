#!/usr/bin/env python3
"""Freeze the Stage 6O longitudinal representation training protocol.

This command is intentionally read-only with respect to training data and model
checkpoints.  It audits the sharded Waymo dataset, verifies frozen evidence
hashes, and writes a pre-training manifest.  It never imports or invokes a
trainer and never writes a checkpoint.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


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
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def verify_file_record(record: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"{label} must contain path and sha256")
    path = resolve_repo_path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={path}")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def validate_protocol_config(config: Mapping[str, Any]) -> None:
    _require_keys(
        config,
        [
            "schema_version",
            "protocol_id",
            "issue",
            "authorization",
            "source_dataset",
            "baseline",
            "expected_dataset",
            "split_and_leakage",
            "coverage_audit",
            "sampling_protocol",
            "representation",
            "objectives",
            "optimization",
            "checkpoint_policy",
            "waymo_acceptance",
            "nuplan_acceptance",
            "replacement_policy",
            "forbidden_information",
        ],
        "config",
    )
    auth = config["authorization"]
    for key in (
        "training_authorized",
        "checkpoint_write_authorized",
        "baseline_overwrite_authorized",
        "nuplan_result_driven_tuning_authorized",
    ):
        if auth.get(key) is not False:
            raise ValueError(f"authorization.{key} must be false during Stage 6O freeze")

    representation = config["representation"]
    if int(representation.get("output_dim", -1)) != 64:
        raise ValueError("representation.output_dim must remain 64")
    subspace_sum = int(representation.get("ego_longitudinal_subspace_dim", -1)) + int(
        representation.get("context_fusion_subspace_dim", -1)
    )
    if subspace_sum != 64:
        raise ValueError(f"representation subspaces must sum to 64, got {subspace_sum}")
    if representation.get("neighbor_context_removed") is not False:
        raise ValueError("neighbor context must not be removed")
    if representation.get("planner_identity_as_input") is not False:
        raise ValueError("planner identity must not be a model input")

    sampling = config["sampling_protocol"]
    shares = [
        float(sampling["hard_negative_pairs"]["share_of_ranking_pairs"]),
        float(sampling["near_boundary_pairs"]["share_of_ranking_pairs"]),
        float(sampling["uniform_pairs"]["share_of_ranking_pairs"]),
    ]
    if not np.isclose(sum(shares), 1.0, atol=1e-12):
        raise ValueError(f"ranking-pair shares must sum to 1.0, got {sum(shares)}")
    if not sampling.get("forbidden_sampling_fields"):
        raise ValueError("sampling_protocol.forbidden_sampling_fields must be non-empty")

    objectives = config["objectives"]
    if objectives.get("raw_bdd_is_training_objective") is not False:
        raise ValueError("raw BDD must not be a training objective")
    if objectives.get("nuplan_metric_is_training_objective") is not False:
        raise ValueError("nuPlan metrics must not be training objectives")
    weights = objectives.get("weights", {})
    if not weights or any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("all frozen objective weights must be present and non-negative")

    optimization = config["optimization"]
    seeds = [int(seed) for seed in optimization.get("seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("optimization.seeds must contain exactly three unique seeds")
    if int(optimization.get("primary_seed", -1)) not in seeds:
        raise ValueError("optimization.primary_seed must be one of optimization.seeds")
    for key in ("max_epochs", "batch_size", "max_wallclock_hours_per_seed", "max_total_optimizer_steps_per_seed"):
        if float(optimization.get(key, 0)) <= 0:
            raise ValueError(f"optimization.{key} must be positive")

    checkpoint = config["checkpoint_policy"]
    if checkpoint.get("baseline_checkpoint_read_only") is not True:
        raise ValueError("checkpoint_policy.baseline_checkpoint_read_only must be true")
    if checkpoint.get("overwrite_existing_seed_directory") is not False:
        raise ValueError("checkpoint_policy.overwrite_existing_seed_directory must be false")
    if not checkpoint.get("required_checkpoint_metadata"):
        raise ValueError("checkpoint metadata contract is missing")

    waymo = config["waymo_acceptance"]
    if waymo.get("model_selection_split") != "val" or waymo.get("final_noninferiority_split") != "test":
        raise ValueError("Waymo model selection must use val and final non-inferiority must use test")
    if int(waymo.get("test_evaluation_runs_per_checkpoint", -1)) != 1:
        raise ValueError("Waymo test evaluation must be frozen to exactly one run per checkpoint")
    if not waymo.get("candidate_gates"):
        raise ValueError("Waymo candidate gates are missing")

    nuplan = config["nuplan_acceptance"]
    if nuplan.get("evaluation_only_after_waymo_candidate_is_locked") is not True:
        raise ValueError("nuPlan evaluation must occur only after the Waymo candidate is locked")
    _require_keys(nuplan, ["paired_dose_gates", "unpaired_release_gates_n400", "forbidden_gate"], "nuplan_acceptance")

    replacement = config["replacement_policy"]
    required_true = (
        "candidate_requires_all_integrity_gates",
        "candidate_requires_all_waymo_gates",
        "candidate_requires_all_nuplan_paired_gates",
        "candidate_requires_all_nuplan_unpaired_gates",
        "candidate_requires_seed_stability",
        "manual_review_required",
    )
    if any(replacement.get(key) is not True for key in required_true):
        raise ValueError("replacement policy must require every integrity/evidence gate and manual review")
    if replacement.get("baseline_deletion_allowed") is not False:
        raise ValueError("baseline deletion must remain forbidden")
    if replacement.get("post_result_threshold_change_allowed") is not False:
        raise ValueError("post-result threshold changes must remain forbidden")


def _path_suffix_after_anchor(path: Path, anchor: str) -> Path:
    parts = path.parts
    indices = [index for index, part in enumerate(parts) if part == anchor]
    if not indices:
        raise ValueError(f"manifest shard path has no anchor '{anchor}': {path}")
    return Path(*parts[indices[-1] + 1 :])


def resolve_shard_path(entry_path: str, dataset_config: Mapping[str, Any]) -> Tuple[Path, str]:
    direct = Path(entry_path).expanduser()
    if direct.is_dir():
        return direct.resolve(), "manifest_direct"
    anchor = str(dataset_config["manifest_path_anchor"])
    suffix = _path_suffix_after_anchor(direct, anchor)
    tried: List[str] = [str(direct)]
    for root_text in dataset_config.get("allowed_shard_roots", []):
        root = (REPO_ROOT / str(root_text)).resolve()
        candidate = root / suffix
        tried.append(str(candidate))
        if candidate.is_dir():
            return candidate.resolve(), f"explicit_allowed_root:{root_text}"
    raise FileNotFoundError(f"Unable to resolve shard path {entry_path}; tried={tried}")


def _as_split_strings(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind in {"U", "S", "O"}:
        return array.astype(str)
    mapping = {0: "train", 1: "val", 2: "test"}
    return np.asarray([mapping.get(int(value), str(int(value))) for value in array], dtype=object)


def split_from_scenario_id(scenario_id: str) -> str:
    value = int(hashlib.md5(str(scenario_id).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if value < 0.8 else ("val" if value < 0.9 else "test")


def _bin_labels(values: np.ndarray, edges: Sequence[float], names: Sequence[str]) -> np.ndarray:
    if len(edges) != len(names) + 1:
        raise ValueError("bin edges must have exactly one more item than bin names")
    indices = np.digitize(values, np.asarray(edges[1:-1], dtype=np.float64), right=False)
    return np.asarray([names[int(index)] for index in indices], dtype=object)


def _counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _audit_build_quality(build_summary: Mapping[str, Any], expected: Mapping[str, Any]) -> Dict[str, Any]:
    gates = expected["quality_gates"]
    slot_ratios = build_summary.get("slot_occupied_window_ratio", {})
    checks = {
        "total_shards_match": int(build_summary.get("total_shards", build_summary.get("n_shards", -1)))
        == int(expected["total_shards"]),
        "total_windows_match": int(build_summary.get("total_windows", build_summary.get("n_windows_kept", -1)))
        == int(expected["total_windows"]),
        "good_lane_context_rate": float(build_summary.get("good_lane_context_rate", -1.0))
        >= float(gates["min_good_lane_context_rate"]),
        "lane_assignment_success_rate": float(build_summary.get("lane_assignment_success_rate", -1.0))
        >= float(gates["min_lane_assignment_success_rate"]),
        "fallback_assignment_rate": float(build_summary.get("fallback_assignment_rate", 1.0))
        <= float(gates["max_fallback_assignment_rate"]),
        "front_slot_coverage": float(slot_ratios.get("front", -1.0))
        >= float(gates["min_front_slot_occupied_window_ratio"]),
        "left_front_slot_coverage": float(slot_ratios.get("left_front", -1.0))
        >= float(gates["min_side_slot_occupied_window_ratio"]),
        "left_rear_slot_coverage": float(slot_ratios.get("left_rear", -1.0))
        >= float(gates["min_side_slot_occupied_window_ratio"]),
        "right_front_slot_coverage": float(slot_ratios.get("right_front", -1.0))
        >= float(gates["min_side_slot_occupied_window_ratio"]),
        "right_rear_slot_coverage": float(slot_ratios.get("right_rear", -1.0))
        >= float(gates["min_side_slot_occupied_window_ratio"]),
        "nonfinite_output_detected": int(build_summary.get("nonfinite_output_detected", -1))
        <= int(gates["max_nonfinite_output_detected"]),
    }
    return {
        "observed": {
            "good_lane_context_rate": build_summary.get("good_lane_context_rate"),
            "lane_assignment_success_rate": build_summary.get("lane_assignment_success_rate"),
            "fallback_assignment_rate": build_summary.get("fallback_assignment_rate"),
            "slot_occupied_window_ratio": slot_ratios,
            "nonfinite_output_detected": build_summary.get("nonfinite_output_detected"),
            "warning_count": len(build_summary.get("warnings", [])),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_shards(config: Mapping[str, Any], manifest_path: Path) -> Dict[str, Any]:
    dataset_config = config["source_dataset"]
    expected = config["expected_dataset"]
    coverage = config["coverage_audit"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("shards") or manifest.get("shard_infos") or manifest.get("shard_paths")
    if not entries:
        raise ValueError(f"No shard entries in manifest: {manifest_path}")
    if len(entries) != int(expected["total_shards"]):
        raise ValueError(f"Shard count mismatch: expected={expected['total_shards']}, observed={len(entries)}")

    required = list(expected["required_files_per_shard"])
    split_counts: Counter = Counter()
    scenario_split: Dict[str, str] = {}
    scenario_agent_split: Dict[Tuple[str, str], str] = {}
    duplicate_window_keys = 0
    window_keys = set()
    resolution_counts: Counter = Counter()
    speed_counts: Dict[str, Counter] = defaultdict(Counter)
    front_counts: Dict[str, Counter] = defaultdict(Counter)
    joint_counts: Dict[str, Counter] = defaultdict(Counter)
    lateral_counts: Dict[str, Counter] = defaultdict(Counter)
    motion_counts: Dict[str, Counter] = defaultdict(Counter)
    train_targets: MutableMapping[str, List[np.ndarray]] = defaultdict(list)
    shard_records: List[Dict[str, Any]] = []
    total_rows = 0

    speed_names = coverage["speed_bin_names"]
    front_names = coverage["front_regime_names"]
    speed_edges = coverage["speed_bins_mps"]
    front_edges = coverage["front_valid_ratio_bins"]
    raw_indices = coverage["raw_feature_indices"]
    speed_channel = int(coverage["ego_channel_indices"]["speed"])

    for shard_index, entry in enumerate(tqdm(entries, desc="Stage6O Waymo shard audit", unit="shard")):
        entry_path = entry["shard_path"] if isinstance(entry, Mapping) else entry
        shard_dir, resolution = resolve_shard_path(str(entry_path), dataset_config)
        resolution_counts[resolution] += 1
        missing = [name for name in required if not (shard_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Shard {shard_dir} missing required files: {missing}")

        context = np.load(shard_dir / "context_traj.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(shard_dir / "context_mask.npy", mmap_mode="r", allow_pickle=False)
        feat = np.load(shard_dir / "interaction_feat_style.npy", mmap_mode="r", allow_pickle=False)
        feat_raw = np.load(shard_dir / "interaction_feat_style_raw.npy", mmap_mode="r", allow_pickle=False)
        split = _as_split_strings(np.load(shard_dir / "split.npy", allow_pickle=True))
        meta = np.load(shard_dir / "meta.npy", allow_pickle=True)
        n_rows = int(context.shape[0])
        expected_shapes = {
            "context_traj.npy": (n_rows, int(expected["sequence_length"]), int(expected["context_dim"])),
            "context_mask.npy": (n_rows, int(expected["sequence_length"]), int(expected["context_mask_dim"])),
            "interaction_feat_style.npy": (n_rows, int(expected["feature_dim"])),
            "interaction_feat_style_raw.npy": (n_rows, int(expected["feature_dim"])),
            "split.npy": (n_rows,),
            "meta.npy": (n_rows,),
        }
        arrays = {
            "context_traj.npy": context,
            "context_mask.npy": mask,
            "interaction_feat_style.npy": feat,
            "interaction_feat_style_raw.npy": feat_raw,
            "split.npy": split,
            "meta.npy": meta,
        }
        shape_mismatches = {
            name: {"expected": list(expected_shapes[name]), "observed": list(array.shape)}
            for name, array in arrays.items()
            if tuple(array.shape) != expected_shapes[name]
        }
        if shape_mismatches:
            raise ValueError(f"Shard shape mismatch at {shard_dir}: {shape_mismatches}")
        finite_checks = {
            "context_traj": bool(np.isfinite(context).all()),
            "context_mask": bool(np.isfinite(mask).all()),
            "interaction_feat_style": bool(np.isfinite(feat).all()),
            "interaction_feat_style_raw": bool(np.isfinite(feat_raw).all()),
        }
        if expected["quality_gates"]["require_all_arrays_finite"] and not all(finite_checks.values()):
            raise ValueError(f"Non-finite values found in shard {shard_dir}: {finite_checks}")
        unknown_splits = sorted(set(split.tolist()) - set(SPLITS))
        if unknown_splits:
            raise ValueError(f"Unknown split labels in {shard_dir}: {unknown_splits}")
        if not meta.dtype.names or not {"scenario_id", "target_agent_id", "start", "split"}.issubset(meta.dtype.names):
            raise ValueError(f"meta.npy missing leakage-audit fields at {shard_dir}: {meta.dtype.names}")
        meta_splits = np.asarray(meta["split"], dtype=str)
        if config["split_and_leakage"]["require_meta_split_matches_split_array"] and not np.array_equal(
            meta_splits, split
        ):
            raise ValueError(f"meta split differs from split.npy in {shard_dir}")

        for split_name, count in zip(*np.unique(split, return_counts=True)):
            split_counts[str(split_name)] += int(count)
        for row in meta:
            scenario = str(row["scenario_id"])
            agent = str(row["target_agent_id"])
            row_split = str(row["split"])
            expected_split = split_from_scenario_id(scenario)
            if row_split != expected_split:
                raise ValueError(
                    f"Frozen split algorithm mismatch: scenario_id={scenario}, expected={expected_split}, "
                    f"observed={row_split}"
                )
            previous = scenario_split.setdefault(scenario, row_split)
            if previous != row_split:
                raise ValueError(f"Scenario leakage: scenario_id={scenario} appears in {previous} and {row_split}")
            previous_agent = scenario_agent_split.setdefault((scenario, agent), row_split)
            if previous_agent != row_split:
                raise ValueError(
                    f"Scenario-agent leakage: scenario_id={scenario}, target_agent_id={agent} appears in "
                    f"{previous_agent} and {row_split}"
                )
            key = (scenario, agent, int(row["start"]))
            if key in window_keys:
                duplicate_window_keys += 1
            else:
                window_keys.add(key)

        speed_mean = np.asarray(context[:, :, speed_channel], dtype=np.float64).mean(axis=1)
        speed_min = np.asarray(context[:, :, speed_channel], dtype=np.float64).min(axis=1)
        speed_max = np.asarray(context[:, :, speed_channel], dtype=np.float64).max(axis=1)
        speed_std = np.asarray(context[:, :, speed_channel], dtype=np.float64).std(axis=1)
        front_valid_ratio = np.asarray(mask[:, :, 0], dtype=np.float64).mean(axis=1)
        speed_labels = _bin_labels(speed_mean, speed_edges, speed_names)
        front_labels = _bin_labels(front_valid_ratio, front_edges, front_names)
        lateral_proxy = np.asarray(feat_raw[:, int(raw_indices["lane_change_count_proxy"])], dtype=np.float64)
        lateral_labels = np.where(lateral_proxy > 0.0, "lateral_motion", "no_lateral_motion")
        for row_split in SPLITS:
            selected = split == row_split
            speed_counts[row_split].update(speed_labels[selected].tolist())
            front_counts[row_split].update(front_labels[selected].tolist())
            lateral_counts[row_split].update(lateral_labels[selected].tolist())
            motion_counts[row_split]["stop_go"] += int(
                np.sum(selected & (speed_min < 1.0) & (speed_max >= 5.0))
            )
            motion_counts[row_split]["low_speed_variable"] += int(
                np.sum(selected & (speed_mean < 5.0) & ((speed_max - speed_min) >= 2.0))
            )
            motion_counts[row_split]["steady_speed"] += int(np.sum(selected & (speed_std <= 0.5)))
            motion_counts[row_split]["dynamic_speed"] += int(np.sum(selected & (speed_std > 0.5)))
            joint_counts[row_split].update(
                f"{speed_name}|{front_name}|{lateral_name}"
                for speed_name, front_name, lateral_name in zip(
                    speed_labels[selected], front_labels[selected], lateral_labels[selected]
                )
            )
        train_mask = split == "train"
        train_targets["mean_speed"].append(speed_mean[train_mask])
        for feature_name in ("rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk", "mean_thw", "min_thw"):
            train_targets[feature_name].append(
                np.asarray(feat_raw[train_mask, int(raw_indices[feature_name])], dtype=np.float64)
            )

        file_records = []
        for name in required:
            file_path = shard_dir / name
            file_records.append(
                {
                    "name": name,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),
                }
            )
        shard_records.append(
            {
                "index": shard_index,
                "manifest_path": str(entry_path),
                "resolved_path": str(shard_dir),
                "resolution": resolution,
                "n_rows": n_rows,
                "split_counts": _counter_to_dict(Counter(split.tolist())),
                "finite_checks": finite_checks,
                "files": file_records,
            }
        )
        total_rows += n_rows

    expected_split_counts = {key: int(value) for key, value in expected["split_counts"].items()}
    observed_split_counts = {key: int(split_counts.get(key, 0)) for key in SPLITS}
    if total_rows != int(expected["total_windows"]):
        raise ValueError(f"Total row count mismatch: expected={expected['total_windows']}, observed={total_rows}")
    if observed_split_counts != expected_split_counts:
        raise ValueError(f"Split counts mismatch: expected={expected_split_counts}, observed={observed_split_counts}")
    ratios = {key: observed_split_counts[key] / total_rows for key in SPLITS}
    tolerance = float(config["split_and_leakage"]["ratio_absolute_tolerance"])
    expected_ratios = config["split_and_leakage"]["expected_ratios"]
    ratio_checks = {key: abs(ratios[key] - float(expected_ratios[key])) <= tolerance for key in SPLITS}
    if not all(ratio_checks.values()):
        raise ValueError(f"Split ratios exceed tolerance: ratios={ratios}, checks={ratio_checks}")

    train_quantiles = [float(value) for value in coverage["train_quantiles_to_freeze"]]
    quantile_records = {}
    for target, chunks in train_targets.items():
        values = np.concatenate(chunks)
        quantile_records[target] = {
            f"q{int(round(q * 100)):02d}": float(np.quantile(values, q)) for q in train_quantiles
        }
    min_speed = int(coverage["min_train_windows_per_speed_bin"])
    min_front = int(coverage["min_train_windows_per_front_regime"])
    min_joint = int(coverage["min_train_windows_per_nonempty_speed_front_cell"])
    speed_gate = {name: int(speed_counts["train"].get(name, 0)) >= min_speed for name in speed_names}
    front_gate = {name: int(front_counts["train"].get(name, 0)) >= min_front for name in front_names}
    speed_front_counts = Counter()
    for key, value in joint_counts["train"].items():
        speed_name, front_name, _ = key.split("|")
        speed_front_counts[f"{speed_name}|{front_name}"] += value
    nonempty_joint_gate = {
        key: int(value) >= min_joint for key, value in speed_front_counts.items() if int(value) > 0
    }
    motion_gate = {
        "stop_go": int(motion_counts["train"].get("stop_go", 0))
        >= int(coverage["min_train_stop_go_windows"]),
        "steady_speed": int(motion_counts["train"].get("steady_speed", 0))
        >= int(coverage["min_train_steady_speed_windows"]),
    }
    coverage_pass = (
        all(speed_gate.values())
        and all(front_gate.values())
        and all(nonempty_joint_gate.values())
        and all(motion_gate.values())
    )
    fingerprint_inputs = [
        {"index": record["index"], "files": record["files"], "n_rows": record["n_rows"]}
        for record in shard_records
    ]
    return {
        "total_shards": len(shard_records),
        "total_windows": total_rows,
        "split_counts": observed_split_counts,
        "split_ratios": ratios,
        "split_ratio_checks": ratio_checks,
        "unique_scenarios": len(scenario_split),
        "unique_scenario_agents": len(scenario_agent_split),
        "scenario_cross_split_overlap_count": 0,
        "scenario_agent_cross_split_overlap_count": 0,
        "duplicate_scenario_agent_start_count": duplicate_window_keys,
        "shard_resolution_counts": _counter_to_dict(resolution_counts),
        "coverage": {
            "speed_bin_counts_by_split": {key: _counter_to_dict(value) for key, value in speed_counts.items()},
            "front_regime_counts_by_split": {key: _counter_to_dict(value) for key, value in front_counts.items()},
            "lateral_nuisance_counts_by_split": {key: _counter_to_dict(value) for key, value in lateral_counts.items()},
            "motion_regime_counts_by_split": {key: _counter_to_dict(value) for key, value in motion_counts.items()},
            "joint_stratum_counts_by_split": {key: _counter_to_dict(value) for key, value in joint_counts.items()},
            "train_speed_front_counts": _counter_to_dict(speed_front_counts),
            "train_quantiles": quantile_records,
            "speed_gate": speed_gate,
            "front_gate": front_gate,
            "nonempty_speed_front_gate": nonempty_joint_gate,
            "motion_gate": motion_gate,
            "pass": coverage_pass,
        },
        "shards": shard_records,
        "dataset_fingerprint_sha256": sha256_json(fingerprint_inputs),
        "pass": True,
    }


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    audit = manifest["waymo_data_audit"]
    quality = manifest["build_quality_audit"]
    coverage = audit["coverage"]
    lines = [
        "# Stage 6O 纵向表征训练协议冻结报告",
        "",
        "## 冻结结论",
        "",
        f"- 状态：`{manifest['status']}`",
        "- 本步骤只完成数据与训练协议冻结；没有启动训练，也没有写入或覆盖 checkpoint。",
        f"- 配置 SHA-256：`{manifest['config_sha256']}`",
        f"- 数据指纹 SHA-256：`{audit['dataset_fingerprint_sha256']}`",
        "",
        "## Waymo 数据规模与质量",
        "",
        f"- shards：{audit['total_shards']}",
        f"- windows：{audit['total_windows']}",
        f"- train/val/test：{audit['split_counts']}",
        f"- unique scenarios：{audit['unique_scenarios']}",
        f"- unique scenario-agent：{audit['unique_scenario_agents']}",
        f"- scenario 跨 split 重叠：{audit['scenario_cross_split_overlap_count']}",
        f"- scenario-agent 跨 split 重叠：{audit['scenario_agent_cross_split_overlap_count']}",
        f"- build quality gate：{quality['pass']}",
        f"- coverage gate：{coverage['pass']}",
        f"- train speed bins：{coverage['speed_bin_counts_by_split'].get('train', {})}",
        f"- train front regimes：{coverage['front_regime_counts_by_split'].get('train', {})}",
        "",
        "## 冻结模型方向",
        "",
        "- 输入仍为 Stage5D 83D context，输出接口仍为 64D。",
        "- 64D 固定为 ego-longitudinal 16D 与 context/fusion 48D 拼接。",
        "- 邻车上下文保留；使用显式 mask 和预冻结 context dropout，ego 通道不做 dropout。",
        "- 训练目标包含纵向 auxiliary、metric alignment、ranking 及 context consistency。",
        "- raw BDD、nuPlan BDD/MMD、planner 名称和 dose 标签都不得进入训练或选 epoch。",
        "",
        "## 替换边界",
        "",
        "- Waymo test 只允许在候选锁定后评估一次。",
        "- nuPlan 只用于候选锁定后的外域验收。",
        "- Waymo 非劣性、nuPlan paired dose、task coverage 和 n=400 unpaired release gate 必须全部通过。",
        "- 任一门槛失败时保留 Stage5D-balanced-v2；部分通过只能作为研究消融。",
        "- 基线 checkpoint 永不删除，任何晋级都需要人工复核。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_freeze(config_path: Path, out_dir: Path, overwrite: bool = False) -> Dict[str, Any]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage6O config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_protocol_config(config)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists and is non-empty; pass --overwrite")
        shutil.rmtree(out_dir)

    source_records = {
        "manifest": verify_file_record(config["source_dataset"]["manifest"], "source_dataset.manifest"),
        "build_summary": verify_file_record(
            config["source_dataset"]["build_summary"], "source_dataset.build_summary"
        ),
        "feature_schema": verify_file_record(
            config["source_dataset"]["feature_schema"], "source_dataset.feature_schema"
        ),
        "standardization": verify_file_record(
            config["source_dataset"]["standardization"], "source_dataset.standardization"
        ),
        "standardization_report": verify_file_record(
            config["source_dataset"]["standardization_report"], "source_dataset.standardization_report"
        ),
        "baseline_checkpoint": verify_file_record(config["baseline"]["checkpoint"], "baseline.checkpoint"),
        "baseline_evaluation_summary": verify_file_record(
            config["baseline"]["evaluation_summary"], "baseline.evaluation_summary"
        ),
        "baseline_category_correlation": verify_file_record(
            config["baseline"]["category_correlation"], "baseline.category_correlation"
        ),
        "baseline_retrieval_metrics": verify_file_record(
            config["baseline"]["retrieval_metrics"], "baseline.retrieval_metrics"
        ),
    }
    for label, record in config["nuplan_acceptance"]["authoritative_evidence"].items():
        source_records[label] = verify_file_record(record, f"nuplan_acceptance.authoritative_evidence.{label}")

    standardization_report = json.loads(
        Path(source_records["standardization_report"]["path"]).read_text(encoding="utf-8")
    )
    if int(standardization_report.get("train_count", -1)) != int(config["expected_dataset"]["split_counts"]["train"]):
        raise ValueError("Standardization train_count does not match the frozen train split count")
    if int(standardization_report.get("feature_dim", -1)) != int(config["expected_dataset"]["feature_dim"]):
        raise ValueError("Standardization feature_dim does not match the frozen feature_dim")

    build_summary = json.loads(Path(source_records["build_summary"]["path"]).read_text(encoding="utf-8"))
    build_quality = _audit_build_quality(build_summary, config["expected_dataset"])
    if not build_quality["pass"]:
        raise ValueError(f"Waymo build quality gate failed: {build_quality['checks']}")
    audit = audit_shards(config, Path(source_records["manifest"]["path"]))

    coverage_pass = bool(audit["coverage"]["pass"])
    validation_pass = bool(build_quality["pass"] and audit["pass"] and coverage_pass)
    manifest = {
        "schema_version": "stage6o_longitudinal_training_protocol_freeze_manifest_v1",
        "protocol_id": config["protocol_id"],
        "issue": int(config["issue"]),
        "status": (
            "FROZEN_READY_FOR_IMPLEMENTATION_NOT_TRAINING"
            if coverage_pass
            else "FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING"
        ),
        "training_authorized": False,
        "checkpoint_write_authorized": False,
        "baseline_overwrite_authorized": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_records": source_records,
        "build_quality_audit": build_quality,
        "waymo_data_audit": audit,
        "frozen_protocol": config,
        "validation": {
            "config_contract_pass": True,
            "source_hashes_pass": True,
            "build_quality_pass": build_quality["pass"],
            "shard_data_audit_pass": audit["pass"],
            "split_leakage_pass": audit["scenario_cross_split_overlap_count"] == 0
            and audit["scenario_agent_cross_split_overlap_count"] == 0,
            "coverage_pass": audit["coverage"]["pass"],
            "pass": validation_pass,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage6o_waymo_data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "stage6o_training_protocol_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(out_dir / "stage6o_training_protocol_report_zh.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Waymo shards and freeze the Stage6O training protocol without starting training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_freeze(args.config, args.out_dir, overwrite=args.overwrite)
    print(json.dumps({
        "status": manifest["status"],
        "training_authorized": manifest["training_authorized"],
        "validation_pass": manifest["validation"]["pass"],
        "out_dir": str(args.out_dir.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
