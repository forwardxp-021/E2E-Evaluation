#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_scenario_conditioned_bdd import (
    biased_mmd2_from_kernel,
    exact_median_bandwidth,
    holm_adjust,
    markdown_table,
    permutation_bdd,
    rbf_kernel,
    sha256_file,
    validate_and_build_pairs,
)


PRETREATMENT_TASKS: Mapping[str, Tuple[str, ...]] = {
    "following_interaction": (
        "following_lane_with_lead",
        "following_lane_with_slow_lead",
        "near_long_vehicle",
    ),
    "lane_change": (
        "changing_lane_to_left",
        "changing_lane_to_right",
    ),
    "stop_go_control": (
        "accelerating_at_traffic_light_with_lead",
        "stationary_at_traffic_light_without_lead",
        "stationary_in_traffic",
        "stopping_at_traffic_light_with_lead",
        "stopping_with_lead",
    ),
    "high_motion_dynamics": (
        "high_lateral_acceleration",
        "high_magnitude_speed",
        "medium_magnitude_speed",
    ),
    "dense_or_vulnerable_interaction": (
        "near_multiple_vehicles",
        "near_pedestrian_on_crosswalk",
    ),
}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_csv_records(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def planner_fingerprints(
    metadata: pd.DataFrame, planners: Sequence[str]
) -> Dict[str, str]:
    required = {"planner_name", "parameters_json"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing planner fingerprint columns: {missing}")
    result: Dict[str, str] = {}
    for planner in planners:
        rows = metadata.loc[metadata["planner_name"].astype(str) == planner]
        if rows.empty:
            raise ValueError(f"metadata contains no rows for planner {planner}")
        canonical_values = set()
        for raw in rows["parameters_json"].astype(str):
            try:
                canonical = json.dumps(
                    json.loads(raw), sort_keys=True, separators=(",", ":")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"planner {planner} has invalid parameters_json"
                ) from exc
            canonical_values.add(canonical)
        if len(canonical_values) != 1:
            raise ValueError(
                f"planner {planner} has multiple parameter configurations"
            )
        result[planner] = hashlib.sha256(
            next(iter(canonical_values)).encode("utf-8")
        ).hexdigest()
    return result


def build_pretreatment_task_masks(
    metadata: pd.DataFrame,
    pair_indices: np.ndarray,
    task_definitions: Mapping[str, Sequence[str]] = PRETREATMENT_TASKS,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    if "scenario_type" not in metadata:
        raise ValueError("metadata missing pre-treatment scenario_type")
    by_row = metadata.set_index("global_row", drop=False)
    pair_types: List[str] = []
    pair_tokens: List[str] = []
    for pair_position, (row_a, row_b) in enumerate(pair_indices):
        type_a = str(by_row.loc[int(row_a), "scenario_type"])
        type_b = str(by_row.loc[int(row_b), "scenario_type"])
        if type_a != type_b:
            raise ValueError(
                f"pair {pair_position} has unequal pre-treatment scenario_type: "
                f"A={type_a}, B={type_b}"
            )
        pair_types.append(type_a)
        pair_tokens.append(str(by_row.loc[int(row_a), "scenario_token"]))

    masks: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    covered = np.zeros(len(pair_indices), dtype=bool)
    pair_types_array = np.asarray(pair_types, dtype=object)
    for task, scenario_types in task_definitions.items():
        mask = np.isin(pair_types_array, np.asarray(scenario_types, dtype=object))
        masks[task] = mask
        covered |= mask
        rows.append(
            {
                "task": task,
                "selection_timing": "pre_treatment",
                "source_field": "metadata.scenario_type",
                "scenario_types": "|".join(scenario_types),
                "n_pairs": int(mask.sum()),
            }
        )
    rows.append(
        {
            "task": "unmapped_scenario_type",
            "selection_timing": "pre_treatment",
            "source_field": "metadata.scenario_type",
            "scenario_types": "|".join(sorted(set(pair_types_array[~covered]))),
            "n_pairs": int((~covered).sum()),
        }
    )
    return masks, pd.DataFrame(rows)


def paired_randomization_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    monte_carlo_repetitions: int,
    seed: int,
    progress_label: str,
    exact_max_pairs: int = 20,
) -> Tuple[Dict[str, Any], np.ndarray]:
    if values_a.shape != values_b.shape:
        raise ValueError("paired randomization requires equal A/B shapes")
    n_pairs = len(values_a)
    if n_pairs < 2:
        raise ValueError("paired randomization requires at least two pairs")
    if n_pairs > exact_max_pairs:
        result, samples = permutation_bdd(
            values_a,
            values_b,
            repetitions=monte_carlo_repetitions,
            seed=seed,
            paired_swap=True,
            progress_label=progress_label,
        )
        result["randomization_mode"] = "monte_carlo"
        result["unique_label_assignments"] = int(2**n_pairs)
        return result, samples

    pooled = np.vstack([values_a, values_b]).astype(np.float64, copy=False)
    bandwidth = exact_median_bandwidth(pooled)
    kernel = rbf_kernel(pooled, bandwidth)
    index_a = np.arange(n_pairs, dtype=np.int64)
    index_b = np.arange(n_pairs, 2 * n_pairs, dtype=np.int64)
    observed = biased_mmd2_from_kernel(kernel, index_a, index_b)
    assignment_count = 2**n_pairs
    samples = np.empty(assignment_count, dtype=np.float64)
    for position, bits in enumerate(itertools.product((False, True), repeat=n_pairs)):
        swap = np.asarray(bits, dtype=bool)
        candidate_a = np.where(swap, index_b, index_a)
        candidate_b = np.where(swap, index_a, index_b)
        samples[position] = biased_mmd2_from_kernel(
            kernel, candidate_a, candidate_b
        )
    exceedance_count = int(np.sum(samples >= observed - 1e-15))
    p_value = float(exceedance_count / assignment_count)
    return {
        "metric": "BDD_MMD",
        "mmd_estimator": "biased_single_rbf_fixed_pooled_median_bandwidth",
        "kernel_type": "single_rbf",
        "mmd2": observed,
        "bandwidth": bandwidth,
        "bandwidth_selection": (
            "exact median of all finite positive off-diagonal pooled Euclidean distances"
        ),
        "n_A": n_pairs,
        "n_B": n_pairs,
        "permutation_scheme": "within_scenario_pair_label_swap",
        "randomization_mode": "exact_enumeration",
        "unique_label_assignments": assignment_count,
        "exceedance_count": exceedance_count,
        "p_value": p_value,
        "exact_randomization_resolution": float(1.0 / assignment_count),
        "null_median": float(np.median(samples)),
        "null_q95": float(np.quantile(samples, 0.95)),
        "null_q99": float(np.quantile(samples, 0.99)),
    }, samples


def robust_pooled_transform(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(source, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("representation has an all-missing/non-finite column")
    filled = np.where(np.isfinite(source), source, median)
    q25, q75 = np.percentile(filled, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.std(filled, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return (filled - median) / scale


def audit_locked_disjointness(
    development: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    planners: Sequence[str],
) -> Dict[str, Any]:
    required = {"scenario_token", "log_name", "planner_name", "parameters_json"}
    for name, frame in (("development", development), ("candidate", candidate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} metadata missing disjointness columns: {missing}")
    development_tokens = set(development["scenario_token"].astype(str))
    candidate_tokens = set(candidate["scenario_token"].astype(str))
    development_logs = set(development["log_name"].astype(str))
    candidate_logs = set(candidate["log_name"].astype(str))
    dev_fingerprints = planner_fingerprints(development, planners)
    candidate_fingerprints = planner_fingerprints(candidate, planners)
    token_overlap = sorted(development_tokens & candidate_tokens)
    log_overlap = sorted(development_logs & candidate_logs)
    result = {
        "scenario_token_overlap_count": len(token_overlap),
        "scenario_token_overlap_examples": token_overlap[:10],
        "log_overlap_count": len(log_overlap),
        "log_overlap_examples": log_overlap[:10],
        "planner_fingerprints_development": dev_fingerprints,
        "planner_fingerprints_candidate": candidate_fingerprints,
        "planner_parameters_identical_to_frozen_treatments": (
            dev_fingerprints == candidate_fingerprints
        ),
    }
    result["passed"] = bool(
        not token_overlap
        and not log_overlap
        and dev_fingerprints == candidate_fingerprints
    )
    return result


def validate_frozen_power_justification(
    payload: Mapping[str, Any],
    *,
    lock_manifest_path: Path,
    pair_count: int,
    task_masks: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    if payload.get("status") != "FROZEN_BEFORE_LOCKED_CONFIRMATION":
        raise ValueError("power justification is not frozen before confirmation")
    expected_lock_hash = sha256_file(lock_manifest_path)
    if payload.get("m6_2_lock_spec_sha256") != expected_lock_hash:
        raise ValueError("power justification references a different M6.2 lock spec")
    required_overall = int(payload["required_complete_pairs_overall"])
    required_by_task = payload["required_complete_pairs_by_task"]
    if set(required_by_task) != set(task_masks):
        raise ValueError("power justification task family differs from frozen task masks")
    observed_by_task = {
        task: int(np.asarray(mask, dtype=bool).sum())
        for task, mask in task_masks.items()
    }
    insufficient = {
        task: {
            "observed": observed_by_task[task],
            "required": int(required_by_task[task]),
        }
        for task in task_masks
        if observed_by_task[task] < int(required_by_task[task])
    }
    result = {
        "required_complete_pairs_overall": required_overall,
        "observed_complete_pairs_overall": int(pair_count),
        "required_complete_pairs_by_task": {
            task: int(value) for task, value in required_by_task.items()
        },
        "observed_complete_pairs_by_task": observed_by_task,
        "insufficient_tasks": insufficient,
        "passed": bool(pair_count >= required_overall and not insufficient),
    }
    if not result["passed"]:
        raise ValueError(f"locked power/sample targets not met: {result}")
    return result


def build_lock_spec(
    *,
    frozen_spec_path: Path,
    development_metadata_path: Path,
    planners: Sequence[str],
    development_metadata: pd.DataFrame,
    tool_path: Path,
    minimum_overall_pairs: int,
    minimum_task_pairs: int,
) -> Dict[str, Any]:
    frozen_spec = read_json(frozen_spec_path)
    primary = frozen_spec.get("primary_analysis", {})
    if primary.get("permutations") != 100000:
        raise ValueError("M6.1 frozen primary must use 100000 permutations")
    return {
        "protocol_id": "stage7_m6_2_locked_confirmation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_NEW_CONFIRMATION_DATA",
        "development_dataset_role": "METHOD_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "locked_dataset_role": "NEW_LOG_AND_SCENARIO_DISJOINT_CONFIRMATION",
        "frozen_m6_1_spec_sha256": sha256_file(frozen_spec_path),
        "development_metadata_sha256": sha256_file(development_metadata_path),
        "analysis_tool_sha256": sha256_file(tool_path),
        "planner_parameter_fingerprints": planner_fingerprints(
            development_metadata, planners
        ),
        "primary_analysis": primary,
        "task_conditioned_secondary": {
            "selection_timing": "pre_treatment",
            "source": "nuPlan scenario_type before either planner rollout",
            "task_definitions": {
                key: list(value) for key, value in PRETREATMENT_TASKS.items()
            },
            "within_task_null": "within_scenario_pair_label_swap",
            "small_subset_rule": "exact enumeration when n_pairs <= 20",
            "large_subset_rule": "100000 Monte Carlo swaps with plus-one p-value",
            "multiplicity": "Holm correction across mapped tasks per representation",
            "family_alpha": 0.05,
        },
        "representation_controls": {
            "learned_embedding": "unchanged original embedding; inferential secondary",
            "interaction_features": (
                "label-blind pooled median/IQR transform; mechanism control"
            ),
            "trajectory_summary": (
                "label-blind pooled median/IQR transform; mechanism control"
            ),
            "controls_do_not_replace_primary": True,
        },
        "locked_intake_requirements": {
            "development_scenario_token_overlap": 0,
            "development_log_overlap": 0,
            "planner_parameter_fingerprints_must_match": True,
            "minimum_overall_complete_pairs_operational_floor": minimum_overall_pairs,
            "minimum_pairs_per_task_operational_floor": minimum_task_pairs,
            "operational_floors_are_not_power_analysis": True,
            "separate_power_justification_file_required": True,
            "selection_and_simulation_config_frozen_before_label_unblinding": True,
        },
        "interpretation_limits": [
            "BDD significance does not imply safety or planner superiority.",
            "Task frequency shift and within-task behavior shift are separate estimands.",
            "Outcome-derived task bins are sensitivity-only and not confirmatory matching strata.",
        ],
    }


def parse_representation_argument(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--representation must use NAME=PATH syntax, got {value!r}"
            )
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate representation name: {name!r}")
        result[name] = Path(raw_path)
    if "learned_embedding" not in result:
        raise ValueError("--representation must include learned_embedding=PATH")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and validate Stage7 M6.2 pre-treatment task-conditioned paired BDD."
        )
    )
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--development_metadata_csv", type=Path, required=True)
    parser.add_argument("--m6_frozen_spec", type=Path, required=True)
    parser.add_argument("--representation", action="append", default=[], required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--analysis_role",
        choices=("development_validation", "locked_confirmation"),
        required=True,
    )
    parser.add_argument("--lock_manifest", type=Path)
    parser.add_argument("--power_justification_file", type=Path)
    parser.add_argument("--planner_a", required=True)
    parser.add_argument("--planner_b", required=True)
    parser.add_argument("--minimum_overall_pairs", type=int, default=80)
    parser.add_argument("--minimum_task_pairs", type=int, default=12)
    parser.add_argument("--task_monte_carlo_permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    metadata = pd.read_csv(args.metadata_csv)
    development_metadata = pd.read_csv(args.development_metadata_csv)
    paired_rows = read_csv_records(args.paired_delta_csv)
    representations_paths = parse_representation_argument(args.representation)
    representations: Dict[str, np.ndarray] = {}
    for name, path in representations_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"representation {name} must be 2D, got {values.shape}")
        if len(values) != len(metadata):
            raise ValueError(
                f"representation {name}/metadata row mismatch: "
                f"{len(values)} != {len(metadata)}"
            )
        if name != "learned_embedding":
            values = robust_pooled_transform(values)
        elif not np.isfinite(values).all():
            raise ValueError("learned_embedding contains non-finite values")
        representations[name] = values

    pair_indices, scenarios = validate_and_build_pairs(
        metadata,
        paired_rows,
        len(metadata),
        planner_a=args.planner_a,
        planner_b=args.planner_b,
    )
    task_masks, task_definition_table = build_pretreatment_task_masks(
        metadata, pair_indices
    )

    tool_path = Path(__file__).resolve()
    power_validation = None
    if args.analysis_role == "development_validation":
        lock_spec = build_lock_spec(
            frozen_spec_path=args.m6_frozen_spec,
            development_metadata_path=args.development_metadata_csv,
            planners=(args.planner_a, args.planner_b),
            development_metadata=development_metadata,
            tool_path=tool_path,
            minimum_overall_pairs=args.minimum_overall_pairs,
            minimum_task_pairs=args.minimum_task_pairs,
        )
        write_json(args.output_dir / "m6_2_locked_confirmation_spec.json", lock_spec)
        (args.output_dir / "m6_2_power_justification_template.md").write_text(
            "\n".join(
                [
                    "# M6.2 Locked Confirmation Power Justification",
                    "",
                    "This file must be completed and frozen before planner labels from the new confirmation set are analyzed.",
                    "",
                    "## Primary endpoint",
                    "",
                    "- Unchanged 64D embedding, frozen M6.1 single-RBF paired-label-swap BDD.",
                    "- Family alpha: 0.05 for the overall primary endpoint.",
                    "",
                    "## Required simulation-based power analysis",
                    "",
                    "Because no closed-form power formula is assumed for the frozen MMD randomization test, document:",
                    "",
                    "1. An alternative-distribution generator specified without using labels from the locked set.",
                    "2. A scientifically meaningful effect grid, including the smallest effect worth detecting.",
                    "3. Expected task proportions, missingness, failed-rollout and quality-tier attrition.",
                    "4. At least 1000 simulated experiments per candidate sample size.",
                    "5. The smallest sample size with at least 80% rejection probability under the frozen test.",
                    "6. Sensitivity to weaker effects and higher attrition.",
                    "",
                    "## Task-conditioned secondary family",
                    "",
                    "- Pre-treatment scenario-type tasks only.",
                    "- Holm correction across the frozen task family.",
                    "- State target complete-pair count for every task and its simulated power.",
                    "",
                    "## Operational floors",
                    "",
                    f"- Overall complete pairs: {args.minimum_overall_pairs}.",
                    f"- Complete pairs per task: {args.minimum_task_pairs}.",
                    "- These are data-quality floors, not evidence of adequate statistical power.",
                    "",
                    "## Freeze record",
                    "",
                    "- Power-analysis code/version:",
                    "- Random seed(s):",
                    "- Selected total sample size:",
                    "- Selected per-task targets:",
                    "- Expected attrition:",
                    "- Approval date and analyst:",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        disjointness = {
            "status": "NOT_APPLICABLE_DEVELOPMENT_VALIDATION",
            "passed": None,
        }
        dataset_role = "METHOD_DEVELOPMENT_ONLY_NOT_CONFIRMATORY"
    else:
        if args.lock_manifest is None:
            raise ValueError("locked_confirmation requires --lock_manifest")
        if args.power_justification_file is None:
            raise ValueError(
                "locked_confirmation requires --power_justification_file"
            )
        lock_spec = read_json(args.lock_manifest)
        if sha256_file(tool_path) != lock_spec.get("analysis_tool_sha256"):
            raise ValueError("analysis tool differs from frozen lock manifest")
        if sha256_file(args.m6_frozen_spec) != lock_spec.get(
            "frozen_m6_1_spec_sha256"
        ):
            raise ValueError("M6.1 frozen spec differs from lock manifest")
        if sha256_file(args.development_metadata_csv) != lock_spec.get(
            "development_metadata_sha256"
        ):
            raise ValueError("development metadata differs from lock manifest")
        if not args.power_justification_file.is_file():
            raise FileNotFoundError(args.power_justification_file)
        power_payload = read_json(args.power_justification_file)
        disjointness = audit_locked_disjointness(
            development_metadata,
            metadata,
            planners=(args.planner_a, args.planner_b),
        )
        if not disjointness["passed"]:
            raise ValueError(
                f"locked confirmation leakage audit failed: {disjointness}"
            )
        requirements = lock_spec["locked_intake_requirements"]
        if len(pair_indices) < int(
            requirements["minimum_overall_complete_pairs_operational_floor"]
        ):
            raise ValueError("locked confirmation is below overall pair floor")
        power_validation = validate_frozen_power_justification(
            power_payload,
            lock_manifest_path=args.lock_manifest,
            pair_count=len(pair_indices),
            task_masks=task_masks,
        )
        dataset_role = "NEW_LOG_AND_SCENARIO_DISJOINT_CONFIRMATION"

    rows: List[Dict[str, Any]] = []
    null_samples: Dict[str, np.ndarray] = {}
    for representation_position, (representation, values) in enumerate(
        representations.items()
    ):
        inferential_p: List[float] = []
        inferential_positions: List[int] = []
        for task_position, (task, mask) in enumerate(task_masks.items()):
            selected = pair_indices[mask]
            record: Dict[str, Any] = {
                "representation": representation,
                "task": task,
                "selection_timing": "pre_treatment",
                "n_pairs": int(len(selected)),
                "minimum_task_pairs_operational_floor": args.minimum_task_pairs,
                "meets_operational_floor": bool(
                    len(selected) >= args.minimum_task_pairs
                ),
                "analysis_role": (
                    "inferential_secondary"
                    if representation == "learned_embedding"
                    else "mechanism_control"
                ),
            }
            if len(selected) < 2:
                record.update(
                    {
                        "status": "INSUFFICIENT_PAIRS",
                        "mmd2": None,
                        "p_value": None,
                    }
                )
            else:
                result, samples = paired_randomization_test(
                    values[selected[:, 0]],
                    values[selected[:, 1]],
                    monte_carlo_repetitions=args.task_monte_carlo_permutations,
                    seed=args.seed
                    + representation_position * 100
                    + task_position,
                    progress_label=f"{representation}/{task}",
                )
                record.update(
                    {
                        "status": "DEVELOPMENT_RESULT"
                        if args.analysis_role == "development_validation"
                        else "LOCKED_CONFIRMATION_RESULT",
                        "mmd2": result["mmd2"],
                        "bandwidth": result["bandwidth"],
                        "randomization_mode": result["randomization_mode"],
                        "unique_label_assignments": result[
                            "unique_label_assignments"
                        ],
                        "exceedance_count": result["exceedance_count"],
                        "p_value": result["p_value"],
                    }
                )
                null_samples[f"{representation}__{task}"] = samples
                if representation == "learned_embedding":
                    inferential_positions.append(len(rows))
                    inferential_p.append(float(result["p_value"]))
            rows.append(record)
        if representation == "learned_embedding" and inferential_p:
            adjusted = holm_adjust(inferential_p)
            for position, value in zip(inferential_positions, adjusted):
                rows[position]["holm_p_within_pretreatment_tasks"] = value
                rows[position]["reject_holm_0_05"] = bool(value <= 0.05)

    result_table = pd.DataFrame(rows)
    result_table.to_csv(args.output_dir / "table_m6_2_task_paired_bdd.csv", index=False)
    (args.output_dir / "table_m6_2_task_paired_bdd.md").write_text(
        markdown_table(
            rows,
            [
                "representation",
                "task",
                "n_pairs",
                "meets_operational_floor",
                "mmd2",
                "randomization_mode",
                "exceedance_count",
                "p_value",
                "holm_p_within_pretreatment_tasks",
            ],
        ),
        encoding="utf-8",
    )
    task_definition_table.to_csv(
        args.output_dir / "m6_2_pretreatment_task_definitions.csv", index=False
    )
    np.savez_compressed(
        args.output_dir / "m6_2_task_null_samples.npz", **null_samples
    )

    learned_rows = result_table.loc[
        result_table["representation"] == "learned_embedding"
    ]
    summary = {
        "milestone": "Stage 7 Milestone 6.2",
        "analysis_role": args.analysis_role,
        "dataset_role": dataset_role,
        "n_pairs": int(len(pair_indices)),
        "n_scenarios": int(len(scenarios)),
        "task_selection_is_pre_treatment": True,
        "post_treatment_task_bins_used_for_confirmatory_selection": False,
        "disjointness_audit": disjointness,
        "power_justification": (
            {
                "path": str(args.power_justification_file),
                "sha256": sha256_file(args.power_justification_file),
                "sample_target_validation": power_validation,
            }
            if args.power_justification_file
            else {
                "status": "REQUIRED_BEFORE_LOCKED_CONFIRMATION",
                "operational_floors_are_not_power_analysis": True,
            }
        ),
        "task_counts": {
            row["task"]: int(row["n_pairs"])
            for row in task_definition_table.to_dict("records")
        },
        "learned_embedding_tasks_meeting_operational_floor": int(
            learned_rows["meets_operational_floor"].sum()
        ),
        "locked_confirmation_ready": bool(
            args.analysis_role == "locked_confirmation"
            and disjointness["passed"]
            and learned_rows["meets_operational_floor"].all()
        ),
        "limitations": [
            "Current development results are not independent confirmation."
            if args.analysis_role == "development_validation"
            else "Confirmation is limited to the frozen scenario-type task family.",
            "Operational pair floors do not replace a documented power analysis.",
            "Task frequency and within-task BDD must be reported separately.",
            "Representation controls are mechanism analyses and do not replace the unchanged embedding primary.",
        ],
    }
    write_json(args.output_dir / "milestone6_2_summary.json", summary)

    report = [
        "# Stage 7 Milestone 6.2 Locked Confirmation and Task-conditioned Paired BDD",
        "",
        "## Status",
        "",
        f"- analysis role: `{args.analysis_role}`",
        f"- dataset role: `{dataset_role}`",
        f"- complete pairs: `{len(pair_indices)}`",
        "- task selection: pre-treatment `metadata.scenario_type`",
        "- outcome-derived task bins are not used to define confirmatory strata",
        "",
        "## Task coverage",
        "",
        markdown_table(
            task_definition_table.to_dict("records"),
            ["task", "selection_timing", "scenario_types", "n_pairs"],
        ),
        "## Task-conditioned paired BDD",
        "",
        markdown_table(
            rows,
            [
                "representation",
                "task",
                "n_pairs",
                "meets_operational_floor",
                "mmd2",
                "randomization_mode",
                "exceedance_count",
                "p_value",
                "holm_p_within_pretreatment_tasks",
            ],
        ),
        "## Interpretation constraints",
        "",
        "- The learned embedding is the inferential secondary representation; handcrafted controls are mechanism checks.",
        "- Subsets with at most 20 pairs use every possible within-pair label assignment.",
        "- Development-set p-values validate implementation and calibrate the method; they are not confirmatory evidence.",
        "- A locked run additionally requires zero development log/scenario overlap, identical frozen planner parameter fingerprints, and a separate power justification.",
        "- Task frequency shift and within-task behavior shift are distinct results.",
        "",
    ]
    (args.output_dir / "milestone6_2_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        f"M6.2 {args.analysis_role}: {len(pair_indices)} pairs, "
        f"{summary['learned_embedding_tasks_meeting_operational_floor']} learned "
        "embedding tasks meet the operational floor"
    )


if __name__ == "__main__":
    main()
