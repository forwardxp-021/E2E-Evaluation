#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7d_extract_neighbors_from_nuplan import find_msgpack


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def token_of(row: Dict[str, Any]) -> str:
    return str(
        row.get("actual_nuplan_scenario_token")
        or row.get("scenario_token")
        or row.get("scenario_id")
        or row.get("scene_token")
        or ""
    ).strip()


def nested_find(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = nested_find(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = nested_find(item, key)
            if found is not None:
                return found
    return None


def physical_anomalies(
    ego_path: Path,
    rollout_mask_path: Path,
    metadata: List[Dict[str, str]],
    planner_count: int,
    dt: float = 0.1,
) -> List[Dict[str, Any]]:
    ego = np.load(ego_path, mmap_mode="r")
    rollout_mask = np.load(rollout_mask_path, mmap_mode="r")
    if ego.shape[0] != len(metadata) or ego.ndim != 3 or ego.shape[-1] < 8:
        raise ValueError(f"ego/metadata mismatch: ego={list(ego.shape)}, metadata={len(metadata)}")
    if rollout_mask.shape[:2] != (len(metadata) // planner_count, planner_count):
        raise ValueError(
            f"rollout mask prefix mismatch: mask={list(rollout_mask.shape)}, "
            f"metadata={len(metadata)}, planner_count={planner_count}"
        )
    anomalies: List[Dict[str, Any]] = []
    for row_index, meta in enumerate(metadata):
        row = np.asarray(ego[row_index], dtype=np.float64)
        lateral_accel = np.diff(row[:, 3], prepend=row[0, 3]) / dt
        curvature = np.full(row.shape[0], np.nan, dtype=np.float64)
        curvature_valid = np.isfinite(row[:, 5]) & np.isfinite(row[:, 7]) & (np.abs(row[:, 5]) >= 0.5)
        curvature[curvature_valid] = row[curvature_valid, 7] / np.abs(row[curvature_valid, 5])
        for metric, values, cap in [
            ("raw_abs_lateral_accel", np.abs(lateral_accel), 8.0),
            ("raw_abs_curvature", np.abs(curvature), 1.0),
        ]:
            for timestep in np.flatnonzero(np.isfinite(values) & (values > cap)):
                anomalies.append(
                    {
                        "global_row": row_index,
                        "tensor_scenario_position": meta.get("tensor_scenario_position", ""),
                        "scenario_index": meta.get("scenario_index", ""),
                        "scenario_token": token_of(meta),
                        "planner_id": meta.get("planner_id", ""),
                        "planner_name": meta.get("planner_name", ""),
                        "timestep": int(timestep),
                        "time_s": float(timestep * dt),
                        "valid_rollout_frame": bool(rollout_mask[row_index // planner_count, row_index % planner_count, timestep]),
                        "metric": metric,
                        "value": float(values[timestep]),
                        "cap": cap,
                    }
                )
    return sorted(anomalies, key=lambda item: float(item["value"]), reverse=True)


def run(args: argparse.Namespace) -> None:
    sim_dir = Path(args.sim_dir)
    context_dir = Path(args.context_dir)
    embedding_dir = Path(args.embedding_dir)
    stage7f_dir = Path(args.stage7f_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    axis_json = read_json(require(sim_dir / "simulated_ego_seq_index.json"))
    scenario_axis = [int(value) for value in axis_json["scenario_axis"]]
    planner_axis = [int(value) for value in axis_json["planner_axis"]]
    planner_names = list(axis_json["planner_axis_names"])
    index_rows = read_csv(require(sim_dir / "scenario_planner_index.csv"))
    context_rows = read_csv(require(context_dir / "metadata.csv"))
    embedding_rows = read_csv(require(embedding_dir / "metadata.csv"))
    by_pair = {
        (int(row["scenario_index"]), int(row["planner_id"])): row
        for row in index_rows
        if row.get("scenario_index", "").strip() and row.get("planner_id", "").strip()
    }

    alignment_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    expected_rows = len(scenario_axis) * len(planner_axis)
    if len(context_rows) != expected_rows:
        errors.append(f"context metadata rows={len(context_rows)} expected={expected_rows}")
    if len(embedding_rows) != expected_rows:
        errors.append(f"embedding metadata rows={len(embedding_rows)} expected={expected_rows}")
    for tensor_position, scenario_index in enumerate(scenario_axis):
        for planner_position, planner_id in enumerate(planner_axis):
            global_row = tensor_position * len(planner_axis) + planner_position
            expected = by_pair.get((scenario_index, planner_id))
            if expected is None:
                errors.append(f"missing Stage7C pair scenario={scenario_index} planner={planner_id}")
                continue
            observed = context_rows[global_row] if global_row < len(context_rows) else {}
            embedded = embedding_rows[global_row] if global_row < len(embedding_rows) else {}
            checks = {
                "status_succeeded": expected.get("status") == "succeeded",
                "tensor_position_matches": str(observed.get("tensor_scenario_position")) == str(tensor_position),
                "scenario_index_matches": str(observed.get("scenario_index")) == str(scenario_index),
                "planner_id_matches": str(observed.get("planner_id")) == str(planner_id),
                "planner_name_matches": observed.get("planner_name") == planner_names[planner_position] == expected.get("planner_name"),
                "token_matches": token_of(observed) == token_of(expected),
                "embedding_metadata_matches": all(
                    str(embedded.get(key, "")) == str(observed.get(key, ""))
                    for key in ["scenario_index", "planner_id", "planner_name", "scenario_token"]
                ),
            }
            msgpack = ""
            try:
                msgpack = str(find_msgpack(sim_dir, expected) or "")
                checks["msgpack_identity_matches"] = bool(msgpack)
            except (ValueError, FileNotFoundError) as exc:
                checks["msgpack_identity_matches"] = False
                errors.append(str(exc))
            passed = all(checks.values())
            if not passed:
                errors.append(f"alignment failed at global_row={global_row}: {checks}")
            alignment_rows.append(
                {
                    "global_row": global_row,
                    "tensor_scenario_position": tensor_position,
                    "scenario_index": scenario_index,
                    "planner_id": planner_id,
                    "planner_name": planner_names[planner_position],
                    "scenario_token": token_of(expected),
                    "msgpack_path": msgpack,
                    "passed": passed,
                }
            )

    context_array = np.load(require(context_dir / "context_traj.npy"), mmap_mode="r")
    context_mask = np.load(require(context_dir / "ego_seq_mask.npy"), mmap_mode="r")
    source_mask = np.load(require(sim_dir / "simulated_ego_seq_mask.npy"), mmap_mode="r")
    embedding_array = np.load(require(embedding_dir / "embedding.npy"), mmap_mode="r")
    if context_array.shape[0] != expected_rows or context_array.shape[-1] != 83:
        errors.append(f"context shape invalid: {list(context_array.shape)}")
    if embedding_array.shape[0] != expected_rows or embedding_array.shape[-1] != 64:
        errors.append(f"embedding shape invalid: {list(embedding_array.shape)}")
    expected_context_mask = np.asarray(source_mask, dtype=bool).reshape(expected_rows, source_mask.shape[-1])
    context_mask_matches_source = bool(
        context_mask.shape == expected_context_mask.shape
        and np.array_equal(np.asarray(context_mask, dtype=bool), expected_context_mask)
    )
    if not context_mask_matches_source:
        errors.append(
            f"context ego_seq_mask does not match Stage7C source mask: "
            f"context={list(context_mask.shape)}, expected={list(expected_context_mask.shape)}"
        )

    warnings_json = read_json(require(context_dir / "warnings.json"))
    strict_json = read_json(require(context_dir / "nuplan_laneaware_strict_filter_summary.json"))
    stage7f_json = read_json(require(stage7f_dir / "report_card" / "stage7f_summary.json"))
    pairwise_json = read_json(require(stage7f_dir / "report_card" / "stage7f_pairwise_summary.json"))
    paired_delta_json = read_json(
        require(stage7f_dir / "paired_delta_assertive_minus_conservative" / "paired_delta_summary.json")
    )
    task_bdd_rows = read_csv(
        require(stage7f_dir / "task_bdd_assertive_minus_conservative_v1" / "task_bdd_summary.csv")
    )
    validation = nested_find(warnings_json, "validation") or {}
    fallback_rate = nested_find(warnings_json, "fallback_assignment_used_rate")
    lane_success = nested_find(warnings_json, "lane_assignment_available")
    physical_warning_path = stage7f_dir / "behavior_events_v2" / "behavior_event_warnings_v2.json"
    behavior_warnings = read_json(require(physical_warning_path))
    physical_warnings = [
        item
        for item in behavior_warnings
        if item.get("warning") == "metric_physical_range_warning"
    ]
    stage6_mask_warning = next(
        (item for item in behavior_warnings if item.get("warning") == "rollout_validity_mask_applied"),
        None,
    )
    anomaly_rows = physical_anomalies(
        context_dir / "ego_seq.npy",
        sim_dir / "simulated_ego_seq_mask.npy",
        context_rows,
        len(planner_axis),
    )
    valid_anomaly_rows = [row for row in anomaly_rows if row["valid_rollout_frame"]]
    padded_anomaly_rows = [row for row in anomaly_rows if not row["valid_rollout_frame"]]

    stage7f_alignment = stage7f_json.get("alignment", {})
    structural_pass = bool(
        not errors
        and validation.get("scenario_planner_token_alignment_strict") is True
        and validation.get("msgpack_global_fallback_disabled") is True
        and context_mask_matches_source
        and stage7f_alignment.get("all_scenarios_have_all_planners") is True
        and stage7f_alignment.get("total_rows") == expected_rows
    )
    limitations = []
    if fallback_rate is not None and float(fallback_rate) >= 0.5:
        limitations.append("high_geometric_fallback_rate")
    if int(strict_json.get("rows_kept", 0)) < 10:
        limitations.append("strict_0_8_sample_too_small")
    if physical_warnings:
        limitations.append("raw_physical_metric_warnings")
    if padded_anomaly_rows and stage6_mask_warning is None:
        limitations.append("stage6_physical_metrics_include_invalid_padding_frames")
    if valid_anomaly_rows:
        limitations.append("valid_frame_physical_outlier")
    overall_verdict = (
        "PASS"
        if structural_pass and not limitations
        else "PASS_WITH_LIMITATIONS"
        if structural_pass
        else "FAIL"
    )
    pairwise_row = pairwise_json.get("rows", [{}])[0] if pairwise_json.get("rows") else {}
    max_task_bdd = max(
        (float(row["bdd_mmd"]) for row in task_bdd_rows if row.get("bdd_mmd", "").strip()),
        default=None,
    )

    summary = {
        "milestone": "Stage 7 Milestone 1 PDM data credibility re-audit",
        "scenario_axis": scenario_axis,
        "scenario_axis_non_contiguous": scenario_axis != list(range(len(scenario_axis))),
        "successful_scenarios": len(scenario_axis),
        "successful_planner_runs": expected_rows,
        "alignment_rows_passed": sum(bool(row["passed"]) for row in alignment_rows),
        "alignment_rows_failed": sum(not bool(row["passed"]) for row in alignment_rows),
        "context_shape": list(context_array.shape),
        "embedding_shape": list(embedding_array.shape),
        "ego_seq_mask_shape": list(context_mask.shape),
        "ego_seq_mask_matches_stage7c": context_mask_matches_source,
        "strict_scenario_planner_token_alignment": structural_pass,
        "msgpack_global_fallback_disabled": validation.get("msgpack_global_fallback_disabled"),
        "fallback_assignment_used_rate": fallback_rate,
        "lane_assignment_available": lane_success,
        "map_name_resolved_rate": nested_find(warnings_json, "map_name_resolved_rate"),
        "map_query_success": nested_find(warnings_json, "map_query_success"),
        "lane_info_count": nested_find(warnings_json, "lane_info_count"),
        "strict_0_8": {
            "rows_kept": strict_json.get("rows_kept"),
            "kept_row_rate": strict_json.get("kept_row_rate"),
            "scenarios_with_all_planners": strict_json.get("scenarios_with_all_planners"),
        },
        "stage7f_results": {
            "bdd_mmd2": pairwise_row.get("bdd_mmd2"),
            "bdd_permutation_p_value": pairwise_row.get("permutation_p_value"),
            "paired_scenarios": paired_delta_json.get("num_paired_scenarios"),
            "assertive_gt_conservative_speed_fraction": paired_delta_json.get(
                "aggressive_gt_conservative_speed_fraction"
            ),
            "assertive_gt_conservative_accel_fraction": paired_delta_json.get(
                "aggressive_gt_conservative_accel_fraction"
            ),
            "assertive_smaller_thw_fraction": paired_delta_json.get(
                "aggressive_smaller_thw_fraction"
            ),
            "valid_task_bdd_count": len(task_bdd_rows),
            "max_task_bdd": max_task_bdd,
        },
        "physical_metric_warnings": physical_warnings,
        "stage6_rollout_validity_mask_applied": stage6_mask_warning is not None,
        "stage6_padding_frames_excluded": (
            stage6_mask_warning.get("padding_frames_excluded") if stage6_mask_warning else None
        ),
        "physical_anomaly_points": len(anomaly_rows),
        "physical_anomaly_points_on_valid_frames": len(valid_anomaly_rows),
        "physical_anomaly_points_on_invalid_padding_frames": len(padded_anomaly_rows),
        "limitations": limitations,
        "errors": errors,
        "overall_verdict": overall_verdict,
    }
    write_csv(output_dir / "alignment_audit.csv", alignment_rows)
    write_csv(
        output_dir / "physical_anomalies.csv",
        anomaly_rows,
        [
            "global_row", "tensor_scenario_position", "scenario_index", "scenario_token",
            "planner_id", "planner_name", "timestep", "time_s", "valid_rollout_frame",
            "metric", "value", "cap",
        ],
    )
    write_json(output_dir / "audit_summary.json", summary)
    report = [
        "# Stage 7 Milestone 1 — PDM Data Credibility Re-audit",
        "",
        f"## Verdict: `{overall_verdict}`",
        "",
        f"- strict alignment rows passed: `{summary['alignment_rows_passed']}/{expected_rows}`",
        f"- scenario axis: `{scenario_axis}`",
        f"- context shape: `{summary['context_shape']}`",
        f"- embedding shape: `{summary['embedding_shape']}`",
        f"- Stage7F complete paired alignment: `{stage7f_alignment.get('all_scenarios_have_all_planners')}`",
        f"- geometric fallback rate: `{fallback_rate}`",
        f"- strict-0.8 rows kept: `{strict_json.get('rows_kept')}`",
        f"- full BDD MMD² / permutation p: `{pairwise_row.get('bdd_mmd2')}` / `{pairwise_row.get('permutation_p_value')}`",
        f"- valid task-conditioned BDD rows: `{len(task_bdd_rows)}`",
        f"- raw physical anomaly points: `{len(anomaly_rows)}`",
        f"- physical anomalies on valid rollout frames: `{len(valid_anomaly_rows)}`",
        f"- physical anomalies caused by invalid/padded frames: `{len(padded_anomaly_rows)}`",
        f"- Stage6C excluded invalid padding frames: `{stage6_mask_warning is not None}`",
        f"- limitations: `{limitations}`",
        "",
        "The repaired main dataset is structurally aligned. Remaining fallback, strict-subset-size, "
        "and physical-range items limit interpretation strength and are not alignment failures.",
        "",
        "Detailed row checks are in `alignment_audit.csv`; physical outliers with scenario/planner/timestep "
        "are in `physical_anomalies.csv`.",
    ]
    (output_dir / "data_quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not structural_pass:
        raise RuntimeError(f"Milestone 1 re-audit failed: {errors}")
    print(f"Stage7 Milestone 1 re-audit {overall_verdict}: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage7 PDM scenario/planner/token/msgpack alignment and data quality.")
    parser.add_argument("--sim_dir", required=True)
    parser.add_argument("--context_dir", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--stage7f_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
