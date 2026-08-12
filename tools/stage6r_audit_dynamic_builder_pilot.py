#!/usr/bin/env python3
"""Compare legacy static slots with Dynamic Interaction Builder v2 pilot outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage5d_context_core import SLOT_NAMES
from tools.stage6q_audit_waymo_raw_interaction_coverage import event_flags


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_shards(root: Path) -> list[Path]:
    summary = read_json(root / "build_summary.json")
    result = []
    for raw in summary["shard_paths"]:
        value = Path(raw)
        workspace_outputs = root.parents[2] / "outputs" if len(root.parents) >= 3 else root.parent
        candidates = [
            value,
            root / "shards" / value.name,
            workspace_outputs / value.parent.parent.name / "shards" / value.name,
        ]
        found = next((item.resolve() for item in candidates if item.is_dir()), None)
        if found is None:
            raise FileNotFoundError(raw)
        result.append(found)
    return result


def runs(values: list[str]) -> list[tuple[int, int, str]]:
    if not values:
        return []
    result = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            result.append((start, index, values[start]))
            start = index
    return result


def summarize_dynamic(roots: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    frame_valid = np.zeros(5, dtype=np.int64)
    frames = 0
    occupied_windows = np.zeros(5, dtype=np.int64)
    switch_counts = np.zeros(5, dtype=np.int64)
    transition_counts = np.zeros(5, dtype=np.int64)
    direction_valid = np.zeros(5, dtype=np.int64)
    finite = True
    shape_pass = True
    derivative_violations = 0
    events = Counter()
    cases: list[dict[str, Any]] = []
    scenario_ids: set[str] = set()
    row_count = 0
    raw_targets = []
    normalized_targets = []
    geometric_fallback_windows = 0
    semantic_manifests_pass = True
    for root in roots:
        build_summary = read_json(root / "build_summary.json")
        manifest = read_json(root / "dynamic_builder_v2_manifest.json")
        geometric_fallback_windows += sum(
            int(slot_counts.get("geometric_fallback", 0))
            for slot_counts in build_summary.get("assignment_method_counts_by_slot", {}).values()
        )
        semantic_manifests_pass &= bool(
            manifest.get("assignment_mode_required") == "lane_aware_only"
            and manifest.get("geometric_adjacent_lane_inference") is False
        )
        for shard in resolve_shards(root):
            meta = np.load(shard / "meta.npy", allow_pickle=True)
            ids = np.load(shard / "slot_track_id_timeline.npy").astype(str)
            valid = np.load(shard / "slot_valid_mask.npy")
            switches = np.load(shard / "slot_identity_switch_mask.npy")
            derivative = np.load(shard / "slot_derivative_valid_mask.npy")
            neighbor = np.load(shard / "neighbor_seq.npy")
            raw = np.load(shard / "longitudinal_supervision_v2_raw.npy")
            normalized = np.load(shard / "longitudinal_supervision_v2.npy")
            expected = (len(meta), 5, neighbor.shape[2])
            shape_pass &= ids.shape == valid.shape == switches.shape == derivative.shape == expected
            finite &= bool(np.isfinite(neighbor).all() and np.isfinite(raw).all() and np.isfinite(normalized).all())
            derivative_violations += int(np.sum((~derivative) & ((neighbor[:, :, :, 12] != 0.0) | (neighbor[:, :, :, 14] != 0.0))))
            row_count += len(meta)
            frames += len(meta) * neighbor.shape[2]
            frame_valid += valid.sum(axis=(0, 2))
            occupied_windows += valid.any(axis=2).sum(axis=0)
            switch_counts += switches.sum(axis=(0, 2))
            transition_counts += (valid[:, :, 1:] & valid[:, :, :-1]).sum(axis=(0, 2))
            conditions = [
                neighbor[:, 0, :, 1] > 0,
                (neighbor[:, 1, :, 1] > 0) & (neighbor[:, 1, :, 2] > 0),
                (neighbor[:, 2, :, 1] <= 0) & (neighbor[:, 2, :, 2] > 0),
                (neighbor[:, 3, :, 1] > 0) & (neighbor[:, 3, :, 2] < 0),
                (neighbor[:, 4, :, 1] <= 0) & (neighbor[:, 4, :, 2] < 0),
            ]
            for slot in range(5):
                direction_valid[slot] += int(np.sum(conditions[slot] & valid[:, slot]))
            raw_targets.append(raw.reshape(-1, 3))
            normalized_targets.append(normalized.reshape(-1, 3))
            for row_index, meta_row in enumerate(meta):
                scenario_id = str(meta_row["scenario_id"])
                scenario_ids.add(scenario_id)
                lead = [None if value == "-1" else value for value in ids[row_index, 0].tolist()]
                front_valid = valid[row_index, 0]
                gaps = np.where(front_valid, neighbor[row_index, 0, :, 5], np.nan)
                closing = np.where(front_valid, neighbor[row_index, 0, :, 8], np.nan)
                flags = event_flags(lead, gaps, closing, 5)
                for name in ["lead_entry", "lead_exit", "intermittent_following_primary", "front_identity_switch", "free_flow_to_closing_to_following", "following_to_free_flow"]:
                    events[name] += int(bool(flags[name]))
                event_names = [name for name in ["lead_entry", "lead_exit", "intermittent_following_primary", "front_identity_switch", "free_flow_to_closing_to_following", "following_to_free_flow"] if flags[name]]
                for slot_index, slot_name in enumerate(SLOT_NAMES):
                    slot_valid = valid[row_index, slot_index]
                    if not np.any(slot_valid):
                        continue
                    slot_rel_x = neighbor[row_index, slot_index, :, 1][slot_valid]
                    slot_rel_y = neighbor[row_index, slot_index, :, 2][slot_valid]
                    slot_expected = conditions[slot_index][row_index][slot_valid]
                    slot_events = event_names if slot_index == 0 else (["identity_switch"] if np.any(switches[row_index, slot_index]) else ["occupied"])
                    cases.append({
                        "scenario_id": scenario_id,
                        "target_agent_id": str(meta_row["target_agent_id"]),
                        "start": int(meta_row["start"]),
                        "split": str(meta_row["split"]),
                        "slot": slot_name,
                        "events": ";".join(slot_events),
                        "slot_valid_ratio": float(np.mean(slot_valid)),
                        "slot_identity_segments": ";".join(f"{a}:{b}:{value}" for a, b, value in runs(ids[row_index, slot_index].tolist())),
                        "rel_x_min": float(np.min(slot_rel_x)),
                        "rel_x_max": float(np.max(slot_rel_x)),
                        "rel_y_min": float(np.min(slot_rel_y)),
                        "rel_y_max": float(np.max(slot_rel_y)),
                        "automated_semantic_check": "PASS" if np.all(slot_expected) else "FAIL",
                        "manual_review": "PENDING",
                    })
    raw = np.concatenate(raw_targets)
    normalized = np.concatenate(normalized_targets)
    summary = {
        "row_count": row_count,
        "scenario_count": len(scenario_ids),
        "slot_valid_frame_ratio": {name: float(frame_valid[i] / max(1, frames)) for i, name in enumerate(SLOT_NAMES)},
        "slot_occupied_window_ratio": {name: float(occupied_windows[i] / max(1, row_count)) for i, name in enumerate(SLOT_NAMES)},
        "slot_identity_switch_count": {name: int(switch_counts[i]) for i, name in enumerate(SLOT_NAMES)},
        "slot_switch_rate": {name: float(switch_counts[i] / max(1, transition_counts[i])) for i, name in enumerate(SLOT_NAMES)},
        "slot_direction_correct_ratio": {name: float(direction_valid[i] / max(1, frame_valid[i])) for i, name in enumerate(SLOT_NAMES)},
        "events": dict(events),
        "finite": finite,
        "shape_pass": bool(shape_pass),
        "derivative_cross_identity_violation_count": derivative_violations,
        "geometric_fallback_window_count": geometric_fallback_windows,
        "semantic_strict_manifest_pass": bool(semantic_manifests_pass),
        "longitudinal_raw_abs_quantiles": {str(q): np.quantile(np.abs(raw), q, axis=0).tolist() for q in [0.5, 0.95, 0.99, 0.999]},
        "longitudinal_normalized_abs_quantiles": {str(q): np.quantile(np.abs(normalized), q, axis=0).tolist() for q in [0.5, 0.95, 0.99, 0.999]},
    }
    return summary, cases, scenario_ids


def summarize_legacy(root: Path, scenario_ids: set[str]) -> dict[str, Any]:
    row_count = 0
    selected_scenarios: set[str] = set()
    static_shape_rows = 0
    valid_frames = np.zeros(5, dtype=np.int64)
    frames = 0
    events = Counter()
    ego_values = []
    for shard in resolve_shards(root):
        meta = np.load(shard / "meta.npy", allow_pickle=True)
        keep = np.asarray([str(row["scenario_id"]) in scenario_ids for row in meta], dtype=bool)
        if not np.any(keep):
            continue
        ids = np.load(shard / "neighbor_slot_ids.npy", allow_pickle=True)[keep]
        neighbor = np.load(shard / "neighbor_seq.npy", mmap_mode="r")[keep]
        ego = np.load(shard / "ego_seq.npy", mmap_mode="r")[keep]
        selected_meta = meta[keep]
        static_shape_rows += int(ids.ndim == 2 and ids.shape[1] == 5) * len(ids)
        row_count += len(ids)
        frames += len(ids) * neighbor.shape[2]
        valid = neighbor[:, :, :, 0] > 0.5
        valid_frames += valid.sum(axis=(0, 2))
        ego_values.append(np.asarray(ego))
        for index, row in enumerate(selected_meta):
            selected_scenarios.add(str(row["scenario_id"]))
            lead_id = str(ids[index, 0])
            lead = [lead_id if valid[index, 0, frame] and lead_id != "-1" else None for frame in range(neighbor.shape[2])]
            gaps = np.where(valid[index, 0], neighbor[index, 0, :, 5], np.nan)
            closing = np.where(valid[index, 0], neighbor[index, 0, :, 8], np.nan)
            flags = event_flags(lead, gaps, closing, 5)
            for name in ["lead_entry", "lead_exit", "intermittent_following_primary", "front_identity_switch", "free_flow_to_closing_to_following", "following_to_free_flow"]:
                events[name] += int(bool(flags[name]))
    ego = np.concatenate(ego_values) if ego_values else np.empty((0, 80, 8))
    old_accel = ego[:, :, 6].reshape(-1) if ego.size else np.empty(0)
    old_jerk = np.diff(ego[:, :, 6], axis=1).reshape(-1) / 0.1 if ego.size else np.empty(0)
    return {
        "row_count": row_count,
        "scenario_count": len(selected_scenarios),
        "five_slots_static_reference_frame_layout_pass": static_shape_rows == row_count,
        "neighbor_slot_ids_shape_contract": "[N,5] fixed per window",
        "slot_valid_frame_ratio": {name: float(valid_frames[i] / max(1, frames)) for i, name in enumerate(SLOT_NAMES)},
        "events": dict(events),
        "front_identity_switch_structurally_representable": False,
        "old_accel_abs_quantiles": {str(q): float(np.quantile(np.abs(old_accel), q)) for q in [0.5, 0.95, 0.99, 0.999]} if old_accel.size else {},
        "old_jerk_abs_quantiles": {str(q): float(np.quantile(np.abs(old_jerk), q)) for q in [0.5, 0.95, 0.99, 0.999]} if old_jerk.size else {},
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    dynamic, cases, scenario_ids = summarize_dynamic([path.resolve() for path in args.dynamic_roots])
    legacy = summarize_legacy(args.legacy_root.resolve(), scenario_ids)
    rng = np.random.default_rng(20260812)
    cases = sorted(cases, key=lambda row: (row["scenario_id"], row["target_agent_id"], row["start"]))
    selected = []
    for slot in SLOT_NAMES:
        candidates = [row for row in cases if row["slot"] == slot and row["slot_valid_ratio"] >= 0.2]
        if len(candidates) < 4:
            candidates = [row for row in cases if row["slot"] == slot]
        count = min(4, len(candidates))
        if count:
            selected.extend(candidates[index] for index in sorted(rng.choice(len(candidates), size=count, replace=False)))
    if selected:
        write_csv(output / "stage6r_manual_semantic_cases.csv", selected)
    gate = config["pilot_gate"]
    checks = {
        "semantic_strict_assignment": dynamic["semantic_strict_manifest_pass"] and dynamic["geometric_fallback_window_count"] == 0,
        "all_five_slots_dynamic": all(dynamic["slot_identity_switch_count"][slot] > 0 for slot in SLOT_NAMES),
        "finite_shape_pass": dynamic["finite"] and dynamic["shape_pass"],
        "front_intermittent_count": dynamic["events"].get("intermittent_following_primary", 0) >= gate["front_intermittent_count_min"],
        "front_entry_count": dynamic["events"].get("lead_entry", 0) >= gate["front_entry_count_min"],
        "front_exit_count": dynamic["events"].get("lead_exit", 0) >= gate["front_exit_count_min"],
        "identity_switch_count": sum(dynamic["slot_identity_switch_count"].values()) >= gate["identity_switch_count_min"],
        "slot_switch_rate": max(dynamic["slot_switch_rate"].values()) <= gate["slot_switch_rate_max"],
        "derivative_cross_identity": dynamic["derivative_cross_identity_violation_count"] <= gate["derivative_cross_identity_violations_max"],
        "manual_case_count": len(selected) >= gate["manual_case_count"],
        "normalized_target_q999": max(dynamic["longitudinal_normalized_abs_quantiles"]["0.999"]) <= gate["normalized_target_abs_q999_max"],
    }
    automated_pass = all(checks.values())
    summary = {
        "schema_version": "stage6r_dynamic_builder_v2_pilot_audit_v1",
        "status": "AUTOMATED_PASS_PENDING_TOPOLOGY_AND_VISUAL_REVIEW" if automated_pass else "PILOT_GATE_FAILED",
        "config_sha256": sha256_file(args.config),
        "legacy": legacy,
        "dynamic_v2": dynamic,
        "gate_checks": checks,
        "topology_reconstruction_complete": False,
        "visual_semantic_review_complete": False,
        "embedding_or_checkpoint_read": False,
        "stage6o_v1_modified": False,
    }
    (output / "stage6r_pilot_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    report = [
        "# Stage 6R Dynamic Interaction Builder v2 pilot审计",
        "",
        f"- 自动门禁：**{summary['status']}**；人工语义复核尚未签字。",
        f"- v2行数/场景：{dynamic['row_count']}/{dynamic['scenario_count']}；旧builder同源场景行数：{legacy['row_count']}。",
        f"- v2 front entry/exit/intermittent/identity switch：{dynamic['events'].get('lead_entry', 0)}/{dynamic['events'].get('lead_exit', 0)}/{dynamic['events'].get('intermittent_following_primary', 0)}/{dynamic['events'].get('front_identity_switch', 0)}。",
        f"- 旧builder `neighbor_slot_ids`为[N,5]整窗固定；front identity switch结构上不可表达：{legacy['front_identity_switch_structurally_representable']}。",
        f"- 五slot逐帧方向正确率：{dynamic['slot_direction_correct_ratio']}。",
        f"- 跨identity导数违规：{dynamic['derivative_cross_identity_violation_count']}；finite/shape：{dynamic['finite']}/{dynamic['shape_pass']}。",
        f"- longitudinal v2 normalized |q999|：{dynamic['longitudinal_normalized_abs_quantiles']['0.999']}。",
        "- 未读取embedding或checkpoint；未修改Stage6O v1。",
        "",
        "## 解释边界",
        "",
        "自动方向检查与随机典型case用于pilot筛查；只有人工复核CSV的20个case后，才允许把pilot记为最终PASS并启动full51。",
    ]
    (output / "stage6r_pilot_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--legacy_root", type=Path, required=True)
    parser.add_argument("--dynamic_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
