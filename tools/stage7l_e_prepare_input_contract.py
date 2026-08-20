#!/usr/bin/env python3
"""Prepare the frozen Stage7L-D rollouts for Stage7L-E representation inference.

This tool does not run nuPlan, read checkpoints, export embeddings, or compute
BDD/MMD.  It only validates the Stage7L-D unlock, replays the frozen C2 task
masks, and creates five read-only Stage7C-compatible views in roster order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_generate_pretreatment_task_masks import (  # noqa: E402
    build_task_mask_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")
TARGET_T = 150
EXPECTED_PROTOCOL_SHA = "f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a"
EXPECTED_ROSTER_SHA = "90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9"
EXPECTED_D_COMMIT = "6279bc742ad527246a945a4b6d5d7090fab591ea"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def git_contains_frozen_commit(repo: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def validate_unlock(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED":
        raise RuntimeError("Stage7L-D planner-level confirmation is not PASS")
    if manifest.get("representation_status") != "STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED":
        raise RuntimeError("Stage7L-E representation evaluation is not unlocked")
    gates = manifest.get("gates", {})
    required = (
        "execution",
        "canonical_identity",
        "mechanism",
        "longitudinal_nuisance",
        "safety_validity",
        "representation_unlock",
    )
    failed = [name for name in required if gates.get(name) is not True]
    if failed:
        raise RuntimeError(f"Stage7L-D unlock gates are not all true: {failed}")
    execution = manifest.get("execution", {})
    if execution.get("successful_cells") != 400 or execution.get("complete_all_five_doses") != 80:
        raise RuntimeError(f"Unexpected frozen Stage7L-D execution inventory: {execution}")


def selected_sources(
    roster: Sequence[Mapping[str, str]], summary_rows: Sequence[Mapping[str, str]]
) -> dict[tuple[str, str], Mapping[str, str]]:
    successful = [
        row
        for row in summary_rows
        if row.get("official_run_status") == "SUCCEEDED"
        and str(row.get("trajectory_available", "")).lower() == "true"
    ]
    by_key: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in successful:
        key = (row.get("scenario_token", ""), row.get("dose", ""))
        if key in by_key:
            raise RuntimeError(f"Duplicate successful Stage7L-D cell: {key}")
        by_key[key] = row
    expected = {(row["scenario_token"], dose) for row in roster for dose in DOSES}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise RuntimeError(f"Frozen Stage7L-D cell mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return by_key


def build_dose_view(
    *,
    dose: str,
    roster: Sequence[Mapping[str, str]],
    sources: Mapping[tuple[str, str], Mapping[str, str]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    official_root = output_dir / "official_nuplan_runs"
    official_root.mkdir()
    shape = (len(roster), 1, TARGET_T, 8)
    seq_out = np.lib.format.open_memmap(
        output_dir / "simulated_ego_seq.npy", mode="w+", dtype=np.float32, shape=shape
    )
    mask_out = np.lib.format.open_memmap(
        output_dir / "simulated_ego_seq_mask.npy", mode="w+", dtype=np.bool_, shape=shape[:-1]
    )
    seq_out[:] = 0.0
    mask_out[:] = False
    combined_index: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    planner_metadata: list[dict[str, str]] | None = None
    planner_name = ""
    raw_lengths: list[int] = []
    for global_index, roster_row in enumerate(roster):
        token = roster_row["scenario_token"]
        source_row = sources[(token, dose)]
        stage7c_dir = Path(source_row["attempt_dir"]).resolve() / "stage7c_output"
        seq = np.load(stage7c_dir / "simulated_ego_seq.npy", mmap_mode="r")
        mask = np.load(stage7c_dir / "simulated_ego_seq_mask.npy", mmap_mode="r")
        if seq.ndim != 4 or seq.shape[:2] != (1, 1) or seq.shape[-1] != 8:
            raise RuntimeError(f"Invalid frozen trajectory tensor for {token}/{dose}: {seq.shape}")
        if mask.shape != seq.shape[:-1] or seq.shape[2] > TARGET_T:
            raise RuntimeError(f"Invalid frozen trajectory mask for {token}/{dose}: {mask.shape}")
        raw_t = int(seq.shape[2])
        raw_lengths.append(raw_t)
        seq_out[global_index, 0, :raw_t] = seq[0, 0]
        mask_out[global_index, 0, :raw_t] = mask[0, 0].astype(bool)
        index_rows = read_csv(stage7c_dir / "scenario_planner_index.csv")
        if len(index_rows) != 1:
            raise RuntimeError(f"Expected one Stage7C index row: {stage7c_dir}")
        index_row = {**index_rows[0], "scenario_index": global_index}
        if index_row.get("scenario_token") != token or index_row.get("log_name") != roster_row.get("log_name"):
            raise RuntimeError(f"Token/log mismatch in frozen Stage7C cell: {token}/{dose}")
        combined_index.append(index_row)
        current_planner_metadata = read_csv(stage7c_dir / "simulated_planner_metadata.csv")
        if len(current_planner_metadata) != 1:
            raise RuntimeError(f"Expected one planner metadata row: {stage7c_dir}")
        if planner_metadata is None:
            planner_metadata = current_planner_metadata
            planner_name = current_planner_metadata[0]["planner_name"]
        elif current_planner_metadata != planner_metadata:
            raise RuntimeError(f"Planner metadata changed within {dose}")
        source_scenario = stage7c_dir / "official_nuplan_runs" / "scenario_0"
        if not source_scenario.is_dir():
            raise FileNotFoundError(f"Missing frozen official scenario directory: {source_scenario}")
        os.symlink(source_scenario.resolve(), official_root / f"scenario_{global_index}", target_is_directory=True)
        msgpacks = sorted(source_scenario.rglob("*.msgpack.xz"))
        if len(msgpacks) != 1 or msgpacks[0].name != f"{token}.msgpack.xz":
            raise RuntimeError(f"Official msgpack identity mismatch for {token}/{dose}: {msgpacks}")
        ledger.append(
            {
                "global_scenario_index": global_index,
                "collection_order": int(roster_row["collection_order"]),
                "scenario_token": token,
                "log_name": roster_row.get("log_name", ""),
                "direction": roster_row.get("direction", ""),
                "dose": dose,
                "selected_attempt_id": source_row.get("attempt_id", ""),
                "stage7c_output_dir": str(stage7c_dir),
                "raw_timesteps": raw_t,
                "valid_timesteps": int(np.asarray(mask[0, 0], dtype=bool).sum()),
                "trajectory_sha256": sha256_file(stage7c_dir / "simulated_ego_seq.npy"),
                "mask_sha256": sha256_file(stage7c_dir / "simulated_ego_seq_mask.npy"),
                "msgpack_sha256": sha256_file(msgpacks[0]),
            }
        )
    seq_out.flush()
    mask_out.flush()
    del seq_out, mask_out
    assert planner_metadata is not None
    write_csv(output_dir / "scenario_planner_index.csv", combined_index)
    write_csv(output_dir / "simulated_planner_metadata.csv", planner_metadata)
    write_csv(output_dir / "source_cell_ledger.csv", ledger)
    write_json(
        output_dir / "simulated_ego_seq_index.json",
        {
            "scenario_axis": [str(index) for index in range(len(roster))],
            "planner_axis": ["0"],
            "planner_axis_names": [planner_name],
            "ego_state_channels": ["x", "y", "yaw", "speed", "velocity_y", "acceleration", "acceleration_y", "time_s"],
            "shape": list(shape),
            "padding_policy": "right-pad 149-step arrays to 150 with zero values and false mask only",
        },
    )
    write_json(
        output_dir / "simulation_schema.json",
        {
            "stage": "Stage7L-E frozen Stage7C view",
            "uses_official_nuplan_simulation": True,
            "pseudo_rollout": False,
            "planner_names": [planner_name],
            "simulated_ego_seq_shape": list(shape),
            "same_scenario_alignment_checked": True,
            "same_scenario_alignment_passed": True,
            "same_log_alignment_passed": True,
            "strict_nuplan_token_alignment_passed": True,
            "source_stage7c_outputs_are_symlinked_not_copied": True,
            "planner_rerun": False,
        },
    )
    write_json(
        output_dir / "warnings.json",
        {
            "warnings": [],
            "pseudo_rollout": False,
            "validation": {
                "pass": True,
                "pseudo_rollout": False,
                "official_success_count": len(roster),
                "expected_pair_count": len(roster),
                "observed_pair_count": len(roster),
            },
        },
    )
    return {
        "dose": dose,
        "scenario_count": len(roster),
        "shape": list(shape),
        "raw_timestep_values": sorted(set(raw_lengths)),
        "padding_policy": "zero_values_false_mask_right_pad_to_150",
        "scenario_order_sha256": sha256_file(output_dir / "source_cell_ledger.csv"),
        "trajectory_view_sha256": sha256_file(output_dir / "simulated_ego_seq.npy"),
        "mask_view_sha256": sha256_file(output_dir / "simulated_ego_seq_mask.npy"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-run-dir", type=Path, default=ROOT / "outputs/stage7l_d_one_time_confirmation_v1")
    parser.add_argument("--d-manifest", type=Path, default=ROOT / "docs/stage7l_d_confirmation_manifest_v1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/stage7l_c_prospective_confirmation_protocol_v1.json")
    parser.add_argument("--roster", type=Path, default=ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_roster.csv")
    parser.add_argument(
        "--pretreatment-source",
        type=Path,
        default=ROOT / "outputs/stage7l_b2_dynamic_clearance_expanded_inventory_v2_pittsburgh/pool_b_strict_development_log_disjoint_dynamic_clean.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sha256_file(args.protocol) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("Frozen Stage7L protocol SHA mismatch")
    if sha256_file(args.roster) != EXPECTED_ROSTER_SHA:
        raise RuntimeError("Frozen Stage7L roster SHA mismatch")
    if not git_contains_frozen_commit(ROOT, EXPECTED_D_COMMIT):
        raise RuntimeError(f"HEAD does not contain frozen Stage7L-D commit {EXPECTED_D_COMMIT}")
    d_manifest = read_json(args.d_manifest)
    validate_unlock(d_manifest)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    roster = read_csv(args.roster)
    if len(roster) != 80:
        raise RuntimeError(f"Expected frozen roster of 80, got {len(roster)}")
    summary_rows = read_csv(args.d_run_dir / "official_run_summary.csv")
    sources = selected_sources(roster, summary_rows)
    task_rows = build_task_mask_rows(roster, read_csv(args.pretreatment_source))
    task_mask_path = args.output_dir / "task_masks.csv"
    write_csv(task_mask_path, task_rows)
    lane_count = sum(bool(row["LAT.LANE_CHANGE"]) for row in task_rows)
    dynamics_count = sum(bool(row["LAT.DYNAMICS"]) for row in task_rows)
    if (lane_count, dynamics_count) != (80, 38):
        raise RuntimeError(f"Frozen task mask count mismatch: lane={lane_count}, dynamics={dynamics_count}")
    view_root = args.output_dir / "stage7c_views"
    view_root.mkdir()
    dose_audits = [
        build_dose_view(dose=dose, roster=roster, sources=sources, output_dir=view_root / dose)
        for dose in DOSES
    ]
    task_audit = {
        "definition_version": "stage7l_c2_pretreatment_task_masks_v1",
        "LAT.LANE_CHANGE": lane_count,
        "LAT.DYNAMICS": dynamics_count,
        "task_mask_sha256": sha256_file(task_mask_path),
        "roster_sha256": sha256_file(args.roster),
        "pretreatment_source_sha256": sha256_file(args.pretreatment_source),
        "forbidden_inputs_read": [],
    }
    write_json(args.output_dir / "task_mask_audit.json", task_audit)
    audit = {
        "schema_version": "stage7l_e_input_contract_preparation_v1",
        "status": "STAGE7L_E_INPUT_VIEWS_PREPARED_CONTEXT_BUILD_NOT_YET_VALIDATED",
        "stage7l_d_unlock_verified": True,
        "stage7l_d_commit_contained_in_head": EXPECTED_D_COMMIT,
        "stage7l_d_manifest_sha256": sha256_file(args.d_manifest),
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "official_rollouts_reused": 400,
        "planner_rerun": False,
        "checkpoint_read": False,
        "embedding_read": False,
        "bdd_or_mmd_computed": False,
        "target_contract": [80, TARGET_T, 83],
        "dose_views": dose_audits,
        "task_masks": task_audit,
        "unsafe_or_offroad_or_collision_filtering": False,
    }
    write_json(args.output_dir / "input_contract_preparation_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
