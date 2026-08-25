#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import (  # noqa: E402
    audit_stage7c_output,
    current_planner_fingerprints,
    sha256_file,
)


SCHEMA_VERSION = "stage6j_pure_longitudinal_view_v1"
EXPECTED_PLANNERS = [
    "pdm_closed_assertive_longitudinal_v1",
    "pdm_closed_conservative_longitudinal_v1",
]
EXPECTED_TASK_COUNTS = {
    "following_interaction": 60,
    "longitudinal_high_motion": 56,
    "stop_go_control": 67,
}
LEDGER_FIELDS = [
    "global_scenario_index",
    "collection_order",
    "source_global_scenario_index",
    "task",
    "source_task",
    "scenario_type",
    "log_name",
    "scenario_token",
    "db_file",
    "attempt",
    "stage7c_output_dir",
    "simulated_ego_seq_sha256",
    "simulated_ego_seq_mask_sha256",
    "scenario_planner_index_sha256",
    "official_msgpack_count",
    "official_msgpack_size_bytes",
    "stage7c_audit_pass",
]


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def planner_metadata_signature(rows: Sequence[Mapping[str, str]]) -> str:
    normalized = [
        dict(sorted(row.items()))
        for row in sorted(rows, key=lambda row: int(row["planner_id"]))
    ]
    return canonical_hash(normalized)


def validate_locked_sources(
    freeze_manifest_path: Path,
    locked_scenarios_path: Path,
    batch_manifest_path: Path,
    batch_state_path: Path,
    batch_status_path: Path,
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    freeze = read_json(freeze_manifest_path)
    batch = read_json(batch_manifest_path)
    state = read_json(batch_state_path)
    locked = read_csv(locked_scenarios_path)
    statuses = read_csv(batch_status_path)

    if freeze.get("status") != "FROZEN_BEFORE_PURE_LONGITUDINAL_ROLLOUTS":
        raise ValueError("freeze manifest is not frozen before pure-longitudinal rollouts")
    if freeze.get("embedding_or_bdd_read") is not False:
        raise ValueError("freeze manifest does not certify embedding_or_bdd_read=false")
    if batch.get("schema_version") != "stage6j_pure_longitudinal_batch_v1":
        raise ValueError("unexpected Stage 6J batch schema_version")
    if batch.get("planners") != EXPECTED_PLANNERS:
        raise ValueError(f"unexpected planner order: {batch.get('planners')}")
    expected_fingerprints = current_planner_fingerprints(EXPECTED_PLANNERS)
    if batch.get("planner_fingerprints") != expected_fingerprints:
        raise ValueError("current planner parameters differ from the batch manifest")
    if batch.get("locked_scenarios_sha256") != sha256_file(locked_scenarios_path):
        raise ValueError("locked scenario CSV changed after the batch was frozen")
    if batch.get("freeze_manifest_sha256") != sha256_file(freeze_manifest_path):
        raise ValueError("freeze manifest changed after the batch was frozen")
    if batch.get("full_embedding_or_bdd_read") is not False:
        raise ValueError("batch manifest does not certify full_embedding_or_bdd_read=false")
    counts = state.get("counts", {})
    succeeded = counts.get("SUCCEEDED", counts.get("succeeded"))
    failed = counts.get("FAILED_REVIEW_REQUIRED", counts.get("failed"))
    pending = counts.get("PENDING", counts.get("pending"))
    if succeeded != 183 or failed != 0 or pending != 0:
        raise ValueError(f"Stage 6J batch is not complete: state={state}")
    if len(locked) != 183 or len(statuses) != 183:
        raise ValueError(
            f"expected 183 locked/status rows, got locked={len(locked)}, status={len(statuses)}"
        )
    task_counts = dict(Counter(row["task"] for row in locked))
    if task_counts != EXPECTED_TASK_COUNTS:
        raise ValueError(f"unexpected locked task composition: {task_counts}")

    locked_by_order = {int(row["collection_order"]): row for row in locked}
    status_by_order = {int(row["collection_order"]): row for row in statuses}
    expected_orders = set(range(1, 184))
    if set(locked_by_order) != expected_orders or set(status_by_order) != expected_orders:
        raise ValueError("collection_order must be unique and contiguous from 1 to 183")

    rows: List[Dict[str, str]] = []
    for order in range(1, 184):
        frozen = locked_by_order[order]
        status = status_by_order[order]
        if status.get("status") != "SUCCEEDED":
            raise ValueError(f"collection_order={order} is not SUCCEEDED")
        for field in (
            "task",
            "source_task",
            "scenario_type",
            "log_name",
            "scenario_token",
            "db_file",
        ):
            if status.get(field) != frozen.get(field):
                raise ValueError(
                    f"collection_order={order} field {field} differs between freeze and status"
                )
        output_dir = Path(status.get("stage7c_output_dir", "")).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"collection_order={order} Stage7C output is missing: {output_dir}"
            )
        rows.append({**frozen, **status, "stage7c_output_dir": str(output_dir)})

    audit = {
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "locked_scenarios_sha256": sha256_file(locked_scenarios_path),
        "batch_manifest_sha256": sha256_file(batch_manifest_path),
        "batch_state_sha256": sha256_file(batch_state_path),
        "batch_status_sha256": sha256_file(batch_status_path),
        "planner_parameter_fingerprints": expected_fingerprints,
        "scenario_count": len(rows),
        "task_counts": task_counts,
        "pass": True,
    }
    return rows, audit


def prepare_view(
    rows: List[Dict[str, str]],
    output_dir: Path,
    expected_planners: Sequence[str] = EXPECTED_PLANNERS,
    ledger_filename: str = "stage6j_scenario_ledger.csv",
    ledger_fields: Sequence[str] = LEDGER_FIELDS,
    stage_label: str = "6J unified pure-longitudinal Stage7C view",
    schema_version: str = SCHEMA_VERSION,
) -> Dict[str, Any]:
    audits: List[Dict[str, Any]] = []
    max_timesteps = 0
    planner_metadata: List[Dict[str, str]] | None = None
    planner_signature = ""

    for global_index, row in enumerate(rows):
        source = Path(row["stage7c_output_dir"])
        audit = audit_stage7c_output(source, list(expected_planners), row)
        if not audit["pass"]:
            raise ValueError(
                f"Stage7C re-audit failed for order={row['collection_order']} "
                f"token={row['scenario_token']}: {audit}"
            )
        seq = np.load(source / "simulated_ego_seq.npy", mmap_mode="r")
        mask = np.load(source / "simulated_ego_seq_mask.npy", mmap_mode="r")
        if seq.ndim != 4 or seq.shape[:2] != (1, 2) or seq.shape[-1] != 8:
            raise ValueError(f"unexpected Stage7C seq shape for {source}: {seq.shape}")
        if mask.shape != seq.shape[:-1]:
            raise ValueError(
                f"Stage7C mask mismatch for {source}: seq={seq.shape}, mask={mask.shape}"
            )
        max_timesteps = max(max_timesteps, int(seq.shape[2]))
        current_metadata = read_csv(source / "simulated_planner_metadata.csv")
        signature = planner_metadata_signature(current_metadata)
        if planner_metadata is None:
            planner_metadata, planner_signature = current_metadata, signature
        elif signature != planner_signature:
            raise ValueError(f"planner metadata differs across Stage7C outputs: {source}")
        msgpacks = sorted(
            (source / "official_nuplan_runs" / "scenario_0").rglob("*.msgpack.xz")
        )
        if len(msgpacks) != 2:
            raise ValueError(f"expected two official msgpacks for {source}, got {len(msgpacks)}")
        audits.append(
            {
                **row,
                "global_scenario_index": global_index,
                "stage7c_audit": audit,
                "msgpacks": msgpacks,
            }
        )

    if planner_metadata is None:
        raise ValueError("no successful Stage7C output was available")
    output_dir.mkdir(parents=True)
    official_root = output_dir / "official_nuplan_runs"
    official_root.mkdir()
    shape = (len(rows), 2, max_timesteps, 8)
    seq_out = np.lib.format.open_memmap(
        output_dir / "simulated_ego_seq.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    mask_out = np.lib.format.open_memmap(
        output_dir / "simulated_ego_seq_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=shape[:-1],
    )
    seq_out[:] = 0.0
    mask_out[:] = False
    combined_index: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []

    for item in audits:
        global_index = int(item["global_scenario_index"])
        source = Path(item["stage7c_output_dir"])
        seq = np.load(source / "simulated_ego_seq.npy", mmap_mode="r")
        mask = np.load(source / "simulated_ego_seq_mask.npy", mmap_mode="r")
        timesteps = int(seq.shape[2])
        seq_out[global_index, :, :timesteps] = seq[0]
        mask_out[global_index, :, :timesteps] = mask[0]
        for source_row in read_csv(source / "scenario_planner_index.csv"):
            combined_index.append({**source_row, "scenario_index": global_index})
        source_official = source / "official_nuplan_runs" / "scenario_0"
        os.symlink(
            source_official.resolve(),
            official_root / f"scenario_{global_index}",
            target_is_directory=True,
        )
        msgpacks: List[Path] = item["msgpacks"]
        ledger.append(
            {
                "global_scenario_index": global_index,
                "collection_order": item["collection_order"],
                "source_collection_order": item.get("source_collection_order", item["collection_order"]),
                "dose": item.get("dose", ""),
                "dose_label": item.get("dose_label", ""),
                "planner_a": item.get("planner_a", expected_planners[0]),
                "planner_b": item.get("planner_b", expected_planners[1]),
                "source_global_scenario_index": item["source_global_scenario_index"],
                "task": item["task"],
                "source_task": item["source_task"],
                "scenario_type": item["scenario_type"],
                "log_name": item["log_name"],
                "scenario_token": item["scenario_token"],
                "db_file": item["db_file"],
                "attempt": item["attempt"],
                "stage7c_output_dir": str(source),
                "simulated_ego_seq_sha256": sha256_file(source / "simulated_ego_seq.npy"),
                "simulated_ego_seq_mask_sha256": sha256_file(
                    source / "simulated_ego_seq_mask.npy"
                ),
                "scenario_planner_index_sha256": sha256_file(
                    source / "scenario_planner_index.csv"
                ),
                "official_msgpack_count": len(msgpacks),
                "official_msgpack_size_bytes": sum(path.stat().st_size for path in msgpacks),
                "stage7c_audit_pass": True,
            }
        )

    seq_out.flush()
    mask_out.flush()
    del seq_out, mask_out

    write_csv(output_dir / "scenario_planner_index.csv", combined_index, list(combined_index[0]))
    write_csv(
        output_dir / "simulated_planner_metadata.csv",
        planner_metadata,
        list(planner_metadata[0]),
    )
    write_csv(output_dir / ledger_filename, ledger, ledger_fields)
    write_json(
        output_dir / "simulated_ego_seq_index.json",
        {
            "scenario_axis": [str(index) for index in range(len(rows))],
            "planner_axis": ["0", "1"],
            "planner_axis_names": list(expected_planners),
            "ego_state_channels": [
                "x",
                "y",
                "yaw",
                "speed",
                "accel",
                "yaw_rate",
                "steering_angle",
                "time_s",
            ],
            "shape": list(shape),
            "padding_policy": (
                f"right-pad arrays shorter than {max_timesteps} timesteps with zero "
                "values and false mask only"
            ),
        },
    )
    write_json(
        output_dir / "simulation_schema.json",
        {
            "stage": stage_label,
            "schema_version": schema_version,
            "uses_official_nuplan_simulation": True,
            "pseudo_rollout": False,
            "planner_names": list(expected_planners),
            "simulated_ego_seq_shape": list(shape),
            "same_log_alignment_passed": True,
            "strict_nuplan_token_alignment_passed": True,
            "source_stage7c_outputs_are_symlinked_not_copied": True,
        },
    )
    mask_total = int(
        np.load(output_dir / "simulated_ego_seq_mask.npy", mmap_mode="r").sum()
    )
    write_json(
        output_dir / "warnings.json",
        {
            "warnings": [],
            "pseudo_rollout": False,
            "validation": {
                "pass": True,
                "pseudo_rollout": False,
                "official_success_count": len(rows) * 2,
                "trajectory_rows": mask_total,
                "tensor_validation": {
                    "passed": True,
                    "expected_pair_count": len(rows) * 2,
                    "observed_pair_count": len(rows) * 2,
                    "missing_pair_count": 0,
                },
            },
        },
    )
    return {
        "scenario_count": len(rows),
        "rollout_count": len(rows) * 2,
        "shape": list(shape),
        "trajectory_rows": mask_total,
        "planner_metadata_canonical_sha256": planner_signature,
        "ledger_sha256": sha256_file(output_dir / ledger_filename),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "distinct_log_count": len({row["log_name"] for row in rows}),
        "reaudited_scenario_count": len(audits),
        "reaudit_failure_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-audit and consolidate all 183 Stage 6J pure-longitudinal scenario pairs."
        )
    )
    base = Path("outputs")
    freeze_dir = base / "stage6j_pure_longitudinal_freeze_v1"
    batch_dir = base / "stage6j_pure_longitudinal_batch_v1"
    parser.add_argument(
        "--freeze_manifest",
        type=Path,
        default=freeze_dir / "stage6j_freeze_manifest.json",
    )
    parser.add_argument(
        "--locked_scenarios_csv",
        type=Path,
        default=freeze_dir / "stage6j_locked_scenarios.csv",
    )
    parser.add_argument(
        "--batch_manifest", type=Path, default=batch_dir / "batch_manifest.json"
    )
    parser.add_argument("--batch_state", type=Path, default=batch_dir / "batch_state.json")
    parser.add_argument(
        "--batch_status_csv",
        type=Path,
        default=batch_dir / "batch_scenario_status.csv",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output_dir already exists: {args.output_dir}; use --overwrite"
            )
        shutil.rmtree(args.output_dir)
    rows, source_audit = validate_locked_sources(
        args.freeze_manifest,
        args.locked_scenarios_csv,
        args.batch_manifest,
        args.batch_state,
        args.batch_status_csv,
    )
    try:
        view = prepare_view(rows, args.output_dir)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PURE_LONGITUDINAL_VIEW_READY",
            "full_embedding_or_bdd_read": False,
            "selection_changed_after_outcome_review": False,
            "locked_source_audit": source_audit,
            "preparation_tool_sha256": sha256_file(Path(__file__).resolve()),
            "input_files": {
                key: {
                    "path": str(getattr(args, key).resolve()),
                    "sha256": sha256_file(getattr(args, key)),
                }
                for key in (
                    "freeze_manifest",
                    "locked_scenarios_csv",
                    "batch_manifest",
                    "batch_state",
                    "batch_status_csv",
                )
            },
            **view,
        }
        write_json(args.output_dir / "stage6j_view_summary.json", summary)
    except Exception:
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        raise
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
