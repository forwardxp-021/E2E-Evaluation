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
    EXPECTED_PLANNERS,
    audit_stage7c_output,
    current_planner_fingerprints,
    sha256_file,
)


SCHEMA_VERSION = "stage7_m6_5_locked_confirmation_view_v1"
EXPECTED_TASK_COUNTS = {
    "following_interaction": 60,
    "lane_change": 60,
    "stop_go_control": 67,
    "high_motion_dynamics": 60,
    "dense_or_vulnerable_interaction": 63,
}
LEDGER_FIELDS = [
    "global_scenario_index",
    "source_group",
    "source_collection_order",
    "task",
    "scenario_type",
    "log_name",
    "scenario_token",
    "db_file",
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
        return [{key: str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def index_by_token(rows: Sequence[Mapping[str, str]], label: str) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        token = row.get("scenario_token", "")
        if not token or token in result:
            raise ValueError(f"{label} has missing or duplicate scenario_token={token!r}")
        result[token] = dict(row)
    return result


def successful_status_rows(path: Path) -> List[Dict[str, str]]:
    rows = read_csv(path)
    return [row for row in rows if row.get("status") == "SUCCEEDED"]


def assemble_sources(args: argparse.Namespace) -> List[Dict[str, str]]:
    primary = index_by_token(read_csv(args.primary_csv), "locked primary")
    reserve = index_by_token(read_csv(args.reserve_csv), "locked reserve")
    supplement = index_by_token(read_csv(args.supplement_primary_csv), "supplement primary")
    specs = [
        ("m6_4b_primary", args.batch_status_csv, primary),
        ("m6_4c_quoted_primary", args.quoted_status_csv, primary),
        ("m6_4c_frozen_reserve", args.reserve_status_csv, reserve),
        ("m6_4d_supplement_primary", args.supplement_status_csv, supplement),
    ]
    rows: List[Dict[str, str]] = []
    for source_group, status_path, frozen_lookup in specs:
        for status in successful_status_rows(status_path):
            token = status.get("scenario_token", "")
            if token not in frozen_lookup:
                raise ValueError(f"{source_group} successful token is not in its frozen source: {token}")
            frozen = frozen_lookup[token]
            for key in ("task", "log_name"):
                if status.get(key) and status.get(key) != frozen.get(key):
                    raise ValueError(f"{source_group} {token} {key} differs from frozen source")
            output_dir = Path(status.get("stage7c_output_dir", "")).resolve()
            rows.append({
                **frozen,
                "source_group": source_group,
                "source_collection_order": status.get("collection_order", status.get("plan_order", "")),
                "stage7c_output_dir": str(output_dir),
            })
    tokens = [row["scenario_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        duplicates = sorted(token for token, count in Counter(tokens).items() if count > 1)
        raise ValueError(f"successful source groups overlap: {duplicates}")
    rows.sort(key=lambda row: (row["task"], row["source_group"], int(row["source_collection_order"])))
    return rows


def validate_development_disjoint(rows: Sequence[Mapping[str, str]], metadata_csv: Path) -> Dict[str, Any]:
    development = read_csv(metadata_csv)
    dev_tokens = {
        row.get("scenario_token") or row.get("scenario_id") or row.get("scene_token")
        for row in development
    }
    dev_logs = {row.get("log_name", "") for row in development if row.get("log_name")}
    tokens = {row["scenario_token"] for row in rows}
    logs = {row["log_name"] for row in rows}
    token_overlap = sorted(tokens & dev_tokens)
    log_overlap = sorted(logs & dev_logs)
    result = {
        "development_metadata_csv": str(metadata_csv.resolve()),
        "development_metadata_sha256": sha256_file(metadata_csv),
        "scenario_token_overlap_count": len(token_overlap),
        "log_name_overlap_count": len(log_overlap),
        "scenario_token_overlap": token_overlap,
        "log_name_overlap": log_overlap,
        "pass": not token_overlap and not log_overlap,
    }
    if not result["pass"]:
        raise ValueError(f"confirmation/development overlap detected: {result}")
    return result


def planner_metadata_signature(rows: Sequence[Mapping[str, str]]) -> str:
    normalized = [dict(sorted(row.items())) for row in sorted(rows, key=lambda row: int(row["planner_id"]))]
    return canonical_hash(normalized)


def prepare_view(rows: List[Dict[str, str]], output_dir: Path) -> Dict[str, Any]:
    audits: List[Dict[str, Any]] = []
    max_timesteps = 0
    planner_metadata: List[Dict[str, str]] | None = None
    planner_signature = ""
    for global_index, row in enumerate(rows):
        source = Path(row["stage7c_output_dir"])
        audit = audit_stage7c_output(source, EXPECTED_PLANNERS, row)
        if not audit["pass"]:
            raise ValueError(f"Stage7C re-audit failed for {row['scenario_token']}: {audit}")
        seq = np.load(source / "simulated_ego_seq.npy", mmap_mode="r")
        mask = np.load(source / "simulated_ego_seq_mask.npy", mmap_mode="r")
        if seq.ndim != 4 or seq.shape[:2] != (1, len(EXPECTED_PLANNERS)) or seq.shape[-1] != 8:
            raise ValueError(f"unexpected Stage7C seq shape for {source}: {seq.shape}")
        if mask.shape != seq.shape[:-1]:
            raise ValueError(f"Stage7C mask mismatch for {source}: seq={seq.shape}, mask={mask.shape}")
        max_timesteps = max(max_timesteps, int(seq.shape[2]))
        current_metadata = read_csv(source / "simulated_planner_metadata.csv")
        signature = planner_metadata_signature(current_metadata)
        if planner_metadata is None:
            planner_metadata, planner_signature = current_metadata, signature
        elif signature != planner_signature:
            raise ValueError(f"planner metadata differs across successful outputs: {source}")
        msgpacks = sorted((source / "official_nuplan_runs" / "scenario_0").rglob("*.msgpack.xz"))
        if len(msgpacks) != len(EXPECTED_PLANNERS):
            raise ValueError(f"expected two official msgpacks for {source}, got {len(msgpacks)}")
        audits.append({
            **row,
            "global_scenario_index": global_index,
            "stage7c_audit": audit,
            "seq_shape": list(seq.shape),
            "mask_shape": list(mask.shape),
            "msgpacks": msgpacks,
        })

    assert planner_metadata is not None
    output_dir.mkdir(parents=True)
    official_root = output_dir / "official_nuplan_runs"
    official_root.mkdir()
    shape = (len(rows), len(EXPECTED_PLANNERS), max_timesteps, 8)
    seq_out = np.lib.format.open_memmap(output_dir / "simulated_ego_seq.npy", mode="w+", dtype=np.float32, shape=shape)
    mask_out = np.lib.format.open_memmap(output_dir / "simulated_ego_seq_mask.npy", mode="w+", dtype=np.bool_, shape=shape[:-1])
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
        target = source / "official_nuplan_runs" / "scenario_0"
        os.symlink(target.resolve(), official_root / f"scenario_{global_index}", target_is_directory=True)
        msgpacks: List[Path] = item["msgpacks"]
        ledger.append({
            "global_scenario_index": global_index,
            "source_group": item["source_group"],
            "source_collection_order": item["source_collection_order"],
            "task": item["task"],
            "scenario_type": item["scenario_type"],
            "log_name": item["log_name"],
            "scenario_token": item["scenario_token"],
            "db_file": item["db_file"],
            "stage7c_output_dir": str(source),
            "simulated_ego_seq_sha256": sha256_file(source / "simulated_ego_seq.npy"),
            "simulated_ego_seq_mask_sha256": sha256_file(source / "simulated_ego_seq_mask.npy"),
            "scenario_planner_index_sha256": sha256_file(source / "scenario_planner_index.csv"),
            "official_msgpack_count": len(msgpacks),
            "official_msgpack_size_bytes": sum(path.stat().st_size for path in msgpacks),
            "stage7c_audit_pass": True,
        })
    seq_out.flush()
    mask_out.flush()
    del seq_out, mask_out

    index_fields = list(combined_index[0])
    write_csv(output_dir / "scenario_planner_index.csv", combined_index, index_fields)
    write_csv(output_dir / "simulated_planner_metadata.csv", planner_metadata, list(planner_metadata[0]))
    write_csv(output_dir / "confirmation_scenario_ledger.csv", ledger, LEDGER_FIELDS)
    write_json(output_dir / "simulated_ego_seq_index.json", {
        "scenario_axis": [str(index) for index in range(len(rows))],
        "planner_axis": [str(index) for index in range(len(EXPECTED_PLANNERS))],
        "planner_axis_names": EXPECTED_PLANNERS,
        "ego_state_channels": ["x", "y", "yaw", "speed", "accel", "yaw_rate", "steering_angle", "time_s"],
        "shape": list(shape),
        "padding_policy": "right-pad shorter 149-step arrays to 150 with zero values and false mask only",
    })
    write_json(output_dir / "simulation_schema.json", {
        "stage": "7 M6.5 unified confirmatory Stage7C view",
        "schema_version": SCHEMA_VERSION,
        "uses_official_nuplan_simulation": True,
        "pseudo_rollout": False,
        "planner_names": EXPECTED_PLANNERS,
        "simulated_ego_seq_shape": list(shape),
        "same_log_alignment_passed": True,
        "strict_nuplan_token_alignment_passed": True,
        "source_stage7c_outputs_are_symlinked_not_copied": True,
    })
    write_json(output_dir / "warnings.json", {
        "warnings": [],
        "pseudo_rollout": False,
        "validation": {
            "pass": True,
            "pseudo_rollout": False,
            "official_success_count": len(rows) * len(EXPECTED_PLANNERS),
            "trajectory_rows": int(np.load(output_dir / "simulated_ego_seq_mask.npy", mmap_mode="r").sum()),
            "tensor_validation": {
                "passed": True,
                "expected_pair_count": len(rows) * len(EXPECTED_PLANNERS),
                "observed_pair_count": len(rows) * len(EXPECTED_PLANNERS),
                "missing_pair_count": 0,
            },
        },
    })
    return {
        "scenario_count": len(rows),
        "row_count": len(rows) * len(EXPECTED_PLANNERS),
        "shape": list(shape),
        "planner_metadata_canonical_sha256": planner_signature,
        "ledger_sha256": sha256_file(output_dir / "confirmation_scenario_ledger.csv"),
        "source_group_counts": dict(Counter(row["source_group"] for row in rows)),
        "task_counts": dict(Counter(row["task"] for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the outcome-locked 310-pair M6.5 confirmation Stage7C view.")
    base = Path("outputs")
    parser.add_argument("--batch_status_csv", type=Path, default=base / "stage7_m6_4b_locked_batch_mac_v2/batch_scenario_status.csv")
    parser.add_argument("--quoted_status_csv", type=Path, default=base / "stage7_m6_4c_quoted_primary_recovery_mac_v1/recovery_status.csv")
    parser.add_argument("--reserve_status_csv", type=Path, default=base / "stage7_m6_4c_frozen_reserve_recovery_mac_v1/recovery_status.csv")
    parser.add_argument("--supplement_status_csv", type=Path, default=base / "stage7_m6_4d_high_motion_supplement_primary_mac_v1/supplement_status.csv")
    parser.add_argument("--primary_csv", type=Path, default=base / "stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_primary_collection.csv")
    parser.add_argument("--reserve_csv", type=Path, default=base / "stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_reserve_collection.csv")
    parser.add_argument("--supplement_primary_csv", type=Path, default=base / "stage7_m6_4d_high_motion_supplement_freeze_v1/m6_4d_locked_primary_collection.csv")
    parser.add_argument("--development_metadata_csv", type=Path, default=base / "stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    rows = assemble_sources(args)
    counts = dict(Counter(row["task"] for row in rows))
    if len(rows) != 310 or counts != EXPECTED_TASK_COUNTS:
        raise ValueError(f"locked confirmation composition mismatch: n={len(rows)}, tasks={counts}")
    disjoint = validate_development_disjoint(rows, args.development_metadata_csv)
    try:
        view = prepare_view(rows, args.output_dir)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "LOCKED_CONFIRMATION_VIEW_READY",
            "selection_is_outcome_blind_and_unchanged": True,
            "quality_gates_are_sensitivity_only": True,
            "expected_task_counts": EXPECTED_TASK_COUNTS,
            "development_disjoint_audit": disjoint,
            "planner_parameter_fingerprints": current_planner_fingerprints(EXPECTED_PLANNERS),
            "preparation_tool_sha256": sha256_file(Path(__file__).resolve()),
            "input_files": {
                key: {"path": str(getattr(args, key).resolve()), "sha256": sha256_file(getattr(args, key))}
                for key in (
                    "batch_status_csv", "quoted_status_csv", "reserve_status_csv", "supplement_status_csv",
                    "primary_csv", "reserve_csv", "supplement_primary_csv", "development_metadata_csv",
                )
            },
            **view,
        }
        write_json(args.output_dir / "m6_5_confirmation_view_summary.json", summary)
    except Exception:
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        raise
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
