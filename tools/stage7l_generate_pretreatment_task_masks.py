#!/usr/bin/env python3
"""Generate frozen Stage7L task masks from pre-treatment metadata only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
TASK_MASK_DEFINITION_VERSION = "stage7l_c2_pretreatment_task_masks_v1"
LAT_DYNAMICS_OFFICIAL_TYPES = frozenset(
    {
        "high_lateral_acceleration",
        "high_magnitude_speed",
        "medium_magnitude_speed",
    }
)


def read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_official_types(raw: str, *, scenario_token: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"scenario {scenario_token} has invalid official_scenario_types_json"
        ) from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"scenario {scenario_token} official_scenario_types_json must be a string list"
        )
    return tuple(sorted(set(value)))


def build_task_mask_rows(
    roster_rows: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
) -> list[Dict[str, Any]]:
    """Build masks without rollout, treatment, embedding, BDD or MMD inputs."""
    if not roster_rows:
        raise ValueError("frozen roster is empty")
    by_token: Dict[str, Mapping[str, str]] = {}
    for row in source_rows:
        token = str(row.get("scenario_token", ""))
        if not token:
            raise ValueError("source metadata contains an empty scenario_token")
        if token in by_token:
            raise ValueError(f"source metadata contains duplicate scenario_token: {token}")
        by_token[token] = row

    output: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for roster_position, roster_row in enumerate(roster_rows, start=1):
        token = str(roster_row.get("scenario_token", ""))
        if not token:
            raise ValueError(f"roster row {roster_position} has empty scenario_token")
        if token in seen:
            raise ValueError(f"frozen roster contains duplicate scenario_token: {token}")
        seen.add(token)
        source = by_token.get(token)
        if source is None:
            raise ValueError(f"frozen roster token missing from pre-treatment source: {token}")
        official_types = parse_official_types(
            str(source.get("official_scenario_types_json", "")),
            scenario_token=token,
        )
        output.append(
            {
                "roster_position": roster_position,
                "scenario_token": token,
                "log_name": str(roster_row.get("log_name", "")),
                "LAT.LANE_CHANGE": True,
                "LAT.DYNAMICS": bool(
                    LAT_DYNAMICS_OFFICIAL_TYPES.intersection(official_types)
                ),
                "official_scenario_types_json": json.dumps(
                    official_types, ensure_ascii=False, separators=(",", ":")
                ),
                "task_mask_definition_version": TASK_MASK_DEFINITION_VERSION,
                "selection_timing": "pre_treatment",
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty task mask")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_roster.csv",
    )
    parser.add_argument(
        "--pretreatment-source",
        type=Path,
        default=ROOT / "outputs/stage7l_b2_dynamic_clearance_expanded_inventory_v2_pittsburgh/pool_b_strict_development_log_disjoint_dynamic_clean.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_task_mask_rows(read_csv(args.roster), read_csv(args.pretreatment_source))
    write_csv(args.output_csv, rows)
    summary = {
        "task_mask_definition_version": TASK_MASK_DEFINITION_VERSION,
        "n_roster": len(rows),
        "n_lat_lane_change": sum(bool(row["LAT.LANE_CHANGE"]) for row in rows),
        "n_lat_dynamics": sum(bool(row["LAT.DYNAMICS"]) for row in rows),
        "roster_sha256": sha256_file(args.roster),
        "pretreatment_source_sha256": sha256_file(args.pretreatment_source),
        "output_csv": str(args.output_csv.resolve()),
        "forbidden_inputs_read": [],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
