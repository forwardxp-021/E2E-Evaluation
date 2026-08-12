#!/usr/bin/env python3
"""Record an explicit Stage 6R visual semantic review bound to an overview image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SLOTS = ["front", "left_front", "left_rear", "right_front", "right_rear"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    with args.reviewed_cases_csv.open(newline="", encoding="utf-8-sig") as handle:
        cases = list(csv.DictReader(handle))
    topology = json.loads(args.topology_summary.read_text())
    if topology.get("status") != "TOPOLOGY_RECONSTRUCTION_PASS_PENDING_VISUAL_REVIEW":
        raise ValueError("topology reconstruction has not passed")
    if len(cases) != 20 or any(sum(row["slot"] == slot for row in cases) != 4 for slot in SLOTS):
        raise ValueError("visual review set must contain exactly four cases per semantic slot")
    if not args.confirm_all_cases_semantically_correct:
        raise ValueError("visual semantic pass requires explicit --confirm_all_cases_semantically_correct")
    result = {
        "schema_version": "stage6r_pilot_visual_semantic_review_v2",
        "status": "VISUAL_SEMANTIC_REVIEW_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_case_count": 20,
        "pass_count_by_slot": {slot: 4 for slot in SLOTS},
        "overview_image": str(args.overview_image.resolve()),
        "overview_image_sha256": sha256_file(args.overview_image),
        "reviewed_cases_csv_sha256": sha256_file(args.reviewed_cases_csv),
        "topology_summary_sha256": sha256_file(args.topology_summary),
        "review_statement": (
            "All frozen cases were visually inspected for ego/slot trajectory and local lane geometry; "
            "automatic reconstruction alone is not accepted as semantic evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed_cases_csv", type=Path, required=True)
    parser.add_argument("--topology_summary", type=Path, required=True)
    parser.add_argument("--overview_image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm_all_cases_semantically_correct", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
