#!/usr/bin/env python3
"""Combine Stage 6R automated, reconstruction, and visual pilot gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    automated = json.loads(args.automated_audit.read_text())
    topology = json.loads(args.topology_reconstruction.read_text())
    visual = json.loads(args.visual_review.read_text())
    if automated.get("status") != "AUTOMATED_PASS_PENDING_TOPOLOGY_AND_VISUAL_REVIEW":
        raise ValueError("Stage 6R automated pilot gate did not pass")
    if topology.get("status") != "TOPOLOGY_RECONSTRUCTION_PASS_PENDING_VISUAL_REVIEW":
        raise ValueError("Stage 6R topology reconstruction did not pass")
    if visual.get("status") != "VISUAL_SEMANTIC_REVIEW_PASS":
        raise ValueError("Stage 6R visual semantic review did not pass")
    if not all(automated.get("gate_checks", {}).values()):
        raise ValueError("Stage 6R automated pilot has a failed gate check")
    if visual.get("reviewed_case_count") != 20 or set(visual.get("pass_count_by_slot", {}).values()) != {4}:
        raise ValueError("Stage 6R visual review is not balanced 4 cases per semantic slot")
    decision = {
        "schema_version": "stage6r_dynamic_builder_v2_pilot_decision_v1",
        "status": "PILOT_PASS_FULL51_REBUILD_ALLOWED_NOT_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "automated_audit_sha256": sha256_file(args.automated_audit),
        "topology_reconstruction_sha256": sha256_file(args.topology_reconstruction),
        "visual_review_sha256": sha256_file(args.visual_review),
        "automated_gate_checks": automated["gate_checks"],
        "visual_reviewed_case_count": visual["reviewed_case_count"],
        "visual_pass_count_by_slot": visual["pass_count_by_slot"],
        "topology_failure_count": topology["topology_failure_count"],
        "reconstruction_mismatch_count": topology["reconstruction_mismatch_count"],
        "full51_rebuild_authorized": True,
        "checkpoint_training_authorized": False,
        "waymo_expansion_authorized": False,
        "stage6o_v1_modification_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--automated_audit", type=Path, required=True)
    parser.add_argument("--topology_reconstruction", type=Path, required=True)
    parser.add_argument("--visual_review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
