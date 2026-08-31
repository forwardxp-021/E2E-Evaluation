#!/usr/bin/env python3
"""Deterministically construct the V2.2 planner from a frozen B2.8-R1 row.

This is execution wiring, not a selector.  It accepts only an exact run ID
already present in the immutable schedule and fails before simulator start for
every missing, ambiguous, or inconsistent binding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from tools.r1_official_technical_smoke_planner_v2_2 import R1OfficialTechnicalSmokePlannerV2_2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_run_binding(binding_manifest_path: str | Path, run_id: str) -> Dict[str, Any]:
    path = Path(binding_manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"B2.8-R1 binding manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "r1_b2_8_r1_execution_bindings_manifest_v1.0":
        raise ValueError("B2.8-R1 binding manifest schema mismatch")
    rows = [row for row in payload.get("frozen_run_bindings", []) if str(row.get("run_id")) == str(run_id)]
    if len(rows) != 1:
        raise ValueError(f"FROZEN_RUN_ID_MATCH_COUNT_MUST_EQUAL_ONE:{run_id}:observed={len(rows)}")
    row = dict(rows[0])
    required = ("run_id", "pair_id", "family", "scenario_token", "log_id", "arm", "future_roster_row")
    if any(key not in row for key in required):
        raise ValueError("FROZEN_RUN_BINDING_REQUIRED_FIELD_MISSING")
    roster = dict(row["future_roster_row"])
    if (str(row["family"]), str(row["scenario_token"]), str(row["log_id"])) != (str(roster.get("family")), str(roster.get("scenario_token")), str(roster.get("log_id"))):
        raise ValueError("FROZEN_SCHEDULE_ROSTER_IDENTITY_MISMATCH")
    arms = {str(value) for value in roster.get("arms", [])}
    if str(row["arm"]) not in arms:
        raise ValueError("FROZEN_SCHEDULE_ARM_NOT_IN_ROSTER_FAMILY_ARMS")
    return row


def build_planner_from_frozen_binding(binding_manifest_path: str, run_id: str, trace_dir: str) -> R1OfficialTechnicalSmokePlannerV2_2:
    """Hydra target: no defaults, substitutions, or fallback identities."""
    if not str(trace_dir).strip():
        raise ValueError("REALIZED_TRACE_DIRECTORY_MUST_BE_EXPLICIT")
    binding = load_frozen_run_binding(binding_manifest_path, run_id)
    return R1OfficialTechnicalSmokePlannerV2_2(
        future_roster_row=binding["future_roster_row"],
        runtime_family=str(binding["family"]),
        smoke_arm=str(binding["arm"]),
        trace_dir=str(trace_dir),
    )


__all__ = ["build_planner_from_frozen_binding", "load_frozen_run_binding", "sha256_file"]
