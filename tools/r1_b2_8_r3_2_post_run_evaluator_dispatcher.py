#!/usr/bin/env python3
"""R3.2 pair dispatcher: HLC clearance is inapplicable to TSB by contract."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_1_official_safety_adapter import adapt_pair_official_safety  # noqa: E402
from tools.r1_official_technical_smoke_evaluator_v2_1 import R1OfficialTechnicalSmokeEvaluatorV2_1  # noqa: E402

TRACE_FILENAME = "realized_current_ego.jsonl"
DISPATCHER_SCHEMA_VERSION = "r1_b2_8_r3_2_post_run_evaluator_dispatcher_v1.0"


def _read_trace(run_dir: str | Path) -> list[Mapping[str, Any]]:
    path = Path(run_dir) / "trace" / TRACE_FILENAME
    if not path.is_file(): raise FileNotFoundError(f"REALIZED_PRIMARY_TRACE_MISSING:{path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("primary_measurement_source") != "REALIZED_CURRENT_EGO" for row in rows): raise ValueError("REALIZED_PRIMARY_TRACE_SOURCE_MISMATCH")
    return rows


def _require_pair_binding(binding: Mapping[str, Any]) -> None:
    required = ("pair_id", "family", "baseline_context", "treatment_context")
    missing = [key for key in required if key not in binding]
    if missing: raise ValueError(f"FROZEN_PAIR_BINDING_MISSING:{','.join(missing)}")
    if binding["family"] == "R-HLC":
        required_hlc = ("pretreatment_clearance", "source_reference_xy", "target_reference_xy", "native_route_reference_xy", "native_route_reference_source")
        missing_hlc = [key for key in required_hlc if key not in binding]
        if missing_hlc: raise ValueError(f"FROZEN_HLC_PAIR_BINDING_MISSING:{','.join(missing_hlc)}")
        if binding["pretreatment_clearance"].get("pretreatment_only") is not True: raise ValueError("POSTHOC_CLEARANCE_RECALCULATION_FORBIDDEN")
    elif binding["family"] == "R-TSB":
        if binding.get("pretreatment_clearance") is not None: raise ValueError("TSB_HLC_CLEARANCE_MUST_BE_NONE")
    else:
        raise ValueError("UNKNOWN_FROZEN_FAMILY")


def evaluate_frozen_pair(*, pair_binding: Mapping[str, Any], baseline_run_dir: str | Path, treatment_run_dir: str | Path) -> dict[str, Any]:
    _require_pair_binding(pair_binding)
    safety = adapt_pair_official_safety(baseline_run_dir, treatment_run_dir)
    result = R1OfficialTechnicalSmokeEvaluatorV2_1().evaluate_pair(
        family=str(pair_binding["family"]), baseline_trace_rows=_read_trace(baseline_run_dir), treatment_trace_rows=_read_trace(treatment_run_dir),
        baseline_context=pair_binding["baseline_context"], treatment_context=pair_binding["treatment_context"], official_safety_canonical_payload=safety,
        pretreatment_clearance=pair_binding["pretreatment_clearance"] if pair_binding["family"] == "R-HLC" else None,
        source_reference_xy=pair_binding.get("source_reference_xy"), target_reference_xy=pair_binding.get("target_reference_xy"),
        native_route_reference_xy=pair_binding.get("native_route_reference_xy"), native_route_reference_source=pair_binding.get("native_route_reference_source"),
    )
    return {"schema_version": DISPATCHER_SCHEMA_VERSION, "pair_id": pair_binding["pair_id"], "dispatch_status": "EVALUATED_NO_POSTHOC_PAIR_DELETION", "evaluation": result, "official_safety_pair_pass": safety["pair_safety_pass"], "posthoc_eligibility_deletion_allowed": False, "manual_pair_splicing_performed": False}
