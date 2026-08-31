#!/usr/bin/env python3
"""Deterministic post-run dispatcher for frozen R1 B2.8-R3.1 pair artifacts."""

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
DISPATCHER_SCHEMA_VERSION = "r1_b2_8_r3_1_post_run_evaluator_dispatcher_v1.0"


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"FROZEN_ARTIFACT_MUST_BE_JSON_OBJECT:{path}")
    return value


def _read_trace(run_dir: str | Path) -> list[Mapping[str, Any]]:
    path = Path(run_dir) / "trace" / TRACE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"REALIZED_PRIMARY_TRACE_MISSING:{path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("REALIZED_PRIMARY_TRACE_EMPTY_OR_INVALID")
    if any(row.get("primary_measurement_source") != "REALIZED_CURRENT_EGO" for row in rows):
        raise ValueError("REALIZED_PRIMARY_TRACE_SOURCE_MISMATCH")
    return rows


def _require_pair_binding(binding: Mapping[str, Any]) -> None:
    required = ("family", "baseline_context", "treatment_context", "pretreatment_clearance")
    if binding.get("family") == "R-HLC":
        required += ("source_reference_xy", "target_reference_xy", "native_route_reference_xy", "native_route_reference_source")
    missing = [key for key in required if key not in binding]
    if missing:
        raise ValueError(f"FROZEN_PAIR_BINDING_MISSING:{','.join(missing)}")
    if binding["pretreatment_clearance"].get("pretreatment_only") is not True:
        raise ValueError("POSTHOC_CLEARANCE_RECALCULATION_FORBIDDEN")


def evaluate_frozen_pair(
    *,
    pair_binding: Mapping[str, Any],
    baseline_run_dir: str | Path,
    treatment_run_dir: str | Path,
) -> dict[str, Any]:
    """Read all artifacts by fixed path, evaluate exactly one scheduled pair."""
    _require_pair_binding(pair_binding)
    baseline_trace, treatment_trace = _read_trace(baseline_run_dir), _read_trace(treatment_run_dir)
    official_safety = adapt_pair_official_safety(baseline_run_dir, treatment_run_dir)
    result = R1OfficialTechnicalSmokeEvaluatorV2_1().evaluate_pair(
        family=str(pair_binding["family"]),
        baseline_trace_rows=baseline_trace,
        treatment_trace_rows=treatment_trace,
        baseline_context=pair_binding["baseline_context"],
        treatment_context=pair_binding["treatment_context"],
        official_safety_canonical_payload=official_safety,
        pretreatment_clearance=pair_binding["pretreatment_clearance"],
        source_reference_xy=pair_binding.get("source_reference_xy"),
        target_reference_xy=pair_binding.get("target_reference_xy"),
        native_route_reference_xy=pair_binding.get("native_route_reference_xy"),
        native_route_reference_source=pair_binding.get("native_route_reference_source"),
    )
    return {
        "schema_version": DISPATCHER_SCHEMA_VERSION,
        "dispatch_status": "EVALUATED_NO_POSTHOC_PAIR_DELETION",
        "pair_binding": dict(pair_binding),
        "evaluation": result,
        "official_safety_pair_pass": official_safety["pair_safety_pass"],
        "posthoc_eligibility_deletion_allowed": False,
        "manual_pair_splicing_performed": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-binding", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--treatment-run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite versioned evaluator output: {args.output}")
    payload = evaluate_frozen_pair(
        pair_binding=_read_json(args.pair_binding),
        baseline_run_dir=args.baseline_run_dir,
        treatment_run_dir=args.treatment_run_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["dispatch_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
