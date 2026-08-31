#!/usr/bin/env python3
"""R1 B2.8-R3.1 adapter for the already-frozen official safety contract.

This module deliberately delegates parsing to the historical canonicalizer.  It
does not calculate a metric, select an identity, or introduce a safety
threshold: the pair rule is the existing B2.x rule that each arm has zero
at-fault collisions and drivable-area compliance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_official_metric_canonicalizer import (  # noqa: E402
    CANONICAL_SCHEMA_VERSION,
    MetricCanonicalizationError,
    canonicalize_official_metrics,
)


ADAPTER_SCHEMA_VERSION = "r1_b2_8_r3_1_official_safety_adapter_v1.0"
FROZEN_PAIR_PASS_SEMANTICS = (
    "each arm: number_of_all_at_fault_collisions_stat_value == 0 AND "
    "drivable_area_compliance_stat_value is true; pair: baseline AND treatment"
)


def _require_payload(payload: Mapping[str, Any]) -> tuple[int, bool]:
    """Defensively require the exact historical canonical payload shape."""
    try:
        collision = payload["collision"]["number_of_all_at_fault_collisions_stat_value"]
        drivable = payload["drivable_area"]["drivable_area_compliance_stat_value"]
    except (KeyError, TypeError) as exc:
        raise MetricCanonicalizationError("FROZEN_CANONICAL_SAFETY_PAYLOAD_MISSING_REQUIRED_FIELD") from exc
    if isinstance(collision, bool) or not isinstance(collision, int) or collision < 0:
        raise MetricCanonicalizationError("FROZEN_CANONICAL_SAFETY_PAYLOAD_INVALID_COLLISION")
    if not isinstance(drivable, bool):
        raise MetricCanonicalizationError("FROZEN_CANONICAL_SAFETY_PAYLOAD_INVALID_DRIVABLE")
    return collision, drivable


def adapt_official_safety(run_dir: str | Path) -> dict[str, Any]:
    """Adapt one actual nuPlan metric-engine output directory, fail closed."""
    canonical = canonicalize_official_metrics(Path(run_dir))
    collision, drivable = _require_payload(canonical["canonical_payload"])
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "historical_canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "raw_metric_source": "actual_nuPlan_metric_engine_parquet_output",
        "canonical_payload": canonical["canonical_payload"],
        "canonical_payload_sha256": canonical["canonical_payload_sha256"],
        "raw_metric_artifact_provenance": canonical["artifact_provenance"],
        "frozen_arm_safety_pass": collision == 0 and drivable,
        "frozen_pair_pass_semantics": FROZEN_PAIR_PASS_SEMANTICS,
        "representation_bdd_rbr_read": False,
        "posthoc_eligibility_applied": False,
        "threshold_or_metric_changed": False,
    }


def adapt_pair_official_safety(baseline_run_dir: str | Path, treatment_run_dir: str | Path) -> dict[str, Any]:
    """Return the single pair payload consumed by the V2.1 evaluator."""
    baseline = adapt_official_safety(baseline_run_dir)
    treatment = adapt_official_safety(treatment_run_dir)
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "baseline": baseline,
        "treatment": treatment,
        "baseline_safety_pass": baseline["frozen_arm_safety_pass"],
        "treatment_safety_pass": treatment["frozen_arm_safety_pass"],
        "pair_safety_pass": baseline["frozen_arm_safety_pass"] and treatment["frozen_arm_safety_pass"],
        "frozen_pair_pass_semantics": FROZEN_PAIR_PASS_SEMANTICS,
        "posthoc_eligibility_applied": False,
        "threshold_or_metric_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--treatment-run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite versioned safety output: {args.output}")
    payload = adapt_pair_official_safety(args.baseline_run_dir, args.treatment_run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "pair_safety_pass": payload["pair_safety_pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
