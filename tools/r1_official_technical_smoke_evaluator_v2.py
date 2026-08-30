#!/usr/bin/env python3
"""Unique realized-first evaluation pipeline for future R1 official smoke V2."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from tools.r1_closed_loop_benchmark_v2_1 import calculate_hlc_option_b_v2_timestamp_aware, calculate_tsb_option_a_v2_timestamp_aware, exact_realized_window_v1_1, hlc_endpoint_v1_1_timestamp_aware, prospective_primary_f_match, timestamp_aware_hlc_engineering, trajectory_arrays_timestamp_aware, trajectory_descriptors_timestamp_aware
from tools.r1_context_mechanism_core import assert_pair_context_identity, qualify_hlc_pair, qualify_tsb_pair


class R1OfficialTechnicalSmokeEvaluatorV2:
    PIPELINE = ("REALIZED_CURRENT_EGO_ITERATIONS_0_79", "TIMESTAMP_AWARE_MECHANISM", "PROSPECTIVE_PRIMARY_F_MATCH", "HLC_TIMESTAMP_AWARE_ENDPOINT", "TIMESTAMP_AWARE_ENGINEERING", "OFFICIAL_SAFETY_CANONICALIZER")

    @staticmethod
    def _states(rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        return exact_realized_window_v1_1(rows)

    def evaluate_pair(self, *, family: str, baseline_trace_rows: Sequence[Mapping[str, Any]], treatment_trace_rows: Sequence[Mapping[str, Any]], baseline_context: Mapping[str, Any], treatment_context: Mapping[str, Any], official_safety_canonical_payload: Mapping[str, Any], pretreatment_clearance: Mapping[str, Any] | None = None, target_reference_xy: Sequence[Sequence[float]] | None = None) -> Dict[str, Any]:
        if any("planner_output_trajectory" in row and row.get("primary_measurement_source") == "PLANNED" for row in list(baseline_trace_rows) + list(treatment_trace_rows)):
            raise ValueError("PLANNED_TRAJECTORY_PRIMARY_FORBIDDEN")
        context_identity = assert_pair_context_identity(baseline_context, treatment_context)
        baseline, treatment = self._states(baseline_trace_rows), self._states(treatment_trace_rows)
        base_desc, treatment_desc = trajectory_descriptors_timestamp_aware(baseline), trajectory_descriptors_timestamp_aware(treatment)
        fmatch = prospective_primary_f_match(base_desc, treatment_desc, family)
        base_time, base_xy, _, base_speed = trajectory_arrays_timestamp_aware(baseline)
        treatment_time, treatment_xy, _, treatment_speed = trajectory_arrays_timestamp_aware(treatment)
        if family == "R-HLC":
            if target_reference_xy is None:
                raise ValueError("HLC_NATIVE_TARGET_REFERENCE_REQUIRED")
            target = np.asarray(target_reference_xy, dtype=np.float64)
            source_delta = target - target[0]
            denominator = max(float(np.sum(source_delta[-1] ** 2)), 1e-12)
            base_progress = np.clip(np.sum((base_xy - base_xy[0]) * source_delta, axis=1) / denominator, 0.0, 1.0)
            treatment_progress = np.clip(np.sum((treatment_xy - treatment_xy[0]) * source_delta, axis=1) / denominator, 0.0, 1.0)
            base_mechanism = calculate_hlc_option_b_v2_timestamp_aware(base_time, base_progress, base_speed)
            treatment_mechanism = calculate_hlc_option_b_v2_timestamp_aware(treatment_time, treatment_progress, treatment_speed)
            mechanism = qualify_hlc_pair(base_mechanism, treatment_mechanism)
            route_delta = abs(float(base_desc["path_length"]) - float(treatment_desc["path_length"]))
            endpoint = {"baseline": hlc_endpoint_v1_1_timestamp_aware(baseline, target, paired_route_progress_delta_m=route_delta), "treatment": hlc_endpoint_v1_1_timestamp_aware(treatment, target, paired_route_progress_delta_m=route_delta)}
            engineering = {"baseline": timestamp_aware_hlc_engineering(baseline), "treatment": timestamp_aware_hlc_engineering(treatment)}
        else:
            base_mechanism = calculate_tsb_option_a_v2_timestamp_aware(base_time, base_speed)
            treatment_mechanism = calculate_tsb_option_a_v2_timestamp_aware(treatment_time, treatment_speed)
            mechanism, endpoint, engineering = qualify_tsb_pair(base_mechanism, treatment_mechanism), None, None
        if pretreatment_clearance is not None and pretreatment_clearance.get("pretreatment_only") is not True:
            raise ValueError("POSTHOC_CLEARANCE_RECALCULATION_FORBIDDEN")
        return {"status": "EVALUATED_REALIZED_FIRST", "primary_measurement_source": "REALIZED_CURRENT_EGO", "pipeline": list(self.PIPELINE), "context_identity": context_identity, "mechanism": mechanism, "f_match": fmatch, "endpoint": endpoint, "engineering": engineering, "official_safety": dict(official_safety_canonical_payload), "pretreatment_clearance_ledger_entry": None if pretreatment_clearance is None else dict(pretreatment_clearance), "planned_trajectory_role": "SECONDARY_GENERATOR_INTENT_ONLY", "posthoc_eligibility_deletion_allowed": False}


__all__ = ["R1OfficialTechnicalSmokeEvaluatorV2"]
