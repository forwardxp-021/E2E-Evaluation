#!/usr/bin/env python3
"""R1 official smoke evaluator V2.1 with native HLC progress semantics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from tools.r1_closed_loop_benchmark_v2_1 import calculate_hlc_option_b_v2_timestamp_aware, calculate_tsb_option_a_v2_timestamp_aware, exact_realized_window_v1_1, hlc_endpoint_v1_1_timestamp_aware, prospective_primary_f_match, timestamp_aware_hlc_engineering, trajectory_arrays_timestamp_aware, trajectory_descriptors_timestamp_aware
from tools.r1_context_mechanism_core import assert_pair_context_identity, qualify_hlc_pair, qualify_tsb_pair
from tools.r1_hlc_measurement_conformance_v1 import hlc_realized_lane_transition_progress_v1_0, terminal_native_route_progress_v1_0


class R1OfficialTechnicalSmokeEvaluatorV2_1:
    PIPELINE = ("REALIZED_CURRENT_EGO_ITERATIONS_0_79", "NATIVE_HLC_REALIZED_PROGRESS", "TIMESTAMP_AWARE_MECHANISM", "PROSPECTIVE_PRIMARY_F_MATCH", "HLC_TIMESTAMP_AWARE_ENDPOINT_WITH_NATIVE_ROUTE_PROGRESS", "TIMESTAMP_AWARE_ENGINEERING", "OFFICIAL_SAFETY_CANONICALIZER")

    @staticmethod
    def _states(rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        return exact_realized_window_v1_1(rows)

    def evaluate_pair(
        self,
        *,
        family: str,
        baseline_trace_rows: Sequence[Mapping[str, Any]],
        treatment_trace_rows: Sequence[Mapping[str, Any]],
        baseline_context: Mapping[str, Any],
        treatment_context: Mapping[str, Any],
        official_safety_canonical_payload: Mapping[str, Any],
        pretreatment_clearance: Mapping[str, Any] | None = None,
        source_reference_xy: Sequence[Sequence[float]] | None = None,
        target_reference_xy: Sequence[Sequence[float]] | None = None,
        native_route_reference_xy: Sequence[Sequence[float]] | None = None,
        native_route_reference_source: str | None = None,
    ) -> Dict[str, Any]:
        rows = list(baseline_trace_rows) + list(treatment_trace_rows)
        if any("planner_output_trajectory" in row and row.get("primary_measurement_source") == "PLANNED" for row in rows):
            raise ValueError("PLANNED_TRAJECTORY_PRIMARY_FORBIDDEN")
        context_identity = assert_pair_context_identity(baseline_context, treatment_context)
        if not context_identity["pair_context_identity_pass"]:
            raise ValueError("PAIR_CONTEXT_IDENTITY_FAIL_CLOSED")
        baseline, treatment = self._states(baseline_trace_rows), self._states(treatment_trace_rows)
        baseline_descriptors = trajectory_descriptors_timestamp_aware(baseline)
        treatment_descriptors = trajectory_descriptors_timestamp_aware(treatment)
        fmatch = prospective_primary_f_match(baseline_descriptors, treatment_descriptors, family)
        base_time, base_xy, _, base_speed = trajectory_arrays_timestamp_aware(baseline)
        treatment_time, treatment_xy, _, treatment_speed = trajectory_arrays_timestamp_aware(treatment)
        progress_audit = route_progress = None
        if family == "R-HLC":
            if source_reference_xy is None or target_reference_xy is None or native_route_reference_xy is None or native_route_reference_source is None:
                raise ValueError("HLC_SOURCE_TARGET_AND_NATIVE_ROUTE_REFERENCES_REQUIRED")
            baseline_progress = hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source_reference_xy, target_reference_xy=target_reference_xy, realized_ego_xy=base_xy)
            treatment_progress = hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source_reference_xy, target_reference_xy=target_reference_xy, realized_ego_xy=treatment_xy)
            base_mechanism = calculate_hlc_option_b_v2_timestamp_aware(base_time, baseline_progress["clipped_progress_for_frozen_mechanism"], base_speed)
            treatment_mechanism = calculate_hlc_option_b_v2_timestamp_aware(treatment_time, treatment_progress["clipped_progress_for_frozen_mechanism"], treatment_speed)
            mechanism = qualify_hlc_pair(base_mechanism, treatment_mechanism)
            route_progress = terminal_native_route_progress_v1_0(baseline_terminal_xy=base_xy[-1], treatment_terminal_xy=treatment_xy[-1], native_route_reference_xy=native_route_reference_xy, route_reference_source=native_route_reference_source)
            route_delta = float(route_progress["paired_route_progress_delta_m"])
            endpoint = {"baseline": hlc_endpoint_v1_1_timestamp_aware(baseline, target_reference_xy, paired_route_progress_delta_m=route_delta), "treatment": hlc_endpoint_v1_1_timestamp_aware(treatment, target_reference_xy, paired_route_progress_delta_m=route_delta), "paired_native_route_progress": route_progress}
            engineering = {"baseline": timestamp_aware_hlc_engineering(baseline), "treatment": timestamp_aware_hlc_engineering(treatment)}
            progress_audit = {"baseline": baseline_progress, "treatment": treatment_progress}
        else:
            base_mechanism = calculate_tsb_option_a_v2_timestamp_aware(base_time, base_speed)
            treatment_mechanism = calculate_tsb_option_a_v2_timestamp_aware(treatment_time, treatment_speed)
            mechanism, endpoint, engineering = qualify_tsb_pair(base_mechanism, treatment_mechanism), None, None
        if pretreatment_clearance is not None and pretreatment_clearance.get("pretreatment_only") is not True:
            raise ValueError("POSTHOC_CLEARANCE_RECALCULATION_FORBIDDEN")
        return {"status": "EVALUATED_REALIZED_FIRST_NATIVE_CONFORMANT", "primary_measurement_source": "REALIZED_CURRENT_EGO", "pipeline": list(self.PIPELINE), "context_identity": context_identity, "native_hlc_progress": progress_audit, "mechanism": mechanism, "f_match": fmatch, "endpoint": endpoint, "native_route_progress": route_progress, "engineering": engineering, "official_safety": dict(official_safety_canonical_payload), "pretreatment_clearance_ledger_entry": None if pretreatment_clearance is None else dict(pretreatment_clearance), "planned_trajectory_role": "SECONDARY_GENERATOR_INTENT_ONLY", "posthoc_eligibility_deletion_allowed": False, "frozen_scientific_numerics_modified": False}


__all__ = ["R1OfficialTechnicalSmokeEvaluatorV2_1"]
