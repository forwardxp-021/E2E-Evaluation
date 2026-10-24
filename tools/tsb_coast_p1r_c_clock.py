#!/usr/bin/env python3
"""Canonical Coast control time, with immutable, disclosed raw-index mapping."""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

STEP_US = 100000
ITERATIONS = 81
MAPPING_POLICY = 'SAME_INDEX_UNIQUE_NEAREST_CELL_v1'


class CoastCanonicalTimeControllerV1(StepSimulationTimeController):
    """Only control time changes. The scenario and its observations stay intact.

    81 controller indices yield 80 planner/history rows and 79 updates under
    official StepSimulationTimeController end semantics. Index 80 is terminal
    metadata support, not an extra observed or propagated scientific state.
    """

    def __init__(self, scenario):
        super().__init__(scenario)
        if int(scenario.get_number_of_iterations()) < ITERATIONS:
            raise ValueError('COAST_CLOCK: scenario needs at least 81 source indices')
        if abs(float(scenario.database_interval)-.1) > 1e-12:
            raise ValueError('COAST_CLOCK: expected nominal 10Hz scenario sampling')
        raw = tuple(scenario.get_time_point(i).time_us for i in range(ITERATIONS))
        if any(type(t) is not int for t in raw):
            raise ValueError('COAST_CLOCK: original timestamps must be integer microseconds, never rounded')
        self._raw_us = raw
        if any(b <= a for a, b in zip(self._raw_us, self._raw_us[1:])):
            raise ValueError('COAST_CLOCK: source timestamps duplicated or reordered')
        self.episode_start_us = self._raw_us[0]
        tokens = getattr(scenario, '_lidarpc_tokens', None)
        self._tokens = tuple(str(x) for x in tokens[:ITERATIONS]) if tokens is not None else (None,)*ITERATIONS
        if len(self._tokens) != ITERATIONS or (tokens is not None and len(set(self._tokens)) != ITERATIONS):
            raise ValueError('COAST_CLOCK: source frame identities missing or duplicated')

    def number_of_iterations(self):
        return ITERATIONS

    def simulation_time(self, index):
        if type(index) is not int or not 0 <= index < ITERATIONS:
            raise ValueError('COAST_CLOCK: index must be integer 0..80')
        return TimePoint(self.episode_start_us + index*STEP_US)

    def get_iteration(self):
        return SimulationIteration(self.simulation_time(self.current_iteration_index), self.current_iteration_index)

    def mapping_report(self):
        rows = []
        for i, raw in enumerate(self._raw_us):
            sim = self.simulation_time(i).time_us
            spacing = [STEP_US]
            if i:
                spacing.append(raw-self._raw_us[i-1])
            if i+1 < ITERATIONS:
                spacing.append(self._raw_us[i+1]-raw)
            delta = raw-sim
            # Strictly within half of BOTH adjacent raw spacings and nominal
            # period. This tests index identity, not sensor accuracy or safety.
            nearest = 2*abs(delta) < min(spacing)
            rows.append(dict(index=i, simulation_time_us=sim, raw_log_time_us=raw,
                lidar_token=self._tokens[i], raw_minus_sim_us=delta,
                log_time_after_sim_time=delta > 0, unique_same_index_cell=nearest,
                observed_scientific_frame=i < 80))
        used = rows[:80]
        return dict(policy=MAPPING_POLICY, status='PASS' if all(r['unique_same_index_cell'] for r in rows) else 'CLOCK_MAPPING_NOT_ACCEPTABLE',
            authority=type(self).__module__+'.'+type(self).__name__, episode_start_us=self.episode_start_us,
            control_step_us=STEP_US, iteration_count=81, realized_frame_count=80, transition_count=79,
            nominal_half_period_us=STEP_US//2,
            half_period_role='INDEX_AMBIGUITY_BOUND_NOT_PHYSICAL_ERROR_OR_CAUSALITY_TOLERANCE',
            positive_skew_count=sum(r['raw_minus_sim_us'] > 0 for r in used),
            negative_skew_count=sum(r['raw_minus_sim_us'] < 0 for r in used),
            min_raw_minus_sim_us=min(r['raw_minus_sim_us'] for r in used),
            max_raw_minus_sim_us=max(r['raw_minus_sim_us'] for r in used),
            max_abs_raw_minus_sim_us=max(abs(r['raw_minus_sim_us']) for r in used),
            raw_80frame_span_us=self._raw_us[79]-self._raw_us[0], simulation_80frame_span_us=7900000,
            strict_event_time_causal_replay=False,
            future_frame_selected=False, observations_retimed_or_interpolated=False, rows=rows)

    def require_mapping(self):
        proof = self.mapping_report()
        if proof['status'] != 'PASS':
            raise ValueError('CLOCK_MAPPING_NOT_ACCEPTABLE: canonical time is outside its own raw-index cell')
        return proof
