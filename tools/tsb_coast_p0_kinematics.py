#!/usr/bin/env python3
"""Exact piecewise-constant kinematics; no planner/controller/runner imports.

All outputs are analytic design fixtures, never realized behavior or F_match
qualification. No speed floor or vehicle dynamics are applied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def moments(segments, horizon=7.9):
    """Return signed impulse, first moment, and absolute impulse, in SI."""
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    previous_end = 0.0
    impulse = moment = absolute = 0.0
    for start, end, acceleration in segments:
        if not all(math.isfinite(x) for x in (start, end, acceleration)):
            raise ValueError("segments must contain finite SI values")
        if start < previous_end or end <= start or end > horizon:
            raise ValueError("segments must be ordered, nonoverlapping and inside horizon")
        impulse += acceleration * (end - start)
        moment += acceleration * (end * end - start * start) / 2
        absolute += abs(acceleration) * (end - start)
        previous_end = end
    return impulse, moment, absolute


def speed_at(segments, time, initial_speed=10.0):
    return initial_speed + sum(a * max(0.0, min(time, end) - start)
                               for start, end, a in segments)


def metrics(segments, initial_speed=10.0, horizon=7.9):
    impulse, moment, absolute = moments(segments, horizon)
    if not math.isfinite(initial_speed) or initial_speed <= 0:
        raise ValueError("initial speed must be finite and positive")
    boundaries = {0.0, horizon} | {x for s, e, _ in segments for x in (s, e)}
    minimum_speed = min(speed_at(segments, t, initial_speed) for t in boundaries)
    if minimum_speed <= 0:
        raise ValueError("analytic design crosses zero speed; no clipping allowed")
    distance = initial_speed * horizon + horizon * impulse - moment
    result = dict(signed_impulse_mps=impulse, signed_first_moment_m=moment,
                  absolute_impulse_mps=absolute, terminal_speed_mps=initial_speed+impulse,
                  straight_path_length_m=distance, continuous_mean_speed_mps=distance/horizon,
                  continuous_mean_abs_accel_mps2=absolute/horizon,
                  minimum_speed_mps=minimum_speed)
    if abs(horizon - 7.9) < 1e-12:
        result['sample80_arithmetic_mean_speed_mps'] = sum(
            speed_at(segments, i/10, initial_speed) for i in range(80))/80
    return result


def coast_reference_valid(segments):
    """Necessary reference sign constraint, NOT a realized-coast qualifier."""
    moments(segments)
    return all(a <= 0 for _, _, a in segments)


def designs():
    # Two named proposals, not a parameter search or an authorized generator.
    return {
        'OLD_FROZEN': {'baseline': [(1.1, 2.9, -1.45)],
                       'treatment': [(1.1, 2., -2.4), (2., 3.3, 1.4), (3.3, 4.2, -2.4)]},
        'NAIVE_REMOVE_RECOVERY': {'baseline': [(1.1, 2.9, -1.45)],
                                  'treatment': [(1.1, 2., -2.4), (2., 3.3, 0.), (3.3, 4.2, -2.4)]},
        'C1_SYMMETRIC_EQUAL_PEAK': {'baseline': [(1.75, 3.55, -1.45)],
                                   'treatment': [(1.1, 2., -1.45), (2., 3.3, 0.), (3.3, 4.2, -1.45)]},
        'C2_LONGER_COAST': {'baseline': [(2.1, 3.9, -1.45)],
                            'treatment': [(1.1, 2.1, -1.305), (2.1, 3.9, 0.), (3.9, 4.9, -1.305)]},
        'C1_KEEP_OLD_BASELINE_TIMING': {'baseline': [(1.1, 2.9, -1.45)],
                                      'treatment': [(1.1, 2., -1.45), (2., 3.3, 0.), (3.3, 4.2, -1.45)]},
        'C1_MILD_DRAG_EQUAL_MOMENTS': {'baseline': [(1.75, 3.55, -1.45)],
                                     'treatment': [(1.1, 2., -2.48/1.8), (2., 3.3, -.1), (3.3, 4.2, -2.48/1.8)]},
    }


def analysis():
    records = {}
    for name, arms in designs().items():
        arm_metrics = {arm: metrics(segments) for arm, segments in arms.items()}
        delta = {k: arm_metrics['treatment'][k]-arm_metrics['baseline'][k]
                 for k in arm_metrics['baseline']}
        records[name] = {'reference_segments_s_s_mps2': arms,
                         'analytic_metrics_v0_10mps_T_7p9s': arm_metrics,
                         'treatment_minus_baseline': delta,
                         'treatment_reference_nonpositive': coast_reference_valid(arms['treatment'])}
    return {'schema_version': 'tsb_coast_p0_analytic_fixture_v1',
            'evidence_kind': 'ANALYTIC_REFERENCE_ONLY_NOT_CLOSED_LOOP',
            'is_official_F_match': False, 'runner_calls': 0,
            'candidate_selection_using_H_or_RBR': False, 'designs': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON path; overwrite refused')
    args = parser.parse_args()
    payload = json.dumps(analysis(), indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(payload)
        print(json.dumps({'status': 'ANALYTIC_FIXTURE_COMPLETE', 'output': str(args.output)}))
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
