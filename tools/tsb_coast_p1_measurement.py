#!/usr/bin/env python3
"""P1 measurement and resolution gate. Pure fixtures only; no scientific runner.

This module cannot execute an arm or authorize one. A failed resolution audit
must block activation of the conditional Owner execution authorization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.r1_context_mechanism_core import median3
from tools.tsb_coast_p0_kinematics import designs, speed_at


VERSION = 'TSB_COAST_P1_MEASUREMENT_v1'
F_LIMITS = dict(mean_speed=.10, end_minus_start_speed=.15,
                path_length=.80, mean_abs_accel=.02)


def validate(time_us, speed):
    if len(time_us) != 80 or np.asarray(speed).shape != (80,):
        raise ValueError('MEASUREMENT_INVALID: exactly 80 frames and 79 intervals required')
    if any(not isinstance(x, (int, np.integer)) for x in time_us):
        raise ValueError('MEASUREMENT_INVALID: integer microsecond timestamps required')
    t = np.asarray([int(x)-int(time_us[0]) for x in time_us], dtype=np.float64)/1e6
    v = np.asarray(speed, dtype=np.float64)
    if not np.isfinite(v).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('MEASUREMENT_INVALID: finite speed and increasing timestamps required')
    if np.max(np.abs(np.diff(t)-.1)) > 1.00001e-6 or abs(t[-1]-7.9) > 1.00001e-6:
        raise ValueError('MEASUREMENT_INVALID: cadence outside fixed one-microsecond tolerance')
    return t, v


def ranges(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [(int(x[0]), int(x[-1])) for x in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)]


def integrate(t, values, start, end):
    return float(np.trapz(values[start:end+1], t[start:end+1]))


def measure(time_us, speed, arm, *, smoothing=True, edge_order=2, floor_triggered=False):
    """Timestamp support only; no interpolated duration or N*dt qualification."""
    if arm not in ('BASELINE', 'TREATMENT'):
        raise ValueError('Unknown P1 arm')
    t, v = validate(time_us, speed)
    used = median3(v) if smoothing else v
    a = np.gradient(used, t, edge_order=edge_order)
    reasons = []
    if floor_triggered or np.any(v <= 0) or any(t[e]-t[s] >= .5-1e-12 for s, e in ranges(v < 1.)):
        reasons.append('LOW_SPEED_ENDSTOP')
    phases = []
    for s, e in ranges(a <= -.6+1e-12):
        if t[e] < 1.1-1e-12 or t[s] > 5.5+1e-12:
            continue
        phase = dict(first_index=s, last_index=e, start_s=float(t[s]), end_s=float(t[e]),
                     support_s=float(t[e]-t[s]), peak_decel_mps2=float(-np.min(a[s:e+1])),
                     dose_mps=integrate(t, np.maximum(-a, 0), s, e))
        if phase['support_s'] >= .4-1e-12 and phase['peak_decel_mps2'] >= .8-1e-12 and phase['dose_mps'] >= .35-1e-12:
            phases.append(phase)
            if t[s] < 1.1-1e-12 or t[e] > 5.5+1e-12 or s == 0 or e == 79:
                reasons.append('PHASE_EDGE_CENSORED')
    expected = 1 if arm == 'BASELINE' else 2
    if len(phases) != expected:
        reasons.append('ONE_BRAKE_PHASE_NOT_REALIZED' if arm == 'BASELINE' else 'TWO_BRAKE_PHASES_NOT_REALIZED')
    gap = None
    if arm == 'TREATMENT' and len(phases) == 2:
        left, right = phases[0]['last_index'], phases[1]['first_index']
        candidates = [(s, e) for s, e in ranges((np.abs(a) <= .2+1e-12)
                                               & (np.arange(80) > left) & (np.arange(80) < right))]
        core = None
        if candidates:
            s, e = min(candidates, key=lambda x: (-(t[x[1]]-t[x[0]]), x[0]))
            core = dict(first_index=s, last_index=e, start_s=float(t[s]), end_s=float(t[e]),
                        support_s=float(t[e]-t[s]), raw_speed_drift_mps=float(v[e]-v[s]),
                        drift_limit_mps=float(.2*(t[e]-t[s])+.05))
        positive_runs = [(s, e) for s, e in ranges((a > .2+1e-12)
                                                  & (np.arange(80) >= left) & (np.arange(80) <= right))]
        max_positive_support = max([float(t[e]-t[s]) for s, e in positive_runs] or [0.])
        positive_dose = integrate(t, np.maximum(a, 0), left, right)
        positive_peak = float(np.max(a[left:right+1]))
        excluded = max_positive_support >= .3-1e-12 or positive_dose > .10+1e-12 or positive_peak > .40+1e-12
        gap = dict(start_s=float(t[left]), end_s=float(t[right]), support_s=float(t[right]-t[left]),
                   raw_speed_drift_mps=float(v[right]-v[left]), core=core,
                   positive_run_support_s=max_positive_support, positive_dose_mps=positive_dose,
                   positive_peak_mps2=positive_peak, positive_recovery_excluded=excluded)
        if gap['support_s'] < .7-1e-12:
            reasons.append('INTERBRAKE_GAP_TOO_SHORT')
        if core is None or core['support_s'] < .5-1e-12:
            reasons.append('COAST_CORE_TOO_SHORT')
        if core is not None and abs(core['raw_speed_drift_mps']) > core['drift_limit_mps']+1e-12:
            reasons.append('COAST_SPEED_DRIFT_EXCEEDED')
        if excluded:
            reasons.append('POSITIVE_RECOVERY_EXCEEDED')
    return dict(version=VERSION, arm=arm, status='FAIL' if reasons else 'PASS', reasons=reasons,
                phases=phases, gap=gap, total_negative_dose_mps=integrate(t, np.maximum(-a, 0), 0, 79),
                time_s=t.tolist(), raw_speed_mps=v.tolist(), measurement_speed_mps=used.tolist(),
                acceleration_mps2=a.tolist())


def f_descriptors(time_us, speed, xy):
    """Independent version of the four frozen timestamp-aware F definitions."""
    t, v = validate(time_us, speed)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (80, 2) or not np.isfinite(xy).all():
        raise ValueError('MEASUREMENT_INVALID: finite rear-axle XY[80,2] required')
    return dict(mean_speed=round(float(v.mean()), 6),
                end_minus_start_speed=round(float(v[-1]-v[0]), 6),
                path_length=round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 6),
                mean_abs_accel=round(float(np.abs(np.gradient(v, t, edge_order=2)).mean()), 6))


def stability(time_us, speed, arm):
    nominal = measure(time_us, speed, arm)
    variants = {}
    reasons = []
    for name, smoothing, edge in [('raw_edge2', False, 2), ('median3_edge1', True, 1)]:
        other = measure(time_us, speed, arm, smoothing=smoothing, edge_order=edge)
        matching_count = len(other['phases']) == len(nominal['phases'])
        peak_error = max([abs(x['peak_decel_mps2']-y['peak_decel_mps2'])
                          for x, y in zip(nominal['phases'], other['phases'])] or [0.])
        duration_error = max([abs(x['support_s']-y['support_s'])
                              for x, y in zip(nominal['phases'], other['phases'])] or [0.])
        ng, og = nominal['gap'], other['gap']
        if ng and og:
            if ng['core'] and og['core']:
                duration_error = max(duration_error, abs(ng['core']['support_s']-og['core']['support_s']))
            elif bool(ng['core']) != bool(og['core']):
                matching_count = False
            recovery_flip = ng['positive_recovery_excluded'] != og['positive_recovery_excluded']
        else:
            recovery_flip = bool(ng) != bool(og)
        unstable = not matching_count or peak_error > .2+1e-12 or duration_error > .2+1e-12 or recovery_flip
        if unstable:
            reasons.append(name)
        variants[name] = dict(peak_difference_mps2=peak_error, duration_difference_s=duration_error,
                              phase_count_equal=matching_count, positive_recovery_flip=recovery_flip,
                              unstable=unstable)
    return dict(status='MEASUREMENT_UNSTABLE' if reasons else 'STABLE', reasons=reasons, variants=variants)


def fixture(arm='TREATMENT', middle_accel=0.):
    """Analytic input, NOT planner output, controller response, or rollout."""
    segments = list(designs()['C2_LONGER_COAST'][arm.lower()])
    if arm == 'TREATMENT':
        segments[1] = (2.1, 3.9, middle_accel)
    t = np.arange(80, dtype=np.float64)/10
    speed = np.asarray([speed_at(segments, x) for x in t])
    # Exact reference path integral at each timestamp.
    x = 10*t + sum(a*((np.maximum(t-s, 0)**2-np.maximum(t-e, 0)**2)/2)
                    for s, e, a in segments)
    return [1_650_000_000_000_000+i*100_000 for i in range(80)], speed, np.column_stack([x, np.zeros(80)])


def resolution_audit():
    """Prespecified P0 ±.01 m/s deterministic noise stress, never outcome search.

    Test all four phases of the fixed period-four quantization-like error and
    both perturbation signs. Independent arm measurement error is allowed.
    Identity/zero-dose fixtures isolate measurement from mechanism differences.
    """
    cases = []
    for arm in ('BASELINE', 'TREATMENT'):
        ts, v, xy = fixture(arm)
        clean = f_descriptors(ts, v, xy)
        for phase in range(4):
            pattern = np.roll(np.tile([.01, .01, -.01, -.01], 20), phase)
            for sign in (-1, 1):
                noisy = v+sign*pattern
                observed = f_descriptors(ts, noisy, xy)
                delta = {k: abs(observed[k]-clean[k]) for k in F_LIMITS}
                failures = [k for k in F_LIMITS if delta[k] > F_LIMITS[k]+1e-12]
                cases.append(dict(fixture_arm=arm, noise_pattern='PERIOD4_PLUS_PLUS_MINUS_MINUS',
                                  phase=phase, sign=sign, max_abs_speed_error_mps=.01,
                                  clean_F=clean, noisy_F=observed, absolute_measurement_error=delta,
                                  exceeded_F_budgets=failures, mechanism_stability=stability(ts, noisy, arm)))
    failed = [c for c in cases if c['exceeded_F_budgets'] or c['mechanism_stability']['status'] != 'STABLE']
    # Pure constant-speed input is an additional null measurement sanity case.
    ts, _, xy = fixture()
    t = np.arange(80)/10
    straight = np.column_stack([10*t, np.zeros(80)])
    null_clean = f_descriptors(ts, np.full(80, 10.), straight)
    null_noisy = f_descriptors(ts, np.full(80, 10.)+np.tile([.01, .01, -.01, -.01], 20), straight)
    return dict(status='PREFLIGHT_BLOCKED' if failed else 'RESOLUTION_GATE_PASS',
                evidence_kind='DETERMINISTIC_MEASUREMENT_FIXTURES_NOT_CLOSED_LOOP',
                noise_budget_source='TSB-Coast P0 Mechanism Proposal: ±.01 m/s',
                actual_nuplan_speed_error_bound_established=False,
                scientific_implication='No C2 feasibility verdict; cannot certify fixed measurement budget under the requested stress envelope',
                cases=cases, failed_case_count=len(failed), total_case_count=len(cases),
                null_clean_F=null_clean, null_noisy_F=null_noisy, F_limits=F_LIMITS,
                max_F_error={k:max(c['absolute_measurement_error'][k] for c in cases) for k in F_LIMITS},
                runner_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution-audit', type=Path, required=True,
                        help='New fixture JSON output. Existing output is never overwritten.')
    args = parser.parse_args()
    result = resolution_audit()
    args.resolution_audit.parent.mkdir(parents=True, exist_ok=True)
    with args.resolution_audit.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases',)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
