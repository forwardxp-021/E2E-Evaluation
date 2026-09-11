#!/usr/bin/env python3
"""S1 zero-run serializer/validator; delegates science to the production evaluator.

No simulator or executor entry point is provided. The schema is a proposed S2
boundary, not evidence of production lifecycle qualification.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / 'docs/stageR/s1/S1_Canonical_Schema_Draft_v0.1.json'


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label}: required exact keys {expected}')


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: finite numeric value required')


def validate_trace(rows):
    schema = json.loads(SCHEMA.read_text())
    if len(rows) != 80:
        raise ValueError('trace: exactly 80 states required')
    previous = None
    for index, row in enumerate(rows):
        _keys(row, schema['trace_keys'], 'trace')
        if row['iteration_index'] != index or type(row['iteration_index']) is not int:
            raise ValueError('trace: expected integer indices 0...79')
        if row['primary_measurement_source'] != 'REALIZED_CURRENT_EGO':
            raise ValueError('trace: realized source required')
        state = row['current_ego']
        _keys(state, schema['state_keys'], 'current_ego')
        _keys(state['rear_axle'], ['x', 'y', 'heading'], 'rear_axle')
        for name, value in state['rear_axle'].items():
            _finite(value, name)
        _finite(state['speed_mps'], 'speed_mps')
        if state['speed_mps'] < 0 or type(state['time_us']) is not int:
            raise ValueError('state: nonnegative speed and integer microseconds required')
        if previous is not None and state['time_us'] <= previous:
            raise ValueError('trace: increasing timestamps required')
        previous = state['time_us']


def serialize_trace(path, rows):
    """The same strict JSONL serializer is used by fixtures and the proposed boundary."""
    validate_trace(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def validate_contract(contract):
    schema = json.loads(SCHEMA.read_text())
    _keys(contract, schema['contract_keys'], 'contract')
    if contract['schema_version'] != schema['instance_version']:
        raise ValueError('schema_version mismatch')
    if contract['units'] != schema['units']:
        raise ValueError('units mismatch')
    for field in ('pair_id', 'log_id', 'scenario_token'):
        if not isinstance(contract[field], str) or not contract[field]:
            raise ValueError(f'{field}: nonempty identity required')
    if type(contract['planner_calls']) is not int or contract['planner_calls'] != 80:
        raise ValueError('planner_calls must equal 80')
    if contract['family'] != 'R-TSB' or contract['pretreatment_clearance'] is not None:
        raise ValueError('TSB only; HLC clearance inapplicable')
    for arm in ('baseline', 'treatment'):
        context = contract[f'{arm}_context']
        for key in ('pre_context_raw_hash', 'canonical_context_json_hash'):
            if not isinstance(context.get(key), str) or len(context[key]) != 64:
                raise ValueError(f'{arm}_context.{key}: hash required')
    if contract['baseline_context'] != contract['treatment_context']:
        raise ValueError('context mismatch')
    if contract['runner_report_succeeded'] is not True:
        raise ValueError('runner report incomplete')


def validate_transitions(rows):
    if len(rows) != 79 or [r.get('iteration') for r in rows] != list(range(79)):
        raise ValueError('controller: exactly 79 transitions 0...78 required')
    for row in rows:
        for name in ('actual_acceleration_command_mps2', 'actual_tire_steering_rate_command_radps'):
            _finite(row.get(name), name)
        if row.get('behavior_changed') is not False:
            raise ValueError('passive recorder noninterference flag required')


def analyze_pair(contract, baseline_root, treatment_root):
    """Validate serialized inputs, then call the real dispatcher and safety parser."""
    validate_contract(contract)
    for root in (Path(baseline_root), Path(treatment_root)):
        rows = [json.loads(line) for line in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()]
        validate_trace(rows)
        actual = [json.loads(line) for line in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]
        validate_transitions(actual)
    result = evaluate_frozen_pair(pair_binding=contract, baseline_run_dir=baseline_root, treatment_run_dir=treatment_root)
    evaluation = result['evaluation']
    result['joint_pass'] = bool(evaluation['mechanism']['pass'] and evaluation['f_match']['pass'] and result['official_safety_pair_pass'])
    return result


def validate_plan(contracts, output_root, allowed_root, run_cap):
    """Static capacity/identity check only; does not claim budget or run anything."""
    target, allowed = Path(output_root).resolve(), Path(allowed_root).resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError('output-root violation')
    if target.exists():
        raise ValueError('output-root must be fresh')
    if type(run_cap) is not int or run_cap < 0 or 2 * len(contracts) > run_cap:
        raise ValueError('budget violation')
    for key in ('pair_id', 'scenario_token', 'log_id'):
        values = [c[key] for c in contracts]
        if len(set(values)) != len(values):
            raise ValueError(f'duplicate identity: {key}')
    for contract in contracts:
        validate_contract(contract)
    return {'planned_runs': 2 * len(contracts), 'executed_runs': 0}


def handcrafted_h(states):
    """One development-informed H; raw 30 columns, no data-dependent feature choice."""
    import numpy as np
    from tools.r1_closed_loop_benchmark_v2_1 import trajectory_arrays_timestamp_aware, trajectory_descriptors_timestamp_aware
    from tools.r1_context_mechanism_core import median3
    from tools.stage6l_prepare_context_representation_ablation import ego_kinematic_features
    time, xy, heading, speed = trajectory_arrays_timestamp_aware(states)
    # Preserve historical ego13 fixed-dt semantics even for irregular timestamps.
    # New temporal columns use physical time; cadence is reported, not thresholded.
    ego = np.zeros((1, 80, 8), dtype=np.float64)
    ego[0, :, :2], ego[0, :, 4], ego[0, :, 5] = xy, heading, speed
    features = list(ego_kinematic_features(ego, np.ones((1, 80), dtype=bool))[0])
    features.append(trajectory_descriptors_timestamp_aware(states)['mean_abs_accel'])
    accel = np.gradient(median3(speed), time, edge_order=2)
    for left in range(8):
        values = accel[(time >= left) & (time < left + 1)]
        if not len(values):
            raise ValueError('H_EMPTY_TEMPORAL_BIN')
        features.append(float(np.mean(values)))
    centered = accel - np.mean(accel)
    denominator = float(np.dot(centered, centered))
    acf_valid = denominator > 1e-12
    for lag in (5, 10, 20):
        features.append(float(np.dot(centered[:-lag], centered[lag:]) / denominator) if acf_valid else 0.0)
    # Trapezoidal quadrature weights over observed physical timestamps.
    weights = np.r_[np.diff(time)[0]/2, (time[2:]-time[:-2])/2, np.diff(time)[-1]/2]
    mass = np.maximum(-accel, 0) * weights
    total = float(mass.sum())
    mass_valid = total > 1e-12
    centroid = float(np.dot(mass, time)/total) if mass_valid else 0.0
    spread = float(np.sqrt(np.dot(mass, (time-centroid)**2)/total)) if mass_valid else 0.0
    features.extend([centroid, spread, float(acf_valid), float(mass_valid), float(np.max(np.abs(np.diff(time)-0.1)))])
    return np.asarray(features, dtype=np.float64)


def development_audit():
    """Only the 16 already exposed runs explicitly bound by frozen DEV-CAL."""
    import hashlib
    import numpy as np
    from tools.r1_closed_loop_benchmark_v2_1 import (
        calculate_tsb_option_a_v2_timestamp_aware, trajectory_arrays_timestamp_aware,
        trajectory_descriptors_timestamp_aware,
    )
    from tools.r2_b_controller_aware_generator_v1 import tsb_controller_aware_acceleration
    path = ROOT/'docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_results_v1.0.json'
    frozen = json.loads(path.read_text())
    runs = frozen['runs']
    if len(runs) != 16 or len({r['log_id'] for r in runs}) != 8:
        raise ValueError('DEV-CAL bound cardinality changed')
    output = {'status': 'DEVELOPMENT_ONLY_NOT_FRESH_QUALIFICATION', 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'runs': [], 'trace_sha256': {}}
    descriptors = {}
    for run in runs:
        trace_path = ROOT/run['trace_path']
        output['trace_sha256'][run['trace_path']] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        raw = [json.loads(s) for s in trace_path.read_text().splitlines()]
        rows = [{k: r[k] for k in ('iteration_index', 'primary_measurement_source', 'current_ego')} for r in raw]
        validate_trace(rows)
        states = [r['current_ego'] for r in rows]
        time, _, _, speed = trajectory_arrays_timestamp_aware(states)
        mechanism = calculate_tsb_option_a_v2_timestamp_aware(time, speed)
        descriptors[(run['pair_id'], run['arm'])] = trajectory_descriptors_timestamp_aware(states)
        try:
            h = handcrafted_h(states)
            h_status = 'FINITE_30' if len(h) == 30 else 'INVALID'
        except ValueError as error:
            h_status = str(error)
        output['runs'].append({'historical_run_id': run['run_id'], 'initial_speed_mps': float(speed[0]), 'minimum_speed_mps': float(speed.min()), 'dt_min_s': float(np.diff(time).min()), 'dt_max_s': float(np.diff(time).max()), 'measurement': mechanism['status'], 'phase_count': mechanism['brake_phase_count'], 'release_fraction': mechanism['interstage_release_fraction'], 'second_peak_ratio': mechanism['second_brake_peak_ratio'], 'H_status': h_status})
    output['signed_F_deltas'] = {}
    for key in ('mean_speed', 'end_minus_start_speed', 'path_length', 'mean_abs_accel'):
        delta = [descriptors[(p['pair_id'], 'TREATMENT')][key]-descriptors[(p['pair_id'], 'BASELINE')][key] for p in frozen['pairs']]
        output['signed_F_deltas'][key] = {'values': delta, 'mean': float(np.mean(delta)), 'sd_ddof1': float(np.std(delta, ddof=1)), 'min': min(delta), 'max': max(delta)}
    output['nominal_command_arithmetic'] = {}
    for arm in ('BASELINE', 'TREATMENT'):
        command = np.array([tsb_controller_aware_acceleration(i*.1, arm, frozen['parameters']) for i in range(79)])
        integral = np.r_[0, np.cumsum(command*.1)]
        output['nominal_command_arithmetic'][arm] = {'total_speed_change_mps': float(integral[-1]), 'maximum_cumulative_speed_loss_mps': float(-integral.min()), 'nonzero_command_samples': int(np.count_nonzero(command)), 'interpretation': 'UNCLAMPED_FIXED_SCHEDULE_ARITHMETIC_NOT_CLOSED_LOOP_OR_APPLICABILITY_GUARANTEE'}
    output['unresolved'] = ['initial full acceleration/steering/controller state not in current_ego payload', 'conservative controller/replanning reachable envelope not established', 'fresh eligible capacity not materialized', 'RBR effect and independent release variance not estimable from these 8 development logs']
    output['simulation'] = output['runner_run'] = output['new_scientific_identities_exposed'] = output['RBR_training'] = 0
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-audit', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(development_audit(), ensure_ascii=False, indent=2, allow_nan=False))
