#!/usr/bin/env python3
"""Read-only source/old-config inventory and closed-form algebra, never a rollout.

No nuPlan module is imported or executed. No controller, model, recorder or
runner object is instantiated. This is capability evidence, not Coast results.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
DEVKIT = ROOT.parent / 'nuplan-devkit'
MODEL = 'nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py'
TRACKER = 'nuplan/planning/simulation/controller/tracker/lqr.py'
CONTROLLER = 'nuplan/planning/simulation/controller/two_stage_controller.py'
DYNAMIC = 'nuplan/common/actor_state/dynamic_car_state.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method(path, class_name, name):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def defaults(path, class_name):
    node = method(path, class_name, '__init__')
    names = [a.arg for a in node.args.args]
    # Only read literal defaults; do not evaluate source expressions.
    result = {}
    for name, value in zip(names[-len(node.args.defaults):], node.args.defaults):
        try:
            result[name] = ast.literal_eval(value)
        except ValueError:
            result[name] = {'source_expression_not_executed': ast.unparse(value)}
    return result


def zero_command_algebra(dt, tau, steps):
    """Normalized analytic filter response, not a sampled vehicle trajectory.

    a_n/a_0 = r**n and delta_v/a_0 = dt*sum(r**j,j=1..n).
    No initial speed, pose, scene, or brake amplitude is selected.
    """
    if not math.isfinite(dt) or not math.isfinite(tau) or dt <= 0 or tau < 0:
        raise ValueError('dt must be positive and tau nonnegative and finite')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    ratio = tau/(dt+tau)
    residual = ratio**steps
    coefficient = dt*ratio*(1-residual)/(1-ratio)
    return dict(filter_ratio=ratio, acceleration_ratio_to_initial=residual,
                speed_change_per_initial_acceleration_s=coefficient)


def lqr_algebra(velocity_error, horizon_s, q, r):
    if not all(math.isfinite(x) for x in (velocity_error, horizon_s, q, r)) or min(horizon_s, q, r) <= 0:
        raise ValueError('finite error and positive horizon/q/r required')
    return q*horizon_s/(r+q*horizon_s**2)*velocity_error


def source_contract():
    model = DEVKIT/MODEL
    update = method(model, 'KinematicBicycleModel', '_update_commands')
    assignments = {n.targets[0].id: ast.unparse(n.value) for n in ast.walk(update)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    if assignments.get('updated_accel_x') != 'dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel':
        raise ValueError('Official acceleration filter changed; re-review required')
    inventory = []
    for path, cls, names in [
        (DEVKIT/MODEL, 'KinematicBicycleModel', ['__init__', '_update_commands', 'get_state_dot', 'propagate_state']),
        (DEVKIT/TRACKER, 'LQRTracker', ['track_trajectory', '_compute_reference_velocity_and_curvature_profile', '_longitudinal_lqr_controller', '_solve_one_step_lqr']),
        (DEVKIT/CONTROLLER, 'TwoStageController', ['update_state']),
        (DEVKIT/DYNAMIC, 'DynamicCarState', ['__init__']),
        (ROOT/'tools/r2_bj_b0_2_passive_actual_lqr_recorder.py', 'PassiveActualLQRRecorderV1', ['install']),
    ]:
        for name in names:
            node = method(path, cls, name)
            inventory.append(dict(path=os.path.relpath(path, ROOT), class_name=cls, symbol=name,
                                  first_line=node.lineno, last_line=node.end_lineno, sha256=sha(path)))
    dynamic_fields = [a.arg for a in method(DEVKIT/DYNAMIC, 'DynamicCarState', '__init__').args.args if a.arg != 'self']
    return dict(method_inventory=inventory, model_constructor_defaults=defaults(model, 'KinematicBicycleModel'),
                dynamic_state_constructor_fields=dynamic_fields, acceleration_filter_source=assignments['updated_accel_x'])


def audit():
    code = source_contract()
    configs = sorted((ROOT/'outputs/stageR/b1_resume_execution_r2_v1').rglob('resolved_config.json'))
    if len(configs) != 6:
        raise ValueError('Expected six existing R2 resolved configs; no new identity selection allowed')
    bindings = []
    tau = code['model_constructor_defaults']['accel_time_constant']
    for path in configs:
        config = json.loads(path.read_text())['ego_controller']
        if config['_target_'] != 'nuplan.planning.simulation.controller.two_stage_controller.TwoStageController':
            raise ValueError('Controller target changed')
        model, tracker = config['motion_model'], config['tracker']
        if model['_target_'] != 'nuplan.planning.simulation.controller.motion_model.kinematic_bicycle.KinematicBicycleModel':
            raise ValueError('Motion model target changed')
        if tracker['_target_'] != 'nuplan.planning.simulation.controller.tracker.lqr.LQRTracker':
            raise ValueError('Tracker target changed')
        actual_tau = model.get('accel_time_constant', tau)
        h = tracker['discretization_time']*tracker['tracking_horizon']
        bindings.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                             controller=config['_target_'], tracker=tracker['_target_'], model=model['_target_'],
                             dt_s=tracker['discretization_time'], horizon_s=h,
                             accel_tau_s=actual_tau, tau_origin='resolved override' if 'accel_time_constant' in model else 'source constructor default',
                             q=tracker['q_longitudinal'][0], r=tracker['r_longitudinal'][0]))
    first = bindings[0]
    if any(any(b[k] != first[k] for k in ['dt_s', 'horizon_s', 'accel_tau_s', 'q', 'r']) for b in bindings):
        raise ValueError('Existing production configuration mismatch')
    return dict(stage='TSB-Coast C0', capability='ABSTRACT_COAST_PROXY_ONLY',
                evidence_scope='SOURCE_AST_OLD_RESOLVED_CONFIG_AND_ALGEBRA_ONLY_NO_RUNTIME_MODEL_CALLS',
                current_LQR_already_has_coast_mode=False,
                future_zero_command_gate='PROPOSAL_ONLY_NOT_IMPLEMENTED_OR_AUTHORIZED',
                physical_drive_brake_channels_observable=False, code=code, production_bindings=bindings,
                algebra=dict(
                    one_step_zero_command=zero_command_algebra(first['dt_s'], first['accel_tau_s'], 1),
                    old_gap_length_only_18_steps=zero_command_algebra(first['dt_s'], first['accel_tau_s'], 18),
                    positive_unit_velocity_error_command=lqr_algebra(1.,first['horizon_s'],first['q'],first['r']),
                    zero_velocity_error_command=lqr_algebra(0.,first['horizon_s'],first['q'],first['r']),
                    note='normalized analytic checks; unit error is not a new physical candidate; no response trace generated'),
                scientific_execution_counts={k:0 for k in ['SIMULATION','RUNNER_RUN','MODEL_PROPAGATE','TRACKER_RUNTIME_CALL','TSB_COAST_ROLLOUT','RBR_TRAINING','RBR_INFERENCE','H_FEATURES','BDD','MMD','PRIMARY','V','C']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON evidence path; overwrite refused')
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'capability':payload['capability'], 'existing_configs':len(payload['production_bindings']), 'runtime_model_calls':0}))


if __name__ == '__main__':
    main()
