#!/usr/bin/env python3
"""Versioned canonical-clock Coast factory. No default scientific execution."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.s2r_production_executor import (ROOT, CONFIG_ROOT, FreshArm, _validate_spec,
    verify_scenario_precontext, sha256_file, canonical_sha256, resolved_override_contract,
    write_manifest, PROTECTED, PROTECTED_SHA, SEED)
from tools.tsb_coast_p1r_c_clock import CoastCanonicalTimeControllerV1

STAGE = 'TSB_COAST_P1R_C_FIRST_PAIR'


def validate_clock_binding(arm):
    clock = arm.simulation._time_controller
    if type(clock) is not CoastCanonicalTimeControllerV1:
        raise ValueError('P1R_C_OLD_OR_UNBOUND_CLOCK_REJECTED')
    if arm.simulation._setup.time_controller is not clock or clock.scenario is not arm.simulation._scenario:
        raise ValueError('P1R_C_MULTIPLE_TIME_AUTHORITIES')
    if clock.episode_start_us != int(arm.spec['start_timestamp_us']):
        raise ValueError('P1R_C_EPISODE_ANCHOR_CHANGED')
    if arm.cfg.simulation_time_controller._target_ != type(clock).__module__+'.'+type(clock).__name__:
        raise ValueError('P1R_C_RESOLVED_CLOCK_CONFIG_MISMATCH')
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    observation = arm.simulation._observations
    if type(observation) is not TracksObservation or observation.scenario is not clock.scenario or observation.current_iteration != 0:
        raise ValueError('P1R_C_OBSERVATION_INDEX_BINDING_MISMATCH')
    if arm.controller._scenario is not clock.scenario:
        raise ValueError('P1R_C_CONTROLLER_SCENARIO_MISMATCH')
    proof = clock.require_mapping()
    proof['observer'] = type(observation).__module__+'.'+type(observation).__name__
    proof['shared_original_scenario'] = True
    return proof


def build_canonical_coast_arm(spec, run_root):
    """Fresh real official objects, never initialize/step/run or read outcomes."""
    from tools.r1_b2_8_r3_prospective_selector import official_env, official_count
    official_env(); _validate_spec(spec)
    run_root = Path(run_root)
    if spec['role'] != 'B' or spec.get('candidate') != 'TSB_COAST_C2_P1R':
        raise ValueError('P1R_C requires the unchanged C2 B-development spec')
    if run_root.exists():
        raise FileExistsError('P1R_C fresh namespace required')
    if sha256_file(PROTECTED) != PROTECTED_SHA or official_count(spec['db_path'], spec['scenario_token']) != 1:
        raise ValueError('P1R_C protected data or unique source identity mismatch')
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    from tools.tsb_coast_zero_command_tracker_v1 import ZeroLongitudinalRequestTrackerV1
    from tools.tsb_coast_p1r_recorder import CoastTransitionRecorderV1
    planner = CoastC2PlannerV1(spec, run_root/'trace', run_root/'telemetry')
    overrides = resolved_override_contract(spec, run_root/'raw')
    overrides = [('planner=tsb_coast_c2_p1r' if x.startswith('planner=') else
        'simulation_time_controller=tsb_coast_canonical_clock_v1' if x.startswith('simulation_time_controller.') else
        'experiment_name=tsb_coast_p1r_c_clock_closure' if x.startswith('experiment_name=') else x) for x in overrides]
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name='default_simulation', overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    common = set_up_common_builder(cfg, 'p1r_c_canonical_clock_factory')
    worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, worker)
    runners = build_simulations(cfg, common.worker, callbacks, worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise ValueError('P1R_C expected exactly one official runner')
    runner = runners[0]; simulation = runner._simulation
    verify_scenario_precontext(simulation._scenario, spec)
    controller = simulation._ego_controller
    if spec['arm'] == 'TREATMENT':
        controller._tracker = ZeroLongitudinalRequestTrackerV1(controller._tracker,
            int(spec['start_timestamp_us']), [(2100000, 3900000)])
    recorder = CoastTransitionRecorderV1(run_root/'telemetry/actual_lqr_controller_telemetry.jsonl',
        spec['arm'], int(spec['start_timestamp_us']))
    arm = FreshArm(spec, runner, planner, simulation, controller, controller._tracker,
        controller._motion_model, callbacks, recorder, random.Random(SEED), run_root,
        canonical_sha256(resolved), spec['precontext_id'], spec['route_id'], common, cfg)
    mapping = validate_clock_binding(arm)
    # The legacy optional clock check only recognizes the old class name.
    # Validate the exact new authority above, then retain its untouched strict
    # command/model/anchor/79-transition checks via the clock-free core install.
    recorder.install(controller)
    recorder._canonical_clock_authority = simulation._time_controller
    write_manifest(run_root/'resolved_config.json', resolved)
    write_manifest(run_root/'canonical_clock_mapping.json', mapping)
    write_manifest(run_root/'coast_binding.json', dict(version='P1R_C_CANONICAL_STACK_v1',
        candidate=spec['candidate'], arm=spec['arm'], episode_anchor_us=mapping['episode_start_us'],
        clock_source_sha256=sha256_file(ROOT/'tools/tsb_coast_p1r_c_clock.py'),
        adapter_enabled=spec['arm']=='TREATMENT', common_precontext_unchanged=True))
    return arm


def validate_canonical_production(arm):
    from tools.tsb_coast_p1r_production import validate_callbacks
    mapping = validate_clock_binding(arm)
    if arm.recorder._canonical_clock_authority is not arm.simulation._time_controller:
        raise ValueError('P1R_C_RECORDER_TIME_AUTHORITY_MISMATCH')
    return dict(status='PASS', callback_recorder=validate_callbacks(arm), clock=mapping)


def validate_scientific_credential(auth, spec, approved_contract, binding_path, preflight_path):
    """An approved amendment alone never supplies execution authorization."""
    if auth.get('stage') != STAGE or auth.get('authorization_status') != 'AUTHORIZED' or auth.get('scientific_owner_approved') is not True:
        raise PermissionError('P1R_C scientific execution NOT_AUTHORIZED; old/PROPOSED_ONLY credentials rejected')
    if auth.get('disclosed_observation_skew_acknowledged') is not True:
        raise PermissionError('P1R_C execution Owner must acknowledge disclosed mixed-time index replay')
    contract = json.loads(Path(approved_contract).read_text())
    if contract.get('D1') != 'APPROVED' or contract.get('D2') != 'APPROVED' or contract.get('D3') != 'CONDITIONAL_ENGINEERING_APPROVAL':
        raise PermissionError('P1R_C exact approved amendment not supplied')
    from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
    if contract.get('approved_mechanism_specification') != SCIENCE_CONTRACT:
        raise PermissionError('P1R_C approved D1/D2 values differ from frozen analyzer')
    for field, path in [('approved_amendment_sha256', approved_contract),
                        ('execution_source_binding_sha256', binding_path), ('production_preflight_sha256', preflight_path)]:
        if auth.get(field) != sha256_file(Path(path)):
            raise PermissionError('P1R_C authorization binding mismatch: '+field)
    expected = ['P1R-C2-DEV-01-BASELINE', 'P1R-C2-DEV-01-TREATMENT']
    if auth.get('approved_run_ids') != expected or spec['run_id'] not in expected or spec['role'] != 'B':
        raise PermissionError('P1R_C only the original first pair can be authorized')
    if auth.get('approved_spec_sha256', {}).get(spec['run_id']) != canonical_sha256(spec):
        raise PermissionError('P1R_C original SESSION/precontext/route spec changed')
    binding = json.loads(Path(binding_path).read_text())
    if binding.get('production_preflight_sha256') != sha256_file(Path(preflight_path)) or binding.get('approved_amendment_sha256') != sha256_file(Path(approved_contract)):
        raise PermissionError('P1R_C source binding refers to a different contract or preflight')
    required={'tools/tsb_coast_p1r_c_executor.py','tools/tsb_coast_p1r_c_clock.py','tools/tsb_coast_p1r_c_production.py',
        'tools/tsb_coast_p1r_planner.py','tools/tsb_coast_p1r_recorder.py','tools/tsb_coast_zero_command_tracker_v1.py',
        'tools/tsb_coast_p1r_analysis.py','tools/tsb_coast_p1_measurement.py',
        'configs/r1_official_technical_smoke_hydra/simulation_time_controller/tsb_coast_canonical_clock_v1.yaml',
        '../nuplan-devkit/nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py',
        '../nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py',
        '../nuplan-devkit/nuplan/planning/simulation/controller/two_stage_controller.py'}
    if not required.issubset(binding.get('source_config_sha256',{})):
        raise PermissionError('P1R_C canonical stack source binding incomplete')
    if str(spec['db_path']) not in binding.get('data_sha256',{}):
        raise PermissionError('P1R_C full source database not bound')
    for group in ('source_config_sha256', 'data_sha256'):
        if not binding.get(group) or any(sha256_file(ROOT/p) != h for p,h in binding[group].items()):
            raise PermissionError('P1R_C bound source/config/data changed')
    proof = json.loads(Path(preflight_path).read_text())
    if proof.get('status') != 'PASS' or len(proof.get('arms', [])) != 6 or any(x['clock']['status'] != 'PASS' or x['clock'].get('authority') != 'tools.tsb_coast_p1r_c_clock.CoastCanonicalTimeControllerV1' for x in proof['arms']):
        raise PermissionError('P1R_C three-pair canonical preflight incomplete')


if __name__ == '__main__':
    raise SystemExit('P1R-C is zero-run entry preparation. No scientific execution CLI is authorized.')
