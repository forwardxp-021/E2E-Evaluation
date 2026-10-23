#!/usr/bin/env python3
"""Read-only checks of the official B1 callback owners and finalization readiness."""
from __future__ import annotations

import ast
import inspect
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.s2r_production_executor import ROOT, sha256_file

VERSION = 'b1_production_callback_preflight_v1'
EXPECTED_LIFECYCLE = ('RUNNER_COMPLETE', 'RECORDER_COMPLETE', 'OFFICIAL_METRICS_COMPLETE',
                      'SERIALIZER_COMPLETE', 'MANIFEST_COMPLETE', 'ARTIFACT_HASHES_VALIDATED')


class CallbackPreflightError(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise CallbackPreflightError(code)


def class_name(obj):
    return type(obj).__module__ + '.' + type(obj).__name__


def bound_method(obj, name, code):
    method = getattr(obj, name, None)
    require(callable(method) and getattr(method, '__self__', None) is obj
            and getattr(method, '__func__', None) is getattr(type(obj), name, None), code)


def lifecycle_source_contract():
    """Verify the unchanged sole B1 runner body, without calling any lifecycle method."""
    from tools import b1_tsb_qualification_executor as executor
    require(tuple(executor.LIFECYCLE_ORDER) == EXPECTED_LIFECYCLE, 'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH')
    source = inspect.getsource(executor.complete_production_arm)
    tree = ast.parse(source)
    expected = ['arm.runner.run', 'arm.recorder.validate_complete', 'save_runner_reports',
                'arm.common_builder.multi_main_callback.on_run_simulation_end',
                'validate_official_metric_outputs', 'validate_serializer_complete', 'validate_and_hash_artifacts']
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    # ast.unparse is available in the frozen Python 3.9 runtime.
    positions = []
    for name in expected:
        matches = [line for line, target in calls if target == name]
        require(len(matches) == 1, 'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH:' + name)
        positions.append(matches[0])
    require(positions == sorted(positions), 'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH:ORDER')
    from nuplan.planning.script.utils import save_runner_reports
    require(callable(save_runner_reports), 'MISSING_REQUIRED_RUNNER_REPORT_WRITER')
    return {'lifecycle_order': list(EXPECTED_LIFECYCLE), 'ordered_call_sites': expected,
            'executor_sha256': sha256_file(Path(executor.__file__))}


def validate_production_callback_binding(arm):
    """Inspect live official objects only; never initialize/run/finalize or write metrics."""
    from omegaconf import OmegaConf
    from nuplan.planning.simulation.callback.metric_callback import MetricCallback
    from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
    from nuplan.planning.simulation.callback.multi_callback import MultiCallback
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
    from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
    from nuplan.planning.metrics.metric_engine import MetricsEngine
    from tools.r2_bj_b0_2_passive_actual_lqr_recorder import PassiveActualLQRRecorderV1
    from tools.b1_offline_metric_finalize import EXPECTED_OFFICIAL_METRICS

    cfg = OmegaConf.to_container(arm.cfg, resolve=True)
    require(arm.runner.simulation is arm.simulation and arm.runner.planner is arm.planner,
            'INVALID_CALLBACK_BINDING:RUNNER_SIMULATION_OWNER')
    graph = arm.simulation.callback
    require(type(graph) is MultiCallback, 'INVALID_CALLBACK_BINDING:SIMULATION_MULTICALLBACK')
    bound_method(graph, 'on_simulation_end', 'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH:SIMULATION_DISPATCH')
    actual = graph.callbacks
    metrics = [c for c in actual if type(c) is MetricCallback]
    serializers = [c for c in actual if type(c) is SimulationLogCallback]
    require(len(metrics) == 1, 'MISSING_OFFICIAL_SIMULATION_METRIC_CALLBACK')
    require(len(serializers) == 1, 'MISSING_OFFICIAL_SIMULATION_LOG_CALLBACK')
    metric, serializer = metrics[0], serializers[0]
    bound_method(metric, 'on_simulation_end', 'INVALID_CALLBACK_BINDING:METRIC_METHOD')
    bound_method(serializer, 'on_simulation_end', 'INVALID_CALLBACK_BINDING:SERIALIZER_METHOD')
    require(type(metric.metric_engine) is MetricsEngine, 'INVALID_CALLBACK_BINDING:METRIC_ENGINE')
    require(metric._pool is None and serializer._pool is None and not metric.futures and not serializer.futures,
            'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH:NOT_FRESH_SEQUENTIAL')

    main = arm.common_builder.multi_main_callback
    require(type(main) is MultiMainCallback, 'INVALID_CALLBACK_BINDING:MAIN_MULTICALLBACK')
    bound_method(main, 'on_run_simulation_end', 'MISSING_REQUIRED_FINALIZER:MAIN_DISPATCH')
    finalizers = [c for c in main._main_callbacks if type(c) is MetricFileCallback]
    require(len(finalizers) == 1, 'MISSING_METRIC_FILE_CALLBACK')
    finalizer = finalizers[0]
    bound_method(finalizer, 'on_run_simulation_end', 'MISSING_REQUIRED_FINALIZER:METRIC_FILE_METHOD')
    finalizer_cfg = cfg.get('main_callback', {}).get('metric_file_callback', {})
    require(finalizer_cfg.get('_target_') == class_name(finalizer), 'INVALID_CALLBACK_BINDING:MAIN_CONFIG_TARGET')

    raw = (arm.run_root / 'raw').resolve()
    metric_dir = raw / 'metrics'
    require(Path(cfg['output_dir']).resolve() == raw == arm.common_builder.output_dir.resolve()
            and cfg.get('metric_dir') == 'metrics' and cfg.get('run_metric') is True,
            'INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION:OUTPUT_ROOT')
    require(metric.metric_engine._main_save_path.resolve() == metric_dir
            and finalizer._metric_file_output_path.resolve() == metric_dir
            and [p.resolve() for p in finalizer._scenario_metric_paths] == [metric_dir],
            'INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION:PRODUCER_CONSUMER_PATHS')
    require(Path(finalizer_cfg['metric_file_output_path']).resolve() == metric_dir
            and [Path(p).resolve() for p in finalizer_cfg['scenario_metric_paths']] == [metric_dir],
            'INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION:RESOLVED_CONFIG')
    metric_names = sorted(m.name for m in metric.metric_engine.metrics)
    # Official drivable-area builder emits two statistics; output names are not engine names.
    configured = dict(cfg['simulation_metric'].get('low_level', {}))
    configured.update(cfg['simulation_metric'].get(arm.simulation.scenario.scenario_type, {}))
    configured.update(cfg['simulation_metric'].get('high_level', {}))
    selected = cfg.get('selected_simulation_metrics')
    if isinstance(selected, str):
        selected = [selected]
    if selected is not None:
        configured = {key: value for key, value in configured.items() if key in selected}
    expected_engines = {value['name']: value['_target_'] for value in configured.values()}
    require({m.name: class_name(m) for m in metric.metric_engine.metrics} == expected_engines
            and (set(EXPECTED_OFFICIAL_METRICS) - {'corners_in_drivable_area'}).issubset(metric_names),
            'INVALID_CALLBACK_BINDING:REQUIRED_METRIC_ENGINES')
    require(metric_dir.is_dir() and os.access(metric_dir, os.W_OK),
            'INVALID_OFFICIAL_METRIC_OUTPUT_CONFIGURATION:NOT_WRITABLE')
    require(finalizer._delete_scenario_metric_files == finalizer_cfg['delete_scenario_metric_files'],
            'INVALID_CALLBACK_BINDING:FINALIZER_CONFIG')
    require(serializer._output_directory.resolve() == raw / 'simulation_log'
            and serializer._serialization_type == 'msgpack' and serializer._file_suffix == '.msgpack.xz',
            'INVALID_CALLBACK_BINDING:OFFICIAL_SERIALIZER')
    require(cfg['callback']['simulation_log_callback']['_target_'] == class_name(serializer),
            'INVALID_CALLBACK_BINDING:SERIALIZER_CONFIG_TARGET')
    require(cfg.get('disable_callback_parallelization') is True
            and cfg['worker']['_target_'] == 'nuplan.planning.utils.multithreading.worker_sequential.Sequential',
            'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH:SEQUENTIAL_CONFIG')
    require(type(arm.recorder) is PassiveActualLQRRecorderV1 and arm.recorder._tracker is arm.tracker
            and arm.controller._tracker is arm.tracker and arm.recorder._original is not None
            and getattr(arm.tracker, '_r2_bj_b0_2_passive_recorder_installed', False)
            and arm.recorder.row_count == 0 and arm.recorder.expected_rows == 79
            and arm.recorder.output_path.resolve() == (arm.run_root / 'telemetry/actual_lqr_controller_telemetry.jsonl').resolve(),
            'INVALID_CALLBACK_BINDING:PASSIVE_RECORDER')
    require(arm.simulation._history_buffer is None and len(arm.simulation.history) == 0,
            'CALLBACK_LIFECYCLE_CONTRACT_MISMATCH:SIMULATION_ALREADY_INITIALIZED')
    contract = lifecycle_source_contract()
    return {'version': VERSION, 'status': 'PASS', 'scope': 'STATIC_PREFLIGHT_VALIDATION',
            'runtime_metric_finalization': 'NOT_EXECUTED_NOT_PROVEN',
            'shared_callback_count': len(arm.callbacks),
            'simulation_callback_owner': class_name(graph),
            'simulation_callback_classes': [class_name(c) for c in actual],
            'main_callback_owner': class_name(main),
            'main_callback_classes': [class_name(c) for c in main._main_callbacks],
            'metric_engine_class': class_name(metric.metric_engine), 'metric_names': metric_names,
            'metric_engine_output_path': str(metric.metric_engine._main_save_path),
            'metric_finalizer_input_paths': [str(p) for p in finalizer._scenario_metric_paths],
            'metric_finalizer_output_path': str(finalizer._metric_file_output_path),
            'serializer_output_directory': str(serializer._output_directory),
            'recorder_bound_to_actual_tracker': True, 'simulation_uninitialized': True,
            'lifecycle_contract': contract}


@contextmanager
def zero_run_guard():
    """Fail before scientific entry, state advance, metric computation or finalization."""
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
    from nuplan.planning.simulation.simulation import Simulation
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    from nuplan.planning.simulation.callback.metric_callback import MetricCallback
    from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
    from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
    from nuplan.planning.metrics.metric_engine import MetricsEngine
    from tools.r2_b_controller_aware_planner_v1 import R2BControllerAwarePlannerV1
    from tools import b1_tsb_qualification_executor as executor
    from tools import b1_resume_execution as resume
    entries = [(SimulationRunner, 'run'), (SimulationRunner, '_initialize'),
               (Simulation, 'initialize'), (Simulation, 'propagate'),
               (R2BControllerAwarePlannerV1, 'initialize'), (R2BControllerAwarePlannerV1, 'compute_trajectory'),
               (R2BControllerAwarePlannerV1, 'compute_planner_trajectory'),
               (TwoStageController, 'update_state'), (LQRTracker, 'track_trajectory'),
               (KinematicBicycleModel, 'propagate_state'), (MetricsEngine, 'compute'), (MetricsEngine, 'write_to_files'),
               (MetricCallback, 'on_simulation_end'), (SimulationLogCallback, 'on_simulation_end'),
               (MetricFileCallback, 'on_run_simulation_end'), (MultiMainCallback, 'on_run_simulation_end'),
               (executor, 'claim_budget'),
               (resume, 'validate_claim'), (resume, 'update_status')]
    calls = {getattr(owner, '__name__', str(owner)) + '.' + name: 0 for owner, name in entries}
    def forbidden(key):
        def refuse(*args, **kwargs):
            calls[key] += 1
            raise RuntimeError('B1_RP_ZERO_RUN_VIOLATION:' + key)
        return refuse
    with ExitStack() as stack:
        for owner, name in entries:
            key = getattr(owner, '__name__', str(owner)) + '.' + name
            stack.enter_context(patch.object(owner, name, forbidden(key)))
        yield calls
