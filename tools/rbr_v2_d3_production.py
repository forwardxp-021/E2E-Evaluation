#!/usr/bin/env python3
"""D3 compatibility factory; unchanged official control/model/metrics with explicit versioned planner binding."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
from tools.rbr_v2_d2_download import ROOT,write,sha


def build(spec,root):
 from tools.tsb_direct_coast_f2_production import build_f2_arm,validate_f2_production
 from tools.rbr_v2_d3_arc_binding import ArcBoundDirectCoastPlannerV1,VERSION
 from omegaconf import OmegaConf
 arm=build_f2_arm(spec,root);validate_f2_production(arm)
 parent_config_hash=sha(arm.run_root/'resolved_config.json')
 planner=ArcBoundDirectCoastPlannerV1(spec,arm.run_root/'trace',arm.run_root/'telemetry')
 arm.planner=planner;arm.runner._planner=planner
 OmegaConf.update(arm.cfg,'planner._target_','tools.rbr_v2_d3_arc_binding.ArcBoundDirectCoastPlannerV1')
 from tools.s2r_production_executor import canonical_sha256
 resolved=OmegaConf.to_container(arm.cfg,resolve=True);arm.config_hash=canonical_sha256(resolved);write(arm.run_root/'resolved_config.json',resolved)
 write(arm.run_root/'engineering_binding.json',dict(version=VERSION,parent_control='F2_DIRECT_REQUEST_STACK_v1',parent_config_hash=parent_config_hash,actual_config_hash=arm.config_hash,
       source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tools/rbr_v2_d3_arc_binding.py',ROOT/'tools/rbr_v2_d3_production.py']},
       change='Rebind endpoint-clamped first-edge arc to the same stitched native reference being sampled; no new path, no LQR/model/longitudinal/clock/metric change',correct_cases='ORIGINAL_CONSISTENT_ARC_USES_EXACT_ORIGINAL_FLOAT'))
 return arm


def validate(arm):
 from tools.tsb_direct_coast_f2_production import validate_callbacks
 from tools.tsb_coast_p1r_c_production import validate_clock_binding
 from tools.rbr_v2_d3_arc_binding import ArcBoundDirectCoastPlannerV1
 from tools.tsb_direct_coast_f2_tracker import ScriptedLongitudinalRequestTrackerV1
 if type(arm.planner) is not ArcBoundDirectCoastPlannerV1 or type(arm.tracker) is not ScriptedLongitudinalRequestTrackerV1:raise TypeError('D3 explicit planner and unchanged F2 adapter required')
 if arm.runner.planner is not arm.planner:raise ValueError('D3 planner owner mismatch')
 return dict(status='PASS',clock=validate_clock_binding(arm),callbacks=validate_callbacks(arm))


def lifecycle(arm,spec,contract):
 import inspect
 import tools.tsb_direct_coast_f2_production as p
 inspect.signature(p.lifecycle_f2).bind(arm,spec,contract)
 try:p.lifecycle_f2(arm,spec,contract)
 except RuntimeError as error:
  if str(error)!='Scientific arm gate FAIL; STOP no retry or replacement' or not (arm.run_root/'arm_science_review.json').exists():raise
 manifest=json.loads((arm.run_root/'execution_manifest.json').read_text());manifest['artifacts']['engineering_binding.json']=sha(arm.run_root/'engineering_binding.json');manifest['engineering_version']='SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1';write(arm.run_root/'execution_manifest.json',manifest)
 return json.loads((arm.run_root/'arm_science_review.json').read_text())
