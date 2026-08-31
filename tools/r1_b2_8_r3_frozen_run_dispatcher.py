#!/usr/bin/env python3
"""Exact v2.1 schedule-to-planner dispatcher; no fallback identities."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from tools.r1_official_technical_smoke_planner_v2_2 import R1OfficialTechnicalSmokePlannerV2_2

def load_frozen_run_binding(path: str | Path, run_id: str) -> Dict[str, Any]:
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 'r1_b2_8_r3_execution_bindings_manifest_v1.0': raise ValueError('R3_BINDING_SCHEMA_MISMATCH')
    rows=[r for r in payload['frozen_run_bindings'] if r['run_id']==run_id]
    if len(rows)!=1: raise ValueError(f'FROZEN_RUN_ID_MATCH_COUNT_MUST_EQUAL_ONE:{run_id}:{len(rows)}')
    row=rows[0]; roster=row['future_roster_row']
    if (row['family'],row['scenario_token'],row['log_id']) != (roster['family'],roster['scenario_token'],roster['log_id']): raise ValueError('FROZEN_SCHEDULE_ROSTER_IDENTITY_MISMATCH')
    if row['arm'] not in roster['arms']: raise ValueError('FROZEN_SCHEDULE_ARM_MISMATCH')
    return row

def build_planner_from_frozen_binding(binding_manifest_path: str, run_id: str, trace_dir: str) -> R1OfficialTechnicalSmokePlannerV2_2:
    row=load_frozen_run_binding(binding_manifest_path,run_id)
    return R1OfficialTechnicalSmokePlannerV2_2(row['future_roster_row'],row['family'],row['arm'],trace_dir)
