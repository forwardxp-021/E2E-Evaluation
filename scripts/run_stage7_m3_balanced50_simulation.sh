#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation
PYTHON_BIN=/home/forwardxp/miniconda3/envs/nuplan/bin/python
NUPLAN_DEVKIT_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit
NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
NUPLAN_MAP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
NUPLAN_MAPS_ROOT="$NUPLAN_MAP_ROOT"
NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
SELECTION_DIR=outputs/stage7_m3_pdm_balanced50_selection_v1
SIM_DIR=outputs/stage7_m3_pdm_balanced50_stage7c_v1
EXPECTED_MANIFEST_SHA256=a59b003ee517237d5a888e9774f939879ce812ac99d09a8f41e23c6d7e196313

export PATH="/home/forwardxp/miniconda3/envs/nuplan/bin:${PATH}"
export NUPLAN_DEVKIT_ROOT NUPLAN_DATA_ROOT NUPLAN_MAP_ROOT NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT
mkdir -p "$NUPLAN_EXP_ROOT"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" - "$SELECTION_DIR/milestone3_selection_summary.json" \
  "$EXPECTED_MANIFEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("overall_verdict") != "PASS":
    raise SystemExit("Milestone 3 frozen selection is not PASS")
if summary.get("selection_manifest_sha256") != sys.argv[2]:
    raise SystemExit(
        "Milestone 3 selection manifest changed after freeze: "
        f"{summary.get('selection_manifest_sha256')} != {sys.argv[2]}"
    )
if summary.get("target_scenarios") != 50:
    raise SystemExit(f"expected 50 selected scenarios, got {summary.get('target_scenarios')}")
print("Milestone 3 frozen selection preflight PASS")
PY

"$PYTHON_BIN" tools/stage7c1_run_nuplan_simulation.py \
  --context_dir "$SELECTION_DIR/stage7c_candidate_context" \
  --nuplan_db_root "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini" \
  --nuplan_map_root "$NUPLAN_MAP_ROOT" \
  --output_dir "$SIM_DIR" \
  --planners pdm_closed_conservative_v1 pdm_closed_assertive_v1 \
  --max_scenarios 50 \
  --min_timesteps 2 \
  --allow_external_planner_name \
  --hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]' \
  --nuplan_simulation_command_template "$PYTHON_BIN $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_filter=all_scenarios {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name=stage7_m3_pdm_balanced50_stage7c_v1 job_name=stage7c_{planner_name_safe} output_dir={output_dir}" \
  --command_timeout_s 600 \
  --progress_json "$SIM_DIR/stage7c_progress.json" \
  --overwrite
