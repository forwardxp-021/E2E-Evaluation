#!/usr/bin/env bash
set -euo pipefail

# ============================================================

# One-click PDM v1 lane-change experiment

# official nuPlan sim -> Stage7E context -> embedding -> Stage7F reports

# ============================================================

cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-/home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit}
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp}
mkdir -p "$NUPLAN_EXP_ROOT"

EXP_NAME=${EXP_NAME:-stage7p_pdm_v1_known_good_lane_change_2scenes_2planners_v2}

ROOT_DIR="outputs/${EXP_NAME}"
CONTEXT_SEED_DIR="${ROOT_DIR}/stage7c_candidate_context"
SIM_DIR="outputs/${EXP_NAME}_stage7c"
CTX_DIR="outputs/${EXP_NAME}_stage7e_context"
EMB_DIR="outputs/${EXP_NAME}_stage7e_embeddings"
STAGE7F_DIR="outputs/${EXP_NAME}_stage7f"
PAIR_DIR="${STAGE7F_DIR}/paired_delta"

CONSERVATIVE="pdm_closed_conservative_v1"
ASSERTIVE="pdm_closed_assertive_v1"

echo "============================================================"
echo "[0/7] Preflight tests"
echo "============================================================"
pytest -q 
tests/test_stage7c_external_planner.py 
tests/test_stage7p_find_lane_change_candidates.py 
tests/test_stage5d_context_core.py 
tests/test_lane_diagnostics.py 
tests/test_stage7f_idm_diagnostics.py 
tests/test_stage7f_report_card.py

echo "============================================================"
echo "[1/7] Create known-good lane-change Stage7C context"
echo "============================================================"
rm -rf "$ROOT_DIR" "$SIM_DIR" "$CTX_DIR" "$EMB_DIR" "$STAGE7F_DIR"
mkdir -p "$CONTEXT_SEED_DIR"

cat > "${CONTEXT_SEED_DIR}/merged_metadata.csv" <<'CSV'
log_name,scenario_token,scene_token,scenario_type,source
2021.06.07.18.53.26_veh-26_00005_00427,a59a8c3490f154e2,a59a8c3490f154e2,changing_lane_to_left,known_good_actual_type
2021.05.25.14.16.10_veh-35_01690_02183,f6f9afda75e251ae,f6f9afda75e251ae,changing_lane_to_right,known_good_actual_type
CSV

echo "Known-good context:"
cat "${CONTEXT_SEED_DIR}/merged_metadata.csv"

echo "============================================================"
echo "[2/7] Run official nuPlan PDM closed-loop simulation"
echo "============================================================"
python tools/stage7c1_run_nuplan_simulation.py 
--context_dir "$CONTEXT_SEED_DIR" 
--nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini 
--nuplan_map_root "$NUPLAN_MAPS_ROOT" 
--output_dir "$SIM_DIR" 
--planners "$CONSERVATIVE" "$ASSERTIVE" 
--max_scenarios 2 
--min_timesteps 2 
--require_same_scenario_alignment 
--allow_external_planner_name 
--hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]' 
--nuplan_simulation_command_template 'python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_filter=all_scenarios {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name='"$EXP_NAME"' job_name=stage7c_{planner_name_safe} output_dir={output_dir}' 
--overwrite

echo "============================================================"
echo "[3/7] Validate Stage7C output and actual scenario types"
echo "============================================================"
python - <<PY
import json
from pathlib import Path

sim_dir = Path("$SIM_DIR")
summary = json.loads((sim_dir / "warnings.json").read_text())
align = json.loads((sim_dir / "scenario_alignment.json").read_text())

assert summary["validation"]["pass"] is True, "Stage7C validation.pass is false"
assert summary["validation"]["official_success_count"] == 4, "Expected 4 official simulation successes"
assert summary["validation"]["pseudo_rollout"] is False, "Pseudo rollout detected"
assert summary["scenario_alignment"]["passed"] is True, "Scenario alignment failed"
assert summary["scenario_alignment"]["strict_nuplan_token_alignment_passed"] is True, "Strict token alignment failed"

allowed = {"changing_lane", "changing_lane_to_left", "changing_lane_to_right"}
bad = []
for r in align["records"]:
t = r.get("actual_scenario_type")
if t not in allowed:
bad.append((r.get("scenario_index"), r.get("planner_name"), t))
if bad:
raise SystemExit(f"Non lane-change actual_scenario_type found: {bad}")

print("Stage7C validation OK")
print("Actual scenario types are all changing_lane*")
PY

echo "============================================================"
echo "[4/7] Build Stage7E Stage5D-compatible context"
echo "============================================================"
python tools/build_nuplan_5neighbor_context_dataset.py 
--sim_dir "$SIM_DIR" 
--output_dir "$CTX_DIR" 
--assignment_mode lane_aware_with_geometric_fallback 
--nuplan_map_root "$NUPLAN_MAPS_ROOT" 
--map_name us-nv-las-vegas-strip 
--slot_sanity_min_coverage 0.05 
--write_projection_debug 
--write_strict_filter_diagnostic 
--strict_filter_min_laneaware_ratio 0.8 
--strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 
--debug_projection_sample_rows 100 
--overwrite

echo "============================================================"
echo "[5/7] Embed with Stage5/Stage6 encoder"
echo "============================================================"
python tools/stage7e_embed_stage6_dataset.py 
--context_dataset_dir "$CTX_DIR" 
--checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt 
--output_dir "$EMB_DIR" 
--overwrite

echo "============================================================"
echo "[6/7] Run Stage7F paired delta"
echo "============================================================"
mkdir -p "$PAIR_DIR"

python tools/stage7f_aggressive_conservative_paired_delta.py 
--context_dataset_dir "$CTX_DIR" 
--embedding_dir "$EMB_DIR" 
--stage7f_dir "$STAGE7F_DIR" 
--planner_a "$ASSERTIVE" 
--planner_b "$CONSERVATIVE" 
--output_dir "$PAIR_DIR" 
--overwrite

echo "============================================================"
echo "[7/7] Optional Stage7F BDD/report-card wrappers"
echo "============================================================"

# The paired-delta report is the currently validated core PDM-style report.

# If task/BDD wrappers are available in this branch, run them; otherwise skip safely.

if [[ -f tools/stage7f_run_report_card.py ]]; then
echo "Running stage7f_run_report_card.py if its CLI matches expected arguments..."
python tools/stage7f_run_report_card.py -h > "${STAGE7F_DIR}/report_card_help.txt" || true
fi

if [[ -f tools/stage7f_run_task_conditioned_bdd.py ]]; then
echo "Running task-conditioned BDD wrapper help dump..."
python tools/stage7f_run_task_conditioned_bdd.py -h > "${STAGE7F_DIR}/task_bdd_help.txt" || true
fi

echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Stage7C simulation report:"
echo "  ${SIM_DIR}/simulation_report.md"
echo "Stage7E context report:"
echo "  ${CTX_DIR}/context_build_report.md"
echo "Stage7E embedding report:"
echo "  ${EMB_DIR}/embedding_report.md"
echo "Stage7F paired delta:"
echo "  ${PAIR_DIR}/paired_delta_report.md"
echo ""
echo "If you want me to wire the exact task-BDD command into this script,"
echo "send the generated:"
echo "  ${STAGE7F_DIR}/task_bdd_help.txt"
echo "  ${STAGE7F_DIR}/report_card_help.txt"
