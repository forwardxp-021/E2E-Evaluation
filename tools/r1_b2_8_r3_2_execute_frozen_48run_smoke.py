#!/usr/bin/env python3
"""The sole future R1 48-run entrypoint; dry-run is default and never simulates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json"; SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v2.1.json"
PAIR_BINDINGS = R1 / "r1_b2_8_r3_2_frozen_pair_evaluation_bindings_v1.0.json"; R3_BINDING = R1 / "r1_b2_8_r3_execution_bindings_manifest_v1.0.json"
FINAL_MANIFEST = R1 / "r1_b2_8_r3_2_final_execution_binding_manifest_v1.1.json"
ROSTER_SHA = "b977b802a7b25f0be37d04f3277cba2b2e98e521a2e30938ec40af9f278c1973"; SCHEDULE_SHA = "6733dc623cce2e2b64b9eb71cd407982b54dcaf5ecd48b644058c767c89d552f"


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path: Path) -> Mapping[str, Any]: return json.loads(path.read_text(encoding="utf-8"))

@dataclass
class FrozenBudgetLedger:
    """No retry, no duplicate claim, and cap-before-runner semantics."""
    cap: int = 48
    claimed: list[str] = field(default_factory=list)
    def claim(self, run_id: str) -> None:
        if len(self.claimed) >= self.cap: raise RuntimeError("HARD_FAIL_BEFORE_RUNNER_RUN_CAP_48")
        if run_id in self.claimed: raise RuntimeError("HARD_FAIL_BEFORE_RUNNER_RUN_DUPLICATE_CLAIM")
        self.claimed.append(run_id)
    def reject_49th(self) -> str:
        saved = list(self.claimed); self.claimed = [f"claim-{i}" for i in range(48)]
        try:
            self.claim("forbidden-49th")
        except RuntimeError as exc:
            return str(exc)
        finally: self.claimed = saved
        raise RuntimeError("49TH_CLAIM_NOT_REJECTED")


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], trace: Path, raw: Path) -> list[str]:
    return ["+simulation=closed_loop_nonreactive_agents", "planner=r1_official_technical_smoke_v2_2_r3", "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{entry['db_path']}]", "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]", "worker=single_machine_thread_pool", "worker.max_workers=1", "scenario_builder.max_workers=1", "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1", "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701", "run_metric=true", "enable_simulation_progress_bar=false", "experiment_name=r1_b2_8_r3_2", f"job_name={run['run_id']}", f"output_dir={raw}", f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"]


def _load_frozen() -> tuple[list[Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if sha(ROSTER) != ROSTER_SHA or sha(SCHEDULE) != SCHEDULE_SHA: raise ValueError("STOP_BEFORE_SIMULATION_ROSTER_OR_SCHEDULE_SHA_MISMATCH")
    schedule, roster, pairs = read(SCHEDULE), read(ROSTER), read(PAIR_BINDINGS)
    runs = sorted(schedule["runs"], key=lambda row: int(row["run_order"]))
    if [int(row["run_order"]) for row in runs] != list(range(1, 49)): raise ValueError("STOP_BEFORE_SIMULATION_RUN_ORDER_MISMATCH")
    entries = {(row["scenario_token"], row["log_id"]): row for row in roster["entries"]}
    pair_by_id = {row["pair_id"]: row for row in pairs["pairs"]}
    if len(runs) != 48 or len(pair_by_id) != 24 or any(row["pair_id"] not in pair_by_id for row in runs): raise ValueError("STOP_BEFORE_SIMULATION_FROZEN_BINDING_MISMATCH")
    return runs, entries, pair_by_id


def _authorize(manifest: Path, authorization: Path | None) -> None:
    if authorization is None: raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_AUTHORIZATION_REQUIRED")
    payload = read(authorization)
    if payload.get("OFFICIAL_SMOKE_AUTHORIZED") is not True or payload.get("final_execution_manifest_sha256") != sha(manifest): raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_AUTHORIZATION_INVALID")
    bound = read(manifest)
    if bound.get("status") != "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION": raise PermissionError("STOP_BEFORE_SIMULATION_MANIFEST_NOT_READY")
    if bound.get("roster", {}).get("sha256") != sha(ROSTER) or bound.get("schedule", {}).get("sha256") != sha(SCHEDULE) or bound.get("frozen_pair_binding", {}).get("sha256") != sha(PAIR_BINDINGS):
        raise PermissionError("STOP_BEFORE_SIMULATION_FROZEN_ARTIFACT_SHA_MISMATCH")
    for relative_path, expected_sha in bound.get("future_execution_components_sha256", {}).items():
        component = ROOT / relative_path
        if not component.is_file() or sha(component) != expected_sha:
            raise PermissionError(f"STOP_BEFORE_SIMULATION_EXECUTION_SHA_CLOSURE_MISMATCH:{relative_path}")


def run(*, execute: bool, output_root: Path | None = None, authorization: Path | None = None, manifest: Path = FINAL_MANIFEST) -> dict[str, Any]:
    runs, entries, pair_by_id = _load_frozen()
    if execute: _authorize(manifest, authorization)
    official_env()
    from hydra import compose, initialize_config_dir
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_b2_8_r3_frozen_run_dispatcher import build_planner_from_frozen_binding
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair
    ledger, completed, audit = FrozenBudgetLedger(), set(), []
    temporary = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="r1_b2_8_r3_2_dry_run_"); root = Path(temporary.name)
    else: root = output_root
    try:
        config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
        for run_row in runs:
            run_id = str(run_row["run_id"]); entry = entries[(run_row["scenario_token"], run_row["log_id"])]
            run_root, trace, raw = root / run_id, root / run_id / "trace", root / run_id / "raw"
            if run_root.exists(): raise RuntimeError(f"STOP_BEFORE_SIMULATION_OUTPUT_REUSE:{run_id}")
            trace.mkdir(parents=True)
            os.environ.update({"R1_B2_8_R3_BINDING_MANIFEST": str(R3_BINDING), "R1_B2_8_R3_RUN_ID": run_id, "R1_B2_8_R3_TRACE_DIR": str(trace)})
            with initialize_config_dir(config_dir=str(config_root)): cfg = compose(config_name="default_simulation", overrides=_overrides(run_row, entry, trace, raw))
            planner = build_planner_from_frozen_binding(str(R3_BINDING), run_id, str(trace)); common = set_up_common_builder(cfg, "r3_2_orchestrator")
            callback_worker = build_callbacks_worker(cfg); callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker); runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
            if len(runners) != 1: raise RuntimeError(f"STOP_BEFORE_SIMULATION_RUNNER_CONSTRUCTION:{run_id}")
            ledger.claim(run_id)
            if execute:
                try: runners[0].run()
                except Exception as exc: raise RuntimeError(f"STOPPED_ON_TECHNICAL_FAILURE_REQUIRES_OWNER_REVIEW:{run_id}:{type(exc).__name__}") from exc
                completed.add(run_id)
                pair = pair_by_id[run_row["pair_id"]]
                if {pair["baseline_run_id"], pair["treatment_run_id"]}.issubset(completed):
                    result = evaluate_frozen_pair(pair_binding=pair, baseline_run_dir=root / pair["baseline_run_id"], treatment_run_dir=root / pair["treatment_run_id"])
                    (root / f"{pair['pair_id']}__evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            audit.append({"run_id": run_id, "run_order": run_row["run_order"], "runner_constructed": True, "budget_claimed_before_runner_run": True, "runner_run_called": bool(execute), "pair_binding_lookup": run_row["pair_id"], "simulation_started": bool(execute)})
        return {"status": "EXECUTION_ORCHESTRATOR_READY" if not execute else "COMPLETE", "runs": audit, "counts": {"runs": len(audit), "pair_bindings": len(pair_by_id)}, "claim_49": ledger.reject_49th(), "simulation_started": bool(execute), "official_runs": len(completed), "consumed_real_budget": len(completed), "dry_claim_count": len(ledger.claimed) if not execute else None, "technical_failure_policy": "STOPPED_ON_TECHNICAL_FAILURE_REQUIRES_OWNER_REVIEW_NO_RETRY_NO_REPLACEMENT", "scientific_outcome_policy": "RECORD_AND_CONTINUE_SCHEDULE_NO_RETRY_NO_REPLACEMENT"}
    finally:
        if temporary is not None: temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute", action="store_true"); parser.add_argument("--authorization", type=Path); parser.add_argument("--output-root", type=Path); parser.add_argument("--manifest", type=Path, default=FINAL_MANIFEST); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); result = run(execute=args.execute, output_root=args.output_root, authorization=args.authorization, manifest=args.manifest)
    if args.output:
        if args.output.exists(): raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{args.output}")
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "simulation_started": result["simulation_started"]}, ensure_ascii=False))

if __name__ == "__main__": main()
