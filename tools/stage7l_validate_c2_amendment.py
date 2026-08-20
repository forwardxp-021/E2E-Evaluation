#!/usr/bin/env python3
"""Validate the final Stage7L-C2 task-population amendment without results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_generate_pretreatment_task_masks import (  # noqa: E402
    LAT_DYNAMICS_OFFICIAL_TYPES,
    TASK_MASK_DEFINITION_VERSION,
    build_task_mask_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SCIENTIFIC_KEYS = (
    "representation",
    "contrast",
    "mode",
    "task",
    "statistic",
    "null",
    "swaps",
    "plus_one_p_value",
    "success_rule",
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def values(rows: Iterable[Mapping[str, str]], key: str) -> set[str]:
    return {row[key] for row in rows if row.get(key)}


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    protocol = read_json(args.protocol_config)
    authorization = read_json(args.authorization_manifest)
    amendment = read_json(args.amendment_manifest)
    summary = read_json(args.freeze_summary)
    roster = read_csv(args.roster)
    source = read_csv(args.pretreatment_source)
    development = read_csv(args.development_ledger)

    provenance = amendment["provenance"]
    invariant = amendment["canonical_invariant_sha256"]
    paired = protocol["paired_bdd"]
    primary = paired["primary_endpoint"]
    family = paired["secondary_family"]
    tasks = paired["secondary_task_definitions"]

    protocol_sha = sha256_file(args.protocol_config)
    authorization_sha = sha256_file(args.authorization_manifest)
    roster_sha = sha256_file(args.roster)
    source_sha = sha256_file(args.pretreatment_source)
    task_tool_sha = sha256_file(args.task_mask_tool)

    require(protocol_sha == provenance["stage7l_c2_protocol_sha256"], "C2 protocol SHA mismatch")
    require(
        authorization_sha == provenance["stage7l_c2_authorization_sha256"],
        "C2 authorization SHA mismatch",
    )
    require(roster_sha == provenance["roster_sha256"], "frozen roster SHA changed")
    require(source_sha == provenance["pretreatment_source_sha256"], "pre-treatment source SHA changed")
    require(task_tool_sha == provenance["task_mask_generation_tool_sha256"], "task-mask tool SHA changed")
    require(authorization["frozen_protocol"]["sha256"] == protocol_sha, "authorization protocol binding mismatch")
    require(authorization["frozen_protocol"]["paired_bdd"] == paired, "authorization paired_bdd mismatch")
    require(
        authorization["frozen_confirmation_artifacts"]["roster"]["sha256"] == roster_sha,
        "authorization roster binding changed",
    )

    require(len(roster) == 80, "confirmation roster N must remain 80")
    require(sum(row["direction"] == "left" for row in roster) == 15, "left must remain 15")
    require(sum(row["direction"] == "right" for row in roster) == 65, "right must remain 65")
    require(len(values(roster, "log_name")) == 79, "unique logs must remain 79")
    require(
        not values(roster, "scenario_token") & values(development, "scenario_token"),
        "development scenario overlap must remain zero",
    )
    development_logs = {
        row["log_name"]
        for row in development
        if row.get("log_name") and "STAGE7L_B_" in row.get("exclusion_reason", "")
    }
    require(len(development_logs) == 26, "expected 26 frozen Stage7L-B development logs")
    require(not values(roster, "log_name") & development_logs, "development log overlap must remain zero")

    for key in (
        "source_assets",
        "treatment",
        "eligibility",
        "selection",
        "mechanism",
        "semantic_uncertainty_reporting",
        "nuisance_gate",
        "safety_validity_gate",
        "failure_policy",
        "representation_lock",
    ):
        require(canonical_sha256(protocol[key]) == invariant[key], f"frozen section changed: {key}")

    scientific_primary = {key: primary[key] for key in PRIMARY_SCIENTIFIC_KEYS}
    require(
        canonical_sha256(scientific_primary) == invariant["primary_endpoint_scientific_definition"],
        "Primary BDD scientific definition changed",
    )
    require(protocol["failure_policy"]["minimum_completed_scenarios"] == 76, "minimum complete changed")
    require(paired["primary_minimum_analyzable_pair_count"] == 76, "Primary minimum pair changed")

    mask_rows = build_task_mask_rows(roster, source)
    require(paired["task_mask_definition_version"] == TASK_MASK_DEFINITION_VERSION, "task-mask version mismatch")
    require(set(tasks["LAT.DYNAMICS"]["pre_treatment_official_scenario_types"]) == set(LAT_DYNAMICS_OFFICIAL_TYPES), "LAT.DYNAMICS type set mismatch")
    lane_tokens = [row["scenario_token"] for row in mask_rows if row["LAT.LANE_CHANGE"]]
    dynamics_tokens = [row["scenario_token"] for row in mask_rows if row["LAT.DYNAMICS"]]
    frozen_masks = amendment["task_population_freeze"]
    require(len(lane_tokens) == 80, "LAT.LANE_CHANGE must equal all frozen roster members")
    require(len(dynamics_tokens) == frozen_masks["LAT.DYNAMICS"]["n_scenarios"], "LAT.DYNAMICS N mismatch")
    require(canonical_sha256(lane_tokens) == frozen_masks["LAT.LANE_CHANGE"]["mask_sha256"], "LAT.LANE_CHANGE mask SHA mismatch")
    require(canonical_sha256(dynamics_tokens) == frozen_masks["LAT.DYNAMICS"]["mask_sha256"], "LAT.DYNAMICS mask SHA mismatch")
    require(canonical_sha256(lane_tokens) != canonical_sha256(dynamics_tokens), "task masks must be distinct")
    require(tasks["LAT.LANE_CHANGE"]["source"] == "frozen_confirmation_roster_membership", "lane-change population source is not roster membership")
    require(tasks["LAT.DYNAMICS"]["source_timing"] == "pre_treatment", "LAT.DYNAMICS is not pre-treatment")
    require(tasks["LAT.DYNAMICS"]["mapping_strength"] == "MIXED_PROXY", "LAT.DYNAMICS boundary changed")

    primary_cell_sha = canonical_sha256(paired["primary_cell_definition"])
    require(primary_cell_sha == primary["cell_definition_sha256"], "Primary cell SHA mismatch")
    require(
        primary_cell_sha == family["excluded_primary_cell_definition_sha256"],
        "Primary and corresponding matrix cell definitions differ",
    )
    require(
        primary["task_population_definition_id"]
        == tasks["LAT.LANE_CHANGE"]["task_population_definition_id"],
        "Primary and secondary LAT.LANE_CHANGE populations differ",
    )

    representations = paired["secondary_representations"]
    contrasts = paired["secondary_contrasts"]
    task_names = paired["secondary_tasks"]
    theoretical = list(itertools.product(representations, contrasts, task_names))
    primary_tuple = ("B_seed3407", "dose100_vs_dose0", "LAT.LANE_CHANGE")
    require(len(theoretical) == 40, "theoretical matrix must have 40 cells")
    require(theoretical.count(primary_tuple) == 1, "Primary must appear exactly once in theoretical matrix")
    require(len([cell for cell in theoretical if cell != primary_tuple]) == 39, "secondary matrix must have 39 cells")
    require(family["theoretical_cells"] == 40, "protocol theoretical count changed")
    require(family["fixed_secondary_test_count"] == 39, "protocol secondary count changed")
    require(family["primary_excluded_exactly_once"] is True, "Primary exclusion rule changed")

    noncomputable = family["non_computable_cell_policy"]
    require(noncomputable["status"] == "NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION", "non-computable status mismatch")
    require(noncomputable["raw_p_for_multiplicity"] == 1.0, "non-computable p must be 1.0")
    require(noncomputable["cell_remains_in_39_test_family"] is True, "non-computable cell was removed")
    low_n = family["low_n_computable_cell_policy"]
    require(low_n["status"] == "LOW_N_SECONDARY_DIAGNOSTIC", "low-N status mismatch")
    require(low_n["compute_normally"] is True and low_n["new_minimum_n_gate"] is None, "low-N policy changed")

    require(authorization["stage7l_d_started"] is False, "Stage7L-D has started")
    require(summary["stage7l_d_started"] is False, "freeze summary says Stage7L-D started")
    require(summary["confirmation_rollout_started"] is False, "confirmation rollout has started")
    require(amendment["change_flags"]["confirmation_result_existed"] is False, "result-free assertion failed")
    require(protocol["immutability"]["c2_is_final_pre_stage7l_d_protocol_consistency_amendment"] is True, "C2 finality missing")
    for prefix in ("stage7l_d", "stage7l_e"):
        found = sorted((args.repo_root / "outputs").glob(f"{prefix}*"))
        require(not found, f"Unexpected confirmation result path exists for {prefix}: {found}")

    return {
        "schema_version": "stage7l_c2_task_population_consistency_validation_v1",
        "status": "STAGE7L_C2_TASK_POPULATION_CONSISTENCY_AMENDMENT_FROZEN",
        "protocol_sha256": protocol_sha,
        "blind_authorization_sha256": authorization_sha,
        "roster_sha256": roster_sha,
        "task_masks": {
            "LAT.LANE_CHANGE": {"n": len(lane_tokens), "sha256": canonical_sha256(lane_tokens)},
            "LAT.DYNAMICS": {"n": len(dynamics_tokens), "sha256": canonical_sha256(dynamics_tokens)},
        },
        "hard_assertions": {
            "primary_and_secondary_corresponding_cell_definition_identical": True,
            "primary_excluded_exactly_once": True,
            "theoretical_cells_equals_40": True,
            "secondary_cells_equals_39": True,
            "lat_lane_change_equals_full_frozen_roster": True,
            "lat_dynamics_uses_pretreatment_metadata_only": True,
            "non_computable_secondary_cell_uses_conservative_p1": True,
            "roster_and_all_scientific_invariants_unchanged": True,
            "stage7l_d_started": False,
        },
        "final_states": amendment["final_states"],
        "forbidden_work_performed": {
            "planner_rollout_or_confirmation": False,
            "embedding_export_or_read": False,
            "bdd_or_mmd": False,
            "training": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--protocol-config", type=Path, default=ROOT / "configs/stage7l_c_prospective_confirmation_protocol_v1.json")
    parser.add_argument("--authorization-manifest", type=Path, default=ROOT / "docs/stage7l_c_blind_confirmation_authorization_manifest_v1.json")
    parser.add_argument("--amendment-manifest", type=Path, default=ROOT / "docs/stage7l_c2_task_population_consistency_amendment_manifest_v1.json")
    parser.add_argument("--roster", type=Path, default=ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_roster.csv")
    parser.add_argument("--pretreatment-source", type=Path, default=ROOT / "outputs/stage7l_b2_dynamic_clearance_expanded_inventory_v2_pittsburgh/pool_b_strict_development_log_disjoint_dynamic_clean.csv")
    parser.add_argument("--development-ledger", type=Path, default=ROOT / "outputs/stage7l_b_final_development_freeze_v1/stage7l_b_final_prior_exclusion_ledger.csv")
    parser.add_argument("--freeze-summary", type=Path, default=ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_freeze_summary.json")
    parser.add_argument("--task-mask-tool", type=Path, default=ROOT / "tools/stage7l_generate_pretreatment_task_masks.py")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(args)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
