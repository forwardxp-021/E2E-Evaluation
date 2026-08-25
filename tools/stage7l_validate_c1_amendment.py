#!/usr/bin/env python3
"""Validate the Stage7L-C1 protocol-only consistency amendment.

This validator reads protocol/freeze metadata only. It never opens rollout,
embedding, BDD or MMD results and never starts Stage7L-D.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def values(rows: Iterable[Mapping[str, str]], key: str) -> set[str]:
    return {row[key] for row in rows if row.get(key)}


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    protocol_path = args.protocol_config.resolve()
    authorization_path = args.authorization_manifest.resolve()
    amendment_path = args.amendment_manifest.resolve()
    roster_path = args.roster.resolve()
    development_path = args.development_ledger.resolve()
    summary_path = args.freeze_summary.resolve()

    protocol = read_json(protocol_path)
    authorization = read_json(authorization_path)
    amendment = read_json(amendment_path)
    summary = read_json(summary_path)
    roster = read_csv(roster_path)
    development = read_csv(development_path)

    expected = amendment["provenance"]
    invariant = amendment["canonical_invariant_sha256"]
    paired = protocol["paired_bdd"]
    primary = paired["primary_endpoint"]
    family = paired["secondary_family"]

    protocol_sha = sha256_file(protocol_path)
    authorization_sha = sha256_file(authorization_path)
    roster_sha = sha256_file(roster_path)

    require(protocol_sha == expected["new_protocol_sha256"], "new protocol SHA mismatch")
    require(
        authorization_sha == expected["new_blind_authorization_sha256"],
        "new blind authorization SHA mismatch",
    )
    require(roster_sha == expected["roster_sha256"], "frozen roster SHA changed")
    require(
        authorization["frozen_protocol"]["sha256"] == protocol_sha,
        "blind authorization is not bound to the amended protocol SHA",
    )
    require(
        authorization["representation_unlock_condition"]
        == paired["evaluation_condition"],
        "top-level representation unlock condition does not match amended protocol",
    )
    require(
        authorization["frozen_confirmation_artifacts"]["roster"]["sha256"]
        == expected["roster_sha256"],
        "blind authorization roster binding changed",
    )

    require(len(roster) == 80, "confirmation roster N must remain 80")
    require(sum(row["direction"] == "left" for row in roster) == 15, "left must remain 15")
    require(sum(row["direction"] == "right" for row in roster) == 65, "right must remain 65")
    require(len(values(roster, "log_name")) == 79, "unique log count must remain 79")
    require(
        not values(roster, "scenario_token") & values(development, "scenario_token"),
        "development scenario overlap must remain zero",
    )
    stage7l_b_development_logs = {
        row["log_name"]
        for row in development
        if row.get("log_name") and "STAGE7L_B_" in row.get("exclusion_reason", "")
    }
    require(len(stage7l_b_development_logs) == 26, "expected 26 Stage7L-B development logs")
    require(
        not values(roster, "log_name") & stage7l_b_development_logs,
        "development log overlap must remain zero",
    )

    for key in (
        "treatment",
        "eligibility",
        "mechanism",
        "nuisance_gate",
        "safety_validity_gate",
        "representation_lock",
    ):
        require(
            canonical_sha256(protocol[key]) == invariant[key],
            f"frozen section changed: {key}",
        )

    scientific_primary = {key: primary[key] for key in PRIMARY_SCIENTIFIC_KEYS}
    require(
        canonical_sha256(scientific_primary)
        == invariant["primary_endpoint_scientific_definition"],
        "Primary BDD scientific definition changed",
    )
    require(protocol["treatment"]["trigger_s_route_m"] == 12.0, "trigger changed")
    require(
        protocol["failure_policy"]["minimum_completed_scenarios"] == 76,
        "minimum complete threshold must be 76",
    )
    require(
        paired["primary_minimum_analyzable_pair_count"] == 76,
        "Primary minimum pair count must be 76",
    )
    require(family["theoretical_cells"] == 40, "theoretical secondary matrix must have 40 cells")
    require(family["fixed_secondary_test_count"] == 39, "secondary Holm family must have 39 tests")
    require(family["primary_excluded"] is True, "Primary must be excluded from secondary Holm")
    require(
        primary["multiplicity_label"] == "PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY",
        "Primary multiplicity label mismatch",
    )
    require("B_seed3407" in paired["secondary_representations"], "B dose curve is missing")

    tasks = paired["secondary_task_definitions"]
    lane_types = set(tasks["LAT.LANE_CHANGE"]["pre_treatment_official_scenario_types"])
    dynamics_types = set(tasks["LAT.DYNAMICS"]["pre_treatment_official_scenario_types"])
    require(lane_types and dynamics_types and lane_types.isdisjoint(dynamics_types), "task scopes are not distinct")

    ci = protocol["semantic_uncertainty_reporting"]
    require(ci["method"] == "log_cluster_bootstrap", "semantic CI method changed")
    require(ci["replicates"] == 10000 and ci["seed"] == 620272, "semantic CI freeze mismatch")
    require(ci["mechanism_gate_unchanged"] is True, "mechanism gate must remain unchanged")

    require(authorization["stage7l_d_started"] is False, "Stage7L-D has started")
    require(summary["stage7l_d_started"] is False, "freeze summary says Stage7L-D started")
    require(summary["confirmation_rollout_started"] is False, "confirmation rollout has started")
    require(amendment["change_flags"]["confirmation_result_existed"] is False, "result-free amendment assertion failed")

    for prefix in ("stage7l_d", "stage7l_e"):
        found = sorted((args.repo_root / "outputs").glob(f"{prefix}*"))
        require(not found, f"Unexpected confirmation result path exists for {prefix}: {found}")

    require(
        authorization["frozen_protocol"]["paired_bdd"] == protocol["paired_bdd"],
        "authorization embedded paired_bdd does not match amended protocol",
    )
    require(
        authorization["frozen_protocol"]["failure_policy"] == protocol["failure_policy"],
        "authorization embedded failure_policy does not match amended protocol",
    )
    require(
        authorization["frozen_protocol"]["semantic_uncertainty_reporting"]
        == protocol["semantic_uncertainty_reporting"],
        "authorization embedded semantic CI rule does not match amended protocol",
    )

    assertions = {
        "roster_sha_unchanged": True,
        "scenario_count_equals_80": True,
        "left_equals_15": True,
        "right_equals_65": True,
        "unique_logs_equals_79": True,
        "development_overlap_equals_0": True,
        "development_log_overlap_equals_0": True,
        "dose_unchanged": True,
        "trigger_unchanged": True,
        "eligibility_unchanged": True,
        "checkpoint_sha_unchanged": True,
        "primary_endpoint_unchanged": True,
        "minimum_complete_threshold_equals_76": True,
        "primary_minimum_pair_count_equals_76": True,
        "secondary_test_count_equals_39": True,
        "primary_excluded_from_secondary_holm": True,
        "stage7l_d_started": False,
    }
    return {
        "schema_version": "stage7l_c1_protocol_consistency_validation_v1",
        "status": "STAGE7L_C1_PROTOCOL_CONSISTENCY_AMENDMENT_FROZEN",
        "protocol_sha256": protocol_sha,
        "blind_authorization_sha256": authorization_sha,
        "roster_sha256": roster_sha,
        "hard_assertions": assertions,
        "stage7l_d": "NOT_STARTED",
        "forbidden_work_performed": {
            "planner_rollout": False,
            "embedding_export_or_read": False,
            "bdd_or_mmd": False,
            "training": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=ROOT / "configs/stage7l_c_prospective_confirmation_protocol_v1.json",
    )
    parser.add_argument(
        "--authorization-manifest",
        type=Path,
        default=ROOT / "docs/stage7l_c_blind_confirmation_authorization_manifest_v1.json",
    )
    parser.add_argument(
        "--amendment-manifest",
        type=Path,
        default=ROOT / "docs/stage7l_c1_protocol_consistency_amendment_manifest_v1.json",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_roster.csv",
    )
    parser.add_argument(
        "--development-ledger",
        type=Path,
        default=ROOT / "outputs/stage7l_b_final_development_freeze_v1/stage7l_b_final_prior_exclusion_ledger.csv",
    )
    parser.add_argument(
        "--freeze-summary",
        type=Path,
        default=ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_freeze_summary.json",
    )
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
