#!/usr/bin/env python3
"""Freeze the outcome-blind R2-A DEV identity roster and excitation design."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402
from tools.r1_future_compliant_smoke_selector_v1_3 import (  # noqa: E402
    MAPS,
    ROOT,
    SOURCE,
    _hydrate_candidate,
    _route_continuous_primary80_audit,
    _tail_candidates,
    canonical_sha,
    read_json,
    sha256,
)
from tools.r1_b2_8_r3_prospective_selector import official_count  # noqa: E402
from tools import r1_b2_7_freeze_official_smoke_roster_v2 as frozen  # noqa: E402


R1 = ROOT / "docs/stageR/r1"
R2 = ROOT / "docs/stageR/r2"
R1_EXCLUSION = R1 / "r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json"
R1_EXPOSURE = R1 / "r1_b3_r1_official_outcome_exposure_ledger_v1.0.json"
R1_ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"

OUTPUTS = {
    "exclusion": R2 / "r2_a_controller_id_permanent_exclusion_ledger_v1.0.json",
    "roster": R2 / "r2_a_controller_id_dev_canary_roster_v1.0.json",
    "hlc_grid": R2 / "r2_a_hlc_excitation_grid_v1.0.json",
    "tsb_grid": R2 / "r2_a_tsb_excitation_grid_v1.0.json",
    "selection_audit": R2 / "r2_a_controller_id_selection_audit_v1.0.json",
    "run_ledger": R2 / "r2_a_controller_transfer_run_ledger_v1.0.json",
}


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _effective_firewall() -> Dict[str, Any]:
    historical = read_json(R1_EXCLUSION)
    exposure = read_json(R1_EXPOSURE)
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in historical["entries"]:
        key = (str(row["scenario_token"]), str(row["log_id"]))
        merged[key] = {
            "scenario_token": key[0],
            "log_id": key[1],
            "sources": [str(R1_EXCLUSION.relative_to(ROOT))],
            "reasons": sorted(set(row.get("reasons", [])) | {"HISTORICAL_PERMANENT_EXCLUSION"}),
            "R2_DEVELOPMENT_USE_FORBIDDEN": True,
            "R2_CONFIRMATORY_USE_FORBIDDEN": True,
        }
    for row in exposure["identities"]:
        key = (str(row["scenario_token"]), str(row["log_id"]))
        target = merged.setdefault(
            key,
            {
                "scenario_token": key[0],
                "log_id": key[1],
                "sources": [],
                "reasons": [],
                "R2_DEVELOPMENT_USE_FORBIDDEN": True,
                "R2_CONFIRMATORY_USE_FORBIDDEN": True,
            },
        )
        source = str(R1_EXPOSURE.relative_to(ROOT))
        if source not in target["sources"]:
            target["sources"].append(source)
        if "R1_OFFICIAL_OUTCOME_EXPOSED" not in target["reasons"]:
            target["reasons"].append("R1_OFFICIAL_OUTCOME_EXPOSED")
    entries = sorted(merged.values(), key=lambda row: (row["scenario_token"], row["log_id"]))
    for row in entries:
        row["sources"].sort()
        row["reasons"].sort()
    if len(entries) != 69:
        raise RuntimeError(f"R2_A_EFFECTIVE_FIREWALL_EXPECTED_69:{len(entries)}")
    return {
        "schema_version": "r2_a_controller_id_permanent_exclusion_ledger_v1.0",
        "status": "FROZEN_ADDITIVE_R2_DATA_FIREWALL",
        "match_rule": "EXCLUDE_IF_SCENARIO_TOKEN_OR_LOG_ID_MATCHES",
        "source_ledgers": [
            {"path": str(R1_EXCLUSION.relative_to(ROOT)), "sha256": sha256(R1_EXCLUSION)},
            {"path": str(R1_EXPOSURE.relative_to(ROOT)), "sha256": sha256(R1_EXPOSURE)},
        ],
        "entries": entries,
        "counts": {
            "historical_permanent_exclusions": len(historical["entries"]),
            "R1_official_outcome_exposed": len(exposure["identities"]),
            "effective_unique_identities": len(entries),
        },
        "entry_removal_or_reduction_allowed": False,
    }


def _hlc_grid() -> Dict[str, Any]:
    rows = [
        {
            "excitation_id": "HLC_MONOTONIC_REFERENCE",
            "kind": "MONOTONIC_REFERENCE",
            "diverge_s": 1.1,
            "transition_duration_s": 2.0,
        },
        {
            "excitation_id": "HLC_HESITATION_CENTER",
            "kind": "HESITATION",
            "diverge_s": 1.1,
            "advance_duration_s": 1.4,
            "advance_progress": 0.38,
            "hold_duration_s": 0.6,
            "retreat_depth": 0.16,
            "retreat_duration_s": 1.0,
            "recommit_duration_s": 2.4,
            "nominal_settling_duration_s": 1.4,
        },
        {
            "excitation_id": "HLC_RETREAT_DEPTH_HIGH",
            "kind": "HESITATION",
            "diverge_s": 1.1,
            "advance_duration_s": 1.4,
            "advance_progress": 0.38,
            "hold_duration_s": 0.6,
            "retreat_depth": 0.28,
            "retreat_duration_s": 1.0,
            "recommit_duration_s": 2.4,
            "nominal_settling_duration_s": 1.4,
        },
        {
            "excitation_id": "HLC_RETREAT_DURATION_LONG",
            "kind": "HESITATION",
            "diverge_s": 1.1,
            "advance_duration_s": 1.4,
            "advance_progress": 0.38,
            "hold_duration_s": 0.6,
            "retreat_depth": 0.16,
            "retreat_duration_s": 1.6,
            "recommit_duration_s": 2.4,
            "nominal_settling_duration_s": 0.8,
        },
        {
            "excitation_id": "HLC_RECOMMIT_FAST_SETTLE_LONG",
            "kind": "HESITATION",
            "diverge_s": 1.1,
            "advance_duration_s": 1.4,
            "advance_progress": 0.38,
            "hold_duration_s": 0.6,
            "retreat_depth": 0.16,
            "retreat_duration_s": 1.0,
            "recommit_duration_s": 1.5,
            "nominal_settling_duration_s": 2.3,
        },
    ]
    return {
        "schema_version": "r2_a_hlc_excitation_grid_v1.0",
        "status": "FROZEN_BEFORE_ANY_R2_A_SIMULATION",
        "design": "ONE_FACTOR_AROUND_GEN1_CENTER_WITH_MONOTONIC_REFERENCE",
        "purpose": "CONTROLLER_TRANSFER_IDENTIFICATION_NOT_FINAL_GENERATOR_SELECTION",
        "time_grid_s": 0.1,
        "primary_horizon_s": [0.0, 7.9],
        "excitations": rows,
        "online_adaptation_allowed": False,
        "scientific_threshold_targeting_allowed": False,
    }


def _tsb_grid() -> Dict[str, Any]:
    rows = [
        {
            "excitation_id": "TSB_SINGLE_BRAKE_REFERENCE",
            "kind": "SINGLE_BRAKE_REFERENCE",
            "start_s": 1.1,
            "first_brake_mps2": -1.0,
            "first_brake_duration_s": 0.95,
            "release_mps2": 0.0,
            "release_duration_s": 0.0,
            "second_brake_mps2": 0.0,
            "second_brake_duration_s": 0.0,
        },
        {
            "excitation_id": "TSB_TWO_PULSE_CENTER",
            "kind": "TWO_PULSE",
            "start_s": 1.1,
            "first_brake_mps2": -0.9,
            "first_brake_duration_s": 0.5,
            "release_mps2": 0.4,
            "release_duration_s": 0.7,
            "second_brake_mps2": -0.9,
            "second_brake_duration_s": 0.5,
        },
        {
            "excitation_id": "TSB_BRAKE_AMPLITUDE_HIGH",
            "kind": "TWO_PULSE",
            "start_s": 1.1,
            "first_brake_mps2": -1.5,
            "first_brake_duration_s": 0.5,
            "release_mps2": 0.4,
            "release_duration_s": 0.7,
            "second_brake_mps2": -1.5,
            "second_brake_duration_s": 0.5,
        },
        {
            "excitation_id": "TSB_BRAKE_DURATION_LONG",
            "kind": "TWO_PULSE",
            "start_s": 1.1,
            "first_brake_mps2": -0.9,
            "first_brake_duration_s": 0.9,
            "release_mps2": 0.4,
            "release_duration_s": 0.7,
            "second_brake_mps2": -0.9,
            "second_brake_duration_s": 0.9,
        },
        {
            "excitation_id": "TSB_RELEASE_STRONG_LONG",
            "kind": "TWO_PULSE",
            "start_s": 1.1,
            "first_brake_mps2": -0.9,
            "first_brake_duration_s": 0.5,
            "release_mps2": 1.0,
            "release_duration_s": 1.1,
            "second_brake_mps2": -0.9,
            "second_brake_duration_s": 0.5,
        },
    ]
    return {
        "schema_version": "r2_a_tsb_excitation_grid_v1.0",
        "status": "FROZEN_BEFORE_ANY_R2_A_SIMULATION",
        "design": "ONE_FACTOR_FRACTIONAL_DESIGN_AROUND_GEN1_CENTER_WITH_SINGLE_BRAKE_REFERENCE",
        "purpose": "GAIN_LAG_DURATION_RELEASE_IDENTIFICATION_NOT_GATE_PARAMETER_SELECTION",
        "time_grid_s": 0.1,
        "primary_horizon_s": [0.0, 7.9],
        "excitations": rows,
        "online_adaptation_allowed": False,
        "scientific_threshold_targeting_allowed": False,
    }


def _schedule(entries: list[Mapping[str, Any]], hlc: Mapping[str, Any], tsb: Mapping[str, Any]) -> list[Dict[str, Any]]:
    runs: list[Dict[str, Any]] = []
    by_family = {"R-HLC": hlc["excitations"], "R-TSB": tsb["excitations"]}
    for family in ("R-HLC", "R-TSB"):
        family_entries = [row for row in entries if row["family"] == family]
        for identity_index, entry in enumerate(family_entries, 1):
            for excitation in by_family[family]:
                run_id = f"R2A-{family[2:]}-{identity_index:02d}-{excitation['excitation_id']}"
                runs.append(
                    {
                        "run_order": len(runs) + 1,
                        "run_id": run_id,
                        "family": family,
                        "scenario_token": entry["scenario_token"],
                        "log_id": entry["log_id"],
                        "excitation_id": excitation["excitation_id"],
                        "status": "PLANNED_FROZEN_PRE_EXECUTION",
                        "attempt": 1,
                        "technical_rerun_of": None,
                    }
                )
    if len(runs) != 80 or [row["run_order"] for row in runs] != list(range(1, 81)):
        raise RuntimeError("R2_A_EXPECTED_EXACT_80_RUN_STRUCTURED_DESIGN")
    return runs


def _select_unique_family_suffix(
    family: str,
    cutoff: str,
    firewall: Mapping[str, Any],
    map_cache: Dict[str, Any],
    used_tokens: set[str],
    used_logs: set[str],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    """Select the first eight complete passes while enforcing global log uniqueness."""
    candidates, source_audit = _tail_candidates(family, cutoff, firewall, used_tokens, used_logs)
    selected: list[Dict[str, Any]] = []
    audits: list[Dict[str, Any]] = []
    failures: Counter[str] = Counter()
    local_tokens, local_logs = set(used_tokens), set(used_logs)
    for position, row in enumerate(candidates, 1):
        token, log_id = str(row["scenario_token"]), str(row["log_id"])
        if token in local_tokens or log_id in local_logs:
            failures["GLOBAL_TOKEN_OR_LOG_UNIQUENESS"] += 1
            continue
        candidate = _hydrate_candidate(row)
        resolution = official_count(str(candidate["db_path"]), token)
        if resolution != 1:
            failures["OFFICIAL_EXACT_RESOLUTION_COUNT_NOT_ONE"] += 1
            continue
        try:
            entry, legacy_audit = frozen.evaluate_candidate(candidate, family, MAPS, map_cache)
        except frozen.EligibilityError as exc:
            failures[str(exc)] += 1
            continue
        if family == "R-HLC":
            try:
                route: Any = _route_continuous_primary80_audit(entry, map_cache)
            except Exception as exc:
                failures[f"HLC_ROUTE_CONTINUOUS_PRIMARY80_FAIL:{type(exc).__name__}:{exc}"] += 1
                continue
        else:
            route = "NOT_APPLICABLE_TSB_FROZEN_SEMANTICS_UNCHANGED"
        entry["selector_rank_sha256"] = row["selector_rank_sha256"]
        entry["official_exact_single_scenario_resolution"] = "PASS"
        entry["Primary80_execution_eligibility"] = "PASS_PENDING_EXACT_CONTROLLER_CONFIRMATION_IN_ZERO_RUN"
        if family == "R-HLC":
            entry["HLC_route_continuous_Primary80_applicability"] = "PASS"
        else:
            entry["TSB_frozen_eligibility_semantics"] = "UNCHANGED_PASS"
        selected.append(entry)
        local_tokens.add(token)
        local_logs.add(log_id)
        audits.append(
            {
                "family": family,
                "scenario_token": token,
                "log_id": log_id,
                "selector_rank_sha256": row["selector_rank_sha256"],
                "legacy_frozen_eligibility": legacy_audit,
                "official_resolution_count": resolution,
                "route_continuous": route,
                "deterministic_rank_position_after_R1_stop": position,
            }
        )
        if len(selected) == 8:
            source_audit.update(
                {
                    "evaluated_after_R1_stop": position,
                    "failure_counts": dict(sorted(failures.items())),
                    "selected_count": 8,
                    "stopping_rank_sha256": row["selector_rank_sha256"],
                    "global_unique_logs_enforced": True,
                }
            )
            return selected, audits, source_audit
    raise RuntimeError(f"INSUFFICIENT_UNIQUE_{family}_R2_A_DEV_IDENTITIES:{len(selected)}")


def main() -> int:
    if any(path.exists() for path in OUTPUTS.values()):
        raise FileExistsError("R2_A_VERSIONED_FREEZE_OUTPUT_ALREADY_EXISTS")
    if hashlib.sha256(PROTECTED_CSV.read_bytes()).hexdigest() != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    firewall = _effective_firewall()
    r1_entries = read_json(R1_ROSTER)["entries"]
    cutoffs = {
        family: max(str(row["selector_rank_sha256"]) for row in r1_entries if row["family"] == family)
        for family in ("R-HLC", "R-TSB")
    }
    map_cache: Dict[str, Any] = {}
    used_tokens: set[str] = set()
    used_logs: set[str] = set()
    selected: list[Dict[str, Any]] = []
    audits: list[Dict[str, Any]] = []
    source_audits: Dict[str, Any] = {}
    for family in ("R-HLC", "R-TSB"):
        rows, row_audits, source_audit = _select_unique_family_suffix(
            family,
            cutoffs[family],
            firewall,
            map_cache,
            used_tokens,
            used_logs,
        )
        for row in rows:
            row.update(
                {
                    "PERMANENT_R2_ENGINEERING_ONLY": True,
                    "R2_CONFIRMATORY_USE_FORBIDDEN": True,
                    "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
                    "selection_role": "OUTCOME_BLIND_CONTROLLER_TRANSFER_IDENTIFICATION",
                }
            )
        selected.extend(rows)
        audits.extend(row_audits)
        source_audits[family] = source_audit
        used_tokens.update(row["scenario_token"] for row in rows)
        used_logs.update(row["log_id"] for row in rows)
    forbidden_tokens = {row["scenario_token"] for row in firewall["entries"]}
    forbidden_logs = {row["log_id"] for row in firewall["entries"]}
    if any(row["scenario_token"] in forbidden_tokens or row["log_id"] in forbidden_logs for row in selected):
        raise RuntimeError("R2_A_SELECTED_IDENTITY_OVERLAPS_FIREWALL")
    if len(selected) != 16 or len(used_tokens) != 16 or len(used_logs) != 16:
        raise RuntimeError("R2_A_ROSTER_CARDINALITY_OR_UNIQUENESS_FAIL")
    pre_selection_firewall_sha = canonical_sha(firewall)
    final_firewall = dict(firewall)
    final_firewall["pre_selection_firewall_canonical_sha256"] = pre_selection_firewall_sha
    final_firewall["entries"] = list(firewall["entries"]) + [
        {
            "scenario_token": row["scenario_token"],
            "log_id": row["log_id"],
            "family": row["family"],
            "sources": [str(OUTPUTS["roster"].relative_to(ROOT))],
            "reasons": ["R2_A_CONTROLLER_IDENTIFICATION_DEV_IDENTITY"],
            "OUTCOME_EXPOSED": False,
            "PERMANENT_R2_ENGINEERING_ONLY": True,
            "R2_CONFIRMATORY_USE_FORBIDDEN": True,
            "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
        }
        for row in selected
    ]
    final_firewall["counts"] = {
        **firewall["counts"],
        "R2_A_fresh_engineering_only": 16,
        "effective_unique_identities": len(final_firewall["entries"]),
    }
    final_firewall["selected_DEV_identity_addition_is_post_selection"] = True
    hlc_grid, tsb_grid = _hlc_grid(), _tsb_grid()
    roster = {
        "schema_version": "r2_a_controller_id_dev_canary_roster_v1.0",
        "status": "FROZEN_PENDING_ZERO_RUN_AND_AUTHORIZED_ENGINEERING_EXECUTION",
        "selection_semantics": "CONTINUE_FROZEN_V1_3_HASH_RANK_AFTER_R1_SELECTED_PREFIX",
        "master_seed": 2026082701,
        "salt_sha256": "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9",
        "source_universe": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE), "reused": True},
        "effective_firewall_canonical_sha256": pre_selection_firewall_sha,
        "entries": selected,
        "counts": {"R-HLC": 8, "R-TSB": 8, "total": 16},
        "selection_inputs": [
            "context",
            "map_and_runtime_applicability",
            "route_and_reference_availability",
            "Primary80_technical_eligibility",
            "exact_single_official_scenario_resolution",
        ],
        "forbidden_selection_inputs_used": [],
        "outcome_or_safety_or_F_match_used": False,
        "confirmatory_roster": False,
    }
    runs = _schedule(selected, hlc_grid, tsb_grid)
    frozen_plan_sha = canonical_sha(runs)
    ledger = {
        "schema_version": "r2_a_controller_transfer_run_ledger_v1.0",
        "status": "FROZEN_PRE_EXECUTION",
        "authorization": "R2_DEV_ENGINEERING_SIMULATION_FRESH_R2_A_IDENTITIES_ONLY",
        "roster_canonical_sha256": canonical_sha(roster),
        "hlc_grid_canonical_sha256": canonical_sha(hlc_grid),
        "tsb_grid_canonical_sha256": canonical_sha(tsb_grid),
        "frozen_run_plan_canonical_sha256": frozen_plan_sha,
        "runs": runs,
        "counts": {"planned": len(runs), "executed": 0, "technical_reruns": 0},
        "online_parameter_adaptation": False,
        "identity_replacement_on_mechanism_outcome": False,
        "scientific_simulation": False,
        "RBR_started": False,
    }
    selection_audit = {
        "schema_version": "r2_a_controller_id_selection_audit_v1.0",
        "status": "PASS_OUTCOME_BLIND_FRESH_DEV_SELECTION",
        "source_universe": source_audits,
        "selected_candidate_audits": audits,
        "counts": {"R-HLC": 8, "R-TSB": 8, "total": 16},
        "overlap_with_R1_official": 0,
        "overlap_with_historical_or_canary_blacklist": 0,
        "scientific_mechanism_or_safety_outcome_opened_for_selection": False,
        "scientific_threshold_changed": False,
        "simulation_started": False,
    }
    for key, value in (
        ("exclusion", final_firewall),
        ("roster", roster),
        ("hlc_grid", hlc_grid),
        ("tsb_grid", tsb_grid),
        ("selection_audit", selection_audit),
        ("run_ledger", ledger),
    ):
        write_new(OUTPUTS[key], value)
    print(
        json.dumps(
            {
                "status": selection_audit["status"],
                "pre_selection_firewall_count": len(firewall["entries"]),
                "final_permanent_exclusion_count": len(final_firewall["entries"]),
                "HLC_DEV": 8,
                "TSB_DEV": 8,
                "planned_engineering_runs": len(runs),
                "frozen_run_plan_sha256": frozen_plan_sha,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
