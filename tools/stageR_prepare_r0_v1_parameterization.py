#!/usr/bin/env python3
"""Generate StageR R0 pre-freeze manifests from verified local development assets.

This tool is deliberately read-only with respect to historical outputs.  It reads
the R0 local contract verification result and frozen development metadata, then
writes only new versioned files under docs/stageR/r0/manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MANIFEST_DIR = Path("docs/stageR/r0/manifests")
RAW33_MANIFEST = Path(
    "outputs/stage6r_dynamic_full51_semantic_strict_v1/"
    "stage6r_dynamic_full51_manifest.json"
)
RAW33_LEDGER = Path(
    "outputs/stage6r_dynamic_full51_semantic_strict_v1/"
    "stage6r_full51_sha256_ledger.json"
)
LOCAL_AUDIT = Path(
    "outputs/stageR/r0_local_audit/r0_local_contract_verification.json"
)
RAW33_SCALER = Path(
    "outputs/stage6t_training_evaluation_protocol_freeze_v1/"
    "stage6t_global_interaction_target_standardization.json"
)
EGO13_SCALER = Path(
    "outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired/"
    "scalers/handcrafted_reference_scalers.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def make_raw33_provenance(root: Path, audit: dict[str, Any]) -> None:
    dataset = next(
        row
        for row in audit["datasets"]
        if row["id"] == "waymo_dynamic_interaction_full51_semantic_strict_v1"
    )
    rows = []
    for item in dataset["per_shard_assets"]:
        if Path(item["path"]).name != "interaction_feat_style_raw.npy":
            continue
        path = Path(item["path"])
        part = next(piece for piece in path.parts if piece.startswith("stage6r_dynamic_full51_semantic_strict_part_"))
        shard = next(piece for piece in path.parts if piece.startswith("shard_"))
        rows.append(
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "part": part,
                "shard": shard,
                "row_count": item["shape"][0],
                "shape": json.dumps(item["shape"], separators=(",", ":")),
                "dtype": item["dtype"],
                "current_manifest": str(RAW33_MANIFEST),
                "current_manifest_sha256": dataset["manifest_sha256"],
                "builder_path": "tools/build_waymo_dynamic_interaction_dataset_v2.py",
                "builder_sha256": "1c0f8d77caf0b48a37fe47c673a4f9b293902fcbf1a58ada159f4572c90d1b79",
                "feature_code_path": "tools/interaction_context_features.py",
                "feature_code_sha256": "ccc6c149f9fa4d9ce7ac541c300415c7c4cc0b43dcb0cee827141fc865ef7293",
                "code_introducing_commit": "fa37948ce909ef83930fb34ef65342b912af93cb",
                "code_binding_scope": "CURRENT_AND_GIT_TRACEABLE_NOT_ARTIFACT_BOUND_AT_GENERATION",
                "discovered_at_utc": audit["generated_at_utc"],
                "current_file_provenance_status": "CURRENT_FILE_PROVENANCE_VERIFIED",
                "historical_ledger_status": "HISTORICAL_LEDGER_ENTRY_NOT_AVAILABLE",
                "historical_ledger": str(RAW33_LEDGER),
                "historical_ledger_sha256": dataset["sha_ledger_sha256"],
                "historical_sha_claimed": "false",
            }
        )
    rows.sort(key=lambda row: (row["part"], row["shard"]))
    if len(rows) != 36 or sum(int(row["row_count"]) for row in rows) != 168700:
        raise ValueError("Expected exactly 36 raw33 shards and 168700 rows")
    write_csv(
        root / MANIFEST_DIR / "r0_raw33_provenance_addendum_v0.1.csv",
        rows,
        list(rows[0]),
    )


EGO13 = [
    ("mean_speed", "Mean speed over valid frames", "m/s"),
    ("std_speed", "Population standard deviation of speed over valid frames", "m/s"),
    ("p95_speed", "95th percentile speed over valid frames", "m/s"),
    ("end_minus_start_speed", "Last valid speed minus first valid speed", "m/s"),
    ("rms_accel", "RMS of speed-derived acceleration with dt=0.1 s", "m/s^2"),
    ("mean_abs_accel", "Mean absolute speed-derived acceleration", "m/s^2"),
    ("p95_abs_accel", "95th percentile absolute speed-derived acceleration", "m/s^2"),
    ("rms_jerk", "RMS of acceleration-derived jerk", "m/s^3"),
    ("p95_abs_jerk", "95th percentile absolute acceleration-derived jerk", "m/s^3"),
    ("rms_yaw_rate", "RMS of wrapped-heading-derived yaw rate", "rad/s"),
    ("mean_abs_yaw_rate", "Mean absolute wrapped-heading-derived yaw rate", "rad/s"),
    ("heading_change_abs_total", "Sum of absolute wrapped heading increments", "rad"),
    ("path_length", "Sum of xy displacement norms across valid frames", "m"),
]


RAW33 = [
    ("rms_accel", "RMS ego acceleration", "m/s^2", "longitudinal"),
    ("rms_jerk", "RMS ego jerk", "m/s^3", "longitudinal"),
    ("max_abs_accel", "Maximum absolute ego acceleration", "m/s^2", "longitudinal"),
    ("max_abs_jerk", "Maximum absolute ego jerk", "m/s^3", "longitudinal"),
    ("mean_thw", "Mean valid-front time headway", "s", "longitudinal"),
    ("min_thw", "Minimum valid-front time headway", "s", "longitudinal"),
    ("mean_front_distance", "Mean valid-front distance", "m", "longitudinal"),
    ("min_front_distance", "Minimum valid-front distance", "m", "longitudinal"),
    ("mean_rel_speed", "Mean valid-front closing-rate proxy", "m/s", "longitudinal"),
    ("p95_rel_speed", "95th percentile valid-front closing-rate proxy", "m/s", "longitudinal"),
    ("rms_yaw_rate", "RMS ego yaw-rate proxy", "rad/s", "lateral"),
    ("rms_curvature", "RMS yaw-rate/speed where abs(speed)>=0.5", "1/m", "lateral"),
    ("heading_change_total", "Total absolute wrapped heading change", "rad", "lateral"),
    ("lane_change_count_proxy", "Count of lateral-offset threshold entries", "count", "lateral"),
    ("lane_change_rate_proxy", "Lane-change count divided by window duration", "1/s", "lateral"),
    ("lane_change_left_count_proxy", "Integer proxy split attributed left", "count", "lateral"),
    ("lane_change_right_count_proxy", "Integer proxy split attributed right", "count", "lateral"),
    ("lane_change_duration_mean_proxy", "Mean transition indicator times dt", "s/frame proxy", "lateral"),
    ("max_lateral_speed", "Maximum absolute local lateral speed", "m/s", "lateral"),
    ("rms_lateral_accel", "RMS per-frame lateral-speed increment", "m/s per frame", "lateral"),
    ("lane_change_oscillation_score_proxy", "Mean absolute change of lane-change proxy", "fraction", "lateral"),
    ("front_pressure_score", "Mean clipped max(0,30-front_distance)", "m proxy", "interaction_context"),
    ("left_front_min_gap", "Minimum left-front gap", "m", "interaction_context"),
    ("left_rear_min_gap", "Minimum left-rear gap", "m", "interaction_context"),
    ("right_front_min_gap", "Minimum right-front gap", "m", "interaction_context"),
    ("right_rear_min_gap", "Minimum right-rear gap", "m", "interaction_context"),
    ("left_gap_min", "Minimum of left-front and left-rear gaps", "m", "interaction_context"),
    ("right_gap_min", "Minimum of right-front and right-rear gaps", "m", "interaction_context"),
    ("left_gap_acceptance_proxy", "Fraction with positive left-front closing rate", "fraction", "interaction_context"),
    ("right_gap_acceptance_proxy", "Fraction with positive right-front closing rate", "fraction", "interaction_context"),
    ("rear_vehicle_pressure_proxy", "Mean max rear closing-rate proxy", "m/s proxy", "interaction_context"),
    ("yielding_score_proxy", "Mean clipped closing-rate/front-distance ratio", "1/s proxy", "interaction_context"),
    ("assertiveness_score_proxy", "Fraction of frames above sequence mean speed", "fraction", "interaction_context"),
]


LONGITUDINAL = [
    ("ego_speed_smoothed", "Median-filtered ego speed", "m/s"),
    ("ego_longitudinal_accel", "Finite-difference longitudinal acceleration", "m/s^2"),
    ("ego_longitudinal_jerk", "Finite-difference longitudinal jerk", "m/s^3"),
]


F_MATCH = [f"ego13.{name}" for name, _, _ in EGO13] + [
    "raw33.mean_thw",
    "raw33.min_thw",
    "raw33.mean_front_distance",
    "raw33.min_front_distance",
    "raw33.mean_rel_speed",
    "raw33.p95_rel_speed",
    "raw33.front_pressure_score",
    "raw33.left_front_min_gap",
    "raw33.left_rear_min_gap",
    "raw33.right_front_min_gap",
    "raw33.right_rear_min_gap",
]


M_BEHAVIOR = [
    "raw33.lane_change_count_proxy",
    "raw33.lane_change_rate_proxy",
    "raw33.lane_change_left_count_proxy",
    "raw33.lane_change_right_count_proxy",
    "raw33.lane_change_duration_mean_proxy",
    "raw33.lane_change_oscillation_score_proxy",
    "raw33.left_gap_acceptance_proxy",
    "raw33.right_gap_acceptance_proxy",
    "raw33.rear_vehicle_pressure_proxy",
    "raw33.yielding_score_proxy",
    "raw33.assertiveness_score_proxy",
]


def target_record(
    target_id: str,
    name: str,
    family: str,
    definition: str,
    source_path: str,
    unit: str,
    valid_frame_rule: str,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "name": name,
        "family": family,
        "definition": definition,
        "source_path": source_path,
        "unit": unit,
        "valid_frame_rule": valid_frame_rule,
        "independence_unit": "scenario; log-clustered when multiple scenarios share a log",
        "used_for": {
            "semantic_probe": True,
            "mechanism_validation": target_id in M_BEHAVIOR,
            "matching": target_id in F_MATCH,
            "leakage_audit": family == "interaction_context",
        },
        "descriptor_role": (
            "F_match" if target_id in F_MATCH else "M_behavior" if target_id in M_BEHAVIOR else "NEITHER"
        ),
    }


def make_target_definition(root: Path) -> None:
    targets = []
    for name, definition, unit in EGO13:
        targets.append(
            target_record(
                f"ego13.{name}", name, "ego13", definition,
                "tools/stage6l_prepare_context_representation_ablation.py:ego_kinematic_features",
                unit, "all frames selected by ego_seq_mask; require at least 2 valid frames; dt=0.1 s",
            )
        )
    for name, definition, unit, family in RAW33:
        targets.append(
            target_record(
                f"raw33.{name}", name, family, definition,
                "tools/interaction_context_features.py:aggregate_interaction_features",
                unit,
                "ego uses the supplied sequence; neighbor aggregates use slot presence>0.5 and NaN-safe reduction; empty valid neighbor set maps to 0.0",
            )
        )
    for name, definition, unit in LONGITUDINAL:
        targets.append(
            target_record(
                f"longitudinal_v2.{name}", name, "longitudinal", definition,
                "outputs/stage6r_dynamic_full51_semantic_strict_v1/longitudinal_supervision_v2_global_schema.json",
                unit,
                "all finite frames in each 80-frame row; train-q01/q99 winsorization and train median/IQR normalization are target preprocessing, not input preprocessing",
            )
        )
    if set(F_MATCH) & set(M_BEHAVIOR):
        raise ValueError("F_match and M_behavior must be disjoint")
    payload = {
        "schema_version": "r0_target_definition_v0.1",
        "status": "COMPLETE_DEFINITION_DRAFT",
        "evidence_level": "R0_DEVELOPMENT_CONTRACT_EVIDENCE",
        "dt_seconds": 0.1,
        "targets": targets,
        "descriptor_sets": {
            "F_match": {
                "purpose": "descriptor/context equivalence and matching only",
                "target_ids": F_MATCH,
            },
            "M_behavior": {
                "purpose": "mechanism qualification only; excluded from core F_match equivalence",
                "target_ids": M_BEHAVIOR,
            },
            "intersection": [],
        },
        "normalization_contracts": {
            "learned_encoder_input": "NONE",
            "raw33_target": "train-only population mean/std; epsilon floor 1e-6",
            "longitudinal_v2_target": "train q01/q99 winsorize then train median/IQR",
            "ego13_stage6l_representation": "dose100 conservative reference median/IQR; valid-mask aggregation",
        },
    }
    write_json(root / MANIFEST_DIR / "r0_target_definition_v0.1.json", payload)


PARAMETERS = [
    ("P001", "Global", "alpha", "0.05", "PHYSICAL_RATIONALE", "Conventional two-sided error rate; family definitions and Holm correction prevent implicit pooling", "READY_FOR_FREEZE"),
    ("P002", "Global", "confidence_level", "0.95", "PHYSICAL_RATIONALE", "Matches alpha=0.05 two-sided inference", "READY_FOR_FREEZE"),
    ("P003", "Global", "multiplicity", "Holm within each predeclared module/family; intersection-union for multi-feature equivalence", "COMPUTATIONAL_PRECISION", "Strong FWER control without assuming independence", "READY_FOR_FREEZE"),
    ("P004", "Global", "bootstrap_repetitions", "5000", "COMPUTATIONAL_PRECISION", "Stable percentile/BCa endpoints with cluster resampling; increase only by predeclared precision rule", "READY_FOR_FREEZE"),
    ("P005", "Global", "permutation_repetitions", "49999", "COMPUTATIONAL_PRECISION", "Minimum attainable plus-one p=0.00002 and MC SE near 0.001 at p=0.05", "READY_FOR_FREEZE"),
    ("P006", "D0", "event_position_bins", "early=0:49; middle=50:99; late=100:149 (frame indices)", "PHYSICAL_RATIONALE", "Three equal 5 s supports at dt=0.1 s; boundaries fixed without outcome access", "READY_FOR_FREEZE"),
    ("P007", "D0", "minimum_temporal_effect", "absolute paired standardized retention difference >=0.10 plus 95% CI excluding 0 and >=2/3 seed direction", "DEVELOPMENT_ESTIMATION", "Small-effect proposal; scientific materiality still requires owner approval", "REQUIRES_SCIENTIFIC_OWNER_APPROVAL"),
    ("P008", "D0/D1", "linear_probe_grid", "ridge alpha={1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000}; fixed linear/logistic family", "COMPUTATIONAL_PRECISION", "Bounded log grid and fixed-capacity refit contract", "READY_FOR_FREEZE"),
    ("P009", "D3", "projection_rank_candidates", "{1,2,4,8,16}; max_rank=16", "COMPUTATIONAL_PRECISION", "Bounded powers-of-two set prevents 64D dimension fishing", "READY_FOR_FREEZE"),
    ("P010", "D3", "projection_rank_selection", "development semantic-retention score subject to null-calibration gate; smallest rank wins ties within 1 SE", "PHYSICAL_RATIONALE", "Favors parsimonious stable readout and isolates audit holdout", "READY_FOR_FREEZE"),
    ("P011", "D3", "primary_kernel", "single RBF", "HISTORICAL_VARIABILITY", "Matches verified Stage7L/Stage6P primary family", "READY_FOR_FREEZE"),
    ("P012", "D3", "bandwidth_rule", "per representation/readout fixed positive off-diagonal median distance on treatment-label-blind R0_DEVELOPMENT reference bank", "DEVELOPMENT_ESTIMATION", "Avoids treatment-cell adaptive bandwidth and is frozen before audit", "READY_FOR_FREEZE"),
    ("P013", "D2", "context_shuffle_matching_strata", "scenario_family x lane_change_direction x initial_speed_tertile x traffic_density_tertile x neighbor_availability_pattern x event_phase_bin", "PHYSICAL_RATIONALE", "All are pre-treatment/context variables; sparse cells follow fixed coarsening order", "READY_FOR_FREEZE"),
    ("P014", "D2", "sparse_strata_rule", "minimum 4 independent units; coarsen event_phase then density then speed; never cross scenario_family", "COMPUTATIONAL_PRECISION", "Prevents singleton shuffle and preserves task family", "READY_FOR_FREEZE"),
    ("P015", "D2", "ood_boundary", "reference 99th percentile per metric; OOD_DOMINATED if >=2 of 4 metrics exceed their boundary", "DEVELOPMENT_ESTIMATION", "Bounded multimetric diagnostic estimated without candidate outcomes", "READY_FOR_FREEZE"),
    ("P016", "D1", "continuous_target_reporting", "held-out R2 primary; MAE/NRMSE, Spearman, calibration slope secondary; log-cluster 95% CI", "PHYSICAL_RATIONALE", "Target-level effects retain units and uncertainty", "READY_FOR_FREEZE"),
    ("P017", "D1", "categorical_target_reporting", "balanced_accuracy primary; AUROC and macro-F1 secondary; log-cluster 95% CI", "PHYSICAL_RATIONALE", "Avoids prevalence-driven accuracy", "READY_FOR_FREEZE"),
    ("P018", "D3", "calibration_fpr_gate", "upper 95% CI <=0.075 at nominal 0.05", "MEASUREMENT_PRECISION", "Proposal allows finite-sample uncertainty while limiting material calibration inflation", "REQUIRES_SCIENTIFIC_OWNER_APPROVAL"),
]


def make_parameterization(root: Path) -> None:
    rows = [
        {
            "parameter_id": pid,
            "module": module,
            "parameter": parameter,
            "proposal": proposal,
            "rationale_category": category,
            "basis": basis,
            "status": status,
            "data_tier": "R0_DEVELOPMENT_ONLY",
            "future_outcome_used": "false",
        }
        for pid, module, parameter, proposal, category, basis, status in PARAMETERS
    ]
    write_csv(root / MANIFEST_DIR / "r0_parameterization_proposal_v0.1.csv", rows, list(rows[0]))


def make_holdout_inventory(root: Path) -> None:
    rows = [
        ("H001", "Waymo Dynamic-v2 train", "24872-scenario source; split=train", "A/B/C training and R0 development", "true", "true", "DIRECT_TRAIN_OVERLAP", "Stage6T/U/V", "scenario; source file/segment grouping", "AVAILABLE", "INELIGIBLE"),
        ("H002", "Waymo Dynamic-v2 val", "split=val", "A/B/C model selection", "true", "true", "DIRECT_VAL_OVERLAP", "Stage6T/U", "scenario; source file/segment grouping", "AVAILABLE", "INELIGIBLE"),
        ("H003", "Waymo Dynamic-v2 historical test", "split=test; 16784 rows", "Stage6V one-time unblinded evaluation", "false", "true", "DIRECT_HISTORICAL_TEST_OVERLAP", "Stage6V", "scenario; source file/segment grouping", "AVAILABLE", "INELIGIBLE"),
        ("H004", "Stage6P frozen 1600-row pool", "800 pairs; 489 logs", "unpaired release calibration/evaluation", "false", "true", "HISTORICAL_EVALUATION_POOL", "Stage6P/Stage6V", "log-disjoint release groups", "AVAILABLE", "INELIGIBLE"),
        ("H005", "Stage7/Stage7 M6 assets", "historical nuPlan scenario/token rosters", "data-quality, BDD, mechanism and confirmation analyses", "false", "true", "NO_WAYMO_SPLIT_CLAIM", "Stage7 and Stage7 M6", "scenario pair; log cluster", "AVAILABLE", "INELIGIBLE"),
        ("H006", "Stage7L E3 roster", "80 scenario rows per dose", "development, one-time confirmation and prospective BDD", "false", "true", "NO_WAYMO_SPLIT_CLAIM", "Stage7L A-E", "same-scenario treatment pair; log cluster", "AVAILABLE", "INELIGIBLE"),
        ("H007", "Existing nuPlan database remainder", "UNSCREENED_REMAINDER; authoritative unused token IDs not frozen", "unknown", "unknown", "unknown", "NO_WAYMO_SPLIT_CLAIM", "overlap cannot be excluded against all Stage6/7/7L ledgers", "scenario; log cluster", "SOURCE_MAY_EXIST_IDENTITY_LEDGER_NOT_AVAILABLE", "NOT_EVALUABLE"),
    ]
    output = []
    for values in rows:
        candidate_id, source, ids, prior, selection, representation, overlap_split, overlap_stages, unit, availability, eligibility = values
        output.append({
            "candidate_id": candidate_id,
            "source_dataset": source,
            "log_scenario_token_id": ids,
            "prior_historical_use": prior,
            "ever_used_for_model_selection": selection,
            "ever_used_for_representation_evaluation": representation,
            "overlap_with_train_val_historical_test": overlap_split,
            "overlap_with_stage6_stage7_stage7l": overlap_stages,
            "independence_unit": unit,
            "availability": availability,
            "eligibility": eligibility,
            "selection_basis": "identity/source/history/split/runnability only",
            "candidate_representation_performance_read": "false",
            "overall_conclusion": "R0_AUDIT_HOLDOUT_UNAVAILABLE_FROM_EXISTING_ASSETS",
        })
    write_csv(root / MANIFEST_DIR / "r0_audit_holdout_candidate_inventory_v0.1.csv", output, list(output[0]))


def make_future_pool(root: Path) -> None:
    rows = [
        {
            "candidate_id": "R4A_EXISTING_UNUSED_POOL",
            "source_class": "A_UNUSED_EXISTING_SOURCE_TOKEN_POOL",
            "source_or_generator": "existing nuPlan/other source remainder",
            "scenario_source_rule": "must have authoritative source inventory and zero overlap with all Stage6/7/7L and R0 ledgers",
            "token_selection_rule": "hash-sorted eligible tokens selected before treatment; fixed seed 2026082601",
            "treatment_family": "NOT_APPLICABLE_UNTIL_SOURCE_LOCK",
            "dose_parameter_family": "NOT_APPLICABLE_UNTIL_SOURCE_LOCK",
            "exclusions": "prior use; identity overlap; missing source/log metadata; known technical unrunnability; protocol exclusions fixed pre-treatment",
            "runnability_rule": "pre-treatment configuration/import/map availability only; no realized mechanism or representation outcome",
            "independence_unit": "scenario with log-disjoint allocation",
            "outcome_blind": "true",
            "availability": "NOT_AVAILABLE_AUTHORITATIVE_UNUSED_LEDGER_MISSING",
            "eligibility": "NOT_AVAILABLE",
        },
        {
            "candidate_id": "R4B_PROSPECTIVE_CONTROLLED_PLANNER_V1",
            "source_class": "B_PROSPECTIVE_CONTROLLED_PLANNER",
            "source_or_generator": "prospective controlled planner rollouts from a pre-locked scenario source",
            "scenario_source_rule": "lock dataset release/map set/log roster and SHA before any treatment rollout",
            "token_selection_rule": "hash-sort all pre-treatment eligible tokens; select whole fixed roster with seed 2026082601",
            "treatment_family": "R-HLC hesitation; R-TSB two-stage braking; R-IP interaction probing, each defined before rollout",
            "dose_parameter_family": "bounded planner parameter grids frozen by family before rollout; no realized-effect adaptation",
            "exclusions": "pre-treatment identity overlap; missing context; unsupported map/config; deterministic preflight failure only",
            "runnability_rule": "pre-treatment smoke/preflight only; weak realized mechanism is not technical failure",
            "independence_unit": "scenario; log-clustered; whole-roster intention-to-evaluate",
            "outcome_blind": "true",
            "availability": "RULESET_DRAFT_SOURCE_AND_ROSTER_NOT_LOCKED",
            "eligibility": "NOT_AVAILABLE",
        },
    ]
    write_csv(root / MANIFEST_DIR / "r0_future_r4_reserved_pool_candidate_v0.1.csv", rows, list(rows[0]))


def make_equivalence(root: Path) -> None:
    raw_scaler = read_json(root / RAW33_SCALER)
    raw_std = dict(zip((name for name, _, _, _ in RAW33), raw_scaler["std"]))
    ego_npz = np.load(root / EGO13_SCALER)
    ego_iqr = dict(zip((name for name, _, _ in EGO13), ego_npz["ego_scale"].tolist()))
    target_lookup = {
        f"ego13.{name}": (definition, unit) for name, definition, unit in EGO13
    }
    target_lookup.update({
        f"raw33.{name}": (definition, unit) for name, definition, unit, _ in RAW33
    })
    rows = []
    for target_id in F_MATCH:
        namespace, name = target_id.split(".", 1)
        definition, unit = target_lookup[target_id]
        if namespace == "ego13":
            variability = f"Stage6L dose100 conservative reference IQR={ego_iqr[name]:.9g} {unit}; n=183"
            variability_source = "HISTORICAL_VARIABILITY"
        else:
            variability = f"Waymo Dynamic-v2 train population SD={raw_std[name]:.9g} {unit}; n=135046"
            variability_source = "HISTORICAL_VARIABILITY"
        rows.append({
            "target_id": target_id,
            "feature": name,
            "definition": definition,
            "physical_scale_unit": unit,
            "measurement_noise_reproducibility": "NOT_QUANTIFIED_FROM_REPEATED_MEASUREMENT",
            "historical_natural_variability": variability,
            "potential_business_behavior_relevance": "candidate matching descriptor; material tolerance requires scientific/domain owner judgment",
            "proposed_equivalence_margin": "",
            "rationale_category": variability_source,
            "status": "REQUIRES_SCIENTIFIC_OWNER_APPROVAL",
            "power_only_margin": "false",
        })
    write_csv(root / MANIFEST_DIR / "r0_equivalence_margin_proposal_v0.1.csv", rows, list(rows[0]))


def make_sap(root: Path) -> None:
    hypotheses = []
    for module, ids in {
        "D0": ["D0_LENGTH_EFFECT", "D0_POSITION_RETENTION_ASSOCIATION", "D0_POOLING_EFFECT", "D0_MASK_PADDING_SENSITIVITY"],
        "D1": ["D1_KNOWN_SEMANTIC_INFORMATION_PRESENT", "D1_CROSS_DOMAIN_SEMANTIC_TRANSFER", "D1_GEOMETRY_DEGENERACY"],
        "D2": ["D2_RESPONSE_SENSITIVITY", "D2_CONTEXT_SENSITIVITY", "D2_PAIRING_SENSITIVITY", "D2_SHORTCUT_RISK", "D2_ABLATION_OOD_RISK"],
        "D3": ["D3_FULL64_SIGNAL_DILUTION", "D3_PROJECTED_READOUT_GAIN", "D3_NULL_CALIBRATION_PRESERVED"],
        "D4": [
            "D4_DESCRIPTOR_EQUIVALENCE_R_HLC", "D4_MECHANISM_DIFFERENCE_R_HLC", "D4_OUTCOME_BLIND_FEASIBILITY_R_HLC",
            "D4_DESCRIPTOR_EQUIVALENCE_R_TSB", "D4_MECHANISM_DIFFERENCE_R_TSB", "D4_OUTCOME_BLIND_FEASIBILITY_R_TSB",
            "D4_DESCRIPTOR_EQUIVALENCE_R_IP", "D4_MECHANISM_DIFFERENCE_R_IP", "D4_OUTCOME_BLIND_FEASIBILITY_R_IP",
        ],
    }.items():
        for hypothesis_id in ids:
            hypotheses.append({
                "hypothesis_id": hypothesis_id,
                "module": module,
                "analysis_family": module,
                "alpha": 0.05,
                "multiplicity": "Holm within module/family; D4 equivalence uses intersection-union across frozen F_match",
                "independence_unit": "scenario or same-scenario pair; log cluster where repeated scenarios share a log",
                "split_unit": "scenario/source grouping; no cross-role identity overlap",
                "bootstrap_cluster": "log (scenario if no log identity exists, explicitly downgraded)",
                "permutation_unit": "same-scenario pair for paired tests; log-disjoint group label for eligible unpaired tests",
                "probe_family": "fixed-capacity linear ridge/logistic; both frozen-probe-across-view and same-capacity-refit-per-view for D0",
                "kernel_family": "single RBF primary",
                "bandwidth_rule": "fixed per representation/readout from treatment-label-blind R0_DEVELOPMENT reference bank positive off-diagonal median",
                "projection_candidate_rule": "ranks {1,2,4,8,16}; smallest within 1 SE after semantic-retention and null-calibration gates",
                "equivalence_method": "TOST or two-sided 90% CI fully inside owner-approved frozen margin; intersection-union across F_match",
                "status_model": "execution COMPLETE/BLOCKED; hypothesis SUPPORTED/NOT_SUPPORTED/MIXED/INCONCLUSIVE/NOT_EVALUABLE",
                "allowed_evidence_level": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE until an eligible R0_AUDIT_HOLDOUT exists",
            })
    payload = {
        "schema_version": "r0_statistical_analysis_plan_v0.1",
        "status": "DRAFT_NOT_FROZEN",
        "data_roles": {
            "parameterization": "R0_DEVELOPMENT_ONLY",
            "r0_audit_holdout": "NOT_AVAILABLE",
            "future_r4_reserved_pool": "NOT_AVAILABLE",
        },
        "global": {
            "alpha": 0.05,
            "confidence_level": 0.95,
            "bootstrap_repetitions": 5000,
            "permutation_repetitions": 49999,
            "fixed_seeds": [2026082601, 3407, 3408, 3409],
            "missing_value_policy": "preserve mask/sentinel semantics; no silent imputation; target-specific rules recorded",
            "outlier_policy": "no outcome-driven trimming; predeclared physical/quality exclusions only",
            "ci_method": "log-cluster bootstrap; BCa when estimable, percentile fallback declared",
            "evidence_level_rule": "no confirmatory wording without R0_AUDIT_HOLDOUT",
        },
        "d0": {
            "event_position_bins": {"early": [0, 49], "middle": [50, 99], "late": [100, 149]},
            "pooling_set": ["last", "mean", "max"],
            "historical_reference": "T150 + final hidden + historical mask/padding behavior",
            "content_windows": ["first80", "last80", "event80", "overlap80", "full_native"],
            "minimum_temporal_effect": "proposal requires scientific owner approval",
        },
        "d2": {
            "matching_strata": ["scenario_family", "lane_change_direction", "initial_speed_tertile", "traffic_density_tertile", "neighbor_availability_pattern", "event_phase_bin"],
            "ood_boundary": "reference q99 per metric; OOD_DOMINATED when >=2/4 exceed",
        },
        "d3": {
            "projection_ranks": [1, 2, 4, 8, 16],
            "maximum_rank": 16,
            "primary_kernel": "RBF",
            "calibration_fpr_gate": "proposal requires scientific owner approval",
        },
        "d4": {
            "f_match_target_ids": F_MATCH,
            "m_behavior_target_ids": M_BEHAVIOR,
            "equivalence_margin_status": "REQUIRES_SCIENTIFIC_OWNER_APPROVAL_ALL_FEATURES",
            "whole_roster_primary": True,
        },
        "hypotheses": hypotheses,
        "training_authorization": {
            "RBR_A": "NOT_AUTHORIZED",
            "RBR_B": "NOT_AUTHORIZED",
            "RBR_C": "NOT_AUTHORIZED",
        },
    }
    write_json(root / MANIFEST_DIR / "r0_statistical_analysis_plan_v0.1.json", payload)


def make_readiness(root: Path) -> None:
    rows = [
        ("R001", "Global", "raw33_current_file_provenance", "READY", "36 current SHAs/rows/shapes bound in non-destructive addendum", "none", "NO"),
        ("R002", "Global", "raw33_historical_ledger_entry", "HISTORICAL_LEDGER_ENTRY_NOT_AVAILABLE", "historical ledger remains unchanged", "retain explicit limitation", "NO"),
        ("R003", "D0", "temporal_audit_policy", "READY_FOR_FREEZE", "D0-A/B/C/D and interpretation levels drafted", "scientific owner approval of minimum effect", "YES"),
        ("R004", "D0", "mask_padding_audit_policy", "READY_FOR_FREEZE", "historical/diagnostic/future layers and measured baseline drafted", "execute diagnostics after v1 freeze", "YES"),
        ("R005", "D0", "minimum_temporal_effect", "REQUIRES_SCIENTIFIC_OWNER_APPROVAL", "0.10 standardized paired effect proposal", "approve or replace using domain materiality", "YES"),
        ("R006", "D0", "event_position_bins", "READY_FOR_FREEZE", "equal 5-second bins at dt=0.1", "owner freeze", "YES"),
        ("R007", "D1", "target_definition", "READY_FOR_FREEZE", "49 targets with roles/units/rules; F_match and M_behavior disjoint", "owner freeze", "YES"),
        ("R008", "Global", "alpha_multiplicity", "READY_FOR_FREEZE", "alpha .05 and Holm by predeclared family", "owner freeze", "YES"),
        ("R009", "Global", "bootstrap_repetitions", "READY_FOR_FREEZE", "5000 proposal", "owner freeze", "YES"),
        ("R010", "Global", "permutation_repetitions", "READY_FOR_FREEZE", "49999 plus-one proposal", "owner freeze", "YES"),
        ("R011", "D1", "probe_capacity_grid", "READY_FOR_FREEZE", "bounded linear ridge/logistic grid", "owner freeze", "YES"),
        ("R012", "D2", "matching_strata", "READY_FOR_FREEZE", "outcome-blind fixed strata/coarsening", "owner freeze", "YES"),
        ("R013", "D2", "ood_boundary", "READY_FOR_FREEZE", "reference q99 and 2-of-4 proposal", "owner freeze", "YES"),
        ("R014", "D3", "kernel_bandwidth", "READY_FOR_FREEZE", "single RBF and fixed label-blind reference median", "owner freeze", "YES"),
        ("R015", "D3", "projection_rank_rule", "READY_FOR_FREEZE", "{1,2,4,8,16}, max16, smallest-within-1SE", "owner freeze", "YES"),
        ("R016", "D3", "calibration_fpr_gate", "REQUIRES_SCIENTIFIC_OWNER_APPROVAL", "upper 95% CI <=.075 proposal", "approve material inflation tolerance", "YES"),
        ("R017", "D4", "equivalence_inference_method", "READY_FOR_FREEZE", "TOST/90% CI plus intersection-union", "owner freeze", "YES"),
        ("R018", "D4", "equivalence_margins", "REQUIRES_SCIENTIFIC_OWNER_APPROVAL", "measurement repeatability/business tolerances unavailable", "approve each of 24 F_match margins", "YES"),
        ("R019", "D4", "r0_audit_holdout", "NOT_AVAILABLE", "all verified pools used/unblinded; unused identity ledger absent", "acquire new disjoint data and freeze identities", "YES"),
        ("R020", "D4/R4", "future_r4_reserved_pool", "NOT_AVAILABLE", "prospective rules drafted but source/token roster not locked", "lock source and roster before RBR training", "YES"),
        ("R021", "Global", "sap", "DRAFT_READY_FOR_REVIEW", "machine and human v0.1 created", "resolve blocking fields and freeze", "YES"),
        ("R022", "Global", "rbr_training_authorization", "NOT_AUTHORIZED", "holdout/reserved/equivalence/operational freeze unresolved", "do not train RBR-A/B/C", "YES"),
    ]
    output = [{
        "readiness_id": pid,
        "module": module,
        "parameter_or_contract": parameter,
        "readiness_status": status,
        "verified_basis": basis,
        "remaining_work": work,
        "blocking": blocking,
        "future_outcome_used": "false",
    } for pid, module, parameter, status, basis, work, blocking in rows]
    write_csv(root / MANIFEST_DIR / "r0_v1_numerical_freeze_readiness_v0.2.csv", output, list(output[0]))


def validate_outputs(root: Path) -> None:
    json_paths = [
        root / MANIFEST_DIR / "r0_target_definition_v0.1.json",
        root / MANIFEST_DIR / "r0_statistical_analysis_plan_v0.1.json",
    ]
    csv_paths = [
        root / MANIFEST_DIR / "r0_raw33_provenance_addendum_v0.1.csv",
        root / MANIFEST_DIR / "r0_parameterization_proposal_v0.1.csv",
        root / MANIFEST_DIR / "r0_audit_holdout_candidate_inventory_v0.1.csv",
        root / MANIFEST_DIR / "r0_future_r4_reserved_pool_candidate_v0.1.csv",
        root / MANIFEST_DIR / "r0_equivalence_margin_proposal_v0.1.csv",
        root / MANIFEST_DIR / "r0_v1_numerical_freeze_readiness_v0.2.csv",
    ]
    for path in json_paths:
        read_json(path)
    for path in csv_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"CSV has no rows: {path}")
    raw_rows = list(csv.DictReader((root / csv_paths[0].relative_to(root)).open(encoding="utf-8")))
    if len(raw_rows) != 36:
        raise ValueError("raw33 provenance row count must be 36")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    audit = read_json(root / LOCAL_AUDIT)
    if audit.get("status") != "COMPLETE_WITH_EXPLICIT_WARNINGS":
        raise ValueError("R0 local verification result is not complete")
    raw33_dataset = next(
        row
        for row in audit["datasets"]
        if row["id"] == "waymo_dynamic_interaction_full51_semantic_strict_v1"
    )
    if sha256_file(root / RAW33_MANIFEST) != raw33_dataset["manifest_sha256"]:
        raise ValueError("raw33 current manifest SHA changed since local audit")
    make_raw33_provenance(root, audit)
    make_target_definition(root)
    make_parameterization(root)
    make_holdout_inventory(root)
    make_future_pool(root)
    make_equivalence(root)
    make_sap(root)
    make_readiness(root)
    validate_outputs(root)
    print("R0 pre-freeze manifests generated and parsed successfully")


if __name__ == "__main__":
    main()
