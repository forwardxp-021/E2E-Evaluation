#!/usr/bin/env python3
"""Build the frozen post-training BDD Style Report Card without recomputation.

This tool only copies and relabels already locked CSV/JSON summaries.  In
particular, it never opens embeddings, runs simulations, trains a model, or
computes BDD/MMD statistics.  It provides the first concrete report using the
``unified_bdd_reporting_schema_v1`` contract for the completed A/B/C study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROFILE_FIELDS = [
    "schema_version", "report_series_id", "dimension_id", "dimension_name_zh",
    "reference", "target", "task_slice", "evaluation_mode", "n_scenarios",
    "n_logs", "representation", "raw_mmd2", "null_q95", "z_bdd", "raw_p",
    "corrected_p", "null_or_calibration", "semantic_delta_target_minus_reference",
    "semantic_ci95", "semantic_direction", "interpretation", "mapping_strength",
    "evidence_status", "reason_code", "parent_bdd_result_id", "source_stage",
    "source_file", "source_sha256", "raw_mmd2_cross_rep_comparison_prohibited",
]

SCORECARD_FIELDS = [
    "schema_version", "representation", "primary_seed", "longitudinal_paired",
    "following_paired", "lane_change", "interaction_confirmation",
    "unpaired_release_n400_detection", "unpaired_release_n400_fpr",
    "unpaired_release_detection_minus_fpr", "unpaired_direction_min",
    "seed_stability", "stage6w_signal_driver", "frozen_gate_result",
    "capability_conclusion", "source_file", "source_sha256",
    "raw_mmd2_cross_rep_comparison_prohibited",
]

GAP_FIELDS = [
    "schema_version", "dimension_id", "dimension_name_zh", "behavior_profile_status",
    "bdd_status", "semantic_status", "representation_status", "reason_code",
    "available_frozen_source", "fixed_boundary",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen input is missing: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    with require_file(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = set(required).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV {path} is missing required columns: {sorted(missing)}")
        return list(reader)


def find_one(rows: list[dict[str, str]], label: str, **expected: str) -> dict[str, str]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in expected.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one row for {label}, found {len(selected)}: {expected}")
    return selected[0]


def as_float(value: str | float | int | None, field: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Frozen value is empty: {field}")
    return float(value)


def fmt(value: float | str | None, digits: int = 6) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}g}"


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def ci(low: float, high: float, unit: str) -> str:
    return f"[{low:.3f}, {high:.3f}] {unit}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def semantic_row(
    rows: list[dict[str, str]], scope: str, metric: str
) -> dict[str, str]:
    return find_one(rows, f"kinematic {scope}/{metric}", dose_label="dose100", scope=scope, metric=metric)


def bdd_row(rows: list[dict[str, str]], representation: str, scope: str) -> dict[str, str]:
    return find_one(rows, f"Stage6J/K {representation}/{scope}", representation=representation,
                    dose_label="dose100", scope=scope)


def profile_base(
    schema: dict[str, Any], dimension_id: str, **values: str
) -> dict[str, str]:
    taxonomy = {item["dimension_id"]: item["name_zh"] for item in schema["behavior_taxonomy"]}
    if dimension_id not in taxonomy:
        raise ValueError(f"Unknown frozen taxonomy dimension: {dimension_id}")
    row = {field: "" for field in PROFILE_FIELDS}
    row.update({
        "schema_version": schema["schema_version"],
        "report_series_id": "posttraining_frozen_behavior_profile_old64_primary",
        "dimension_id": dimension_id,
        "dimension_name_zh": taxonomy[dimension_id],
        "raw_mmd2_cross_rep_comparison_prohibited": "True",
    })
    row.update(values)
    return row


def profile_gap(schema: dict[str, Any], dimension_id: str, reason: str, detail: str) -> dict[str, str]:
    return profile_base(
        schema, dimension_id,
        reference="N/A", target="N/A", task_slice="fixed taxonomy row", evaluation_mode="N/A",
        n_scenarios="N/A", n_logs="N/A", representation="N/A", raw_mmd2="N/A", null_q95="N/A",
        z_bdd="N/A", raw_p="N/A", corrected_p="N/A", null_or_calibration="N/A",
        semantic_delta_target_minus_reference="N/A", semantic_ci95="N/A", semantic_direction="N/A",
        interpretation=detail, mapping_strength="INSUFFICIENT_FOR_DIMENSION_CLAIM",
        evidence_status=reason, reason_code=reason, parent_bdd_result_id="N/A",
        source_stage="schema gap audit", source_file="docs/unified_bdd_reporting_schema_freeze_v1.json",
        source_sha256="recorded in output manifest",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    schema = read_json(args.schema)
    freeze = read_json(args.schema_freeze)
    if schema.get("status") != "UNIFIED_BDD_REPORTING_SCHEMA_FROZEN":
        raise RuntimeError("Unified BDD schema is not frozen")
    if freeze.get("status") != "UNIFIED_BDD_REPORTING_SCHEMA_FROZEN":
        raise RuntimeError("Unified BDD schema freeze is not frozen")

    paired_manifest = read_json(args.stage6jk_manifest)
    unpaired_manifest = read_json(args.stage6p_manifest)
    stage6s_mechanism = read_json(args.stage6s_mechanism)
    stage6s_increment = read_json(args.stage6s_increment)
    stage7_summary = read_json(args.stage7_summary)
    if paired_manifest.get("status") != "FROZEN_STAGE6J_K_PAIRED_BLIND_COMPLETE":
        raise RuntimeError("Stage6J/K blind results are not frozen complete")
    if unpaired_manifest.get("status") != "FROZEN_STAGE6P_UNPAIRED_BLIND_COMPLETE":
        raise RuntimeError("Stage6P blind results are not frozen complete")
    if stage6s_mechanism.get("status") != "STAGE6S_V3_MECHANISM_GATE_PASS_REPRESENTATION_EVALUATION_AUTHORIZED":
        raise RuntimeError("Stage6S-v3 mechanism confirmation did not pass")
    if stage6s_increment.get("raw_mmd2_difference_computed") is not False:
        raise RuntimeError("Forbidden cross-representation raw MMD2 difference found")
    if stage7_summary.get("status") != "PASS_WITH_QUALITY_LIMITATIONS":
        raise RuntimeError("Stage7 confirmation evidence status changed")

    paired = read_csv(args.stage6jk_results, ["representation", "dose_label", "scope", "mmd2",
                                              "paired_null_q95", "null_standardized_z_bdd", "raw_p", "holm_p"])
    paired_decisions = read_csv(args.stage6jk_decisions, ["representation", "overall_holm_pass_doses_out_of_4",
                                                           "task_dose_holm_pass_cells_out_of_12",
                                                           "minimum_detectable_nominal_dose",
                                                           "median_overall_z_bdd", "frozen_longitudinal_gate_pass"])
    kinematics = read_csv(args.stage6jk_kinematics, ["dose_label", "scope", "metric", "pair_count",
                                                      "distinct_log_count", "mean_delta_A_minus_B",
                                                      "cluster_bootstrap_ci95_low", "cluster_bootstrap_ci95_high"])
    operating = read_csv(args.stage6p_operating, ["representation", "target_scenarios_per_release", "method",
                                                  "aa_false_positive_rate", "ab_detection_rate",
                                                  "detection_minus_false_positive"])
    primary = read_csv(args.stage6p_primary, ["representation", "context_balanced_fpr",
                                              "context_balanced_detection", "context_balanced_direction_min",
                                              "frozen_n400_gate_pass"])
    seed_rows = read_csv(args.stage6p_seed_stability, ["candidate", "seed", "context_balanced_detection",
                                                        "context_balanced_fpr"])
    interaction = read_csv(args.stage6s_results, ["representation", "n_pairs", "mmd2", "paired_null_q95",
                                                   "null_standardized_z_bdd", "raw_p", "candidate_detection_gate_pass"])
    stage6w = read_csv(args.stage6w_driver, ["representation", "method", "standardized_signal_ratio",
                                              "relative_null_noise_ratio", "primary_driver"])
    stage7_tasks = read_csv(args.stage7_tasks, ["task", "n_pairs", "mmd2", "p_value",
                                                "holm_p_within_pretreatment_tasks", "reject_holm_0_05"])
    stage7_kin = read_csv(args.stage7_kinematics, ["metric", "n_finite_pairs", "mean_delta_A_minus_B",
                                                    "mean_ci95_low", "mean_ci95_high"])

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite report output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "schema": args.schema, "schema_freeze": args.schema_freeze,
        "stage6jk_manifest": args.stage6jk_manifest, "stage6jk_results": args.stage6jk_results,
        "stage6jk_decisions": args.stage6jk_decisions, "stage6jk_kinematics": args.stage6jk_kinematics,
        "stage6p_manifest": args.stage6p_manifest, "stage6p_operating": args.stage6p_operating,
        "stage6p_primary": args.stage6p_primary, "stage6p_seed_stability": args.stage6p_seed_stability,
        "stage6s_mechanism": args.stage6s_mechanism, "stage6s_results": args.stage6s_results,
        "stage6s_increment": args.stage6s_increment, "stage6w_driver": args.stage6w_driver,
        "stage7_summary": args.stage7_summary, "stage7_tasks": args.stage7_tasks,
        "stage7_kinematics": args.stage7_kinematics,
    }
    source_sha = {key: source_entry(value) for key, value in source_paths.items()}

    # All profile values use one fixed historical representation (old64).  The
    # post-training A/B/C comparison is deliberately kept in the scorecard.
    old64_overall = bdd_row(paired, "old64", "overall")
    old64_following = bdd_row(paired, "old64", "following_interaction")
    kin_overall_speed = semantic_row(kinematics, "overall", "delta_mean_speed")
    kin_overall_accel = semantic_row(kinematics, "overall", "delta_rms_accel")
    kin_overall_jerk = semantic_row(kinematics, "overall", "delta_rms_jerk")
    kin_follow_speed = semantic_row(kinematics, "following_interaction", "delta_mean_speed")
    kin_follow_accel = semantic_row(kinematics, "following_interaction", "delta_rms_accel")
    interaction_old64 = find_one(interaction, "Stage6S-v3 old64", representation="old64")
    stage7_task = {row["task"]: row for row in stage7_tasks}
    stage7_kin_by_metric = {row["metric"]: row for row in stage7_kin}
    overall_stage7 = stage7_summary["primary_endpoint_preserved_without_recomputation"]

    old_ref = "pdm_closed_longitudinal_conservative_v2"
    old_target = "pdm_closed_longitudinal_assertive_v2"
    interaction_ref = "pdm_closed_interaction_long_headway_v2"
    interaction_target = "pdm_closed_interaction_short_headway_v2"
    base_null = "paired randomization null; 100000 permutations; Holm within frozen family"
    profiles: list[dict[str, str]] = []
    profiles.append(profile_base(
        schema, "OVR.ALL", reference="pdm_closed_conservative_v1", target="pdm_closed_assertive_v1",
        task_slice="Stage7 locked 5-task confirmation", evaluation_mode="paired", n_scenarios="310",
        n_logs="257", representation="old64", raw_mmd2=fmt(float(overall_stage7["original_mmd2"])),
        null_q95="N/A", z_bdd="N/A", raw_p=fmt(float(overall_stage7["original_monte_carlo_p"])),
        corrected_p="N/A", null_or_calibration="Legacy locked Monte-Carlo paired null; q95/Z not archived",
        semantic_delta_target_minus_reference=(
            f"mean speed +{float(stage7_kin_by_metric['delta_mean_speed']['mean_delta_A_minus_B']):.3f} m/s; "
            f"RMS accel +{float(stage7_kin_by_metric['delta_rms_accel']['mean_delta_A_minus_B']):.3f} m/s²"),
        semantic_ci95=(
            f"speed {ci(float(stage7_kin_by_metric['delta_mean_speed']['mean_ci95_low']), float(stage7_kin_by_metric['delta_mean_speed']['mean_ci95_high']), 'm/s')}; "
            f"RMS accel {ci(float(stage7_kin_by_metric['delta_rms_accel']['mean_ci95_low']), float(stage7_kin_by_metric['delta_rms_accel']['mean_ci95_high']), 'm/s²')}"),
        semantic_direction="TARGET_HIGHER_SPEED_AND_LONGITUDINAL_EXCITATION; overall=MIXED",
        interpretation="总体行为分布显著漂移；不表示安全性、优劣或单一风格标签。",
        mapping_strength="EXACT_DIMENSION", evidence_status="AVAILABLE", reason_code="",
        parent_bdd_result_id="stage7_overall_old64", source_stage="Stage7 locked confirmation",
        source_file=str(args.stage7_summary), source_sha256=source_sha["stage7_summary"]["sha256"],
    ))
    profiles.append(profile_gap(schema, "LON.FREE_FLOW_SPEED", "EVIDENCE_GAP_BDD_NOT_COMPUTED",
                                "没有冻结的纯自由流BDD及绑定的自由流semantic delta。"))
    profiles.append(profile_base(
        schema, "LON.ACCEL_DECEL", reference=old_ref, target=old_target,
        task_slice="Stage6J/K pure-longitudinal dose100 overall", evaluation_mode="paired",
        n_scenarios=old64_overall["n_pairs"], n_logs=kin_overall_speed["distinct_log_count"], representation="old64",
        raw_mmd2=old64_overall["mmd2"], null_q95=old64_overall["paired_null_q95"],
        z_bdd=old64_overall["null_standardized_z_bdd"], raw_p=old64_overall["raw_p"],
        corrected_p=old64_overall["holm_p"], null_or_calibration=base_null,
        semantic_delta_target_minus_reference=(
            f"mean speed +{float(kin_overall_speed['mean_delta_A_minus_B']):.3f} m/s; "
            f"RMS accel +{float(kin_overall_accel['mean_delta_A_minus_B']):.3f} m/s²"),
        semantic_ci95=(
            f"speed {ci(float(kin_overall_speed['cluster_bootstrap_ci95_low']), float(kin_overall_speed['cluster_bootstrap_ci95_high']), 'm/s')}; "
            f"RMS accel {ci(float(kin_overall_accel['cluster_bootstrap_ci95_low']), float(kin_overall_accel['cluster_bootstrap_ci95_high']), 'm/s²')}"),
        semantic_direction="TARGET_HIGHER_LONGITUDINAL_EXCITATION",
        interpretation="纯纵向处置产生显著总体纵向行为漂移。", mapping_strength="TREATMENT_ALIGNED_PROXY",
        evidence_status="AVAILABLE", reason_code="", parent_bdd_result_id="stage6jk_old64_dose100_overall",
        source_stage="Stage6J/K blind", source_file=str(args.stage6jk_results),
        source_sha256=source_sha["stage6jk_results"]["sha256"],
    ))
    profiles.append(profile_base(
        schema, "LON.CAR_FOLLOWING", reference=old_ref, target=old_target,
        task_slice="Stage6J/K following_interaction dose100", evaluation_mode="paired",
        n_scenarios=old64_following["n_pairs"], n_logs=kin_follow_speed["distinct_log_count"], representation="old64",
        raw_mmd2=old64_following["mmd2"], null_q95=old64_following["paired_null_q95"],
        z_bdd=old64_following["null_standardized_z_bdd"], raw_p=old64_following["raw_p"],
        corrected_p=old64_following["holm_p"], null_or_calibration=base_null,
        semantic_delta_target_minus_reference=(
            f"following mean speed +{float(kin_follow_speed['mean_delta_A_minus_B']):.3f} m/s; "
            f"RMS accel +{float(kin_follow_accel['mean_delta_A_minus_B']):.3f} m/s²"),
        semantic_ci95=(
            f"speed {ci(float(kin_follow_speed['cluster_bootstrap_ci95_low']), float(kin_follow_speed['cluster_bootstrap_ci95_high']), 'm/s')}; "
            f"RMS accel {ci(float(kin_follow_accel['cluster_bootstrap_ci95_low']), float(kin_follow_accel['cluster_bootstrap_ci95_high']), 'm/s²')}"),
        semantic_direction="TARGET_CLOSER_OR_MORE_ACTIVE_FOLLOWING (speed/accel only)",
        interpretation="跟车slice BDD显著；前车间距/THW方向不稳定，不能把本行解释为稳定更近。",
        mapping_strength="TASK_SLICE_PROXY", evidence_status="AVAILABLE", reason_code="",
        parent_bdd_result_id="stage6jk_old64_dose100_following", source_stage="Stage6J/K blind",
        source_file=str(args.stage6jk_results), source_sha256=source_sha["stage6jk_results"]["sha256"],
    ))
    for dimension_id, metric_key, direction, conclusion in [
        ("LON.CLOSING_RESPONSE", "delta_mean_accel_during_closing_mps2", "TARGET_HIGHER_CLOSING_ACCELERATION", "逼近阶段维持更多加速度。"),
        ("INT.LONG_FOLLOWING", "delta_mean_accel_during_following_pressure_mps2", "TARGET_HIGHER_FOLLOWING_PRESSURE_ACCELERATION", "跟车压力阶段维持更多加速度。"),
    ]:
        bootstrap = stage6s_mechanism["log_cluster_bootstrap"][metric_key.replace("_mps2", "").replace("delta_", "delta_")]
        # JSON bootstrap keys omit the unit suffix, while aggregate keeps it.
        value = float(stage6s_mechanism["aggregate"][metric_key])
        profiles.append(profile_base(
            schema, dimension_id, reference=interaction_ref, target=interaction_target,
            task_slice="Stage6S-v3 following interaction confirmation", evaluation_mode="paired",
            n_scenarios=interaction_old64["n_pairs"], n_logs=str(stage6s_mechanism["distinct_logs"]),
            representation="old64", raw_mmd2=interaction_old64["mmd2"], null_q95=interaction_old64["paired_null_q95"],
            z_bdd=interaction_old64["null_standardized_z_bdd"], raw_p=interaction_old64["raw_p"], corrected_p="N/A",
            null_or_calibration="paired randomization null; 100000 permutations; frozen Stage6S-v3 confirmation",
            semantic_delta_target_minus_reference=f"{metric_key} +{value:.3f} m/s²",
            semantic_ci95=ci(float(bootstrap["bootstrap95_low"]), float(bootstrap["bootstrap95_high"]), "m/s²"),
            semantic_direction=direction, interpretation=conclusion, mapping_strength="EXACT_DIMENSION",
            evidence_status="AVAILABLE", reason_code="", parent_bdd_result_id="stage6s_v3_old64_following",
            source_stage="Stage6S-v3 confirmation", source_file=str(args.stage6s_results),
            source_sha256=source_sha["stage6s_results"]["sha256"],
        ))
    profiles.append(profile_base(
        schema, "LON.COMFORT", reference=old_ref, target=old_target,
        task_slice="Stage6J/K pure-longitudinal dose100 overall (shared parent)", evaluation_mode="paired",
        n_scenarios=old64_overall["n_pairs"], n_logs=kin_overall_jerk["distinct_log_count"], representation="old64",
        raw_mmd2=old64_overall["mmd2"], null_q95=old64_overall["paired_null_q95"],
        z_bdd=old64_overall["null_standardized_z_bdd"], raw_p=old64_overall["raw_p"],
        corrected_p=old64_overall["holm_p"], null_or_calibration=base_null,
        semantic_delta_target_minus_reference=f"RMS jerk +{float(kin_overall_jerk['mean_delta_A_minus_B']):.3f} m/s³",
        semantic_ci95=ci(float(kin_overall_jerk["cluster_bootstrap_ci95_low"]), float(kin_overall_jerk["cluster_bootstrap_ci95_high"]), "m/s³"),
        semantic_direction="TARGET_HIGHER_LONGITUDINAL_JERK", interpretation="与LON.ACCEL_DECEL共享parent BDD；不是独立平顺性BDD检验。",
        mapping_strength="TREATMENT_ALIGNED_PROXY", evidence_status="PROXY_ONLY_NOT_CONFIRMATORY",
        reason_code="PROXY_ONLY_NOT_CONFIRMATORY", parent_bdd_result_id="stage6jk_old64_dose100_overall",
        source_stage="Stage6J/K blind", source_file=str(args.stage6jk_kinematics),
        source_sha256=source_sha["stage6jk_kinematics"]["sha256"],
    ))
    profiles.append(profile_gap(schema, "LAT.LANE_KEEPING", "EVIDENCE_GAP_BDD_NOT_COMPUTED",
                                "现有冻结库存没有精确lane-keeping BDD和语义增量。"))
    for dimension_id, task, strength, status, reason, conclusion in [
        ("LAT.LANE_CHANGE", "lane_change", "TASK_SLICE_PROXY", "AVAILABLE", "EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED", "变道场景slice BDD显著，但无冻结task级方向。"),
        ("LAT.DYNAMICS", "high_motion_dynamics", "MIXED_PROXY", "PROXY_ONLY_NOT_CONFIRMATORY", "EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED", "混合高运动slice显著，不能称纯横向BDD。"),
        ("INT.MERGE_YIELD_CUTIN", "dense_or_vulnerable_interaction", "INSUFFICIENT_FOR_DIMENSION_CLAIM", "PROXY_ONLY_NOT_CONFIRMATORY", "PROXY_ONLY_NOT_CONFIRMATORY", "仅broad dense/vulnerable interaction proxy，不能确认具体汇入/让行/切入。"),
    ]:
        task_row = stage7_task[task]
        profiles.append(profile_base(
            schema, dimension_id, reference="pdm_closed_conservative_v1", target="pdm_closed_assertive_v1",
            task_slice=f"Stage7 {task}", evaluation_mode="paired", n_scenarios=task_row["n_pairs"], n_logs="N/A (legacy field not archived)",
            representation="old64", raw_mmd2=task_row["mmd2"], null_q95="N/A", z_bdd="N/A", raw_p=task_row["p_value"],
            corrected_p=task_row["holm_p_within_pretreatment_tasks"], null_or_calibration="Legacy locked Monte-Carlo paired null; q95/Z not archived",
            semantic_delta_target_minus_reference="N/A", semantic_ci95="N/A", semantic_direction="N/A", interpretation=conclusion,
            mapping_strength=strength, evidence_status=status, reason_code=reason, parent_bdd_result_id=f"stage7_old64_{task}",
            source_stage="Stage7 locked confirmation", source_file=str(args.stage7_tasks),
            source_sha256=source_sha["stage7_tasks"]["sha256"],
        ))
    gap_front = stage6s_mechanism["aggregate"]
    front_boot = stage6s_mechanism["log_cluster_bootstrap"]
    profiles.append(profile_base(
        schema, "INT.FRONT_GAP_THW", reference=interaction_ref, target=interaction_target,
        task_slice="Stage6S-v3 following interaction confirmation", evaluation_mode="paired",
        n_scenarios=interaction_old64["n_pairs"], n_logs=str(stage6s_mechanism["distinct_logs"]), representation="old64",
        raw_mmd2=interaction_old64["mmd2"], null_q95=interaction_old64["paired_null_q95"],
        z_bdd=interaction_old64["null_standardized_z_bdd"], raw_p=interaction_old64["raw_p"], corrected_p="N/A",
        null_or_calibration="paired randomization null; 100000 permutations; frozen Stage6S-v3 confirmation",
        semantic_delta_target_minus_reference=(
            f"median front gap {float(gap_front['delta_median_front_gap_m']):+.3f} m; "
            f"finite THW {float(gap_front['delta_median_finite_thw_s']):+.3f} s"),
        semantic_ci95=(
            f"gap {ci(float(front_boot['delta_median_front_gap']['bootstrap95_low']), float(front_boot['delta_median_front_gap']['bootstrap95_high']), 'm')}; "
            f"THW {ci(float(front_boot['delta_median_finite_thw']['bootstrap95_low']), float(front_boot['delta_median_finite_thw']['bootstrap95_high']), 's')}"),
        semantic_direction="TARGET_SHORTER_FRONT_GAP_AND_FINITE_THW", interpretation="interaction mechanism与BDD均有冻结证据；THW排除sentinel/cap。",
        mapping_strength="EXACT_DIMENSION", evidence_status="AVAILABLE", reason_code="", parent_bdd_result_id="stage6s_v3_old64_following",
        source_stage="Stage6S-v3 confirmation", source_file=str(args.stage6s_mechanism),
        source_sha256=source_sha["stage6s_mechanism"]["sha256"],
    ))
    profiles.append(profile_gap(schema, "INT.LATERAL_GAP", "EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED",
                                "没有冻结的横向gap acceptance BDD及同slice semantic delta。"))
    if len(profiles) != len(schema["behavior_taxonomy"]):
        raise AssertionError(f"Expected 13 profile rows, built {len(profiles)}")

    decisions_by_rep = {row["representation"]: row for row in paired_decisions}
    primary_by_rep = {row["representation"]: row for row in primary}
    interaction_by_rep = {row["representation"]: row for row in interaction}
    drivers_by_rep = {row["representation"].replace("_3407", ""): row for row in stage6w if row["method"] == "context_balanced"}
    representations = ["old64", "A", "B", "C", "ego13"]
    scorecards: list[dict[str, str]] = []
    for rep in representations:
        decision = decisions_by_rep[rep]
        primary_key = rep if rep in {"old64", "ego13"} else f"{rep}_3407"
        primary_row = primary_by_rep[primary_key]
        op_row = find_one(operating, f"Stage6P n400 {primary_key}", representation=primary_key,
                          target_scenarios_per_release="400", method="context_balanced")
        interaction_row = interaction_by_rep[rep]
        following_passes = sum(
            row["reject_holm_0_05"] == "True" for row in paired
            if row["representation"] == rep and row["scope"] == "following_interaction"
        )
        seed_details = "N/A: single frozen historical representation"
        if rep in {"A", "B", "C"}:
            relevant = [row for row in seed_rows if row["candidate"] == rep]
            if len(relevant) != 3:
                raise ValueError(f"Expected 3 frozen seed-stability rows for {rep}")
            detections = [as_float(row["context_balanced_detection"], "seed detection") for row in relevant]
            fprs = [as_float(row["context_balanced_fpr"], "seed FPR") for row in relevant]
            seed_details = (
                f"seeds 3407/3408/3409: detection {percent(min(detections))}–{percent(max(detections))}; "
                f"FPR {percent(min(fprs))}–{percent(max(fprs))}"
            )
        interaction_text = f"Z={float(interaction_row['null_standardized_z_bdd']):.3f}; detected={interaction_row['candidate_detection_gate_pass']}"
        if rep == "C":
            interaction_text += (
                f"; full−neighbor-zero ΔZ={float(stage6s_increment['delta_z_bdd']):.3f}, "
                f"CI=[{float(stage6s_increment['log_cluster_bootstrap95_lower']):.3f}, {float(stage6s_increment['log_cluster_bootstrap95_upper']):.3f}], pass=False"
            )
        if rep == "B":
            conclusion = "当前最简单、最强release-level learned工程候选；不是universal/final validated representation。"
        elif rep == "C":
            conclusion = "release-level signal强，但未证明full-context相对neighbor-zero的增量interaction信息。"
        elif rep == "ego13":
            conclusion = "controlled-longitudinal诊断参考；不是完整context style模型。"
        elif rep == "A":
            conclusion = "动态数据修复候选；release detection改善，但整体门禁不通过。"
        else:
            conclusion = "历史baseline；release-level unpaired detection不足。"
        driver_text = "baseline (Stage6W ratio relative to itself is not reported)"
        if rep in drivers_by_rep:
            driver_text = (
                f"{drivers_by_rep[rep]['primary_driver']}; "
                f"signal={float(drivers_by_rep[rep]['standardized_signal_ratio']):.3f}×, "
                f"null noise={float(drivers_by_rep[rep]['relative_null_noise_ratio']):.3f}× vs old64"
            )
        scorecards.append({
            "schema_version": schema["schema_version"], "representation": rep,
            "primary_seed": "3407" if rep in {"A", "B", "C"} else "N/A",
            "longitudinal_paired": (
                f"overall {decision['overall_holm_pass_doses_out_of_4']}/4; task×dose "
                f"{decision['task_dose_holm_pass_cells_out_of_12']}/12; MDD={decision['minimum_detectable_nominal_dose']}; "
                f"median Z={float(decision['median_overall_z_bdd']):.3f}"),
            "following_paired": f"Holm pass {following_passes}/4 dose cells",
            "lane_change": "EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED: no fair locked A/B/C/ego13 comparison",
            "interaction_confirmation": interaction_text,
            "unpaired_release_n400_detection": percent(as_float(primary_row["context_balanced_detection"], "n400 detection")),
            "unpaired_release_n400_fpr": percent(as_float(primary_row["context_balanced_fpr"], "n400 FPR")),
            "unpaired_release_detection_minus_fpr": percent(as_float(op_row["detection_minus_false_positive"], "n400 margin")),
            "unpaired_direction_min": percent(as_float(primary_row["context_balanced_direction_min"], "n400 direction")),
            "seed_stability": seed_details,
            "stage6w_signal_driver": driver_text,
            "frozen_gate_result": primary_row["frozen_n400_gate_pass"],
            "capability_conclusion": conclusion,
            "source_file": f"{args.stage6jk_decisions}; {args.stage6p_primary}; {args.stage6s_results}",
            "source_sha256": f"{source_sha['stage6jk_decisions']['sha256']};{source_sha['stage6p_primary']['sha256']};{source_sha['stage6s_results']['sha256']}",
            "raw_mmd2_cross_rep_comparison_prohibited": "True",
        })

    profile_by_id = {row["dimension_id"]: row for row in profiles}
    gaps: list[dict[str, str]] = []
    for item in schema["behavior_taxonomy"]:
        row = profile_by_id[item["dimension_id"]]
        available = row["evidence_status"] == "AVAILABLE"
        gaps.append({
            "schema_version": schema["schema_version"], "dimension_id": item["dimension_id"],
            "dimension_name_zh": item["name_zh"],
            "behavior_profile_status": "AVAILABLE" if available else row["evidence_status"],
            "bdd_status": "AVAILABLE" if row["raw_mmd2"] != "N/A" else "EVIDENCE_GAP_BDD_NOT_COMPUTED",
            "semantic_status": "AVAILABLE" if row["semantic_delta_target_minus_reference"] != "N/A" else "EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED",
            "representation_status": ("AVAILABLE_PRIMARY_ONLY" if item["dimension_id"] in {"LON.ACCEL_DECEL", "LON.CAR_FOLLOWING", "INT.LONG_FOLLOWING", "INT.FRONT_GAP_THW"}
                                      else "EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED"),
            "reason_code": row["reason_code"] or "", "available_frozen_source": row["source_stage"],
            "fixed_boundary": row["interpretation"],
        })

    profile_path = output / "behavior_drift_profile.csv"
    score_path = output / "representation_scorecard.csv"
    gap_path = output / "evidence_gap_matrix.csv"
    write_csv(profile_path, PROFILE_FIELDS, profiles)
    write_csv(score_path, SCORECARD_FIELDS, scorecards)
    write_csv(gap_path, GAP_FIELDS, gaps)

    def md_profile(row: dict[str, str]) -> str:
        return (f"| `{row['dimension_id']}` {row['dimension_name_zh']} | {row['reference']} → {row['target']} | "
                f"{row['task_slice']} / {row['evaluation_mode']} | {row['n_scenarios']}/{row['n_logs']} | "
                f"{row['representation']} | {row['z_bdd']} | {row['corrected_p'] or row['raw_p']} | "
                f"{row['semantic_delta_target_minus_reference']} | {row['semantic_direction']} | {row['interpretation']} |\n")

    def md_score(row: dict[str, str]) -> str:
        return (f"| {row['representation']} | {row['longitudinal_paired']} | {row['following_paired']} | "
                f"{row['interaction_confirmation']} | {row['unpaired_release_n400_detection']} | "
                f"{row['unpaired_release_n400_fpr']} | {row['unpaired_release_detection_minus_fpr']} | "
                f"{row['capability_conclusion']} |\n")

    report_path = output / "unified_bdd_posttraining_report_zh.md"
    report = f"""# 新训练模型后对比试验：统一 BDD Style Report Card

> Schema：`{schema['schema_version']}`
> 状态：`FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE`
> 生成方式：只读取已冻结结果；不训练、不仿真、不读取embedding、不重算BDD/MMD。
> 全部semantic delta为 **Target − Reference**。

## 1. 读表前必须区分的三件事

- **表A 行为漂移画像**回答Target相对Reference的行为变化和方向。本次采用old64作为历史主行为报告的固定representation；这不表示old64是最佳检测器。
- **BDD统计量**只在同一行的Reference、Target、task、representation、null下有效。禁止跨representation比较raw MMD²。
- **表B 表示能力评分卡**回答old64/A/B/C/ego13检测已知处置的可靠性，不能反过来当作Target的风格方向报告。

## 2. 表A：BDD Behavior Profile / Style Report Card

| Behavior dimension | Reference→Target | Task / mode | N scenario/log | Rep. | Z_BDD | corrected p / raw p | Semantic Δ (Target−Reference) | Direction | Conclusion |
|---|---|---|---:|---|---:|---|---|---|---|
{''.join(md_profile(row) for row in profiles)}

说明：`N/A`和`EVIDENCE_GAP_*`表示尚无冻结证据，**不是没有行为差异**。Stage7历史任务行未归档null q95/Z，不能事后由其他representation或其他task补填。

## 3. 表B：BDD Evaluator / Representation Scorecard

| Representation | Pure-longitudinal paired | Following paired | Interaction confirmation | n=400 detection | A/A FPR | detection−FPR | Capability conclusion |
|---|---|---|---|---:|---:|---:|---|
{''.join(md_score(row) for row in scorecards)}

固定解释：Stage6P是context-balanced、log-disjoint unpaired release监测，n=400；A/A calibration独立完成。Stage6J/K与Stage6S-v3是paired条件。B/C的release-level提升主要由标准化signal增强驱动，不能用“raw MMD²更大”解释。

## 4. 新训练模型后比较：直接结论

1. **release-level longitudinal drift**：old64的n=400 detection为66.5%，A/B/C为90.5%/100.0%/99.5%，对应FPR为3.0%/5.0%/6.5%。B是当前最简单且最强的learned release-level工程候选。
2. **controlled paired longitudinal**：ego13仍最强（4/4 overall、12/12 task×dose）；A保持old64级别，B/C为3/4 overall、2/12 task×dose。因此A/B/C不是全局、最终验证representation。
3. **interaction**：Stage6S-v3的轨迹机制门禁通过；但是C full-context相对C neighbor-zero的ΔZ为−7.852，log-cluster 95% CI为[−33.393, 29.219]，没有证明增量interaction信息。
4. **最终模型判断不变**：`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。本报告只统一表达，绝不改变冻结结论。

## 5. 固定证据缺口

| Dimension | Behavior profile | BDD | Semantic direction | Representation comparison | Reason |
|---|---|---|---|---|---|
{''.join(f"| `{row['dimension_id']}` {row['dimension_name_zh']} | {row['behavior_profile_status']} | {row['bdd_status']} | {row['semantic_status']} | {row['representation_status']} | {row['reason_code'] or '—'} |\n" for row in gaps)}

完整机器可读表见：

- `behavior_drift_profile.csv`（表A，含raw MMD²、null、p、semantic和provenance字段）
- `representation_scorecard.csv`（表B，不用raw MMD²跨表示排序）
- `evidence_gap_matrix.csv`（13维完整coverage）

`FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE`
"""
    report_path.write_text(report, encoding="utf-8")

    output_paths = [profile_path, score_path, gap_path, report_path]
    manifest = {
        "schema_version": "unified_bdd_posttraining_report_v1",
        "status": "FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": {"name": schema["schema_version"], "sha256": sha256(args.schema)},
        "frozen_schema_freeze_sha256": sha256(args.schema_freeze),
        "input_sources": source_sha,
        "output_files": {path.name: sha256(path) for path in output_paths},
        "profile_representation": "old64",
        "scorecard_representations": representations,
        "profile_rows": len(profiles),
        "scorecard_rows": len(scorecards),
        "evidence_gap_rows": len(gaps),
        "training_run": False,
        "simulation_run": False,
        "embedding_read": False,
        "bdd_or_mmd_recomputed": False,
        "cross_representation_raw_mmd2_comparison_performed": False,
        "existing_frozen_conclusions_modified": False,
        "stage6v_joint_model_decision": "NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE",
    }
    manifest_path = output / "unified_bdd_posttraining_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=ROOT / "configs/unified_bdd_reporting_schema_v1.json")
    parser.add_argument("--schema-freeze", type=Path, default=ROOT / "docs/unified_bdd_reporting_schema_freeze_v1.json")
    parser.add_argument("--stage6jk-manifest", type=Path, default=ROOT / "outputs/stage6v_stage6jk_paired_blind_v1/stage6v_stage6jk_result_manifest.json")
    parser.add_argument("--stage6jk-results", type=Path, default=ROOT / "outputs/stage6v_stage6jk_paired_blind_v1/stage6v_stage6jk_paired_results.csv")
    parser.add_argument("--stage6jk-decisions", type=Path, default=ROOT / "outputs/stage6v_stage6jk_paired_blind_v1/stage6v_stage6jk_decisions.csv")
    parser.add_argument("--stage6jk-kinematics", type=Path, default=ROOT / "outputs/stage6k_realized_longitudinal_dose_curve_v2_runtime_repaired/stage6k_kinematic_contrasts.csv")
    parser.add_argument("--stage6p-manifest", type=Path, default=ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_result_manifest.json")
    parser.add_argument("--stage6p-operating", type=Path, default=ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_operating_characteristics.csv")
    parser.add_argument("--stage6p-primary", type=Path, default=ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_primary_decisions.csv")
    parser.add_argument("--stage6p-seed-stability", type=Path, default=ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_seed_stability_n400.csv")
    parser.add_argument("--stage6s-mechanism", type=Path, default=ROOT / "outputs/stage6s_v3_confirmation_mechanism_v1/stage6s_v3_confirmation_mechanism_summary.json")
    parser.add_argument("--stage6s-results", type=Path, default=ROOT / "outputs/stage6s_v3_confirmation_representations_v1/stage6s_v3_confirmation_representation_results.csv")
    parser.add_argument("--stage6s-increment", type=Path, default=ROOT / "outputs/stage6s_v3_confirmation_representations_v1/stage6s_v3_c_context_increment.json")
    parser.add_argument("--stage6w-driver", type=Path, default=ROOT / "outputs/stage6w_a_context_balanced_driver_addendum_v2/stage6w_a_unpaired_driver_by_method.csv")
    parser.add_argument("--stage7-summary", type=Path, default=ROOT / "outputs/stage7_m6_6_confirmation_evidence_v1/m6_6_confirmation_evidence_summary.json")
    parser.add_argument("--stage7-tasks", type=Path, default=ROOT / "outputs/stage7_m6_6_confirmation_evidence_v1/table_m6_6_task_bdd.csv")
    parser.add_argument("--stage7-kinematics", type=Path, default=ROOT / "outputs/stage7_m6_6_confirmation_evidence_v1/table_m6_6_kinematic_contrasts.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/unified_bdd_posttraining_report_v1")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "profile_rows": result["profile_rows"],
                      "scorecard_rows": result["scorecard_rows"]}, ensure_ascii=False))
