#!/usr/bin/env python3
"""Freeze the Stage6W-A and Stage6S-v3 combined interpretation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def by_rep(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["representation"]: row for row in rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    w_manifest = read_json(args.stage6w_manifest)
    w_addendum = read_json(args.stage6w_addendum_manifest)
    v3_freeze = read_json(args.stage6s_v3_freeze)
    v3_state = read_json(args.stage6s_v3_batch_state)
    mechanism = read_json(args.stage6s_v3_mechanism)
    representation = read_json(args.stage6s_v3_representation_manifest)
    increment = read_json(args.stage6s_v3_increment)
    v2_failure = read_json(args.stage6s_v2_failure)
    if w_manifest.get("status") != "FROZEN_STAGE6W_A_PAIRED_UNPAIRED_MECHANISM_COMPLETE":
        raise RuntimeError("Stage6W-A base analysis is not frozen complete")
    if w_addendum.get("status") != "FROZEN_STAGE6W_A_CONTEXT_BALANCED_DRIVER_ADDENDUM_V2_COMPLETE":
        raise RuntimeError("Stage6W-A context-balanced addendum is not frozen complete")
    if v3_freeze.get("status") != "STAGE6S_V3_CONFIRMATION_ROSTER_FROZEN_NOT_RUN":
        raise RuntimeError("Stage6S-v3 roster freeze changed")
    if v3_state.get("counts") != {"SUCCEEDED": 80, "FAILED_REVIEW_REQUIRED": 0, "PENDING": 0}:
        raise RuntimeError("Stage6S-v3 official rollout is incomplete")
    if mechanism.get("status") != "STAGE6S_V3_MECHANISM_GATE_PASS_REPRESENTATION_EVALUATION_AUTHORIZED":
        raise RuntimeError("Stage6S-v3 mechanism did not pass")
    if representation.get("status") != "FROZEN_STAGE6S_V3_REPRESENTATION_EVALUATION_COMPLETE":
        raise RuntimeError("Stage6S-v3 representation result is not frozen complete")
    if v2_failure.get("status") != "CONFIRMATION_EXECUTION_INCOMPLETE_STOP_NO_MECHANISM_OR_EMBEDDING":
        raise RuntimeError("Stage6S-v2 permanent failure record changed")
    if increment.get("raw_mmd2_difference_computed") is not False:
        raise RuntimeError("forbidden cross-representation raw MMD2 difference was computed")

    w_summary = by_rep(read_csv(args.stage6w_summary))
    w_decomp = by_rep(read_csv(args.stage6w_decomposition))
    driver_rows = read_csv(args.stage6w_addendum_driver)
    drivers = {(row["representation"], row["method"]): row for row in driver_rows}
    rep_rows = by_rep(read_csv(args.stage6s_v3_representation_results))
    b_signal = float(drivers[("B_3407", "context_balanced")]["standardized_signal_ratio"])
    c_signal = float(drivers[("C_3407", "context_balanced")]["standardized_signal_ratio"])
    b_noise = float(drivers[("B_3407", "context_balanced")]["relative_null_noise_ratio"])
    c_noise = float(drivers[("C_3407", "context_balanced")]["relative_null_noise_ratio"])
    b_signal_share = float(drivers[("B_3407", "context_balanced")]["log_gain_from_signal"]) / (
        float(drivers[("B_3407", "context_balanced")]["log_gain_from_signal"])
        + float(drivers[("B_3407", "context_balanced")]["log_gain_from_noise_reduction"])
    )
    c_signal_share = float(drivers[("C_3407", "context_balanced")]["log_gain_from_signal"]) / (
        float(drivers[("C_3407", "context_balanced")]["log_gain_from_signal"])
        + float(drivers[("C_3407", "context_balanced")]["log_gain_from_noise_reduction"])
    )
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "stage6w_stage6s_v3_final_decision_v1",
        "status": "FROZEN_STAGE6W_STAGE6S_V3_COMPLETE_NO_NEW_CHECKPOINT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paired_unpaired_mechanism": {
            "same_pool_same_n_result": "B/C paired sensitivity is strong on the Stage6P pool; historical separation is benchmark/treatment/task-dose specific, not an intrinsic paired-statistic weakness.",
            "same_pool_n": 400,
            "old64_paired_z_median": float(w_summary["old64"]["median_paired_z_same_support"]),
            "B_paired_z_median": float(w_summary["B_3407"]["median_paired_z_same_support"]),
            "C_paired_z_median": float(w_summary["C_3407"]["median_paired_z_same_support"]),
            "old64_release_direction_coherence": float(w_summary["old64"]["release_direction_resultant_length"]),
            "B_release_direction_coherence": float(w_summary["B_3407"]["release_direction_resultant_length"]),
            "C_release_direction_coherence": float(w_summary["C_3407"]["release_direction_resultant_length"]),
            "old64_planner_energy_fraction": float(w_decomp["old64"]["planner_signal_energy_fraction"]),
            "B_planner_energy_fraction": float(w_decomp["B_3407"]["planner_signal_energy_fraction"]),
            "C_planner_energy_fraction": float(w_decomp["C_3407"]["planner_signal_energy_fraction"]),
        },
        "unpaired_driver": {
            "primary_driver": "signal_enhancement",
            "context_balanced_B_signal_ratio_vs_old64": b_signal,
            "context_balanced_C_signal_ratio_vs_old64": c_signal,
            "context_balanced_B_null_noise_ratio_vs_old64": b_noise,
            "context_balanced_C_null_noise_ratio_vs_old64": c_noise,
            "context_balanced_B_log_gain_share_from_signal": b_signal_share,
            "context_balanced_C_log_gain_share_from_signal": c_signal_share,
        },
        "stage6s_v3": {
            "official_rollout_succeeded": 80, "official_rollout_failed": 0,
            "mechanism_gate_passed": True,
            "delta_mean_speed_mps": mechanism["aggregate"]["delta_mean_speed_mps"],
            "delta_rms_accel_mps2": mechanism["aggregate"]["delta_rms_accel_mps2"],
            "delta_median_front_gap_m": mechanism["aggregate"]["delta_median_front_gap_m"],
            "delta_median_finite_thw_s": mechanism["aggregate"]["delta_median_finite_thw_s"],
            "interaction_checks": mechanism["interaction_checks"],
            "C_z_bdd": float(rep_rows["C"]["null_standardized_z_bdd"]),
            "C_neighbor_zero_z_bdd": float(rep_rows["C_neighbor_zero"]["null_standardized_z_bdd"]),
            "C_full_minus_neighbor_zero_delta_z": increment["delta_z_bdd"],
            "C_increment_cluster_bootstrap95": [increment["log_cluster_bootstrap95_lower"], increment["log_cluster_bootstrap95_upper"]],
            "C_incremental_interaction_information_pass": increment["incremental_interaction_information_pass"],
        },
        "paper_decision": {
            "can_close_as_mixed_result": True,
            "cannot_claim_interaction_aware_C_validated": True,
            "cannot_override_stage6v_joint_model_decision": True,
            "stage6v_joint_model_decision_remains": "NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE",
            "best_engineering_release_detector_under_current_evidence": "B, as the simpler model with 100% Stage6P n=400 detection; this is not a final-model qualification.",
        },
        "future_v3_training_decision": {
            "automatic_training_authorized": False,
            "scientific_reason_exists_if_interaction_aware_main_claim_is_required": True,
            "recommendation": "Do not train automatically. Close the current paper as a mixed/negative interaction result unless an interaction-aware final-model claim is essential; only then preregister a genuinely new runnable confirmation inventory before any v3 training.",
            "unused_officially_runnable_candidates_in_current_frozen_inventory": int(v3_freeze["official_runnable_candidate_count"] - v3_freeze["scenario_count"]),
            "current_inventory_can_supply_new_60_pair_confirmation": False,
        },
        "training_or_checkpoint_write": False,
        "protocol_or_existing_benchmark_modified": False,
        "cross_representation_raw_mmd2_comparison_performed": False,
        "source_sha256": {
            "stage6w_manifest": sha256(args.stage6w_manifest),
            "stage6w_addendum_manifest": sha256(args.stage6w_addendum_manifest),
            "stage6s_v3_freeze": sha256(args.stage6s_v3_freeze),
            "stage6s_v3_batch_state": sha256(args.stage6s_v3_batch_state),
            "stage6s_v3_mechanism": sha256(args.stage6s_v3_mechanism),
            "stage6s_v3_representation_manifest": sha256(args.stage6s_v3_representation_manifest),
            "stage6s_v3_increment": sha256(args.stage6s_v3_increment),
            "stage6s_v2_failure": sha256(args.stage6s_v2_failure),
        },
    }
    manifest_path = output / "stage6w_stage6s_v3_final_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f"""# Stage6W-A 与 Stage6S-v3 中文总报告

## 1. paired / unpaired 分离机制

在同一Stage6P 800-pair pool、相同n=400下，B/C的paired median Z分别为
{result['paired_unpaired_mechanism']['B_paired_z_median']:.3f}/{result['paired_unpaired_mechanism']['C_paired_z_median']:.3f}，
高于old64的{result['paired_unpaired_mechanism']['old64_paired_z_median']:.3f}。因此历史Stage6J/K较弱并非
paired statistic天然压低B/C，也不是183 vs 400样本量本身造成；主要是Stage6J/K窄纵向dose/task与
Stage6P广义assertive/conservative treatment、场景池和estimand不同。B/C在Stage6P的release shift方向
一致性为{result['paired_unpaired_mechanism']['B_release_direction_coherence']:.3f}/
{result['paired_unpaired_mechanism']['C_release_direction_coherence']:.3f}，高于old64的
{result['paired_unpaired_mechanism']['old64_release_direction_coherence']:.3f}。日志异质性仍占主导，但400场景
聚合能平均局部异质性并保留一致planner shift。

## 2. B/C unpaired提升的驱动

context-balanced口径下，B/C标准化signal相对old64为{b_signal:.3f}×/{c_signal:.3f}×，null noise为
{b_noise:.3f}×/{c_noise:.3f}×。按log-Z增益分解，signal贡献约{100*b_signal_share:.1f}%/
{100*c_signal_share:.1f}%，所以主要驱动是signal增强；null方差下降只提供次要贡献。raw-marginal口径下
null noise甚至没有下降，结论相同。

## 3. Stage6S-v3机制确认

80/80 official pairs成功。短headway减长headway的median Δmean speed=
{mechanism['aggregate']['delta_mean_speed_mps']:+.3f} m/s、ΔRMS accel=
{mechanism['aggregate']['delta_rms_accel_mps2']:+.3f} m/s²；Δfront gap=
{mechanism['aggregate']['delta_median_front_gap_m']:+.3f} m、Δfinite THW=
{mechanism['aggregate']['delta_median_finite_thw_s']:+.3f} s。front-gap、finite-THW、closing accel和
following accel四项均通过，机制确认成功。

## 4. C是否含增量interaction信息

未证明。C full-context Z={float(rep_rows['C']['null_standardized_z_bdd']):.3f}，C neighbor-zero Z=
{float(rep_rows['C_neighbor_zero']['null_standardized_z_bdd']):.3f}；预冻结ΔZ=
{increment['delta_z_bdd']:+.3f}，log-cluster bootstrap 95% CI=[
{increment['log_cluster_bootstrap95_lower']:.3f}, {increment['log_cluster_bootstrap95_upper']:.3f}]，下界不大于0。
两者各自均能检出treatment，不等于full-context提供了额外interaction信息。

## 5. 论文是否可以收口

可以按“强unpaired release正结果 + paired/Waymo/interaction增量负结果”的诚实口径收口。不能写成
“interaction-aware C已验证”或“新64D全面恢复纵向敏感性”。Stage6V联合门禁结论仍为
`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`，本阶段不能事后改写。B可作为当前最简单、最强的
release-level工程候选讨论，但不等于通过论文最终主模型资格。

## 6. 是否值得训练v3

不自动启动。若论文必须坚持“interaction-conditioned主模型”这一更强主张，C相对neighbor-zero无增量是
足够明确的全新模型研究理由；但必须在训练前先扩展并冻结一套真正未使用、100% runnable的新confirmation。
当前库存仅余{result['future_v3_training_decision']['unused_officially_runnable_candidates_in_current_frozen_inventory']}个
未使用runnable候选，不足60-pair最低规模，因此不能复用本次80对或只在原库存补抽。若接受混合/负结果，
当前论文无需再训练v3即可收口。
"""
    report_path = output / "stage6w_stage6s_v3_report_zh.md"
    report_path.write_text(report, encoding="utf-8")
    result["final_manifest_sha256"] = sha256(manifest_path)
    result["report_sha256"] = sha256(report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6w_manifest", type=Path, required=True)
    parser.add_argument("--stage6w_summary", type=Path, required=True)
    parser.add_argument("--stage6w_decomposition", type=Path, required=True)
    parser.add_argument("--stage6w_addendum_manifest", type=Path, required=True)
    parser.add_argument("--stage6w_addendum_driver", type=Path, required=True)
    parser.add_argument("--stage6s_v3_freeze", type=Path, required=True)
    parser.add_argument("--stage6s_v3_batch_state", type=Path, required=True)
    parser.add_argument("--stage6s_v3_mechanism", type=Path, required=True)
    parser.add_argument("--stage6s_v3_representation_manifest", type=Path, required=True)
    parser.add_argument("--stage6s_v3_representation_results", type=Path, required=True)
    parser.add_argument("--stage6s_v3_increment", type=Path, required=True)
    parser.add_argument("--stage6s_v2_failure", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
