#!/usr/bin/env python3
"""Integrate frozen Stage7L-E results into final reporting without recomputation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("old64", "A_seed3407", "B_seed3407", "C_seed3407", "ego13")
REP_SHORT = {
    "old64": "old64",
    "A_seed3407": "A",
    "B_seed3407": "B",
    "C_seed3407": "C",
    "ego13": "ego13",
}
DOSES = ("dose25", "dose50", "dose75", "dose100")
TASKS = ("LAT.LANE_CHANGE", "LAT.DYNAMICS")
TASK_TO_DIMENSION = {
    "LAT.LANE_CHANGE": ("LAT.LANE_CHANGE", "变道行为", "Lateral"),
    "LAT.DYNAMICS": ("LAT.DYNAMICS", "横向动态", "Lateral"),
}
TARGET_BY_DOSE = {
    "dose25": "pure_lateral_execution_dose25_58.5m",
    "dose50": "pure_lateral_execution_dose50_57.0m",
    "dose75": "pure_lateral_execution_dose75_55.5m",
    "dose100": "pure_lateral_execution_sharp_dose100_54.0m",
}
BEHAVIOR_REFERENCE = "pure_lateral_execution_gentle_dose0_60.0m"
NULL_REFERENCE = (
    "representation-specific same-scenario within-pair label-swap; "
    "100000 randomizations; plus-one p; own null q95"
)
FINAL_STATUS = "STAGE7L_E_PROSPECTIVE_REPRESENTATION_EVALUATION_COMPLETE"
INTEGRATION_STATUS = "STAGE7L_E_PROSPECTIVE_EVIDENCE_INTEGRATED_FOR_THESIS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> str:
    if not rows:
        return "N/A"
    columns = list(fields or rows[0].keys())

    def cell(value: Any) -> str:
        if value is None or value == "":
            return "N/A"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend("| " + " | ".join(cell(row.get(name)) for name in columns) + " |" for row in rows)
    return "\n".join(lines)


def verify_machine_freeze(repo: Path, result_dir: Path) -> tuple[dict[str, Any], str]:
    path = repo / "docs/stage7l_e_machine_result_freeze_manifest_v1.json"
    manifest = read_json(path)
    if manifest.get("status") != "STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING":
        raise RuntimeError("Stage7L-E machine results are not frozen for E3")
    for name, expected in manifest["result_artifact_sha256"].items():
        actual_path = result_dir / name
        if not actual_path.is_file() or sha256(actual_path) != expected:
            raise RuntimeError(f"Frozen E2 artifact mismatch: {actual_path}")
    if manifest["primary"]["final_primary_status"] != "STAGE7L_E_PRIMARY_BDD_FAILED":
        raise RuntimeError("Unexpected frozen Stage7L-E primary status")
    if manifest["scientific_boundaries"]["stage6v_qualification_changed"] is not False:
        raise RuntimeError("Stage6V qualification boundary changed")
    return manifest, sha256(path)


def validate_stage7l_mapping(repo: Path) -> tuple[Path, str]:
    path = repo / "configs/unified_bdd_stage7l_evidence_mapping_v1.csv"
    rows = read_csv_rows(path)
    if len(rows) != 2:
        raise RuntimeError("Stage7L evidence mapping must contain exactly two task rows")
    actual = {(row["source_task"], row["dimension_id"]) for row in rows}
    expected = {("LAT.LANE_CHANGE", "LAT.LANE_CHANGE"), ("LAT.DYNAMICS", "LAT.DYNAMICS")}
    if actual != expected:
        raise RuntimeError("Stage7L task-to-dimension mapping changed")
    if any(row["evaluation_mode"] != "paired" for row in rows):
        raise RuntimeError("Stage7L mapping must remain paired")
    return path, sha256(path)


def load_cells(result_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, str]]]:
    frame = pd.read_csv(result_dir / "all_bdd_cells.csv", keep_default_na=False)
    if len(frame) != 40:
        raise RuntimeError(f"Expected 40 frozen BDD cells, got {len(frame)}")
    keys = set(zip(frame["representation"], frame["dose"], frame["task"]))
    expected = {(rep, dose, task) for rep in REPRESENTATIONS for dose in DOSES for task in TASKS}
    if keys != expected:
        raise RuntimeError("Frozen Stage7L-E cell identity matrix changed")
    index = {
        (str(row["representation"]), str(row["dose"]), str(row["task"])): {
            key: str(value) for key, value in row.to_dict().items()
        }
        for _, row in frame.iterrows()
    }
    return frame, index


def mechanism_by_dose(mechanism: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {"dose100": mechanism["primary"]}
    for row in mechanism["secondary_dose_response"]:
        if row["metric"] not in {
            "lane_change_duration_s",
            "rms_lateral_accel_mps2",
            "peak_yaw_rate_radps",
        }:
            continue
        output.setdefault(row["dose"], {})[row["metric"]] = row
    if set(output) != set(DOSES):
        raise RuntimeError("Stage7L-D semantic dose curve is incomplete")
    return output


def semantic_fields(
    task: str,
    dose: str,
    mechanisms: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str, str]:
    if task == "LAT.DYNAMICS":
        return (
            "N/A_TASK_SPECIFIC_SEMANTIC_DELTA_NOT_COMPUTED",
            "N/A",
            "N/A_TASK_SPECIFIC_SEMANTIC_DELTA_NOT_COMPUTED",
            "MIXED_PROXY",
        )
    metrics = mechanisms[dose]
    duration = metrics["lane_change_duration_s"]
    accel = metrics["rms_lateral_accel_mps2"]
    yaw = metrics["peak_yaw_rate_radps"]
    semantic_metric = "lane_change_duration_s + rms_lateral_accel_mps2 + peak_yaw_rate_radps"
    delta = (
        f"duration {float(duration['paired_median_delta']):+.6f} s; "
        f"RMS lateral accel {float(accel['paired_median_delta']):+.6f} m/s²; "
        f"peak yaw rate {float(yaw['paired_median_delta']):+.6f} rad/s"
    )
    ci = (
        f"duration {duration['log_cluster_bootstrap_median_ci_95']}; "
        f"RMS lateral accel {accel['log_cluster_bootstrap_median_ci_95']}; "
        f"peak yaw rate {yaw['log_cluster_bootstrap_median_ci_95']}"
    )
    return semantic_metric, delta, ci, "TARGET_SHORTER_DURATION_HIGHER_LATERAL_EXCITATION"


def evidence_status(row: Mapping[str, str]) -> str:
    key = (row["representation"], row["dose"], row["task"])
    if key == ("B_seed3407", "dose100", "LAT.LANE_CHANGE"):
        return "PROSPECTIVE_PRE_REGISTERED_PRIMARY_FAILED"
    significant = str(row.get("holm_significant_0_05", "")).lower() == "true"
    suffix = "HOLM_SIGNIFICANT" if significant else "HOLM_NOT_SIGNIFICANT"
    if row["task"] == "LAT.DYNAMICS":
        return f"PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_{suffix}"
    return f"PROSPECTIVE_SECONDARY_{suffix}"


def build_stage7l_long_rows(
    cells: pd.DataFrame,
    mechanisms: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in cells.iterrows():
        raw = {key: str(value) for key, value in source.to_dict().items()}
        task = raw["task"]
        dose = raw["dose"]
        representation = raw["representation"]
        dimension_id, behavior_dimension, level = TASK_TO_DIMENSION[task]
        semantic_metric, semantic_delta, semantic_ci, semantic_direction = semantic_fields(
            task, dose, mechanisms
        )
        is_primary = (
            representation == "B_seed3407" and dose == "dose100" and task == "LAT.LANE_CHANGE"
        )
        corrected = "N/A_PRIMARY_EXCLUDED_FROM_HOLM" if is_primary else raw["holm_p"]
        detected = float(raw["raw_p"]) < 0.05 if is_primary else raw["holm_significant_0_05"].lower() == "true"
        rows.append(
            {
                "schema_version": "standardized_fixed_dimension_bdd_matrix_v2_stage7l",
                "report_id": "stage7l_e_prospective_evidence_addendum_v1",
                "result_id": f"stage7l_e:{REP_SHORT[representation]}:{dose}:{task}",
                "parent_bdd_result_id": f"stage7l_e:{REP_SHORT[representation]}:{dose}:{task}",
                "dimension_id": dimension_id,
                "behavior_dimension": behavior_dimension,
                "behavior_level": level,
                "behavior_reference": BEHAVIOR_REFERENCE,
                "target": TARGET_BY_DOSE[dose],
                "contrast_label": f"{TARGET_BY_DOSE[dose]} | {BEHAVIOR_REFERENCE}",
                "task_id": task,
                "evaluation_mode": "paired",
                "n_pairs": int(raw["N_pair"]),
                "n_scenarios": int(raw["N_pair"]),
                "n_logs": 79 if task == "LAT.LANE_CHANGE" else 38,
                "representation_id": REP_SHORT[representation],
                "representation_baseline": (
                    "old64" if representation == "old64" else "old64_capability_baseline_not_raw_mmd2_reference"
                ),
                "statistic_name": "biased_single_rbf_mmd2",
                "null_reference": NULL_REFERENCE,
                "raw_mmd2": float(raw["raw_mmd2"]),
                "null_q95": float(raw["null_q95"]),
                "bdd_to_null_q95_ratio": float(raw["bdd_over_null_q95"]),
                "z_bdd": float(raw["z_bdd"]),
                "raw_p_value": float(raw["raw_p"]),
                "corrected_p_value": corrected,
                "detection_or_pass": detected,
                "semantic_metric": semantic_metric,
                "semantic_delta_target_minus_reference": semantic_delta,
                "semantic_95ci": semantic_ci,
                "semantic_direction": semantic_direction,
                "mapping_strength": "DIRECT_PROSPECTIVE_TREATMENT" if task == "LAT.LANE_CHANGE" else "MIXED_PROXY",
                "evidence_status": evidence_status(raw),
                "provenance_path": "outputs/stage7l_e_prospective_bdd_v1/all_bdd_cells.csv",
                "shared_parent_bdd": False,
                "interpretation": (
                    "Prospective frozen Stage7L pure-lateral BDD. Physical direction comes from Stage7L-D "
                    "semantic mechanism, never from the MMD sign or magnitude."
                ),
                "bandwidth": float(raw["bandwidth"]),
                "null_repetitions": int(raw["null_reps"]),
            }
        )
    return pd.DataFrame(rows)


def cell_text(row: Mapping[str, str]) -> str:
    return f"{float(row['bdd_over_null_q95']):.2f}× / Z={float(row['z_bdd']):.2f}"


def update_primary_matrix(
    base: pd.DataFrame,
    index: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> pd.DataFrame:
    matrix = base.copy()
    for task, dimension in (("LAT.LANE_CHANGE", "LAT.LANE_CHANGE"), ("LAT.DYNAMICS", "LAT.DYNAMICS")):
        row_index = matrix.index[matrix["dimension_id"] == dimension]
        if len(row_index) != 1:
            raise RuntimeError(f"Base matrix lacks unique {dimension}")
        position = row_index[0]
        for representation in REPRESENTATIONS:
            matrix.at[position, REP_SHORT[representation]] = cell_text(
                index[(representation, "dose100", task)]
            )
        matrix.at[position, "source"] = "stage7l_e"
        matrix.at[position, "evidence_status"] = (
            "PROSPECTIVE_PRIMARY_B_FAILED_SECONDARY_EGO13_HOLM_SIGNIFICANT"
            if task == "LAT.LANE_CHANGE"
            else "PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_ONLY_EGO13_HOLM_SIGNIFICANT"
        )
        ego_z = float(index[("ego13", "dose100", task)]["z_bdd"])
        matrix.at[position, "highest_standardized_sensitivity_on_this_treatment"] = (
            f"ego13 (within-null Z={ego_z:.2f})"
        )
    matrix["prospective_evidence_precedence"] = matrix["dimension_id"].apply(
        lambda value: "STAGE7L_E_PRIMARY_DISPLAY" if value in {"LAT.LANE_CHANGE", "LAT.DYNAMICS"} else "UNCHANGED_BASE_EVIDENCE"
    )
    matrix["historical_posthoc_preserved"] = matrix["dimension_id"].apply(
        lambda value: bool(value in {"LAT.LANE_CHANGE", "LAT.DYNAMICS"})
    )
    return matrix


def update_style_card(
    base: pd.DataFrame,
    index: Mapping[tuple[str, str, str], Mapping[str, str]],
    mechanisms: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    style = base.copy()
    for task, dimension in (("LAT.LANE_CHANGE", "LAT.LANE_CHANGE"), ("LAT.DYNAMICS", "LAT.DYNAMICS")):
        row_index = style.index[style["dimension_id"] == dimension]
        if len(row_index) != 1:
            raise RuntimeError(f"Base style card lacks unique {dimension}")
        position = row_index[0]
        result = index[("B_seed3407", "dose100", task)]
        semantic_metric, semantic_delta, semantic_ci, semantic_direction = semantic_fields(
            task, "dose100", mechanisms
        )
        values = {
            "behavior_reference": BEHAVIOR_REFERENCE,
            "target": TARGET_BY_DOSE["dose100"],
            "evaluation_mode": "paired",
            "primary_representation": "B",
            "null_reference": NULL_REFERENCE,
            "n_scenarios": int(result["N_pair"]),
            "n_logs": 79 if task == "LAT.LANE_CHANGE" else 38,
            "raw_mmd2": float(result["raw_mmd2"]),
            "null_q95": float(result["null_q95"]),
            "bdd_to_null_q95_ratio": float(result["bdd_over_null_q95"]),
            "z_bdd": float(result["z_bdd"]),
            "significance": (
                "不显著（预注册Primary FAIL；raw p=0.411906）"
                if task == "LAT.LANE_CHANGE"
                else "不显著（secondary Holm p=1.0；LOW_N mixed proxy）"
            ),
            "semantic_delta_target_minus_reference": semantic_delta,
            "semantic_95ci": semantic_ci,
            "semantic_direction": semantic_direction,
            "evidence_status": (
                "PROSPECTIVE_PLANNER_MECHANISM_POSITIVE_B_PRIMARY_BDD_FAILED"
                if task == "LAT.LANE_CHANGE"
                else "PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_BDD_NOT_SIGNIFICANT"
            ),
            "parent_bdd_result_id": f"stage7l_e:B:dose100:{task}",
            "shared_parent_bdd": False,
            "evidence_gap": (
                "N/A"
                if task == "LAT.LANE_CHANGE"
                else "38场景slice没有单独冻结semantic delta；不得借用all-80语义指标冒充task-specific结果。"
            ),
        }
        for column, value in values.items():
            style.at[position, column] = value
        if "semantic_metric" in style.columns:
            style.at[position, "semantic_metric"] = semantic_metric
    return style


def update_qualification(
    base: pd.DataFrame,
    index: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> pd.DataFrame:
    output = base.copy()
    reverse = {value: key for key, value in REP_SHORT.items()}
    statuses = {
        "old64": "SECONDARY_HOLM_NOT_SIGNIFICANT",
        "A": "SECONDARY_HOLM_NOT_SIGNIFICANT",
        "B": "PRE_REGISTERED_PRIMARY_FAILED",
        "C": "SECONDARY_HOLM_NOT_SIGNIFICANT",
        "ego13": "SECONDARY_HOLM_SIGNIFICANT",
    }
    output["stage7l_e_dose100_lane_change_status"] = output["representation_id"].map(statuses)
    output["stage7l_e_dose100_lane_change_bdd_over_null_q95"] = output["representation_id"].map(
        lambda rep: float(index[(reverse[rep], "dose100", "LAT.LANE_CHANGE")]["bdd_over_null_q95"])
    )
    output["stage7l_e_dose100_lane_change_z_bdd"] = output["representation_id"].map(
        lambda rep: float(index[(reverse[rep], "dose100", "LAT.LANE_CHANGE")]["z_bdd"])
    )
    output["stage7l_e_dose100_lane_change_raw_p"] = output["representation_id"].map(
        lambda rep: float(index[(reverse[rep], "dose100", "LAT.LANE_CHANGE")]["raw_p"])
    )
    output["stage7l_e_changes_stage6v_joint_gate"] = False
    return output


def dose_curve_rows(index: Mapping[tuple[str, str, str], Mapping[str, str]], task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for dose in DOSES:
            raw = index[(representation, dose, task)]
            primary = representation == "B_seed3407" and dose == "dose100" and task == "LAT.LANE_CHANGE"
            rows.append(
                {
                    "Representation": REP_SHORT[representation],
                    "Dose": dose,
                    "N": raw["N_pair"],
                    "BDD/q95": f"{float(raw['bdd_over_null_q95']):.3f}×",
                    "Z_BDD": f"{float(raw['z_bdd']):.3f}",
                    "raw p": f"{float(raw['raw_p']):.6g}",
                    "Holm p": "Primary—excluded" if primary else f"{float(raw['holm_p']):.6g}",
                    "Status": evidence_status(raw),
                }
            )
    return rows


def build_stage7l_report(
    machine: Mapping[str, Any],
    index: Mapping[tuple[str, str, str], Mapping[str, str]],
    mechanism: Mapping[str, Any],
    nuisance: Mapping[str, Any],
) -> str:
    primary = machine["primary"]
    comparison = []
    for rep in REPRESENTATIONS:
        raw = index[(rep, "dose100", "LAT.LANE_CHANGE")]
        comparison.append(
            {
                "Representation": REP_SHORT[rep],
                "raw MMD²（仅本rep内解释）": f"{float(raw['raw_mmd2']):.9f}",
                "null q95": f"{float(raw['null_q95']):.9f}",
                "BDD/q95": f"{float(raw['bdd_over_null_q95']):.3f}×",
                "Z_BDD": f"{float(raw['z_bdd']):.3f}",
                "p": f"{float(raw['raw_p']):.6g}",
                "身份": "Primary" if rep == "B_seed3407" else "Secondary",
            }
        )
    primary_mechanism = mechanism["primary"]
    lines = [
        "# Stage7L-E Prospective Representation / BDD 最终中文报告",
        "",
        f"> 最终状态：`{FINAL_STATUS}`",
        "> Primary状态：`STAGE7L_E_PRIMARY_BDD_FAILED`",
        "> 本报告接受预注册失败结果；没有换checkpoint、换task、调kernel、重训或rescue experiment。",
        "",
        "## 1. Frozen provenance",
        "",
        f"- Stage7L-D冻结commit：`{machine['frozen_provenance']['stage7l_d_commit']}`。",
        f"- Stage7L protocol SHA：`{machine['frozen_provenance']['protocol_sha256']}`。",
        f"- roster SHA：`{machine['frozen_provenance']['roster_sha256']}`。",
        f"- E2 preformal commit：`{machine['frozen_provenance']['e2_preformal_resume_commit']}`。",
        "- 400条Stage7L-D official rollout原样复用；planner rerun=No，replacement=0，outcome filtering=No。",
        "",
        "## 2. Stage7L-D unlock evidence",
        "",
        "execution、canonical identity、mechanism、longitudinal nuisance、safety/validity和representation unlock六项均为PASS。80/80场景五档完整，400/400 official rollout成功。",
        "",
        "## 3. Representation contracts",
        "",
        "仅使用old64、A3407、B3407、C3407、ego13。四个learned representation复用Stage6V/W冻结83D→64D inference；ego13复用冻结13D scaler。Primary固定B3407，没有按结果换seed。",
        "",
        "## 4. Pair populations与输入合同",
        "",
        "五档context均为`[80,150,83]`、float32、finite、scenario order一致。`LAT.LANE_CHANGE`四档均80 pair/79 log；`LAT.DYNAMICS`四档均38 pair/38 log，后者继续是pre-treatment high-motion `MIXED_PROXY`。",
        "",
        "## 5. 预注册Primary B结果",
        "",
        markdown_table(
            [
                {
                    "Rep": "B3407",
                    "Contrast": "dose100 − dose0",
                    "Task": "LAT.LANE_CHANGE",
                    "N": primary["N_pair"],
                    "raw MMD²": f"{primary['raw_mmd2']:.9f}",
                    "null mean": f"{primary['null_mean']:.9f}",
                    "null SD": f"{primary['null_sd']:.9f}",
                    "null q95": f"{primary['null_q95']:.9f}",
                    "BDD/q95": f"{primary['bdd_over_null_q95']:.3f}×",
                    "Z_BDD": f"{primary['z_bdd']:.3f}",
                    "plus-one p": f"{primary['raw_p']:.6f}",
                    "结论": "FAIL",
                }
            ]
        ),
        "",
        "planner-level pure-lateral treatment confirmation成功，但Candidate B未通过预先指定的prospective paired BDD endpoint。BDD failure不否定物理mechanism；它否定的是B在该冻结任务上的预注册检测主张。",
        "",
        "## 6. dose100五representation对照",
        "",
        markdown_table(comparison),
        "",
        "ego13具有该Treatment下最高within-null标准化敏感度。该treatment直接改变ego横向运动学，因此不能写成ego13全局最佳，也不能据此否定neighbor/context。raw MMD²没有跨representation排序。",
        "",
        "## 7. LAT.LANE_CHANGE四档dose curve",
        "",
        markdown_table(dose_curve_rows(index, "LAT.LANE_CHANGE")),
        "",
        "中间dose不要求严格单调。learned64四种表示均未在该任务获得显著secondary或Primary结果；ego13四档均通过Holm。",
        "",
        "## 8. LAT.DYNAMICS secondary",
        "",
        markdown_table(dose_curve_rows(index, "LAT.DYNAMICS")),
        "",
        "该38场景slice全部标记`LOW_N_SECONDARY_DIAGNOSTIC`。A/C的部分raw p低于0.05，但39-test Holm后均不显著；只有ego13四档通过。不得把该slice称为pure lateral dynamics ground truth。",
        "",
        "## 9. 固定39-test Holm family",
        "",
        "理论矩阵为5 representations×4 doses×2 tasks=40格；唯一B×dose100×LAT.LANE_CHANGE Primary只排除一次，secondary family固定39格。Holm通过8格，全部来自ego13；NOT_COMPUTABLE=0，LOW_N=20。",
        "",
        "## 10. Semantic mechanism与BDD方向分离",
        "",
        "所有semantic delta为Sharp dose100−Gentle dose0，方向来自Stage7L-D trajectory mechanism，而不是MMD正值：",
        "",
        markdown_table(
            [
                {
                    "Metric": name,
                    "paired median Δ": f"{values['paired_median_delta']:+.6f}",
                    "direction consistency": f"{100*values['directional_consistency']:.2f}%",
                    "95% log-cluster CI": values["log_cluster_bootstrap_median_ci_95"],
                    "Gate": "PASS",
                }
                for name, values in primary_mechanism.items()
            ]
        ),
        "",
        f"纵向nuisance gate仍为{nuisance['nuisance_gate_pass']}。因此正确组合结论是：planner-level横向机制positive，但B representation Primary detection negative。",
        "",
        "## 11. Positive / negative findings",
        "",
        "- Positive：prospective pure-lateral planner treatment在80个冻结场景上产生方向正确、纵向副作用极小的横向机制差异。",
        "- Positive：ego13在两个task的四档secondary均显著，证明冻结统计管线能够检出运动学处置。",
        "- Negative：B的唯一Primary未通过；old64/A/C在dose100 lane-change secondary也未通过。",
        "- Negative：learned64没有在本前瞻pure-lateral任务中证明稳定检测能力。",
        "",
        "## 12. Stage6V compatibility与claim boundary",
        "",
        "Stage6V联合结论继续是`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。B仍是最简单、最强的release-level learned工程候选，但Stage7L没有为其增加pure-lateral paired成功证据。A仍是dynamic-data repair ablation；C仍是dual-branch ablation，Stage7L不是interaction-specific benchmark，不能改写Stage6S-v3的context增量负结果。",
        "",
        "允许写：经前瞻冻结的pure-lateral planner treatment得到物理确认，但B未通过预注册BDD主端点；ego13在该运动学处置中最敏感。禁止写：B完成横向验证、ego13全局最佳、context无价值、新64D全面优于old64。",
        "",
        "## 13. Thesis implication",
        "",
        "Stage7L补齐的是一个可信的确认性负结果：框架成功区分了‘行为确实变化’与‘指定representation能否检出’。这直接支撑task-conditioned representation qualification，而不是削弱论文主线。Stage7L实验链到此关闭，后续只允许论文写作、图表和claim cleanup。",
        "",
        "## 14. 任务要求29项核对",
        "",
        "1. Stage7L-D unlock：已验证，六项required gate均PASS。",
        "2. 原400条rollout：是；未重新仿真。",
        "3. input contract：通过，五档均`[80,150,83]`且finite。",
        "4. old64/A/B/C/ego13：全部推理成功。",
        "5. LAT.LANE_CHANGE N_pair：25/50/75/100均80。",
        "6. LAT.DYNAMICS N_pair：25/50/75/100均38。",
        f"7. Primary raw MMD²：`{primary['raw_mmd2']}`。",
        f"8. Primary null q95：`{primary['null_q95']}`。",
        f"9. Primary BDD/q95：`{primary['bdd_over_null_q95']}`。",
        f"10. Primary Z_BDD：`{primary['z_bdd']}`。",
        f"11. Primary plus-one p：`{primary['raw_p']}`。",
        "12. Primary：FAIL。",
        f"13. old64 dose100：ratio/Z/p=`{float(index[('old64','dose100','LAT.LANE_CHANGE')]['bdd_over_null_q95']):.6f}/{float(index[('old64','dose100','LAT.LANE_CHANGE')]['z_bdd']):.6f}/{float(index[('old64','dose100','LAT.LANE_CHANGE')]['raw_p']):.6f}`。",
        f"14. A dose100：ratio/Z/p=`{float(index[('A_seed3407','dose100','LAT.LANE_CHANGE')]['bdd_over_null_q95']):.6f}/{float(index[('A_seed3407','dose100','LAT.LANE_CHANGE')]['z_bdd']):.6f}/{float(index[('A_seed3407','dose100','LAT.LANE_CHANGE')]['raw_p']):.6f}`。",
        f"15. B dose100：ratio/Z/p=`{float(index[('B_seed3407','dose100','LAT.LANE_CHANGE')]['bdd_over_null_q95']):.6f}/{float(index[('B_seed3407','dose100','LAT.LANE_CHANGE')]['z_bdd']):.6f}/{float(index[('B_seed3407','dose100','LAT.LANE_CHANGE')]['raw_p']):.6f}`。",
        f"16. C dose100：ratio/Z/p=`{float(index[('C_seed3407','dose100','LAT.LANE_CHANGE')]['bdd_over_null_q95']):.6f}/{float(index[('C_seed3407','dose100','LAT.LANE_CHANGE')]['z_bdd']):.6f}/{float(index[('C_seed3407','dose100','LAT.LANE_CHANGE')]['raw_p']):.6f}`。",
        f"17. ego13 dose100：ratio/Z/p=`{float(index[('ego13','dose100','LAT.LANE_CHANGE')]['bdd_over_null_q95']):.6f}/{float(index[('ego13','dose100','LAT.LANE_CHANGE')]['z_bdd']):.6f}/{float(index[('ego13','dose100','LAT.LANE_CHANGE')]['raw_p']):.6g}`。",
        "18. 最高标准化敏感度：ego13；仅限该kinematic-heavy treatment。",
        "19. B Z曲线：`-0.914/-0.818/0.433/-0.065`。",
        "20. 五representation曲线：完整列于第7节；learned64不显著，ego13四档显著。",
        "21. LAT.DYNAMICS：38场景/38 log，完整列于第8节。",
        "22. Holm通过：8/39。",
        "23. NOT_COMPUTABLE：0。",
        "24. LOW_N：20。",
        "25. 跨representation比较raw MMD²：No。",
        "26. Stage6V qualification改变：No。",
        f"27. Stage7L-E最终状态：`{FINAL_STATUS}`。",
        "28. 最终commit SHA：由E3提交后写入Git历史；manifest绑定提交前全部证据SHA。",
        "29. 远端同步：由E3提交后完成并在最终回复确认。",
        "",
        f"`{FINAL_STATUS}`",
    ]
    return "\n".join(lines) + "\n"


def build_integrated_style_report(
    matrix: pd.DataFrame,
    style: pd.DataFrame,
    qualification: pd.DataFrame,
) -> str:
    matrix_fields = [
        "behavior_dimension", "old64", "A", "B", "C", "ego13",
        "highest_standardized_sensitivity_on_this_treatment", "evidence_status",
    ]
    style_fields = [
        "behavior_dimension", "behavior_reference", "target", "evaluation_mode",
        "primary_representation", "null_reference", "n_scenarios", "n_logs",
        "bdd_to_null_q95_ratio", "z_bdd", "significance",
        "semantic_delta_target_minus_reference", "semantic_direction", "evidence_status",
    ]
    qualification_fields = [
        "representation_id", "stage6p_n400_detection", "stage6p_n400_aa_fpr",
        "stage6jk_paired_gate_pass", "stage6p_unpaired_gate_pass", "waymo_gate_pass",
        "interaction_increment_gate_pass", "stage7l_e_dose100_lane_change_status",
        "stage7l_e_dose100_lane_change_bdd_over_null_q95",
        "stage7l_e_dose100_lane_change_z_bdd", "stage6v_joint_candidate_gate_pass",
        "applicability_boundary",
    ]
    return "\n".join(
        [
            "# Final Standardized BDD Style Report Card — Stage7L Prospective Evidence Addendum",
            "",
            "> 基础schema状态：`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`",
            f"> Stage7L证据整合状态：`{INTEGRATION_STATUS}`",
            "> 没有重算既有统计；Stage7L-E数值逐值继承E2冻结结果。",
            "",
            "## 第一页：Behavior Drift / Style Report Card",
            "",
            "Primary Representation仍为B；B只是测量representation，不是被评价的planner。每行独立声明Behavior Reference、Target和Null Reference。semantic方向只来自Target−Reference物理指标。",
            "",
            markdown_table(style.to_dict("records"), style_fields),
            "",
            "Stage7L prospective更新要点：Sharp相对Gentle的换道时长缩短、RMS横向加速度和峰值yaw-rate升高，但B的BDD Primary不显著。这不是矛盾：前者回答行为是否物理变化，后者回答B是否能检出该分布变化。",
            "",
            "† Stage6S-v3的closing/front-gap/following三行共享同一parent task-level BDD，不是独立检验。",
            "",
            "## 第二页：Representation Qualification Matrix",
            "",
            "单元格只比较各representation自身null下的BDD/q95和Z_BDD；禁止跨representation比较raw MMD²。`该Treatment下最高标准化检测敏感度`不表示全局最佳。",
            "",
            markdown_table(matrix.to_dict("records"), matrix_fields),
            "",
            "### Release、联合门禁与Stage7L资格",
            "",
            markdown_table(qualification.to_dict("records"), qualification_fields),
            "",
            "## 证据优先级与历史保留",
            "",
            "- `LAT.LANE_CHANGE`和`LAT.DYNAMICS`主显示使用Stage7L prospective dose100 evidence。",
            "- 原Stage7 60场景post-hoc lane-change/lateral结果完整保留在`historical_stage7_posthoc_lateral_evidence.csv`和combined long table中，但不再作为横向主显示。",
            "- Stage7L B Primary失败不会被ego13 secondary成功替代；Primary身份保持B。",
            "- Free-flow speed、lane keeping、lateral gap interaction继续N/A，不补实验。",
            "",
            "## 一眼可答的最终结论",
            "",
            "1. 跟车BDD（B）：`1.72× / Z=5.25`，Stage6J/K confirmatory。",
            "2. 变道BDD（B）：Stage7L prospective `0.436× / Z=-0.065 / p=0.411906`，Primary FAIL；物理mechanism同时PASS。",
            "3. 纵向BDD（B）：`2.74× / Z=10.33`，Stage6J/K confirmatory。",
            "4. interaction BDD（B）：`7.39× / Z=30.60 †`，Stage6S-v3 confirmatory。",
            "5. 横向最高within-null标准化敏感度为ego13（Stage7L dose100 Z=40.201），不等于全局最佳。",
            "6. Stage7历史post-hoc横向结果保留，但证据等级低于Stage7L prospective。",
            "7. Stage6V联合决策仍为`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。",
            "",
            f"`{INTEGRATION_STATUS}`",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--base-matrix-dir", type=Path)
    parser.add_argument("--base-final-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    result_dir = (args.result_dir or repo / "outputs/stage7l_e_prospective_bdd_v1").resolve()
    base_matrix_dir = (
        args.base_matrix_dir or repo / "outputs/standardized_fixed_dimension_bdd_matrix_v1"
    ).resolve()
    base_final_dir = (
        args.base_final_dir or repo / "outputs/final_standardized_bdd_style_report_card_v1"
    ).resolve()
    output_dir = (
        args.output_dir or repo / "outputs/final_standardized_bdd_style_report_card_v2_stage7l"
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage7L integrated output: {output_dir}")

    machine, machine_sha = verify_machine_freeze(repo, result_dir)
    mapping_path, mapping_sha = validate_stage7l_mapping(repo)
    cells, index = load_cells(result_dir)
    mechanism_path = repo / "outputs/stage7l_d_one_time_confirmation_v1/mechanism_summary.json"
    nuisance_path = repo / "outputs/stage7l_d_one_time_confirmation_v1/longitudinal_nuisance_summary.json"
    mechanism = read_json(mechanism_path)
    nuisance = read_json(nuisance_path)
    mechanisms = mechanism_by_dose(mechanism)

    base_manifest_path = base_final_dir / "final_standardized_bdd_reporting_manifest.json"
    base_manifest = read_json(base_manifest_path)
    if base_manifest["status"] != "FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN":
        raise RuntimeError("Base standardized BDD reporting system is not frozen")

    output_dir.mkdir(parents=True)
    stage7l_long = build_stage7l_long_rows(cells, mechanisms)
    base_long = pd.read_csv(base_matrix_dir / "standardized_bdd_long.csv", keep_default_na=False)
    if list(stage7l_long.columns) != list(base_long.columns):
        raise RuntimeError("Stage7L long-table schema does not match frozen standardized schema")
    combined_long = pd.concat([base_long, stage7l_long], ignore_index=True)
    historical = base_long[base_long["dimension_id"].isin(["LAT.LANE_CHANGE", "LAT.DYNAMICS"])]

    base_matrix = pd.read_csv(base_final_dir / "final_fixed_dimension_primary_matrix.csv", keep_default_na=False)
    base_style = pd.read_csv(base_final_dir / "final_behavior_style_report_card.csv", keep_default_na=False)
    base_qualification = pd.read_csv(
        base_final_dir / "final_representation_qualification_matrix.csv", keep_default_na=False
    )
    matrix = update_primary_matrix(base_matrix, index)
    style = update_style_card(base_style, index, mechanisms)
    qualification = update_qualification(base_qualification, index)
    shared_audit = pd.read_csv(
        base_final_dir / "final_shared_parent_bdd_audit.csv", keep_default_na=False
    )

    stage7l_long.to_csv(output_dir / "stage7l_e_prospective_bdd_long.csv", index=False)
    combined_long.to_csv(output_dir / "standardized_bdd_long_with_stage7l.csv", index=False)
    historical.to_csv(output_dir / "historical_stage7_posthoc_lateral_evidence.csv", index=False)
    matrix.to_csv(output_dir / "final_fixed_dimension_primary_matrix.csv", index=False)
    style.to_csv(output_dir / "final_behavior_style_report_card.csv", index=False)
    qualification.to_csv(output_dir / "final_representation_qualification_matrix.csv", index=False)
    shared_audit.to_csv(output_dir / "final_shared_parent_bdd_audit.csv", index=False)

    integrated_report = build_integrated_style_report(matrix, style, qualification)
    integrated_report_path = output_dir / "final_standardized_bdd_style_report_card_zh.md"
    integrated_report_path.write_text(integrated_report, encoding="utf-8")

    stage7l_report = build_stage7l_report(machine, index, mechanism, nuisance)
    stage7l_report_path = repo / "docs/stage7l_e_prospective_representation_bdd_report_zh.md"
    stage7l_report_path.write_text(stage7l_report, encoding="utf-8")

    output_names = (
        "stage7l_e_prospective_bdd_long.csv",
        "standardized_bdd_long_with_stage7l.csv",
        "historical_stage7_posthoc_lateral_evidence.csv",
        "final_fixed_dimension_primary_matrix.csv",
        "final_behavior_style_report_card.csv",
        "final_representation_qualification_matrix.csv",
        "final_shared_parent_bdd_audit.csv",
        "final_standardized_bdd_style_report_card_zh.md",
    )
    integration_manifest = {
        "schema_version": "final_standardized_bdd_reporting_stage7l_addendum_v1",
        "status": INTEGRATION_STATUS,
        "base_schema_status": "FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN",
        "stage7l_e_final_status": FINAL_STATUS,
        "base_final_manifest_sha256": sha256(base_manifest_path),
        "stage7l_e_machine_manifest_sha256": machine_sha,
        "stage7l_evidence_mapping_path": str(mapping_path.relative_to(repo)),
        "stage7l_evidence_mapping_sha256": mapping_sha,
        "finalizer_tool_sha256": sha256(repo / "tools/stage7l_e_finalize_reporting.py"),
        "statistics_recomputed": False,
        "training_run": False,
        "simulation_run": False,
        "checkpoint_created_or_modified": False,
        "scenario_selection_modified": False,
        "raw_mmd2_cross_representation_ranking_performed": False,
        "stage6v_joint_conclusion_modified": False,
        "stage7_historical_posthoc_preserved": True,
        "dimension_count": int(len(matrix)),
        "stage7l_prospective_row_count": int(len(stage7l_long)),
        "output_files": {name: sha256(output_dir / name) for name in output_names},
    }
    integration_manifest_path = output_dir / "final_standardized_bdd_reporting_manifest.json"
    write_json(integration_manifest_path, integration_manifest)

    stage7l_manifest = {
        "schema_version": "stage7l_e_prospective_bdd_manifest_v1",
        "status": FINAL_STATUS,
        "primary_status": "STAGE7L_E_PRIMARY_BDD_FAILED",
        "branch": "20260611_stage7_conclusion",
        "frozen_provenance": machine["frozen_provenance"],
        "machine_result_freeze_manifest_sha256": machine_sha,
        "stage7l_evidence_mapping_sha256": mapping_sha,
        "finalizer_tool_sha256": sha256(repo / "tools/stage7l_e_finalize_reporting.py"),
        "stage7l_d_mechanism_summary_sha256": sha256(mechanism_path),
        "stage7l_d_longitudinal_nuisance_summary_sha256": sha256(nuisance_path),
        "checkpoint_or_scaler_sha256": machine["checkpoint_or_scaler_sha256"],
        "null_settings": machine["execution_contract"],
        "task_populations": machine["task_populations"],
        "primary": machine["primary"],
        "secondary_family": machine["secondary_family"],
        "dose100_lane_change_standardized_comparison": machine[
            "dose100_lane_change_standardized_comparison"
        ],
        "claim_boundary": {
            "planner_level_pure_lateral_mechanism_confirmed": True,
            "candidate_b_prospective_primary_detected": False,
            "ego13_highest_standardized_sensitivity_on_treatment": True,
            "ego13_global_best_claim_allowed": False,
            "stage6v_qualification_changed": False,
            "new_stage7l_model_development_allowed": False,
        },
        "report_sha256": sha256(stage7l_report_path),
        "integrated_reporting_manifest_sha256": sha256(integration_manifest_path),
        "integrated_reporting_output_sha256": integration_manifest["output_files"],
        "next_authorized_step": "THESIS_WRITING_FIGURES_TABLES_AND_CLAIM_CLEANUP_ONLY",
    }
    stage7l_manifest_path = repo / "docs/stage7l_e_prospective_bdd_manifest_v1.json"
    write_json(stage7l_manifest_path, stage7l_manifest)

    print(
        json.dumps(
            {
                "status": FINAL_STATUS,
                "integration_status": INTEGRATION_STATUS,
                "primary_status": "STAGE7L_E_PRIMARY_BDD_FAILED",
                "output_dir": str(output_dir),
                "report": str(stage7l_report_path),
                "manifest": str(stage7l_manifest_path),
                "statistics_recomputed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
