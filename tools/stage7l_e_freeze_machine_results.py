#!/usr/bin/env python3
"""Audit and freeze Stage7L-E machine results without recomputing statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPRESENTATIONS = ("old64", "A_seed3407", "B_seed3407", "C_seed3407", "ego13")
DOSES = ("dose25", "dose50", "dose75", "dose100")
TASKS = ("LAT.LANE_CHANGE", "LAT.DYNAMICS")
PRIMARY_KEY = ("B_seed3407", "dose100", "LAT.LANE_CHANGE")
EXPECTED_D_COMMIT = "6279bc742ad527246a945a4b6d5d7090fab591ea"
EXPECTED_PREFORMAL_COMMIT = "a85314a34518aaec627dca7baf5b73b15483553c"
EXPECTED_PROTOCOL_SHA = "f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a"
EXPECTED_ROSTER_SHA = "90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9"
EXPECTED_TASK_MASK_SHA = "74206af6b0d7bc5be4b16e8ef8343feebad6ced722d1d3232e4e72c6396a3ec3"
NULL_REPETITIONS = 100_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty Stage7L-E provenance ledger")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_stage7l_d(repo: Path) -> tuple[dict[str, Any], str]:
    path = repo / "docs/stage7l_d_confirmation_manifest_v1.json"
    manifest = read_json(path)
    if manifest["status"] != "STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED":
        raise RuntimeError("Stage7L-D planner confirmation is not passed")
    if manifest["representation_status"] != "STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED":
        raise RuntimeError("Stage7L-E representation evaluation is not unlocked")
    required_gates = (
        "execution",
        "canonical_identity",
        "mechanism",
        "longitudinal_nuisance",
        "safety_validity",
        "representation_unlock",
    )
    if not all(manifest["gates"].get(name) is True for name in required_gates):
        raise RuntimeError("One or more Stage7L-D unlock gates are not true")
    provenance = manifest["frozen_provenance"]
    if provenance["protocol_sha256"] != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("Stage7L protocol SHA mismatch")
    if provenance["roster_sha256"] != EXPECTED_ROSTER_SHA:
        raise RuntimeError("Stage7L roster SHA mismatch")
    return manifest, sha256_file(path)


def validate_results(result_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(result_dir / "all_bdd_cells.csv")
    decision = read_json(result_dir / "stage7l_e_final_decision.json")
    expected_keys = {(rep, dose, task) for rep in REPRESENTATIONS for dose in DOSES for task in TASKS}
    actual_keys = {(row["representation"], row["dose"], row["task"]) for row in rows}
    if len(rows) != 40 or actual_keys != expected_keys:
        raise RuntimeError("Stage7L-E fixed 40-cell matrix mismatch")
    primary_rows = [row for row in rows if row["multiplicity_role"] == "PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY"]
    if len(primary_rows) != 1 or tuple(primary_rows[0][key] for key in ("representation", "dose", "task")) != PRIMARY_KEY:
        raise RuntimeError("Stage7L-E primary exclusion is not unique or has changed")
    secondary = [row for row in rows if row["multiplicity_role"] == "SECONDARY_HOLM_39"]
    if len(secondary) != 39:
        raise RuntimeError("Stage7L-E secondary Holm family is not 39")
    expected_n = {"LAT.LANE_CHANGE": "80", "LAT.DYNAMICS": "38"}
    if any(row["N_pair"] != expected_n[row["task"]] for row in rows):
        raise RuntimeError("Frozen Stage7L-E task population changed")
    if sum(row["status"] == "LOW_N_SECONDARY_DIAGNOSTIC" for row in rows) != 20:
        raise RuntimeError("Unexpected Stage7L-E low-N count")
    if sum(row["status"] == "NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION" for row in rows) != 0:
        raise RuntimeError("Unexpected Stage7L-E not-computable cell")
    if decision["primary_status"] != "STAGE7L_E_PRIMARY_BDD_FAILED":
        raise RuntimeError("Unexpected Stage7L-E primary status")
    if decision["secondary_holm_pass_count"] != 8:
        raise RuntimeError("Unexpected Stage7L-E Holm pass count")
    if decision["cross_representation_raw_mmd2_comparison_performed"] is not False:
        raise RuntimeError("Forbidden cross-representation raw MMD comparison was recorded")
    if decision["stage6v_qualification_changed"] is not False:
        raise RuntimeError("Stage6V qualification was unexpectedly changed")
    if decision["planner_rerun"] is not False or decision["checkpoint_or_training_modified"] is not False:
        raise RuntimeError("Planner/checkpoint/training immutability was violated")
    return rows, decision


def embedding_contracts(result_dir: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "old64": result_dir / "embedding_manifest_old64.json",
        "A_seed3407": result_dir / "embedding_manifest_A.json",
        "B_seed3407": result_dir / "embedding_manifest_B.json",
        "C_seed3407": result_dir / "embedding_manifest_C.json",
        "ego13": result_dir / "embedding_manifest_ego13.json",
    }
    manifests: dict[str, dict[str, Any]] = {}
    for representation, path in paths.items():
        manifest = read_json(path)
        if manifest["representation"] != representation or manifest["finite"] is not True:
            raise RuntimeError(f"Invalid embedding manifest: {path}")
        if manifest["scenario_order_sha256"] != EXPECTED_TASK_MASK_SHA:
            raise RuntimeError(f"Scenario order mismatch: {path}")
        manifests[representation] = {
            "path": str(path),
            "manifest_sha256": sha256_file(path),
            "checkpoint_or_scaler_sha256": manifest["checkpoint_or_scaler"]["sha256"],
            "doses": {item["dose"]: item for item in manifest["doses"]},
        }
    return manifests


def build_provenance_ledger(
    result_dir: Path,
    rows: Sequence[Mapping[str, str]],
    embeddings: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for row in rows:
        representation = row["representation"]
        dose = row["dose"]
        task_slug = row["task"].replace(".", "_")
        cell_path = result_dir / "cell_ledger" / f"{representation}__{dose}__{task_slug}.json"
        null_path = result_dir / "cell_ledger" / f"{representation}__{dose}__{task_slug}_null.npy"
        raw_cell = read_json(cell_path)
        for field in ("representation", "dose", "task", "N_pair", "null_seed", "null_reps"):
            expected: Any = row[field]
            actual: Any = raw_cell[field]
            if field in ("N_pair", "null_seed", "null_reps"):
                expected = int(expected)
            if actual != expected:
                raise RuntimeError(f"Cell aggregate mismatch for {cell_path}: {field}")
        samples = np.load(null_path, mmap_mode="r")
        if samples.shape != (NULL_REPETITIONS,) or not np.isfinite(samples).all():
            raise RuntimeError(f"Invalid paired-null sample artifact: {null_path}")
        rep_manifest = embeddings[representation]
        dose0 = rep_manifest["doses"]["dose0"]
        target = rep_manifest["doses"][dose]
        enriched = dict(row)
        enriched.update(
            {
                "checkpoint_or_scaler_sha256": rep_manifest["checkpoint_or_scaler_sha256"],
                "reference_embedding_sha256": dose0["embedding_sha256"],
                "target_embedding_sha256": target["embedding_sha256"],
                "reference_input_feature_sha256": dose0["input_feature_sha256"],
                "target_input_feature_sha256": target["input_feature_sha256"],
                "scenario_order_and_task_mask_sha256": EXPECTED_TASK_MASK_SHA,
                "cell_json_sha256": sha256_file(cell_path),
                "paired_null_samples_sha256": sha256_file(null_path),
            }
        )
        ledger.append(enriched)
    return ledger


def number(value: str) -> float:
    return float(value)


def format_report(rows: Sequence[Mapping[str, str]], decision: Mapping[str, Any]) -> str:
    index = {(row["representation"], row["dose"], row["task"]): row for row in rows}
    primary = decision["primary"]
    lines = [
        "# Stage7L-E E2机器结果冻结报告",
        "",
        "> 状态：`STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING`",
        "",
        "本报告只冻结E2机器结果；没有修改模型、checkpoint、planner、roster、task、kernel、阈值或Stage6V结论。完整论文叙事、统一BDD矩阵与Style Report Card更新留给E3。",
        "",
        "## 1. 冻结范围与输入",
        "",
        "- Stage7L-D六项解锁条件全部通过。",
        "- 仅复用400条冻结official rollout；nuPlan planner未重新运行。",
        "- 五档输入均为`[80,150,83]`、float32、finite，scenario order一致。",
        "- old64、A3407、B3407、C3407均成功生成`[80,64]`；ego13成功生成`[80,13]`。",
        "- `LAT.LANE_CHANGE`四档均80对；`LAT.DYNAMICS`四档均38对。",
        "",
        "## 2. 预注册Primary",
        "",
        "| Rep | Contrast | Task | N | raw MMD² | null q95 | BDD/q95 | Z_BDD | plus-one p | 结论 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        f"| B3407 | dose100−dose0 | LAT.LANE_CHANGE | {primary['N_pair']} | {primary['raw_mmd2']:.9f} | {primary['null_q95']:.9f} | {primary['bdd_over_null_q95']:.3f}× | {primary['z_bdd']:.3f} | {primary['raw_p']:.6f} | FAIL |",
        "",
        "冻结解释：planner-level pure-lateral treatment confirmation成功，但Candidate B未通过预先指定的prospective paired BDD Primary endpoint。不得据此更换checkpoint、task、kernel或重新训练。",
        "",
        "## 3. dose100同一Treatment五representation对照",
        "",
        "raw MMD²只在各representation内部解释，禁止跨representation排序。",
        "",
        "| Representation | BDD/q95 | Z_BDD | raw p | Multiplicity |",
        "|---|---:|---:|---:|---|",
    ]
    for rep in REPRESENTATIONS:
        row = index[(rep, "dose100", "LAT.LANE_CHANGE")]
        role = "Primary（不进入Holm）" if rep == "B_seed3407" else f"Secondary（Holm p={number(row['holm_p']):.6f}）"
        lines.append(f"| {rep} | {number(row['bdd_over_null_q95']):.3f}× | {number(row['z_bdd']):.3f} | {number(row['raw_p']):.6g} | {role} |")
    lines.extend(
        [
            "",
            "ego13在该kinematic-heavy treatment下具有最高within-null标准化敏感度；这不表示ego13是全局最佳或最完整的behavior representation。",
            "",
            "## 4. 四档Z_BDD曲线",
            "",
        ]
    )
    for task in TASKS:
        lines.extend(
            [
                f"### {task}",
                "",
                "| Representation | dose25 | dose50 | dose75 | dose100 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for rep in REPRESENTATIONS:
            values = [number(index[(rep, dose, task)]["z_bdd"]) for dose in DOSES]
            lines.append(f"| {rep} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        lines.append("")
    lines.extend(
        [
            "## 5. Multiplicity与边界",
            "",
            f"- 固定理论矩阵40格；唯一Primary移除后，secondary Holm家族39格。",
            f"- Holm通过{decision['secondary_holm_pass_count']}格，均为ego13的四档×两个task。",
            f"- `NOT_COMPUTABLE`：{decision['not_computable_count']}格；`LOW_N_SECONDARY_DIAGNOSTIC`：{decision['low_n_count']}格。",
            "- `LAT.DYNAMICS`仍是38场景的pre-treatment high-motion `MIXED_PROXY`，不是pure lateral dynamics ground truth。",
            "- 未跨representation比较raw MMD²；未改变Stage6V资格结论；未重跑planner；未修改训练或checkpoint。",
            "",
            "## 6. E2冻结结论",
            "",
            "`STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING`",
            "",
            "下一步仅允许E3报告整合与论文表达，不允许任何rescue experiment或模型返工。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--docs-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    result_dir = (args.result_dir or repo / "outputs/stage7l_e_prospective_bdd_v1").resolve()
    docs_dir = (args.docs_dir or repo / "docs").resolve()
    d_manifest, d_manifest_sha = validate_stage7l_d(repo)
    implementation_path = repo / "docs/stage7l_e_implementation_freeze_manifest_v1.json"
    implementation = read_json(implementation_path)
    if implementation["next_authorized_step"] != "E2_FORMAL_FIVE_REPRESENTATION_INFERENCE_AND_FROZEN_40_CELL_BDD_ONLY":
        raise RuntimeError("Stage7L-E E2 was not authorized by the implementation freeze")
    rows, decision = validate_results(result_dir)
    embeddings = embedding_contracts(result_dir)
    ledger = build_provenance_ledger(result_dir, rows, embeddings)
    ledger_path = result_dir / "cell_provenance_ledger.csv"
    atomic_write_csv(ledger_path, ledger)
    report_path = docs_dir / "stage7l_e_machine_result_freeze_zh.md"
    atomic_write_text(report_path, format_report(rows, decision))
    index = {(row["representation"], row["dose"], row["task"]): row for row in rows}
    artifact_names = (
        "all_bdd_cells.csv",
        "primary_bdd_result.json",
        "secondary_bdd_cells.csv",
        "secondary_holm_results.csv",
        "dose_response_standardized_sensitivity.csv",
        "representation_comparison_dose100_lane_change.csv",
        "stage7l_e_final_decision.json",
        "paired_null_samples.npz",
        "input_contract_audit.json",
        "task_mask_audit.json",
        "cell_provenance_ledger.csv",
    )
    manifest = {
        "schema_version": "stage7l_e_machine_result_freeze_manifest_v1",
        "status": "STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING",
        "branch": "20260611_stage7_conclusion",
        "scope": "E2_MACHINE_RESULTS_ONLY_NO_THESIS_OR_UNIFIED_REPORTING_UPDATE",
        "frozen_provenance": {
            "stage7l_d_commit": EXPECTED_D_COMMIT,
            "e2_preformal_resume_commit": EXPECTED_PREFORMAL_COMMIT,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA,
            "roster_sha256": EXPECTED_ROSTER_SHA,
            "stage7l_d_manifest_sha256": d_manifest_sha,
            "stage7l_e_implementation_manifest_sha256": sha256_file(implementation_path),
            "task_mask_sha256": EXPECTED_TASK_MASK_SHA,
            "formal_inference_and_bdd_sha256": implementation["implementation"]["formal_inference_and_bdd"]["sha256"],
            "machine_freeze_tool_sha256": sha256_file(repo / "tools/stage7l_e_freeze_machine_results.py"),
            "inherited_kernel_analysis_sha256": implementation["inherited_statistics"]["kernel_analysis_sha256"],
        },
        "stage7l_d_unlock": {
            "status": d_manifest["status"],
            "representation_status": d_manifest["representation_status"],
            "all_required_gates_true": True,
        },
        "execution_contract": {
            "official_rollouts_reused": 400,
            "planner_rerun": False,
            "checkpoint_or_training_modified": False,
            "representations": list(REPRESENTATIONS),
            "null": "same_scenario_within_pair_label_swap",
            "null_repetitions": NULL_REPETITIONS,
            "plus_one_p": True,
            "biased_rbf_mmd2": True,
            "representation_specific_median_heuristic": True,
        },
        "task_populations": {
            "LAT.LANE_CHANGE": {dose: 80 for dose in DOSES},
            "LAT.DYNAMICS": {dose: 38 for dose in DOSES},
        },
        "checkpoint_or_scaler_sha256": {
            rep: embeddings[rep]["checkpoint_or_scaler_sha256"] for rep in REPRESENTATIONS
        },
        "primary": decision["primary"],
        "dose100_lane_change_standardized_comparison": [
            {
                "representation": rep,
                "bdd_over_null_q95": number(index[(rep, "dose100", "LAT.LANE_CHANGE")]["bdd_over_null_q95"]),
                "z_bdd": number(index[(rep, "dose100", "LAT.LANE_CHANGE")]["z_bdd"]),
                "raw_p": number(index[(rep, "dose100", "LAT.LANE_CHANGE")]["raw_p"]),
                "multiplicity_role": index[(rep, "dose100", "LAT.LANE_CHANGE")]["multiplicity_role"],
            }
            for rep in REPRESENTATIONS
        ],
        "secondary_family": {
            "theoretical_cells": 40,
            "holm_tests": 39,
            "holm_pass_count": 8,
            "not_computable_count": 0,
            "low_n_count": 20,
        },
        "scientific_boundaries": {
            "cross_representation_raw_mmd2_comparison_performed": False,
            "stage6v_qualification_changed": False,
            "primary_failure_accepted_without_rescue": True,
            "ego13_global_best_claim_allowed": False,
            "stage7l_model_development_allowed": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "result_artifact_sha256": {name: sha256_file(result_dir / name) for name in artifact_names},
        "embedding_manifest_sha256": {rep: embeddings[rep]["manifest_sha256"] for rep in REPRESENTATIONS},
        "machine_report": {"path": str(report_path.relative_to(repo)), "sha256": sha256_file(report_path)},
        "next_authorized_step": "E3_REPORTING_AND_THESIS_INTEGRATION_ONLY_NO_NEW_EXPERIMENT",
    }
    manifest_path = docs_dir / "stage7l_e_machine_result_freeze_manifest_v1.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
