#!/usr/bin/env python3
"""Apply the pre-frozen Stage6T decision rule to the one-time blind results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def by_name(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    return {row[field]: row for row in rows}


def pct(value: str) -> str:
    return f"{100 * float(value):.1f}%"


def run(args: argparse.Namespace) -> dict[str, Any]:
    auth = read_json(args.authorization)
    if sha256(args.authorization) != AUTH_SHA or auth.get("status") != "AUTHORIZED_STAGE6_ONE_TIME_BLIND_EVALUATION":
        raise ValueError("unexpected blind evaluation authorization")
    if auth.get("immutability_statement") != "evaluation results cannot trigger retraining or protocol changes":
        raise ValueError("immutability statement changed")

    waymo_manifest = read_json(args.waymo_manifest)
    paired_manifest = read_json(args.paired_manifest)
    unpaired_manifest = read_json(args.unpaired_manifest)
    execution = read_json(args.confirmation_execution)
    if waymo_manifest.get("status") != "FROZEN_WAYMO_DYNAMIC_V2_TEST_COMPLETE":
        raise ValueError("Waymo result is not frozen")
    if paired_manifest.get("status") != "FROZEN_STAGE6J_K_PAIRED_BLIND_COMPLETE":
        raise ValueError("Stage6J/K result is not frozen")
    if unpaired_manifest.get("status") != "FROZEN_STAGE6P_UNPAIRED_BLIND_COMPLETE":
        raise ValueError("Stage6P result is not frozen")

    waymo = by_name(read_csv(args.waymo_decisions), "representation")
    paired = by_name(read_csv(args.paired_decisions), "representation")
    unpaired = by_name(read_csv(args.unpaired_decisions), "representation")
    seed_stability = read_csv(args.unpaired_seed_stability)
    primary = {candidate: f"{candidate}_3407" for candidate in "ABC"}

    candidate_cards: dict[str, Any] = {}
    for candidate in "ABC":
        candidate_cards[candidate] = {
            "waymo_primary_all_gates_pass": truth(waymo[primary[candidate]]["all_waymo_gates_pass"]),
            "waymo_primary_noninferiority_pass": truth(waymo[primary[candidate]]["noninferiority_pass"]),
            "waymo_primary_longitudinal_delta": float(waymo[primary[candidate]]["longitudinal_delta"]),
            "stage6jk_longitudinal_gate_pass": truth(paired[candidate]["frozen_longitudinal_gate_pass"]),
            "stage6p_n400_gate_pass": truth(unpaired[primary[candidate]]["frozen_n400_gate_pass"]),
            "stage6p_context_detection": float(unpaired[primary[candidate]]["context_balanced_detection"]),
            "stage6p_context_fpr": float(unpaired[primary[candidate]]["context_balanced_fpr"]),
            "stage6p_direction_min": float(unpaired[primary[candidate]]["context_balanced_direction_min"]),
        }

    confirmation_complete = execution.get("status") == "CONFIRMATION_EXECUTION_COMPLETE_MECHANISM_EVALUATION_AUTHORIZED"
    mechanism_pass = False
    interaction_increment_pass = False
    c = candidate_cards["C"]
    c_all = bool(
        c["waymo_primary_all_gates_pass"]
        and c["stage6jk_longitudinal_gate_pass"]
        and c["stage6p_n400_gate_pass"]
        and confirmation_complete
        and mechanism_pass
        and interaction_increment_pass
    )
    b = candidate_cards["B"]
    b_longitudinal = bool(
        b["waymo_primary_all_gates_pass"]
        and b["stage6jk_longitudinal_gate_pass"]
        and b["stage6p_n400_gate_pass"]
    )
    if c_all:
        final_decision = "C_SELECTED_AS_FINAL_THESIS_MODEL"
    elif b_longitudinal and mechanism_pass and not interaction_increment_pass:
        final_decision = "B_SELECTED_AS_SIMPLER_FINAL_THESIS_MODEL"
    else:
        final_decision = "NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE"

    result: dict[str, Any] = {
        "schema_version": "stage6v_one_time_blind_evaluation_final_v1",
        "status": "FROZEN_STAGE6V_ONE_TIME_BLIND_EVALUATION_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutability_statement": "evaluation results cannot trigger retraining or protocol changes",
        "primary_seed": 3407,
        "candidate_scorecards": candidate_cards,
        "stage6s_v2": {
            "execution_complete": confirmation_complete,
            "locked_roster_count": int(execution["locked_roster_count"]),
            "succeeded": int(execution["succeeded"]),
            "failed_review_required": int(execution["failed_review_required"]),
            "failure_categories": execution["failure_categories"],
            "mechanism_gate_evaluated": False,
            "mechanism_gate_passed": False,
            "representation_evaluation_run": False,
            "c_full_vs_neighbor_zero_increment_evaluated": False,
            "c_full_vs_neighbor_zero_increment_passed": False,
        },
        "final_model_decision": final_decision,
        "training_or_protocol_modified": False,
        "seed_epoch_loss_architecture_or_benchmark_changed": False,
        "source_sha256": {
            "authorization": sha256(args.authorization),
            "waymo_manifest": sha256(args.waymo_manifest),
            "waymo_decisions": sha256(args.waymo_decisions),
            "paired_manifest": sha256(args.paired_manifest),
            "paired_decisions": sha256(args.paired_decisions),
            "unpaired_manifest": sha256(args.unpaired_manifest),
            "unpaired_decisions": sha256(args.unpaired_decisions),
            "unpaired_seed_stability": sha256(args.unpaired_seed_stability),
            "confirmation_execution": sha256(args.confirmation_execution),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "stage6v_blind_evaluation_final_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    old = unpaired["old64"]
    report = f"""# Stage6V 一次性盲测中文总报告

## 总结结论

状态：`{result['status']}`

预冻结最终决策：`{final_decision}`

本轮最重要的正结果是：在 Stage6P 真正非配对发布条件下，n=400 的 context-balanced 检出率从
old64 的 {pct(old['context_balanced_detection'])} 提升到 A/B/C 的
{pct(unpaired['A_3407']['context_balanced_detection'])}/
{pct(unpaired['B_3407']['context_balanced_detection'])}/
{pct(unpaired['C_3407']['context_balanced_detection'])}。但三者都未同时通过 Waymo primary longitudinal
提升门禁和 Stage6J/K paired 完整门禁；Stage6S-v2 冻结 roster 又因建榜时遗漏 nuPlan 官方
`valid_scenes` 边界规则而未能完整执行。因此没有 A/B/C candidate 满足预冻结的最终论文主模型规则。

## 1. Waymo test：A/B/C分别发生了什么

| Candidate（primary 3407） | longitudinal delta | 其余非劣性 | Waymo完整门禁 |
|---|---:|---|---|
| A | {float(waymo['A_3407']['longitudinal_delta']):+.4f} | 通过 | 未通过；纵向显著下降 |
| B | {float(waymo['B_3407']['longitudinal_delta']):+.4f} | 通过 | 未通过；正向但未达到冻结幅度 |
| C | {float(waymo['C_3407']['longitudinal_delta']):+.4f} | 通过 | 未通过；正向但未达到冻结幅度 |

三个candidate的3/3 seed均通过following/lateral/behavior/retrieval综合非劣性。只有B-3409通过全部Waymo
门禁，但primary seed已固定为3407，不能事后换seed，因此B的primary结论仍是未通过。A与B/C的candidate-specific
total loss定义不同，未作横向比较。

## 2. Stage6J/K paired纵向能力

| Representation | overall Holm | task×dose Holm | 最小检出剂量 | median Z_BDD | 冻结门禁 |
|---|---:|---:|---:|---:|---|
| old64 | {paired['old64']['overall_holm_pass_doses_out_of_4']}/4 | {paired['old64']['task_dose_holm_pass_cells_out_of_12']}/12 | {paired['old64']['minimum_detectable_nominal_dose']} | {float(paired['old64']['median_overall_z_bdd']):.3f} | 未通过 |
| A | {paired['A']['overall_holm_pass_doses_out_of_4']}/4 | {paired['A']['task_dose_holm_pass_cells_out_of_12']}/12 | {paired['A']['minimum_detectable_nominal_dose']} | {float(paired['A']['median_overall_z_bdd']):.3f} | 未通过 |
| B | {paired['B']['overall_holm_pass_doses_out_of_4']}/4 | {paired['B']['task_dose_holm_pass_cells_out_of_12']}/12 | {paired['B']['minimum_detectable_nominal_dose']} | {float(paired['B']['median_overall_z_bdd']):.3f} | 未通过 |
| C | {paired['C']['overall_holm_pass_doses_out_of_4']}/4 | {paired['C']['task_dose_holm_pass_cells_out_of_12']}/12 | {paired['C']['minimum_detectable_nominal_dose']} | {float(paired['C']['median_overall_z_bdd']):.3f} | 未通过 |
| ego13 | {paired['ego13']['overall_holm_pass_doses_out_of_4']}/4 | {paired['ego13']['task_dose_holm_pass_cells_out_of_12']}/12 | {paired['ego13']['minimum_detectable_nominal_dose']} | {float(paired['ego13']['median_overall_z_bdd']):.3f} | 通过 |

ego13仍是明确最强的纵向representation。learned64中A最好，但只与old64持平于4/4 overall和7/12 task cell；
B/C反而只有3/4 overall和2/12 task cell。这说明B/C没有在冻结paired benchmark中恢复丢失的纵向敏感性。
所有representation均使用自己的bandwidth/null，未跨representation比较raw MMD²。

## 3. Stage6P unpaired release

| Representation | n=400 A/A FPR | A/B detection | 两方向最小值 | detection−FPR | 冻结门禁 |
|---|---:|---:|---:|---:|---|
| old64 | {pct(old['context_balanced_fpr'])} | {pct(old['context_balanced_detection'])} | {pct(old['context_balanced_direction_min'])} | {100*(float(old['context_balanced_detection'])-float(old['context_balanced_fpr'])):.1f} pp | 未通过 |
| A-3407 | {pct(unpaired['A_3407']['context_balanced_fpr'])} | {pct(unpaired['A_3407']['context_balanced_detection'])} | {pct(unpaired['A_3407']['context_balanced_direction_min'])} | {100*(float(unpaired['A_3407']['context_balanced_detection'])-float(unpaired['A_3407']['context_balanced_fpr'])):.1f} pp | 通过 |
| B-3407 | {pct(unpaired['B_3407']['context_balanced_fpr'])} | {pct(unpaired['B_3407']['context_balanced_detection'])} | {pct(unpaired['B_3407']['context_balanced_direction_min'])} | {100*(float(unpaired['B_3407']['context_balanced_detection'])-float(unpaired['B_3407']['context_balanced_fpr'])):.1f} pp | 通过 |
| C-3407 | {pct(unpaired['C_3407']['context_balanced_fpr'])} | {pct(unpaired['C_3407']['context_balanced_detection'])} | {pct(unpaired['C_3407']['context_balanced_direction_min'])} | {100*(float(unpaired['C_3407']['context_balanced_detection'])-float(unpaired['C_3407']['context_balanced_fpr'])):.1f} pp | 通过 |
| ego13 | {pct(unpaired['ego13']['context_balanced_fpr'])} | {pct(unpaired['ego13']['context_balanced_detection'])} | {pct(unpaired['ego13']['context_balanced_direction_min'])} | {100*(float(unpaired['ego13']['context_balanced_detection'])-float(unpaired['ego13']['context_balanced_fpr'])):.1f} pp | 通过 |

C满足冻结的≥80%整体、双方向各≥75%、FPR≤7.5%和raw detection≥75%门槛。三个seed的C detection均为
99.5%，FPR为6.5%/5.5%/6.0%，结果稳定。B三个seed均为100% context-balanced detection，FPR为
5.0%/6.0%/7.5%。这是新64D纵向signal recovery最强的工程证据，但不能抵消paired和Waymo门禁失败。

## 4. Stage6S-v2 confirmation

冻结80对中{execution['succeeded']}对成功，{execution['failed_review_required']}对在官方nuPlan查询前失败；失败均为
`NUPLAN_VALID_SCENES_BOUNDARY_EXCLUSION`。官方查询要求scene排序满足`row_num >= 3 AND row_num < scene_count - 1`，
而pre-treatment inventory没有复用该规则。失败token均真实存在于DB，但位于首个或倒数第二个scene。

对失败项完成原token重试后结论不变。由于roster已冻结，不能用成功子集重新定义confirmation，也不能换场景或
修改门槛。因此trajectory mechanism gate未执行、interaction embedding/BDD未读取，C full-context相对
C neighbor-zero的null-standardized ΔZ及bootstrap CI均未评估。这里的答案不是“没有interaction增量”，而是
“本轮确认实验未产生可判定证据”。

## 5. 最终模型决策

- C：Stage6P通过，但Waymo primary、Stage6J/K和Stage6S-v2完整证据链未通过，不能成为最终论文主模型。
- B：Stage6P最强且稳定，但同样未通过Waymo primary与Stage6J/K，不能依据简化优先规则入选。
- A：Stage6P明显改善，但Waymo纵向下降且paired不优于old64，不入选。

因此预冻结决策为：**A/B/C均不具备成为最终论文主模型的充分证据**。old64继续作为冻结历史baseline，ego13继续
作为纵向敏感性参考上界；不能把ego13解释为interaction/context无用。

## 6. 可写入论文与必须作为限制的内容

可以写入论文：Dynamic v2训练与纵向目标使新64D在异log/异场景的非配对发布检测上产生大幅、跨seed稳定提升；
B和C把old64的66.5% context-balanced检出率提升到100%和99.5%，同时A/A FPR受控。该结果支持“表示学习可以改善
受控纵向版本差异的release-level检出”，但只限于冻结nuPlan/PDM treatment与本统计口径。

必须作为限制或负结果：A/B/C未通过完整Waymo纵向提升与paired task-coverage联合门禁；paired与unpaired结论不一致；
Stage6S-v2 confirmation因roster建榜遗漏官方scene边界可运行性规则而失败，无法证明C的context增量；本轮不能选出符合
全部预注册条件的最终64D模型。结果不能外推为通用BDD阈值、真实整车厂发布可靠性或安全有效性。

本轮没有训练返工、换seed、换epoch、改loss、改architecture、修改benchmark或读取被条件禁止的Stage6S-v2 embedding。
"""
    report_path = args.output_dir / "stage6v_blind_evaluation_final_report_zh.md"
    report_path.write_text(report, encoding="utf-8")
    result["manifest_sha256"] = sha256(manifest)
    result["report_sha256"] = sha256(report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--waymo_manifest", type=Path, required=True)
    parser.add_argument("--waymo_decisions", type=Path, required=True)
    parser.add_argument("--paired_manifest", type=Path, required=True)
    parser.add_argument("--paired_decisions", type=Path, required=True)
    parser.add_argument("--unpaired_manifest", type=Path, required=True)
    parser.add_argument("--unpaired_decisions", type=Path, required=True)
    parser.add_argument("--unpaired_seed_stability", type=Path, required=True)
    parser.add_argument("--confirmation_execution", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
