#!/usr/bin/env python3
"""Reconstruct and numerically audit Stage 6M pre-treatment covariate balance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage6d_unpaired_version_bdd as stage6d
from tools import stage6e_calibrate_unpaired_release as stage6e


IDENTITY = [
    "target_scenarios_per_release",
    "experiment_set",
    "family",
    "repetition",
    "split_seed",
    "planner_A",
    "planner_B",
]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> pd.DataFrame:
    freeze = read_json(args.freeze_manifest.resolve())
    if freeze.get("status") != "FROZEN_BEFORE_STAGE6M_AGGREGATED_RELIABILITY_RESULTS":
        raise ValueError("Stage 6M freeze manifest is not authoritative")
    source = freeze["sources"]
    config = stage6e.validate_config(read_json(Path(source["stage6h_config"]["path"])))
    design = config["stage6d_design"]
    metadata = pd.read_csv(source["embedding_pool_metadata"]["path"])
    pairmeta = metadata.drop_duplicates(config["pair_id_column"]).copy()
    pairmeta = pairmeta[
        [config["pair_id_column"], config["cluster_column"], "map_name", "scenario_type"]
    ].reset_index(drop=True)
    assignments = pd.read_csv(source["log_assignments"]["path"])
    rows: list[dict[str, Any]] = []
    for identity, assigned in assignments.groupby(IDENTITY, sort=False, dropna=False):
        base = dict(zip(IDENTITY, identity))
        logs_a = set(assigned.loc[assigned["release_group"] == "A", "log_name"].astype(str))
        logs_b = set(assigned.loc[assigned["release_group"] == "B", "log_name"].astype(str))
        if logs_a & logs_b:
            raise ValueError(f"log overlap in trial: {base}")
        frame_a = pairmeta.loc[pairmeta[config["cluster_column"]].astype(str).isin(logs_a)].copy()
        frame_b = pairmeta.loc[pairmeta[config["cluster_column"]].astype(str).isin(logs_b)].copy()
        frame_a["_release_group"] = "A"
        frame_b["_release_group"] = "B"
        frame = pd.concat([frame_a, frame_b], ignore_index=True)
        scopes: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(frame), dtype=bool))]
        scopes.extend(
            (str(task["name"]), stage6d.task_mask(frame, task)) for task in config["tasks"]
        )
        for scope, mask in scopes:
            current, _ = stage6d.coarsen_covariates(frame.loc[mask].reset_index(drop=True), design)
            standardization = stage6d.build_standardization(current, design)
            balances = pd.DataFrame(stage6d.balance_table(current, design, standardization["weights"]))
            for covariate, covariate_rows in balances.groupby("covariate", sort=False):
                raw_abs = float(np.nanmax(np.abs(covariate_rows["raw_difference"])))
                balanced_abs = float(
                    np.nanmax(np.abs(covariate_rows["standardized_difference"]))
                )
                rows.append(
                    {
                        **base,
                        "scope": scope,
                        "covariate": covariate,
                        "standardization_status": standardization["status"],
                        "max_absolute_raw_level_difference": raw_abs,
                        "max_absolute_balanced_level_difference": balanced_abs,
                        "support_fraction_A": standardization["group_A"]["support_fraction"],
                        "support_fraction_B": standardization["group_B"]["support_fraction"],
                        "ess_ratio_A": standardization["group_A"]["ess_ratio"],
                        "ess_ratio_B": standardization["group_B"]["ess_ratio"],
                        "max_weight_ratio_A": standardization["group_A"]["max_weight_ratio"],
                        "max_weight_ratio_B": standardization["group_B"]["max_weight_ratio"],
                    }
                )
    audit = pd.DataFrame(rows)
    comparable = audit["standardization_status"] == stage6d.PASS_STATUS
    not_comparable_scope_trials = int(
        audit.loc[~comparable, IDENTITY + ["scope"]].drop_duplicates().shape[0]
    )
    tolerance = 1e-12
    maximum = float(audit.loc[comparable, "max_absolute_balanced_level_difference"].max())
    if maximum > tolerance:
        raise ValueError(f"balanced covariate difference exceeds tolerance: {maximum}")
    summary_rows: list[dict[str, Any]] = []
    for (size, scope, covariate), group in audit.groupby(
        ["target_scenarios_per_release", "scope", "covariate"], sort=True
    ):
        summary_rows.append(
            {
                "target_scenarios_per_release": int(size),
                "scope": scope,
                "covariate": covariate,
                "trials": int(len(group)),
                "raw_max_abs_difference_median": float(
                    group["max_absolute_raw_level_difference"].median()
                ),
                "raw_max_abs_difference_q95": float(
                    group["max_absolute_raw_level_difference"].quantile(0.95)
                ),
                "raw_max_abs_difference_max": float(
                    group["max_absolute_raw_level_difference"].max()
                ),
                "balanced_max_abs_difference_max": float(
                    group["max_absolute_balanced_level_difference"].max()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "stage6m_covariate_balance_by_trial.csv", index=False)
    summary.to_csv(args.output_dir / "stage6m_covariate_balance_summary.csv", index=False)
    (args.output_dir / "stage6m_covariate_balance_audit_zh.md").write_text(
        "# Stage 6M 协变量平衡审计\n\n"
        "- matching变量仅为pre-treatment的`map_name`和`scenario_type`；task也是冻结的pre-treatment分层。\n"
        f"- 重建scope-trial-covariate审计行：`{len(audit)}`。\n"
        f"- standardized后最大类别比例差：`{maximum:.3g}`（容差`{tolerance:.1g}`），PASS。\n"
        f"- 冻结门禁下不可比scope-trial：`{not_comparable_scope_trials}`；这些项不进入对应方法的A/A阈值或A/B检出率。\n"
        "- 可比trial的common support、ESS和最大权重均按冻结门禁重新计算并PASS。\n"
        "- 该平衡只消除已测量的map/scenario-type构成差异，不排除未测量ODD混杂。\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(audit), "not_comparable_scope_trials": not_comparable_scope_trials, "maximum_balanced_difference": maximum}, ensure_ascii=False))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
