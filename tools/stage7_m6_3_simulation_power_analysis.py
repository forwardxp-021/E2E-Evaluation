#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_2_locked_task_bdd import (
    PRETREATMENT_TASKS,
    build_pretreatment_task_masks,
    read_csv_records,
)
from tools.stage7_m6_scenario_conditioned_bdd import (
    holm_adjust,
    markdown_table,
    sha256_file,
    validate_and_build_pairs,
)


def parse_numeric_grid(raw: str, *, cast: type) -> List[Any]:
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("numeric grid cannot be empty")
    if any(value <= 0 for value in values):
        raise ValueError("numeric grid values must be positive")
    if len(set(values)) != len(values):
        raise ValueError("numeric grid values must be unique")
    return sorted(values)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def paired_kernel_quadratic(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[np.ndarray, float]:
    if values_a.shape != values_b.shape:
        raise ValueError("paired kernel requires equal A/B shapes")
    if len(values_a) < 2:
        raise ValueError("paired kernel requires at least two pairs")
    pooled = np.vstack([values_a, values_b]).astype(np.float64, copy=False)
    squared_norm = np.sum(pooled * pooled, axis=1)
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (pooled @ pooled.T)
    )
    np.maximum(squared_distance, 0.0, out=squared_distance)
    upper = np.sqrt(
        squared_distance[np.triu_indices(len(pooled), k=1)]
    )
    positive = upper[np.isfinite(upper) & (upper > 0)]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    if bandwidth <= 1e-8:
        bandwidth = 1.0
    kernel = np.exp(-squared_distance / (2.0 * bandwidth * bandwidth))
    n_pairs = len(values_a)
    aa = kernel[:n_pairs, :n_pairs]
    bb = kernel[n_pairs:, n_pairs:]
    ab = kernel[:n_pairs, n_pairs:]
    ba = kernel[n_pairs:, :n_pairs]
    quadratic = (aa + bb - ab - ba) / float(n_pairs * n_pairs)
    observed = float(np.sum(quadratic))
    return quadratic, observed


def fast_paired_permutation_p(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    quadratic, observed = paired_kernel_quadratic(values_a, values_b)
    signs = rng.choice((-1.0, 1.0), size=(permutations, len(values_a)))
    null = np.sum((signs @ quadratic) * signs, axis=1)
    exceedances = int(np.sum(null >= observed - 1e-15))
    return float((exceedances + 1) / (permutations + 1))


class EmpiricalPairedGenerator:
    def __init__(self, values_a: np.ndarray, values_b: np.ndarray):
        if values_a.shape != values_b.shape or len(values_a) < 2:
            raise ValueError("empirical generator requires at least two equal-shape pairs")
        midpoint = 0.5 * (values_a + values_b)
        difference = values_a - values_b
        self.midpoint_mean = np.mean(midpoint, axis=0)
        self.midpoint_residual = midpoint - self.midpoint_mean
        self.difference_mean = np.mean(difference, axis=0)
        self.difference_residual = difference - self.difference_mean
        self.pilot_pairs = len(values_a)

    def sample(
        self,
        n_pairs: int,
        *,
        effect_scale: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if n_pairs < 2:
            raise ValueError("n_pairs must be at least two")
        midpoint_indices = rng.integers(0, self.pilot_pairs, size=n_pairs)
        difference_indices = rng.integers(0, self.pilot_pairs, size=n_pairs)
        midpoint = self.midpoint_mean + self.midpoint_residual[midpoint_indices]
        difference = (
            effect_scale * self.difference_mean
            + self.difference_residual[difference_indices]
        )
        return midpoint + 0.5 * difference, midpoint - 0.5 * difference


def simulate_power_grid(
    generators: Dict[str, EmpiricalPairedGenerator],
    *,
    candidate_pairs: Sequence[int],
    effect_scales: Sequence[float],
    simulations: int,
    permutations: int,
    alpha: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "overall_primary" not in generators:
        raise ValueError("generators must contain overall_primary")
    task_names = [name for name in generators if name != "overall_primary"]
    rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    master_rng = np.random.default_rng(seed)

    for effect_scale in effect_scales:
        for n_pairs in candidate_pairs:
            overall_reject = 0
            task_raw_reject = {task: 0 for task in task_names}
            task_holm_reject = {task: 0 for task in task_names}
            all_tasks_holm_reject = 0
            any_task_holm_reject = 0
            progress = tqdm(
                range(simulations),
                desc=f"power scale={effect_scale:g} n={n_pairs}",
                unit="sim",
                leave=False,
            )
            for _ in progress:
                simulation_seed = int(
                    master_rng.integers(0, np.iinfo(np.int64).max)
                )
                simulation_rng = np.random.default_rng(simulation_seed)
                overall_a, overall_b = generators["overall_primary"].sample(
                    n_pairs, effect_scale=effect_scale, rng=simulation_rng
                )
                overall_p = fast_paired_permutation_p(
                    overall_a,
                    overall_b,
                    permutations=permutations,
                    rng=simulation_rng,
                )
                overall_reject += int(overall_p <= alpha)

                task_p_values = []
                for task in task_names:
                    values_a, values_b = generators[task].sample(
                        n_pairs, effect_scale=effect_scale, rng=simulation_rng
                    )
                    p_value = fast_paired_permutation_p(
                        values_a,
                        values_b,
                        permutations=permutations,
                        rng=simulation_rng,
                    )
                    task_p_values.append(p_value)
                    task_raw_reject[task] += int(p_value <= alpha)
                adjusted = holm_adjust(task_p_values)
                decisions = [value <= alpha for value in adjusted]
                for task, decision in zip(task_names, decisions):
                    task_holm_reject[task] += int(decision)
                all_tasks_holm_reject += int(all(decisions))
                any_task_holm_reject += int(any(decisions))

            endpoints = [
                (
                    "overall_primary",
                    "overall_primary",
                    overall_reject,
                    "unadjusted_primary_alpha",
                )
            ]
            endpoints.extend(
                (
                    task,
                    "task_conditioned_secondary",
                    task_holm_reject[task],
                    "holm_across_five_tasks",
                )
                for task in task_names
            )
            for endpoint, family, successes, multiplicity in endpoints:
                low, high = wilson_interval(successes, simulations)
                raw_successes = (
                    overall_reject
                    if endpoint == "overall_primary"
                    else task_raw_reject[endpoint]
                )
                rows.append(
                    {
                        "endpoint": endpoint,
                        "family": family,
                        "effect_scale_vs_development_pilot_mean_shift": effect_scale,
                        "candidate_pairs": n_pairs,
                        "simulations": simulations,
                        "planning_permutations": permutations,
                        "alpha": alpha,
                        "multiplicity": multiplicity,
                        "rejections": successes,
                        "power": successes / simulations,
                        "power_ci95_low": low,
                        "power_ci95_high": high,
                        "raw_unadjusted_power": raw_successes / simulations,
                        "pilot_pairs": generators[endpoint].pilot_pairs,
                    }
                )
            family_low, family_high = wilson_interval(
                all_tasks_holm_reject, simulations
            )
            family_rows.append(
                {
                    "endpoint": "all_five_tasks_reject_after_holm",
                    "effect_scale_vs_development_pilot_mean_shift": effect_scale,
                    "candidate_pairs_per_task": n_pairs,
                    "simulations": simulations,
                    "rejections": all_tasks_holm_reject,
                    "power": all_tasks_holm_reject / simulations,
                    "power_ci95_low": family_low,
                    "power_ci95_high": family_high,
                    "any_task_power": any_task_holm_reject / simulations,
                }
            )
    return rows, {"task_family_rows": family_rows, "task_names": task_names}


def choose_targets(
    rows: Sequence[Dict[str, Any]],
    family_rows: Sequence[Dict[str, Any]],
    *,
    target_effect_scale: float,
    target_power: float,
    attrition_rate: float,
) -> Dict[str, Any]:
    endpoints = sorted(set(row["endpoint"] for row in rows))
    selections: Dict[str, Any] = {}
    for endpoint in endpoints:
        eligible = sorted(
            (
                row
                for row in rows
                if row["endpoint"] == endpoint
                and math.isclose(
                    row["effect_scale_vs_development_pilot_mean_shift"],
                    target_effect_scale,
                )
                and row["power"] >= target_power
            ),
            key=lambda row: row["candidate_pairs"],
        )
        if eligible:
            selected = dict(eligible[0])
            selected["gross_pairs_with_attrition"] = int(
                math.ceil(selected["candidate_pairs"] / (1.0 - attrition_rate))
            )
            selections[endpoint] = selected
        else:
            selections[endpoint] = {
                "status": "NOT_REACHED_WITHIN_CANDIDATE_GRID",
                "target_effect_scale": target_effect_scale,
                "target_power": target_power,
            }

    family_eligible = sorted(
        (
            row
            for row in family_rows
            if math.isclose(
                row["effect_scale_vs_development_pilot_mean_shift"],
                target_effect_scale,
            )
            and row["power"] >= target_power
        ),
        key=lambda row: row["candidate_pairs_per_task"],
    )
    family_selection: Dict[str, Any]
    if family_eligible:
        family_selection = dict(family_eligible[0])
        family_selection["gross_total_pairs_for_five_disjoint_task_quotas"] = int(
            math.ceil(
                5
                * family_selection["candidate_pairs_per_task"]
                / (1.0 - attrition_rate)
            )
        )
    else:
        family_selection = {
            "status": "NOT_REACHED_WITHIN_CANDIDATE_GRID",
            "target_effect_scale": target_effect_scale,
            "target_power": target_power,
        }
    return {
        "selection_rule": (
            "smallest candidate n with simulated rejection probability >= target power"
        ),
        "target_effect_scale_vs_development_pilot_mean_shift": target_effect_scale,
        "target_power": target_power,
        "assumed_pair_attrition_rate": attrition_rate,
        "endpoint_selections": selections,
        "simultaneous_all_five_task_selection": family_selection,
        "design_ready": bool(
            all("candidate_pairs" in selection for selection in selections.values())
            and "candidate_pairs_per_task" in family_selection
        ),
    }


def plot_power_curves(
    rows: pd.DataFrame,
    *,
    target_effect_scale: float,
    target_power: float,
    output_path: Path,
) -> None:
    selected = rows.loc[
        np.isclose(
            rows["effect_scale_vs_development_pilot_mean_shift"],
            target_effect_scale,
        )
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for endpoint, group in selected.groupby("endpoint", sort=False):
        group = group.sort_values("candidate_pairs")
        ax.plot(
            group["candidate_pairs"],
            group["power"],
            marker="o",
            label=endpoint,
        )
    ax.axhline(target_power, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Complete pairs (overall or per task)")
    ax.set_ylabel("Simulated rejection probability")
    ax.set_title(
        f"M6.3 pilot power at {target_effect_scale:g}x development mean shift"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation-based power planning for frozen Stage7 M6 paired BDD."
    )
    parser.add_argument("--embedding_path", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--m6_2_lock_spec", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--planner_a", required=True)
    parser.add_argument("--planner_b", required=True)
    parser.add_argument("--candidate_pairs", default="12,20,30,45,60,80,120")
    parser.add_argument("--effect_scales", default="0.5,0.75,1.0")
    parser.add_argument("--target_effect_scale", type=float, default=0.75)
    parser.add_argument("--target_power", type=float, default=0.80)
    parser.add_argument("--attrition_rate", type=float, default=0.20)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--planning_permutations", type=int, default=999)
    parser.add_argument("--blas_threads", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_pairs = parse_numeric_grid(args.candidate_pairs, cast=int)
    effect_scales = parse_numeric_grid(args.effect_scales, cast=float)
    if args.target_effect_scale not in effect_scales:
        raise ValueError("target_effect_scale must appear in effect_scales")
    if not 0 < args.target_power < 1:
        raise ValueError("target_power must be between zero and one")
    if not 0 <= args.attrition_rate < 1:
        raise ValueError("attrition_rate must be in [0, 1)")
    if args.simulations < 100:
        raise ValueError("formal planning requires at least 100 simulations")
    if args.planning_permutations < 199:
        raise ValueError("formal planning requires at least 199 permutations")
    if args.blas_threads <= 0:
        raise ValueError("blas_threads must be positive")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir()

    lock_spec = json.loads(args.m6_2_lock_spec.read_text(encoding="utf-8"))
    if lock_spec.get("status") != "FROZEN_BEFORE_NEW_CONFIRMATION_DATA":
        raise ValueError("M6.2 lock spec is not frozen")
    if lock_spec["task_conditioned_secondary"]["task_definitions"] != {
        key: list(value) for key, value in PRETREATMENT_TASKS.items()
    }:
        raise ValueError("M6.2 task definitions differ from analysis tool")

    metadata = pd.read_csv(args.metadata_csv)
    paired_rows = read_csv_records(args.paired_delta_csv)
    embedding = np.asarray(
        np.load(args.embedding_path, mmap_mode="r"), dtype=np.float64
    )
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("embedding must be a finite 2D array")
    pair_indices, _ = validate_and_build_pairs(
        metadata,
        paired_rows,
        len(embedding),
        planner_a=args.planner_a,
        planner_b=args.planner_b,
    )
    task_masks, task_table = build_pretreatment_task_masks(metadata, pair_indices)
    generators = {
        "overall_primary": EmpiricalPairedGenerator(
            embedding[pair_indices[:, 0]], embedding[pair_indices[:, 1]]
        )
    }
    for task, mask in task_masks.items():
        selected = pair_indices[mask]
        if len(selected) < 5:
            raise ValueError(
                f"task {task} has only {len(selected)} pilot pairs; "
                "at least five are required for empirical planning"
            )
        generators[task] = EmpiricalPairedGenerator(
            embedding[selected[:, 0]], embedding[selected[:, 1]]
        )

    with threadpool_limits(limits=args.blas_threads):
        rows, extras = simulate_power_grid(
            generators,
            candidate_pairs=candidate_pairs,
            effect_scales=effect_scales,
            simulations=args.simulations,
            permutations=args.planning_permutations,
            alpha=args.alpha,
            seed=args.seed,
        )
    family_rows = extras["task_family_rows"]
    selection = choose_targets(
        rows,
        family_rows,
        target_effect_scale=args.target_effect_scale,
        target_power=args.target_power,
        attrition_rate=args.attrition_rate,
    )

    power_frame = pd.DataFrame(rows)
    family_frame = pd.DataFrame(family_rows)
    power_frame.to_csv(args.output_dir / "m6_3_power_grid.csv", index=False)
    family_frame.to_csv(
        args.output_dir / "m6_3_simultaneous_task_family_power.csv", index=False
    )
    task_table.to_csv(args.output_dir / "m6_3_pilot_task_counts.csv", index=False)
    plot_power_curves(
        power_frame,
        target_effect_scale=args.target_effect_scale,
        target_power=args.target_power,
        output_path=plot_dir / "m6_3_power_curves_target_effect.png",
    )

    provenance = {
        "analysis_role": "DEVELOPMENT_PILOT_POWER_PLANNING_NOT_CONFIRMATORY",
        "generator": {
            "midpoint": "independent empirical bootstrap of centered pair midpoints",
            "difference": (
                "effect_scale * development mean pair difference plus independent "
                "empirical bootstrap of centered pair-difference residual"
            ),
            "added_parametric_noise": False,
            "labels_from_future_locked_set_used": False,
        },
        "embedding_sha256": sha256_file(args.embedding_path),
        "metadata_sha256": sha256_file(args.metadata_csv),
        "paired_delta_sha256": sha256_file(args.paired_delta_csv),
        "m6_2_lock_spec_sha256": sha256_file(args.m6_2_lock_spec),
        "analysis_tool_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_pairs": candidate_pairs,
        "effect_scales": effect_scales,
        "simulations": args.simulations,
        "planning_permutations": args.planning_permutations,
        "blas_threads": args.blas_threads,
        "alpha": args.alpha,
        "seed": args.seed,
        "limitations": [
            "Power is conditional on the empirical development-pilot generator.",
            "Task pilots contain only 8-9 pairs and therefore have wide model uncertainty.",
            "Effect multipliers are sensitivity assumptions, not estimates from locked data.",
            "Planning permutations need not equal the frozen 100000 final-test permutations.",
        ],
    }
    (args.output_dir / "m6_3_power_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "m6_3_selected_sample_targets.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    family_selection = selection["simultaneous_all_five_task_selection"]
    if selection["design_ready"]:
        common_task_target = int(family_selection["candidate_pairs_per_task"])
        operational_overall_floor = int(
            lock_spec["locked_intake_requirements"][
                "minimum_overall_complete_pairs_operational_floor"
            ]
        )
        power_selected_overall = int(
            selection["endpoint_selections"]["overall_primary"]["candidate_pairs"]
        )
        locked_power_justification = {
            "status": "FROZEN_BEFORE_LOCKED_CONFIRMATION",
            "analysis_role": "DEVELOPMENT_PILOT_POWER_PLANNING_NOT_CONFIRMATORY",
            "m6_2_lock_spec_sha256": sha256_file(args.m6_2_lock_spec),
            "power_analysis_tool_sha256": sha256_file(Path(__file__).resolve()),
            "target_effect_scale_vs_development_pilot_mean_shift": (
                args.target_effect_scale
            ),
            "target_power": args.target_power,
            "simulations_per_grid_cell": args.simulations,
            "planning_permutations": args.planning_permutations,
            "final_confirmation_permutations": 100000,
            "assumed_pair_attrition_rate": args.attrition_rate,
            "power_selected_complete_pairs_overall": power_selected_overall,
            "required_complete_pairs_overall": max(
                power_selected_overall, operational_overall_floor
            ),
            "required_complete_pairs_by_task": {
                task: common_task_target for task in PRETREATMENT_TASKS
            },
            "required_complete_pairs_total_across_disjoint_task_quotas": (
                common_task_target * len(PRETREATMENT_TASKS)
            ),
            "planned_gross_pairs_per_task_with_attrition": int(
                math.ceil(common_task_target / (1.0 - args.attrition_rate))
            ),
            "planned_gross_pairs_total_with_attrition": int(
                family_selection[
                    "gross_total_pairs_for_five_disjoint_task_quotas"
                ]
            ),
            "simultaneous_all_task_power": family_selection["power"],
            "simultaneous_all_task_power_ci95": [
                family_selection["power_ci95_low"],
                family_selection["power_ci95_high"],
            ],
            "freeze_rules": [
                "Do not change task mapping after inspecting locked planner labels.",
                "Do not stop collection based on observed locked effect sizes.",
                "Quality failures may be replaced only under the predeclared intake rules.",
                "The two frozen planner treatment parameter fingerprints must remain unchanged.",
            ],
        }
        (args.output_dir / "m6_3_locked_power_justification.json").write_text(
            json.dumps(
                locked_power_justification, indent=2, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        quota_rows = []
        gross_per_task = int(
            math.ceil(common_task_target / (1.0 - args.attrition_rate))
        )
        pilot_counts = {
            str(row["task"]): int(row["n_pairs"])
            for row in task_table.to_dict("records")
        }
        for task, scenario_types in PRETREATMENT_TASKS.items():
            quota_rows.append(
                {
                    "task": task,
                    "scenario_types": "|".join(scenario_types),
                    "development_pilot_pairs": pilot_counts[task],
                    "required_new_complete_pairs": common_task_target,
                    "planned_gross_pairs_with_attrition": gross_per_task,
                    "required_new_logs_and_scenarios_disjoint": True,
                    "planner_parameter_fingerprints_must_match": True,
                }
            )
        pd.DataFrame(quota_rows).to_csv(
            args.output_dir / "m6_3_locked_collection_quotas.csv", index=False
        )

    target_rows = [
        row
        for row in rows
        if math.isclose(
            row["effect_scale_vs_development_pilot_mean_shift"],
            args.target_effect_scale,
        )
    ]
    report = [
        "# Stage 7 Milestone 6.3 Simulation-based Power Analysis",
        "",
        "## Status",
        "",
        "`DEVELOPMENT_PILOT_POWER_PLANNING_NOT_CONFIRMATORY`",
        "",
        "The alternative generator uses only the 45-pair development pilot. No future locked-set labels are used.",
        "",
        "## Target assumption",
        "",
        f"- smallest planning effect: `{args.target_effect_scale}x` development-pilot mean paired shift",
        f"- target rejection probability: `{args.target_power}`",
        f"- simulations per grid cell: `{args.simulations}`",
        f"- planning permutations per simulated experiment: `{args.planning_permutations}`",
        f"- assumed complete-pair attrition: `{args.attrition_rate}`",
        "",
        "## Target-effect power grid",
        "",
        markdown_table(
            target_rows,
            [
                "endpoint",
                "candidate_pairs",
                "power",
                "power_ci95_low",
                "power_ci95_high",
                "multiplicity",
            ],
        ),
        "## Selected targets",
        "",
        "```json",
        json.dumps(selection, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Limitations",
        "",
        "- This is empirical-pilot planning, not achieved power and not confirmation.",
        "- Task generators are based on only 8-9 pilot pairs.",
        "- If a task does not reach the target within the candidate grid, sample-size increase alone is not yet justified as a solution.",
        "- The final locked analysis still uses the M6.1 frozen 100000-permutation primary test.",
        "",
    ]
    (args.output_dir / "milestone6_3_power_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        f"M6.3 power planning complete: design_ready={selection['design_ready']}, "
        f"target_effect_scale={args.target_effect_scale}"
    )


if __name__ == "__main__":
    main()
