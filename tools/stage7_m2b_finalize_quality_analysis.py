#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CORE_ARRAYS = (
    "context_traj.npy",
    "ego_seq.npy",
    "ego_seq_mask.npy",
    "neighbor_seq.npy",
    "interaction_feat_style.npy",
    "neighbor_slot_ids.npy",
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_core_arrays(baseline_dir: Path, rebuilt_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in CORE_ARRAYS:
        baseline = baseline_dir / name
        rebuilt = rebuilt_dir / name
        if not baseline.is_file():
            raise FileNotFoundError(baseline)
        if not rebuilt.is_file():
            raise FileNotFoundError(rebuilt)
        baseline_hash = sha256(baseline)
        rebuilt_hash = sha256(rebuilt)
        rows.append({
            "file": name,
            "baseline_bytes": baseline.stat().st_size,
            "rebuilt_bytes": rebuilt.stat().st_size,
            "baseline_sha256": baseline_hash,
            "rebuilt_sha256": rebuilt_hash,
            "byte_identical": baseline_hash == rebuilt_hash,
        })
    return rows


def validate_bdd(
    name: str,
    value: Dict[str, Any],
    *,
    expected_rows_per_planner: int,
    alpha: float,
) -> Dict[str, Any]:
    required = {"mmd2", "p_value", "n_A", "n_B", "embedding_dim"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{name} BDD summary missing fields: {missing}")
    n_a = int(value["n_A"])
    n_b = int(value["n_B"])
    if n_a != expected_rows_per_planner or n_b != expected_rows_per_planner:
        raise ValueError(
            f"{name} BDD row count mismatch: n_A={n_a}, n_B={n_b}, "
            f"expected={expected_rows_per_planner}"
        )
    return {
        "name": name,
        "paired_scenarios": expected_rows_per_planner,
        "n_A": n_a,
        "n_B": n_b,
        "embedding_dim": int(value["embedding_dim"]),
        "mmd2": float(value["mmd2"]),
        "permutation_p_value": float(value["p_value"]),
        "significant_at_alpha": float(value["p_value"]) < alpha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize Stage7 Milestone 2B with BDD sensitivity and byte-identity checks."
    )
    parser.add_argument("--quality_dir", type=Path, required=True)
    parser.add_argument("--baseline_context_dir", type=Path, required=True)
    parser.add_argument("--rebuilt_context_dir", type=Path, required=True)
    parser.add_argument("--full_bdd_summary", type=Path, required=True)
    parser.add_argument("--tier_a_bdd_summary", type=Path)
    parser.add_argument("--tier_b_inclusive_bdd_summary", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError(f"--alpha must be in (0,1), got {args.alpha}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    quality = read_json(args.quality_dir / "milestone2b_summary.json")
    if quality.get("overall_verdict") != "PASS":
        raise ValueError("base Milestone 2B quality gate is not PASS")
    tier_a_path = args.tier_a_bdd_summary or (
        args.quality_dir / "bdd_sensitivity/tier_a_assertive_vs_conservative/bdd_summary.json"
    )
    tier_b_path = args.tier_b_inclusive_bdd_summary or (
        args.quality_dir
        / "bdd_sensitivity/tier_b_inclusive_assertive_vs_conservative/bdd_summary.json"
    )
    analyses = [
        validate_bdd(
            "full",
            read_json(args.full_bdd_summary),
            expected_rows_per_planner=int(quality["full_pairs"]),
            alpha=args.alpha,
        ),
        validate_bdd(
            "tier_a",
            read_json(tier_a_path),
            expected_rows_per_planner=int(quality["tier_a_pairs"]),
            alpha=args.alpha,
        ),
        validate_bdd(
            "tier_b_inclusive",
            read_json(tier_b_path),
            expected_rows_per_planner=int(quality["tier_b_inclusive_pairs"]),
            alpha=args.alpha,
        ),
    ]
    dimensions = {row["embedding_dim"] for row in analyses}
    significance_classes = {row["significant_at_alpha"] for row in analyses}
    core_rows = compare_core_arrays(args.baseline_context_dir, args.rebuilt_context_dir)
    checks = {
        "base_quality_gate_pass": True,
        "all_bdd_pair_counts_match_quality_gate": True,
        "embedding_dimensions_match": len(dimensions) == 1,
        "m2a_m2b_core_arrays_byte_identical": all(row["byte_identical"] for row in core_rows),
        "bdd_significance_conclusion_stable": len(significance_classes) == 1,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "milestone": "Stage 7 Milestone 2B final audit",
        "overall_verdict": verdict,
        "scale_readiness": quality["scale_readiness"] if verdict == "PASS" else "BLOCKED",
        "analysis_policy": {
            "primary": "full planner-paired scenarios",
            "sensitivity_only": ["tier_a", "tier_b_inclusive"],
            "reason": (
                "Lane-context quality is measured on realized planner rollouts; "
                "quality subsets are therefore symmetric pair-level sensitivity analyses, "
                "not replacements for the primary full-pair estimate."
            ),
        },
        "alpha": args.alpha,
        "bdd_analyses": analyses,
        "checks": checks,
        "conclusion": (
            "Small, non-significant embedding-distribution BDD is stable across full, "
            "Tier A, and Tier B-inclusive paired datasets."
            if len(significance_classes) == 1 and not analyses[0]["significant_at_alpha"]
            else "BDD significance interpretation changes across quality tiers."
        ),
        "limitations": [
            "17 paired scenarios remain exploratory; thesis-level distributional claims require scale-up."
        ],
    }
    (args.output_dir / "milestone2b_final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "core_array_byte_identity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(core_rows[0]))
        writer.writeheader()
        writer.writerows(core_rows)
    report = [
        "# Stage 7 Milestone 2B Final Audit",
        "",
        f"## Verdict: `{verdict}`",
        "",
        f"- scale readiness: `{summary['scale_readiness']}`",
        f"- full / Tier A / Tier B-inclusive pairs: "
        f"`{quality['full_pairs']} / {quality['tier_a_pairs']} / "
        f"{quality['tier_b_inclusive_pairs']}`",
        "",
        "## BDD sensitivity",
        "",
        "| dataset | pairs | MMD² | permutation p | significant |",
        "| --- | ---: | ---: | ---: | --- |",
        *[
            f"| {row['name']} | {row['paired_scenarios']} | {row['mmd2']:.8f} | "
            f"{row['permutation_p_value']:.6f} | {row['significant_at_alpha']} |"
            for row in analyses
        ],
        "",
        f"Conclusion: {summary['conclusion']}",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in checks.items()],
        "",
        "## Interpretation policy",
        "",
        "- Use all 17 complete planner pairs for the primary estimate.",
        "- Use Tier A and Tier B-inclusive results only as symmetric paired sensitivity checks.",
        "- Do not filter individual planner rows by realized lane-context quality.",
        "- The 17-pair result remains exploratory even though data-quality checks pass.",
    ]
    (args.output_dir / "milestone2b_final_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if verdict != "PASS":
        raise RuntimeError(f"Milestone 2B final audit failed: {checks}")
    print(f"Stage7 Milestone 2B final audit PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
