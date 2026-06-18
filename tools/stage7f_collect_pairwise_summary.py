#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROW_SEMANTICS = "scenario × planner-controlled nuPlan ego rollout"
OUTPUT_CSV = "stage7f_pairwise_summary.csv"
OUTPUT_JSON = "stage7f_pairwise_summary.json"
OUTPUT_MD = "stage7f_pairwise_summary.md"


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def scalar(value):
    if pd.isna(value):
        return None
    try:
        if hasattr(value, "item"):
            return value.item()
    except ValueError:
        pass
    return value


def top_abs_row(df: Optional[pd.DataFrame], column: str):
    if df is None or df.empty or column not in df.columns:
        return None
    vals = pd.to_numeric(df[column], errors="coerce").abs()
    if vals.dropna().empty:
        return None
    return df.loc[vals.idxmax()]


def pair_from_name(pair_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    name = pair_dir.name
    if "_vs_" not in name:
        return None, None
    a, b = name.split("_vs_", 1)
    return a, b


def warning_preview(path: Path) -> Tuple[int, Optional[str]]:
    raw = read_json(path)
    warnings = raw.get("warnings", []) if isinstance(raw, dict) else []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    preview = "; ".join(str(x) for x in warnings[:3]) if warnings else None
    return len(warnings), preview


def build_pair_row(pair_dir: Path) -> Dict:
    bdd = read_json(pair_dir / "bdd_summary.json")
    cat = read_csv_optional(pair_dir / "category_delta.csv")
    feat = read_csv_optional(pair_dir / "feature_delta.csv")
    scenario_slice = read_csv_optional(pair_dir / "scenario_slice_delta.csv")
    top_cases = read_csv_optional(pair_dir / "top_drift_cases.csv")
    cat1 = top_abs_row(cat, "delta")
    cat2 = None
    if cat is not None and not cat.empty and "delta" in cat.columns:
        vals = pd.to_numeric(cat["delta"], errors="coerce").abs().sort_values(ascending=False)
        if len(vals.dropna()) >= 2:
            cat2 = cat.loc[vals.index[1]]
    feat1 = top_abs_row(feat, "delta_normalized")
    planner_a, planner_b = pair_from_name(pair_dir)
    warning_count, warnings_preview = warning_preview(pair_dir / "stage6_warnings.json")
    p_value = bdd.get("p_value", bdd.get("permutation_p_value"))
    return {
        "planner_a": planner_a,
        "planner_b": planner_b,
        "pair_name": pair_dir.name,
        "pair_output_dir": str(pair_dir),
        "bdd_mmd2": bdd.get("mmd2", bdd.get("bdd_mmd2")),
        "ci95_low": bdd.get("ci95_low"),
        "ci95_high": bdd.get("ci95_high"),
        "permutation_p_value": p_value,
        "n_A": bdd.get("n_A"),
        "n_B": bdd.get("n_B"),
        "embedding_dim": bdd.get("embedding_dim"),
        "top_category_1": scalar(cat1.get("category")) if cat1 is not None and "category" in cat1 else None,
        "top_category_1_delta": scalar(cat1.get("delta")) if cat1 is not None and "delta" in cat1 else None,
        "top_category_1_cohen_d": scalar(cat1.get("cohen_d")) if cat1 is not None and "cohen_d" in cat1 else None,
        "top_category_1_p_value": scalar(cat1.get("p_value")) if cat1 is not None and "p_value" in cat1 else None,
        "top_category_2": scalar(cat2.get("category")) if cat2 is not None and "category" in cat2 else None,
        "top_category_2_delta": scalar(cat2.get("delta")) if cat2 is not None and "delta" in cat2 else None,
        "top_feature_1": scalar(feat1.get("feature")) if feat1 is not None and "feature" in feat1 else None,
        "top_feature_1_delta_normalized": scalar(feat1.get("delta_normalized")) if feat1 is not None and "delta_normalized" in feat1 else None,
        "top_feature_1_cohen_d": scalar(feat1.get("cohen_d")) if feat1 is not None and "cohen_d" in feat1 else None,
        "top_feature_1_p_value": scalar(feat1.get("permutation_p_value")) if feat1 is not None and "permutation_p_value" in feat1 else None,
        "warning_count": warning_count,
        "warnings_preview": warnings_preview,
        "has_scenario_slice_summary": bool(scenario_slice is not None and not scenario_slice.empty),
        "has_top_drift_cases": bool(top_cases is not None and not top_cases.empty),
    }


def add_interpretation(rows: List[Dict]) -> List[Dict]:
    ordered = sorted(rows, key=lambda r: (float(r["bdd_mmd2"]) if r.get("bdd_mmd2") is not None else float("-inf")), reverse=True)
    top_name = ordered[0]["pair_name"] if ordered else None
    for rank, row in enumerate(ordered, 1):
        row["bdd_rank_desc"] = rank
        p = row.get("permutation_p_value")
        row["significance_label"] = "nominal_p_lt_0.05_exploratory_only" if p is not None and float(p) < 0.05 else "exploratory_only"
        row["effect_size_label"] = "larger_relative_drift" if row["pair_name"] == top_name else "small_or_uncalibrated"
        row["interpretation_note"] = "Sample sizes are reported from the actual pairwise outputs; BDD measures distribution drift magnitude only, not direction; category/feature deltas are interpretation layers."
    return ordered


def collect_pairwise_summary(stage7f_dir: Path, output_dir: Optional[Path] = None, overwrite: bool = False) -> Dict:
    stage7f_dir = Path(stage7f_dir)
    output_dir = Path(output_dir) if output_dir else stage7f_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [OUTPUT_CSV, OUTPUT_JSON, OUTPUT_MD]:
        if (output_dir / name).exists() and not overwrite:
            raise FileExistsError(f"Output exists: {output_dir / name}. Use --overwrite.")
    pair_root = stage7f_dir / "stage6_pairwise"
    pair_dirs = sorted([p for p in pair_root.glob("*") if p.is_dir()]) if pair_root.exists() else []
    rows = add_interpretation([build_pair_row(p) for p in pair_dirs if (p / "bdd_summary.json").exists()])
    summary = {"stage7f_dir": str(stage7f_dir), "pairwise_root": str(pair_root), "num_pairs": len(rows), "rows": rows}
    pd.DataFrame(rows).to_csv(output_dir / OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    (output_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(stage7f_dir, output_dir / OUTPUT_MD, rows)
    return {"csv": str(output_dir / OUTPUT_CSV), "json": str(output_dir / OUTPUT_JSON), "md": str(output_dir / OUTPUT_MD), "num_pairs": len(rows)}


def format_n_range(rows: List[Dict]) -> str:
    vals = []
    for key in ["n_A", "n_B"]:
        nums = [int(r[key]) for r in rows if r.get(key) is not None]
        if not nums:
            vals.append(f"{key}=unavailable")
        elif min(nums) == max(nums):
            vals.append(f"{key}={nums[0]}")
        else:
            vals.append(f"{key}={min(nums)}-{max(nums)}")
    return ", ".join(vals)


def fallback_rate_text(summary: Dict) -> str:
    fallback = summary.get("fallback", {}) if isinstance(summary, dict) else {}
    val = fallback.get("fallback_rate")
    return "unavailable" if val is None else str(val)


def write_markdown(stage7f_dir: Path, out_path: Path, rows: List[Dict]) -> None:
    s = read_json(stage7f_dir / "stage7f_summary.json")
    align, fallback = s.get("alignment", {}), s.get("fallback", {})
    df = pd.DataFrame(rows)
    table_cols = ["bdd_rank_desc", "pair_name", "bdd_mmd2", "permutation_p_value", "top_category_1", "top_feature_1", "warning_count"]
    table = df[table_cols].to_markdown(index=False) if not df.empty else "_No pairwise rows found._"
    top = rows[0]["pair_name"] if rows else "unavailable"
    low = rows[-1]["pair_name"] if rows else "unavailable"
    warning_total = sum(int(r.get("warning_count") or 0) for r in rows)
    mode_text = s.get("mode", "unavailable")
    fallback_rate = fallback_rate_text(s)
    n_range = format_n_range(rows)
    if mode_text == "full" and fallback.get("fallback_preserving_status") is True:
        mode_text = "full fallback-preserving"
    lines = [
        "# Stage7F Pairwise Summary", "",
        f"- source Stage7F directory: `{stage7f_dir}`",
        f"- mode: `{mode_text}`",
        f"- row semantics: `{s.get('row_semantics', ROW_SEMANTICS)}`",
        f"- number of scenarios: `{align.get('num_scenarios', 'unavailable')}`",
        f"- number of planners: `{align.get('num_planners', 'unavailable')}`",
        f"- total rows: `{align.get('total_rows', 'unavailable')}`",
        f"- fallback_rate: `{fallback_rate}`",
        f"- map_name_resolved_rate: `{fallback.get('map_name_resolved_rate', 'unavailable')}`",
        f"- map_query_success: `{fallback.get('map_query_success', 'unavailable')}`",
        f"- lane_info_count: `{fallback.get('lane_info_count', 'unavailable')}`", "",
        "## Ranked pairwise table", "", table, "",
        f"- top pair by BDD: `{top}`",
        f"- lowest pair by BDD: `{low}`", "",
        "## Warnings summary", "", f"- total warning entries across pairs: `{warning_total}`", "",
        "## Limitations", "",
        f"- actual pairwise sample sizes from pair outputs: `{n_range}`.",
        "- permutation p-values are low power.",
        "- BDD scale is uncalibrated without negative/positive controls.",
        f"- mode `{mode_text}` uses actual fallback_rate `{fallback_rate}`; if unavailable, inspect stage7f_summary.json and accompany full main result with strict-filter sensitivity.",
        "- BDD measures distribution drift magnitude only, not direction; category/feature deltas are interpretation layers.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Collect existing Stage6 pairwise outputs into Stage7F pairwise summary files.")
    p.add_argument("--stage7f_dir", required=True)
    p.add_argument("--output_dir")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_pairwise_summary(Path(args.stage7f_dir), Path(args.output_dir) if args.output_dir else None, args.overwrite)
