#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if False:  # imported for static contract checks; avoid importing numpy-heavy Stage5D core at CLI import time
    from tools.stage5d_context_core import SLOT_NAMES


def slot_names() -> List[str]:
    return list(("front", "left_front", "left_rear", "right_front", "right_rear"))




def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_shard_paths(dataset_dir: Path) -> List[Path]:
    manifest = read_json(dataset_dir / "shard_manifest.json")
    out: List[Path] = []
    for row in manifest.get("shards", []) if isinstance(manifest.get("shards"), list) else []:
        sp = row.get("shard_path") if isinstance(row, dict) else None
        if sp:
            p = Path(sp)
            out.append(p if p.is_absolute() else dataset_dir / p)
    if out:
        return out
    shards_dir = dataset_dir / "shards"
    if shards_dir.exists():
        return sorted(p for p in shards_dir.iterdir() if p.is_dir())
    return [dataset_dir]


def slot_coverage_and_switches(dataset_dir: Path, max_rows: Optional[int] = None) -> Dict[str, Any]:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return {
            "slot_coverage_by_slot": {},
            "slot_switch_rate_by_slot": {},
            "slot_switch_count_by_slot": {},
            "array_scan_warning": "numpy is unavailable; skipped neighbor_seq.npy and neighbor_slot_ids.npy array diagnostics",
        }
    valid_counts = Counter()
    total_counts = Counter()
    transitions = Counter()
    switches = Counter()
    rows_seen = 0
    for shard in load_shard_paths(dataset_dir):
        nbr_path = shard / "neighbor_seq.npy"
        ids_path = shard / "neighbor_slot_ids.npy"
        if nbr_path.exists():
            nbr = np.load(nbr_path, mmap_mode="r")
            take = nbr.shape[0] if max_rows is None else max(0, min(nbr.shape[0], max_rows - rows_seen))
            if take > 0:
                for si, sn in enumerate(slot_names()):
                    valid = nbr[:take, si, :, 0] > 0.5
                    valid_counts[sn] += int(np.sum(valid))
                    total_counts[sn] += int(valid.size)
                rows_seen += take
        if ids_path.exists():
            ids = np.load(ids_path, allow_pickle=True)
            take = ids.shape[0] if max_rows is None else min(ids.shape[0], max_rows)
            for row in ids[:take]:
                for si, sn in enumerate(slot_names()):
                    seq = list(row[si]) if np.asarray(row[si]).ndim else [str(row[si])]
                    prev = "-1"
                    for tok in seq:
                        tok = str(tok)
                        if tok in {"", "None", "nan"}:
                            tok = "-1"
                        if tok != "-1" and prev != "-1":
                            transitions[sn] += 1
                            if tok != prev:
                                switches[sn] += 1
                        prev = tok
    return {
        "slot_coverage_by_slot": {sn: safe_rate(valid_counts[sn], total_counts[sn]) for sn in slot_names()},
        "slot_switch_rate_by_slot": {sn: safe_rate(switches[sn], transitions[sn]) for sn in slot_names()},
        "slot_switch_count_by_slot": {sn: int(switches[sn]) for sn in slot_names()},
    }


def summarize_lane_debug_csv(dataset_dir: Path) -> Dict[str, Any]:
    method_counts = {sn: Counter() for sn in slot_names()}
    reason_counts = Counter()
    for shard in load_shard_paths(dataset_dir):
        csv_path = shard / "lane_assignment_debug.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sn = row.get("slot_name", "")
                if sn in method_counts:
                    method = row.get("assignment_method") or row.get("method") or "unknown"
                    method_counts[sn][method] += 1
                reason = row.get("rejection_reason") or row.get("fallback_reason") or row.get("reason") or ""
                if reason:
                    reason_counts[reason] += 1
    return {
        "assignment_method_counts_by_slot_from_debug_csv": {sn: dict(method_counts[sn]) for sn in slot_names()},
        "debug_csv_reason_counts": dict(reason_counts),
    }


def summarize_waymo(dataset_dir: Path, max_rows: Optional[int]) -> Dict[str, Any]:
    exported = read_json(dataset_dir / "waymo_lane_aware_diagnostics.json")
    if exported:
        exported.setdefault("dataset", "waymo_stage5")
        exported.setdefault("path", str(dataset_dir))
        exported.setdefault("source_files", [str(dataset_dir / "waymo_lane_aware_diagnostics.json")])
        exported.setdefault("filtering_mode", detect_filtering_mode(exported, "waymo"))
        exported.setdefault("diagnostic_source_note", diagnostic_source_note(exported, "waymo"))
        return exported
    summary = read_json(dataset_dir / "build_summary.json") or read_json(dataset_dir / "neighbor_context_summary.json")
    n = int(summary.get("n_windows_kept", 0) or summary.get("n_windows", 0) or 0)
    metrics = {
        "dataset": "waymo_stage5",
        "path": str(dataset_dir),
        "n_rows": n,
        "lane_assignment_available": bool(summary.get("lane_assignment_success_rate", 0.0) > 0 or summary.get("lane_assignment_success_count_kept", 0) > 0),
        "lane_assignment_available_rate": as_float(summary.get("lane_assignment_success_rate"), 0.0),
        "fallback_assignment_used_rate": as_float(summary.get("fallback_assignment_rate"), as_float(summary.get("fallback_assignment_used_rate"), None)),
        "candidate_projection_success_rate": as_float(summary.get("lane_projection_success_rate"), None),
        "adjacency_source_counts": summary.get("adjacency_source_counts", {}),
        "lane_context_quality_counts": summary.get("lane_context_quality_counts", {}),
        "lane_context_quality_reason_counts": summary.get("lane_context_quality_reason_counts", {}),
        "rejection_reason_counts": summary.get("slot_rejection_reason_counts", {}),
        "slot_coverage_by_slot": summary.get("slot_valid_ratio") or {sn: 1.0 - as_float((summary.get("empty_slot_ratio_by_slot") or {}).get(sn), 1.0) for sn in slot_names()},
        "slot_switch_rate_by_slot": summary.get("slot_id_switch_rate_by_slot", {}),
        "source_files": [str(dataset_dir / "build_summary.json"), str(dataset_dir / "neighbor_context_summary.json")],
    }
    metrics["filtering_mode"] = detect_filtering_mode({**summary, **metrics}, "waymo")
    metrics["diagnostic_source_note"] = diagnostic_source_note(metrics, "waymo")
    cov_switch = slot_coverage_and_switches(dataset_dir, max_rows=max_rows)
    if cov_switch.get("slot_coverage_by_slot"):
        metrics["slot_coverage_metric_source"] = "array_derived"
    else:
        metrics["slot_coverage_metric_source"] = "summary_derived"
    metrics.update(cov_switch)
    metrics.update(summarize_lane_debug_csv(dataset_dir))
    return metrics


def flatten_rejection_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out = Counter()
    for d in rows:
        for by_slot in (d.get("slot_rejection_reason_counts") or {}).values():
            if isinstance(by_slot, dict):
                out.update({str(k): int(v) for k, v in by_slot.items()})
    return dict(out)


def summarize_nuplan(dataset_dir: Path, max_rows: Optional[int]) -> Dict[str, Any]:
    strict = read_json(dataset_dir / "nuplan_laneaware_strict_filter_summary.json")
    if strict:
        strict.setdefault("dataset", "nuplan_stage7_adapter")
        strict.setdefault("path", str(dataset_dir))
        strict.setdefault("source_files", [str(dataset_dir / "nuplan_laneaware_strict_filter_summary.json")])
        strict.setdefault("filtering_mode", "strict_filter_lane_aware_only")
        strict.setdefault("diagnostic_source_note", diagnostic_source_note(strict, "nuplan"))
        return strict
    warnings = read_json(dataset_dir / "warnings.json")
    validation = warnings.get("validation", {}) if isinstance(warnings.get("validation"), dict) else {}
    assign_rows = read_json(dataset_dir / "assignment_debug.json")
    rows = assign_rows if isinstance(assign_rows, list) else []
    projection_debug = read_json(dataset_dir / "nuplan_lane_projection_debug_summary.json")
    metrics = {
        "dataset": "nuplan_stage7_adapter",
        "path": str(dataset_dir),
        "n_rows": len(rows) or validation.get("n_rows"),
        "lane_assignment_available": bool(validation.get("lane_assignment_available", False)),
        "lane_assignment_available_rate": as_float(validation.get("ego_lane_projection_success_rate"), None),
        "fallback_assignment_used_rate": as_float(validation.get("fallback_assignment_used_rate"), None),
        "candidate_projection_success_rate": as_float(validation.get("candidate_lane_projection_success_rate"), None),
        "adjacency_source_counts": validation.get("adjacency_source_counts", warnings.get("adjacency_source_counts", {})),
        "lane_context_quality_counts": dict(Counter(d.get("lane_context_quality", "unknown") for d in rows)) if rows else {},
        "lane_context_quality_reason_counts": dict(Counter(r for d in rows for r in d.get("lane_context_quality_reasons", []))) if rows else {},
        "rejection_reason_counts": flatten_rejection_counts(rows),
        "slot_coverage_by_slot": validation.get("slot_coverage_by_slot", {}),
        "slot_switch_rate_by_slot": validation.get("slot_id_switch_rate_by_slot", {}),
        "source_files": [str(dataset_dir / "warnings.json"), str(dataset_dir / "assignment_debug.json"), str(dataset_dir / "slot_assignment_report.md")],
    }
    metrics["filtering_mode"] = detect_filtering_mode({**warnings, **metrics}, "nuplan")
    metrics["diagnostic_source_note"] = diagnostic_source_note(metrics, "nuplan")
    if projection_debug:
        metrics["projection_debug_summary_available"] = True
        metrics["projection_debug_candidate_projection_success_rate"] = as_float(projection_debug.get("candidate_projection_success_rate"), None)
        if metrics["candidate_projection_success_rate"] is None:
            metrics["candidate_projection_success_rate"] = metrics["projection_debug_candidate_projection_success_rate"]
        metrics["projection_debug_summary"] = projection_debug
    if not metrics["lane_context_quality_counts"]:
        metrics.update(summarize_lane_debug_csv(dataset_dir))
    cov_switch = slot_coverage_and_switches(dataset_dir, max_rows=max_rows)
    if cov_switch["slot_coverage_by_slot"]:
        metrics["slot_coverage_by_slot_from_arrays"] = cov_switch["slot_coverage_by_slot"]
        metrics["slot_coverage_metric_source"] = "array_derived"
    else:
        metrics["slot_coverage_metric_source"] = "summary_derived"
    if cov_switch["slot_switch_rate_by_slot"]:
        metrics["slot_switch_rate_by_slot_from_arrays"] = cov_switch["slot_switch_rate_by_slot"]
    return metrics


def detect_filtering_mode(metrics: Dict[str, Any], dataset: str) -> str:
    explicit = str(metrics.get("filtering_mode") or metrics.get("laneaware_filtering_mode") or "").strip()
    if explicit:
        return explicit
    assignment_mode = str(metrics.get("assignment_mode") or "").strip()
    filters = metrics.get("strict_filters") or metrics.get("drop_filters") or {}
    if assignment_mode == "lane_aware_only" and (
        metrics.get("strict_filter_diagnostic")
        or metrics.get("drop_if_no_lane_map")
        or metrics.get("drop_if_ego_lane_missing")
        or (isinstance(filters, dict) and any(filters.values()))
    ):
        return "strict_filter_lane_aware_only"
    # Do not infer Waymo strict filtering from assignment_mode alone; historical exports
    # need explicit metadata or concrete drop-filter fields for fair comparability.
    if assignment_mode == "lane_aware_with_geometric_fallback":
        return "fallback_preserving"
    return "unknown"


def diagnostic_source_note(metrics: Dict[str, Any], dataset: str) -> str:
    mode = str(metrics.get("filtering_mode") or detect_filtering_mode(metrics, dataset))
    if dataset == "waymo" and mode == "strict_filter_lane_aware_only":
        return "Waymo diagnostic source is strict-filtered lane_aware_only when Stage5 drop_if_* filters are present/detected."
    if dataset == "nuplan" and mode == "strict_filter_lane_aware_only":
        return "nuPlan diagnostic source is the Stage5-style strict-filter diagnostic, not the default official-rollout-preserving output."
    if dataset == "nuplan" and mode == "fallback_preserving":
        return "nuPlan source is lane_aware_with_geometric_fallback and preserves official scenario × planner rollout rows."
    return f"{dataset} filtering mode is {mode}; comparability may be limited."


def diagnose(waymo: Dict[str, Any], nuplan: Dict[str, Any], fallback_gap_threshold: float) -> Dict[str, Any]:
    wf = as_float(waymo.get("fallback_assignment_used_rate"), None)
    nf = as_float(nuplan.get("fallback_assignment_used_rate"), None)
    wc = as_float(waymo.get("candidate_projection_success_rate"), None)
    nc = as_float(nuplan.get("candidate_projection_success_rate"), None)
    fallback_comparable = wf is not None and nf is not None
    projection_comparable = wc is not None and nc is not None
    comparable_metrics_available = fallback_comparable or projection_comparable
    missing_waymo_metrics = {
        "fallback_assignment_used_rate": wf is None,
        "candidate_projection_success_rate": wc is None,
    }
    waymo_mode = str(waymo.get("filtering_mode") or "unknown")
    nuplan_mode = str(nuplan.get("filtering_mode") or "unknown")
    filtering_modes_match = waymo_mode == nuplan_mode and waymo_mode != "unknown"
    filtering_mismatch = waymo_mode != "unknown" and nuplan_mode != "unknown" and waymo_mode != nuplan_mode

    if filtering_mismatch:
        return {
            "verdict": "inconclusive_due_to_filtering_mismatch",
            "reason": "Waymo and nuPlan diagnostics use different filtering philosophies, so fallback=0 on strict-filtered Waymo must not be treated as directly comparable to fallback-preserving nuPlan.",
            "waymo_fallback_rate": wf,
            "nuplan_fallback_rate": nf,
            "waymo_candidate_projection_success_rate": wc,
            "nuplan_candidate_projection_success_rate": nc,
            "fallback_rate_comparable": False,
            "candidate_projection_success_comparable": projection_comparable,
            "waymo_candidate_projection_success_rate_available": wc is not None,
            "nuplan_candidate_projection_success_rate_available": nc is not None,
            "fallback_rates_comparable": False,
            "projection_success_rates_comparable": projection_comparable,
            "waymo_slot_coverage_metric_source": waymo.get("slot_coverage_metric_source", "unknown"),
            "nuplan_slot_coverage_metric_source": nuplan.get("slot_coverage_metric_source", "unknown"),
            "confidence": "downgraded",
            "missing_waymo_metrics": missing_waymo_metrics,
            "waymo_filtering_mode": waymo_mode,
            "nuplan_filtering_mode": nuplan_mode,
            "filtering_modes_match": False,
        }

    if filtering_modes_match and waymo_mode == "strict_filter_lane_aware_only":
        keep_rate = as_float(nuplan.get("kept_row_rate"), None)
        verdict = "comparable_strict_filter_pass"
        reason = "Both sides use strict-filtered lane_aware_only diagnostics; fallback-preserving mismatch is removed."
        confidence = "medium"
        if keep_rate is not None and keep_rate < 0.5:
            verdict = "nuplan_strict_filter_low_keep_rate"
            reason = "Both sides use strict filtering, but nuPlan keeps too few official rollout rows for strong evidence."
            confidence = "low"
        return {
            "verdict": verdict,
            "reason": reason,
            "waymo_fallback_rate": wf,
            "nuplan_fallback_rate": nf,
            "waymo_candidate_projection_success_rate": wc,
            "nuplan_candidate_projection_success_rate": nc,
            "fallback_rate_comparable": True,
            "candidate_projection_success_comparable": projection_comparable,
            "waymo_candidate_projection_success_rate_available": wc is not None,
            "nuplan_candidate_projection_success_rate_available": nc is not None,
            "fallback_rates_comparable": True,
            "projection_success_rates_comparable": projection_comparable,
            "waymo_slot_coverage_metric_source": waymo.get("slot_coverage_metric_source", "unknown"),
            "nuplan_slot_coverage_metric_source": nuplan.get("slot_coverage_metric_source", "unknown"),
            "confidence": confidence,
            "missing_waymo_metrics": missing_waymo_metrics,
            "waymo_filtering_mode": waymo_mode,
            "nuplan_filtering_mode": nuplan_mode,
            "filtering_modes_match": True,
            "nuplan_kept_row_rate": keep_rate,
        }

    if not comparable_metrics_available:
        verdict = "inconclusive_missing_comparable_metrics"
        reason = "Waymo fallback/projection metrics are unavailable, so nuPlan cannot be compared conservatively under the shared Stage5D assignment interface."
        confidence = "inconclusive"
    elif waymo_mode == "unknown":
        verdict = "inconclusive_unknown_waymo_filtering_mode"
        reason = "Waymo filtering_mode is unknown, so fallback/projection metrics have limited comparability with nuPlan diagnostics."
        confidence = "low"
    elif projection_comparable and nc + 0.20 < wc:
        verdict = "nuplan_adapter_or_map_projection_issue"
        reason = "nuPlan candidate projection success is substantially lower than Waymo under the shared Stage5D assignment interface."
        confidence = "high" if fallback_comparable else "medium"
    elif fallback_comparable and nf > wf + fallback_gap_threshold:
        verdict = "nuplan_adapter_or_map_projection_issue"
        reason = "nuPlan geometric fallback rate is substantially higher than Waymo under the same Stage5D assignment interface."
        confidence = "medium" if not projection_comparable else "high"
    elif projection_comparable and fallback_comparable and wc < 0.5 and nc < 0.5 and wf > 0.2 and nf > 0.2:
        verdict = "generic_stage5_lane_aware_limitation_or_dataset_common_issue"
        reason = "Both datasets show low projection success and elevated fallback, suggesting a shared Stage5D limitation or common data/map issue rather than a nuPlan-only adapter issue."
        confidence = "medium"
    else:
        verdict = "generic_stage5_lane_aware_limitation_or_inconclusive" if comparable_metrics_available else "inconclusive_missing_comparable_metrics"
        reason = "nuPlan is not clearly worse than Waymo on comparable fallback/projection metrics, or comparable metrics are incomplete."
        confidence = "low" if not (fallback_comparable and projection_comparable) else "medium"
    return {
        "verdict": verdict,
        "reason": reason,
        "waymo_fallback_rate": wf,
        "nuplan_fallback_rate": nf,
        "waymo_candidate_projection_success_rate": wc,
        "nuplan_candidate_projection_success_rate": nc,
        "fallback_rate_comparable": fallback_comparable,
        "candidate_projection_success_comparable": projection_comparable,
        "waymo_candidate_projection_success_rate_available": wc is not None,
        "nuplan_candidate_projection_success_rate_available": nc is not None,
        "fallback_rates_comparable": fallback_comparable,
        "projection_success_rates_comparable": projection_comparable,
        "waymo_slot_coverage_metric_source": waymo.get("slot_coverage_metric_source", "unknown"),
        "nuplan_slot_coverage_metric_source": nuplan.get("slot_coverage_metric_source", "unknown"),
        "confidence": confidence,
        "missing_waymo_metrics": missing_waymo_metrics,
        "waymo_filtering_mode": waymo_mode,
        "nuplan_filtering_mode": nuplan_mode,
        "filtering_modes_match": filtering_modes_match,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    diagnosis = payload["diagnosis"]
    lines = [
        "# Stage5D Lane-Aware Assignment Cross-Dataset Diagnostics",
        "",
        "## Architecture constraint",
        "",
        "- Stage5D CORE / `tools.lane_aware_assignment.py` remains the only lane-aware assignment implementation.",
        "- Stage7 nuPlan is treated only as an adapter that emits Stage5-compatible `LaneInfo` and candidate states.",
        "",
        "## Diagnosis",
        "",
        f"- verdict: `{diagnosis['verdict']}`",
        f"- reason: {diagnosis['reason']}",
        f"- fallback_rate_comparable: `{diagnosis.get('fallback_rate_comparable')}`",
        f"- candidate_projection_success_comparable: `{diagnosis.get('candidate_projection_success_comparable')}`",
        f"- confidence: `{diagnosis.get('confidence')}`",
        f"- Waymo filtering mode: `{diagnosis.get('waymo_filtering_mode')}`",
        f"- nuPlan filtering mode: `{diagnosis.get('nuplan_filtering_mode')}`",
        f"- filtering modes match: `{diagnosis.get('filtering_modes_match')}`",
        f"- Waymo candidate_projection_success_rate available: `{diagnosis.get('waymo_candidate_projection_success_rate_available')}`",
        f"- nuPlan candidate_projection_success_rate available: `{diagnosis.get('nuplan_candidate_projection_success_rate_available')}`",
        f"- fallback rates comparable: `{diagnosis.get('fallback_rates_comparable')}`",
        f"- projection success rates comparable: `{diagnosis.get('projection_success_rates_comparable')}`",
        f"- Waymo slot coverage metric source: `{diagnosis.get('waymo_slot_coverage_metric_source')}`",
        f"- nuPlan slot coverage metric source: `{diagnosis.get('nuplan_slot_coverage_metric_source')}`",
        f"- missing_waymo_metrics: `{diagnosis.get('missing_waymo_metrics')}`",
        "",
        "## Comparable metrics",
    ]
    for name in ("waymo", "nuplan"):
        m = payload[name]
        lines += ["", f"### {name}", "", f"- path: `{m.get('path')}`", f"- filtering_mode: `{m.get('filtering_mode')}`", f"- diagnostic_source_note: {m.get('diagnostic_source_note')}", f"- lane_assignment_available: `{m.get('lane_assignment_available')}`", f"- lane_assignment_available_rate: `{m.get('lane_assignment_available_rate')}`", f"- fallback_assignment_used_rate: `{m.get('fallback_assignment_used_rate')}`", f"- candidate_projection_success_rate: `{m.get('candidate_projection_success_rate')}`", f"- adjacency_source_counts: `{m.get('adjacency_source_counts')}`", f"- lane_context_quality counts: `{m.get('lane_context_quality_counts')}`", f"- rejection reason counts: `{m.get('rejection_reason_counts')}`", f"- slot coverage by slot: `{m.get('slot_coverage_by_slot')}`", f"- slot switch rate by slot: `{m.get('slot_switch_rate_by_slot')}`"]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare shared Stage5D lane-aware assignment diagnostics between Waymo Stage5 output and nuPlan Stage7 adapter output.")
    p.add_argument("--waymo_dir", type=Path, required=True, help="Waymo Stage5/Stage5D output directory containing build_summary.json and/or shards.")
    p.add_argument("--nuplan_dir", type=Path, required=True, help="nuPlan Stage7E/Stage5D-compatible output directory containing warnings.json and optional assignment_debug.json.")
    p.add_argument("--out_dir", type=Path, required=True, help="Directory for lane_aware_diagnostic_comparison.json/.md.")
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap when scanning neighbor arrays for coverage/switch diagnostics.")
    p.add_argument("--fallback_gap_threshold", type=float, default=0.20, help="Minimum nuPlan-minus-Waymo fallback-rate gap used for adapter-issue diagnosis.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.waymo_dir.exists():
        raise FileNotFoundError(f"Waymo output directory does not exist: {args.waymo_dir}")
    if not args.nuplan_dir.exists():
        raise FileNotFoundError(f"nuPlan output directory does not exist: {args.nuplan_dir}")
    waymo = summarize_waymo(args.waymo_dir, args.max_rows)
    nuplan = summarize_nuplan(args.nuplan_dir, args.max_rows)
    payload = {"waymo": waymo, "nuplan": nuplan, "diagnosis": diagnose(waymo, nuplan, args.fallback_gap_threshold)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "lane_aware_diagnostic_comparison.json", payload)
    (args.out_dir / "lane_aware_diagnostic_comparison.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote diagnostics: {args.out_dir / 'lane_aware_diagnostic_comparison.json'}")
    print(f"Wrote report: {args.out_dir / 'lane_aware_diagnostic_comparison.md'}")


if __name__ == "__main__":
    main()
