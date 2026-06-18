#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
CLASS_NAMES = ["PDMClosedPlanner", "AbstractPDMClosedPlanner", "AbstractPDMPlanner", "PDMGenerator", "PDMProposal", "PDMProposalManager", "PDMSimulator", "PDMScorer"]
GROUP_RULES = [
    ("route/path following", ["route", "path", "centerline", "lane", "map"]),
    ("longitudinal / speed / progress", ["speed", "velocity", "accel", "decel", "progress", "idm", "headway"]),
    ("lateral / offset / lane-change-like behavior", ["lateral", "offset", "lane_change", "steer", "yaw"]),
    ("proposal generation / trajectory sampling", ["proposal", "sample", "trajectory", "horizon", "interval"]),
    ("scoring / weights", ["score", "weight", "cost"]),
    ("comfort", ["comfort", "jerk"]),
    ("collision / safety / emergency brake", ["collision", "safety", "brake", "emergency", "ttc"]),
    ("simulator / dynamics", ["simulator", "simulation", "dynamics", "vehicle"]),
]


def parse_scalar(text: str) -> Any:
    raw = text.strip().strip('"\'')
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if NUM_RE.match(raw):
        return float(raw) if any(c in raw for c in ".eE") else int(raw)
    if raw.startswith("[") and raw.endswith("]"):
        vals = [parse_scalar(v.strip()) for v in raw[1:-1].split(",") if v.strip()]
        return vals
    return raw


def parse_simple_yaml(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full_key = ".".join([item[1] for item in stack] + [key.strip()])
        value = value.strip()
        if value == "":
            stack.append((indent, key.strip()))
            continue
        parsed = parse_scalar(value)
        kind = "unknown"
        if key.strip() == "_target_" or full_key.endswith("._target_"):
            kind = "target_path"
        elif isinstance(parsed, bool):
            kind = "boolean"
        elif isinstance(parsed, (int, float)):
            kind = "numeric_scalar"
        elif isinstance(parsed, list) and parsed and all(isinstance(v, (int, float)) for v in parsed):
            kind = "numeric_list"
        rows.append({"source": "yaml", "path": str(path), "line": lineno, "name": full_key, "value": parsed, "kind": kind, "group": classify(full_key)})
    return rows


def classify(name: str) -> str:
    lower = name.lower()
    for group, keys in GROUP_RULES:
        if any(k in lower for k in keys):
            return group
    return "unknown"


def iter_py_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob("*.py")


def inspect_class_signatures(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    wanted = set(CLASS_NAMES)
    for path in iter_py_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in wanted:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        args = item.args.args
                        defaults = [None] * (len(args) - len(item.args.defaults)) + list(item.args.defaults)
                        for arg, default in zip(args, defaults):
                            if arg.arg == "self":
                                continue
                            try:
                                default_value = ast.literal_eval(default) if default is not None else "<required>"
                            except (ValueError, TypeError):
                                default_value = ast.unparse(default) if default is not None else "<required>"
                            kind = "verified_class_arg"
                            rows.append({"source": "class_signature", "path": str(path), "line": item.lineno, "class": node.name, "name": f"{node.name}.__init__.{arg.arg}", "value": default_value, "kind": kind, "group": classify(arg.arg)})
    return rows


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, config_path: Path) -> None:
    summary = {"planner_config": str(config_path), "parameter_count": len(rows), "groups": {}}
    for row in rows:
        summary["groups"].setdefault(row["group"], 0)
        summary["groups"][row["group"]] += 1
    (out_dir / "pdm_closed_parameter_summary.json").write_text(json.dumps({"summary": summary, "parameters": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["source", "path", "line", "class", "name", "value", "kind", "group", "override_status"]
    with (out_dir / "pdm_closed_parameter_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = {**row, "value": json.dumps(row.get("value"), ensure_ascii=False), "override_status": "verified_config_key" if row["source"] == "yaml" else "verified_class_arg"}
            writer.writerow(row)
    lines = ["# Stage7P PDM Closed Parameter Report", "", f"- config: `{config_path}`", f"- parameters: `{len(rows)}`", "", "## Parameters"]
    for row in rows:
        lines.append(f"- `{row['name']}` = `{row.get('value')}` ({row['kind']}, {row['group']}, source={row['source']})")
    (out_dir / "pdm_closed_parameter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    blueprint = ["# PDM Closed Variant Blueprint", "", "These override groups are candidates only; do not treat them as final runnable configs until names are verified in local Hydra.", ""]
    for name in ["pdm_closed_default", "pdm_closed_conservative_candidate", "pdm_closed_assertive_candidate", "pdm_closed_comfort_candidate"]:
        blueprint.extend([f"## {name}", "- status: inferred_candidate", "- safe overrides: no safe override yet unless listed below as verified_config_key / verified_class_arg."])
        matches = [r for r in rows if r["group"] != "unknown"][:20]
        for row in matches:
            status = "verified_config_key" if row["source"] == "yaml" else "verified_class_arg"
            blueprint.append(f"- `{row['name']}`: {status}")
        if not matches:
            blueprint.append("- no safe override yet")
        blueprint.append("")
    (out_dir / "pdm_closed_variant_blueprint.md").write_text("\n".join(blueprint), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(args.tuplan_garage_root).expanduser().resolve()
    config_path = root / "tuplan_garage" / "planning" / "script" / "config" / "simulation" / "planner" / f"{args.planner_config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing PDM planner config: {config_path}")
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir exists and is not empty: {out_dir}. Use --overwrite.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_simple_yaml(config_path) + inspect_class_signatures(root / "tuplan_garage")
    write_outputs(rows, out_dir, config_path)
    print(json.dumps({"output_dir": str(out_dir), "parameter_count": len(rows)}, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage7P PDM closed planner config parameter discovery report.")
    parser.add_argument("--tuplan_garage_root", required=True)
    parser.add_argument("--planner_config_name", default="pdm_closed_planner")
    parser.add_argument("--output_dir", default="outputs/stage7p_pdm_closed_config_params_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
