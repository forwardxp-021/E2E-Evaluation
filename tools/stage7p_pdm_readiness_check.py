#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".cfg", ".ini"}
KNOWN_PLANNER_CONFIGS = ["simple_planner", "idm_planner", "log_future_planner", "ml_planner"]
STAGE7C_MODULES = [
    "nuplan.planning.simulation.planner.abstract_planner",
    "nuplan.planning.simulation.planner.simple_planner",
    "nuplan.planning.simulation.planner.log_future_planner",
    "nuplan.planning.simulation.planner.idm_planner",
    "nuplan.planning.script.run_simulation",
]
IMPORT_CANDIDATES = [
    "nuplan",
    "nuplan.planning.simulation.planner",
    "nuplan.planning.simulation.planner.pdm_planner",
    "nuplan.planning.simulation.planner.pdm_planner.pdm_planner",
    "nuplan.planning.simulation.planner.pdm_planner.pdm_closed_planner",
    "nuplan.planning.script.config.common.planner.pdm_planner",
]
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*PDM[A-Za-z0-9_]*|PDM[A-Za-z0-9_]*)\b", re.MULTILINE)


def path_to_str(path: Path) -> str:
    return str(path.expanduser().resolve())


def safe_read_text(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def scan_root(root: Path) -> Dict[str, Any]:
    files = list(iter_files(root)) if root.exists() else []
    pdm_file_candidates: List[str] = []
    pdm_content_candidates: List[str] = []
    pdm_class_candidates: List[Dict[str, str]] = []
    planner_config_candidates: List[str] = []
    known_configs: Dict[str, List[str]] = {name: [] for name in KNOWN_PLANNER_CONFIGS}

    config_root_token = str(Path("nuplan/planning/script/config"))
    for path in files:
        lower_name = path.name.lower()
        lower_path = str(path).lower()
        if "pdm" in lower_name:
            pdm_file_candidates.append(str(path))
        is_text = path.suffix.lower() in TEXT_SUFFIXES or path.name.lower() in {"hydra.yaml", "config.yaml"}
        text = safe_read_text(path) if is_text else ""
        if text and "pdm" in text.lower():
            pdm_content_candidates.append(str(path))
        if path.suffix.lower() == ".py" and text:
            for klass in CLASS_RE.findall(text):
                pdm_class_candidates.append({"class_name": klass, "path": str(path)})
        if config_root_token in lower_path.replace("\\", "/") and path.suffix.lower() in {".yaml", ".yml"}:
            if "planner" in lower_path or "planner" in lower_name:
                planner_config_candidates.append(str(path))
            for name in KNOWN_PLANNER_CONFIGS:
                if name in lower_path or name in text.lower():
                    known_configs[name].append(str(path))

    return {
        "root": str(root),
        "exists": root.exists(),
        "pdm_file_candidates": sorted(set(pdm_file_candidates)),
        "pdm_content_candidates": sorted(set(pdm_content_candidates)),
        "pdm_class_candidates": pdm_class_candidates,
        "planner_config_candidates": sorted(set(planner_config_candidates)),
        "known_planner_configs": {k: sorted(set(v)) for k, v in known_configs.items()},
    }


def import_status(module: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"module": module, "spec_exists": False, "imported": False, "error": ""}
    try:
        spec_exists = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["spec_exists"] = spec_exists
    if not spec_exists:
        return out
    try:
        imported = importlib.import_module(module)
        out["imported"] = True
        out["file"] = getattr(imported, "__file__", "") or ""
        pdm_attrs = [name for name in dir(imported) if "pdm" in name.lower()]
        if pdm_attrs:
            out["pdm_attrs"] = pdm_attrs
    except Exception as exc:  # safe diagnostic: report but do not crash
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def flatten_unique(scans: List[Dict[str, Any]], key: str) -> List[Any]:
    seen: Set[str] = set()
    out: List[Any] = []
    for scan in scans:
        for item in scan.get(key, []):
            marker = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, dict) else str(item)
            if marker not in seen:
                seen.add(marker)
                out.append(item)
    return out


def write_report(summary: Dict[str, Any], path: Path) -> None:
    status = "PASS: PDM appears available" if summary["pdm_available"] else "PENDING: PDM is not available yet"
    lines = [
        "# Stage7P PDM Readiness Report",
        "",
        f"## Status\n{status}",
        "",
        "## Required next action",
        summary["required_next_action"],
        "",
        "## PDM availability",
        f"- pdm_available: `{str(summary['pdm_available']).lower()}`",
        f"- pdm_config_candidates: `{len(summary['pdm_config_candidates'])}`",
        f"- pdm_module_candidates: `{len(summary['pdm_module_candidates'])}`",
        f"- pdm_class_candidates: `{len(summary['pdm_class_candidates'])}`",
        "",
        "## Safety notes",
        "- 本脚本只做只读发现，不安装包、不 clone 外部仓库、不修改环境。",
        "- 在 `pdm_available=true` 之前，不要运行 PDM smoke template，也不要假设 `planner=pdm_planner` 可用。",
        "",
        "## Search roots",
    ]
    for root in summary["search_roots"]:
        lines.append(f"- `{root}`")
    lines.extend(["", "## PDM config candidates"])
    lines.extend([f"- `{p}`" for p in summary["pdm_config_candidates"]] or ["- none"])
    lines.extend(["", "## PDM module candidates"])
    lines.extend([f"- `{m.get('module')}` imported={m.get('imported')} spec_exists={m.get('spec_exists')} error={m.get('error', '')}" for m in summary["pdm_module_candidates"]] or ["- none"])
    lines.extend(["", "## PDM class candidates"])
    lines.extend([f"- `{c.get('class_name')}` in `{c.get('path')}`" for c in summary["pdm_class_candidates"]] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir exists and is not empty: {out_dir}. Use --overwrite.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(args.nuplan_devkit_root), Path(args.repo_root)] + [Path(p) for p in args.extra_search_roots]
    scans = [scan_root(root.expanduser().resolve()) for root in roots]
    imports = [import_status(m) for m in STAGE7C_MODULES]
    candidate_imports = [import_status(m) for m in IMPORT_CANDIDATES]
    pdm_module_candidates = [m for m in candidate_imports if ("pdm" in m["module"].lower() and m["spec_exists"]) or m.get("pdm_attrs")]
    pdm_config_candidates = sorted(set(flatten_unique(scans, "pdm_file_candidates") + flatten_unique(scans, "pdm_content_candidates")))
    pdm_class_candidates = flatten_unique(scans, "pdm_class_candidates")
    pdm_available = bool(pdm_config_candidates and (pdm_module_candidates or pdm_class_candidates))
    required_next_action = "ready_for_pdm_smoke" if pdm_available else ("configure_external_planner_path" if pdm_config_candidates or pdm_module_candidates else "install_external_pdm_implementation")
    known: Dict[str, List[str]] = {name: [] for name in KNOWN_PLANNER_CONFIGS}
    for scan in scans:
        for name, paths in scan["known_planner_configs"].items():
            known[name].extend(paths)
    summary = {
        "pdm_available": pdm_available,
        "required_next_action": required_next_action,
        "repo_root": path_to_str(Path(args.repo_root)),
        "nuplan_devkit_root": path_to_str(Path(args.nuplan_devkit_root)),
        "search_roots": [path_to_str(p) for p in roots],
        "pdm_config_candidates": pdm_config_candidates,
        "pdm_module_candidates": pdm_module_candidates,
        "pdm_class_candidates": pdm_class_candidates,
        "available_planner_configs": {k: sorted(set(v)) for k, v in known.items() if v},
        "stage7c_import_discovery": imports,
        "root_scans": scans,
        "environment_modified": False,
    }
    (out_dir / "pdm_readiness_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, out_dir / "pdm_readiness_report.md")
    print(json.dumps({"pdm_available": pdm_available, "required_next_action": required_next_action, "output_dir": str(out_dir)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage7P read-only PDM planner readiness and discovery check.")
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--nuplan_devkit_root", required=True)
    parser.add_argument("--output_dir", default="outputs/stage7p_pdm_readiness_check_v1")
    parser.add_argument("--extra_search_roots", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
