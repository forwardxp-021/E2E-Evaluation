import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import tools.stage7c1_run_nuplan_simulation as stage7c


def test_unknown_planner_without_allow_external_fails_clearly(tmp_path, monkeypatch):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "merged_metadata.csv").write_text(
        "scenario_id,db_name,scene_token,sample_id\nscenario,log.db,scene,sample\n",
        encoding="utf-8",
    )
    db_root = tmp_path / "db"
    map_root = tmp_path / "maps"
    db_root.mkdir()
    map_root.mkdir()
    monkeypatch.setattr(stage7c, "discover_modules", lambda: {"nuplan.planning.script.run_simulation": {"available": True}})
    args = stage7c.parse_args([
        "--context_dir", str(context_dir),
        "--nuplan_db_root", str(db_root),
        "--nuplan_map_root", str(map_root),
        "--output_dir", str(tmp_path / "out"),
        "--planners", "external_demo_planner",
        "--nuplan_simulation_command_template", "python -c pass {planner_hydra_overrides}",
        "--overwrite",
    ])

    rc = stage7c.run(args)

    assert rc == 2
    warnings = json.loads((tmp_path / "out" / "warnings.json").read_text(encoding="utf-8"))["warnings"]
    assert any(w["type"] == "unknown_planner" and "Use --allow_external_planner_name" in w["message"] for w in warnings)


def test_unknown_planner_with_allow_external_uses_planner_override(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, shell, text, stdout, stderr, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(stage7c.subprocess, "run", fake_run)
    planner_name = "external_demo_planner_for_test"
    stage7c.PLANNER_PROFILES.pop(planner_name, None)
    warnings = []

    ok, log_path, return_code = stage7c.run_official_nuplan_cli(
        "python -m dummy {planner_hydra_overrides}",
        planner_name,
        {"scenario_index": "0"},
        tmp_path,
        10,
        warnings,
    )

    assert ok is True
    assert return_code == 0
    assert calls[0][-1] == f"planner={planner_name}"
    assert f"planner={planner_name}" in Path(log_path).read_text(encoding="utf-8")


def test_hydra_searchpath_is_appended_to_official_command(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, shell, text, stdout, stderr, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(stage7c.subprocess, "run", fake_run)
    searchpath = "[pkg://tuplan_garage.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
    warnings = []

    ok, log_path, _ = stage7c.run_official_nuplan_cli(
        "python -m dummy {planner_hydra_overrides}",
        "simple_planner",
        {"scenario_index": "0"},
        tmp_path,
        10,
        warnings,
        hydra_searchpath=searchpath,
    )

    assert ok is True
    command_text = " ".join(calls[0])
    assert "planner=simple_planner" in command_text
    assert f"hydra.searchpath='{searchpath}'" in Path(log_path).read_text(encoding="utf-8")


def test_existing_simple_and_idm_planner_overrides_remain_unchanged():
    assert stage7c.format_planner_hydra_overrides("simple_planner") == "planner=simple_planner"
    idm = stage7c.format_planner_hydra_overrides("idm_longitudinal_comfort")
    assert idm.startswith("planner=idm_planner ")
    assert "planner.idm_planner.target_velocity=10.0" in idm
    assert "hydra.searchpath" not in idm


def test_quick_reference_contains_pdm_closed_smoke_command():
    text = Path("QUICK_REFERENCE.md").read_text(encoding="utf-8")
    assert "Stage7P — PDM closed planner smoke" in text
    assert "--planners pdm_closed_planner" in text
    assert "--allow_external_planner_name" in text
    assert "--hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common" in text
