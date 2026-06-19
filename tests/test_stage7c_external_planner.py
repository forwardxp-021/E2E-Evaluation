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


def test_expandvars_expands_nuplan_devkit_root_in_command_template(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, shell, text, stdout, stderr, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setenv("NUPLAN_DEVKIT_ROOT", "/abs/nuplan-devkit")
    monkeypatch.setattr(stage7c.subprocess, "run", fake_run)
    warnings = []
    ok, log_path, _ = stage7c.run_official_nuplan_cli(
        "python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py {scenario_hydra_overrides}",
        "simple_planner",
        {"scenario_index": "0", "scenario_token": "tok"},
        tmp_path,
        10,
        warnings,
        require_same_scenario_alignment=True,
    )
    assert ok is True
    assert calls[0][1] == "/abs/nuplan-devkit/nuplan/planning/script/run_simulation.py"
    assert "$NUPLAN_DEVKIT_ROOT" not in Path(log_path).read_text(encoding="utf-8")


def test_scenario_hydra_overrides_prefers_token():
    info = stage7c.scenario_hydra_override_info({"scenario_token": "abc", "db_name": "log.db"}, True)
    assert info["control_mode"] == "token"
    assert info["scenario_hydra_overrides"] == "scenario_filter.scenario_tokens=[abc]"


def test_scenario_hydra_overrides_falls_back_to_log_name():
    info = stage7c.scenario_hydra_override_info({"db_name": "2021.01.01_veh-1.db"}, True)
    assert info["control_mode"] == "log_name"
    assert "scenario_filter.log_names=[2021.01.01_veh-1]" in info["scenario_hydra_overrides"]
    assert "scenario_filter.limit_total_scenarios=1" in info["scenario_hydra_overrides"]


def test_missing_scenario_hydra_placeholder_with_required_alignment_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="Command template must include"):
        stage7c.run_official_nuplan_cli(
            "python -m dummy",
            "simple_planner",
            {"scenario_token": "abc"},
            tmp_path,
            10,
            [],
            require_same_scenario_alignment=True,
        )


def test_pdm_external_planner_command_generation_with_required_alignment(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, shell, text, stdout, stderr, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(stage7c.subprocess, "run", fake_run)
    ok, _, _ = stage7c.run_official_nuplan_cli(
        "python -m dummy {planner_hydra_overrides} {scenario_hydra_overrides}",
        "pdm_closed_planner",
        {"scenario_index": "0", "scenario_token": "abc"},
        tmp_path,
        10,
        [],
        require_same_scenario_alignment=True,
    )
    assert ok is True
    joined = " ".join(calls[0])
    assert "planner=pdm_closed_planner" in joined
    assert "scenario_filter.scenario_tokens=[abc]" in joined


def test_quick_reference_contains_fixed_pdm_commands():
    text = Path("QUICK_REFERENCE.md").read_text(encoding="utf-8")
    assert "{scenario_hydra_overrides}" in text
    assert "stage7p_pdm_config_parameter_report.py" in text


def test_pdm_closed_conservative_variant_hydra_overrides_and_metadata():
    overrides = stage7c.format_planner_hydra_overrides("pdm_closed_conservative_v1")
    assert overrides.startswith("planner=pdm_closed_planner ")
    assert "planner.pdm_closed_planner.idm_policies.speed_limit_fraction=[0.2,0.4,0.6,0.8]" in overrides
    assert "planner.pdm_closed_planner.idm_policies.fallback_target_velocity=10.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.min_gap_to_lead_agent=2.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.headway_time=2.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.accel_max=1.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.decel_max=3.0" in overrides
    assert "planner.pdm_closed_planner.lateral_offsets=[-0.5,0.5]" in overrides
    profile = stage7c.PLANNER_PROFILES["pdm_closed_conservative_v1"]
    assert profile["planner_type"] == "pdm_closed_variant"
    assert profile["policy_style"] == "conservative"
    assert profile["style_scope"] == "full_closed_loop_planner"
    assert profile["nuplan_planner_config"] == "pdm_closed_planner"
    assert profile["parameters"]["source"] == "tuplan_garage"
    assert profile["parameters"]["checkpoint_required"] is False


def test_pdm_closed_assertive_variant_hydra_overrides_and_metadata():
    overrides = stage7c.format_planner_hydra_overrides("pdm_closed_assertive_v1")
    assert overrides.startswith("planner=pdm_closed_planner ")
    assert "planner.pdm_closed_planner.idm_policies.speed_limit_fraction=[0.4,0.6,0.8,1.0]" in overrides
    assert "planner.pdm_closed_planner.idm_policies.fallback_target_velocity=18.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.min_gap_to_lead_agent=0.5" in overrides
    assert "planner.pdm_closed_planner.idm_policies.headway_time=1.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.accel_max=2.0" in overrides
    assert "planner.pdm_closed_planner.idm_policies.decel_max=3.5" in overrides
    assert "planner.pdm_closed_planner.lateral_offsets=[-1.5,1.5]" in overrides
    profile = stage7c.PLANNER_PROFILES["pdm_closed_assertive_v1"]
    assert profile["planner_type"] == "pdm_closed_variant"
    assert profile["policy_style"] == "assertive"
    assert profile["style_scope"] == "full_closed_loop_planner"
    assert profile["nuplan_planner_config"] == "pdm_closed_planner"


def test_pdm_closed_variant_requested_label_and_safe_slug_are_preserved():
    replacements = stage7c.build_command_replacements(
        "pdm_closed_assertive_v1",
        {"scenario_index": "0", "scenario_token": "abc"},
        Path("outputs/demo"),
        require_same_scenario_alignment=True,
    )
    assert replacements["planner_name"] == "pdm_closed_assertive_v1"
    assert replacements["planner_name_safe"] == "pdm_closed_assertive_v1"
    assert replacements["planner_hydra_overrides"].startswith("planner=pdm_closed_planner ")
    assert "planner=pdm_closed_assertive_v1" not in replacements["planner_hydra_overrides"]


def test_existing_pdm_closed_planner_default_still_uses_base_config_only():
    assert stage7c.format_planner_hydra_overrides("pdm_closed_planner") == "planner=pdm_closed_planner"
    assert stage7c.format_planner_hydra_overrides("pdm_closed_default") == "planner=pdm_closed_planner"
