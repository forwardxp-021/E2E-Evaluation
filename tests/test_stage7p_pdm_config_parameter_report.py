import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.stage7p_pdm_config_parameter_report as report


def make_fake_tuplan(root: Path) -> Path:
    cfg = root / "tuplan_garage" / "planning" / "script" / "config" / "simulation" / "planner"
    cfg.mkdir(parents=True)
    (cfg / "pdm_closed_planner.yaml").write_text(
        """
_target_: tuplan_garage.planning.simulation.planner.pdm_planner.PDMClosedPlanner
speed_limit_fraction: 0.8
proposal:
  lateral_offsets: [-1.0, 0.0, 1.0]
  emergency_brake: true
scorer:
  progress_weight: 5.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    mod = root / "tuplan_garage" / "planning" / "simulation" / "planner" / "pdm_planner"
    mod.mkdir(parents=True)
    (mod / "pdm_closed_planner.py").write_text(
        "class PDMClosedPlanner:\n    def __init__(self, speed_limit_fraction=1.0, comfort_weight=2.0):\n        pass\n",
        encoding="utf-8",
    )
    return root


def test_pdm_parameter_parser_detects_fake_numeric_keys_and_writes_outputs(tmp_path):
    root = make_fake_tuplan(tmp_path / "tuplan_garage_root")
    out = tmp_path / "out"
    class Args:
        tuplan_garage_root = str(root)
        planner_config_name = "pdm_closed_planner"
        output_dir = str(out)
        overwrite = True
    assert report.run(Args) == 0
    for name in ["pdm_closed_parameter_report.md", "pdm_closed_parameter_summary.json", "pdm_closed_parameter_table.csv", "pdm_closed_variant_blueprint.md"]:
        assert (out / name).is_file()
    summary = json.loads((out / "pdm_closed_parameter_summary.json").read_text(encoding="utf-8"))
    names = {p["name"] for p in summary["parameters"]}
    assert "speed_limit_fraction" in names
    assert "proposal.lateral_offsets" in names
    assert any(p["kind"] == "numeric_scalar" for p in summary["parameters"])


def test_pdm_parser_strips_inline_comments_and_classifies_values(tmp_path):
    cfg = tmp_path / "pdm_closed_planner.yaml"
    cfg.write_text(
        """
trajectory_sampling:
  num_poses: 80  # sample count
  interval_length: 0.1  # seconds
proposal_sampling:
  num_poses: 40 # proposals
speed_limit_fraction: [0.2, 0.4, 0.6, 0.8, 1.0]  # fractions
fallback_target_velocity: 15.0  # mps
min_gap_to_lead_agent: 1.0
headway_time: 1.5
accel_max: 1.5
decel_max: 3.0
lateral_offsets: [-1.0, 1.0] # meters
map_radius: 50 # meters
use_idm: true # bool
""".strip()
        + "\n",
        encoding="utf-8",
    )
    rows = report.parse_simple_yaml(cfg)
    by_name = {r["name"]: r for r in rows}
    assert by_name["trajectory_sampling.num_poses"]["value"] == 80
    assert by_name["trajectory_sampling.interval_length"]["value"] == 0.1
    assert by_name["speed_limit_fraction"]["value"] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert by_name["fallback_target_velocity"]["value"] == 15.0
    assert by_name["min_gap_to_lead_agent"]["value"] == 1.0
    assert by_name["headway_time"]["value"] == 1.5
    assert by_name["accel_max"]["kind"] == "numeric_scalar"
    assert by_name["lateral_offsets"]["kind"] == "numeric_list"
    assert by_name["speed_limit_fraction"]["kind"] == "numeric_list"
    assert by_name["map_radius"]["kind"] == "numeric_scalar"
    assert by_name["use_idm"]["kind"] == "bool"
    assert by_name["lateral_offsets"]["group"] == "lateral / offset / lane-change-like behavior"
    assert by_name["map_radius"]["group"] == "route/path following"


def test_variant_blueprint_concrete_overrides_are_verified_config_keys(tmp_path):
    root = make_fake_tuplan(tmp_path / "tuplan_garage_root")
    cfg = root / "tuplan_garage" / "planning" / "script" / "config" / "simulation" / "planner" / "pdm_closed_planner.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8")
        + "fallback_target_velocity: 15.0 # clean\nmin_gap_to_lead_agent: 1.0\nheadway_time: 1.5\naccel_max: 1.5\ndecel_max: 3.0\nlateral_offsets: [-1.0, 1.0] # clean\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    class Args:
        tuplan_garage_root = str(root)
        planner_config_name = "pdm_closed_planner"
        output_dir = str(out)
        overwrite = True
    report.run(Args)
    blueprint = (out / "pdm_closed_variant_blueprint.md").read_text(encoding="utf-8")
    assert "verified_config_key" in blueprint
    assert "unsafe_unknown" in blueprint
    concrete_lines = [line for line in blueprint.splitlines() if line.strip().startswith("- `+planner.")]
    assert concrete_lines
    assert all("verified_config_key" in line for line in concrete_lines)
