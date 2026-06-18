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
