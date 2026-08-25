import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "stage7p_pdm_readiness_check.py"
QUICK_REFERENCE = ROOT / "QUICK_REFERENCE.md"
PROTECTED = [
    ROOT / "tools" / "stage5d_context_core.py",
    ROOT / "tools" / "lane_aware_assignment.py",
    ROOT / "tools" / "stage6_compare_unpaired_style.py",
    ROOT / "tools" / "stage6_generate_report_card.py",
]


def run_tool(repo_root: Path, nuplan_root: Path, out: Path):
    return subprocess.run(
        [sys.executable, str(TOOL), "--repo_root", str(repo_root), "--nuplan_devkit_root", str(nuplan_root), "--output_dir", str(out), "--overwrite"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_readiness_false_on_fake_repo_without_pdm(tmp_path):
    repo = tmp_path / "repo"
    nuplan = tmp_path / "nuplan-devkit"
    repo.mkdir()
    (nuplan / "nuplan" / "planning" / "script" / "config" / "planner").mkdir(parents=True)
    (nuplan / "nuplan" / "planning" / "script" / "config" / "planner" / "idm_planner.yaml").write_text("planner: idm_planner\n", encoding="utf-8")
    proc = run_tool(repo, nuplan, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tmp_path / "out" / "pdm_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["pdm_available"] is False
    assert summary["required_next_action"] == "install_external_pdm_implementation"


def test_readiness_detects_fake_pdm_config_and_outputs(tmp_path):
    repo = tmp_path / "repo"
    nuplan = tmp_path / "nuplan-devkit"
    repo.mkdir()
    planner_dir = nuplan / "nuplan" / "planning" / "script" / "config" / "planner"
    planner_dir.mkdir(parents=True)
    (planner_dir / "pdm_planner.yaml").write_text("planner: pdm_planner\n_target_: fake.PDMPlanner\n", encoding="utf-8")
    src = nuplan / "fake_pdm.py"
    src.write_text("class PDMPlanner:\n    pass\n", encoding="utf-8")
    proc = run_tool(repo, nuplan, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    summary_path = tmp_path / "out" / "pdm_readiness_summary.json"
    report_path = tmp_path / "out" / "pdm_readiness_report.md"
    assert summary_path.is_file()
    assert report_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert any("pdm_planner.yaml" in p for p in summary["pdm_config_candidates"])
    assert any(c["class_name"] == "PDMPlanner" for c in summary["pdm_class_candidates"])


def test_quick_reference_contains_pdm_readiness_command():
    text = QUICK_REFERENCE.read_text(encoding="utf-8")
    assert "Stage7P — PDM readiness and smoke preparation" in text
    assert "python tools/stage7p_pdm_readiness_check.py" in text
    assert "--nuplan_devkit_root /home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit" in text


def test_protected_stage5d_lane_stage6_files_unmodified_by_stage7p_patch():
    stage7p = TOOL.read_text(encoding="utf-8")
    assert "stage5d_context_core" not in stage7p
    assert "lane_aware_assignment" not in stage7p
    assert "def compute_mmd" not in stage7p
    assert "def mmd" not in stage7p
    for path in PROTECTED:
        assert path.is_file()
