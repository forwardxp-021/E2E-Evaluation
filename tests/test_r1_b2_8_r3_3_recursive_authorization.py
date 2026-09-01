import hashlib
import json
from pathlib import Path

import pytest

from tools.r1_b2_8_r3_3_execute_frozen_48run_smoke import authorize


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "docs/stageR/r1/r1_b2_8_r3_3_final_execution_binding_manifest_v1.2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    payload = json.loads(REAL.read_text())
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps(payload))
    owner = tmp_path / "owner.json"; owner.write_text(json.dumps({"OFFICIAL_SMOKE_AUTHORIZED": True, "final_execution_manifest_sha256": _sha(manifest)}))
    return manifest, owner, payload


def test_current_recursive_authorization_chain_passes(tmp_path: Path) -> None:
    manifest, owner, _ = _fixture(tmp_path)
    assert authorize(manifest, owner) == 34


def test_owner_manifest_sha_mismatch_fails(tmp_path: Path) -> None:
    manifest, owner, _ = _fixture(tmp_path)
    owner.write_text(json.dumps({"OFFICIAL_SMOKE_AUTHORIZED": True, "final_execution_manifest_sha256": "wrong"}))
    with pytest.raises(PermissionError, match="OWNER_AUTHORIZATION_INVALID"):
        authorize(manifest, owner)


@pytest.mark.parametrize("mutation,pattern", [
    ("wrong_inherited_sha", "INHERITED_MANIFEST_SHA_MISMATCH"),
    ("missing_inherited", "INHERITED_MANIFEST_SHA_MISMATCH"),
    ("wrong_inherited_component", "INHERITED_RUNTIME_COMPONENT_SHA_MISMATCH"),
    ("missing_inherited_component", "INHERITED_RUNTIME_COMPONENT_SHA_MISMATCH"),
    ("wrong_current_component", "EXECUTION_SHA_CLOSURE_MISMATCH"),
    ("wrong_roster", "FROZEN_ARTIFACT_SHA_MISMATCH"),
    ("wrong_schedule", "FROZEN_ARTIFACT_SHA_MISMATCH"),
    ("wrong_pair", "FROZEN_ARTIFACT_SHA_MISMATCH"),
])
def test_recursive_authorization_negative_cases(tmp_path: Path, mutation: str, pattern: str) -> None:
    manifest, owner, payload = _fixture(tmp_path)
    if mutation == "wrong_inherited_sha":
        payload["inherits_r3_1"]["sha256"] = "wrong"
    elif mutation == "missing_inherited":
        payload["inherits_r3_1"]["path"] = "missing-inherited.json"
    elif mutation in {"wrong_inherited_component", "missing_inherited_component"}:
        inherited = ROOT / payload["inherits_r3_1"]["path"]
        copied = json.loads(inherited.read_text())
        key = next(iter(copied["future_execution_components_sha256"]))
        if mutation == "wrong_inherited_component":
            copied["future_execution_components_sha256"][key] = "wrong"
        else:
            copied["future_execution_components_sha256"]["missing_component.py"] = "deadbeef"
        fake = tmp_path / "inherited.json"; fake.write_text(json.dumps(copied))
        payload["inherits_r3_1"] = {"path": str(fake), "sha256": _sha(fake)}
    elif mutation == "wrong_current_component":
        key = next(iter(payload["future_execution_components_sha256"])); payload["future_execution_components_sha256"][key] = "wrong"
    elif mutation == "wrong_roster": payload["roster"]["sha256"] = "wrong"
    elif mutation == "wrong_schedule": payload["schedule"]["sha256"] = "wrong"
    elif mutation == "wrong_pair": payload["frozen_pair_binding"]["sha256"] = "wrong"
    manifest.write_text(json.dumps(payload)); owner.write_text(json.dumps({"OFFICIAL_SMOKE_AUTHORIZED": True, "final_execution_manifest_sha256": _sha(manifest)}))
    with pytest.raises(PermissionError, match=pattern):
        authorize(manifest, owner)
