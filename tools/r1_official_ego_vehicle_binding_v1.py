#!/usr/bin/env python3
"""Official nuPlan runtime ego footprint binding used by B2.5 clearance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def official_ego_vehicle_binding_v1() -> Dict[str, Any]:
    from nuplan.common.actor_state import vehicle_parameters as module

    parameters = module.get_pacifica_parameters()
    source = Path(module.__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if parameters.length <= 0 or parameters.width <= 0:
        raise ValueError("OFFICIAL_EGO_FOOTPRINT_MISSING")
    return {"status": "OFFICIAL_EGO_FOOTPRINT_BOUND", "length_m": float(parameters.length), "width_m": float(parameters.width), "vehicle_name": str(parameters.vehicle_name), "vehicle_type": str(parameters.vehicle_type), "source": "nuplan.common.actor_state.vehicle_parameters.get_pacifica_parameters", "source_file": str(source), "source_file_sha256": digest, "nuplan_devkit_commit": "e9241677997dd86bfc0bcd44817ab04fe631405b", "generic_fallback_used": False}


__all__ = ["official_ego_vehicle_binding_v1"]
