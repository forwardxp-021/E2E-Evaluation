"""Per-frame input feature builders for TrajectoryEncoder.

Currently provides:
    build_rel_kinematics  – 12-dim relative kinematics from aligned ego/front windows.
"""

import math

import torch

REL_KINEMATICS_DIM = 12

_EPS = 1e-6


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle tensor to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def build_rel_kinematics(
    ego: torch.Tensor,
    front: torch.Tensor,
    lengths: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    """Build 12-dim per-frame relative kinematics features from ego and front trajectories.

    Args:
        ego:     [B, T, 4] padded ego trajectories   – columns [x, y, vx, vy].
        front:   [B, T, 4] padded front trajectories – columns [x, y, vx, vy].
        lengths: [B] valid sequence lengths (padding positions will be zeroed).
        dt:      time step in seconds (Waymo 10 Hz → 0.1 s).

    Returns:
        features: [B, T, 12] float32 tensor.

    Feature layout (0-indexed):
        0  ego_v          = sqrt(vx² + vy²)
        1  front_v        = sqrt(front_vx² + front_vy²)
        2  v_rel          = ego_v − front_v
        3  dx             = front_x − ego_x
        4  dy             = front_y − ego_y
        5  dist           = sqrt(dx² + dy²)
        6  closing_rate   = diff(dist) / dt        (t=0 → 0)
        7  ego_a          = diff(ego_v) / dt        (t=0 → 0)
        8  front_a        = diff(front_v) / dt      (t=0 → 0)
        9  ego_heading    = atan2(vy, vx)
        10 ego_yaw_rate   = wrap(diff(ego_heading)) / dt  (t=0 → 0)
        11 thw            = dist / max(ego_v, eps)
    """
    _B, T, _ = ego.shape

    ego_vx = ego[:, :, 2]
    ego_vy = ego[:, :, 3]
    front_vx = front[:, :, 2]
    front_vy = front[:, :, 3]

    ego_v = torch.sqrt(ego_vx ** 2 + ego_vy ** 2)
    front_v = torch.sqrt(front_vx ** 2 + front_vy ** 2)
    v_rel = ego_v - front_v

    dx = front[:, :, 0] - ego[:, :, 0]
    dy = front[:, :, 1] - ego[:, :, 1]
    dist = torch.sqrt(dx ** 2 + dy ** 2)

    closing_rate = torch.zeros_like(dist)
    closing_rate[:, 1:] = (dist[:, 1:] - dist[:, :-1]) / dt

    ego_a = torch.zeros_like(ego_v)
    ego_a[:, 1:] = (ego_v[:, 1:] - ego_v[:, :-1]) / dt

    front_a = torch.zeros_like(front_v)
    front_a[:, 1:] = (front_v[:, 1:] - front_v[:, :-1]) / dt

    ego_heading = torch.atan2(ego_vy, ego_vx)
    ego_yaw_rate = torch.zeros_like(ego_heading)
    ego_yaw_rate[:, 1:] = _wrap_angle(ego_heading[:, 1:] - ego_heading[:, :-1]) / dt

    thw = dist / torch.clamp(ego_v, min=_EPS)

    features = torch.stack(
        [ego_v, front_v, v_rel, dx, dy, dist, closing_rate, ego_a, front_a, ego_heading, ego_yaw_rate, thw],
        dim=-1,
    )

    # Zero out padding positions.
    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
    features = features * mask.unsqueeze(-1)

    return features
