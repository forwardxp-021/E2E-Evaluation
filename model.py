import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from input_features import REL_KINEMATICS_DIM, build_rel_kinematics


class TrajectoryEncoder(nn.Module):
    """GRU encoder for variable-length trajectories.

    Supports two input modes selected at construction time:

    ``raw_xyv`` (default)
        Feed the raw ego trajectory [x, y, vx, vy] directly to the GRU.
        ``input_dim`` controls the feature width (default 4).

    ``rel_kinematics``
        Compute 12-dim per-frame relative kinematics from aligned ego and
        front trajectories before feeding the GRU.  ``input_dim`` is ignored
        in this mode; the GRU always receives ``REL_KINEMATICS_DIM`` (12)
        features per step.  The ``front`` argument must be supplied to
        :py:meth:`forward`.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        mlp_dim: int = 128,
        emb_dim: int = 64,
        dropout: float = 0.1,
        input_mode: str = "raw_xyv",
        dt: float = 0.1,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.dt = dt
        actual_input_dim = REL_KINEMATICS_DIM if input_mode == "rel_kinematics" else input_dim
        self.gru = nn.GRU(
            input_size=actual_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, emb_dim),
        )

    def forward(
        self,
        traj: torch.Tensor,
        lengths: torch.Tensor,
        front: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of trajectories.

        Args:
            traj:    [B, T, input_dim] padded ego trajectories.
            lengths: [B] valid sequence lengths.
            front:   [B, T, 4] padded front trajectories.  Required when
                     ``input_mode='rel_kinematics'``, ignored otherwise.

        Returns:
            z: [B, emb_dim] L2-normalised embeddings.
        """
        if self.input_mode == "rel_kinematics":
            if front is None:
                raise ValueError(
                    "input_mode='rel_kinematics' requires front trajectories; "
                    "pass front=[B,T,4] to TrajectoryEncoder.forward()."
                )
            x = build_rel_kinematics(traj, front, lengths, dt=self.dt)
        else:
            x = traj
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        z = self.head(h[-1])
        z = nn.functional.normalize(z, dim=-1)
        return z
