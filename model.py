import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TrajectoryEncoder(nn.Module):
    """GRU encoder for variable-length trajectories."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        mlp_dim: int = 128,
        emb_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
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

    def forward(self, traj: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(traj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        z = self.head(h[-1])
        z = nn.functional.normalize(z, dim=-1)
        return z
