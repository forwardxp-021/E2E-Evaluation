import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import TrajectoryEncoder


class TrajOnlyDataset(Dataset):
    def __init__(self, traj_path: str, front_path: str | None = None):
        traj_loaded = np.load(traj_path, allow_pickle=True)
        self.traj = [np.asarray(t, dtype=np.float32) for t in traj_loaded]
        self.front = None
        if front_path is not None:
            front_loaded = np.load(front_path, allow_pickle=True)
            self.front = [np.asarray(f, dtype=np.float32) for f in front_loaded]
            if len(self.front) != len(self.traj):
                raise ValueError(
                    f"front length {len(self.front)} != traj length {len(self.traj)}"
                )

    def __len__(self) -> int:
        return len(self.traj)

    def __getitem__(self, idx: int):
        front = None if self.front is None else self.front[idx]
        return self.traj[idx], front, idx


def collate_fn(batch):
    trajs = [torch.as_tensor(item[0], dtype=torch.float32) for item in batch]
    fronts_raw = [item[1] for item in batch]
    idxs = torch.as_tensor([item[2] for item in batch], dtype=torch.long)
    lengths = torch.as_tensor([t.shape[0] for t in trajs], dtype=torch.long)
    padded = pad_sequence(trajs, batch_first=True)
    front_padded = None
    if fronts_raw[0] is not None:
        fronts = [torch.as_tensor(f, dtype=torch.float32) for f in fronts_raw]
        front_padded = pad_sequence(fronts, batch_first=True)
    return padded, lengths, idxs, front_padded


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"

    parser = argparse.ArgumentParser(description="Export full-split embeddings aligned with traj/feat/split rows")
    parser.add_argument("--traj_path", type=str, default=str(out_dir / "traj.npy"))
    parser.add_argument("--front_path", type=str, default=None, help="Path to front.npy; required for --input_mode rel_kinematics")
    parser.add_argument("--checkpoint", type=str, default=str(out_dir / "best_model.pth"))
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Backward-compatible alias of --checkpoint",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Optional split file for row-count validation only",
    )
    parser.add_argument("--output_path", type=str, default=str(out_dir / "embeddings_all.npy"))

    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["raw_xyv", "rel_kinematics"],
        default="raw_xyv",
        help=(
            "Input representation matching the trained model. "
            "'raw_xyv' (default): raw ego [x,y,vx,vy]. "
            "'rel_kinematics': 12-dim relative kinematics; requires --front_path."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step in seconds between frames (Waymo 10 Hz → 0.1 s). Used by rel_kinematics.",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: str) -> TrajectoryEncoder:
    model = TrajectoryEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        emb_dim=args.emb_dim,
        input_mode=args.input_mode,
        dt=args.dt,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.checkpoint_path:
        args.checkpoint = args.checkpoint_path

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.input_mode == "rel_kinematics" and args.front_path is None:
        raise ValueError("--input_mode rel_kinematics requires --front_path to be provided.")

    dataset = TrajOnlyDataset(args.traj_path, front_path=args.front_path)
    if args.split_path:
        split = np.load(args.split_path, allow_pickle=True)
        if len(split) != len(dataset):
            raise ValueError(
                f"split rows ({len(split)}) != traj rows ({len(dataset)}). "
                "embeddings_all.npy must align with traj/feat/split rows."
            )

    if args.input_mode == "rel_kinematics":
        print(f"Input mode: rel_kinematics (12-dim) | dt={args.dt}s | front loaded: {len(dataset.front)} windows")
    else:
        print(f"Input mode: raw_xyv ({args.input_dim}-dim)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = load_model(args, device)

    emb_all = np.zeros((len(dataset), args.emb_dim), dtype=np.float32)

    with torch.no_grad():
        for traj, lengths, idx, front in loader:
            traj = traj.to(device)
            lengths = lengths.to(device)
            front_t = front.to(device) if front is not None else None
            z = model(traj, lengths, front=front_t)
            z = torch.nn.functional.normalize(z, dim=-1)
            emb_all[idx.numpy()] = z.cpu().numpy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb_all)
    print(f"Saved embeddings: {output_path}")
    print(f"Shape: {emb_all.shape}")


if __name__ == "__main__":
    main()
