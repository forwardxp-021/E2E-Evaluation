import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import TrajectoryEncoder


class TrajOnlyDataset(Dataset):
    def __init__(self, traj_path: str):
        traj_loaded = np.load(traj_path, allow_pickle=True)
        self.traj = [np.asarray(t, dtype=np.float32) for t in traj_loaded]

    def __len__(self) -> int:
        return len(self.traj)

    def __getitem__(self, idx: int):
        return self.traj[idx], idx


def collate_fn(batch):
    trajs = [torch.as_tensor(item[0], dtype=torch.float32) for item in batch]
    idxs = torch.as_tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.as_tensor([t.shape[0] for t in trajs], dtype=torch.long)
    padded = pad_sequence(trajs, batch_first=True)
    return padded, lengths, idxs


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"

    parser = argparse.ArgumentParser(description="Export full-split embeddings aligned with traj/feat/split rows")
    parser.add_argument("--traj_path", type=str, default=str(out_dir / "traj.npy"))
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

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: str) -> TrajectoryEncoder:
    model = TrajectoryEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        emb_dim=args.emb_dim,
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

    dataset = TrajOnlyDataset(args.traj_path)
    if args.split_path:
        split = np.load(args.split_path, allow_pickle=True)
        if len(split) != len(dataset):
            raise ValueError(
                f"split rows ({len(split)}) != traj rows ({len(dataset)}). "
                "embeddings_all.npy must align with traj/feat/split rows."
            )

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
        for traj, lengths, idx in loader:
            traj = traj.to(device)
            lengths = lengths.to(device)
            z = model(traj, lengths)
            z = torch.nn.functional.normalize(z, dim=-1)
            emb_all[idx.numpy()] = z.cpu().numpy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb_all)
    print(f"Saved embeddings: {output_path}")
    print(f"Shape: {emb_all.shape}")


if __name__ == "__main__":
    main()
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

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: str) -> TrajectoryEncoder:
    model = TrajectoryEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        emb_dim=args.emb_dim,
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

    dataset = TrajOnlyDataset(args.traj_path)
    if args.split_path:
        split = np.load(args.split_path, allow_pickle=True)
        if len(split) != len(dataset):
            raise ValueError(
                f"split rows ({len(split)}) != traj rows ({len(dataset)}). "
                "embeddings_all.npy must align with traj/feat/split rows."
            )

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
        for traj, lengths, idx in loader:
            traj = traj.to(device)
            lengths = lengths.to(device)
            z = model(traj, lengths)
            z = torch.nn.functional.normalize(z, dim=-1)
            emb_all[idx.numpy()] = z.cpu().numpy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb_all)
    print(f"Saved embeddings: {output_path}")
    print(f"Shape: {emb_all.shape}")


if __name__ == "__main__":
    main()
    parser.add_argument("--checkpoint", type=str, default=str(out_dir / "best_model.pth"))
    parser.add_argument("--output_path", type=str, default=str(out_dir / "embeddings_all.npy"))

    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: str) -> TrajectoryEncoder:
    model = TrajectoryEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        emb_dim=args.emb_dim,
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
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajOnlyDataset(args.traj_path)
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
        for traj, lengths, idx in loader:
            traj = traj.to(device)
            lengths = lengths.to(device)
            z = model(traj, lengths)
            z = torch.nn.functional.normalize(z, dim=-1)
            emb_all[idx.numpy()] = z.cpu().numpy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb_all)
    print(f"Saved embeddings: {output_path}")
    print(f"Shape: {emb_all.shape}")


if __name__ == "__main__":
    main()
