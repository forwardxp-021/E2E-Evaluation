#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_human_behavior_embedding import Enc
from trajectory_preprocessing import (
    assert_finite_array,
    compute_traj_nan_stats,
    load_traj_as_dense_array,
    normalize_local,
    sanitize_trajectory_array,
)


def _check_checkpoint_metadata(obj, allow_nan_checkpoint: bool):
    bad = False
    for k in ["best_val_loss", "final_train_loss", "final_val_loss"]:
        if k in obj and obj[k] is not None and not np.isfinite(float(obj[k])):
            bad = True
    if bool(obj.get("has_nan_loss", False)):
        bad = True
    if bad and not allow_nan_checkpoint:
        raise RuntimeError(
            "Checkpoint appears to come from NaN training. Please retrain after fixing trajectory sanitization."
        )
    return bad


def run(args):
    warnings = []
    if args.smoke_test:
        data_dir = Path(tempfile.mkdtemp()) / "d"
        data_dir.mkdir(parents=True)
        n, t = 48, 20
        traj = np.random.randn(n, t, 4).astype(np.float32)
        traj[1, 2:7, 0] = np.nan
        traj[2, 0:3, 3] = np.inf
        traj[3, 0:2, 1] = np.nan
        np.save(data_dir / "traj.npy", traj)
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        if ckpt is None or not ckpt.exists():
            m = Enc(64)
            tmp = Path(tempfile.mkdtemp()) / "m.pt"
            torch.save({"model": m.state_dict(), "embedding_dim": 64, "best_val_loss": 1.0}, tmp)
            ckpt = tmp
    else:
        data_dir = Path(args.data_dir)
        ckpt = Path(args.checkpoint)
        traj = np.load(data_dir / "traj.npy", allow_pickle=True)

    traj_raw = load_traj_as_dense_array(traj)
    raw_stats = compute_traj_nan_stats(traj_raw)
    traj_clean, sdiag = sanitize_trajectory_array(
        traj,
        mode=args.traj_nan_mode,
        max_nan_ratio=args.max_traj_nan_ratio,
    )
    retained = sdiag["retained_indices"]
    dropped = sdiag["dropped_indices"]
    if len(dropped) > 0 and not args.allow_drop:
        raise RuntimeError(
            f"sanitize dropped {len(dropped)} rows; row-aligned export requires zero dropped rows. "
            "Use different sanitization or explicitly set --allow_drop."
        )
    clean_stats = compute_traj_nan_stats(traj_clean)

    if args.fail_on_nonfinite:
        assert_finite_array(traj_clean, "traj_clean")

    obj = torch.load(ckpt, map_location="cpu")
    bad_ckpt = _check_checkpoint_metadata(obj, args.allow_nan_checkpoint)
    if bad_ckpt:
        warnings.append("checkpoint metadata indicates NaN losses")

    emb_dim = int(obj.get("embedding_dim", 64))
    model = Enc(emb_dim)
    model.load_state_dict(obj["model"], strict=False)
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()

    print(f"traj shape: {traj_raw.shape}")
    print(f"raw NaN count: {raw_stats['nan_count']}")
    print(f"raw Inf count: {raw_stats['inf_count']}")
    print(f"raw finite ratio: {raw_stats['finite_ratio']:.6f}")
    print(f"sanitized NaN count: {clean_stats['nan_count']}")
    print(f"sanitized Inf count: {clean_stats['inf_count']}")
    print(f"repaired samples: {sdiag['repaired_count']}")
    print(f"unrepaired/dropped samples: {sdiag['dropped_count']}")
    print(f"checkpoint path: {ckpt}")
    print(f"embedding_dim: {emb_dim}")
    print(f"output path: {args.out_path}")

    out = []
    debug = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(traj_clean), bs), desc="Exporting embeddings", leave=False):
            batch_np = traj_clean[i : i + bs]
            x = torch.from_numpy(batch_np).float().to(dev)
            x_local = normalize_local(x)
            if args.fail_on_nonfinite:
                assert_finite_array(x_local, "batch_local")
            z = model(x_local)
            assert_finite_array(z, "batch_embedding_raw")
            z = F.normalize(z, dim=1, eps=1e-8)
            assert_finite_array(z, "batch_embedding_norm")
            if len(debug) < 3:
                debug.append({
                    "batch_start": int(i),
                    "input_min": float(x.min().item()),
                    "input_max": float(x.max().item()),
                    "local_min": float(x_local.min().item()),
                    "local_max": float(x_local.max().item()),
                    "embedding_min": float(z.min().item()),
                    "embedding_max": float(z.max().item()),
                    "has_nan": bool(torch.isnan(z).any().item()),
                    "has_inf": bool(torch.isinf(z).any().item()),
                })
            out.append(z.detach().cpu().numpy())

    emb = np.concatenate(out, axis=0) if out else np.zeros((0, emb_dim), dtype=np.float32)
    assert_finite_array(emb, "embeddings")

    op = Path(args.out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {op}; pass --overwrite")
    np.save(op, emb)

    if len(dropped) > 0:
        np.save(op.parent / "retained_indices.npy", retained)

    row_aligned = len(traj_raw) == len(emb) and len(dropped) == 0
    summary = {
        "data_dir": str(data_dir),
        "checkpoint": str(ckpt),
        "out_path": str(op),
        "n_rows_input": int(len(traj_raw)),
        "n_rows_exported": int(len(emb)),
        "embedding_dim": int(emb.shape[1]),
        "row_aligned": bool(row_aligned),
        "traj_nan_count_raw": raw_stats["nan_count"],
        "traj_inf_count_raw": raw_stats["inf_count"],
        "traj_nan_count_after_sanitize": clean_stats["nan_count"],
        "traj_inf_count_after_sanitize": clean_stats["inf_count"],
        "repaired_sample_count": int(sdiag["repaired_count"]),
        "dropped_sample_count": int(sdiag["dropped_count"]),
        "device": str(dev),
        "batch_size": int(args.batch_size),
        "warnings": warnings,
    }
    summary_path = Path(args.summary_path) if args.summary_path else op.with_name(f"{op.stem}_export_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (op.parent / "embedding_export_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")

    if args.smoke_test:
        assert emb.shape[0] == len(traj_raw), "smoke test must be row-aligned"
        assert np.isfinite(emb).all(), "smoke test embeddings must be finite"

    print("export_done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="outputs/waymo_human_v1_full51")
    p.add_argument("--checkpoint")
    p.add_argument("--out_path", required=True)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--traj_nan_mode", choices=["interpolate", "zero", "drop"], default="interpolate")
    p.add_argument("--max_traj_nan_ratio", type=float, default=0.2)
    p.add_argument("--fail_on_nonfinite", action="store_true")
    p.add_argument("--allow_drop", action="store_true")
    p.add_argument("--allow_nan_checkpoint", action="store_true")
    p.add_argument("--summary_path", default=None)
    run(p.parse_args())
