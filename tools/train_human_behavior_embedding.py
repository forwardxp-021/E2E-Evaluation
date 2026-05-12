#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tools.trajectory_preprocessing import (
    load_traj_as_dense_array,
    normalize_local,
    sanitize_trajectory_array,
)


class HumanDS(Dataset):
    def __init__(self, traj, feat):
        self.traj = traj.astype(np.float32)
        self.feat = feat.astype(np.float32)

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, i):
        return self.traj[i], self.feat[i]


class Enc(nn.Module):
    def __init__(self, emb=64, hid=128):
        super().__init__()
        self.gru = nn.GRU(4, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, emb))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def soft_loss(z, f, temperature, feature_temperature):
    z = F.normalize(z, dim=1, eps=1e-8)
    logits = z @ z.T / temperature
    f = F.normalize(f, dim=1, eps=1e-8)
    dist2 = torch.cdist(f, f, p=2) ** 2
    target_logits = -dist2 / feature_temperature
    b = z.shape[0]
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -1e9)
    target_logits = target_logits.masked_fill(eye, -1e9)
    log_q = F.log_softmax(logits, dim=1)
    p = F.softmax(target_logits, dim=1)
    row_valid = (~eye).any(dim=1)
    loss = -(p[row_valid] * log_q[row_valid]).sum(dim=1).mean()
    return loss, logits, target_logits


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    if args.smoke_test:
        td = Path(tempfile.mkdtemp()) / 'd'
        td.mkdir(parents=True)
        n, t = 96, 80
        traj = np.random.randn(n, t, 4).astype(np.float32)
        feat = np.random.randn(n, 16).astype(np.float32)
        traj[1, 2:5, 2] = np.nan
        traj[5, :2, 1] = np.nan
        traj[7, :, 3] = np.nan
        feat[3, 0] = np.inf
        feat[4, 1] = np.nan
        feat[8, 2] = 1e8
        split = np.array(['train'] * 64 + ['val'] * 16 + ['test'] * 16)
    else:
        d = Path(args.data_dir)
        traj = np.load(d / 'traj.npy', allow_pickle=True)
        feat = np.load(d / 'feat_style.npy')
        split = np.load(d / 'split.npy').astype(str)

    traj_raw = load_traj_as_dense_array(traj)
    traj_nan_raw = int(np.isnan(traj_raw).sum())
    traj_inf_raw = int(np.isinf(traj_raw).sum())
    feat_nan_raw = int(np.isnan(feat).sum())
    feat_inf_raw = int(np.isinf(feat).sum())

    traj_clean, sdiag = sanitize_trajectory_array(traj, mode=args.traj_nan_mode, max_nan_ratio=args.max_traj_nan_ratio)
    retained = sdiag["retained_indices"]
    dropped = sdiag["dropped_indices"]
    feat = feat[retained]
    split = split[retained]

    feat_finite_mask = np.isfinite(feat).all(axis=1)
    dropped_feat_idx = np.where(~feat_finite_mask)[0]
    if dropped_feat_idx.size > 0:
        drop_global = retained[dropped_feat_idx]
        keep_feat = feat_finite_mask
        traj_clean = traj_clean[keep_feat]
        split = split[keep_feat]
        feat = feat[keep_feat]
        retained = retained[keep_feat]
        dropped = np.unique(np.concatenate([dropped, drop_global]))

    np.save(out / 'retained_indices.npy', retained)
    np.save(out / 'dropped_indices.npy', dropped)

    if args.fail_on_nonfinite and (not np.isfinite(traj_clean).all() or not np.isfinite(feat).all()):
        raise RuntimeError("Non-finite values remain after sanitization")

    train_mask = split == 'train'
    mu = feat[train_mask].mean(0, keepdims=True)
    sd = feat[train_mask].std(0, keepdims=True)
    sd = np.maximum(sd, args.feature_std_eps)
    feat_norm = (feat - mu) / sd
    feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0)
    min_before, max_before = float(feat_norm.min()), float(feat_norm.max())
    clipped_count = int((np.abs(feat_norm) > args.feature_clip).sum())
    feat_norm = np.clip(feat_norm, -args.feature_clip, args.feature_clip)
    min_after, max_after = float(feat_norm.min()), float(feat_norm.max())

    print(f"feat raw finite ratio: {np.isfinite(feat).mean():.6f}")
    print(f"feat norm before clip: min={min_before:.4f}, max={max_before:.4f}")
    print(f"feat norm after clip: min={min_after:.4f}, max={max_after:.4f}, clipped={clipped_count}")
    print(f"feat has_nan={np.isnan(feat_norm).any()}, has_inf={np.isinf(feat_norm).any()}")

    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    tr_ds = HumanDS(traj_clean[split == 'train'], feat_norm[split == 'train'])
    va_ds = HumanDS(traj_clean[split == 'val'], feat_norm[split == 'val'])
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = Enc(args.embedding_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, bad, logs, debug = 1e9, 0, [], []

    def run_batch(tb, fb, batch_i, train_mode):
        tb, fb = tb.to(dev), fb.to(dev)
        tb = normalize_local(tb)
        assert torch.isfinite(tb).all(), "non-finite trajectory batch before forward"
        z = model(tb)
        loss, logits, tlogits = soft_loss(z, fb, args.temperature, args.feature_temperature)
        finite = bool(torch.isfinite(loss)) and torch.isfinite(logits).all() and torch.isfinite(tlogits).all()
        if (len(debug) < args.debug_n_batches) and train_mode:
            debug.append({
                "batch": int(batch_i), "traj_min": float(tb.min().item()), "traj_max": float(tb.max().item()),
                "traj_has_nan": bool(torch.isnan(tb).any()), "traj_has_inf": bool(torch.isinf(tb).any()),
                "feat_min": float(fb.min().item()), "feat_max": float(fb.max().item()),
                "feat_has_nan": bool(torch.isnan(fb).any()), "feat_has_inf": bool(torch.isinf(fb).any()),
                "embedding_min": float(z.min().item()), "embedding_max": float(z.max().item()),
                "logits_min": float(logits.min().item()), "logits_max": float(logits.max().item()),
                "target_logits_min": float(tlogits.min().item()), "target_logits_max": float(tlogits.max().item()),
                "loss_finite": finite,
            })
        if not finite:
            msg = f"non-finite batch={batch_i}, logits[{logits.min().item():.4f},{logits.max().item():.4f}], tlogits[{tlogits.min().item():.4f},{tlogits.max().item():.4f}]"
            if args.skip_bad_batches:
                warnings.append(msg)
                print("[warn]", msg)
                return None
            raise RuntimeError(msg)
        return loss

    for ep in range(args.epochs):
        model.train(); tl = []
        for bi, (tb, fb) in enumerate(tr):
            loss = run_batch(tb, fb, bi, train_mode=True)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss.item()))
        model.eval(); vl = []
        with torch.no_grad():
            for bi, (tb, fb) in enumerate(va):
                loss = run_batch(tb, fb, bi, train_mode=False)
                if loss is not None:
                    vl.append(float(loss.item()))
        t = float(np.mean(tl)) if tl else float('nan')
        v = float(np.mean(vl)) if vl else t
        logs.append({'epoch': ep + 1, 'train_loss': t, 'val_loss': v})
        if np.isfinite(v) and v < best:
            best = v; bad = 0
            torch.save({'model': model.state_dict(), 'embedding_dim': args.embedding_dim}, out / 'model.pt')
        else:
            bad += 1
        if bad >= args.patience:
            break

    pd.DataFrame(logs).to_csv(out / 'train_log.csv', index=False)
    (out / 'train_debug.json').write_text(json.dumps(debug, indent=2))
    summary = {
        'data_dir': args.data_dir,
        'n_total': int(len(traj_raw)), 'n_retained': int(len(retained)), 'n_dropped': int(len(dropped)),
        'n_train': int((split == 'train').sum()), 'n_val': int((split == 'val').sum()),
        'traj_shape': list(traj_clean.shape), 'feat_shape': list(feat_norm.shape),
        'traj_nan_count_raw': traj_nan_raw, 'traj_inf_count_raw': traj_inf_raw,
        'traj_nan_count_after_sanitize': int(np.isnan(traj_clean).sum()), 'traj_inf_count_after_sanitize': int(np.isinf(traj_clean).sum()),
        'feat_nan_count_raw': feat_nan_raw, 'feat_inf_count_raw': feat_inf_raw,
        'feat_norm_min_before_clip': min_before, 'feat_norm_max_before_clip': max_before,
        'feat_norm_min_after_clip': min_after, 'feat_norm_max_after_clip': max_after,
        'feature_clip': args.feature_clip, 'temperature': args.temperature, 'feature_temperature': args.feature_temperature,
        'batch_size': args.batch_size, 'epochs': args.epochs, 'best_val_loss': best,
        'final_train_loss': logs[-1]['train_loss'] if logs else None, 'final_val_loss': logs[-1]['val_loss'] if logs else None,
        'stopped_early': bad >= args.patience,
        'warnings': warnings,
        'dropped_due_to_nonfinite_traj': int(sdiag['dropped_count']),
        'dropped_due_to_nonfinite_feat': int(len(dropped_feat_idx)),
        'traj_repaired_count': int(sdiag['repaired_count']),
        'feature_clipped_values': clipped_count,
    }
    (out / 'train_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'val_metrics.json').write_text(json.dumps({'best_val_loss': best}, indent=2))
    plt.figure(); plt.plot([x['epoch'] for x in logs], [x['train_loss'] for x in logs], label='train'); plt.plot([x['epoch'] for x in logs], [x['val_loss'] for x in logs], label='val'); plt.legend(); plt.tight_layout(); plt.savefig(out / 'training_curve.png'); plt.close()
    print('train_done')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='outputs/waymo_human_v1_full51')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--feature_temperature', type=float, default=1.0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--traj_nan_mode', choices=['interpolate', 'zero', 'drop'], default='interpolate')
    p.add_argument('--max_traj_nan_ratio', type=float, default=0.2)
    p.add_argument('--fail_on_nonfinite', action='store_true')
    p.add_argument('--feature_clip', type=float, default=10.0)
    p.add_argument('--feature_std_eps', type=float, default=1e-6)
    p.add_argument('--skip_bad_batches', action='store_true')
    p.add_argument('--debug_n_batches', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run(a)
