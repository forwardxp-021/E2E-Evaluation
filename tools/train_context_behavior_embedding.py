#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import random
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.context_shard_dataset import ContextShardDataset, inspect_shard_manifest


FEATURE_GROUPS = {
    "longitudinal_comfort": ["rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk"],
    "following_interaction": [
        "mean_thw", "min_thw", "mean_front_distance", "min_front_distance",
        "mean_rel_speed", "p95_rel_speed", "front_pressure_score", "rear_vehicle_pressure_proxy"
    ],
    "lateral_dynamics": [
        "rms_yaw_rate", "rms_curvature", "heading_change_total", "lane_change_count_proxy",
        "lane_change_rate_proxy", "max_lateral_speed", "rms_lateral_accel", "lane_change_oscillation_score_proxy"
    ],
    "lateral_gap_interaction": [
        "left_front_min_gap", "left_rear_min_gap", "right_front_min_gap", "right_rear_min_gap",
        "left_gap_min", "right_gap_min", "left_gap_acceptance_proxy", "right_gap_acceptance_proxy"
    ],
    "behavior_proxy": ["yielding_score_proxy", "assertiveness_score_proxy"],
}


class ContextFlattenGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))

    def forward(self, context_seq):
        _, h = self.gru(context_seq)
        return self.proj(h[-1])


def soft_contrastive_loss(z, f, temperature=0.1, feature_temperature=1.0):
    z = F.normalize(z, dim=1, eps=1e-8)
    f = F.normalize(f, dim=1, eps=1e-8)
    logits = (z @ z.T) / temperature
    target_logits = -(torch.cdist(f, f, p=2) ** 2) / feature_temperature
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    target_logits = target_logits.masked_fill(eye, -1e9)
    p = F.softmax(target_logits, dim=1)
    logq = F.log_softmax(logits, dim=1)
    return -(p * logq).sum(dim=1).mean()


def metric_alignment_loss(z, tgt, loss_type='huber'):
    dz = torch.cdist(F.normalize(z, dim=1, eps=1e-8), F.normalize(z, dim=1, eps=1e-8), p=2)
    dt = torch.cdist(tgt, tgt, p=2)
    m = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    dz, dt = dz[m], dt[m]
    dz = (dz - dz.mean()) / (dz.std(unbiased=False) + 1e-8)
    dt = (dt - dt.mean()) / (dt.std(unbiased=False) + 1e-8)
    return F.mse_loss(dz, dt) if loss_type == 'mse' else F.smooth_l1_loss(dz, dt)


def _load_feature_schema(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"feature_schema.json not found: {path}")
    data = json.loads(path.read_text(encoding='utf-8'))
    names = data.get('feature_names', [])
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid feature_schema.json at {path}: missing feature_names list")
    return names


def _resolve_group_indices(feature_names):
    fmap = {name: i for i, name in enumerate(feature_names)}
    resolved = {}
    for group, names in FEATURE_GROUPS.items():
        missing = [n for n in names if n not in fmap]
        if missing:
            raise ValueError(f"feature_schema missing names for {group}: {missing}")
        resolved[group] = [fmap[n] for n in names]
    return resolved


def run(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists, pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    feature_names = _load_feature_schema(Path(args.feature_schema))
    group_indices = _resolve_group_indices(feature_names)

    train_ds = ContextShardDataset(args.shard_manifest, split='train', max_samples=args.max_train_samples, cache_shards=args.cache_shards, mmap_mode="r")
    val_ds = ContextShardDataset(args.shard_manifest, split='val', max_samples=args.max_val_samples, cache_shards=args.cache_shards, mmap_mode="r")
    sample = train_ds[0]
    context_dim = int(sample['context'].shape[-1]); feature_dim = int(sample['feat'].shape[-1])
    if feature_dim != len(feature_names):
        raise ValueError(f"feature_dim mismatch: dataset={feature_dim}, feature_schema={len(feature_names)}")

    model = ContextFlattenGRUEncoder(context_dim, args.hidden_dim, args.embedding_dim, args.num_layers, args.dropout).to(dev)
    aux_heads = nn.ModuleDict({
        'longitudinal': nn.Linear(args.embedding_dim, len(group_indices['longitudinal_comfort'])),
        'following': nn.Linear(args.embedding_dim, len(group_indices['following_interaction'])),
        'lateral_dynamics': nn.Linear(args.embedding_dim, len(group_indices['lateral_dynamics'])),
        'lateral_gap': nn.Linear(args.embedding_dim, len(group_indices['lateral_gap_interaction'])),
        'behavior_proxy': nn.Linear(args.embedding_dim, len(group_indices['behavior_proxy'])),
    }).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(aux_heads.parameters()), lr=args.lr)

    def _compute_losses(z, feat):
        loss = args.style_loss_weight * soft_contrastive_loss(z, feat, args.temperature, args.feature_temperature)
        losses = {'style_loss': loss.detach().item()}
        groups = {
            'longitudinal': ('longitudinal_comfort', args.aux_longitudinal_weight, args.metric_longitudinal_weight),
            'following': ('following_interaction', args.aux_following_weight, args.metric_following_weight),
            'lateral_dynamics': ('lateral_dynamics', args.aux_lateral_dynamics_weight, args.metric_lateral_dynamics_weight),
            'lateral_gap': ('lateral_gap_interaction', args.aux_lateral_gap_weight, args.metric_lateral_gap_weight),
            'behavior_proxy': ('behavior_proxy', args.aux_behavior_proxy_weight, args.metric_behavior_proxy_weight),
        }
        for head_key, (group_key, aux_w, metric_w) in groups.items():
            idx = group_indices[group_key]
            tgt = feat[:, idx]
            pred = aux_heads[head_key](z)
            aux_l = F.smooth_l1_loss(pred, tgt)
            met_l = metric_alignment_loss(z, tgt, args.metric_loss_type)
            loss = loss + aux_w * aux_l + metric_w * met_l
            losses[f'aux_{head_key}'] = aux_l.detach().item()
            losses[f'metric_{head_key}'] = met_l.detach().item()
        return loss, losses

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    best_val, best_epoch = float('inf'), -1
    train_losses, val_losses = [], []
    for ep in range(args.epochs):
        model.train(); aux_heads.train(); total=0.0; n=0
        for b in train_loader:
            x = b['context'].float().to(dev); feat = b['feat'].float().to(dev)
            z = model(x)
            loss, _ = _compute_losses(z, feat)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch={ep}")
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        tr = total / max(1, n); train_losses.append(tr)

        model.eval(); aux_heads.eval(); total=0.0; n=0
        with torch.no_grad():
            for b in val_loader:
                x = b['context'].float().to(dev); feat = b['feat'].float().to(dev)
                z = model(x)
                vl, _ = _compute_losses(z, feat)
                if not torch.isfinite(vl):
                    raise RuntimeError(f"Non-finite val loss at epoch={ep}")
                total += vl.item() * x.size(0); n += x.size(0)
        va = total / max(1, n); val_losses.append(va)
        if va < best_val:
            best_val, best_epoch = va, ep + 1
            state = {'model': model.state_dict(), 'aux_heads': aux_heads.state_dict(), 'embedding_dim': args.embedding_dim, 'context_dim': context_dim}
            torch.save(state, out / 'best_model.pt')
            torch.save(state, out / 'model.pt')

    with (out / 'train_log.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['epoch','train_loss','val_loss'])
        for i,(a,b) in enumerate(zip(train_losses,val_losses),1): w.writerow([i,a,b])

    (out / 'training_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    group_cfg = {'feature_schema': args.feature_schema, 'feature_names': feature_names, 'feature_groups': FEATURE_GROUPS, 'group_indices': group_indices, 'aux_loss': 'SmoothL1Loss', 'metric_loss_type': args.metric_loss_type}
    (out / 'feature_group_config.json').write_text(json.dumps(group_cfg, indent=2), encoding='utf-8')

    summary = {'manifest_path': args.shard_manifest, 'total_train_samples': len(train_ds), 'total_val_samples': len(val_ds), 'context_dim': context_dim, 'feature_dim': feature_dim, 'embedding_dim': args.embedding_dim, 'best_val_loss': best_val, 'best_epoch': best_epoch, 'final_train_loss': train_losses[-1], 'final_val_loss': val_losses[-1], 'device': str(dev)}
    (out / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--feature_schema', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--feature_temperature', type=float, default=1.0)
    p.add_argument('--metric_loss_type', choices=['mse','huber'], default='huber')
    p.add_argument('--style_loss_weight', type=float, default=1.0)
    p.add_argument('--aux_longitudinal_weight', type=float, default=0.5)
    p.add_argument('--aux_following_weight', type=float, default=1.5)
    p.add_argument('--aux_lateral_dynamics_weight', type=float, default=1.0)
    p.add_argument('--aux_lateral_gap_weight', type=float, default=1.0)
    p.add_argument('--aux_behavior_proxy_weight', type=float, default=0.5)
    p.add_argument('--metric_longitudinal_weight', type=float, default=0.5)
    p.add_argument('--metric_following_weight', type=float, default=2.0)
    p.add_argument('--metric_lateral_dynamics_weight', type=float, default=1.0)
    p.add_argument('--metric_lateral_gap_weight', type=float, default=1.0)
    p.add_argument('--metric_behavior_proxy_weight', type=float, default=0.5)
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--max_val_samples', type=int, default=None)
    p.add_argument('--cache_shards', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    run(p.parse_args())
