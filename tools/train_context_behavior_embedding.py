#!/usr/bin/env python3
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

from context_shard_dataset import ContextShardDataset, inspect_shard_manifest


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


def run(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists, pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    train_ds = ContextShardDataset(
        args.shard_manifest,
        split='train',
        max_samples=args.max_train_samples,
        cache_shards=args.cache_shards,
        mmap_mode="r",
    )
    val_ds = ContextShardDataset(
        args.shard_manifest,
        split='val',
        max_samples=args.max_val_samples,
        cache_shards=args.cache_shards,
        mmap_mode="r",
    )
    sample = train_ds[0]
    context_dim = int(sample['context'].shape[-1]); feature_dim = int(sample['feat'].shape[-1])

    model = ContextFlattenGRUEncoder(context_dim, args.hidden_dim, args.embedding_dim, args.num_layers, args.dropout).to(dev)
    aux_head = nn.Linear(args.embedding_dim, feature_dim).to(dev) if args.aux_regression else None
    opt = torch.optim.Adam(list(model.parameters()) + ([] if aux_head is None else list(aux_head.parameters())), lr=args.lr)

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader_kwargs["persistent_workers"] = args.persistent_workers
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    best_val, best_epoch = float('inf'), -1
    train_losses, val_losses = [], []
    warnings = []
    for ep in range(args.epochs):
        model.train(); total=0.0; n=0
        for b in train_loader:
            x = b['context'].float().to(dev); feat = b['feat'].float().to(dev)
            z = model(x)
            loss = soft_contrastive_loss(z, feat, args.temperature, args.feature_temperature)
            if args.metric_alignment:
                loss = loss + args.metric_loss_weight * metric_alignment_loss(z, feat, args.metric_loss_type)
            if args.aux_regression:
                pred = aux_head(z)
                loss = loss + args.aux_loss_weight * F.mse_loss(pred, feat)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch={ep}")
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        tr = total / max(1, n); train_losses.append(tr)

        model.eval(); total=0.0; n=0
        with torch.no_grad():
            for b in val_loader:
                x = b['context'].float().to(dev); feat = b['feat'].float().to(dev)
                z = model(x)
                vl = soft_contrastive_loss(z, feat, args.temperature, args.feature_temperature)
                if args.metric_alignment:
                    vl = vl + args.metric_loss_weight * metric_alignment_loss(z, feat, args.metric_loss_type)
                if args.aux_regression:
                    vl = vl + args.aux_loss_weight * F.mse_loss(aux_head(z), feat)
                if not torch.isfinite(vl):
                    raise RuntimeError(f"Non-finite val loss at epoch={ep}")
                total += vl.item() * x.size(0); n += x.size(0)
        va = total / max(1, n); val_losses.append(va)
        if va < best_val:
            best_val, best_epoch = va, ep + 1
            torch.save({'model': model.state_dict(), 'embedding_dim': args.embedding_dim, 'context_dim': context_dim}, out / 'model.pt')

    with (out / 'train_log.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['epoch','train_loss','val_loss'])
        for i,(a,b) in enumerate(zip(train_losses,val_losses),1): w.writerow([i,a,b])
    for vals, name in [(train_losses,'loss_curve.png'), (val_losses,'val_loss_curve.png')]:
        if plt is not None:
            xs = list(range(1, len(vals) + 1))
            plt.figure()
            plt.plot(xs, vals, marker='o')
            if len(xs) == 1:
                plt.xlim(xs[0] - 0.5, xs[0] + 0.5)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.tight_layout()
            plt.savefig(out / name)
            plt.close()

    manifest_info = inspect_shard_manifest(args.shard_manifest)
    (out / 'feature_stats_used.json').write_text(json.dumps({'use_standardized_features': True, 'feature_dim': feature_dim}, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    summary = {
        'manifest_path': args.shard_manifest, 'total_train_samples': len(train_ds), 'total_val_samples': len(val_ds),
        'context_dim': context_dim, 'feature_dim': feature_dim, 'embedding_dim': args.embedding_dim,
        'best_val_loss': best_val, 'best_epoch': best_epoch, 'final_train_loss': train_losses[-1], 'final_val_loss': val_losses[-1],
        'device': str(dev), 'warnings': warnings, 'manifest_report': manifest_info,
        'cache_shards': args.cache_shards,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'prefetch_factor': args.prefetch_factor,
        'persistent_workers': args.persistent_workers,
        'mmap_mode_enabled': True,
    }
    (out / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard_manifest', required=True)
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
    p.add_argument('--metric_alignment', action='store_true')
    p.add_argument('--metric_loss_weight', type=float, default=0.1)
    p.add_argument('--metric_loss_type', choices=['mse','huber'], default='huber')
    p.add_argument('--metric_targets', default='all')
    p.add_argument('--aux_regression', action='store_true')
    p.add_argument('--aux_loss_weight', type=float, default=0.1)
    p.add_argument('--aux_targets', default='all')
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--max_val_samples', type=int, default=None)
    p.add_argument('--cache_shards', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--persistent_workers', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke_test_real_data', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    run(p.parse_args())
