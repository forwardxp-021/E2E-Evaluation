#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from context_shard_dataset import ContextShardDataset
from train_context_behavior_embedding import ContextFlattenGRUEncoder


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb_root = out / 'embeddings'
    emb_root.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ds = ContextShardDataset(args.shard_manifest, split=args.split)
    sample = ds[0]
    context_dim = int(sample['context'].shape[-1])
    ckpt_context_dim = ckpt.get('context_dim')
    if ckpt_context_dim is None:
        raise ValueError("Checkpoint is missing required context_dim.")
    if int(ckpt_context_dim) != context_dim:
        raise ValueError(
            f"checkpoint['context_dim']={ckpt_context_dim} does not match input context_traj last dimension={context_dim}."
        )
    emb_dim = int(ckpt.get('embedding_dim', 64))

    model = ContextFlattenGRUEncoder(context_dim, embedding_dim=emb_dim)
    model.load_state_dict(ckpt['model'], strict=False)
    dev = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model.to(dev).eval()

    by_shard = {}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    nonfinite = False
    with torch.no_grad():
        for b in loader:
            z = model(b['context'].float().to(dev)).cpu().numpy().astype(np.float32)
            if not np.isfinite(z).all():
                nonfinite = True
            metas = b['meta']
            for i in range(z.shape[0]):
                sid = int(metas['shard_id'][i])
                lid = int(metas['local_index'][i])
                by_shard.setdefault(sid, []).append((lid, z[i]))

    paths = []
    total = 0
    merged = []
    for sid in sorted(by_shard.keys()):
        rows = sorted(by_shard[sid], key=lambda x: x[0])
        arr = np.stack([x[1] for x in rows], axis=0) if rows else np.zeros((0, emb_dim), dtype=np.float32)
        shard_dir = emb_root / f'shard_{sid:06d}'
        shard_dir.mkdir(parents=True, exist_ok=True)
        p = shard_dir / 'embeddings.npy'
        np.save(p, arr)
        paths.append(str(p.relative_to(out)))
        total += arr.shape[0]
        if args.merge_embeddings:
            merged.append(arr)

    if args.merge_embeddings:
        np.save(out / 'embeddings.npy', np.concatenate(merged, axis=0) if merged else np.zeros((0, emb_dim), dtype=np.float32))

    manifest = {
        'source_shard_manifest': args.shard_manifest,
        'checkpoint': args.checkpoint,
        'embedding_dim': emb_dim,
        'checkpoint_context_dim': int(ckpt_context_dim),
        'input_context_dim': int(context_dim),
        'total_rows': total,
        'split': args.split,
        'embedding_shard_paths': paths,
        'row_alignment': 'Each embedding shard follows source shard row order for selected split.',
        'nonfinite_embedding_detected': int(nonfinite),
    }
    (out / 'embedding_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all')
    p.add_argument('--merge_embeddings', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    run(p.parse_args())