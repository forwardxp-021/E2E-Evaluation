import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from torch.utils.data import DataLoader, Subset

from dataset import TrajFeatureDataset, collate_variable_traj
from loss import SoftContrastiveLoss
from model import TrajectoryEncoder


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "output"

    parser = argparse.ArgumentParser(description="Feature-guided contrastive training for driving style representation")
    parser.add_argument("--traj_path", type=str, default=str(out_dir / "traj.npy"))
    parser.add_argument("--feat_path", type=str, default=str(out_dir / "feat.npy"))
    parser.add_argument("--feat_raw_path", type=str, default=None)
    parser.add_argument("--front_path", type=str, default=None, help="Path to front.npy for condition vector computation")
    parser.add_argument("--split_path", type=str, default=str(out_dir / "split.npy"))
    parser.add_argument("--pair_cache_path", type=str, default=str(out_dir / "pair_index.npz"))
    parser.add_argument("--output_dir", type=str, default=str(out_dir))

    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)

    parser.add_argument("--k_pos", type=int, default=8)
    parser.add_argument("--k_neg", type=int, default=32)
    parser.add_argument("--distance", type=str, choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tau_feat_min", type=float, default=1e-3, help="Lower bound for adaptive feature temperature")
    parser.add_argument(
        "--feat_norm",
        type=str,
        choices=["none", "batch_std", "l2"],
        default="none",
        help="Feature normalization strategy inside SoftContrastiveLoss",
    )
    parser.add_argument(
        "--tau_mode",
        type=str,
        choices=["batch_median", "anchor_median"],
        default="anchor_median",
        help="Adaptive tau strategy for feature-distance softmax",
    )
    parser.add_argument(
        "--gate_topm",
        type=int,
        default=0,
        help="If >0, only top-M nearest feature neighbors participate in softmax per anchor",
    )
    parser.add_argument(
        "--feat_sim",
        type=str,
        choices=["tau", "local_scale"],
        default="tau",
        help="Feature similarity soft-target mode",
    )
    parser.add_argument("--ls_k", type=int, default=10, help="k-th neighbor distance used as local scaling sigma")
    parser.add_argument("--ls_mode", type=str, choices=["row", "sym"], default="row", help="Local scaling mode")
    parser.add_argument("--ls_sigma_min", type=float, default=1e-3, help="Lower bound for local scaling sigma")
    parser.add_argument("--ls_alpha", type=float, default=1.0, help="Sharpening coefficient for local_scale logits")
    parser.add_argument("--feat_dist_mode", type=str, choices=["plain", "masked"], default="plain")
    parser.add_argument("--min_common_dims", type=int, default=5)
    parser.add_argument("--debug_sim_feat", action="store_true", help="Print first-row sim_feat values once for sanity check")
    parser.add_argument("--debug_sim_topk", type=int, default=10, help="How many sim_feat entries to print for debug")

    # Condition gating arguments.
    parser.add_argument(
        "--cond_mode",
        type=str,
        choices=["off", "hard_box", "knn"],
        default="off",
        help="Condition gating mode: off (disabled), hard_box (within-tolerance filtering), or knn (robust kNN-based gating)",
    )
    parser.add_argument("--cond_speed_tol", type=float, default=2.0, help="Speed tolerance for condition gating (m/s)")
    parser.add_argument("--cond_dist_tol", type=float, default=5.0, help="Distance tolerance for condition gating (m)")
    parser.add_argument("--cond_vrel_tol", type=float, default=1.0, help="Relative speed tolerance for condition gating (m/s)")
    parser.add_argument(
        "--cond_cf_bucket_edges",
        type=str,
        default="0.2,0.6",
        help="Comma-separated edges for cf_valid_frac bucketing (e.g. '0.2,0.6')",
    )
    parser.add_argument(
        "--min_cond_candidates",
        type=int,
        default=8,
        help="Minimum number of condition-compatible candidates before falling back to ungated behavior",
    )
    parser.add_argument("--cond_k", type=int, default=24, help="Number of nearest compatible samples per anchor for cond_mode=knn")
    parser.add_argument(
        "--cond_scale_mode",
        type=str,
        choices=["mad", "iqr", "std"],
        default="mad",
        help="Robust scale estimator for cond_mode=knn distance normalization",
    )

    # Loss mode arguments.
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["softkl", "supcon", "hybrid"],
        default="softkl",
        help="Loss mode: softkl (soft KL), supcon (SupCon with topk positives), hybrid (weighted sum)",
    )
    parser.add_argument("--pos_topk", type=int, default=8, help="Number of top-K positives for supcon/hybrid loss mode")
    parser.add_argument("--w_supcon", type=float, default=1.0, help="Weight for supcon term in hybrid loss")
    parser.add_argument("--w_soft", type=float, default=0.2, help="Weight for softkl term in hybrid loss")

    # Two-stage training schedule.
    # stage1_epochs: hook for future condition-invariance pretraining; currently accepted but not implemented.
    # Set to 0 (default) to use the standard single-stage schedule.
    parser.add_argument("--stage1_epochs", type=int, default=0, help="Epochs for stage 1 invariance pretraining (hook for future use; currently accepted but not active)")
    parser.add_argument("--stage2_epochs", type=int, default=0, help="Epochs for stage 2; if 0, uses --epochs")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument(
        "--skip_val_clustering",
        action="store_true",
        help="Skip val KMeans clustering metrics and silhouette-based best checkpoint selection",
    )

    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def encode_subset(model: TrajectoryEncoder, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            traj = batch["traj"].to(device)
            lengths = batch["lengths"].to(device)
            z = model(traj, lengths)
            embs.append(z.cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, model.head[-1].out_features), dtype=np.float32)


def clustering_metrics(emb: np.ndarray, n_clusters: int) -> dict:
    if len(emb) < max(10, n_clusters + 1):
        return {"sil": float("nan"), "ch": float("nan"), "db": float("nan")}

    sil_scores, ch_scores, db_scores = [], [], []
    for seed in range(5):
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(emb)
        sil_scores.append(silhouette_score(emb, labels))
        ch_scores.append(calinski_harabasz_score(emb, labels))
        db_scores.append(davies_bouldin_score(emb, labels))

    return {
        "sil": float(np.mean(sil_scores)),
        "ch": float(np.mean(ch_scores)),
        "db": float(np.mean(db_scores)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse cf bucket edges.
    cf_bucket_edges: list[float] = []
    if args.cond_cf_bucket_edges:
        try:
            cf_bucket_edges = [float(x.strip()) for x in args.cond_cf_bucket_edges.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(f"Invalid --cond_cf_bucket_edges '{args.cond_cf_bucket_edges}': {e}") from e

    # Determine total epochs (stage2_epochs overrides epochs if set).
    total_epochs = args.stage2_epochs if args.stage2_epochs > 0 else args.epochs

    dataset = TrajFeatureDataset(
        traj_path=args.traj_path,
        feat_path=args.feat_path,
        split_path=args.split_path,
        k_pos=args.k_pos,
        k_neg=args.k_neg,
        metric=args.distance,
        pair_cache_path=args.pair_cache_path,
        build_pairs=False,
        feat_raw_path=args.feat_raw_path,
        front_path=args.front_path,
    )

    train_idx = dataset.indices_by_split("train")
    val_idx = dataset.indices_by_split("val")
    test_idx = dataset.indices_by_split("test")
    print(f"Split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    if dataset.cond is not None:
        print(f"Condition vector shape: {dataset.cond.shape} (speed_mean, dist_mean, vrel_mean, cf_valid_frac)")
    else:
        if args.cond_mode != "off":
            print("[WARN] --cond_mode is not 'off' but --front_path was not provided; condition gating will be skipped.")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_variable_traj,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_traj,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_traj,
        drop_last=False,
    )

    model = TrajectoryEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        emb_dim=args.emb_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SoftContrastiveLoss(
        temperature=args.temperature,
        tau_feat_min=args.tau_feat_min,
        feat_norm=args.feat_norm,
        tau_mode=args.tau_mode,
        gate_topm=args.gate_topm,
        feat_sim=args.feat_sim,
        ls_k=args.ls_k,
        ls_mode=args.ls_mode,
        ls_sigma_min=args.ls_sigma_min,
        ls_alpha=args.ls_alpha,
        feat_dist_mode=args.feat_dist_mode,
        min_common_dims=args.min_common_dims,
        debug_sim=args.debug_sim_feat,
        debug_topk=args.debug_sim_topk,
        cond_mode=args.cond_mode,
        cond_speed_tol=args.cond_speed_tol,
        cond_dist_tol=args.cond_dist_tol,
        cond_vrel_tol=args.cond_vrel_tol,
        cond_cf_bucket_edges=cf_bucket_edges,
        min_cond_candidates=args.min_cond_candidates,
        cond_k=args.cond_k,
        cond_scale_mode=args.cond_scale_mode,
        loss_mode=args.loss_mode,
        pos_topk=args.pos_topk,
        w_supcon=args.w_supcon,
        w_soft=args.w_soft,
    )

    best_sil = -1.0
    best_ckpt = out_dir / "best_model.pth"

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        running_steps = 0
        running_valid = 0
        stats_acc: dict[str, float] = {}

        for batch in train_loader:
            traj = batch["traj"].to(device)
            lengths = batch["lengths"].to(device)
            feat = batch["feat"].to(device)
            feat_valid = batch.get("feat_valid", None)
            if feat_valid is not None:
                feat_valid = feat_valid.to(device)
            cond = batch.get("cond", None)
            if cond is not None:
                cond = cond.to(device)

            z = model(traj, lengths)
            # Replace hard pair contrastive with soft feature-guided contrastive loss.
            # feat is supervision signal only; it is never fed into the trajectory encoder forward path.
            loss, stats = criterion(z, feat, feat_valid=feat_valid, cond=cond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_steps += 1
            running_valid += stats["valid_anchors"]
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stats_acc[key] = stats_acc.get(key, 0.0) + float(value)

        avg_loss = running_loss / max(1, running_steps)
        avg_valid = running_valid / max(1, running_steps)
        avg_stats: dict[str, Any] = {
            k: (v / max(1, running_steps)) for k, v in stats_acc.items() if k != "valid_anchors"
        }
        epoch_msg = f"Epoch {epoch}/{total_epochs} | loss={avg_loss:.4f} | valid_anchors/batch={avg_valid:.2f}"
        if "tau_feat" in avg_stats:
            epoch_msg += f" | tau_feat={avg_stats['tau_feat']:.4f}"
        if "tau_feat_mean" in avg_stats:
            epoch_msg += f" | tau_feat_mean={avg_stats['tau_feat_mean']:.4f}"
        if "tau_feat_median" in avg_stats:
            epoch_msg += f" | tau_feat_median={avg_stats['tau_feat_median']:.4f}"
        if "sim_feat_entropy_mean" in avg_stats:
            epoch_msg += f" | entropy={avg_stats['sim_feat_entropy_mean']:.4f}"
        if "effective_k_mean" in avg_stats:
            epoch_msg += f" | effective_k={avg_stats['effective_k_mean']:.2f}"
        if "sigma_mean" in avg_stats:
            epoch_msg += f" | sigma_mean={avg_stats['sigma_mean']:.4f}"
        if "sigma_median" in avg_stats:
            epoch_msg += f" | sigma_median={avg_stats['sigma_median']:.4f}"
        if "ls_alpha" in avg_stats:
            epoch_msg += f" | ls_alpha={avg_stats['ls_alpha']:.3f}"
        if "cond_candidates_mean" in avg_stats:
            epoch_msg += f" | cond_cands={avg_stats['cond_candidates_mean']:.1f}"
        if "cond_fallback_frac" in avg_stats:
            epoch_msg += f" | cond_fallback={avg_stats['cond_fallback_frac']:.2f}"
        if "supcon_loss" in avg_stats:
            epoch_msg += f" | supcon={avg_stats['supcon_loss']:.4f}"
        if "softkl_loss" in avg_stats:
            epoch_msg += f" | softkl={avg_stats['softkl_loss']:.4f}"
        print(epoch_msg)

        if epoch % args.eval_every == 0:
            if args.skip_val_clustering:
                print("Val metrics skipped (--skip_val_clustering).")
            else:
                emb_val = encode_subset(model, val_loader, device)
                metrics_val = clustering_metrics(emb_val, args.n_clusters)
                print(
                    "Val metrics | "
                    f"sil={metrics_val['sil']:.4f} "
                    f"ch={metrics_val['ch']:.2f} "
                    f"db={metrics_val['db']:.4f}"
                )

                if not np.isnan(metrics_val["sil"]) and metrics_val["sil"] > best_sil:
                    best_sil = metrics_val["sil"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_silhouette": best_sil,
                            "args": vars(args),
                        },
                        best_ckpt,
                    )
                    print(f"Saved best checkpoint: {best_ckpt} (sil={best_sil:.4f})")

    final_ckpt = out_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model: {final_ckpt}")
    if args.skip_val_clustering:
        torch.save(
            {
                "epoch": total_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_silhouette": -1.0,
                "val_clustering_skipped": True,
                "args": vars(args),
            },
            best_ckpt,
        )
        print(f"Saved best checkpoint fallback (skip val clustering): {best_ckpt}")

    emb_test = encode_subset(model, test_loader, device)
    metrics_test = clustering_metrics(emb_test, args.n_clusters)
    print(
        "Test metrics | "
        f"sil={metrics_test['sil']:.4f} "
        f"ch={metrics_test['ch']:.2f} "
        f"db={metrics_test['db']:.4f}"
    )

    emb_out = out_dir / "embeddings_test.npy"
    np.save(emb_out, emb_test)
    print(f"Saved test embeddings: {emb_out}")


if __name__ == "__main__":
    main()
