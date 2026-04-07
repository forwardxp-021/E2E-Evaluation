import argparse
from pathlib import Path

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
    parser.add_argument("--debug_sim_feat", action="store_true", help="Print first-row sim_feat values once for sanity check")
    parser.add_argument("--debug_sim_topk", type=int, default=10, help="How many sim_feat entries to print for debug")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--n_clusters", type=int, default=3)

    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_pair_masks(global_idx: torch.Tensor, pos_global: torch.Tensor, neg_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = global_idx.shape[0]
    device = global_idx.device

    idx_to_local = {int(g.item()): i for i, g in enumerate(global_idx)}

    pos_mask = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)
    neg_mask = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)

    for i in range(bsz):
        for p in pos_global[i].tolist():
            j = idx_to_local.get(int(p))
            if j is not None and j != i:
                pos_mask[i, j] = True

        for n in neg_global[i].tolist():
            j = idx_to_local.get(int(n))
            if j is not None and j != i:
                neg_mask[i, j] = True

    return pos_mask, neg_mask


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

    dataset = TrajFeatureDataset(
        traj_path=args.traj_path,
        feat_path=args.feat_path,
        split_path=args.split_path,
        k_pos=args.k_pos,
        k_neg=args.k_neg,
        metric=args.distance,
        pair_cache_path=args.pair_cache_path,
        build_pairs=False,
    )

    train_idx = dataset.indices_by_split("train")
    val_idx = dataset.indices_by_split("val")
    test_idx = dataset.indices_by_split("test")
    print(f"Split sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

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
        debug_sim=args.debug_sim_feat,
        debug_topk=args.debug_sim_topk,
    )

    best_sil = -1.0
    best_ckpt = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_steps = 0
        running_valid = 0

        for batch in train_loader:
            traj = batch["traj"].to(device)
            lengths = batch["lengths"].to(device)
            feat = batch["feat"].to(device)

            z = model(traj, lengths)
            # Replace hard pair contrastive with soft feature-guided contrastive loss.
            # feat is supervision signal only; it is never fed into the trajectory encoder forward path.
            loss, stats = criterion(z, feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_steps += 1
            running_valid += stats["valid_anchors"]

        avg_loss = running_loss / max(1, running_steps)
        avg_valid = running_valid / max(1, running_steps)
        print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | valid_anchors/batch={avg_valid:.2f}")

        if epoch % args.eval_every == 0:
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
feat_val = feat[val_idx]

print(f"Train: {len(traj_train)}, Val: {len(traj_val)}")

# =========================
# Model
# =========================
class TrajEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.traj_enc = TrajEncoder()
        self.feat_enc = FeatureEncoder(feat_dim)
        self.projector = nn.Linear(96, 32)

    def forward(self, traj, feat):
        z_traj = self.traj_enc(traj)
        z_feat = self.feat_enc(feat)

        z = torch.cat([z_traj, z_feat], dim=1)
        z = self.projector(z)

        return z


# =========================
# Loss
# =========================
def contrastive_loss(z1, z2, temperature=0.1):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0)).to(device)

    return nn.CrossEntropyLoss()(logits, labels)


def feature_structure_loss(z, feat, temperature=0.1):
    """让 embedding 空间保留特征空间的结构
    z: (batch_size, emb_dim)
    feat: (batch_size, feat_dim)
    """
    # normalize
    z = nn.functional.normalize(z, dim=1)
    feat = nn.functional.normalize(feat, dim=1)

    # similarity matrix
    sim_z = z @ z.T                  # embedding similarity
    sim_f = feat @ feat.T           # feature similarity

    # 去掉对角线（自己和自己）
    mask = torch.eye(z.size(0), dtype=torch.bool).to(z.device)

    sim_z = sim_z[~mask].view(z.size(0), -1)
    sim_f = sim_f[~mask].view(z.size(0), -1)

    # soft target（关键）
    target = nn.functional.softmax(sim_f / temperature, dim=1)

    log_prob = nn.functional.log_softmax(sim_z / temperature, dim=1)

    loss = - (target * log_prob).sum(dim=1).mean()

    return loss


# =========================
# Model Init
# =========================
model = Model(feat.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 初始化最佳模型跟踪变量
best_silhouette = -1
best_epoch = 0

# =========================
# Training
# =========================
for epoch in range(50):

    model.train()
    idx = torch.randperm(len(traj_train))

    total_loss = 0
    batch_count = 0

    for i in range(0, len(traj_train), 64):
        batch = idx[i:i+64]

        t = traj_train[batch].to(device)
        f = feat_train[batch].to(device)

        # ===== 数据增强 =====
        T = t.shape[1]

        # 1. crop
        crop_len = int(T * 0.8)
        start = np.random.randint(0, T - crop_len + 1)

        t_crop = t[:, start:start+crop_len, :]
        pad_len = T - crop_len
        pad = torch.zeros((t.shape[0], pad_len, t.shape[2]), device=device)
        t_aug1 = torch.cat([t_crop, pad], dim=1)

        # 2. scale
        scale = 0.9 + np.random.rand() * 0.2
        t_aug2 = t.clone()
        t_aug2[:, :, 2:] *= scale

        # 3. noise
        t_aug3 = t + torch.randn_like(t) * 0.01

        # ===== forward =====
        z = model(t, f)
        z1 = model(t_aug1, f)
        z2 = model(t_aug2, f)
        z3 = model(t_aug3, f)

        # ===== loss =====
        loss_contrastive1 = contrastive_loss(z, z1)
        loss_contrastive2 = contrastive_loss(z, z2)
        loss_contrastive3 = contrastive_loss(z, z3)

        loss_structure = feature_structure_loss(z, f)

        loss = (loss_contrastive1 + loss_contrastive2 + loss_contrastive3) * 0.5 + loss_structure * 1.0

        # ===== backward =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    print(f"Epoch {epoch+1}, Loss {total_loss / batch_count:.4f}")

    # =========================
    # Validation
    # =========================
    if (epoch + 1) % 2 == 0:

        model.eval()

        with torch.no_grad():
            embeds = []

            for i in range(0, len(traj_val), 64):
                t = traj_val[i:i+64].to(device)
                f = feat_val[i:i+64].to(device)

                z = model(t, f)
                z = nn.functional.normalize(z, dim=1)
                embeds.append(z.cpu().numpy())

        embeds = np.concatenate(embeds, axis=0)

        # =========================
        # KMeans (多次运行)
        # =========================
        sil_scores = []
        ch_scores = []
        db_scores = []

        for seed in range(5):
            kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(embeds)

            sil_scores.append(silhouette_score(embeds, labels))
            ch_scores.append(calinski_harabasz_score(embeds, labels))
            db_scores.append(davies_bouldin_score(embeds, labels))

        current_silhouette = np.mean(sil_scores)
        print("\nEmbedding Metrics:")
        print(f"Silhouette: {current_silhouette:.4f}")
        print(f"CH Score: {np.mean(ch_scores):.2f}")
        print(f"DB Score: {np.mean(db_scores):.4f}")

        # 保存最佳模型
        if current_silhouette > best_silhouette:
            best_silhouette = current_silhouette
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'silhouette': best_silhouette,
            }, 'best_model.pth')
            print(f"Best model saved at epoch {best_epoch} with silhouette {best_silhouette:.4f}")
        else:
            # 如果连续3次没有改进，提前停止
            if epoch + 1 - best_epoch >= 6:
                print(f"No improvement for 3 validation steps. Early stopping at epoch {epoch+1}.")
                break

        # =========================
        # Feature Baseline
        # =========================
        feat_np = feat_val.numpy()

        # 标准化
        scaler = StandardScaler()
        feat_np = scaler.fit_transform(feat_np)

        kmeans_feat = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_feat = kmeans_feat.fit_predict(feat_np)

        print("\nFeature Baseline:")
        print(f"Silhouette: {silhouette_score(feat_np, labels_feat):.4f}")
        print(f"CH Score: {calinski_harabasz_score(feat_np, labels_feat):.2f}")
        print(f"DB Score: {davies_bouldin_score(feat_np, labels_feat):.4f}")

        model.train()

# 训练结束后打印最佳模型信息
print(f"\nTraining completed.")
print(f"Best model was saved at epoch {best_epoch} with silhouette {best_silhouette:.4f}")
print(f"Best model details:")
print(f"- Epoch: {best_epoch}")
print(f"- Silhouette Score: {best_silhouette:.4f}")
print(f"- Model saved as: best_model.pth")

# =========================
# Save final model
# =========================
torch.save(model.state_dict(), "model.pth")
print("Final model saved as: model.pth")
