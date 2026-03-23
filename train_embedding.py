import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load Data
# =========================
traj = np.load("traj.npy")
feat = np.load("feat.npy")

traj = torch.tensor(traj, dtype=torch.float32)
feat = torch.tensor(feat, dtype=torch.float32)

# =========================
# Train / Val Split
# =========================
np.random.seed(42)
idx = np.random.permutation(len(traj))

train_idx = idx[:int(len(traj)*0.8)]
val_idx = idx[int(len(traj)*0.8):]

traj_train = traj[train_idx]
feat_train = feat[train_idx]

traj_val = traj[val_idx]
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