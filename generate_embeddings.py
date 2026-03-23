import torch
import numpy as np

# 定义模型结构
class TrajEncoder(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 64)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class FeatureEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )

    def forward(self, x):
        return self.net(x)

class Model(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.traj_enc = TrajEncoder()
        self.feat_enc = FeatureEncoder(feat_dim)
        self.projector = torch.nn.Linear(64 + 32, 32)

    def forward(self, traj, feat):
        z_traj = self.traj_enc(traj)
        z_feat = self.feat_enc(feat)
        z = torch.cat([z_traj, z_feat], dim=1)
        z = self.projector(z)
        return z

# 加载数据
traj = torch.tensor(np.load("traj.npy"), dtype=torch.float32)
feat = torch.tensor(np.load("feat.npy"), dtype=torch.float32)

# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载最佳模型
model = Model(feat.shape[1]).to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best model loaded from epoch {checkpoint['epoch']} with silhouette {checkpoint['silhouette']:.4f}")

# 生成嵌入向量
embeddings = []
batch_size = 128

print("Generating embeddings...")
with torch.no_grad():
    for i in range(0, len(traj), batch_size):
        t = traj[i:i+batch_size].to(device)
        f = feat[i:i+batch_size].to(device)
        z = model(t, f)
        embeddings.append(z.cpu().numpy())
        if (i + batch_size) % 10000 == 0:
            print(f"Processed {i + batch_size}/{len(traj)} samples")

embeddings = np.concatenate(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# 保存嵌入向量
np.save("embeddings.npy", embeddings)
print("Embeddings saved as embeddings.npy")