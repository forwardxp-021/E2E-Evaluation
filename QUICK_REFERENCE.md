# E2E-Evaluation 项目快速参考

## 项目结构
```
E2E-Evaluation/
├── build_dataset.py           # 从 Waymo 数据集构建轨迹和特征数据
├── train_embedding.py         # 训练深度学习嵌入模型
├── generate_embeddings.py     # 使用训练好的模型生成嵌入向量
├── visualize_umap.py          # UMAP 降维可视化
├── analyze_style_embedding.py # 嵌入空间分析和探针
├── data/                      # 数据目录
└── README.md                  # 项目说明
```

## 工作流

### 第1步: 数据准备
```bash
# 设置 Waymo 数据路径
export WAYMO_DATA_PATH=/path/to/waymo/data

# 从原始数据生成训练数据
python build_dataset.py
```
**输出**: `traj.npy` (轨迹), `feat.npy` (特征 - 19维)

### 第2步: 模型训练
```bash
python train_embedding.py
```
**特征**: 对比学习 + 特征结构保留  
**输出**: `best_model.pth`, `model.pth`

### 第3步: 生成嵌入向量
```bash
python generate_embeddings.py
```
**输入**: `traj.npy`, `feat.npy`, `best_model.pth`  
**输出**: `embeddings.npy` (32维)

### 第4步: 可视化分析
```bash
python visualize_umap.py
```
**输出**: `umap_visualization.png`

### 第5步: 深度分析
```bash
python analyze_style_embedding.py \
  --embedding_path embeddings.npy \
  --feature_path feat.npy \
  --output_dir results/
```

## 关键参数

### build_dataset.py
- `WINDOW = 50`: 轨迹窗口大小
- **特征维度**: 19维
  - 相对速度 (3): 均值, 标差, 正向比例
  - THW (3): 均值, 标差, 最小值
  - Jerk (3): 均值, 标差, 95分位
  - 相对加速度 (2): 均值, 标差
  - 反应时间 (1)
  - 横向 (3): 偏航率标差, 变道次数, 变道时长
  - 速度归一化 (2): 均值, 标差
  - 稳定性 (2): 速度标差, 加速度标差

### train_embedding.py
- `Model`: TrajEncoder (LSTM) + FeatureEncoder
  - TrajEncoder 输出: 64维
  - FeatureEncoder 输出: 32维
  - **最终嵌入**: 32维
- 训练: 对比学习 + 特征结构损失
- 评估: Silhouette Score, Calinski-Harabasz, Davies-Bouldin

### visualize_umap.py
- `n_neighbors=10`: UMAP 局部邻域大小
- `min_dist=0.1`: UMAP 最小距离
- 处理前100K样本

### analyze_style_embedding.py
- **分析方法**:
  1. Embedding-Feature Spearman 关联矩阵
  2. UMAP 降维可视化
  3. 邻域一致性分析 (embedding vs feature vs random)
  4. 线性探针回归 (probe)

## 数据维度总结

| 模块 | 输入维度 | 输出维度 |
|------|---------|---------|
| 轨迹 | 50×4 | - |
| 特征 | - | 19 |
| TrajEncoder | 50×4 | 64 |
| FeatureEncoder | 19 | 32 |
| Model | (64+32)=96 | 32 |
| UMAP | 32 | 2 |

## 常见问题

### Q: 如何修改特征维度?
A: 修改 `build_dataset.py` 中 `compute_features()` 函数的返回列表

### Q: 如何修改嵌入维度?
A: 修改 `train_embedding.py` 中 `Model.projector` 的 output_features

### Q: 数据路径错误怎么办?
A: 设置环境变量 `export WAYMO_DATA_PATH=/correct/path`

### Q: 如何调整模型超参数?
A: 修改 `train_embedding.py` 中的:
- 学习率: `lr=1e-3`
- 批大小: `64`
- 训练轮数: `50`
- 聚类数: `3`

## 依赖库版本

- torch == 2.10.0+cpu
- tensorflow-cpu == 2.21.0
- waymo-open-dataset
- scikit-learn >= 0.24
- numpy >= 1.19
- pandas >= 1.1
- scipy >= 1.5
- umap-learn >= 0.5
- matplotlib >= 3.3

## 最近修复 (2026-03-23)

✅ 修复 Model 维度不一致 (feat_dim 硬编码问题)  
✅ 修正特征维度标注 (20D → 19D)  
✅ 数据路径添加环境变量支持  
✅ 改进 reaction_time 异常处理  
✅ 移除无效的 plt.show() 调用

详见: [CODE_REVIEW_FIXES.md](CODE_REVIEW_FIXES.md)
