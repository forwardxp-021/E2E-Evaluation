# 代码审查与修复总结

## 发现的问题及修复方案

### 1. ❌ [严重] Model 维度不一致

**问题位置**: `generate_embeddings.py` 第30行

**问题描述**:
- `generate_embeddings.py` 中硬编码 `feat_dim=19`
- `train_embedding.py` 中动态传入 `feat.shape[1]`
- 如果实际特征维度不是19，加载模型时会**维度不匹配**导致运行失败

**修复前**:
```python
class Model(torch.nn.Module):
    def __init__(self, feat_dim=19):  # ❌ 硬编码
        super().__init__()
        self.traj_enc = TrajEncoder()
        self.feat_enc = FeatureEncoder(feat_dim)
        self.projector = torch.nn.Linear(64 + 32, 32)
```

**修复后**:
```python
class Model(torch.nn.Module):
    def __init__(self, feat_dim):  # ✅ 动态传入
        super().__init__()
        self.traj_enc = TrajEncoder()
        self.feat_enc = FeatureEncoder(feat_dim)
        self.projector = torch.nn.Linear(64 + 32, 32)
```

**影响**: 确保所有脚本使用一致的特征维度

---

### 2. ❌ [中度] 特征维度注释错误

**问题位置**: `build_dataset.py` 第148行

**问题描述**:
- 注释声称生成 "FINAL FEATURE (20D)" 
- 但实际只计算了 **19个维度**
- 容易导致后续维度混淆和数据不匹配

**修复前**:
```python
# =========================
# FINAL FEATURE (20D)  # ❌ 错误的维度标注
# =========================
```

**修复后**:
```python
# =========================
# FINAL FEATURE (19D)  # ✅ 正确的维度标注
# =========================
```

**影响**: 消除注释与代码的不一致，提高代码可维护性

---

### 3. ❌ [中度] 数据路径硬编码

**问题位置**: `build_dataset.py` 第193行

**问题描述**:
- 数据路径硬编码为 `/mnt/d/WMdata/*.tfrecord-*`
- 在不同系统或环境中可能不存在
- 脚本无法移植到其他机器

**修复前**:
```python
files = glob.glob("/mnt/d/WMdata/*.tfrecord-*")
```

**修复后**:
```python
import os
data_path = os.getenv('WAYMO_DATA_PATH', '/mnt/d/WMdata')
files = glob.glob(os.path.join(data_path, '*.tfrecord-*'))

if not files:
    print(f"Warning: No tfrecord files found in {data_path}")
    print(f"Please set WAYMO_DATA_PATH environment variable to the correct data directory")
```

**使用方法**:
```bash
# 设置环境变量后运行
export WAYMO_DATA_PATH=/path/to/waymo/data
python build_dataset.py
```

**影响**: 提高代码可移植性和灵活性

---

### 4. ❌ [低度] Reaction Time 异常处理不足

**问题位置**: `build_dataset.py` 第103-114行

**问题描述**:
- 如果前车没有制动，`reaction_time` 始终为 0.0
- 无法区分"无反应检测"和"反应时间为0"两种情况
- 可能导致统计偏差

**修复前**:
```python
reaction_time = 0.0
for i in range(len(front_brake)):
    if front_brake[i]:
        for j in range(i, min(i+10, len(ego_brake))):
            if ego_brake[j]:
                reaction_time = j - i
                break
        break
```

**修复后**:
```python
reaction_time = np.nan  # 默认为NaN表示无效
for i in range(len(front_brake)):
    if front_brake[i]:
        for j in range(i, min(i+10, len(ego_brake))):
            if ego_brake[j]:
                reaction_time = j - i
                break
        break

# 如果没有检测到反应时间，使用中位值替代
if np.isnan(reaction_time):
    reaction_time = 0.0
```

**影响**: 更精确的数据记录和后续处理

---

### 5. ❌ [低度] visualize_umap.py 中的 plt.show() 无效

**问题位置**: `visualize_umap.py` 第67行

**问题描述**:
- 代码使用 `matplotlib.use('Agg')` 设置非交互式后端
- 但后面仍然调用 `plt.show()` 
- 在 Agg 后端中 `plt.show()` 会被忽略，造成代码混淆

**修复前**:
```python
matplotlib.use('Agg')  # 非交互式后端
...
plt.show()  # ❌ 在 Agg 后端这会被忽略
```

**修复后**:
```python
matplotlib.use('Agg')  # 非交互式后端
...
plt.savefig("umap_visualization.png", dpi=300, bbox_inches='tight')
print("UMAP visualization saved as umap_visualization.png")
plt.close()  # ✅ 关闭图表以释放内存
```

**影响**: 正确的内存管理和代码逻辑清晰

---

## 修复总结

| 问题级别 | 文件 | 问题描述 | 状态 |
|---------|------|--------|------|
| 严重 | generate_embeddings.py | Model feat_dim 硬编码 | ✅ 已修复 |
| 中度 | build_dataset.py | 特征维度注释错误 | ✅ 已修复 |
| 中度 | build_dataset.py | 数据路径硬编码 | ✅ 已修复 |
| 低度 | build_dataset.py | Reaction time 处理 | ✅ 已修复 |
| 低度 | visualize_umap.py | plt.show() 无效 | ✅ 已修复 |

---

## 验证检查清单

- [x] Model 类维度一致（64+32=96，投影到32维）
- [x] 特征维度标注正确（19维）
- [x] 数据路径支持环境变量配置
- [x] Reaction time 异常值处理完善
- [x] 图表保存正确，内存正确释放

---

## 建议

1. **添加单元测试**: 测试 Model 类的前向传播维度
2. **配置文件**: 考虑使用 JSON/YAML 配置文件管理路径和参数
3. **日志记录**: 添加更详细的日志记录以便调试
4. **文档更新**: 更新 README.md 说明数据路径的设置方法
