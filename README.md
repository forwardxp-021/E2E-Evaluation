# E2E-Evaluation

E2E Evaluation research in automated driving

## 预期节省空间 / Expected Space Savings

下表列出了上次优化（针对"animated space orbit"Codespace，初始大小 **14.8 GB**）每项措施预计节省的磁盘空间。

| 措施 | 节省方式 | 预计节省 |
|------|---------|---------|
| 切换到 `python:3.11-slim` 基础镜像（重建 Codespace 后生效）| 默认 universal 镜像 ≈ 10–12 GB；slim 镜像 ≈ 300 MB | **~10 GB** |
| `pip cache purge`（`cleanup_codespace.sh` 步骤 1）| 清除已下载的 wheel 缓存 | ~0.5–2 GB |
| `apt-get clean`（步骤 2）| 清除 deb 包缓存 | ~0.3–1 GB |
| Docker 镜像 prune（步骤 4，若已安装 Docker）| 移除悬空镜像和构建缓存 | 0–2 GB |
| Jupyter / IPython 缓存（步骤 5）| 移除 notebook 工作区快照 | 0–0.5 GB |
| 大文件 / 模型权重（步骤 6–7）| 项目数据集与模型文件 | 视情况而定 |
| `.gitignore` 防止大文件进入 git 历史 | 避免未来膨胀 | 长期维护 |

**总结：**

- **仅运行 `cleanup_codespace.sh`（不重建）**：可释放约 **1–5 GB**，具体取决于缓存积累情况。  
- **重建 Codespace（使用新的 `devcontainer.json`）**：从 14.8 GB 降至约 **2–4 GB**，节省约 **10–13 GB（~70–85%）**。

> 推荐路径：先运行清理脚本，再在 VS Code 中执行 **Codespaces: Rebuild Container**，或直接从本分支新建 Codespace。

---

## Codespace / Dev Container

This repository ships a minimal `.devcontainer` configuration so that GitHub
Codespaces stays as lean as possible:

- **Base image**: `mcr.microsoft.com/devcontainers/python:3.11-slim`
- **No heavy ML libraries are pre-installed** – add only what you need to
  `requirements.txt` and they will be installed automatically on first launch.

### Freeing up space in an existing Codespace

If your Codespace is consuming too much disk space, run the interactive cleanup
script from the repository root:

```bash
bash scripts/cleanup_codespace.sh
```

Pass `--yes` to skip all confirmation prompts:

```bash
bash scripts/cleanup_codespace.sh --yes
```

The script will offer to:

1. Clear the **pip** package cache
2. Clear the **apt** package cache
3. Clean **conda / mamba** caches (if installed)
4. Prune unused **Docker** images and build cache (if installed)
5. Remove **Jupyter / IPython** workspace caches
6. Delete large untracked files in `data/` (> 50 MB)
7. Delete large model / checkpoint files (`*.pt`, `*.pth`, `*.ckpt`, …)
8. Remove log and temporary directories (`logs/`, `runs/`, `wandb/`, …)

> **Tip:** For maximum savings, rebuild the Codespace after cleanup
> (*Codespaces: Rebuild Container* in VS Code) or create a fresh Codespace.

## Repository layout

```
.
├── data/          # Local dataset files (ignored by git – see .gitignore)
├── scripts/
│   └── cleanup_codespace.sh   # Disk-space cleanup helper
└── .devcontainer/
    └── devcontainer.json      # Minimal Codespace configuration
```
