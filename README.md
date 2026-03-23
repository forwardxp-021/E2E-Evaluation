# E2E-Evaluation

E2E Evaluation research in automated driving

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
