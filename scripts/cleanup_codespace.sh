#!/usr/bin/env bash
# =============================================================================
# cleanup_codespace.sh — Free up disk space in the "animated space orbit"
#                         GitHub Codespace (or any Linux dev environment).
#
# Usage:
#   bash scripts/cleanup_codespace.sh          # interactive (confirm each step)
#   bash scripts/cleanup_codespace.sh --yes    # skip all confirmations
# =============================================================================

set -euo pipefail

YES=false
[[ "${1:-}" == "--yes" ]] && YES=true

confirm() {
    if $YES; then
        return 0
    fi
    read -r -p "$1 [y/N] " ans
    [[ "${ans,,}" == "y" ]]
}

hr() { printf '\n%s\n' "────────────────────────────────────────────────"; }

# ── Helper: print before/after size ──────────────────────────────────────────
print_size() { df -h / | awk 'NR==2 {printf "  Disk used: %s / %s (%s free)\n", $3, $2, $4}'; }

echo "=== Codespace Space Cleanup ==="
print_size

# 1. pip cache ─────────────────────────────────────────────────────────────────
hr
echo "[1/8] pip cache"
pip cache info 2>/dev/null || true
if confirm "  Clear pip cache?"; then
    pip cache purge
    echo "  Done."
fi

# 2. apt cache ─────────────────────────────────────────────────────────────────
hr
echo "[2/8] apt / dpkg cache"
if confirm "  Clear apt cache (sudo required)?"; then
    sudo apt-get clean -y
    sudo rm -rf /var/lib/apt/lists/*
    echo "  Done."
fi

# 3. Conda / mamba cache ───────────────────────────────────────────────────────
hr
echo "[3/8] Conda / mamba package cache"
if command -v conda &>/dev/null; then
    if confirm "  Clean conda package cache?"; then
        conda clean --all --yes
        echo "  Done."
    fi
else
    echo "  conda not found, skipping."
fi

# 4. Docker images & build cache ──────────────────────────────────────────────
hr
echo "[4/8] Docker images & build cache"
if command -v docker &>/dev/null; then
    docker system df 2>/dev/null || true
    if confirm "  Prune all unused Docker images and build cache?"; then
        docker system prune -af
        echo "  Done."
    fi
else
    echo "  docker not found, skipping."
fi

# 5. Jupyter / IPython caches ─────────────────────────────────────────────────
hr
echo "[5/8] Jupyter / IPython caches"
JUPYTER_DIRS=(
    "$HOME/.jupyter/lab/workspaces"
    "$HOME/.local/share/jupyter/nbconvert"
    "$HOME/.ipython/profile_default/history.sqlite"
)
for d in "${JUPYTER_DIRS[@]}"; do
    if [[ -e "$d" ]]; then
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        if confirm "  Remove $d ($SIZE)?"; then
            rm -rf "$d"
            echo "  Removed $d"
        fi
    fi
done

# 6. Large files in data/ not tracked by git ──────────────────────────────────
hr
echo "[6/8] Untracked large files in data/ (>50 MB)"
if [[ -d data ]]; then
    find data -type f -size +50M 2>/dev/null | while read -r f; do
        SIZE=$(du -sh "$f" | cut -f1)
        if confirm "  Delete $f ($SIZE)?"; then
            rm -f "$f"
            echo "  Deleted $f"
        fi
    done
else
    echo "  No data/ directory found."
fi

# 7. Model / checkpoint files in the workspace ─────────────────────────────────
hr
echo "[7/8] Model / checkpoint files in workspace (*.pt *.pth *.ckpt *.bin *.onnx)"
find . -not -path './.git/*' \
    \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \
       -o -name "*.bin" -o -name "*.onnx" -o -name "*.h5" \) \
    -size +10M 2>/dev/null | while read -r f; do
    SIZE=$(du -sh "$f" | cut -f1)
    if confirm "  Delete $f ($SIZE)?"; then
        rm -f "$f"
        echo "  Deleted $f"
    fi
done

# 8. Temporary / log files ────────────────────────────────────────────────────
hr
echo "[8/8] Log and temp files"
LOG_DIRS=("logs" "runs" "wandb" "mlruns" "outputs" "tmp" "/tmp/codespace-*")
for d in "${LOG_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        if confirm "  Remove $d/ ($SIZE)?"; then
            rm -rf "$d"
            echo "  Removed $d/"
        fi
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
hr
echo "=== Cleanup complete ==="
print_size
echo
echo "Tip: rebuild the Codespace ('Codespaces: Rebuild Container' in VS Code)"
echo "     or use a fresh Codespace to get the maximum space savings."
