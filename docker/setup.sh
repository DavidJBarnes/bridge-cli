#!/bin/bash
# Bridge CLI - RunPod Setup Script
# Run this once after pod starts to prepare the workspace.
# Copies datasets, creates directories, and optionally pre-downloads the model.
#
# Usage: /opt/bridge-cli/setup.sh [--download-model]

set -euo pipefail

echo "=== Bridge CLI Training Setup ==="

# Create workspace directories
mkdir -p /workspace/bridge-cli/config \
         /workspace/bridge-cli/datasets \
         /workspace/bridge-cli/scripts \
         /workspace/datasets \
         /workspace/outputs \
         /workspace/prepared_data \
         /workspace/.cache/huggingface \
         /workspace/tmp

# Copy project files to workspace (preserves any local modifications)
cp -n /opt/bridge-cli/config/* /workspace/bridge-cli/config/ 2>/dev/null || true
cp -n /opt/bridge-cli/datasets/* /workspace/bridge-cli/datasets/ 2>/dev/null || true
cp -n /opt/bridge-cli/scripts/* /workspace/bridge-cli/scripts/ 2>/dev/null || true
cp -n /opt/bridge-cli/train.sh /workspace/bridge-cli/train.sh 2>/dev/null || true

# Copy datasets to the path expected by axolotl config
cp /opt/bridge-cli/datasets/*.jsonl /workspace/datasets/

# Symlink HF cache to workspace if not already done
if [ ! -L /root/.cache/huggingface ] && [ ! -d /root/.cache/huggingface ]; then
    mkdir -p /root/.cache
    ln -sf /workspace/.cache/huggingface /root/.cache/huggingface
fi

echo "Workspace prepared."
echo "  Config:   /workspace/bridge-cli/config/"
echo "  Datasets: /workspace/datasets/"
echo "  Outputs:  /workspace/outputs/"

# Optionally pre-download the base model
if [[ "${1:-}" == "--download-model" ]]; then
    echo ""
    echo "Downloading base model (deepseek-ai/deepseek-coder-6.7b-instruct)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/deepseek-coder-6.7b-instruct', cache_dir='/workspace/.cache/huggingface')
print('Model download complete.')
"
fi

echo ""
echo "=== Setup Complete ==="
echo "Run training with: /workspace/bridge-cli/train.sh"
