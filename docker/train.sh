#!/bin/bash
# Bridge CLI - Training Entrypoint
# Runs the full training pipeline: setup, train, and optionally test.
#
# Usage:
#   /workspace/bridge-cli/docker/train.sh              # Train only
#   /workspace/bridge-cli/docker/train.sh --test       # Train + test
#   /workspace/bridge-cli/docker/train.sh --publish    # Train + test + publish

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="/workspace/bridge-cli/config/axolotl-config.yaml"
OUTPUT_DIR="/workspace/outputs/bridge-cli"

# Ensure environment
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"

# Run setup if datasets aren't in place
if [ ! -f /workspace/datasets/spring-boot-dataset.jsonl ]; then
    echo "Running first-time setup..."
    /opt/bridge-cli/setup.sh
fi

echo "=== Bridge CLI Training ==="
echo "Config:  $CONFIG"
echo "Output:  $OUTPUT_DIR"
echo ""

# Train
echo "Starting training..."
accelerate launch -m axolotl.cli.train "$CONFIG"

echo ""
echo "Training complete. Adapter saved to: $OUTPUT_DIR"

# Test if requested
if [[ "${1:-}" == "--test" || "${1:-}" == "--publish" ]]; then
    echo ""
    echo "=== Running Model Tests ==="
    python3 /workspace/bridge-cli/scripts/test_model.py \
        --lora-path "$OUTPUT_DIR"
fi

# Publish if requested
if [[ "${1:-}" == "--publish" ]]; then
    echo ""
    echo "=== Publishing Model ==="
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: Set HF_TOKEN environment variable before publishing."
        echo "  export HF_TOKEN=hf_your_token_here"
        exit 1
    fi
    python3 /workspace/bridge-cli/scripts/publish_model.py \
        --repo-id DavidJBarnes/bridge-cli \
        --adapter-path "$OUTPUT_DIR"
fi

echo ""
echo "=== Done ==="
