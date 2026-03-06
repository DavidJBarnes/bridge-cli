# Bridge CLI Training Container
# Pre-configured environment for fine-tuning DeepSeek-Coder with Axolotl on RunPod.
# All dependencies are pinned and tested together to eliminate version conflicts.
#
# Usage on RunPod:
#   Template: davidjbarnes/bridge-cli-training:latest
#   Volume: 100GB mounted at /workspace
#   Then: /workspace/bridge-cli/docker/train.sh
#
# Build locally:
#   docker build -t davidjbarnes/bridge-cli-training -f docker/Dockerfile .

FROM winglian/axolotl:main

LABEL maintainer="DavidJBarnes"
LABEL description="Bridge CLI - Fine-tuning environment for Java/React code model"

# Set environment variables for RunPod compatibility
ENV HF_HOME=/workspace/.cache/huggingface \
    TMPDIR=/workspace/tmp \
    PYTHONUNBUFFERED=1 \
    AXOLOTL_DISABLE_TELEMETRY=1

# Install additional utilities
RUN pip install --no-cache-dir huggingface_hub && \
    pip cache purge

# Copy project files
COPY config/ /opt/bridge-cli/config/
COPY datasets/ /opt/bridge-cli/datasets/
COPY scripts/ /opt/bridge-cli/scripts/
COPY docker/train.sh /opt/bridge-cli/train.sh
COPY docker/setup.sh /opt/bridge-cli/setup.sh
RUN chmod +x /opt/bridge-cli/train.sh /opt/bridge-cli/setup.sh

# Default working directory
WORKDIR /workspace

# On RunPod, the entrypoint is managed by the platform.
# For local testing, default to bash.
CMD ["/bin/bash"]
