# Bridge CLI

Fine-tuned LLM for Java/Spring Boot coding assistance.

## Overview

This project fine-tunes an open-source code model (DeepSeek-Coder) on Spring Boot examples to create a specialized coding assistant. Using QLoRA on RunPod, the entire training costs under $5.

## Quick Start

### 1. Prepare Your RunPod Environment

1. Go to [RunPod](https://runpod.io) and create an account
2. Navigate to **Pods** → **Deploy**
3. Select a GPU:
   - **Budget option**: RTX 4090 or RTX A5000 (24GB) ~$0.50-0.75/hr
   - **Faster option**: A40 (48GB) ~$0.79/hr
4. Choose template: **RunPod Pytorch** or **Axolotl**
5. Set volume disk to at least **50GB**
6. Deploy and connect via Jupyter or SSH

### 2. Upload Training Files

```bash
# Clone this repo to your RunPod pod
git clone https://github.com/DavidJBarnes/bridge-cli.git /workspace/bridge-cli

# Or upload files manually to /workspace/
```

### 3. Install Dependencies

```bash
pip install axolotl accelerate bitsandbytes peft transformers datasets
```

### 4. Run Training

```bash
cd /workspace/bridge-cli

# Start fine-tuning
accelerate launch -m axolotl.cli.train config/axolotl-config.yaml
```

Training takes approximately **2-4 hours** on an RTX 4090.

### 5. Test Your Model

```bash
# After training completes
accelerate launch -m axolotl.cli.inference config/axolotl-config.yaml \
  --lora_model_dir /workspace/outputs/spring-boot-coder
```

### 6. Deploy with vLLM (Optional)

```bash
vllm serve deepseek-ai/deepseek-coder-6.7b-instruct \
  --enable-lora \
  --lora-modules spring-boot=/workspace/outputs/spring-boot-coder \
  --port 8000
```

## Project Structure

```
bridge-cli/
├── config/
│   └── axolotl-config.yaml    # Training configuration
├── datasets/
│   └── spring-boot-dataset.jsonl  # Training data (20 examples included)
├── scripts/
│   └── generate_dataset.py    # Generate more training data
└── README.md
```

## Expanding the Dataset

The included dataset has 20 high-quality Spring Boot examples. For better results, generate more:

```bash
# Generate from GitHub repos
python scripts/generate_dataset.py \
  --output datasets/spring-boot-extended.jsonl \
  --add-synthetic

# Add your own code
python scripts/generate_dataset.py \
  --local-dir /path/to/your/spring-boot-project \
  --output datasets/custom-dataset.jsonl
```

Then update `axolotl-config.yaml` to include the new dataset:

```yaml
datasets:
  - path: /workspace/datasets/spring-boot-dataset.jsonl
    type: alpaca
  - path: /workspace/datasets/spring-boot-extended.jsonl
    type: alpaca
```

## Configuration Options

### For Smaller GPUs (16GB)

Edit `axolotl-config.yaml`:
```yaml
micro_batch_size: 1
gradient_accumulation_steps: 8
sequence_len: 1024
```

### For Larger GPUs (48GB+)

```yaml
load_in_4bit: false
load_in_8bit: true
micro_batch_size: 4
```

### Adjusting Quality vs Speed

More epochs = better quality but longer training:
```yaml
num_epochs: 5      # Higher quality
learning_rate: 0.0001  # Lower for stability
```

## Dataset Format

Training data follows the Alpaca format:

```json
{
  "instruction": "Create a Spring Boot REST controller for User management",
  "input": "",
  "output": "@RestController\n@RequestMapping(\"/api/users\")\npublic class UserController {\n    // ... full implementation\n}"
}
```

## Estimated Costs

| GPU | Cost/hr | Training Time | Total Cost |
|-----|---------|---------------|------------|
| RTX 4090 | $0.74 | 3 hours | ~$2.25 |
| RTX A5000 | $0.50 | 4 hours | ~$2.00 |
| A40 | $0.79 | 2 hours | ~$1.60 |

## Tips for Better Results

1. **Quality over quantity**: 500 excellent examples beat 5000 mediocre ones
2. **Diverse patterns**: Include controllers, services, repos, configs, tests
3. **Real-world code**: Use production patterns, not toy examples
4. **Consistent style**: Follow Spring Boot best practices throughout

## Troubleshooting

### Out of Memory
- Reduce `micro_batch_size` to 1
- Reduce `sequence_len` to 1024
- Enable `gradient_checkpointing: true`

### Slow Training
- Ensure `flash_attention: true`
- Use `sample_packing: true`
- Check GPU utilization with `nvidia-smi`

### Poor Results
- Add more diverse training examples
- Increase `num_epochs` to 5
- Lower `learning_rate` to 0.0001

## License

MIT

## Contributing

PRs welcome! Especially:
- Additional Spring Boot training examples
- Configuration optimizations
- Deployment guides
