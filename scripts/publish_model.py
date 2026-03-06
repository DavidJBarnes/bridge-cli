#!/usr/bin/env python3
"""
Model publishing script for Bridge CLI.

Merges the QLoRA adapter with the base model, quantizes to GGUF formats,
and publishes all artifacts to Hugging Face Hub. Ships three artifacts:

1. LoRA adapter only (lightweight, for vLLM/advanced users)
2. Merged GGUF Q8 (high quality, ~7GB)
3. Merged GGUF Q4_K_M (compact, ~4GB, runs on laptops)

Usage:
    python scripts/publish_model.py --repo-id DavidJBarnes/bridge-cli
    python scripts/publish_model.py --repo-id DavidJBarnes/bridge-cli --adapter-only
    python scripts/publish_model.py --repo-id DavidJBarnes/bridge-cli --skip-quantize
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "merge_adapter",
    "convert_to_gguf",
    "quantize_gguf",
    "create_model_card",
    "publish_to_hub",
    "main",
]

logger = logging.getLogger(__name__)


def merge_adapter(
    base_model: str,
    adapter_path: str,
    output_dir: str,
) -> Path:
    """Merge a QLoRA adapter into the base model to produce a standalone model.

    Loads the base model in FP16, applies the LoRA adapter weights, and saves
    the merged result as a standard Hugging Face model directory.

    Args:
        base_model: Hugging Face model ID or local path for the base model.
        adapter_path: Path to the trained LoRA adapter directory.
        output_dir: Directory to save the merged model.

    Returns:
        Path to the merged model directory.

    Raises:
        ImportError: If required libraries are not installed.
        RuntimeError: If model loading or merging fails.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Required packages not installed. Run: "
            "pip install torch transformers peft accelerate bitsandbytes"
        ) from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete.")
    return output_path


def convert_to_gguf(
    model_dir: str,
    output_file: str,
    llama_cpp_path: Optional[str] = None,
) -> Path:
    """Convert a Hugging Face model directory to GGUF FP16 format.

    Uses llama.cpp's convert_hf_to_gguf.py script to produce the
    intermediate FP16 GGUF file used as input for quantization.

    Args:
        model_dir: Path to the merged Hugging Face model directory.
        output_file: Path for the output GGUF file.
        llama_cpp_path: Path to the llama.cpp repository. If None, assumes
            llama.cpp is cloned at /workspace/llama.cpp.

    Returns:
        Path to the generated GGUF file.

    Raises:
        FileNotFoundError: If llama.cpp conversion script is not found.
        subprocess.CalledProcessError: If conversion fails.
    """
    if llama_cpp_path is None:
        llama_cpp_path = "/workspace/llama.cpp"

    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found at {convert_script}. "
            f"Clone llama.cpp first: git clone https://github.com/ggerganov/llama.cpp {llama_cpp_path}"
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting to GGUF FP16: %s -> %s", model_dir, output_file)
    cmd = [
        sys.executable,
        str(convert_script),
        model_dir,
        "--outfile",
        str(output_path),
        "--outtype",
        "f16",
    ]
    subprocess.run(cmd, check=True)

    logger.info("GGUF FP16 conversion complete: %s", output_path)
    return output_path


def quantize_gguf(
    input_gguf: str,
    output_gguf: str,
    quantization: str = "Q4_K_M",
    llama_cpp_path: Optional[str] = None,
) -> Path:
    """Quantize a GGUF FP16 file to a smaller quantization format.

    Uses llama.cpp's llama-quantize binary to reduce model size while
    preserving quality.

    Args:
        input_gguf: Path to the FP16 GGUF file.
        output_gguf: Path for the quantized output file.
        quantization: Quantization type (e.g., Q4_K_M, Q8_0).
        llama_cpp_path: Path to the llama.cpp repository. If None, assumes
            llama.cpp is at /workspace/llama.cpp.

    Returns:
        Path to the quantized GGUF file.

    Raises:
        FileNotFoundError: If llama-quantize binary is not found.
        subprocess.CalledProcessError: If quantization fails.
    """
    if llama_cpp_path is None:
        llama_cpp_path = "/workspace/llama.cpp"

    quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        # Try alternate location
        quantize_bin = Path(llama_cpp_path) / "llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize not found. Build llama.cpp first: "
            f"cd {llama_cpp_path} && cmake -B build && cmake --build build --target llama-quantize"
        )

    output_path = Path(output_gguf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing %s -> %s (%s)", input_gguf, output_gguf, quantization)
    cmd = [str(quantize_bin), input_gguf, str(output_path), quantization]
    subprocess.run(cmd, check=True)

    logger.info("Quantization complete: %s", output_path)
    return output_path


def create_model_card(
    repo_id: str,
    base_model: str,
    output_path: str,
) -> Path:
    """Generate a MODEL_CARD.md for the Hugging Face repository.

    Creates a standardized model card documenting the fine-tuning process,
    intended use, training data, and usage instructions.

    Args:
        repo_id: Hugging Face repository ID (e.g., DavidJBarnes/bridge-cli).
        base_model: Name of the base model used for fine-tuning.
        output_path: File path to write the model card.

    Returns:
        Path to the generated model card file.
    """
    card_path = Path(output_path)
    card_path.parent.mkdir(parents=True, exist_ok=True)

    card_content = f"""---
language:
  - en
license: mit
tags:
  - code
  - java
  - spring-boot
  - react
  - typescript
  - fine-tuned
  - qlora
base_model: {base_model}
model_type: causal-lm
pipeline_tag: text-generation
---

# Bridge CLI - Java & React Code Assistant

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) using QLoRA
for specialized Java/Spring Boot and React/TypeScript code generation.

## Model Variants

| Variant | Size | Use Case |
|---------|------|----------|
| `adapter/` | ~200MB | LoRA adapter for vLLM or PEFT loading |
| `bridge-cli-Q8_0.gguf` | ~7GB | High-quality local inference |
| `bridge-cli-Q4_K_M.gguf` | ~4GB | Laptop-friendly, runs with Ollama |

## Usage

### With Transformers + PEFT (Adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base_model}", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}", subfolder="adapter")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

### With vLLM (Adapter)

```bash
vllm serve {base_model} \\
  --enable-lora \\
  --lora-modules bridge-cli={repo_id}/adapter \\
  --port 8000
```

### With Ollama (GGUF)

```bash
# Create Modelfile
echo 'FROM ./bridge-cli-Q4_K_M.gguf' > Modelfile
ollama create bridge-cli -f Modelfile
ollama run bridge-cli "Create a Spring Boot REST controller for User management"
```

### With llama.cpp (GGUF)

```bash
./llama-cli -m bridge-cli-Q4_K_M.gguf \\
  -p "### Instruction:\\nCreate a React hook for authentication\\n\\n### Response:\\n" \\
  -n 2048
```

## Training Details

- **Method**: QLoRA (4-bit quantization, LoRA rank 16, alpha 32)
- **Hardware**: RunPod GPU (RTX 4090 / A40)
- **Data**: Curated Java/Spring Boot and React/TypeScript instruction-response pairs
- **Format**: Alpaca (instruction / input / output)
- **Epochs**: 3
- **Optimizer**: AdamW 8-bit with cosine LR schedule

## Intended Use

Code generation and assistance for:
- Java/Spring Boot: REST controllers, services, repositories, entities, security, caching, scheduling, WebSocket, Kafka, testing
- React/TypeScript: functional components, custom hooks, context providers, Redux, routing, API services, testing

## Limitations

- Trained on specific patterns; may not generalize to all Java/React code styles
- Best results with Alpaca-format prompts (instruction/response)
- Not a replacement for code review

## License

MIT
"""

    card_path.write_text(card_content)
    logger.info("Model card written to: %s", card_path)
    return card_path


def publish_to_hub(
    repo_id: str,
    adapter_path: str,
    merged_dir: Optional[str] = None,
    gguf_files: Optional[list] = None,
    model_card_path: Optional[str] = None,
) -> str:
    """Upload all model artifacts to Hugging Face Hub.

    Publishes the LoRA adapter, merged GGUF files, and model card to a
    single Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID (e.g., DavidJBarnes/bridge-cli).
        adapter_path: Path to the LoRA adapter directory.
        merged_dir: Path to the merged model directory (optional).
        gguf_files: List of GGUF file paths to upload (optional).
        model_card_path: Path to the model card file (optional).

    Returns:
        URL of the published Hugging Face repository.

    Raises:
        ImportError: If huggingface_hub is not installed.
        Exception: If upload fails.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        ) from exc

    api = HfApi()

    logger.info("Creating/verifying repo: %s", repo_id)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    # Upload model card
    if model_card_path and Path(model_card_path).exists():
        logger.info("Uploading model card...")
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
        )

    # Upload adapter
    logger.info("Uploading LoRA adapter from: %s", adapter_path)
    api.upload_folder(
        folder_path=adapter_path,
        path_in_repo="adapter",
        repo_id=repo_id,
    )

    # Upload GGUF files
    if gguf_files:
        for gguf_path in gguf_files:
            if Path(gguf_path).exists():
                filename = Path(gguf_path).name
                logger.info("Uploading GGUF: %s", filename)
                api.upload_file(
                    path_or_fileobj=gguf_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                )
            else:
                logger.warning("GGUF file not found, skipping: %s", gguf_path)

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info("Published to: %s", repo_url)
    return repo_url


def main():
    """Entry point for the model publishing CLI.

    Parses command-line arguments and orchestrates the full publish pipeline:
    merge adapter, convert to GGUF, quantize, generate model card, and upload
    to Hugging Face Hub.
    """
    parser = argparse.ArgumentParser(
        description="Publish Bridge CLI model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo ID (e.g., DavidJBarnes/bridge-cli)",
    )
    parser.add_argument(
        "--base-model",
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        default="/workspace/outputs/spring-boot-coder",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/publish",
        help="Working directory for merge and conversion artifacts",
    )
    parser.add_argument(
        "--llama-cpp-path",
        default="/workspace/llama.cpp",
        help="Path to llama.cpp repository",
    )
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Only publish the LoRA adapter (skip merge and quantize)",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Publish merged model but skip GGUF quantization",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run merge and quantize but do not upload to Hub",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged"
    gguf_dir = output_dir / "gguf"
    gguf_files = []

    # Step 1: Merge adapter (unless adapter-only)
    if not args.adapter_only:
        merge_adapter(args.base_model, args.adapter_path, str(merged_dir))

        # Step 2: Convert to GGUF FP16
        if not args.skip_quantize:
            fp16_gguf = str(gguf_dir / "bridge-cli-f16.gguf")
            convert_to_gguf(str(merged_dir), fp16_gguf, args.llama_cpp_path)

            # Step 3: Quantize to Q8_0 and Q4_K_M
            q8_gguf = str(gguf_dir / "bridge-cli-Q8_0.gguf")
            quantize_gguf(fp16_gguf, q8_gguf, "Q8_0", args.llama_cpp_path)
            gguf_files.append(q8_gguf)

            q4_gguf = str(gguf_dir / "bridge-cli-Q4_K_M.gguf")
            quantize_gguf(fp16_gguf, q4_gguf, "Q4_K_M", args.llama_cpp_path)
            gguf_files.append(q4_gguf)

            # Clean up FP16 intermediate
            fp16_path = Path(fp16_gguf)
            if fp16_path.exists():
                fp16_path.unlink()
                logger.info("Cleaned up intermediate FP16 GGUF")

    # Step 4: Generate model card
    model_card_path = str(output_dir / "MODEL_CARD.md")
    create_model_card(args.repo_id, args.base_model, model_card_path)

    # Step 5: Publish
    if not args.skip_upload:
        repo_url = publish_to_hub(
            repo_id=args.repo_id,
            adapter_path=args.adapter_path,
            merged_dir=str(merged_dir) if not args.adapter_only else None,
            gguf_files=gguf_files,
            model_card_path=model_card_path,
        )
        logger.info("Done! Model available at: %s", repo_url)
    else:
        logger.info("Skipped upload. Artifacts saved to: %s", output_dir)


if __name__ == "__main__":
    main()
