"""Tests for the model publishing script.

Covers merge_adapter, convert_to_gguf, quantize_gguf, create_model_card,
publish_to_hub, and CLI argument parsing. All external dependencies
(torch, transformers, peft, huggingface_hub, subprocess) are mocked.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing the module under test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_ml_libs(monkeypatch):
    """Mock torch, transformers, peft, and huggingface_hub globally."""
    mock_torch = MagicMock()
    mock_torch.float16 = "float16"

    mock_transformers = MagicMock()
    mock_peft = MagicMock()
    mock_hf_hub = MagicMock()

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
    monkeypatch.setitem(sys.modules, "peft", mock_peft)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf_hub)

    # Re-import so the module picks up mocks
    import importlib
    if "scripts.publish_model" in sys.modules:
        importlib.reload(sys.modules["scripts.publish_model"])


from scripts.publish_model import (
    merge_adapter,
    convert_to_gguf,
    quantize_gguf,
    create_model_card,
    publish_to_hub,
    main,
)


# ===========================================================================
# merge_adapter
# ===========================================================================
class TestMergeAdapter:
    """Tests for merging a LoRA adapter into the base model."""

    def test_merge_creates_output_dir(self, tmp_path):
        """Verify merge_adapter creates the output directory and saves model."""
        output_dir = tmp_path / "merged"
        result = merge_adapter("base-model", "/fake/adapter", str(output_dir))
        assert result == output_dir
        assert output_dir.exists()

    def test_merge_raises_import_error(self, tmp_path, monkeypatch):
        """Verify ImportError is raised when torch/transformers/peft missing."""
        import importlib

        # Temporarily remove torch so the local import fails
        monkeypatch.delitem(sys.modules, "torch")
        monkeypatch.setitem(sys.modules, "torch", None)
        importlib.reload(sys.modules["scripts.publish_model"])
        from scripts.publish_model import merge_adapter as merge_fresh

        with pytest.raises(ImportError, match="Required packages not installed"):
            merge_fresh("base", "/fake", str(tmp_path / "out"))

    def test_merge_calls_transformers_and_peft(self, tmp_path):
        """Verify merge_adapter loads base model, applies adapter, and saves."""
        import transformers
        import peft

        output_dir = tmp_path / "merged"
        merge_adapter("base-model", "/fake/adapter", str(output_dir))

        transformers.AutoTokenizer.from_pretrained.assert_called_once()
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        peft.PeftModel.from_pretrained.assert_called_once()

    def test_merge_calls_merge_and_unload(self, tmp_path):
        """Verify the adapter is merged and unloaded before saving."""
        import peft

        output_dir = tmp_path / "merged"
        mock_model = MagicMock()
        peft.PeftModel.from_pretrained.return_value = mock_model

        merge_adapter("base-model", "/fake/adapter", str(output_dir))
        mock_model.merge_and_unload.assert_called_once()

    def test_merge_saves_model_and_tokenizer(self, tmp_path):
        """Verify both model and tokenizer are saved to output directory."""
        import transformers
        import peft

        output_dir = tmp_path / "merged"
        mock_merged = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged
        peft.PeftModel.from_pretrained.return_value = mock_peft_model

        merge_adapter("base-model", "/fake/adapter", str(output_dir))
        mock_merged.save_pretrained.assert_called_once_with(output_dir)
        transformers.AutoTokenizer.from_pretrained.return_value.save_pretrained.assert_called_once_with(
            output_dir
        )


# ===========================================================================
# convert_to_gguf
# ===========================================================================
class TestConvertToGguf:
    """Tests for GGUF FP16 conversion."""

    def test_raises_if_script_not_found(self, tmp_path):
        """Verify FileNotFoundError when llama.cpp convert script is missing."""
        with pytest.raises(FileNotFoundError, match="convert script not found"):
            convert_to_gguf("/fake/model", str(tmp_path / "out.gguf"), str(tmp_path))

    def test_successful_conversion(self, tmp_path):
        """Verify conversion calls subprocess with correct arguments."""
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        convert_script = llama_dir / "convert_hf_to_gguf.py"
        convert_script.write_text("# stub")

        output_file = tmp_path / "output" / "model.gguf"

        with patch("scripts.publish_model.subprocess.run") as mock_run:
            result = convert_to_gguf(
                "/fake/model", str(output_file), str(llama_dir)
            )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "/fake/model" in cmd
        assert str(output_file) in cmd
        assert "f16" in cmd
        assert result == output_file
        assert output_file.parent.exists()

    def test_uses_default_llama_cpp_path(self):
        """Verify default llama.cpp path is /workspace/llama.cpp."""
        with pytest.raises(FileNotFoundError, match="/workspace/llama.cpp"):
            convert_to_gguf("/fake/model", "/fake/out.gguf")


# ===========================================================================
# quantize_gguf
# ===========================================================================
class TestQuantizeGguf:
    """Tests for GGUF quantization."""

    def test_raises_if_quantize_binary_not_found(self, tmp_path):
        """Verify FileNotFoundError when llama-quantize is missing."""
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="llama-quantize not found"):
            quantize_gguf(
                "/fake/input.gguf",
                str(tmp_path / "out.gguf"),
                "Q4_K_M",
                str(llama_dir),
            )

    def test_successful_quantization_build_bin(self, tmp_path):
        """Verify quantization with binary at build/bin/llama-quantize."""
        llama_dir = tmp_path / "llama.cpp"
        bin_dir = llama_dir / "build" / "bin"
        bin_dir.mkdir(parents=True)
        quantize_bin = bin_dir / "llama-quantize"
        quantize_bin.write_text("# stub")

        output_file = tmp_path / "quantized.gguf"

        with patch("scripts.publish_model.subprocess.run") as mock_run:
            result = quantize_gguf(
                "/fake/input.gguf", str(output_file), "Q8_0", str(llama_dir)
            )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "/fake/input.gguf" in cmd
        assert "Q8_0" in cmd
        assert result == output_file

    def test_successful_quantization_alt_location(self, tmp_path):
        """Verify quantization with binary at llama.cpp/llama-quantize."""
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        quantize_bin = llama_dir / "llama-quantize"
        quantize_bin.write_text("# stub")

        output_file = tmp_path / "quantized.gguf"

        with patch("scripts.publish_model.subprocess.run") as mock_run:
            result = quantize_gguf(
                "/fake/input.gguf", str(output_file), "Q4_K_M", str(llama_dir)
            )

        mock_run.assert_called_once()
        assert result == output_file

    def test_uses_default_llama_cpp_path(self):
        """Verify default llama.cpp path is /workspace/llama.cpp."""
        with pytest.raises(FileNotFoundError):
            quantize_gguf("/fake/input.gguf", "/fake/out.gguf")


# ===========================================================================
# create_model_card
# ===========================================================================
class TestCreateModelCard:
    """Tests for model card generation."""

    def test_creates_model_card_file(self, tmp_path):
        """Verify model card file is created with correct content."""
        card_path = tmp_path / "MODEL_CARD.md"
        result = create_model_card(
            "DavidJBarnes/bridge-cli",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            str(card_path),
        )
        assert result == card_path
        assert card_path.exists()

    def test_model_card_contains_repo_id(self, tmp_path):
        """Verify model card references the repo ID."""
        card_path = tmp_path / "MODEL_CARD.md"
        create_model_card("DavidJBarnes/bridge-cli", "base-model", str(card_path))
        content = card_path.read_text()
        assert "DavidJBarnes/bridge-cli" in content

    def test_model_card_contains_base_model(self, tmp_path):
        """Verify model card references the base model."""
        card_path = tmp_path / "MODEL_CARD.md"
        create_model_card("repo/name", "deepseek-ai/deepseek-coder-6.7b-instruct", str(card_path))
        content = card_path.read_text()
        assert "deepseek-ai/deepseek-coder-6.7b-instruct" in content

    def test_model_card_contains_usage_sections(self, tmp_path):
        """Verify model card includes all usage sections."""
        card_path = tmp_path / "MODEL_CARD.md"
        create_model_card("repo/name", "base-model", str(card_path))
        content = card_path.read_text()
        assert "Transformers" in content
        assert "vLLM" in content
        assert "Ollama" in content
        assert "llama.cpp" in content or "llama-cli" in content

    def test_model_card_creates_parent_dirs(self, tmp_path):
        """Verify parent directories are created if they don't exist."""
        card_path = tmp_path / "nested" / "dir" / "MODEL_CARD.md"
        create_model_card("repo/name", "base-model", str(card_path))
        assert card_path.exists()

    def test_model_card_contains_variants_table(self, tmp_path):
        """Verify model card includes the model variants table."""
        card_path = tmp_path / "MODEL_CARD.md"
        create_model_card("repo/name", "base-model", str(card_path))
        content = card_path.read_text()
        assert "Q8_0" in content
        assert "Q4_K_M" in content
        assert "adapter" in content


# ===========================================================================
# publish_to_hub
# ===========================================================================
class TestPublishToHub:
    """Tests for publishing artifacts to Hugging Face Hub."""

    def test_raises_import_error(self, tmp_path, monkeypatch):
        """Verify ImportError when huggingface_hub is not installed."""
        import importlib

        monkeypatch.delitem(sys.modules, "huggingface_hub")
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        importlib.reload(sys.modules["scripts.publish_model"])
        from scripts.publish_model import publish_to_hub as publish_fresh

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(ImportError, match="huggingface_hub not installed"):
            publish_fresh("user/repo", str(adapter_dir))

    def test_creates_repo_and_uploads_adapter(self, tmp_path):
        """Verify repo creation and adapter upload."""
        import huggingface_hub

        mock_api = MagicMock()
        huggingface_hub.HfApi.return_value = mock_api

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        result = publish_to_hub("user/repo", str(adapter_dir))

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/repo", exist_ok=True, repo_type="model"
        )
        mock_api.upload_folder.assert_called_once()
        assert result == "https://huggingface.co/user/repo"

    def test_uploads_model_card(self, tmp_path):
        """Verify model card is uploaded as README.md."""
        import huggingface_hub

        mock_api = MagicMock()
        huggingface_hub.HfApi.return_value = mock_api

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        card_path = tmp_path / "card.md"
        card_path.write_text("# Card")

        publish_to_hub("user/repo", str(adapter_dir), model_card_path=str(card_path))

        upload_calls = mock_api.upload_file.call_args_list
        assert any(c.kwargs.get("path_in_repo") == "README.md" or
                    (len(c.args) == 0 and c[1].get("path_in_repo") == "README.md")
                    for c in upload_calls)

    def test_uploads_gguf_files(self, tmp_path):
        """Verify GGUF files are uploaded."""
        import huggingface_hub

        mock_api = MagicMock()
        huggingface_hub.HfApi.return_value = mock_api

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        gguf1 = tmp_path / "model-Q8.gguf"
        gguf1.write_text("fake")
        gguf2 = tmp_path / "model-Q4.gguf"
        gguf2.write_text("fake")

        publish_to_hub(
            "user/repo", str(adapter_dir), gguf_files=[str(gguf1), str(gguf2)]
        )

        # adapter folder upload + 2 gguf file uploads
        assert mock_api.upload_file.call_count >= 2

    def test_skips_missing_gguf_files(self, tmp_path):
        """Verify missing GGUF files are skipped with a warning."""
        import huggingface_hub

        mock_api = MagicMock()
        huggingface_hub.HfApi.return_value = mock_api

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        publish_to_hub(
            "user/repo",
            str(adapter_dir),
            gguf_files=["/nonexistent/model.gguf"],
        )

        # Only adapter upload, no gguf uploads
        assert mock_api.upload_folder.call_count == 1

    def test_skips_model_card_if_not_provided(self, tmp_path):
        """Verify no model card upload when path is None."""
        import huggingface_hub

        mock_api = MagicMock()
        huggingface_hub.HfApi.return_value = mock_api

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        publish_to_hub("user/repo", str(adapter_dir))

        # No upload_file calls for model card
        for c in mock_api.upload_file.call_args_list:
            kwargs = c[1] if len(c) > 1 else c.kwargs
            assert kwargs.get("path_in_repo") != "README.md"


# ===========================================================================
# main (CLI)
# ===========================================================================
class TestMain:
    """Tests for the CLI entry point."""

    def test_adapter_only_skips_merge(self, tmp_path, monkeypatch):
        """Verify --adapter-only skips merge and quantize steps."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "publish_model.py",
                "--repo-id", "user/repo",
                "--adapter-path", str(tmp_path),
                "--output-dir", str(tmp_path / "out"),
                "--adapter-only",
                "--skip-upload",
            ],
        )

        with patch("scripts.publish_model.merge_adapter") as mock_merge, \
             patch("scripts.publish_model.create_model_card") as mock_card:
            main()

        mock_merge.assert_not_called()
        mock_card.assert_called_once()

    def test_full_pipeline(self, tmp_path, monkeypatch):
        """Verify full pipeline calls merge, convert, quantize, card, publish."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "publish_model.py",
                "--repo-id", "user/repo",
                "--adapter-path", str(tmp_path / "adapter"),
                "--output-dir", str(tmp_path / "out"),
                "--llama-cpp-path", str(tmp_path / "llama"),
            ],
        )

        with patch("scripts.publish_model.merge_adapter") as mock_merge, \
             patch("scripts.publish_model.convert_to_gguf") as mock_convert, \
             patch("scripts.publish_model.quantize_gguf") as mock_quantize, \
             patch("scripts.publish_model.create_model_card") as mock_card, \
             patch("scripts.publish_model.publish_to_hub") as mock_publish, \
             patch("pathlib.Path.unlink"):  # don't actually delete
            mock_convert.return_value = tmp_path / "fp16.gguf"
            main()

        mock_merge.assert_called_once()
        mock_convert.assert_called_once()
        assert mock_quantize.call_count == 2  # Q8_0 and Q4_K_M
        mock_card.assert_called_once()
        mock_publish.assert_called_once()

    def test_skip_quantize(self, tmp_path, monkeypatch):
        """Verify --skip-quantize skips GGUF conversion."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "publish_model.py",
                "--repo-id", "user/repo",
                "--adapter-path", str(tmp_path / "adapter"),
                "--output-dir", str(tmp_path / "out"),
                "--skip-quantize",
                "--skip-upload",
            ],
        )

        with patch("scripts.publish_model.merge_adapter") as mock_merge, \
             patch("scripts.publish_model.convert_to_gguf") as mock_convert, \
             patch("scripts.publish_model.create_model_card"):
            main()

        mock_merge.assert_called_once()
        mock_convert.assert_not_called()

    def test_skip_upload(self, tmp_path, monkeypatch):
        """Verify --skip-upload skips publishing to Hub."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "publish_model.py",
                "--repo-id", "user/repo",
                "--adapter-path", str(tmp_path / "adapter"),
                "--output-dir", str(tmp_path / "out"),
                "--adapter-only",
                "--skip-upload",
            ],
        )

        with patch("scripts.publish_model.create_model_card"), \
             patch("scripts.publish_model.publish_to_hub") as mock_publish:
            main()

        mock_publish.assert_not_called()

    def test_fp16_cleanup(self, tmp_path, monkeypatch):
        """Verify intermediate FP16 GGUF is cleaned up after quantization."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "publish_model.py",
                "--repo-id", "user/repo",
                "--adapter-path", str(tmp_path / "adapter"),
                "--output-dir", str(tmp_path / "out"),
                "--llama-cpp-path", str(tmp_path / "llama"),
                "--skip-upload",
            ],
        )

        fp16_path = tmp_path / "out" / "gguf" / "bridge-cli-f16.gguf"

        with patch("scripts.publish_model.merge_adapter"), \
             patch("scripts.publish_model.convert_to_gguf") as mock_convert, \
             patch("scripts.publish_model.quantize_gguf"), \
             patch("scripts.publish_model.create_model_card"):
            mock_convert.return_value = fp16_path
            # Create the file so unlink works
            fp16_path.parent.mkdir(parents=True, exist_ok=True)
            fp16_path.write_text("fake")
            main()

        assert not fp16_path.exists()

    def test_default_args(self, tmp_path, monkeypatch):
        """Verify default argument values are passed correctly."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "publish_model.py",
                "--repo-id", "user/repo",
                "--adapter-only",
                "--skip-upload",
            ],
        )

        with patch("scripts.publish_model.create_model_card") as mock_card:
            main()

        card_call = mock_card.call_args
        assert "deepseek-ai/deepseek-coder-6.7b-instruct" in card_call[0]
