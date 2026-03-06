"""Tests for scripts/test_model.py -- model loading, inference, and CLI."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure importability
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We need to mock heavy dependencies before importing the module
# because `import torch` / `transformers` / `peft` will fail in test
# environments without GPUs.


@pytest.fixture(autouse=True)
def _mock_heavy_deps(monkeypatch):
    """Provide lightweight mocks for torch, transformers, and peft.

    This fixture runs automatically for every test in this module and
    ensures that the real packages are never imported.
    """
    mock_torch = MagicMock()
    mock_torch.float16 = "float16"
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()

    mock_transformers = MagicMock()
    mock_peft = MagicMock()

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)
    monkeypatch.setitem(sys.modules, "peft", mock_peft)


def _import_test_model():
    """Import (or re-import) the test_model module under mocked dependencies."""
    # Remove cached import so the mocked modules are used
    sys.modules.pop("scripts.test_model", None)
    from scripts.test_model import (
        TEST_PROMPTS,
        generate_response,
        interactive_mode,
        load_model,
        main,
        run_tests,
    )
    return load_model, generate_response, run_tests, interactive_mode, main, TEST_PROMPTS


# -----------------------------------------------------------------------
# load_model
# -----------------------------------------------------------------------
class TestLoadModel:
    """Tests for load_model with mocked transformers and peft."""

    def test_loads_base_model_without_lora(self):
        """When lora_path is None, PeftModel should NOT be applied."""
        load_model, *_ = _import_test_model()
        import transformers
        import peft

        model, tokenizer = load_model("base-model")

        transformers.AutoTokenizer.from_pretrained.assert_called_once()
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        peft.PeftModel.from_pretrained.assert_not_called()

    def test_loads_model_with_lora(self):
        """When lora_path is provided, PeftModel.from_pretrained should wrap the model."""
        load_model, *_ = _import_test_model()
        import transformers
        import peft

        # Reset call tracking from previous test
        transformers.AutoTokenizer.from_pretrained.reset_mock()
        transformers.AutoModelForCausalLM.from_pretrained.reset_mock()
        peft.PeftModel.from_pretrained.reset_mock()

        model, tokenizer = load_model("base-model", lora_path="/path/to/lora")

        peft.PeftModel.from_pretrained.assert_called_once()


# -----------------------------------------------------------------------
# generate_response
# -----------------------------------------------------------------------
class TestGenerateResponse:
    """Tests for generate_response with a mocked model and tokenizer."""

    def _make_mocks(self):
        """Create lightweight model and tokenizer mocks."""
        import torch

        model = MagicMock()
        model.device = "cpu"

        tokenizer = MagicMock()
        # tokenizer(...) should return a dict-like object with .to()
        token_output = MagicMock()
        token_output.to.return_value = token_output
        tokenizer.return_value = token_output
        tokenizer.eos_token_id = 0

        # model.generate returns a tensor-like object
        model.generate.return_value = [MagicMock()]
        # tokenizer.decode returns a string containing the response marker
        tokenizer.decode.return_value = (
            "### Instruction:\nDo something\n\n### Response:\npublic class Foo {}"
        )

        return model, tokenizer

    def test_returns_response_string(self):
        """generate_response should return the text after '### Response:'."""
        _, generate_response, *_ = _import_test_model()
        model, tokenizer = self._make_mocks()

        result = generate_response(model, tokenizer, "Create a controller")

        assert isinstance(result, str)
        assert "public class Foo" in result

    def test_handles_response_without_marker(self):
        """If '### Response:' is absent from output, the full decoded text is returned."""
        _, generate_response, *_ = _import_test_model()
        model, tokenizer = self._make_mocks()
        tokenizer.decode.return_value = "just some raw output"

        result = generate_response(model, tokenizer, "prompt")

        assert result == "just some raw output"

    def test_respects_max_length(self):
        """The max_new_tokens kwarg passed to model.generate should honour max_length."""
        _, generate_response, *_ = _import_test_model()
        model, tokenizer = self._make_mocks()

        generate_response(model, tokenizer, "prompt", max_length=512)

        _, kwargs = model.generate.call_args
        assert kwargs["max_new_tokens"] == 512


# -----------------------------------------------------------------------
# run_tests
# -----------------------------------------------------------------------
class TestRunTests:
    """Tests for run_tests -- ensures it iterates over all prompts."""

    def test_calls_generate_response_for_each_prompt(self):
        """run_tests should invoke generate_response once per prompt."""
        load_model, generate_response_fn, run_tests, *_ = _import_test_model()

        model = MagicMock()
        tokenizer = MagicMock()

        prompts = ["prompt1", "prompt2", "prompt3"]

        with patch("scripts.test_model.generate_response", return_value="response") as mock_gen:
            run_tests(model, tokenizer, prompts)
            assert mock_gen.call_count == len(prompts)

    def test_uses_default_prompts_when_none_given(self):
        """When prompts is None, the module-level TEST_PROMPTS list should be used."""
        *_, run_tests, _, main_fn, TEST_PROMPTS = _import_test_model()

        model = MagicMock()
        tokenizer = MagicMock()

        with patch("scripts.test_model.generate_response", return_value="response") as mock_gen:
            run_tests(model, tokenizer)
            assert mock_gen.call_count == len(TEST_PROMPTS)

    def test_handles_long_response(self):
        """Responses longer than 1000 chars should be truncated in output (no crash)."""
        *_, run_tests, _, main_fn, TEST_PROMPTS = _import_test_model()

        model = MagicMock()
        tokenizer = MagicMock()
        long_response = "x" * 2000

        with patch("scripts.test_model.generate_response", return_value=long_response):
            # Should not raise
            run_tests(model, tokenizer, ["short prompt"])


# -----------------------------------------------------------------------
# interactive_mode
# -----------------------------------------------------------------------
class TestInteractiveMode:
    """Tests for interactive_mode with mocked input."""

    def test_quit_exits_loop(self):
        """Typing 'quit' should break out of the interactive loop."""
        *_, interactive_mode, main_fn, _ = _import_test_model()
        model = MagicMock()
        tokenizer = MagicMock()

        with patch("builtins.input", return_value="quit"):
            with patch("scripts.test_model.generate_response") as mock_gen:
                interactive_mode(model, tokenizer)
                mock_gen.assert_not_called()

    def test_exit_command(self):
        """Typing 'exit' should also break the loop."""
        *_, interactive_mode, main_fn, _ = _import_test_model()
        model = MagicMock()
        tokenizer = MagicMock()

        with patch("builtins.input", return_value="exit"):
            with patch("scripts.test_model.generate_response") as mock_gen:
                interactive_mode(model, tokenizer)
                mock_gen.assert_not_called()

    def test_q_command(self):
        """Typing 'q' should also break the loop."""
        *_, interactive_mode, main_fn, _ = _import_test_model()
        model = MagicMock()
        tokenizer = MagicMock()

        with patch("builtins.input", return_value="q"):
            with patch("scripts.test_model.generate_response") as mock_gen:
                interactive_mode(model, tokenizer)
                mock_gen.assert_not_called()

    def test_processes_prompt_then_quits(self):
        """A valid prompt should trigger generate_response, then quit exits."""
        *_, interactive_mode, main_fn, _ = _import_test_model()
        model = MagicMock()
        tokenizer = MagicMock()

        with patch("builtins.input", side_effect=["Create a service", "quit"]):
            with patch("scripts.test_model.generate_response", return_value="output") as mock_gen:
                interactive_mode(model, tokenizer)
                mock_gen.assert_called_once()


# -----------------------------------------------------------------------
# main (CLI argument parsing)
# -----------------------------------------------------------------------
class TestMainCLI:
    """Tests for the main function's argument parsing and dispatch logic."""

    def test_default_runs_tests(self, mocker):
        """Without --interactive, main should call run_tests."""
        *_, main_fn, _ = _import_test_model()

        mocker.patch("sys.argv", ["test_model.py", "--no-lora"])
        mock_load = mocker.patch("scripts.test_model.load_model", return_value=(MagicMock(), MagicMock()))
        mock_run = mocker.patch("scripts.test_model.run_tests")
        mock_interactive = mocker.patch("scripts.test_model.interactive_mode")

        main_fn()

        mock_load.assert_called_once()
        mock_run.assert_called_once()
        mock_interactive.assert_not_called()

    def test_interactive_flag(self, mocker):
        """With --interactive, main should call interactive_mode instead of run_tests."""
        *_, main_fn, _ = _import_test_model()

        mocker.patch("sys.argv", ["test_model.py", "--interactive", "--no-lora"])
        mocker.patch("scripts.test_model.load_model", return_value=(MagicMock(), MagicMock()))
        mock_run = mocker.patch("scripts.test_model.run_tests")
        mock_interactive = mocker.patch("scripts.test_model.interactive_mode")

        main_fn()

        mock_interactive.assert_called_once()
        mock_run.assert_not_called()

    def test_no_lora_flag(self, mocker):
        """With --no-lora, lora_path should be None when calling load_model."""
        *_, main_fn, _ = _import_test_model()

        mocker.patch("sys.argv", ["test_model.py", "--no-lora"])
        mock_load = mocker.patch(
            "scripts.test_model.load_model", return_value=(MagicMock(), MagicMock())
        )
        mocker.patch("scripts.test_model.run_tests")

        main_fn()

        _, kwargs = mock_load.call_args
        # lora_path is passed as second positional arg
        call_args = mock_load.call_args
        assert call_args[0][1] is None or call_args.kwargs.get("lora_path") is None

    def test_custom_base_model(self, mocker):
        """--base-model should be forwarded to load_model."""
        *_, main_fn, _ = _import_test_model()

        mocker.patch("sys.argv", ["test_model.py", "--base-model", "custom/model", "--no-lora"])
        mock_load = mocker.patch(
            "scripts.test_model.load_model", return_value=(MagicMock(), MagicMock())
        )
        mocker.patch("scripts.test_model.run_tests")

        main_fn()

        assert mock_load.call_args[0][0] == "custom/model"
