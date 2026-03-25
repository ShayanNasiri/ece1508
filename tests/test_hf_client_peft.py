"""Tests for HFClient LoRA adapter (PEFT) loading path.

When --model is a local directory containing adapter_config.json, HFClient
must load via peft.PeftModel + AutoModelForCausalLM rather than the plain
transformers pipeline. Plain HF Hub IDs must continue to use the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_evaluation"))
from hf_client import HFClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_dir(tmp_path: Path) -> Path:
    """Create a minimal fake adapter directory with adapter_config.json."""
    adapter_dir = tmp_path / "finetuned_model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct"}'
    )
    return adapter_dir


def _make_tokenizer_mock(prompt: str):
    """Return a mock tokenizer that encodes prompt and decodes continuation."""
    tok = MagicMock()
    tok.return_value = {"input_ids": MagicMock()}
    tok.decode.return_value = prompt + "B"
    tok.eos_token_id = 0
    return tok


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

class TestAdapterDetection:
    """HFClient correctly identifies local adapter paths vs HF Hub IDs."""

    def test_local_adapter_path_detected(self, tmp_path):
        adapter_dir = _make_adapter_dir(tmp_path)
        client = HFClient()
        assert client._is_adapter_path(str(adapter_dir)) is True

    def test_hf_hub_id_not_detected_as_adapter(self, tmp_path):
        client = HFClient()
        assert client._is_adapter_path("HuggingFaceTB/SmolLM2-135M-Instruct") is False

    def test_nonexistent_path_not_detected_as_adapter(self, tmp_path):
        client = HFClient()
        assert client._is_adapter_path(str(tmp_path / "does_not_exist")) is False

    def test_local_dir_without_adapter_config_not_detected(self, tmp_path):
        plain_dir = tmp_path / "plain_model"
        plain_dir.mkdir()
        (plain_dir / "config.json").write_text("{}")
        client = HFClient()
        assert client._is_adapter_path(str(plain_dir)) is False


# ---------------------------------------------------------------------------
# PEFT loading path
# ---------------------------------------------------------------------------

class TestPeftLoading:
    """When adapter_config.json is present, load via peft, not pipeline."""

    def test_peft_model_loaded_for_adapter_path(self, tmp_path):
        adapter_dir = _make_adapter_dir(tmp_path)
        prompt = "Choose A-E: "

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer = _make_tokenizer_mock(prompt)
        mock_tokenizer.decode.return_value = prompt + "A"

        with (
            patch("hf_client.AutoModelForCausalLM") as mock_auto,
            patch("hf_client.PeftModel") as mock_peft,
            patch("hf_client.AutoTokenizer") as mock_tok_cls,
            patch("hf_client.pipeline") as mock_pipeline,
        ):
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_peft.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            client = HFClient()
            client.query(str(adapter_dir), prompt)

            # peft path used — plain pipeline must NOT be called
            mock_pipeline.assert_not_called()
            mock_peft.from_pretrained.assert_called_once()

    def test_plain_pipeline_used_for_hub_id(self, tmp_path):
        prompt = "Choose A-E: "
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": prompt + "C"}]

        with (
            patch("hf_client.pipeline", return_value=mock_pipe) as mock_pipeline_factory,
            patch("hf_client.PeftModel") as mock_peft,
        ):
            client = HFClient()
            client.query("HuggingFaceTB/SmolLM2-135M-Instruct", prompt)

            mock_pipeline_factory.assert_called_once()
            mock_peft.from_pretrained.assert_not_called()

    def test_peft_model_cached_after_first_load(self, tmp_path):
        adapter_dir = _make_adapter_dir(tmp_path)
        prompt = "Q: "

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer = _make_tokenizer_mock(prompt)
        mock_tokenizer.decode.return_value = prompt + "A"

        with (
            patch("hf_client.AutoModelForCausalLM") as mock_auto,
            patch("hf_client.PeftModel") as mock_peft,
            patch("hf_client.AutoTokenizer") as mock_tok_cls,
        ):
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_peft.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            client = HFClient()
            client.query(str(adapter_dir), prompt)
            client.query(str(adapter_dir), prompt)

            # loaded only once despite two queries
            assert mock_peft.from_pretrained.call_count == 1

    def test_peft_uses_base_model_from_adapter_config(self, tmp_path):
        adapter_dir = _make_adapter_dir(tmp_path)
        prompt = "Q: "

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer = _make_tokenizer_mock(prompt)
        mock_tokenizer.decode.return_value = prompt + "A"

        with (
            patch("hf_client.AutoModelForCausalLM") as mock_auto,
            patch("hf_client.PeftModel") as mock_peft,
            patch("hf_client.AutoTokenizer") as mock_tok_cls,
        ):
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_peft.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            client = HFClient()
            client.query(str(adapter_dir), prompt)

            # base model ID read from adapter_config.json
            base_model_call_arg = mock_auto.from_pretrained.call_args[0][0]
            assert base_model_call_arg == "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestPeftQueryReturn:
    """PEFT path returns only the continuation (prompt prefix stripped)."""

    def test_returns_continuation_only(self, tmp_path):
        adapter_dir = _make_adapter_dir(tmp_path)
        prompt = "Choose A-E: "
        continuation = "B"

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.decode.return_value = prompt + continuation
        mock_tokenizer.eos_token_id = 0

        with (
            patch("hf_client.AutoModelForCausalLM") as mock_auto,
            patch("hf_client.PeftModel") as mock_peft,
            patch("hf_client.AutoTokenizer") as mock_tok_cls,
        ):
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_peft.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            client = HFClient()
            result = client.query(str(adapter_dir), prompt)

        assert result == continuation
