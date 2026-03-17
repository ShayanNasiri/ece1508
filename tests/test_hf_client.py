"""Tests for HFClient — HuggingFace inference backend."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_evaluation"))
from hf_client import HFClient


def _make_pipeline_mock(generated_text):
    """Return a mock pipeline callable that produces generated_text."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": generated_text}]
    return mock_pipe


class TestHFClientQuery:
    """HFClient.query() returns only the generated continuation, not the prompt."""

    def test_returns_continuation_after_prompt(self):
        prompt = "What is the answer? Respond with A-E.\n"
        continuation = "B"
        with patch("hf_client.pipeline", return_value=_make_pipeline_mock(prompt + continuation)):
            client = HFClient()
            result = client.query("some/model", prompt, temperature=0)
        assert result == continuation

    def test_strips_prompt_prefix_correctly(self):
        prompt = "Choose A, B, C, D, or E.\n"
        continuation = "  The answer is C."
        with patch("hf_client.pipeline", return_value=_make_pipeline_mock(prompt + continuation)):
            client = HFClient()
            result = client.query("some/model", prompt)
        assert result == continuation

    def test_empty_continuation(self):
        prompt = "Pick one: A B C D E\n"
        with patch("hf_client.pipeline", return_value=_make_pipeline_mock(prompt)):
            client = HFClient()
            result = client.query("some/model", prompt)
        assert result == ""


class TestHFClientTemperature:
    """Temperature=0 uses greedy decoding; >0 enables sampling."""

    def test_temperature_zero_sets_do_sample_false(self):
        prompt = "Answer: "
        mock_pipe = _make_pipeline_mock(prompt + "A")
        with patch("hf_client.pipeline", return_value=mock_pipe):
            client = HFClient()
            client.query("some/model", prompt, temperature=0)
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert "temperature" not in call_kwargs

    def test_nonzero_temperature_sets_do_sample_true(self):
        prompt = "Answer: "
        mock_pipe = _make_pipeline_mock(prompt + "B")
        with patch("hf_client.pipeline", return_value=mock_pipe):
            client = HFClient()
            client.query("some/model", prompt, temperature=0.7)
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == pytest.approx(0.7)


class TestHFClientModelCaching:
    """Pipeline is loaded once per model and reused on subsequent queries."""

    def test_pipeline_loaded_once_for_same_model(self):
        prompt = "Q: "
        mock_pipe = _make_pipeline_mock(prompt + "A")
        with patch("hf_client.pipeline", return_value=mock_pipe) as mock_pipeline_factory:
            client = HFClient()
            client.query("org/model-a", prompt)
            client.query("org/model-a", prompt)
        assert mock_pipeline_factory.call_count == 1

    def test_separate_pipelines_for_different_models(self):
        prompt = "Q: "
        mock_pipe = _make_pipeline_mock(prompt + "A")
        with patch("hf_client.pipeline", return_value=mock_pipe) as mock_pipeline_factory:
            client = HFClient()
            client.query("org/model-a", prompt)
            client.query("org/model-b", prompt)
        assert mock_pipeline_factory.call_count == 2


class TestHFClientInterface:
    """HFClient has the same interface as OllamaClient."""

    def test_query_method_exists(self):
        assert hasattr(HFClient, "query")

    def test_query_signature_matches_ollama_client(self):
        import inspect
        sig = inspect.signature(HFClient.query)
        params = list(sig.parameters.keys())
        # Must have self, model_name, prompt, temperature
        assert "model_name" in params
        assert "prompt" in params
        assert "temperature" in params


class TestMcEvalBackendArgument:
    """mc_eval.py --backend argument is parsed correctly."""

    def test_backend_defaults_to_ollama(self):
        import argparse
        from mc_eval import build_arg_parser
        args = build_arg_parser().parse_args(["--model", "some/model"])
        assert args.backend == "ollama"

    def test_backend_huggingface_accepted(self):
        from mc_eval import build_arg_parser
        args = build_arg_parser().parse_args(["--model", "some/model", "--backend", "huggingface"])
        assert args.backend == "huggingface"

    def test_backend_rejects_invalid_value(self):
        from mc_eval import build_arg_parser
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--model", "some/model", "--backend", "invalid"])
