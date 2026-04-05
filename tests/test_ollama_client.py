"""Tests for llm_evaluation/ollama_client.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure llm_evaluation is importable
ROOT = Path(__file__).resolve().parents[1]
LLM_EVAL = ROOT / "llm_evaluation"
if str(LLM_EVAL) not in sys.path:
    sys.path.insert(0, str(LLM_EVAL))

from ollama_client import OllamaClient


class TestOllamaClientSuccess:
    def test_returns_response_text(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "A"}
        with patch("ollama_client.requests.post", return_value=mock_resp) as mock_post:
            client = OllamaClient()
            result = client.query("test-model", "What is 1+1?")
        assert result == "A"
        mock_post.assert_called_once()

    def test_passes_model_and_prompt_in_payload(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("ollama_client.requests.post", return_value=mock_resp) as mock_post:
            OllamaClient().query("my-model", "hello", temperature=0.5)
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "my-model"
        assert payload["prompt"] == "hello"
        assert payload["temperature"] == 0.5
        assert payload["stream"] is False

    def test_uses_custom_timeout(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("ollama_client.requests.post", return_value=mock_resp) as mock_post:
            OllamaClient(timeout=30).query("m", "p")
        assert mock_post.call_args[1]["timeout"] == 30


class TestOllamaClientRetry:
    def test_retries_on_request_exception(self):
        import requests as req

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        with patch("ollama_client.requests.post") as mock_post, \
             patch("ollama_client.time.sleep") as mock_sleep:
            mock_post.side_effect = [req.ConnectionError("down"), mock_resp]
            result = OllamaClient(max_retries=2, retry_delay=1).query("m", "p")
        assert result == "ok"
        mock_sleep.assert_called_once_with(1)

    def test_raises_after_max_retries_exhausted(self):
        import requests as req

        with patch("ollama_client.requests.post") as mock_post, \
             patch("ollama_client.time.sleep"):
            mock_post.side_effect = req.ConnectionError("down")
            with pytest.raises(RuntimeError, match="failed after 2 attempts"):
                OllamaClient(max_retries=2).query("m", "p")


class TestOllamaClientResponseValidation:
    def test_raises_on_missing_response_key(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "model not found"}
        with patch("ollama_client.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="missing 'response' key"):
                OllamaClient().query("bad-model", "p")

    def test_does_not_retry_on_missing_response_key(self):
        """Response format errors should fail immediately, not retry."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "bad"}
        with patch("ollama_client.requests.post", return_value=mock_resp) as mock_post:
            with pytest.raises(RuntimeError):
                OllamaClient(max_retries=3).query("m", "p")
        # Should only call once — no retries for parse errors
        assert mock_post.call_count == 1
