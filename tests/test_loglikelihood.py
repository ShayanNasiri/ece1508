"""Tests for log-likelihood evaluation mode."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_evaluation"))
from hf_client import HFClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model_and_tokenizer(logits_dict: dict[str, float]):
    """Build mock model + tokenizer where logits_dict maps token strings to scores.

    logits_dict example: {"A": 5.0, "B": 2.0, "C": 8.0, "D": 1.0, "E": 0.5}
    """
    import types

    # Fake tokenizer: encode each letter to a unique token id
    token_to_id = {}
    for i, letter in enumerate("ABCDE"):
        token_to_id[letter] = 100 + i  # A=100, B=101, ...

    mock_tokenizer = MagicMock()

    def fake_encode(text, add_special_tokens=True):
        # Return a list containing the token id for the letter
        if text in token_to_id:
            return [token_to_id[text]]
        return [999]  # unknown

    mock_tokenizer.encode = fake_encode

    # Fake tokenizer __call__: return dict with input_ids
    fake_input_ids = MagicMock()
    fake_input_ids.to = MagicMock(return_value=fake_input_ids)

    mock_tokenizer_output = MagicMock()
    mock_tokenizer_output.__getitem__ = lambda self, key: fake_input_ids if key == "input_ids" else MagicMock()
    mock_tokenizer_output.to = MagicMock(return_value=mock_tokenizer_output)
    mock_tokenizer.return_value = mock_tokenizer_output

    # Build logits tensor-like object: shape [1, seq_len, vocab_size]
    # The code does: logits[0, -1, :] then [token_id].item()
    vocab_size = 200
    last_position_logits = [0.0] * vocab_size
    for letter, score in logits_dict.items():
        last_position_logits[token_to_id[letter]] = score

    class FakeScalar:
        def __init__(self, val):
            self._val = val

        def item(self):
            return self._val

    class FakeLastLogits:
        """Represents logits[0, -1, :] — indexable by token_id."""
        def __getitem__(self, idx):
            return FakeScalar(last_position_logits[idx])

    class FakeLogits:
        """Represents model output logits with shape [1, seq_len, vocab_size]."""
        def __getitem__(self, idx):
            # Called as logits[0, -1, :] — a tuple index
            if isinstance(idx, tuple):
                return FakeLastLogits()
            return MagicMock()

    mock_model_output = MagicMock()
    mock_model_output.logits = FakeLogits()

    mock_model = MagicMock()
    mock_model.return_value = mock_model_output
    mock_model.device = "cpu"

    return mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# HFClient.query_loglikelihood tests
# ---------------------------------------------------------------------------

class TestHFClientQueryLoglikelihood:
    """HFClient.query_loglikelihood scores choices by logits and returns the best."""

    def test_returns_highest_scoring_letter(self):
        mock_model, mock_tokenizer = _make_mock_model_and_tokenizer(
            {"A": 1.0, "B": 2.0, "C": 8.0, "D": 3.0, "E": 0.5}
        )
        with patch.object(HFClient, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):
            client = HFClient()
            result = client.query_loglikelihood("some/base-model", "prompt text", ["A", "B", "C", "D", "E"])
        assert result == "C"

    def test_returns_different_letter_when_scores_change(self):
        mock_model, mock_tokenizer = _make_mock_model_and_tokenizer(
            {"A": 10.0, "B": 2.0, "C": 3.0, "D": 1.0, "E": 0.5}
        )
        with patch.object(HFClient, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):
            client = HFClient()
            result = client.query_loglikelihood("some/base-model", "prompt text", ["A", "B", "C", "D", "E"])
        assert result == "A"

    def test_works_with_subset_of_choices(self):
        """Should work even if only a subset of letters is provided."""
        mock_model, mock_tokenizer = _make_mock_model_and_tokenizer(
            {"A": 1.0, "B": 9.0, "C": 3.0, "D": 1.0, "E": 0.5}
        )
        with patch.object(HFClient, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):
            client = HFClient()
            result = client.query_loglikelihood("some/base-model", "prompt text", ["A", "C", "E"])
        assert result == "C"

    def test_caches_model_across_calls(self):
        mock_model, mock_tokenizer = _make_mock_model_and_tokenizer(
            {"A": 5.0, "B": 2.0, "C": 3.0, "D": 1.0, "E": 0.5}
        )
        with patch.object(HFClient, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)) as mock_load:
            client = HFClient()
            client.query_loglikelihood("some/model", "p1", ["A", "B", "C", "D", "E"])
            client.query_loglikelihood("some/model", "p2", ["A", "B", "C", "D", "E"])
        assert mock_load.call_count == 1

    def test_method_exists_on_hfclient(self):
        assert hasattr(HFClient, "query_loglikelihood")


class TestHFClientQueryLoglikelihoodInterface:
    """query_loglikelihood has no side effects on generative query method."""

    def test_query_method_still_works(self):
        """Generative query is not broken by the new method."""
        prompt = "What is the answer?\n"
        continuation = "B"
        with patch("hf_client.pipeline", return_value=_make_pipeline_mock(prompt + continuation)):
            client = HFClient()
            result = client.query("some/model", prompt, temperature=0)
        assert result == continuation


def _make_pipeline_mock(generated_text):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": generated_text}]
    return mock_pipe


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------

class TestEvalModeArgument:
    """--eval-mode argument is parsed correctly in mc_eval CLI."""

    def test_eval_mode_defaults_to_generative(self):
        from mc_eval import build_arg_parser
        args = build_arg_parser().parse_args(["--model", "some/model"])
        assert args.eval_mode == "generative"

    def test_eval_mode_generative_accepted(self):
        from mc_eval import build_arg_parser
        args = build_arg_parser().parse_args(["--model", "m", "--eval-mode", "generative"])
        assert args.eval_mode == "generative"

    def test_eval_mode_loglikelihood_accepted(self):
        from mc_eval import build_arg_parser
        args = build_arg_parser().parse_args(["--model", "m", "--eval-mode", "loglikelihood"])
        assert args.eval_mode == "loglikelihood"

    def test_eval_mode_rejects_invalid_value(self):
        from mc_eval import build_arg_parser
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--model", "m", "--eval-mode", "invalid"])

    def test_loglikelihood_requires_huggingface_backend(self):
        """loglikelihood mode only works with huggingface backend."""
        from mc_eval import build_arg_parser
        args = build_arg_parser().parse_args([
            "--model", "m", "--eval-mode", "loglikelihood", "--backend", "huggingface"
        ])
        assert args.eval_mode == "loglikelihood"
        assert args.backend == "huggingface"


# ---------------------------------------------------------------------------
# Evaluation flow tests
# ---------------------------------------------------------------------------

class TestLoglikelihoodEvalFlow:
    """run_evaluation uses loglikelihood scoring when --eval-mode loglikelihood."""

    def test_loglikelihood_mode_calls_query_loglikelihood(self, tmp_path):
        """In loglikelihood mode, client.query_loglikelihood is called instead of client.query."""
        from mc_eval import build_arg_parser, run_evaluation
        from recipe_mpr_qa.data.models import RecipeOption, RecipeExample
        from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES

        # Create a minimal example
        options = tuple(
            RecipeOption(option_id=f"opt-{i}", text=f"Recipe {i}")
            for i in range(1, 6)
        )
        flags = {qt: (qt == "Specific") for qt in QUERY_TYPE_NAMES}
        example = RecipeExample(
            example_id="rmpr-0001",
            query="Find me a pasta recipe",
            options=options,
            answer_option_id="opt-3",
            query_type_flags=flags,
            correctness_explanation={"reason": "test"},
        )

        mock_client = MagicMock()
        mock_client.query_loglikelihood = MagicMock(return_value="C")

        output_path = str(tmp_path / "test_output.json")

        args = build_arg_parser().parse_args([
            "--model", "some/base-model",
            "--backend", "huggingface",
            "--eval-mode", "loglikelihood",
            "--output", output_path,
        ])

        with patch("recipe_mpr_qa.evaluation.mc_eval.HFClient", return_value=mock_client), \
             patch("recipe_mpr_qa.evaluation.mc_eval.load_dataset"), \
             patch("recipe_mpr_qa.evaluation.mc_eval.load_split_manifest"), \
             patch("recipe_mpr_qa.evaluation.mc_eval.get_split_examples", return_value=(example,)):
            result = run_evaluation(args)

        mock_client.query_loglikelihood.assert_called_once()
        # query (generative) should NOT be called
        mock_client.query.assert_not_called()

    def test_loglikelihood_result_row_has_eval_mode(self, tmp_path):
        """Result JSON includes eval_mode field."""
        from mc_eval import build_arg_parser, run_evaluation
        from recipe_mpr_qa.data.models import RecipeOption, RecipeExample
        from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
        import json

        options = tuple(
            RecipeOption(option_id=f"opt-{i}", text=f"Recipe {i}")
            for i in range(1, 6)
        )
        flags = {qt: (qt == "Specific") for qt in QUERY_TYPE_NAMES}
        example = RecipeExample(
            example_id="rmpr-0001",
            query="Find me a pasta recipe",
            options=options,
            answer_option_id="opt-3",
            query_type_flags=flags,
            correctness_explanation={"reason": "test"},
        )

        mock_client = MagicMock()
        mock_client.query_loglikelihood = MagicMock(return_value="C")

        output_path = str(tmp_path / "test_output.json")

        args = build_arg_parser().parse_args([
            "--model", "some/base-model",
            "--backend", "huggingface",
            "--eval-mode", "loglikelihood",
            "--output", output_path,
        ])

        with patch("recipe_mpr_qa.evaluation.mc_eval.HFClient", return_value=mock_client), \
             patch("recipe_mpr_qa.evaluation.mc_eval.load_dataset"), \
             patch("recipe_mpr_qa.evaluation.mc_eval.load_split_manifest"), \
             patch("recipe_mpr_qa.evaluation.mc_eval.get_split_examples", return_value=(example,)):
            run_evaluation(args)

        with open(output_path) as f:
            output = json.load(f)

        assert output["eval_mode"] == "loglikelihood"

    def test_generative_mode_still_calls_query(self, tmp_path):
        """In generative mode (default), client.query is called as before."""
        from mc_eval import build_arg_parser, run_evaluation
        from recipe_mpr_qa.data.models import RecipeOption, RecipeExample
        from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES

        options = tuple(
            RecipeOption(option_id=f"opt-{i}", text=f"Recipe {i}")
            for i in range(1, 6)
        )
        flags = {qt: (qt == "Specific") for qt in QUERY_TYPE_NAMES}
        example = RecipeExample(
            example_id="rmpr-0001",
            query="Find me a pasta recipe",
            options=options,
            answer_option_id="opt-3",
            query_type_flags=flags,
            correctness_explanation={"reason": "test"},
        )

        mock_client = MagicMock()
        mock_client.query = MagicMock(return_value="C")

        output_path = str(tmp_path / "test_output.json")

        args = build_arg_parser().parse_args([
            "--model", "some/base-model",
            "--backend", "huggingface",
            "--eval-mode", "generative",
            "--output", output_path,
        ])

        with patch("recipe_mpr_qa.evaluation.mc_eval.HFClient", return_value=mock_client), \
             patch("recipe_mpr_qa.evaluation.mc_eval.load_dataset"), \
             patch("recipe_mpr_qa.evaluation.mc_eval.load_split_manifest"), \
             patch("recipe_mpr_qa.evaluation.mc_eval.get_split_examples", return_value=(example,)):
            run_evaluation(args)

        mock_client.query.assert_called_once()

    def test_loglikelihood_with_ollama_backend_raises_error(self, tmp_path):
        """loglikelihood + ollama should raise an error — only HF supports it."""
        from mc_eval import build_arg_parser, run_evaluation

        output_path = str(tmp_path / "test_output.json")
        args = build_arg_parser().parse_args([
            "--model", "smollm2:135m",
            "--backend", "ollama",
            "--eval-mode", "loglikelihood",
            "--output", output_path,
        ])

        with pytest.raises(ValueError, match="loglikelihood.*huggingface"):
            run_evaluation(args)
