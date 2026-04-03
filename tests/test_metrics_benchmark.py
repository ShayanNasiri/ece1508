from __future__ import annotations

from recipe_mpr_qa.data.models import PreparedDataset, RecipeExample, RecipeOption
from recipe_mpr_qa.evaluation.metrics import (
    summarize_pairwise_prediction_comparison,
    summarize_prediction_metrics,
)
from recipe_mpr_qa.evaluation.records import PredictionRecord


def _dataset() -> PreparedDataset:
    return PreparedDataset(
        examples=(
            RecipeExample(
                example_id="rmpr-0001",
                query="query one",
                options=(
                    RecipeOption("a", "A"),
                    RecipeOption("b", "B"),
                    RecipeOption("c", "C"),
                    RecipeOption("d", "D"),
                    RecipeOption("e", "E"),
                ),
                answer_option_id="a",
                query_type_flags={
                    "Specific": False,
                    "Commonsense": True,
                    "Negated": False,
                    "Analogical": False,
                    "Temporal": False,
                },
                correctness_explanation={"x": "y"},
            ),
            RecipeExample(
                example_id="rmpr-0002",
                query="query two",
                options=(
                    RecipeOption("a2", "A2"),
                    RecipeOption("b2", "B2"),
                    RecipeOption("c2", "C2"),
                    RecipeOption("d2", "D2"),
                    RecipeOption("e2", "E2"),
                ),
                answer_option_id="b2",
                query_type_flags={
                    "Specific": True,
                    "Commonsense": False,
                    "Negated": False,
                    "Analogical": False,
                    "Temporal": False,
                },
                correctness_explanation={"x": "y"},
            ),
        )
    )


def test_prediction_metrics_include_parse_failures_and_ci() -> None:
    dataset = _dataset()
    records = (
        PredictionRecord(
            run_id="run",
            phase="phase2",
            provider="ollama",
            model_name="demo",
            split="test",
            example_id="rmpr-0001",
            prompt_version="recipe-mpr-mc-v2",
            raw_response="A",
            parsed_choice="A",
            predicted_option_id="a",
            gold_option_id="a",
            is_correct=True,
            latency_ms=10.0,
            parse_status="parsed",
        ),
        PredictionRecord(
            run_id="run",
            phase="phase2",
            provider="ollama",
            model_name="demo",
            split="test",
            example_id="rmpr-0002",
            prompt_version="recipe-mpr-mc-v2",
            raw_response="A or B",
            parsed_choice=None,
            predicted_option_id=None,
            gold_option_id="b2",
            is_correct=False,
            latency_ms=20.0,
            parse_status="ambiguous",
        ),
    )

    metrics = summarize_prediction_metrics(records, dataset)

    assert metrics["accuracy"] == 0.5
    assert metrics["correct_count"] == 1
    assert metrics["parse_failure_count"] == 1
    assert metrics["accuracy_ci95_low"] <= metrics["accuracy"] <= metrics["accuracy_ci95_high"]


def test_pairwise_prediction_comparison_counts_discordant_examples() -> None:
    left = (
        PredictionRecord(
            run_id="left",
            phase="phase2",
            provider="ollama",
            model_name="left",
            split="test",
            example_id="rmpr-0001",
            prompt_version="recipe-mpr-mc-v2",
            raw_response="A",
            parsed_choice="A",
            predicted_option_id="a",
            gold_option_id="a",
            is_correct=True,
            latency_ms=None,
        ),
    )
    right = (
        PredictionRecord(
            run_id="right",
            phase="phase2",
            provider="ollama",
            model_name="right",
            split="test",
            example_id="rmpr-0001",
            prompt_version="recipe-mpr-mc-v2",
            raw_response="B",
            parsed_choice="B",
            predicted_option_id="b",
            gold_option_id="a",
            is_correct=False,
            latency_ms=None,
        ),
    )

    comparison = summarize_pairwise_prediction_comparison(left, right)

    assert comparison["left_only_correct"] == 1
    assert comparison["right_only_correct"] == 0
