from __future__ import annotations

from pathlib import Path

import pytest

from recipe_mpr_qa.data.models import RecipeOption
from recipe_mpr_qa.llm.prompts import (
    build_augmentation_prompt,
    build_causal_multiple_choice_prompt,
    build_judge_prompt,
    DEFAULT_PROMPT_SPEC,
    JUDGE_PROMPT_SPEC,
    AUGMENTATION_PROMPT_SPEC,
    CAUSAL_SLM_PROMPT_SPEC,
    build_multiple_choice_prompt,
    parse_augmentation_response,
    parse_judge_response,
    parse_multiple_choice_response,
)
from recipe_mpr_qa.data.preparation import read_prepared_dataset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"


def test_build_multiple_choice_prompt_renders_expected_format() -> None:
    options = (
        RecipeOption("id-a", "Option A"),
        RecipeOption("id-b", "Option B"),
        RecipeOption("id-c", "Option C"),
        RecipeOption("id-d", "Option D"),
        RecipeOption("id-e", "Option E"),
    )

    prompt, letter_to_option_id = build_multiple_choice_prompt(
        query="Need a roasted fish recipe",
        options=options,
        prompt_spec=DEFAULT_PROMPT_SPEC,
    )

    assert "Query: Need a roasted fish recipe" in prompt
    assert "A) Option A" in prompt
    assert "E) Option E" in prompt
    assert prompt.endswith("Respond with only the letter (A-E) of the best option.")
    assert letter_to_option_id == {
        "A": "id-a",
        "B": "id-b",
        "C": "id-c",
        "D": "id-d",
        "E": "id-e",
    }


@pytest.mark.parametrize(
    ("response_text", "expected"),
    [
        ("A", "A"),
        ("b", "B"),
        ("Answer: C", "C"),
        ("I choose option d.", "D"),
        ("(E)", "E"),
        ("A) because it matches oysters", "A"),
        ("No valid answer", None),
    ],
)
def test_parse_multiple_choice_response_handles_noisy_output(
    response_text: str, expected: str | None
) -> None:
    assert parse_multiple_choice_response(response_text) == expected


def test_augmentation_prompt_and_parser_round_trip() -> None:
    example = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[0]
    prompt = build_augmentation_prompt(
        example,
        requested_count=2,
        prompt_spec=AUGMENTATION_PROMPT_SPEC,
    )

    assert "Requested rewrites: 2" in prompt
    assert parse_augmentation_response('{"rewrites": ["one", "two"]}') == ("one", "two")


def test_judge_prompt_and_parser_round_trip() -> None:
    example = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[0]
    prompt = build_judge_prompt(
        example=example,
        predicted_option_text=example.options[0].text,
        model_rationale="It fits the query.",
        prompt_spec=JUDGE_PROMPT_SPEC,
    )

    assert example.query in prompt
    parsed = parse_judge_response(
        '{"ingredient_alignment": 5, "constraint_satisfaction": 4, '
        '"reasoning_quality": 3, "overall_verdict": "partially_correct", '
        '"rationale": "Mostly aligned."}'
    )
    assert parsed["overall_verdict"] == "partially_correct"


def test_causal_prompt_matches_chat_style_letter_response() -> None:
    options = (
        RecipeOption("id-a", "Option A"),
        RecipeOption("id-b", "Option B"),
        RecipeOption("id-c", "Option C"),
        RecipeOption("id-d", "Option D"),
        RecipeOption("id-e", "Option E"),
    )

    prompt, letter_to_option_id = build_causal_multiple_choice_prompt(
        query="I want a quick vegetarian breakfast",
        options=options,
        prompt_spec=CAUSAL_SLM_PROMPT_SPEC,
    )

    assert "User request: I want a quick vegetarian breakfast" in prompt
    assert "Reply with only one letter: A, B, C, D, or E." in prompt
    assert letter_to_option_id["B"] == "id-b"
