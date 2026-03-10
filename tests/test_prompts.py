from __future__ import annotations

import pytest

from recipe_mpr_qa.data.models import RecipeOption
from recipe_mpr_qa.formats import (
    DEFAULT_PROMPT_SPEC,
    build_multiple_choice_prompt,
    parse_multiple_choice_response,
)


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
