"""Tests for mc_eval result row format — Phase 1 output improvements."""
from __future__ import annotations

from recipe_mpr_qa.data.models import RecipeOption, RecipeExample
from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES


# Import the helper we'll create
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_evaluation"))
from mc_eval import build_result_row


def _make_example(example_id="rmpr-0001", answer_option_id="opt-3"):
    """Create a minimal RecipeExample for testing."""
    options = tuple(
        RecipeOption(option_id=f"opt-{i}", text=f"Recipe {i} description")
        for i in range(1, 6)
    )
    flags = {qt: (qt == "Specific") for qt in QUERY_TYPE_NAMES}
    return RecipeExample(
        example_id=example_id,
        query="Find me a quick pasta recipe",
        options=options,
        answer_option_id=answer_option_id,
        query_type_flags=flags,
        correctness_explanation={"reason": "Best match for pasta"},
    )


def _make_letter_to_id():
    return {"A": "opt-1", "B": "opt-2", "C": "opt-3", "D": "opt-4", "E": "opt-5"}


class TestParseFailureFlag:
    """Phase 1 item 1: each result row has a parse_failure boolean."""

    def test_successful_parse_has_parse_failure_false(self):
        example = _make_example()
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=0, example=example, parsed_letter="C",
            letter_to_id=letter_to_id, raw_response="C",
        )
        assert row["parse_failure"] is False

    def test_none_parsed_letter_has_parse_failure_true(self):
        example = _make_example()
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=0, example=example, parsed_letter=None,
            letter_to_id=letter_to_id, raw_response="I don't know the answer",
        )
        assert row["parse_failure"] is True

    def test_parse_failure_with_empty_response(self):
        example = _make_example()
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=0, example=example, parsed_letter=None,
            letter_to_id=letter_to_id, raw_response="",
        )
        assert row["parse_failure"] is True


class TestOptionTextsForWrongAnswers:
    """Phase 1 item 2: wrong answers include option texts in the result row."""

    def test_wrong_answer_includes_options(self):
        example = _make_example(answer_option_id="opt-3")
        letter_to_id = _make_letter_to_id()
        # Predict A (opt-1) but correct is C (opt-3) — wrong answer
        row = build_result_row(
            index=0, example=example, parsed_letter="A",
            letter_to_id=letter_to_id, raw_response="A",
        )
        assert row["is_correct"] is False
        assert "options" in row
        # Options should be a dict mapping letter -> text
        assert row["options"] == {
            "A": "Recipe 1 description",
            "B": "Recipe 2 description",
            "C": "Recipe 3 description",
            "D": "Recipe 4 description",
            "E": "Recipe 5 description",
        }

    def test_correct_answer_has_no_options(self):
        example = _make_example(answer_option_id="opt-3")
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=0, example=example, parsed_letter="C",
            letter_to_id=letter_to_id, raw_response="C",
        )
        assert row["is_correct"] is True
        assert "options" not in row

    def test_parse_failure_includes_options(self):
        """Parse failures are also 'wrong' — should include options for inspection."""
        example = _make_example(answer_option_id="opt-3")
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=0, example=example, parsed_letter=None,
            letter_to_id=letter_to_id, raw_response="gibberish",
        )
        assert row["is_correct"] is False
        assert "options" in row
        assert len(row["options"]) == 5


class TestResultRowBasicFields:
    """Ensure existing fields still work correctly after refactoring."""

    def test_basic_fields_present(self):
        example = _make_example()
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=3, example=example, parsed_letter="C",
            letter_to_id=letter_to_id, raw_response="The answer is C.",
        )
        assert row["index"] == 3
        assert row["example_id"] == "rmpr-0001"
        assert row["query"] == "Find me a quick pasta recipe"
        assert row["correct_answer_id"] == "opt-3"
        assert row["correct_letter"] == "C"
        assert row["predicted_letter"] == "C"
        assert row["predicted_id"] == "opt-3"
        assert row["is_correct"] is True
        assert row["raw_response"] == "The answer is C."

    def test_raw_response_newlines_stripped(self):
        example = _make_example()
        letter_to_id = _make_letter_to_id()
        row = build_result_row(
            index=0, example=example, parsed_letter="C",
            letter_to_id=letter_to_id, raw_response="  C\n\nis my answer  ",
        )
        assert row["raw_response"] == "C  is my answer"
