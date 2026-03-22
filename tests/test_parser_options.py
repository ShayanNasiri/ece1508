"""Tests for parse_multiple_choice_response() with optional options parameter.

Phase 3: recover Pattern 1 parse failures where SmolLM2 outputs
"The best option is <description>" instead of a letter.
"""
from __future__ import annotations

import pytest

from recipe_mpr_qa.formats import parse_multiple_choice_response

OPTIONS = {
    "A": "Cheddar cheese and elbow macaroni with onions",
    "B": "Some other dip recipe",
    "C": "Another cheesy option",
    "D": "A fourth option",
    "E": "A fifth option",
}

BREAKFAST_OPTIONS = {
    "A": "Baked breakfast bars made with nuts and seeds",
    "B": "Scrambled eggs with toast",
    "C": "Oatmeal with fruit",
    "D": "Pancakes with syrup",
    "E": "Yogurt parfait",
}


class TestExistingBehaviourUnchanged:
    """Passing no options should behave exactly as before."""

    def test_plain_letter_still_works(self):
        assert parse_multiple_choice_response("B") == "B"

    def test_noisy_letter_still_works(self):
        assert parse_multiple_choice_response("Answer: C") == "C"

    def test_no_letter_still_returns_none(self):
        assert parse_multiple_choice_response("No valid answer") is None

    def test_empty_still_returns_none(self):
        assert parse_multiple_choice_response("") is None


class TestLetterParsingWithOptionsPassed:
    """Existing letter patterns still work when options dict is provided."""

    def test_plain_letter_with_options(self):
        assert parse_multiple_choice_response("A", options=OPTIONS) == "A"

    def test_noisy_letter_with_options(self):
        assert parse_multiple_choice_response("Answer: B", options=OPTIONS) == "B"


class TestBestOptionIsPattern:
    """Pattern 1 recovery: 'The best option is <text>' fuzzy-matched against options."""

    def test_recovers_from_real_smollm2_failure_cheesy_dip(self):
        # Real SmolLM2 response from rmpr-0069
        raw = (
            "## Solution  The best option is Cheddar cheese and elbow macaroni with onions.  "
            "## Solution  The best option is Cheddar cheese and elbow macaroni with onions."
        )
        assert parse_multiple_choice_response(raw, options=OPTIONS) == "A"

    def test_recovers_from_real_smollm2_failure_breakfast_bars(self):
        # Real SmolLM2 response from rmpr-0335
        raw = (
            "## Solution  The best option is Baked breakfast bars made with nuts and seeds.  "
            "## Explanation  The best option is Baked breakfast bars made with nuts and seeds.  "
            "## Solution  The best option is Baked breakfast bars made with nuts and seeds."
        )
        assert parse_multiple_choice_response(raw, options=BREAKFAST_OPTIONS) == "A"

    def test_case_insensitive_match(self):
        raw = "the best option is cheddar cheese and elbow macaroni with onions."
        assert parse_multiple_choice_response(raw, options=OPTIONS) == "A"

    def test_no_match_returns_none(self):
        raw = "## Solution  The best option is something completely unrelated."
        assert parse_multiple_choice_response(raw, options=OPTIONS) is None

    def test_without_options_returns_none_for_pattern1(self):
        # Without options dict, Pattern 1 responses are unrecoverable — no regression
        raw = "## Solution  The best option is Cheddar cheese and elbow macaroni with onions."
        assert parse_multiple_choice_response(raw) is None

    def test_letter_match_takes_priority_over_text_match(self):
        # If a letter is already parseable, it wins — text fallback is last resort
        raw = "The answer is B. The best option is Cheddar cheese and elbow macaroni with onions."
        assert parse_multiple_choice_response(raw, options=OPTIONS) == "B"
