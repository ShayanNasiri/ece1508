"""Tests for llm_evaluation/utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_EVAL = ROOT / "llm_evaluation"
if str(LLM_EVAL) not in sys.path:
    sys.path.insert(0, str(LLM_EVAL))

from utils import compute_accuracy


def _make_item(query_types: dict[str, int]) -> dict:
    """Build a minimal dataset item with the given query_type flags."""
    return {"query_type": query_types}


class TestComputeAccuracyOverall:
    def test_all_correct(self):
        preds = ["r1", "r2", "r3"]
        golds = ["r1", "r2", "r3"]
        items = [_make_item({}) for _ in range(3)]
        result = compute_accuracy(preds, golds, items)
        assert result["overall"] == 1.0
        assert result["total_correct"] == 3
        assert result["total"] == 3
        assert result["parse_failures"] == 0

    def test_all_wrong(self):
        preds = ["r2", "r3", "r1"]
        golds = ["r1", "r2", "r3"]
        items = [_make_item({}) for _ in range(3)]
        result = compute_accuracy(preds, golds, items)
        assert result["overall"] == 0.0
        assert result["total_correct"] == 0

    def test_partial_accuracy(self):
        preds = ["r1", "wrong", "r3", None]
        golds = ["r1", "r2", "r3", "r4"]
        items = [_make_item({}) for _ in range(4)]
        result = compute_accuracy(preds, golds, items)
        assert result["overall"] == 0.5
        assert result["total_correct"] == 2
        assert result["parse_failures"] == 1


class TestComputeAccuracyPerType:
    def test_breakdown_by_query_type(self):
        preds = ["r1", "wrong"]
        golds = ["r1", "r2"]
        items = [
            _make_item({"Specific": 1, "Commonsense": 0}),
            _make_item({"Specific": 1, "Commonsense": 1}),
        ]
        result = compute_accuracy(preds, golds, items)
        assert result["Specific"]["accuracy"] == 0.5
        assert result["Specific"]["correct"] == 1
        assert result["Specific"]["total"] == 2
        assert result["Commonsense"]["accuracy"] == 0.0
        assert result["Commonsense"]["correct"] == 0
        assert result["Commonsense"]["total"] == 1

    def test_missing_query_type_gives_zero(self):
        preds = ["r1"]
        golds = ["r1"]
        items = [_make_item({"Specific": 1})]
        result = compute_accuracy(preds, golds, items)
        # Types not present in any item get total=0
        assert result["Negated"]["accuracy"] == 0
        assert result["Negated"]["total"] == 0


class TestComputeAccuracyEdgeCases:
    def test_empty_inputs(self):
        result = compute_accuracy([], [], [])
        assert result["overall"] == 0
        assert result["total"] == 0
        assert result["parse_failures"] == 0

    def test_all_parse_failures(self):
        preds = [None, None, None]
        golds = ["r1", "r2", "r3"]
        items = [_make_item({}) for _ in range(3)]
        result = compute_accuracy(preds, golds, items)
        assert result["parse_failures"] == 3
        assert result["overall"] == 0.0
