"""Tests for recipe_mpr_qa.evaluation.results.

These tests cover the small helper that loads a saved evaluation result JSON
(from llm_evaluation/results/) and exposes a structured summary. The helper is
the only new code added for the demo notebook, and it lives in the package so
both the notebook and any future tooling can reuse it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipe_mpr_qa.evaluation.results import (
    EvaluationResultSummary,
    load_evaluation_result,
)


def _write_result(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _full_metrics(overall: float = 0.8266666666666667) -> dict:
    return {
        "overall": overall,
        "total_correct": 62,
        "total": 75,
        "parse_failures": 0,
        "Specific": {"accuracy": 0.95, "correct": 22, "total": 23},
        "Commonsense": {"accuracy": 0.77, "correct": 30, "total": 39},
        "Negated": {"accuracy": 0.82, "correct": 14, "total": 17},
        "Analogical": {"accuracy": 1.0, "correct": 4, "total": 4},
        "Temporal": {"accuracy": 0.75, "correct": 3, "total": 4},
    }


class TestLoadEvaluationResult:
    def test_returns_summary_with_top_level_fields(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "fake-model",
                "split": "test",
                "eval_mode": "generative",
                "metrics": _full_metrics(),
                "results": [],
            },
        )

        summary = load_evaluation_result(path)

        assert isinstance(summary, EvaluationResultSummary)
        assert summary.model == "fake-model"
        assert summary.split == "test"
        assert summary.eval_mode == "generative"
        assert summary.overall_accuracy == pytest.approx(0.8266666666666667)
        assert summary.total_correct == 62
        assert summary.total == 75
        assert summary.parse_failures == 0

    def test_per_query_type_breakdown_is_exposed(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "fake-model",
                "split": "test",
                "eval_mode": "generative",
                "metrics": _full_metrics(),
                "results": [],
            },
        )

        summary = load_evaluation_result(path)

        assert set(summary.per_query_type.keys()) == {
            "Specific",
            "Commonsense",
            "Negated",
            "Analogical",
            "Temporal",
        }
        assert summary.per_query_type["Specific"]["correct"] == 22
        assert summary.per_query_type["Analogical"]["accuracy"] == pytest.approx(1.0)

    def test_eval_mode_defaults_to_generative_when_missing(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "fake-model",
                "split": "test",
                "metrics": _full_metrics(),
                "results": [],
            },
        )

        summary = load_evaluation_result(path)

        assert summary.eval_mode == "generative"

    def test_missing_metrics_raises(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "fake-model",
                "split": "test",
                "results": [],
            },
        )

        with pytest.raises(KeyError):
            load_evaluation_result(path)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "fake-model",
                "split": "test",
                "eval_mode": "generative",
                "metrics": _full_metrics(),
                "results": [],
            },
        )

        summary = load_evaluation_result(str(path))

        assert summary.total == 75


class TestFormatReport:
    def test_report_contains_key_fields(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "smollm2-1.7b-finetuned-synthetic",
                "split": "test",
                "eval_mode": "generative",
                "metrics": _full_metrics(),
                "results": [],
            },
        )
        summary = load_evaluation_result(path)

        report = summary.format_report()

        assert "smollm2-1.7b-finetuned-synthetic" in report
        assert "test" in report
        assert "82.7%" in report or "82.67%" in report
        assert "62/75" in report
        # Per-type lines must be present
        assert "Specific" in report
        assert "Commonsense" in report
        assert "Temporal" in report

    def test_report_is_a_string_with_multiple_lines(self, tmp_path: Path) -> None:
        path = _write_result(
            tmp_path,
            {
                "model": "fake-model",
                "split": "test",
                "eval_mode": "generative",
                "metrics": _full_metrics(),
                "results": [],
            },
        )
        summary = load_evaluation_result(path)

        report = summary.format_report()

        assert isinstance(report, str)
        assert report.count("\n") >= 5
