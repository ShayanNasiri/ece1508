"""Loaders for saved evaluation result JSON files.

The multiple-choice evaluation script (``llm_evaluation/mc_eval.py``) writes
result files under ``llm_evaluation/results/`` with a stable schema:

    {
        "model": "<model id or local path>",
        "split": "<train|validation|test>",
        "eval_mode": "<generative|loglikelihood>",
        "metrics": {
            "overall": <float>,
            "total_correct": <int>,
            "total": <int>,
            "parse_failures": <int>,
            "Specific":    {"accuracy": ..., "correct": ..., "total": ...},
            "Commonsense": {...},
            "Negated":     {...},
            "Analogical":  {...},
            "Temporal":    {...},
        },
        "results": [...per-example records...]
    }

This module exposes a small loader so the demo notebook (and any future tools)
can read those files into a structured object instead of digging through nested
dictionaries by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES


@dataclass(frozen=True)
class EvaluationResultSummary:
    """Structured view of a saved evaluation result file."""

    model: str
    split: str
    eval_mode: str
    overall_accuracy: float
    total_correct: int
    total: int
    parse_failures: int
    per_query_type: Mapping[str, Mapping[str, Any]]

    def format_report(self) -> str:
        """Render a short multi-line report suitable for printing in a notebook."""
        lines = [
            f"Model:           {self.model}",
            f"Split:           {self.split} (mode={self.eval_mode})",
            (
                f"Overall:         {self.overall_accuracy:.1%} "
                f"({self.total_correct}/{self.total})"
            ),
            f"Parse failures:  {self.parse_failures}",
            "Per query type:",
        ]
        for query_type in QUERY_TYPE_NAMES:
            stats = self.per_query_type.get(query_type)
            if not stats:
                continue
            accuracy = float(stats.get("accuracy", 0.0))
            correct = int(stats.get("correct", 0))
            total = int(stats.get("total", 0))
            lines.append(
                f"  {query_type:<12} {accuracy:.1%} ({correct}/{total})"
            )
        return "\n".join(lines)


def load_evaluation_result(path: Path | str) -> EvaluationResultSummary:
    """Load a saved evaluation result JSON file into an EvaluationResultSummary.

    Args:
        path: Path to a JSON file produced by ``llm_evaluation/mc_eval.py``.

    Returns:
        EvaluationResultSummary with the top-level metadata and metrics. The
        per-example ``results`` array is intentionally not exposed here because
        the demo only needs the summary; downstream code that needs records can
        load the JSON directly.

    Raises:
        KeyError: if required keys (``model``, ``split``, ``metrics``,
            ``metrics.overall``, ``metrics.total_correct``, ``metrics.total``)
            are missing.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    per_query_type = {
        query_type: dict(metrics[query_type])
        for query_type in QUERY_TYPE_NAMES
        if query_type in metrics
    }
    return EvaluationResultSummary(
        model=payload["model"],
        split=payload["split"],
        eval_mode=payload.get("eval_mode", "generative"),
        overall_accuracy=float(metrics["overall"]),
        total_correct=int(metrics["total_correct"]),
        total=int(metrics["total"]),
        parse_failures=int(metrics.get("parse_failures", 0)),
        per_query_type=per_query_type,
    )
