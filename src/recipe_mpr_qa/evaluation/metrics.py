from __future__ import annotations

from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.models import PreparedDataset, RecipeExample
from recipe_mpr_qa.evaluation.records import JudgmentRecord, PredictionRecord


def _dataset_lookup(dataset: PreparedDataset) -> dict[str, RecipeExample]:
    return {example.example_id: example for example in dataset.examples}


def summarize_prediction_metrics(
    records: Sequence[PredictionRecord],
    dataset: PreparedDataset,
) -> dict[str, Any]:
    if not records:
        return {
            "example_count": 0,
            "accuracy": 0.0,
            "per_query_type_accuracy": {},
            "per_signature_accuracy": {},
        }

    examples_by_id = _dataset_lookup(dataset)
    correct = 0
    query_type_counts: dict[str, list[int]] = {}
    signature_counts: dict[str, list[int]] = {}

    for record in records:
        is_correct = int(record.is_correct)
        correct += is_correct
        example = examples_by_id[record.example_id]
        for query_type, active in example.query_type_flags.items():
            if active:
                query_type_counts.setdefault(query_type, []).append(is_correct)
        signature_counts.setdefault(example.query_type_signature, []).append(is_correct)

    return {
        "example_count": len(records),
        "accuracy": correct / len(records),
        "per_query_type_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(query_type_counts.items())
        },
        "per_signature_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(signature_counts.items())
        },
    }


def summarize_judgment_metrics(records: Sequence[JudgmentRecord]) -> dict[str, Any]:
    if not records:
        return {
            "judgment_count": 0,
            "average_scores": {},
            "verdict_distribution": {},
        }
    verdict_distribution: dict[str, int] = {}
    ingredient_scores = []
    constraint_scores = []
    reasoning_scores = []
    for record in records:
        verdict_distribution[record.overall_verdict] = (
            verdict_distribution.get(record.overall_verdict, 0) + 1
        )
        ingredient_scores.append(record.ingredient_alignment)
        constraint_scores.append(record.constraint_satisfaction)
        reasoning_scores.append(record.reasoning_quality)
    count = len(records)
    return {
        "judgment_count": count,
        "average_scores": {
            "ingredient_alignment": sum(ingredient_scores) / count,
            "constraint_satisfaction": sum(constraint_scores) / count,
            "reasoning_quality": sum(reasoning_scores) / count,
        },
        "verdict_distribution": dict(sorted(verdict_distribution.items())),
    }
