from __future__ import annotations

import math
from statistics import mean, median
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
            "correct_count": 0,
            "accuracy": 0.0,
            "accuracy_ci95_low": 0.0,
            "accuracy_ci95_high": 0.0,
            "parse_failure_count": 0,
            "parse_failure_rate": 0.0,
            "per_query_type_accuracy": {},
            "per_signature_accuracy": {},
            "latency_ms": {},
        }

    examples_by_id = _dataset_lookup(dataset)
    correct = 0
    parse_failures = 0
    query_type_counts: dict[str, list[int]] = {}
    signature_counts: dict[str, list[int]] = {}
    latencies = []

    for record in records:
        is_correct = int(record.is_correct)
        correct += is_correct
        if record.parse_status not in (None, "parsed", "not_applicable"):
            parse_failures += 1
        if record.latency_ms is not None:
            latencies.append(float(record.latency_ms))
        example = examples_by_id[record.example_id]
        for query_type, active in example.query_type_flags.items():
            if active:
                query_type_counts.setdefault(query_type, []).append(is_correct)
        signature_counts.setdefault(example.query_type_signature, []).append(is_correct)
    ci_low, ci_high = _wilson_interval(correct, len(records))

    return {
        "example_count": len(records),
        "correct_count": correct,
        "accuracy": correct / len(records),
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / len(records),
        "per_query_type_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(query_type_counts.items())
        },
        "per_signature_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(signature_counts.items())
        },
        "latency_ms": _summarize_latencies(latencies),
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


def summarize_pairwise_prediction_comparison(
    left_records: Sequence[PredictionRecord],
    right_records: Sequence[PredictionRecord],
) -> dict[str, Any]:
    left_by_id = {record.example_id: record for record in left_records}
    right_by_id = {record.example_id: record for record in right_records}
    shared_ids = sorted(set(left_by_id).intersection(right_by_id))
    left_only_correct = 0
    right_only_correct = 0
    ties = 0
    for example_id in shared_ids:
        left_correct = bool(left_by_id[example_id].is_correct)
        right_correct = bool(right_by_id[example_id].is_correct)
        if left_correct and not right_correct:
            left_only_correct += 1
        elif right_correct and not left_correct:
            right_only_correct += 1
        else:
            ties += 1
    discordant = left_only_correct + right_only_correct
    p_value = _exact_binomial_two_sided(left_only_correct, discordant)
    return {
        "shared_example_count": len(shared_ids),
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "ties": ties,
        "discordant_count": discordant,
        "exact_two_sided_p_value": p_value,
    }


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = correct / total
    denominator = 1 + (z * z) / total
    center = (p_hat + (z * z) / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1 - p_hat) / total) + ((z * z) / (4 * total * total)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _summarize_latencies(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": float(len(values)),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _exact_binomial_two_sided(successes: int, total: int) -> float:
    if total <= 0:
        return 1.0
    tail_probability = sum(
        math.comb(total, k) * (0.5**total)
        for k in range(0, min(successes, total - successes) + 1)
    )
    return min(1.0, 2 * tail_probability)
