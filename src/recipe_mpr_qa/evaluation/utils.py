from __future__ import annotations

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES


def compute_accuracy(predictions, ground_truth, dataset):
    """Compute overall accuracy and breakdown by query type."""
    assert len(predictions) == len(ground_truth) == len(dataset)

    total_correct = 0
    total = len(predictions)
    type_correct = {query_type: 0 for query_type in QUERY_TYPE_NAMES}
    type_total = {query_type: 0 for query_type in QUERY_TYPE_NAMES}

    for prediction, gold, item in zip(predictions, ground_truth, dataset):
        correct = prediction == gold
        if correct:
            total_correct += 1

        for query_type in QUERY_TYPE_NAMES:
            if item["query_type"].get(query_type, 0) == 1:
                type_total[query_type] += 1
                if correct:
                    type_correct[query_type] += 1

    results = {
        "overall": total_correct / total if total > 0 else 0,
        "total_correct": total_correct,
        "total": total,
        "parse_failures": sum(1 for prediction in predictions if prediction is None),
    }

    for query_type in QUERY_TYPE_NAMES:
        if type_total[query_type] > 0:
            results[query_type] = {
                "accuracy": type_correct[query_type] / type_total[query_type],
                "correct": type_correct[query_type],
                "total": type_total[query_type],
            }
        else:
            results[query_type] = {"accuracy": 0, "correct": 0, "total": 0}

    return results
