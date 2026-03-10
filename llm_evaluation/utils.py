import json
import csv


QUERY_TYPES = ["Specific", "Commonsense", "Negated", "Analogical", "Temporal"]


def load_dataset(path):
    """Load the 500QA.json dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(predictions, ground_truth, dataset):
    """Compute overall accuracy and breakdown by query type.

    Args:
        predictions: list of predicted recipe IDs (or None for parse failures).
        ground_truth: list of correct recipe IDs.
        dataset: the original dataset list (for query_type info).

    Returns:
        dict with 'overall' accuracy and per-type accuracies.
    """
    assert len(predictions) == len(ground_truth) == len(dataset)

    total_correct = 0
    total = len(predictions)
    type_correct = {qt: 0 for qt in QUERY_TYPES}
    type_total = {qt: 0 for qt in QUERY_TYPES}

    for pred, gt, item in zip(predictions, ground_truth, dataset):
        correct = pred == gt
        if correct:
            total_correct += 1

        for qt in QUERY_TYPES:
            if item["query_type"].get(qt, 0) == 1:
                type_total[qt] += 1
                if correct:
                    type_correct[qt] += 1

    results = {
        "overall": total_correct / total if total > 0 else 0,
        "total_correct": total_correct,
        "total": total,
        "parse_failures": sum(1 for p in predictions if p is None),
    }

    for qt in QUERY_TYPES:
        if type_total[qt] > 0:
            results[qt] = {
                "accuracy": type_correct[qt] / type_total[qt],
                "correct": type_correct[qt],
                "total": type_total[qt],
            }
        else:
            results[qt] = {"accuracy": 0, "correct": 0, "total": 0}

    return results


def save_results(rows, path):
    """Save evaluation results to CSV.

    Args:
        rows: list of dicts with keys like query, raw_response, parsed_letter, etc.
        path: output CSV path.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
