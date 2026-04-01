from __future__ import annotations

import json
from pathlib import Path

from recipe_mpr_qa.cli import main
from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.synthetic import read_synthetic_full_dataset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"


class _FakeOpenAIClient:
    def create_structured_output(self, *, schema_name: str, input_text: str, **_: object) -> dict[str, object]:
        payload = json.loads(input_text)
        if schema_name == "synthetic_query_generation":
            parent_query = payload["parent_example"]["query"]
            return {
                "candidates": [
                    {
                        "query": f"Which recipe best matches this request: {parent_query}",
                        "intended_query_type_target": payload["target_query_type_signature"],
                        "method_tag": "answer_aware_rewrite",
                        "rationale": "Preserves the same request while changing framing.",
                    }
                ]
            }
        if schema_name == "synthetic_query_review":
            return {
                "review_status": "approved",
                "review_scores": {
                    "semantic_preservation": 0.96,
                    "constraint_preservation": 0.95,
                    "answer_preservation": 0.99,
                    "leakage_risk": 0.05,
                    "language_quality": 0.97,
                },
                "failure_modes": [],
                "review_summary": "Approved synthetic query.",
            }
        if schema_name == "synthetic_full_generation":
            target_signature = payload["target_query_type_signature"]
            return {
                "candidates": [
                    {
                        "query": "Which recipe suits someone who wants a quick shrimp pasta dinner?",
                        "options": [
                            "Quick shrimp spaghetti with garlic and parsley",
                            "Slow-braised beef short ribs with potatoes",
                            "Chocolate pudding with whipped cream",
                            "Roasted cauliflower with paprika",
                            "Creamy mushroom soup with thyme",
                        ],
                        "answer_index": 0,
                        "query_type_flags": {
                            "Specific": True,
                            "Commonsense": True,
                            "Negated": False,
                            "Analogical": False,
                            "Temporal": True,
                        },
                        "correctness_explanation": {
                            "quick shrimp pasta dinner": "quick shrimp spaghetti"
                        },
                        "intended_query_type_target": target_signature,
                        "distractor_generation_method": "template_conditioned_generation",
                        "rationale": "One answer clearly fits; distractors are plausible but wrong.",
                    }
                ]
            }
        if schema_name == "synthetic_full_review":
            return {
                "review_status": "approved",
                "review_scores": {
                    "single_answer_validity": 0.99,
                    "distractor_plausibility": 0.94,
                    "leakage_risk": 0.05,
                    "distribution_fit": 0.88,
                    "language_quality": 0.97,
                },
                "distribution_fit_score": 0.88,
                "failure_modes": [],
                "review_summary": "Approved synthetic full record.",
            }
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


def test_cli_synthetic_query_flow(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("recipe_mpr_qa.cli._build_openai_client", lambda: _FakeOpenAIClient())
    candidate_path = tmp_path / "query_candidates.jsonl"
    reviewed_path = tmp_path / "query_reviewed.jsonl"
    approved_path = tmp_path / "query_approved.jsonl"
    train_path = tmp_path / "query_train.jsonl"

    assert main(
        [
            "generate-synthetic-query",
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--output",
            str(candidate_path),
            "--limit",
            "2",
            "--max-candidates-per-parent",
            "1",
        ]
    ) == 0
    assert main(
        [
            "review-synthetic-query",
            "--input",
            str(candidate_path),
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--output",
            str(reviewed_path),
        ]
    ) == 0
    assert main(
        [
            "approve-synthetic-query",
            "--input",
            str(reviewed_path),
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--output",
            str(approved_path),
            "--approval-batch-id",
            "query-batch",
            "--max-examples",
            "1",
        ]
    ) == 0
    assert main(
        [
            "build-synthetic-train",
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--query-approved-path",
            str(approved_path),
            "--output",
            str(train_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "query-batch" in captured.out
    train_dataset = read_prepared_dataset(train_path)
    assert len(train_dataset.examples) == 1
    assert train_dataset.examples[0].source_metadata["synthetic_mode"] == "query_only"


def test_cli_synthetic_full_flow(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("recipe_mpr_qa.cli._build_openai_client", lambda: _FakeOpenAIClient())
    candidate_path = tmp_path / "full_candidates.jsonl"
    reviewed_path = tmp_path / "full_reviewed.jsonl"
    approved_path = tmp_path / "full_approved.jsonl"
    train_path = tmp_path / "mixed_train.jsonl"

    assert main(
        [
            "generate-synthetic-full",
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--output",
            str(candidate_path),
            "--limit",
            "1",
        ]
    ) == 0
    assert main(
        [
            "review-synthetic-full",
            "--input",
            str(candidate_path),
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--output",
            str(reviewed_path),
        ]
    ) == 0
    assert main(
        [
            "approve-synthetic-full",
            "--input",
            str(reviewed_path),
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--output",
            str(approved_path),
            "--approval-batch-id",
            "full-batch",
        ]
    ) == 0
    assert main(
        [
            "build-synthetic-train",
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--full-approved-path",
            str(approved_path),
            "--output",
            str(train_path),
            "--target-ratio",
            "0.05",
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "full-batch" in captured.out
    full_dataset = read_synthetic_full_dataset(approved_path)
    assert len(full_dataset.records) == 1
    train_dataset = read_prepared_dataset(train_path)
    assert len(train_dataset.examples) == 1
    assert train_dataset.examples[0].source_metadata["synthetic_mode"] == "full_generation"
