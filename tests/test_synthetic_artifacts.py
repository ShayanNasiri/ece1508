from __future__ import annotations

from pathlib import Path

from recipe_mpr_qa.data.models import PreparedDataset
from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.synthetic import (
    QUERY_ONLY_SYNTHETIC_MODE,
    SyntheticFullDataset,
    build_synthetic_full_record,
    build_synthetic_query_example,
    build_synthetic_training_artifact,
    convert_synthetic_full_records_to_examples,
    read_synthetic_full_dataset,
    validate_synthetic_query_dataset,
    write_synthetic_full_dataset,
    write_synthetic_query_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"


def test_build_synthetic_query_example_marks_required_provenance() -> None:
    parent_example = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[0]

    synthetic_example = build_synthetic_query_example(
        parent_example,
        query="Need a warm dish with oysters",
        candidate_index=1,
        generator_model="gpt-5.4-mini",
        generation_prompt_version="synthetic-query-v1",
        created_at="2026-03-31T00:00:00Z",
        intended_query_type_target=parent_example.query_type_signature,
        generation_method="constrained_paraphrase",
    )

    dataset = validate_synthetic_query_dataset(
        PreparedDataset(examples=(synthetic_example,), metadata={})
    )
    assert dataset.examples[0].source_metadata["synthetic_mode"] == QUERY_ONLY_SYNTHETIC_MODE
    assert dataset.examples[0].source_metadata["parent_example_id"] == parent_example.example_id


def test_synthetic_full_dataset_roundtrip_and_conversion(tmp_path: Path) -> None:
    record = build_synthetic_full_record(
        example_id="synfull-0001",
        query="Which recipe works for someone who wants a quick salmon dinner?",
        option_texts=(
            "Fast roasted salmon with lemon and dill",
            "Slow-cooked beef stew with potatoes",
            "Chocolate pudding with whipped cream",
            "Garlic bread with mozzarella",
            "Vegetable curry with chickpeas",
        ),
        answer_index=0,
        query_type_flags={
            "Specific": True,
            "Commonsense": True,
            "Negated": False,
            "Analogical": False,
            "Temporal": True,
        },
        correctness_explanation={"quick salmon dinner": "fast roasted salmon"},
        generator_model="gpt-5.4-mini",
        generation_prompt_version="synthetic-full-v1",
        created_at="2026-03-31T00:00:00Z",
        intended_query_type_target="Specific|Commonsense|Temporal",
        seed_example_ids=("rmpr-0001",),
        distractor_generation_method="template_conditioned_generation",
        review_status="approved",
        review_scores={
            "single_answer_validity": 0.99,
            "distractor_plausibility": 0.95,
            "leakage_risk": 0.05,
            "distribution_fit": 0.88,
            "language_quality": 0.97,
        },
        distribution_fit_score=0.88,
    )
    dataset = SyntheticFullDataset(records=(record,), metadata={})
    output_path = tmp_path / "synthetic_full.jsonl"

    write_synthetic_full_dataset(dataset, output_path)
    loaded_dataset = read_synthetic_full_dataset(output_path)
    converted_examples = convert_synthetic_full_records_to_examples(loaded_dataset.records)

    assert len(loaded_dataset.records) == 1
    assert loaded_dataset.records[0].provenance["seed_example_ids"] == ("rmpr-0001",)
    assert converted_examples[0].example_id == "synfull-0001"


def test_build_synthetic_training_artifact_combines_query_and_full_examples(tmp_path: Path) -> None:
    dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)
    parent_example = dataset.examples[0]
    query_example = build_synthetic_query_example(
        parent_example,
        query="Need a comforting oyster soup idea",
        candidate_index=1,
        generator_model="gpt-5.4-mini",
        generation_prompt_version="synthetic-query-v1",
        created_at="2026-03-31T00:00:00Z",
        intended_query_type_target=parent_example.query_type_signature,
        generation_method="constrained_paraphrase",
        review_status="approved",
        review_scores={
            "semantic_preservation": 0.95,
            "constraint_preservation": 0.95,
            "answer_preservation": 0.99,
            "leakage_risk": 0.05,
            "language_quality": 0.96,
        },
        approval_batch_id="pilot-q",
    )
    query_path = tmp_path / "approved_query.jsonl"
    write_synthetic_query_dataset(PreparedDataset(examples=(query_example,), metadata={}), query_path)

    full_record = build_synthetic_full_record(
        example_id="synfull-0001",
        query="Which recipe fits a fast shrimp pasta dinner?",
        option_texts=(
            "Quick shrimp spaghetti with garlic and parsley",
            "Slow-braised beef short ribs",
            "Vanilla cupcakes with frosting",
            "Roasted cauliflower salad",
            "Mushroom risotto cooked slowly",
        ),
        answer_index=0,
        query_type_flags={
            "Specific": True,
            "Commonsense": True,
            "Negated": False,
            "Analogical": False,
            "Temporal": True,
        },
        correctness_explanation={"fast shrimp pasta dinner": "quick shrimp spaghetti"},
        generator_model="gpt-5.4-mini",
        generation_prompt_version="synthetic-full-v1",
        created_at="2026-03-31T00:00:00Z",
        intended_query_type_target="Specific|Commonsense|Temporal",
        seed_example_ids=("rmpr-0005",),
        distractor_generation_method="template_conditioned_generation",
        review_status="approved",
        review_scores={
            "single_answer_validity": 0.99,
            "distractor_plausibility": 0.95,
            "leakage_risk": 0.05,
            "distribution_fit": 0.88,
            "language_quality": 0.97,
        },
        approval_batch_id="pilot-f",
        distribution_fit_score=0.88,
    )
    full_path = tmp_path / "approved_full.jsonl"
    write_synthetic_full_dataset(SyntheticFullDataset(records=(full_record,), metadata={}), full_path)

    output_path = tmp_path / "train_synthetic.jsonl"
    summary = build_synthetic_training_artifact(
        dataset_path=PROCESSED_DATASET_PATH,
        split_manifest_path=SPLIT_MANIFEST_PATH,
        query_approved_path=query_path,
        full_approved_path=full_path,
        output_path=output_path,
        max_query_examples=1,
        max_full_examples=1,
    )
    output_dataset = read_prepared_dataset(output_path)

    assert summary["synthetic_query_count"] == 1
    assert summary["synthetic_full_count"] == 1
    assert len(output_dataset.examples) == 2
