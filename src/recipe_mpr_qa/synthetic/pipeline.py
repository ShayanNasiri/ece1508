from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from recipe_mpr_qa.artifacts import write_json
from recipe_mpr_qa.data.loaders import get_split_examples, load_dataset, load_split_manifest
from recipe_mpr_qa.data.models import DatasetValidationError, PreparedDataset, RecipeExample
from recipe_mpr_qa.evaluation.records import PredictionRecord, read_prediction_records
from recipe_mpr_qa.llm.inference import run_llm_predictions
from recipe_mpr_qa.llm.prompts import AUGMENTATION_PROMPT_SPEC, build_augmentation_prompt, parse_augmentation_response
from recipe_mpr_qa.synthetic.artifacts import (
    APPROVED_SYNTHETIC_STAGE,
    CANDIDATE_SYNTHETIC_STAGE,
    DEFAULT_SYNTHETIC_SELECTION_SEED,
    QUERY_ONLY_SYNTHETIC_MODE,
    REVIEWED_SYNTHETIC_STAGE,
    TRAIN_READY_SYNTHETIC_STAGE,
    build_synthetic_query_example,
    deterministic_sample,
    is_near_duplicate,
    normalize_text_key,
    read_synthetic_query_dataset,
    stratified_sample_examples,
    validate_synthetic_query_dataset,
    with_stage,
    write_synthetic_query_dataset,
)


def _authentic_query_keys(dataset: PreparedDataset) -> set[str]:
    return {normalize_text_key(example.query) for example in dataset.examples}


def generate_synthetic_query_candidates(
    *,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    client,
    model_name: str,
    parent_limit: int = 75,
    variants_per_example: int = 2,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
    temperature: float = 0.2,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_examples = get_split_examples(dataset, split_manifest, "train")
    selected_parents = stratified_sample_examples(train_examples, limit=parent_limit, seed=seed)
    seen_query_keys = _authentic_query_keys(dataset)
    created_examples: list[RecipeExample] = []
    rejection_counts: Counter[str] = Counter()
    from recipe_mpr_qa.benchmark.provenance import utc_now_iso

    created_at = utc_now_iso()
    for parent_example in selected_parents:
        prompt = build_augmentation_prompt(
            parent_example,
            requested_count=variants_per_example,
            prompt_spec=AUGMENTATION_PROMPT_SPEC,
        )
        response_text = client.generate(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
        )
        rewrites = parse_augmentation_response(response_text)
        accepted_for_parent = 0
        for rewrite in rewrites:
            if accepted_for_parent >= variants_per_example:
                break
            normalized_query = normalize_text_key(rewrite)
            if not normalized_query:
                rejection_counts["empty"] += 1
                continue
            if normalized_query in seen_query_keys:
                rejection_counts["duplicate_to_authentic_or_candidate"] += 1
                continue
            accepted_for_parent += 1
            seen_query_keys.add(normalized_query)
            created_examples.append(
                build_synthetic_query_example(
                    parent_example,
                    query=rewrite,
                    candidate_index=accepted_for_parent,
                    generator_model=model_name,
                    generation_prompt_version=AUGMENTATION_PROMPT_SPEC.version,
                    created_at=created_at,
                    stage=CANDIDATE_SYNTHETIC_STAGE,
                )
            )
    synthetic_dataset = PreparedDataset(
        examples=tuple(created_examples),
        metadata={
            "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
            "synthetic_stage": CANDIDATE_SYNTHETIC_STAGE,
            "generator_model_name": model_name,
            "generation_prompt_version": AUGMENTATION_PROMPT_SPEC.version,
            "parent_limit": parent_limit,
            "variants_per_example": variants_per_example,
            "parent_count": len(selected_parents),
            "candidate_count": len(created_examples),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
    )
    write_synthetic_query_dataset(synthetic_dataset, output_path)
    return dict(synthetic_dataset.metadata)


def review_synthetic_query_candidates(
    *,
    input_path: Path | str,
    reviewed_output_path: Path | str,
    review_predictions_path: Path | str,
    reviewer_client,
    reviewer_provider: str,
    reviewer_model_name: str,
    split: str = "train",
    temperature: float = 0.0,
    resume: bool = True,
) -> dict[str, Any]:
    candidate_dataset = read_synthetic_query_dataset(input_path)
    validate_synthetic_query_dataset(candidate_dataset, expected_stage=CANDIDATE_SYNTHETIC_STAGE)
    prediction_records = run_llm_predictions(
        examples=candidate_dataset.examples,
        client=reviewer_client,
        run_id=f"review-{reviewer_model_name.replace('/', '_').replace(':', '_')}",
        provider=reviewer_provider,
        model_name=reviewer_model_name,
        split=split,
        output_path=review_predictions_path,
        temperature=temperature,
        resume=resume,
    )
    prediction_by_id = {record.example_id: record for record in prediction_records}
    reviewed_examples = []
    review_status_counts: Counter[str] = Counter()
    for example in candidate_dataset.examples:
        review_record = prediction_by_id[example.example_id]
        review_is_consistent = review_record.predicted_option_id == example.answer_option_id
        review_status = "approved" if review_is_consistent else "rejected"
        review_status_counts[review_status] += 1
        reviewed_examples.append(
            with_stage(
                example,
                stage=REVIEWED_SYNTHETIC_STAGE,
                extra_metadata={
                    "review_model_name": reviewer_model_name,
                    "review_provider": reviewer_provider,
                    "review_prediction_option_id": review_record.predicted_option_id,
                    "review_parse_status": review_record.parse_status,
                    "review_is_consistent": review_is_consistent,
                    "review_status": review_status,
                },
            )
        )
    reviewed_dataset = PreparedDataset(
        examples=tuple(reviewed_examples),
        metadata={
            **dict(candidate_dataset.metadata),
            "synthetic_stage": REVIEWED_SYNTHETIC_STAGE,
            "review_model_name": reviewer_model_name,
            "review_provider": reviewer_provider,
            "review_prediction_path": Path(review_predictions_path).as_posix(),
            "review_status_counts": dict(sorted(review_status_counts.items())),
        },
    )
    write_synthetic_query_dataset(reviewed_dataset, reviewed_output_path)
    return dict(reviewed_dataset.metadata)


def _contains_option_leakage(example: RecipeExample, parent_example: RecipeExample) -> bool:
    query_key = normalize_text_key(example.query)
    if any(token in query_key for token in (" a ", " b ", " c ", " d ", " e ")):
        return True
    for option in parent_example.options:
        option_key = normalize_text_key(option.text)
        if option_key and option_key in query_key:
            return True
    return False


def approve_synthetic_query_candidates(
    *,
    input_path: Path | str,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    approval_batch_id: str,
    max_examples: int | None = None,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
    near_duplicate_threshold: float = 0.95,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_example_ids = set(split_manifest.splits["train"])
    authentic_keys = _authentic_query_keys(dataset)
    examples_by_id = {example.example_id: example for example in dataset.examples}
    reviewed_dataset = read_synthetic_query_dataset(input_path)
    validate_synthetic_query_dataset(reviewed_dataset, expected_stage=REVIEWED_SYNTHETIC_STAGE)
    approved_examples: list[RecipeExample] = []
    rejection_counts: Counter[str] = Counter()
    for candidate in reviewed_dataset.examples:
        metadata = dict(candidate.source_metadata)
        parent_example_id = str(metadata["parent_example_id"])
        parent_example = examples_by_id.get(parent_example_id)
        if parent_example is None:
            rejection_counts["missing_parent"] += 1
            continue
        if parent_example_id not in train_example_ids:
            rejection_counts["parent_not_in_train"] += 1
            continue
        if metadata.get("review_status") != "approved" or not metadata.get("review_is_consistent"):
            rejection_counts["review_failed"] += 1
            continue
        normalized_query = normalize_text_key(candidate.query)
        if not normalized_query:
            rejection_counts["empty"] += 1
            continue
        if normalized_query in authentic_keys:
            rejection_counts["exact_duplicate_to_authentic"] += 1
            continue
        if is_near_duplicate(candidate.query, parent_example.query, threshold=near_duplicate_threshold):
            rejection_counts["near_duplicate_to_parent"] += 1
            continue
        if any(
            is_near_duplicate(candidate.query, approved.query, threshold=near_duplicate_threshold)
            for approved in approved_examples
        ):
            rejection_counts["near_duplicate_to_approved"] += 1
            continue
        if _contains_option_leakage(candidate, parent_example):
            rejection_counts["option_leakage"] += 1
            continue
        approved_examples.append(
            with_stage(
                candidate,
                stage=APPROVED_SYNTHETIC_STAGE,
                extra_metadata={
                    "approval_batch_id": approval_batch_id,
                },
            )
        )
    if max_examples is not None:
        approved_examples = list(
            deterministic_sample(
                approved_examples,
                limit=max_examples,
                seed=seed,
                key_fn=lambda item: item.example_id,
            )
        )
    approved_dataset = PreparedDataset(
        examples=tuple(approved_examples),
        metadata={
            **dict(reviewed_dataset.metadata),
            "synthetic_stage": APPROVED_SYNTHETIC_STAGE,
            "approval_batch_id": approval_batch_id,
            "approved_count": len(approved_examples),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
    )
    validate_synthetic_query_dataset(approved_dataset, expected_stage=APPROVED_SYNTHETIC_STAGE)
    write_synthetic_query_dataset(approved_dataset, output_path)
    return dict(approved_dataset.metadata)


def build_train_ready_dataset(
    *,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    approved_input_path: Path | str,
    output_path: Path | str,
    manifest_output_path: Path | str,
    synthetic_ratio: float,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
) -> dict[str, Any]:
    if synthetic_ratio < 0:
        raise DatasetValidationError("synthetic_ratio must be >= 0")
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_examples = get_split_examples(dataset, split_manifest, "train")
    approved_dataset = read_synthetic_query_dataset(approved_input_path)
    validate_synthetic_query_dataset(approved_dataset, expected_stage=APPROVED_SYNTHETIC_STAGE)
    synthetic_target = int(round(len(train_examples) * synthetic_ratio))
    sampled_examples = deterministic_sample(
        approved_dataset.examples,
        limit=synthetic_target,
        seed=seed,
        key_fn=lambda item: item.example_id,
    )
    train_ready_examples = [
        with_stage(
            example,
            stage=TRAIN_READY_SYNTHETIC_STAGE,
            extra_metadata={
                "train_ready_ratio": synthetic_ratio,
            },
        )
        for example in sampled_examples
    ]
    train_ready_dataset = PreparedDataset(
        examples=tuple(train_ready_examples),
        metadata={
            "train_example_count": len(train_examples),
            "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
            "synthetic_ratio": synthetic_ratio,
            "synthetic_target": synthetic_target,
            "synthetic_selected_count": len(train_ready_examples),
            "approved_input_path": Path(approved_input_path).as_posix(),
        },
    )
    write_synthetic_query_dataset(train_ready_dataset, output_path)
    write_json(dict(train_ready_dataset.metadata), manifest_output_path)
    return dict(train_ready_dataset.metadata)
