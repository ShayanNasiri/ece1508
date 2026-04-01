"""Synthetic-data generation, review, approval, and train-admission pipeline.

The public functions in this module map directly to the CLI lifecycle:

- generate candidates
- review candidates with a model grader
- approve reviewed artifacts through repo-side gates
- build one train-ready artifact from approved sources

That split is intentional. A reviewed artifact is not automatically training
eligible; approval is where the repo's deterministic admission policy is
applied.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
from recipe_mpr_qa.data.loaders import get_split_examples, load_dataset, load_split_manifest
from recipe_mpr_qa.data.models import DatasetValidationError, PreparedDataset, RecipeExample
from recipe_mpr_qa.data.preparation import write_prepared_dataset
from recipe_mpr_qa.synthetic.artifacts import (
    DEFAULT_SYNTHETIC_SELECTION_SEED,
    FULL_GENERATION_SYNTHETIC_MODE,
    QUERY_ONLY_SYNTHETIC_MODE,
    REVIEW_STATUSES,
    SyntheticFullDataset,
    SyntheticFullRecord,
    build_synthetic_full_record,
    build_synthetic_query_example,
    convert_synthetic_full_records_to_examples,
    deterministic_sample,
    is_near_duplicate,
    normalize_text_key,
    read_synthetic_full_dataset,
    read_synthetic_query_dataset,
    stratified_sample_examples,
    utc_now_iso,
    validate_synthetic_query_dataset,
    write_synthetic_full_dataset,
    write_synthetic_query_dataset,
)
from recipe_mpr_qa.synthetic.openai import OpenAIResponsesClient

DEFAULT_QUERY_GENERATION_MODEL = "gpt-5.4-mini"
DEFAULT_QUERY_REVIEW_MODEL = "gpt-5.4"
DEFAULT_FULL_GENERATION_MODEL = "gpt-5.4-mini"
DEFAULT_FULL_REVIEW_MODEL = "gpt-5.4"

QUERY_GENERATION_PROMPT_VERSION = "synthetic-query-v1"
QUERY_REVIEW_PROMPT_VERSION = "synthetic-query-review-v1"
FULL_GENERATION_PROMPT_VERSION = "synthetic-full-v1"
FULL_REVIEW_PROMPT_VERSION = "synthetic-full-review-v1"


def generate_synthetic_query_candidates(
    *,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    client: OpenAIResponsesClient,
    model: str = DEFAULT_QUERY_GENERATION_MODEL,
    limit: int = 75,
    max_candidates_per_parent: int = 3,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
) -> dict[str, Any]:
    """Generate query-only candidate examples from train parents."""
    if max_candidates_per_parent < 1:
        raise DatasetValidationError("max_candidates_per_parent must be >= 1")
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_examples = get_split_examples(dataset, split_manifest, "train")
    selected_examples = stratified_sample_examples(train_examples, limit=limit, seed=seed)
    authentic_query_keys = {normalize_text_key(example.query) for example in dataset.examples}
    seen_query_keys = set(authentic_query_keys)
    created_at = utc_now_iso()
    generated_examples: list[RecipeExample] = []
    for parent_example in selected_examples:
        response_payload = client.create_structured_output(
            model=model,
            instructions=_query_generation_instructions(max_candidates=max_candidates_per_parent),
            input_text=json.dumps(
                {
                    "parent_example": _serialize_recipe_example(parent_example),
                    "target_query_type_signature": parent_example.query_type_signature,
                    "max_candidates": max_candidates_per_parent,
                },
                ensure_ascii=True,
                indent=2,
            ),
            schema_name="synthetic_query_generation",
            schema=_query_generation_schema(max_candidates=max_candidates_per_parent),
        )
        accepted_for_parent = 0
        for candidate in response_payload.get("candidates", []):
            candidate_query = str(candidate["query"]).strip()
            normalized_query = normalize_text_key(candidate_query)
            if not normalized_query or normalized_query in seen_query_keys:
                continue
            accepted_for_parent += 1
            synthetic_example = build_synthetic_query_example(
                parent_example,
                query=candidate_query,
                candidate_index=accepted_for_parent,
                generator_model=model,
                generation_prompt_version=QUERY_GENERATION_PROMPT_VERSION,
                created_at=created_at,
                intended_query_type_target=str(
                    candidate.get("intended_query_type_target", parent_example.query_type_signature)
                ).strip()
                or parent_example.query_type_signature,
                generation_method=str(candidate.get("method_tag", "query_generation")).strip()
                or "query_generation",
                extra_metadata={
                    "generation_notes": str(candidate.get("rationale", "")).strip(),
                },
            )
            seen_query_keys.add(normalized_query)
            generated_examples.append(synthetic_example)
    output_dataset = PreparedDataset(
        examples=tuple(generated_examples),
        metadata={
            "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
            "generator_model": model,
            "generation_prompt_version": QUERY_GENERATION_PROMPT_VERSION,
            "selected_parent_count": len(selected_examples),
            "candidate_count": len(generated_examples),
        },
    )
    write_synthetic_query_dataset(output_dataset, output_path)
    return {
        "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
        "generator_model": model,
        "selected_parent_count": len(selected_examples),
        "candidate_count": len(generated_examples),
        "output_path": str(output_path),
    }


def review_synthetic_query_candidates(
    *,
    input_path: Path | str,
    dataset_path: Path | str,
    output_path: Path | str,
    client: OpenAIResponsesClient,
    model: str = DEFAULT_QUERY_REVIEW_MODEL,
) -> dict[str, Any]:
    """Attach structured reviewer judgments to query-only candidates."""
    dataset = load_dataset(dataset_path)
    examples_by_id = {example.example_id: example for example in dataset.examples}
    candidate_dataset = read_synthetic_query_dataset(input_path)
    reviewed_examples: list[RecipeExample] = []
    for candidate in candidate_dataset.examples:
        parent_example_id = str(candidate.source_metadata["parent_example_id"])
        if parent_example_id not in examples_by_id:
            raise DatasetValidationError(
                f"{candidate.example_id} references missing parent example {parent_example_id!r}"
            )
        parent_example = examples_by_id[parent_example_id]
        review_payload = client.create_structured_output(
            model=model,
            instructions=_query_review_instructions(),
            input_text=json.dumps(
                {
                    "parent_example": _serialize_recipe_example(parent_example),
                    "candidate_example": _serialize_recipe_example(candidate),
                    "gold_answer_text": _gold_answer_text(parent_example),
                },
                ensure_ascii=True,
                indent=2,
            ),
            schema_name="synthetic_query_review",
            schema=_query_review_schema(),
        )
        reviewed_examples.append(
            _replace_recipe_example_metadata(
                candidate,
                {
                    **dict(candidate.source_metadata),
                    "review_status": review_payload["review_status"],
                    "review_scores": dict(review_payload["review_scores"]),
                    "review_summary": str(review_payload["review_summary"]).strip(),
                    "failure_modes": list(review_payload.get("failure_modes", [])),
                    "reviewed_at": utc_now_iso(),
                    "reviewer_model": model,
                    "generation_prompt_version": QUERY_GENERATION_PROMPT_VERSION,
                    "review_prompt_version": QUERY_REVIEW_PROMPT_VERSION,
                },
            )
        )
    reviewed_dataset = PreparedDataset(
        examples=tuple(reviewed_examples),
        metadata={
            "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
            "review_model": model,
            "review_prompt_version": QUERY_REVIEW_PROMPT_VERSION,
            "review_status_counts": _count_review_statuses(reviewed_examples),
        },
    )
    write_synthetic_query_dataset(reviewed_dataset, output_path)
    return {
        "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
        "review_model": model,
        "candidate_count": len(reviewed_examples),
        "review_status_counts": _count_review_statuses(reviewed_examples),
        "output_path": str(output_path),
    }


def approve_synthetic_query_candidates(
    *,
    input_path: Path | str,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    approval_batch_id: str,
    max_examples: int | None = None,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
    min_answer_preservation: float = 0.95,
    min_semantic_preservation: float = 0.90,
    min_constraint_preservation: float = 0.90,
    min_language_quality: float = 0.90,
    max_leakage_risk: float = 0.20,
    near_duplicate_threshold: float = 0.995,
) -> dict[str, Any]:
    """Apply deterministic repo-side gates to reviewed query-only artifacts."""
    if near_duplicate_threshold < 0.0 or near_duplicate_threshold > 1.0:
        raise DatasetValidationError("near_duplicate_threshold must be between 0 and 1")
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_example_ids = set(split_manifest.splits["train"])
    authentic_query_keys = {normalize_text_key(example.query) for example in dataset.examples}
    examples_by_id = {example.example_id: example for example in dataset.examples}
    candidate_dataset = read_synthetic_query_dataset(input_path)
    approved_examples: list[RecipeExample] = []
    rejection_counts: Counter[str] = Counter()
    for candidate in candidate_dataset.examples:
        metadata = dict(candidate.source_metadata)
        parent_example_id = str(metadata["parent_example_id"])
        parent_example = examples_by_id.get(parent_example_id)
        if parent_example is None:
            rejection_counts["missing_parent"] += 1
            continue
        if parent_example_id not in train_example_ids:
            rejection_counts["parent_not_in_train"] += 1
            continue
        if metadata["review_status"] != "approved":
            rejection_counts["review_not_approved"] += 1
            continue
        review_scores = dict(metadata["review_scores"])
        if review_scores.get("answer_preservation", 0.0) < min_answer_preservation:
            rejection_counts["low_answer_preservation"] += 1
            continue
        if review_scores.get("semantic_preservation", 0.0) < min_semantic_preservation:
            rejection_counts["low_semantic_preservation"] += 1
            continue
        if review_scores.get("constraint_preservation", 0.0) < min_constraint_preservation:
            rejection_counts["low_constraint_preservation"] += 1
            continue
        if review_scores.get("language_quality", 0.0) < min_language_quality:
            rejection_counts["low_language_quality"] += 1
            continue
        if review_scores.get("leakage_risk", 1.0) > max_leakage_risk:
            rejection_counts["high_leakage_risk"] += 1
            continue
        normalized_query = normalize_text_key(candidate.query)
        if normalized_query in authentic_query_keys:
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
        approved_examples.append(
            _replace_recipe_example_metadata(
                candidate,
                {
                    **metadata,
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
                key_fn=lambda example: example.example_id,
            )
        )
    output_dataset = PreparedDataset(
        examples=tuple(approved_examples),
        metadata={
            "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
            "approval_batch_id": approval_batch_id,
            "approved_count": len(approved_examples),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
    )
    validate_synthetic_query_dataset(output_dataset, expected_review_status="approved")
    write_synthetic_query_dataset(output_dataset, output_path)
    return {
        "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
        "approval_batch_id": approval_batch_id,
        "approved_count": len(approved_examples),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "output_path": str(output_path),
    }


def generate_synthetic_full_candidates(
    *,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    client: OpenAIResponsesClient,
    model: str = DEFAULT_FULL_GENERATION_MODEL,
    limit: int = 40,
    max_candidates_per_seed: int = 1,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
) -> dict[str, Any]:
    """Generate full synthetic MCQ candidates from train seed examples."""
    if max_candidates_per_seed < 1:
        raise DatasetValidationError("max_candidates_per_seed must be >= 1")
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_examples = get_split_examples(dataset, split_manifest, "train")
    selected_examples = stratified_sample_examples(train_examples, limit=limit, seed=seed)
    authentic_query_keys = {normalize_text_key(example.query) for example in dataset.examples}
    seen_query_keys = set(authentic_query_keys)
    created_at = utc_now_iso()
    generated_records: list[SyntheticFullRecord] = []
    next_example_index = 1
    for seed_example in selected_examples:
        response_payload = client.create_structured_output(
            model=model,
            instructions=_full_generation_instructions(max_candidates=max_candidates_per_seed),
            input_text=json.dumps(
                {
                    "seed_example": _serialize_recipe_example(seed_example),
                    "target_query_type_signature": seed_example.query_type_signature,
                    "max_candidates": max_candidates_per_seed,
                },
                ensure_ascii=True,
                indent=2,
            ),
            schema_name="synthetic_full_generation",
            schema=_full_generation_schema(max_candidates=max_candidates_per_seed),
        )
        for candidate in response_payload.get("candidates", []):
            candidate_query = str(candidate["query"]).strip()
            normalized_query = normalize_text_key(candidate_query)
            option_texts = [str(option_text).strip() for option_text in candidate["options"]]
            normalized_option_texts = {normalize_text_key(option_text) for option_text in option_texts}
            if (
                not normalized_query
                or normalized_query in seen_query_keys
                or len(option_texts) != 5
                or len(normalized_option_texts) != 5
            ):
                continue
            generated_records.append(
                build_synthetic_full_record(
                    example_id=f"synfull-{next_example_index:04d}",
                    query=candidate_query,
                    option_texts=option_texts,
                    answer_index=int(candidate["answer_index"]),
                    query_type_flags=_normalize_query_type_flags(candidate["query_type_flags"]),
                    correctness_explanation=_normalize_correctness_explanation(
                        candidate["correctness_explanation"]
                    ),
                    generator_model=model,
                    generation_prompt_version=FULL_GENERATION_PROMPT_VERSION,
                    created_at=created_at,
                    intended_query_type_target=str(
                        candidate.get("intended_query_type_target", seed_example.query_type_signature)
                    ).strip()
                    or seed_example.query_type_signature,
                    seed_example_ids=[seed_example.example_id],
                    distractor_generation_method=str(
                        candidate.get("distractor_generation_method", "template_conditioned_generation")
                    ).strip()
                    or "template_conditioned_generation",
                    extra_provenance={
                        "generation_notes": str(candidate.get("rationale", "")).strip(),
                    },
                )
            )
            seen_query_keys.add(normalized_query)
            next_example_index += 1
    output_dataset = SyntheticFullDataset(
        records=tuple(generated_records),
        metadata={
            "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
            "generator_model": model,
            "generation_prompt_version": FULL_GENERATION_PROMPT_VERSION,
            "selected_seed_count": len(selected_examples),
            "candidate_count": len(generated_records),
        },
    )
    write_synthetic_full_dataset(output_dataset, output_path)
    return {
        "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
        "generator_model": model,
        "selected_seed_count": len(selected_examples),
        "candidate_count": len(generated_records),
        "output_path": str(output_path),
    }


def review_synthetic_full_candidates(
    *,
    input_path: Path | str,
    dataset_path: Path | str,
    output_path: Path | str,
    client: OpenAIResponsesClient,
    model: str = DEFAULT_FULL_REVIEW_MODEL,
) -> dict[str, Any]:
    """Attach structured reviewer judgments to full-generation candidates."""
    dataset = load_dataset(dataset_path)
    examples_by_id = {example.example_id: example for example in dataset.examples}
    candidate_dataset = read_synthetic_full_dataset(input_path)
    reviewed_records: list[SyntheticFullRecord] = []
    for record in candidate_dataset.records:
        seed_examples = [
            _serialize_recipe_example(examples_by_id[seed_example_id])
            for seed_example_id in record.provenance["seed_example_ids"]
            if seed_example_id in examples_by_id
        ]
        review_payload = client.create_structured_output(
            model=model,
            instructions=_full_review_instructions(),
            input_text=json.dumps(
                {
                    "seed_examples": seed_examples,
                    "candidate_record": record.to_dict(),
                },
                ensure_ascii=True,
                indent=2,
            ),
            schema_name="synthetic_full_review",
            schema=_full_review_schema(),
        )
        updated_provenance = {
            **dict(record.provenance),
            "review_status": review_payload["review_status"],
            "review_scores": dict(review_payload["review_scores"]),
            "distribution_fit_score": float(review_payload["distribution_fit_score"]),
            "review_summary": str(review_payload["review_summary"]).strip(),
            "failure_modes": list(review_payload.get("failure_modes", [])),
            "reviewed_at": utc_now_iso(),
            "reviewer_model": model,
            "review_prompt_version": FULL_REVIEW_PROMPT_VERSION,
        }
        reviewed_records.append(_replace_full_record_provenance(record, updated_provenance))
    reviewed_dataset = SyntheticFullDataset(
        records=tuple(reviewed_records),
        metadata={
            "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
            "review_model": model,
            "review_prompt_version": FULL_REVIEW_PROMPT_VERSION,
            "review_status_counts": _count_review_statuses(
                [record.recipe_example for record in reviewed_records]
            ),
        },
    )
    write_synthetic_full_dataset(reviewed_dataset, output_path)
    return {
        "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
        "review_model": model,
        "candidate_count": len(reviewed_records),
        "review_status_counts": _count_review_statuses(
            [record.recipe_example for record in reviewed_records]
        ),
        "output_path": str(output_path),
    }


def approve_synthetic_full_candidates(
    *,
    input_path: Path | str,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    approval_batch_id: str,
    max_examples: int | None = None,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
    min_single_answer_validity: float = 0.95,
    min_distractor_plausibility: float = 0.90,
    min_distribution_fit: float = 0.70,
    min_language_quality: float = 0.90,
    max_leakage_risk: float = 0.20,
    near_duplicate_threshold: float = 0.995,
) -> dict[str, Any]:
    """Apply deterministic repo-side gates to reviewed full-generation artifacts."""
    dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_example_ids = set(split_manifest.splits["train"])
    authentic_query_keys = {normalize_text_key(example.query) for example in dataset.examples}
    candidate_dataset = read_synthetic_full_dataset(input_path)
    approved_records: list[SyntheticFullRecord] = []
    rejection_counts: Counter[str] = Counter()
    for record in candidate_dataset.records:
        if str(record.provenance["review_status"]) != "approved":
            rejection_counts["review_not_approved"] += 1
            continue
        if not set(record.provenance["seed_example_ids"]).issubset(train_example_ids):
            rejection_counts["seed_not_in_train"] += 1
            continue
        review_scores = dict(record.provenance["review_scores"])
        if review_scores.get("single_answer_validity", 0.0) < min_single_answer_validity:
            rejection_counts["low_single_answer_validity"] += 1
            continue
        if review_scores.get("distractor_plausibility", 0.0) < min_distractor_plausibility:
            rejection_counts["low_distractor_plausibility"] += 1
            continue
        if float(record.provenance["distribution_fit_score"] or 0.0) < min_distribution_fit:
            rejection_counts["low_distribution_fit"] += 1
            continue
        if review_scores.get("language_quality", 0.0) < min_language_quality:
            rejection_counts["low_language_quality"] += 1
            continue
        if review_scores.get("leakage_risk", 1.0) > max_leakage_risk:
            rejection_counts["high_leakage_risk"] += 1
            continue
        normalized_query = normalize_text_key(record.recipe_example.query)
        if normalized_query in authentic_query_keys:
            rejection_counts["exact_duplicate_to_authentic"] += 1
            continue
        if any(
            is_near_duplicate(
                record.recipe_example.query,
                approved_record.recipe_example.query,
                threshold=near_duplicate_threshold,
            )
            for approved_record in approved_records
        ):
            rejection_counts["near_duplicate_to_approved"] += 1
            continue
        normalized_option_texts = {
            normalize_text_key(option.text) for option in record.recipe_example.options
        }
        if len(normalized_option_texts) != len(record.recipe_example.options):
            rejection_counts["duplicate_option_texts"] += 1
            continue
        approved_records.append(
            _replace_full_record_provenance(
                record,
                {
                    **dict(record.provenance),
                    "approval_batch_id": approval_batch_id,
                },
            )
        )
    if max_examples is not None:
        approved_records = list(
            deterministic_sample(
                approved_records,
                limit=max_examples,
                seed=seed,
                key_fn=lambda record: record.recipe_example.example_id,
            )
        )
    output_dataset = SyntheticFullDataset(
        records=tuple(approved_records),
        metadata={
            "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
            "approval_batch_id": approval_batch_id,
            "approved_count": len(approved_records),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
    )
    write_synthetic_full_dataset(output_dataset, output_path)
    return {
        "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
        "approval_batch_id": approval_batch_id,
        "approved_count": len(approved_records),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "output_path": str(output_path),
    }


def build_synthetic_training_artifact(
    *,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    output_path: Path | str,
    query_approved_path: Path | str | None = None,
    full_approved_path: Path | str | None = None,
    max_query_examples: int | None = None,
    max_full_examples: int | None = None,
    target_ratio: float | None = None,
    full_share: float = 0.0,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
) -> dict[str, Any]:
    """Combine approved synthetic inputs into one train-ready RecipeExample artifact."""
    if query_approved_path is None and full_approved_path is None:
        raise DatasetValidationError("At least one approved synthetic artifact path is required")
    if target_ratio is not None and (max_query_examples is not None or max_full_examples is not None):
        raise DatasetValidationError("target_ratio cannot be combined with max_query_examples/max_full_examples")
    if target_ratio is not None and target_ratio < 0:
        raise DatasetValidationError("target_ratio must be >= 0")
    if full_share < 0.0 or full_share > 1.0:
        raise DatasetValidationError("full_share must be between 0 and 1")

    base_dataset = load_dataset(dataset_path)
    split_manifest = load_split_manifest(split_manifest_path)
    train_examples = get_split_examples(base_dataset, split_manifest, "train")
    authentic_example_ids = {example.example_id for example in base_dataset.examples}

    query_examples = ()
    if query_approved_path is not None:
        query_examples = validate_synthetic_query_dataset(
            read_synthetic_query_dataset(query_approved_path),
            expected_review_status="approved",
        ).examples

    full_records = ()
    if full_approved_path is not None:
        full_dataset = read_synthetic_full_dataset(full_approved_path)
        for record in full_dataset.records:
            if record.provenance["review_status"] != "approved":
                raise DatasetValidationError(
                    f"{record.recipe_example.example_id} full synthetic review_status must be 'approved'"
                )
        full_records = full_dataset.records

    if target_ratio is not None:
        total_target = round(len(train_examples) * target_ratio)
        if query_examples and full_records:
            full_target = round(total_target * full_share)
            query_target = total_target - full_target
        elif query_examples:
            query_target = total_target
            full_target = 0
        else:
            query_target = 0
            full_target = total_target
    else:
        query_target = max_query_examples
        full_target = max_full_examples

    selected_query_examples = deterministic_sample(
        list(query_examples),
        limit=query_target,
        seed=seed,
        key_fn=lambda example: example.example_id,
    )
    selected_full_records = deterministic_sample(
        list(full_records),
        limit=full_target,
        seed=seed,
        key_fn=lambda record: record.recipe_example.example_id,
    )
    selected_full_examples = convert_synthetic_full_records_to_examples(selected_full_records)

    synthetic_examples = tuple(selected_query_examples) + tuple(selected_full_examples)
    synthetic_example_ids = [example.example_id for example in synthetic_examples]
    if len(set(synthetic_example_ids)) != len(synthetic_example_ids):
        raise DatasetValidationError("Selected synthetic training examples contain duplicate ids")
    collisions = sorted(authentic_example_ids.intersection(synthetic_example_ids))
    if collisions:
        raise DatasetValidationError(
            f"Synthetic training examples reuse authentic example ids: {collisions[:5]}"
        )
    training_dataset = PreparedDataset(
        examples=synthetic_examples,
        metadata={
            "synthetic_query_count": len(selected_query_examples),
            "synthetic_full_count": len(selected_full_examples),
            "total_synthetic_count": len(synthetic_examples),
            "target_ratio": target_ratio,
            "full_share": full_share,
            "query_source_path": str(query_approved_path) if query_approved_path is not None else None,
            "full_source_path": str(full_approved_path) if full_approved_path is not None else None,
        },
    )
    write_prepared_dataset(training_dataset, output_path)
    return {
        "synthetic_query_count": len(selected_query_examples),
        "synthetic_full_count": len(selected_full_examples),
        "total_synthetic_count": len(synthetic_examples),
        "train_example_count": len(train_examples),
        "target_ratio": target_ratio,
        "full_share": full_share,
        "output_path": str(output_path),
    }


def _serialize_recipe_example(example: RecipeExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "query": example.query,
        "options": [option.text for option in example.options],
        "answer_option_id": example.answer_option_id,
        "answer_text": _gold_answer_text(example),
        "query_type_signature": example.query_type_signature,
        "query_type_flags": dict(example.query_type_flags),
        "correctness_explanation": dict(example.correctness_explanation),
    }


def _gold_answer_text(example: RecipeExample) -> str:
    return next(
        option.text for option in example.options if option.option_id == example.answer_option_id
    )


def _normalize_query_type_flags(payload: Mapping[str, Any]) -> dict[str, bool]:
    normalized_flags: dict[str, bool] = {}
    for query_type_name in QUERY_TYPE_NAMES:
        if query_type_name not in payload:
            raise DatasetValidationError(
                f"synthetic query_type_flags must include {QUERY_TYPE_NAMES}, missing {query_type_name!r}"
            )
        query_type_value = payload[query_type_name]
        if query_type_value not in (0, 1, False, True):
            raise DatasetValidationError(
                f"synthetic query_type flag {query_type_name!r} must be boolean-like"
            )
        normalized_flags[query_type_name] = bool(query_type_value)
    return normalized_flags


def _normalize_correctness_explanation(payload: Mapping[str, Any]) -> dict[str, str]:
    normalized_payload: dict[str, str] = {}
    if isinstance(payload, Mapping):
        items = payload.items()
    elif isinstance(payload, (list, tuple)):
        items = []
        for row in payload:
            if not isinstance(row, Mapping):
                raise DatasetValidationError(
                    "synthetic correctness_explanation list rows must be mappings"
                )
            if "key" not in row or "value" not in row:
                raise DatasetValidationError(
                    "synthetic correctness_explanation rows require key and value"
                )
            items.append((row["key"], row["value"]))
    else:
        raise DatasetValidationError(
            "synthetic correctness_explanation must be a mapping or key/value list"
        )
    for key, value in items:
        if not isinstance(key, str) or not key.strip():
            raise DatasetValidationError("synthetic correctness_explanation keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(
                "synthetic correctness_explanation values must be non-empty strings"
            )
        normalized_payload[key.strip()] = value.strip()
    if not normalized_payload:
        raise DatasetValidationError("synthetic correctness_explanation must be non-empty")
    return normalized_payload


def _replace_recipe_example_metadata(
    example: RecipeExample, updated_metadata: Mapping[str, Any]
) -> RecipeExample:
    return RecipeExample(
        example_id=example.example_id,
        query=example.query,
        options=example.options,
        answer_option_id=example.answer_option_id,
        query_type_flags=example.query_type_flags,
        correctness_explanation=example.correctness_explanation,
        source_metadata=updated_metadata,
    )


def _replace_full_record_provenance(
    record: SyntheticFullRecord, updated_provenance: Mapping[str, Any]
) -> SyntheticFullRecord:
    updated_recipe_example = _replace_recipe_example_metadata(
        record.recipe_example,
        {
            **dict(record.recipe_example.source_metadata),
            "review_status": updated_provenance["review_status"],
            "review_scores": dict(updated_provenance["review_scores"]),
            "approval_batch_id": updated_provenance["approval_batch_id"],
            "reviewed_at": updated_provenance.get("reviewed_at"),
            "reviewer_model": updated_provenance.get("reviewer_model"),
            "review_summary": updated_provenance.get("review_summary"),
            "review_prompt_version": updated_provenance.get("review_prompt_version"),
        },
    )
    return SyntheticFullRecord(
        recipe_example=updated_recipe_example,
        provenance=updated_provenance,
    )


def _count_review_statuses(examples: Sequence[RecipeExample]) -> dict[str, int]:
    counts = Counter(str(example.source_metadata.get("review_status", "unknown")) for example in examples)
    return dict(sorted(counts.items()))


def _query_generation_instructions(*, max_candidates: int) -> str:
    return (
        "You are generating train-only synthetic Recipe-MPR queries. "
        "Preserve the original correct answer against the existing five options, preserve the same "
        "query-type intent, avoid copying option text verbatim into the query, and return at most "
        f"{max_candidates} diverse candidates. Prefer constrained paraphrase or answer-aware rewrite."
    )


def _query_generation_schema(*, max_candidates: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "intended_query_type_target": {"type": "string", "minLength": 1},
                        "method_tag": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "query",
                        "intended_query_type_target",
                        "method_tag",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def _query_review_instructions() -> str:
    return (
        "You are reviewing a synthetic query against a real Recipe-MPR parent example. "
        "Approve only if the candidate preserves semantics and constraints, keeps the same correct "
        "answer against the authentic options, does not leak the answer trivially, and is fluent."
    )


def _query_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "review_status": {"type": "string", "enum": list(REVIEW_STATUSES[1:])},
            "review_scores": {
                "type": "object",
                "properties": {
                    "semantic_preservation": {"type": "number", "minimum": 0, "maximum": 1},
                    "constraint_preservation": {"type": "number", "minimum": 0, "maximum": 1},
                    "answer_preservation": {"type": "number", "minimum": 0, "maximum": 1},
                    "leakage_risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "language_quality": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "semantic_preservation",
                    "constraint_preservation",
                    "answer_preservation",
                    "leakage_risk",
                    "language_quality",
                ],
                "additionalProperties": False,
            },
            "failure_modes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "review_summary": {"type": "string", "minLength": 1},
        },
        "required": ["review_status", "review_scores", "failure_modes", "review_summary"],
        "additionalProperties": False,
    }


def _full_generation_instructions(*, max_candidates: int) -> str:
    return (
        "You are generating brand-new Recipe-MPR-style multiple-choice examples. "
        "Use the seed example only as a structural template. Create a new query, five options, exactly one "
        "defensible correct answer, and four plausible but incorrect distractors. "
        "Keep the query-type flags internally consistent with the wording of the query. "
        "Do not introduce temporal, negation, or analogy cues unless they are intentionally reflected in the flags. "
        "Make distractors close competitors from similar cuisine, meal role, or ingredient space rather than obviously unrelated foods. "
        "Prefer recipe-style option texts, not generic food names. Keep the example aligned "
        "with Recipe-MPR style and return at most "
        f"{max_candidates} candidates."
    )


def _full_generation_schema(*, max_candidates: int) -> dict[str, Any]:
    query_type_flag_schema = {
        "type": "object",
        "properties": {
            query_type_name: {"type": "boolean"} for query_type_name in QUERY_TYPE_NAMES
        },
        "required": list(QUERY_TYPE_NAMES),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "options": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "answer_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 4,
                        },
                        "query_type_flags": query_type_flag_schema,
                        "correctness_explanation": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "minLength": 1},
                                    "value": {"type": "string", "minLength": 1},
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False,
                            },
                        },
                        "intended_query_type_target": {"type": "string", "minLength": 1},
                        "distractor_generation_method": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "query",
                        "options",
                        "answer_index",
                        "query_type_flags",
                        "correctness_explanation",
                        "intended_query_type_target",
                        "distractor_generation_method",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def _full_review_instructions() -> str:
    return (
        "You are reviewing a synthetic full-generation Recipe-MPR-style record. "
        "Approve only if there is exactly one clearly correct option, the distractors are plausible but wrong, "
        "the item is fluent, the query type labels are sensible, there is no trivial answer leakage, and the "
        "record fits the style and difficulty of Recipe-MPR. "
        "Use this scoring direction consistently: 1.0 means very strong for validity, plausibility, distribution fit, and language quality; "
        "0.0 means very weak. For leakage_risk, 1.0 means severe leakage risk and 0.0 means little or no leakage risk. "
        "The review_status should match the scores and summary."
    )


def _full_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "review_status": {"type": "string", "enum": list(REVIEW_STATUSES[1:])},
            "review_scores": {
                "type": "object",
                "properties": {
                    "single_answer_validity": {"type": "number", "minimum": 0, "maximum": 1},
                    "distractor_plausibility": {"type": "number", "minimum": 0, "maximum": 1},
                    "leakage_risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "distribution_fit": {"type": "number", "minimum": 0, "maximum": 1},
                    "language_quality": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "single_answer_validity",
                    "distractor_plausibility",
                    "leakage_risk",
                    "distribution_fit",
                    "language_quality",
                ],
                "additionalProperties": False,
            },
            "distribution_fit_score": {"type": "number", "minimum": 0, "maximum": 1},
            "failure_modes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "review_summary": {"type": "string", "minLength": 1},
        },
        "required": [
            "review_status",
            "review_scores",
            "distribution_fit_score",
            "failure_modes",
            "review_summary",
        ],
        "additionalProperties": False,
    }
