"""Artifact types and validation helpers for synthetic Recipe-MPR data.

The synthetic workflow uses two deliberately different shapes:

- query-only artifacts reuse ``RecipeExample`` because they preserve authentic
  options and the authentic gold label
- full-generation artifacts wrap a generated ``RecipeExample`` in a provenance
  envelope so review and admission metadata stay explicit

This module owns the invariants that keep those artifacts safe to hand off to
later review, approval, and train-admission steps.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
from recipe_mpr_qa.data.models import DatasetValidationError, PreparedDataset, RecipeExample, RecipeOption
from recipe_mpr_qa.data.preparation import read_prepared_dataset, write_prepared_dataset

QUERY_ONLY_SYNTHETIC_MODE = "query_only"
FULL_GENERATION_SYNTHETIC_MODE = "full_generation"
REVIEW_STATUSES = ("pending", "approved", "rejected")
DEFAULT_SYNTHETIC_SELECTION_SEED = 1508

SHARED_PROVENANCE_FIELDS = (
    "synthetic_mode",
    "generator_model",
    "generation_prompt_version",
    "approval_batch_id",
    "review_status",
    "review_scores",
    "created_at",
    "intended_query_type_target",
)
QUERY_ONLY_PROVENANCE_FIELDS = SHARED_PROVENANCE_FIELDS + ("parent_example_id",)
FULL_GENERATION_PROVENANCE_FIELDS = SHARED_PROVENANCE_FIELDS + (
    "seed_example_ids",
    "distractor_generation_method",
    "distribution_fit_score",
)

_TOKEN_PATTERN = re.compile(r"\w+")
_TRAILING_PUNCTUATION_PATTERN = re.compile(r"[?.!,;:]+$")
_T = TypeVar("_T")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip().casefold()
    return _TRAILING_PUNCTUATION_PATTERN.sub("", collapsed)


def near_duplicate_score(left: str, right: str) -> float:
    normalized_left = normalize_text_key(left)
    normalized_right = normalize_text_key(right)
    if not normalized_left or not normalized_right:
        return 0.0
    sequence_ratio = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    left_tokens = set(_TOKEN_PATTERN.findall(normalized_left))
    right_tokens = set(_TOKEN_PATTERN.findall(normalized_right))
    if not left_tokens and not right_tokens:
        token_overlap = 1.0
    elif not left_tokens or not right_tokens:
        token_overlap = 0.0
    else:
        token_overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(sequence_ratio, token_overlap)


def is_near_duplicate(left: str, right: str, *, threshold: float = 0.97) -> bool:
    return near_duplicate_score(left, right) >= threshold


def deterministic_sample(
    items: Sequence[_T],
    *,
    limit: int | None,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
    key_fn: Callable[[_T], str],
) -> tuple[_T, ...]:
    if limit is None or limit >= len(items):
        return tuple(items)
    if limit < 0:
        raise DatasetValidationError("limit must be >= 0")
    ranked_items = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{seed}:{key_fn(item)}".encode("utf-8")).hexdigest(),
    )
    return tuple(ranked_items[:limit])


def stratified_sample_examples(
    examples: Sequence[RecipeExample],
    *,
    limit: int,
    seed: int = DEFAULT_SYNTHETIC_SELECTION_SEED,
) -> tuple[RecipeExample, ...]:
    if limit < 0:
        raise DatasetValidationError("limit must be >= 0")
    if limit >= len(examples):
        return tuple(examples)
    grouped_examples: dict[str, list[RecipeExample]] = {}
    for example in examples:
        grouped_examples.setdefault(example.query_type_signature, []).append(example)
    group_sizes = {signature: len(group) for signature, group in grouped_examples.items()}
    allocated = _allocate_group_counts(group_sizes, limit)
    selected_ids: set[str] = set()
    for signature, group in grouped_examples.items():
        derived_seed = int(
            hashlib.sha256(f"{seed}:{signature}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        shuffled_group = list(group)
        random.Random(derived_seed).shuffle(shuffled_group)
        selected_ids.update(example.example_id for example in shuffled_group[: allocated[signature]])
    return tuple(example for example in examples if example.example_id in selected_ids)


def _allocate_group_counts(group_sizes: Mapping[str, int], target_total: int) -> dict[str, int]:
    total_size = sum(group_sizes.values())
    if target_total < 0 or target_total > total_size:
        raise DatasetValidationError("invalid target_total for stratified sampling")
    if target_total == 0:
        return {group_name: 0 for group_name in group_sizes}
    exact_counts = {
        group_name: (group_size * target_total) / total_size
        for group_name, group_size in group_sizes.items()
    }
    allocated = {
        group_name: min(group_sizes[group_name], math.floor(exact_counts[group_name]))
        for group_name in group_sizes
    }
    assigned = sum(allocated.values())
    ranked_groups = sorted(
        group_sizes,
        key=lambda group_name: (-(exact_counts[group_name] - allocated[group_name]), group_name),
    )
    index = 0
    while assigned < target_total:
        group_name = ranked_groups[index % len(ranked_groups)]
        if allocated[group_name] < group_sizes[group_name]:
            allocated[group_name] += 1
            assigned += 1
        index += 1
    return allocated


def build_synthetic_query_example(
    parent_example: RecipeExample,
    *,
    query: str,
    candidate_index: int,
    generator_model: str,
    generation_prompt_version: str,
    created_at: str,
    intended_query_type_target: str,
    generation_method: str,
    review_status: str = "pending",
    review_scores: Mapping[str, Any] | None = None,
    approval_batch_id: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RecipeExample:
    """Build one query-only synthetic example anchored to a real train parent."""
    if candidate_index < 1:
        raise DatasetValidationError("candidate_index must be >= 1")
    synthetic_example = RecipeExample(
        example_id=f"{parent_example.example_id}-sq-{candidate_index:02d}",
        query=query,
        options=parent_example.options,
        answer_option_id=parent_example.answer_option_id,
        query_type_flags=dict(parent_example.query_type_flags),
        correctness_explanation=dict(parent_example.correctness_explanation),
        source_metadata={
            **dict(parent_example.source_metadata),
            "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
            "generator_model": generator_model,
            "generation_prompt_version": generation_prompt_version,
            "approval_batch_id": approval_batch_id,
            "review_status": review_status,
            "review_scores": dict(review_scores or {}),
            "created_at": created_at,
            "intended_query_type_target": intended_query_type_target,
            "parent_example_id": parent_example.example_id,
            "generation_method": generation_method,
            **dict(extra_metadata or {}),
        },
    )
    validate_synthetic_query_example(synthetic_example)
    return synthetic_example


def validate_synthetic_query_example(example: RecipeExample) -> None:
    provenance = dict(example.source_metadata)
    missing_fields = [field_name for field_name in QUERY_ONLY_PROVENANCE_FIELDS if field_name not in provenance]
    if missing_fields:
        raise DatasetValidationError(
            f"{example.example_id} synthetic query metadata missing fields: {missing_fields}"
        )
    if provenance["synthetic_mode"] != QUERY_ONLY_SYNTHETIC_MODE:
        raise DatasetValidationError(
            f"{example.example_id} synthetic query mode must be {QUERY_ONLY_SYNTHETIC_MODE!r}"
        )
    _validate_shared_provenance(
        provenance,
        expected_mode=QUERY_ONLY_SYNTHETIC_MODE,
        example_id=example.example_id,
    )
    parent_example_id = provenance["parent_example_id"]
    if not isinstance(parent_example_id, str) or not parent_example_id.strip():
        raise DatasetValidationError(f"{example.example_id} parent_example_id must be a non-empty string")


def validate_synthetic_query_dataset(
    dataset: PreparedDataset,
    *,
    expected_review_status: str | None = None,
) -> PreparedDataset:
    if expected_review_status is not None and expected_review_status not in REVIEW_STATUSES:
        raise DatasetValidationError(f"Unsupported expected_review_status: {expected_review_status}")
    for example in dataset.examples:
        validate_synthetic_query_example(example)
        if expected_review_status is not None and example.source_metadata["review_status"] != expected_review_status:
            raise DatasetValidationError(
                f"{example.example_id} review_status must be {expected_review_status!r}"
            )
    return dataset


def read_synthetic_query_dataset(dataset_path: Path | str) -> PreparedDataset:
    return validate_synthetic_query_dataset(read_prepared_dataset(dataset_path))


def write_synthetic_query_dataset(dataset: PreparedDataset, output_path: Path | str) -> None:
    validate_synthetic_query_dataset(dataset)
    write_prepared_dataset(dataset, output_path)


@dataclass(frozen=True)
class SyntheticFullRecord:
    """One full-generation synthetic example plus its provenance envelope."""

    recipe_example: RecipeExample
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, Mapping):
            raise DatasetValidationError("SyntheticFullRecord provenance must be a mapping")
        normalized_provenance = dict(self.provenance)
        missing_fields = [
            field_name
            for field_name in FULL_GENERATION_PROVENANCE_FIELDS
            if field_name not in normalized_provenance
        ]
        if missing_fields:
            raise DatasetValidationError(
                f"{self.recipe_example.example_id} synthetic full provenance missing fields: {missing_fields}"
            )
        _validate_shared_provenance(
            normalized_provenance,
            expected_mode=FULL_GENERATION_SYNTHETIC_MODE,
            example_id=self.recipe_example.example_id,
        )
        seed_example_ids = normalized_provenance["seed_example_ids"]
        if (
            not isinstance(seed_example_ids, (list, tuple))
            or not seed_example_ids
            or not all(isinstance(item, str) and item.strip() for item in seed_example_ids)
        ):
            raise DatasetValidationError(
                f"{self.recipe_example.example_id} seed_example_ids must be a non-empty list of strings"
            )
        distractor_generation_method = normalized_provenance["distractor_generation_method"]
        if not isinstance(distractor_generation_method, str) or not distractor_generation_method.strip():
            raise DatasetValidationError(
                f"{self.recipe_example.example_id} distractor_generation_method must be a non-empty string"
            )
        distribution_fit_score = normalized_provenance["distribution_fit_score"]
        if distribution_fit_score is not None:
            if not isinstance(distribution_fit_score, (int, float)) or not 0.0 <= float(distribution_fit_score) <= 1.0:
                raise DatasetValidationError(
                    f"{self.recipe_example.example_id} distribution_fit_score must be between 0 and 1"
                )
            normalized_provenance["distribution_fit_score"] = float(distribution_fit_score)
        normalized_provenance["seed_example_ids"] = tuple(seed_example_ids)
        object.__setattr__(self, "provenance", normalized_provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_example": self.recipe_example.to_dict(),
            "provenance": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.provenance.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticFullRecord":
        if "recipe_example" not in payload or "provenance" not in payload:
            raise DatasetValidationError("synthetic full records require recipe_example and provenance")
        return cls(
            recipe_example=RecipeExample.from_dict(payload["recipe_example"]),
            provenance=payload["provenance"],
        )


@dataclass(frozen=True)
class SyntheticFullDataset:
    """Validated collection of full-generation synthetic records."""

    records: tuple[SyntheticFullRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        example_ids = [record.recipe_example.example_id for record in self.records]
        if len(set(example_ids)) != len(example_ids):
            raise DatasetValidationError("SyntheticFullDataset contains duplicate example ids")
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_synthetic_full_record(
    *,
    example_id: str,
    query: str,
    option_texts: Sequence[str],
    answer_index: int,
    query_type_flags: Mapping[str, bool],
    correctness_explanation: Mapping[str, Any],
    generator_model: str,
    generation_prompt_version: str,
    created_at: str,
    intended_query_type_target: str,
    seed_example_ids: Sequence[str],
    distractor_generation_method: str,
    review_status: str = "pending",
    review_scores: Mapping[str, Any] | None = None,
    approval_batch_id: str | None = None,
    distribution_fit_score: float | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> SyntheticFullRecord:
    """Build one full-generation synthetic record with required provenance."""
    if len(option_texts) != 5:
        raise DatasetValidationError("Synthetic full records must contain exactly 5 option texts")
    if answer_index < 0 or answer_index >= len(option_texts):
        raise DatasetValidationError("answer_index must be in the range [0, 4]")
    recipe_example = RecipeExample(
        example_id=example_id,
        query=query,
        options=tuple(
            RecipeOption(option_id=f"{example_id}-opt-{index + 1}", text=option_text)
            for index, option_text in enumerate(option_texts)
        ),
        answer_option_id=f"{example_id}-opt-{answer_index + 1}",
        query_type_flags=query_type_flags,
        correctness_explanation=correctness_explanation,
        source_metadata={
            "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
            "generator_model": generator_model,
            "generation_prompt_version": generation_prompt_version,
            "approval_batch_id": approval_batch_id,
            "review_status": review_status,
            "review_scores": dict(review_scores or {}),
            "created_at": created_at,
            "intended_query_type_target": intended_query_type_target,
        },
    )
    return SyntheticFullRecord(
        recipe_example=recipe_example,
        provenance={
            "synthetic_mode": FULL_GENERATION_SYNTHETIC_MODE,
            "generator_model": generator_model,
            "generation_prompt_version": generation_prompt_version,
            "approval_batch_id": approval_batch_id,
            "review_status": review_status,
            "review_scores": dict(review_scores or {}),
            "created_at": created_at,
            "intended_query_type_target": intended_query_type_target,
            "seed_example_ids": list(seed_example_ids),
            "distractor_generation_method": distractor_generation_method,
            "distribution_fit_score": distribution_fit_score,
            **dict(extra_provenance or {}),
        },
    )


def read_synthetic_full_dataset(dataset_path: Path | str) -> SyntheticFullDataset:
    resolved_path = Path(dataset_path)
    lines = [
        line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    records = tuple(SyntheticFullRecord.from_dict(json.loads(line)) for line in lines)
    return SyntheticFullDataset(
        records=records,
        metadata={
            "record_count": len(records),
            "synthetic_full_dataset_path": resolved_path.as_posix(),
        },
    )


def write_synthetic_full_dataset(dataset: SyntheticFullDataset, output_path: Path | str) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=True, separators=(",", ":"))
        for record in dataset.records
    ]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_synthetic_full_records_to_examples(
    records: Sequence[SyntheticFullRecord],
) -> tuple[RecipeExample, ...]:
    """Flatten approved full-generation records into train-ready RecipeExamples."""
    return tuple(
        RecipeExample(
            example_id=record.recipe_example.example_id,
            query=record.recipe_example.query,
            options=record.recipe_example.options,
            answer_option_id=record.recipe_example.answer_option_id,
            query_type_flags=record.recipe_example.query_type_flags,
            correctness_explanation=record.recipe_example.correctness_explanation,
            source_metadata={
                **dict(record.recipe_example.source_metadata),
                **dict(record.provenance),
            },
        )
        for record in records
    )


def _validate_shared_provenance(
    provenance: Mapping[str, Any], *, expected_mode: str, example_id: str
) -> None:
    if provenance["synthetic_mode"] != expected_mode:
        raise DatasetValidationError(
            f"{example_id} synthetic_mode must be {expected_mode!r}, got {provenance['synthetic_mode']!r}"
        )
    generator_model = provenance["generator_model"]
    if not isinstance(generator_model, str) or not generator_model.strip():
        raise DatasetValidationError(f"{example_id} generator_model must be a non-empty string")
    prompt_version = provenance["generation_prompt_version"]
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise DatasetValidationError(
            f"{example_id} generation_prompt_version must be a non-empty string"
        )
    approval_batch_id = provenance["approval_batch_id"]
    if approval_batch_id is not None and (
        not isinstance(approval_batch_id, str) or not approval_batch_id.strip()
    ):
        raise DatasetValidationError(f"{example_id} approval_batch_id must be a string or null")
    review_status = provenance["review_status"]
    if review_status not in REVIEW_STATUSES:
        raise DatasetValidationError(f"{example_id} review_status must be one of {REVIEW_STATUSES}")
    review_scores = provenance["review_scores"]
    if not isinstance(review_scores, Mapping):
        raise DatasetValidationError(f"{example_id} review_scores must be a mapping")
    for score_name, score_value in review_scores.items():
        if not isinstance(score_name, str) or not score_name.strip():
            raise DatasetValidationError(f"{example_id} review score names must be non-empty strings")
        if not isinstance(score_value, (int, float)) or not 0.0 <= float(score_value) <= 1.0:
            raise DatasetValidationError(
                f"{example_id} review score {score_name!r} must be between 0 and 1"
            )
    created_at = provenance["created_at"]
    if not isinstance(created_at, str) or not created_at.strip():
        raise DatasetValidationError(f"{example_id} created_at must be a non-empty string")
    intended_query_type_target = provenance["intended_query_type_target"]
    if (
        not isinstance(intended_query_type_target, str)
        or not intended_query_type_target.strip()
    ):
        raise DatasetValidationError(
            f"{example_id} intended_query_type_target must be a non-empty string"
        )
