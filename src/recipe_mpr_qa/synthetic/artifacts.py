from __future__ import annotations

import re
import unicodedata
from hashlib import sha256
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from recipe_mpr_qa.data.models import DatasetValidationError, PreparedDataset, RecipeExample
from recipe_mpr_qa.data.preparation import read_prepared_dataset, write_prepared_dataset

QUERY_ONLY_SYNTHETIC_MODE = "query_only"
CANDIDATE_SYNTHETIC_STAGE = "candidate"
REVIEWED_SYNTHETIC_STAGE = "reviewed"
APPROVED_SYNTHETIC_STAGE = "approved"
TRAIN_READY_SYNTHETIC_STAGE = "train_ready"
DEFAULT_SYNTHETIC_SELECTION_SEED = 1508


def normalize_text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return normalized.strip()


def is_near_duplicate(left: str, right: str, *, threshold: float = 0.95) -> bool:
    if normalize_text_key(left) == normalize_text_key(right):
        return True
    return SequenceMatcher(a=normalize_text_key(left), b=normalize_text_key(right)).ratio() >= threshold


def deterministic_sample(
    items: Sequence[Any],
    *,
    limit: int,
    seed: int,
    key_fn: Callable[[Any], str],
) -> tuple[Any, ...]:
    if limit <= 0:
        return ()
    decorated = sorted(
        items,
        key=lambda item: (
            sha256(f"{seed}:{key_fn(item)}".encode("utf-8")).hexdigest(),
            key_fn(item),
        ),
    )
    return tuple(decorated[:limit])


def stratified_sample_examples(
    examples: Sequence[RecipeExample],
    *,
    limit: int,
    seed: int,
) -> tuple[RecipeExample, ...]:
    if limit <= 0:
        return ()
    grouped: dict[str, list[RecipeExample]] = {}
    for example in examples:
        grouped.setdefault(example.query_type_signature, []).append(example)
    selected: list[RecipeExample] = []
    group_names = sorted(grouped)
    while len(selected) < min(limit, len(examples)):
        progressed = False
        for group_name in group_names:
            if len(selected) >= limit:
                break
            group_items = deterministic_sample(
                grouped[group_name],
                limit=len(grouped[group_name]),
                seed=seed,
                key_fn=lambda item: item.example_id,
            )
            current = [example.example_id for example in selected]
            next_item = next((item for item in group_items if item.example_id not in current), None)
            if next_item is None:
                continue
            selected.append(next_item)
            progressed = True
        if not progressed:
            break
    return tuple(selected)


def build_synthetic_query_example(
    parent_example: RecipeExample,
    *,
    query: str,
    candidate_index: int,
    generator_model: str,
    generation_prompt_version: str,
    created_at: str,
    stage: str = CANDIDATE_SYNTHETIC_STAGE,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RecipeExample:
    if candidate_index <= 0:
        raise DatasetValidationError("candidate_index must be > 0")
    metadata = {
        **dict(parent_example.source_metadata),
        "synthetic": True,
        "synthetic_mode": QUERY_ONLY_SYNTHETIC_MODE,
        "synthetic_stage": stage,
        "parent_example_id": parent_example.example_id,
        "synthetic_candidate_index": candidate_index,
        "generator_model_name": generator_model,
        "generation_prompt_version": generation_prompt_version,
        "created_at": created_at,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return RecipeExample(
        example_id=f"{parent_example.example_id}-synq-{candidate_index:03d}",
        query=query.strip(),
        options=parent_example.options,
        answer_option_id=parent_example.answer_option_id,
        query_type_flags=parent_example.query_type_flags,
        correctness_explanation=parent_example.correctness_explanation,
        source_metadata=metadata,
    )


def write_synthetic_query_dataset(
    dataset: PreparedDataset,
    output_path: Path | str,
) -> None:
    write_prepared_dataset(dataset, output_path)


def read_synthetic_query_dataset(dataset_path: Path | str) -> PreparedDataset:
    return read_prepared_dataset(dataset_path)


def validate_synthetic_query_dataset(
    dataset: PreparedDataset,
    *,
    expected_stage: str | None = None,
) -> None:
    seen_ids: set[str] = set()
    for example in dataset.examples:
        if example.example_id in seen_ids:
            raise DatasetValidationError(f"Duplicate synthetic example id: {example.example_id}")
        seen_ids.add(example.example_id)
        metadata = dict(example.source_metadata)
        if not metadata.get("synthetic"):
            raise DatasetValidationError(f"{example.example_id} is missing synthetic=true")
        if metadata.get("synthetic_mode") != QUERY_ONLY_SYNTHETIC_MODE:
            raise DatasetValidationError(f"{example.example_id} has unsupported synthetic_mode")
        stage = metadata.get("synthetic_stage")
        if expected_stage is not None and stage != expected_stage:
            raise DatasetValidationError(
                f"{example.example_id} expected stage {expected_stage!r}, got {stage!r}"
            )
        if not isinstance(metadata.get("parent_example_id"), str):
            raise DatasetValidationError(f"{example.example_id} is missing parent_example_id")
        if not isinstance(metadata.get("synthetic_candidate_index"), int):
            raise DatasetValidationError(f"{example.example_id} is missing synthetic_candidate_index")


def with_stage(example: RecipeExample, *, stage: str, extra_metadata: Mapping[str, Any] | None = None) -> RecipeExample:
    metadata = dict(example.source_metadata)
    metadata["synthetic_stage"] = stage
    if extra_metadata:
        metadata.update(extra_metadata)
    return replace(example, source_metadata=metadata)
