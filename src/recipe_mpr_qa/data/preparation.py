from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.constants import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_SEED,
    QUERY_TYPE_NAMES,
)
from recipe_mpr_qa.data.models import (
    DatasetValidationError,
    PreparedDataset,
    RecipeExample,
    RecipeOption,
    SplitManifest,
)


def _sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_example_id(index: int) -> str:
    return f"rmpr-{index + 1:04d}"


def _validate_raw_record(record: Mapping[str, Any], index: int) -> None:
    required = {"query", "query_type", "options", "answer", "correctness_explanation"}
    missing = required.difference(record.keys())
    if missing:
        raise DatasetValidationError(f"raw record {index} missing keys: {sorted(missing)}")

    query = record["query"]
    if not isinstance(query, str) or not query.strip():
        raise DatasetValidationError(f"raw record {index} query must be a non-empty string")

    query_type = record["query_type"]
    if not isinstance(query_type, Mapping):
        raise DatasetValidationError(f"raw record {index} query_type must be a mapping")
    if set(query_type.keys()) != set(QUERY_TYPE_NAMES):
        raise DatasetValidationError(
            f"raw record {index} query_type keys must be {QUERY_TYPE_NAMES}"
        )
    for key in QUERY_TYPE_NAMES:
        if query_type[key] not in (0, 1, False, True):
            raise DatasetValidationError(
                f"raw record {index} query_type[{key!r}] must be boolean-like"
            )

    options = record["options"]
    if not isinstance(options, Mapping):
        raise DatasetValidationError(f"raw record {index} options must be a mapping")
    if len(options) != 5:
        raise DatasetValidationError(f"raw record {index} must contain exactly 5 options")
    option_ids = list(options.keys())
    if len(set(option_ids)) != len(option_ids):
        raise DatasetValidationError(f"raw record {index} contains duplicate option ids")
    for option_id, option_text in options.items():
        if not isinstance(option_id, str) or not option_id.strip():
            raise DatasetValidationError(f"raw record {index} has an invalid option id")
        if not isinstance(option_text, str) or not option_text.strip():
            raise DatasetValidationError(
                f"raw record {index} option {option_id!r} must be a non-empty string"
            )

    answer = record["answer"]
    if not isinstance(answer, str) or answer not in options:
        raise DatasetValidationError(f"raw record {index} answer must match one option id")

    explanation = record["correctness_explanation"]
    if not isinstance(explanation, Mapping) or not explanation:
        raise DatasetValidationError(
            f"raw record {index} correctness_explanation must be a non-empty mapping"
        )
    for key, value in explanation.items():
        if not isinstance(key, str) or not key.strip():
            raise DatasetValidationError(f"raw record {index} explanation keys must be non-empty")
        if isinstance(value, str) and value.strip():
            continue
        if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
            continue
        raise DatasetValidationError(
            f"raw record {index} explanation values must be non-empty strings or non-empty string lists"
        )


def prepare_examples(raw_records: Sequence[Mapping[str, Any]], raw_path: Path) -> tuple[RecipeExample, ...]:
    prepared_examples: list[RecipeExample] = []
    for index, record in enumerate(raw_records):
        _validate_raw_record(record, index)
        normalized_query = record["query"].strip()
        normalization_steps: list[str] = []
        if normalized_query != record["query"]:
            normalization_steps.append("strip_query_outer_whitespace")
        prepared_examples.append(
            RecipeExample(
                example_id=_build_example_id(index),
                query=normalized_query,
                options=tuple(
                    RecipeOption(option_id=option_id, text=option_text)
                    for option_id, option_text in record["options"].items()
                ),
                answer_option_id=record["answer"],
                query_type_flags={name: bool(record["query_type"][name]) for name in QUERY_TYPE_NAMES},
                correctness_explanation=dict(record["correctness_explanation"]),
                source_metadata={
                    "raw_dataset_path": _normalize_source_path(raw_path),
                    "raw_index": index,
                    "normalization": normalization_steps,
                },
            )
        )
    return tuple(prepared_examples)


def build_dataset_metadata(
    examples: Sequence[RecipeExample], *, source_path: Path, raw_sha256: str | None = None
) -> dict[str, Any]:
    query_type_counts = {
        query_type: sum(1 for example in examples if example.query_type_flags[query_type])
        for query_type in QUERY_TYPE_NAMES
    }
    signature_counts: dict[str, int] = {}
    for example in examples:
        signature_counts[example.query_type_signature] = (
            signature_counts.get(example.query_type_signature, 0) + 1
        )
    metadata = {
        "schema_version": "phase1.v1",
        "source_path": _normalize_source_path(source_path),
        "example_count": len(examples),
        "query_type_counts": query_type_counts,
        "query_type_signature_counts": dict(sorted(signature_counts.items())),
    }
    if raw_sha256 is not None:
        metadata["raw_sha256"] = raw_sha256
    return metadata


def prepare_dataset(raw_path: Path | str) -> PreparedDataset:
    resolved_path = Path(raw_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DatasetValidationError("raw dataset root must be a list")
    examples = prepare_examples(payload, resolved_path)
    metadata = build_dataset_metadata(
        examples,
        source_path=resolved_path,
        raw_sha256=_sha256_hex(resolved_path),
    )
    return PreparedDataset(examples=examples, metadata=metadata)


def write_prepared_dataset(dataset: PreparedDataset, output_path: Path | str) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(example.to_dict(), ensure_ascii=True, separators=(",", ":"))
        for example in dataset.examples
    ]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_prepared_dataset(dataset_path: Path | str) -> PreparedDataset:
    resolved_path = Path(dataset_path)
    lines = [
        line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    examples = tuple(RecipeExample.from_dict(json.loads(line)) for line in lines)
    metadata = build_dataset_metadata(examples, source_path=resolved_path)
    metadata["prepared_dataset_path"] = _normalize_source_path(resolved_path)
    return PreparedDataset(examples=examples, metadata=metadata)


def _allocate_group_counts(group_sizes: Mapping[str, int], target_total: int) -> dict[str, int]:
    total_size = sum(group_sizes.values())
    if target_total < 0 or target_total > total_size:
        raise DatasetValidationError("invalid target_total for split allocation")
    if target_total == 0:
        return {group: 0 for group in group_sizes}
    exact_counts = {
        group: (group_size * target_total) / total_size
        for group, group_size in group_sizes.items()
    }
    allocated = {
        group: min(group_sizes[group], math.floor(exact_counts[group]))
        for group in group_sizes
    }
    assigned = sum(allocated.values())
    ranked_groups = sorted(
        group_sizes,
        key=lambda group: (-(exact_counts[group] - allocated[group]), group),
    )
    index = 0
    while assigned < target_total:
        group = ranked_groups[index % len(ranked_groups)]
        if allocated[group] < group_sizes[group]:
            allocated[group] += 1
            assigned += 1
        index += 1
    return allocated


def _shuffle_strata(examples: Sequence[RecipeExample], seed: int) -> dict[str, list[str]]:
    grouped_ids: dict[str, list[str]] = {}
    for example in examples:
        grouped_ids.setdefault(example.query_type_signature, []).append(example.example_id)
    shuffled: dict[str, list[str]] = {}
    for signature, example_ids in grouped_ids.items():
        derived_seed = int(
            hashlib.sha256(f"{seed}:{signature}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        shuffled_ids = list(example_ids)
        random.Random(derived_seed).shuffle(shuffled_ids)
        shuffled[signature] = shuffled_ids
    return shuffled


def _build_signature_counts(
    examples: Sequence[RecipeExample], split_ids: Sequence[str]
) -> dict[str, int]:
    split_id_set = set(split_ids)
    counts: dict[str, int] = {}
    for example in examples:
        if example.example_id in split_id_set:
            counts[example.query_type_signature] = counts.get(example.query_type_signature, 0) + 1
    return dict(sorted(counts.items()))


def generate_primary_split(
    examples: Sequence[RecipeExample],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> SplitManifest:
    total_examples = len(examples)
    train_target = int(total_examples * ratios["train"])
    validation_target = int(total_examples * ratios["validation"])
    test_target = total_examples - train_target - validation_target

    shuffled_strata = _shuffle_strata(examples, seed)
    group_sizes = {signature: len(example_ids) for signature, example_ids in shuffled_strata.items()}
    train_counts = _allocate_group_counts(group_sizes, train_target)
    remaining_after_train = {
        signature: group_sizes[signature] - train_counts[signature] for signature in group_sizes
    }
    validation_counts = _allocate_group_counts(remaining_after_train, validation_target)

    split_sets = {"train": set(), "validation": set(), "test": set()}
    for signature, example_ids in shuffled_strata.items():
        train_cutoff = train_counts[signature]
        validation_cutoff = train_cutoff + validation_counts[signature]
        split_sets["train"].update(example_ids[:train_cutoff])
        split_sets["validation"].update(example_ids[train_cutoff:validation_cutoff])
        split_sets["test"].update(example_ids[validation_cutoff:])

    splits = {
        split_name: tuple(
            example.example_id for example in examples if example.example_id in split_sets[split_name]
        )
        for split_name in ("train", "validation", "test")
    }

    manifest_metadata = {
        "strategy": "stratified_by_query_type_signature",
        "seed": seed,
        "ratios": dict(ratios),
        "counts": {
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
        "query_type_signature_counts": {
            split_name: _build_signature_counts(examples, splits[split_name])
            for split_name in ("train", "validation", "test")
        },
    }
    if manifest_metadata["counts"]["test"] != test_target:
        raise DatasetValidationError("Generated test split size does not match target")
    return SplitManifest(splits=splits, metadata=manifest_metadata)


def write_split_manifest(split_manifest: SplitManifest, output_path: Path | str) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(split_manifest.to_dict(), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_split_manifest(split_manifest_path: Path | str) -> SplitManifest:
    resolved_path = Path(split_manifest_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return SplitManifest.from_dict(payload)
