from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES


class DatasetValidationError(ValueError):
    """Raised when raw or prepared dataset content violates the schema."""


def _validate_query_type_flags(flags: Mapping[str, Any]) -> dict[str, bool]:
    if set(flags.keys()) != set(QUERY_TYPE_NAMES):
        raise DatasetValidationError(
            f"query_type_flags must contain exactly {QUERY_TYPE_NAMES}, got {sorted(flags.keys())}"
        )
    normalized_flags: dict[str, bool] = {}
    for key in QUERY_TYPE_NAMES:
        value = flags[key]
        if value not in (0, 1, False, True):
            raise DatasetValidationError(
                f"query_type flag {key!r} must be boolean-like, got {value!r}"
            )
        normalized_flags[key] = bool(value)
    return normalized_flags


def _validate_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise DatasetValidationError(f"{name} must be non-empty")
    return value


@dataclass(frozen=True)
class RecipeOption:
    option_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_text("option_id", self.option_id)
        _validate_text("option_text", self.text)

    def to_dict(self) -> dict[str, str]:
        return {
            "option_id": self.option_id,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecipeOption":
        if "option_id" not in payload or "text" not in payload:
            raise DatasetValidationError("prepared option rows require option_id and text")
        return cls(
            option_id=_validate_text("option_id", payload["option_id"]),
            text=_validate_text("option_text", payload["text"]),
        )


@dataclass(frozen=True)
class RecipeExample:
    example_id: str
    query: str
    options: tuple[RecipeOption, ...]
    answer_option_id: str
    query_type_flags: Mapping[str, bool]
    correctness_explanation: Mapping[str, Any]
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_text("example_id", self.example_id)
        _validate_text("query", self.query)
        if len(self.options) != 5:
            raise DatasetValidationError(
                f"{self.example_id} must contain exactly 5 options, got {len(self.options)}"
            )
        option_ids = [option.option_id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise DatasetValidationError(f"{self.example_id} contains duplicate option ids")
        _validate_text("answer_option_id", self.answer_option_id)
        if self.answer_option_id not in option_ids:
            raise DatasetValidationError(
                f"{self.example_id} answer_option_id {self.answer_option_id!r} not found in options"
            )
        normalized_flags = _validate_query_type_flags(self.query_type_flags)
        object.__setattr__(self, "query_type_flags", normalized_flags)
        if not isinstance(self.correctness_explanation, Mapping) or not self.correctness_explanation:
            raise DatasetValidationError("correctness_explanation must be a non-empty mapping")
        normalized_explanation: dict[str, Any] = {}
        for key, value in self.correctness_explanation.items():
            normalized_key = _validate_text("correctness_explanation key", key)
            if isinstance(value, str):
                normalized_explanation[normalized_key] = _validate_text(
                    "correctness_explanation value", value
                )
                continue
            if isinstance(value, (list, tuple)) and value:
                normalized_explanation[normalized_key] = tuple(
                    _validate_text("correctness_explanation value", item) for item in value
                )
                continue
            raise DatasetValidationError(
                "correctness_explanation values must be non-empty strings or non-empty string lists"
            )
        object.__setattr__(self, "correctness_explanation", normalized_explanation)
        if not isinstance(self.source_metadata, Mapping):
            raise DatasetValidationError("source_metadata must be a mapping")
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))

    @property
    def query_type_signature(self) -> str:
        active_flags = [name for name in QUERY_TYPE_NAMES if self.query_type_flags[name]]
        return "|".join(active_flags) if active_flags else "None"

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "query": self.query,
            "options": [option.to_dict() for option in self.options],
            "answer_option_id": self.answer_option_id,
            "query_type_flags": {name: self.query_type_flags[name] for name in QUERY_TYPE_NAMES},
            "correctness_explanation": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.correctness_explanation.items()
            },
            "source_metadata": dict(self.source_metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecipeExample":
        required = {
            "example_id",
            "query",
            "options",
            "answer_option_id",
            "query_type_flags",
            "correctness_explanation",
            "source_metadata",
        }
        missing = required.difference(payload.keys())
        if missing:
            raise DatasetValidationError(f"prepared example missing keys: {sorted(missing)}")
        options_payload = payload["options"]
        if not isinstance(options_payload, list):
            raise DatasetValidationError("prepared example options must be a list")
        return cls(
            example_id=_validate_text("example_id", payload["example_id"]),
            query=_validate_text("query", payload["query"]),
            options=tuple(RecipeOption.from_dict(option) for option in options_payload),
            answer_option_id=_validate_text("answer_option_id", payload["answer_option_id"]),
            query_type_flags=payload["query_type_flags"],
            correctness_explanation=payload["correctness_explanation"],
            source_metadata=payload["source_metadata"],
        )


@dataclass(frozen=True)
class PreparedDataset:
    examples: tuple[RecipeExample, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        example_ids = [example.example_id for example in self.examples]
        if len(set(example_ids)) != len(example_ids):
            raise DatasetValidationError("PreparedDataset contains duplicate example ids")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def get_example(self, example_id: str) -> RecipeExample:
        for example in self.examples:
            if example.example_id == example_id:
                return example
        raise KeyError(example_id)

    def to_jsonl_rows(self) -> list[dict[str, Any]]:
        return [example.to_dict() for example in self.examples]


@dataclass(frozen=True)
class SplitManifest:
    splits: Mapping[str, tuple[str, ...]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_splits: dict[str, tuple[str, ...]] = {}
        all_ids: list[str] = []
        for split_name, example_ids in self.splits.items():
            if split_name not in {"train", "validation", "test"}:
                raise DatasetValidationError(f"Unexpected split name: {split_name}")
            normalized_splits[split_name] = tuple(example_ids)
            all_ids.extend(example_ids)
        if len(set(all_ids)) != len(all_ids):
            raise DatasetValidationError("SplitManifest contains duplicate example ids across splits")
        object.__setattr__(self, "splits", normalized_splits)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "splits": {name: list(example_ids) for name, example_ids in self.splits.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitManifest":
        if "metadata" not in payload or "splits" not in payload:
            raise DatasetValidationError("split manifest requires metadata and splits")
        splits = payload["splits"]
        if not isinstance(splits, Mapping):
            raise DatasetValidationError("split manifest splits must be a mapping")
        return cls(
            splits={name: tuple(example_ids) for name, example_ids in splits.items()},
            metadata=payload["metadata"],
        )


@dataclass(frozen=True)
class OptionScoringExample:
    example_id: str
    option_id: str
    option_index: int
    group_size: int
    query: str
    option_text: str
    label: int
    tokenized_inputs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_text("example_id", self.example_id)
        _validate_text("option_id", self.option_id)
        _validate_text("query", self.query)
        _validate_text("option_text", self.option_text)
        if self.option_index < 0:
            raise DatasetValidationError("option_index must be >= 0")
        if self.group_size <= 0:
            raise DatasetValidationError("group_size must be > 0")
        if self.label not in (0, 1):
            raise DatasetValidationError("label must be 0 or 1")
        if self.tokenized_inputs is not None and not isinstance(self.tokenized_inputs, Mapping):
            raise DatasetValidationError("tokenized_inputs must be a mapping when provided")

    def to_model_input(self) -> dict[str, Any]:
        payload = dict(self.tokenized_inputs or {})
        payload.update(
            {
                "example_id": self.example_id,
                "option_id": self.option_id,
                "option_index": self.option_index,
                "group_size": self.group_size,
                "label": self.label,
            }
        )
        return payload
