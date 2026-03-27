from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.models import RecipeOption

LETTER_MAP = ("A", "B", "C", "D", "E")
OPTION_ORDER_SHUFFLE_SEED = 1508


@dataclass(frozen=True)
class PromptSpec:
    version: str
    template: str

    def render(
        self,
        *,
        query: str,
        options: Sequence[RecipeOption] | Mapping[str, str],
        shuffle_key: str | None = None,
        shuffle_seed: int = OPTION_ORDER_SHUFFLE_SEED,
    ) -> tuple[str, dict[str, str]]:
        return build_multiple_choice_prompt(
            query=query,
            options=options,
            prompt_spec=self,
            shuffle_key=shuffle_key,
            shuffle_seed=shuffle_seed,
        )


DEFAULT_PROMPT_SPEC = PromptSpec(
    version="recipe-mpr-mc-v1",
    template=(
        "Given the following recipe preference query, select the best matching recipe.\n\n"
        "Query: {query}\n\n"
        "Options:\n"
        "A) {option_a}\n"
        "B) {option_b}\n"
        "C) {option_c}\n"
        "D) {option_d}\n"
        "E) {option_e}\n\n"
        "Respond with only the letter (A-E) of the best option."
    ),
)


def build_multiple_choice_prompt(
    *,
    query: str,
    options: Sequence[RecipeOption] | Mapping[str, str],
    prompt_spec: PromptSpec = DEFAULT_PROMPT_SPEC,
    shuffle_key: str | None = None,
    shuffle_seed: int = OPTION_ORDER_SHUFFLE_SEED,
) -> tuple[str, dict[str, str]]:
    if isinstance(options, Mapping):
        option_items = list(options.items())
    else:
        option_items = [(option.option_id, option.text) for option in options]
    if len(option_items) != 5:
        raise ValueError(f"Expected exactly 5 options, got {len(option_items)}")
    if shuffle_key is not None:
        derived_seed = int(
            hashlib.sha256(f"{shuffle_seed}:{shuffle_key}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        shuffled_option_items = list(option_items)
        random.Random(derived_seed).shuffle(shuffled_option_items)
        option_items = shuffled_option_items
    letter_to_option_id = {
        LETTER_MAP[index]: option_items[index][0] for index in range(len(LETTER_MAP))
    }
    prompt = prompt_spec.template.format(
        query=query,
        option_a=option_items[0][1],
        option_b=option_items[1][1],
        option_c=option_items[2][1],
        option_d=option_items[3][1],
        option_e=option_items[4][1],
    )
    return prompt, letter_to_option_id


def parse_multiple_choice_response(
    response_text: str,
    options: dict[str, str] | None = None,
) -> str | None:
    text = response_text.strip().upper()
    if not text:
        return None
    exact_match = re.fullmatch(r"\(?([A-E])\)?[\)\.\:\-]?", text)
    if exact_match:
        return exact_match.group(1)
    option_match = re.fullmatch(r"OPTION\s*([A-E])[\)\.\:\-]?", text)
    if option_match:
        return option_match.group(1)

    explicit_patterns = (
        r"\\BOXED\{([A-E])\}",
        r"\b(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+ANSWER|BEST\s+OPTION|BEST\s+MATCHING\s+RECIPE|PREDICTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-E])\)?[\)\.\:\-]?(?=\s|$)",
        r"\b(?:I\s+CHOOSE|I\s+PICK|I\s+SELECT|CHOOSE|PICK|SELECT)\s+(?:OPTION\s*)?\(?([A-E])\)?(?=\b)",
        r"\b(?:MY\s+ANSWER|MY\s+CHOICE)\s*(?:IS|:)\s*\(?([A-E])\)?[\)\.\:\-]?(?=\s|$)",
    )
    latest_match: tuple[int, str] | None = None
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(1)
            if latest_match is None or match.end() > latest_match[0]:
                latest_match = (match.end(), candidate)
    if latest_match is not None:
        return latest_match[1]

    tail_match = re.search(r"(?:^|[\s>])\(?([A-E])\)?[\)\.\:\-]?\s*$", text)
    if tail_match:
        prefix = text[:tail_match.start(1)]
        if not re.search(r"(?:OPTION|CHOICE)\s*$", prefix):
            return tail_match.group(1)

    mentioned_letters = {
        match.group(1) or match.group(2)
        for match in re.finditer(r"OPTION\s+([A-E])\b|(?:^|[\s])([A-E])\)", text)
    }
    if len(mentioned_letters) < 3:
        leading_match = re.match(r"\s*\(?([A-E])\)?[\)\.\:\-]\s+\S", text)
        if leading_match:
            return leading_match.group(1)

    # Fallback: if options provided, try to match "the best option is <text>"
    if options:
        desc_match = None
        for match in re.finditer(r"BEST OPTION IS\s+(.+?)(?:\.|##|\Z)", text):
            desc_match = match
        if desc_match:
            candidate = desc_match.group(1).strip()
            best_letter = None
            best_score = 0
            for letter, option_text in options.items():
                option_upper = option_text.upper()
                # Score by number of words in option text that appear in candidate
                words = [w for w in option_upper.split() if len(w) > 3]
                if not words:
                    continue
                score = sum(1 for w in words if w in candidate) / len(words)
                if score > best_score:
                    best_score = score
                    best_letter = letter
            # Require at least 50% word overlap to avoid spurious matches
            if best_score >= 0.5:
                return best_letter

    return None


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    phase: str
    provider: str
    model_name: str
    split: str
    example_id: str
    prompt_version: str
    raw_response: str
    parsed_choice: str | None
    predicted_option_id: str | None
    gold_option_id: str
    is_correct: bool
    latency_ms: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "provider": self.provider,
            "model_name": self.model_name,
            "split": self.split,
            "example_id": self.example_id,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
            "parsed_choice": self.parsed_choice,
            "predicted_option_id": self.predicted_option_id,
            "gold_option_id": self.gold_option_id,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionRecord":
        return cls(
            run_id=payload["run_id"],
            phase=payload["phase"],
            provider=payload["provider"],
            model_name=payload["model_name"],
            split=payload["split"],
            example_id=payload["example_id"],
            prompt_version=payload["prompt_version"],
            raw_response=payload["raw_response"],
            parsed_choice=payload.get("parsed_choice"),
            predicted_option_id=payload.get("predicted_option_id"),
            gold_option_id=payload["gold_option_id"],
            is_correct=bool(payload["is_correct"]),
            latency_ms=payload.get("latency_ms"),
            metadata=payload.get("metadata", {}),
        )


def write_prediction_records(
    records: list[PredictionRecord] | tuple[PredictionRecord, ...], output_path: Path | str
) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=True, separators=(",", ":")) for record in records]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_prediction_records(input_path: Path | str) -> tuple[PredictionRecord, ...]:
    resolved_path = Path(input_path)
    lines = [line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(PredictionRecord.from_dict(json.loads(line)) for line in lines)
