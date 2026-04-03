from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.records import (
    PredictionRecord,
    read_prediction_records,
    write_prediction_records,
)
from recipe_mpr_qa.llm.prompts import (
    DEFAULT_PROMPT_SPEC,
    OPTION_ORDER_SHUFFLE_SEED,
    benchmark_prompt_metadata,
    build_multiple_choice_prompt,
    parse_multiple_choice_response_detail,
)


def run_llm_predictions(
    *,
    examples: Sequence[RecipeExample],
    client,
    run_id: str,
    provider: str,
    model_name: str,
    split: str,
    output_path: Path | str,
    prompt_version: str = DEFAULT_PROMPT_SPEC.version,
    temperature: float = 0.0,
    resume: bool = True,
    decoding_mode: str = "generate",
) -> tuple[PredictionRecord, ...]:
    output_path = Path(output_path)
    existing_records: dict[str, PredictionRecord] = {}
    if resume and output_path.exists():
        existing_records = {
            record.example_id: record for record in read_prediction_records(output_path)
        }

    completed = dict(existing_records)
    for example in examples:
        if example.example_id in completed:
            continue
        prompt_metadata = benchmark_prompt_metadata(
            example_id=example.example_id,
            prompt_version=prompt_version,
            shuffle_seed=OPTION_ORDER_SHUFFLE_SEED,
        )
        prompt, letter_to_option_id = build_multiple_choice_prompt(
            query=example.query,
            options=example.options,
            shuffle_key=prompt_metadata["shuffle_key"],
            shuffle_seed=prompt_metadata["shuffle_seed"],
        )
        started = time.perf_counter()
        response_text = client.generate(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        parse_result = parse_multiple_choice_response_detail(response_text)
        parsed_choice = parse_result.parsed_choice
        predicted_option_id = (
            letter_to_option_id[parsed_choice] if parsed_choice in letter_to_option_id else None
        )
        completed[example.example_id] = PredictionRecord(
            run_id=run_id,
            phase="phase3",
            provider=provider,
            model_name=model_name,
            split=split,
            example_id=example.example_id,
            prompt_version=prompt_version,
            raw_response=response_text,
            parsed_choice=parsed_choice,
            predicted_option_id=predicted_option_id,
            gold_option_id=example.answer_option_id,
            is_correct=predicted_option_id == example.answer_option_id,
            latency_ms=latency_ms,
            model_interface="generative",
            decoding_mode=decoding_mode,
            parse_status=parse_result.status,
            contract_version=prompt_metadata["contract_version"],
            parser_version=prompt_metadata["parser_version"],
            shuffle_key=prompt_metadata["shuffle_key"],
            shuffle_seed=prompt_metadata["shuffle_seed"],
            metadata={"option_mapping": letter_to_option_id},
        )
        ordered_records = tuple(completed[key] for key in sorted(completed))
        write_prediction_records(ordered_records, output_path)
    return tuple(completed[key] for key in sorted(completed))
