from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.benchmark.provenance import BENCHMARK_CONTRACT_VERSION
from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.records import PredictionRecord, write_prediction_records


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use DistilBERT evaluation") from exc
    return torch


def _require_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use DistilBERT evaluation") from exc
    return AutoModel, AutoTokenizer


def mean_pool_embeddings(last_hidden_state, attention_mask):
    torch = _require_torch()
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def score_example_with_embeddings(
    *,
    example: RecipeExample,
    tokenizer,
    model,
    max_length: int,
    device: str,
) -> tuple[str, list[float]]:
    torch = _require_torch()
    model.eval()
    with torch.no_grad():
        query_inputs = tokenizer(
            example.query,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        option_inputs = tokenizer(
            [option.text for option in example.options],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        query_inputs = {key: value.to(device) for key, value in query_inputs.items()}
        option_inputs = {key: value.to(device) for key, value in option_inputs.items()}
        query_outputs = model(**query_inputs)
        option_outputs = model(**option_inputs)
        query_embedding = mean_pool_embeddings(
            query_outputs.last_hidden_state,
            query_inputs["attention_mask"],
        )
        option_embeddings = mean_pool_embeddings(
            option_outputs.last_hidden_state,
            option_inputs["attention_mask"],
        )
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.expand_as(option_embeddings),
            option_embeddings,
            dim=1,
        )
        scores = similarities.detach().cpu().tolist()
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return example.options[best_index].option_id, scores


def evaluate_vanilla_slm(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    output_path: Path | str | None,
    model_name: str = "distilbert-base-uncased",
    prompt_version: str = "embedding-similarity-v1",
    batch_size: int = 8,
    max_length: int = 128,
    tokenizer=None,
    model=None,
    device: str | None = None,
) -> tuple[PredictionRecord, ...]:
    del batch_size
    torch = _require_torch()
    if tokenizer is None or model is None:
        AutoModel, AutoTokenizer = _require_transformers()
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        model = model or AutoModel.from_pretrained(model_name)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)

    records: list[PredictionRecord] = []
    for example in examples:
        started = time.perf_counter()
        predicted_option_id, scores = score_example_with_embeddings(
            example=example,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            device=resolved_device,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        records.append(
            PredictionRecord(
                run_id=run_id,
                phase="phase2",
                provider="slm",
                model_name=model_name,
                split=split,
                example_id=example.example_id,
                prompt_version=prompt_version,
                raw_response=str(scores),
                parsed_choice=None,
                predicted_option_id=predicted_option_id,
                gold_option_id=example.answer_option_id,
                is_correct=predicted_option_id == example.answer_option_id,
                latency_ms=latency_ms,
                model_interface="classifier",
                decoding_mode="embedding_similarity",
                parse_status="not_applicable",
                contract_version=BENCHMARK_CONTRACT_VERSION,
                metadata={"scores": scores},
            )
        )
    if output_path is not None:
        write_prediction_records(tuple(records), output_path)
    return tuple(records)
