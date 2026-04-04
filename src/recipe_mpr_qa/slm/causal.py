from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.benchmark.provenance import BENCHMARK_CONTRACT_VERSION
from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.metrics import summarize_prediction_metrics
from recipe_mpr_qa.evaluation.records import PredictionRecord, write_prediction_records
from recipe_mpr_qa.llm.prompts import (
    CAUSAL_SLM_PROMPT_SPEC,
    LETTER_MAP,
    OPTION_ORDER_SHUFFLE_SEED,
    benchmark_prompt_metadata,
    build_causal_multiple_choice_prompt,
    parse_multiple_choice_response_detail,
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use causal SLM experiments") from exc
    return torch


def _require_transformers():
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use causal SLM experiments") from exc
    return AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator


def _require_peft():
    try:
        from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "Install recipe-mpr-qa[train] with peft to use LoRA causal fine-tuning"
        ) from exc
    return AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model


class _RowDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def _ensure_tokenizer_padding(tokenizer) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _apply_chat_template(tokenizer, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except ValueError:
            pass
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].capitalize()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n\n".join(rendered)


def build_causal_chat_example(
    example: RecipeExample,
    tokenizer=None,
    *,
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
) -> dict[str, Any]:
    prompt_metadata = benchmark_prompt_metadata(
        example_id=example.example_id,
        prompt_version=prompt_version,
        shuffle_seed=OPTION_ORDER_SHUFFLE_SEED,
    )
    prompt_text, letter_to_option_id = build_causal_multiple_choice_prompt(
        query=example.query,
        options=example.options,
        prompt_spec=CAUSAL_SLM_PROMPT_SPEC,
        shuffle_key=prompt_metadata["shuffle_key"],
        shuffle_seed=prompt_metadata["shuffle_seed"],
    )
    gold_letter = next(
        letter
        for letter, option_id in letter_to_option_id.items()
        if option_id == example.answer_option_id
    )
    user_messages = [{"role": "user", "content": prompt_text}]
    generation_prompt = _apply_chat_template(
        tokenizer,
        user_messages,
        add_generation_prompt=True,
    )
    training_text = _apply_chat_template(
        tokenizer,
        user_messages + [{"role": "assistant", "content": gold_letter}],
        add_generation_prompt=False,
    )
    return {
        "prompt_text": prompt_text,
        "generation_prompt": generation_prompt,
        "training_text": training_text,
        "gold_letter": gold_letter,
        "letter_to_option_id": letter_to_option_id,
        "prompt_metadata": prompt_metadata,
    }


def _move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _decode_generated_text(outputs, encoded_prompt: Mapping[str, Any], tokenizer) -> str:
    input_ids = encoded_prompt["input_ids"]
    prompt_length = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids[0])
    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _candidate_variants_for_letter(letter: str) -> tuple[str, ...]:
    return (
        letter,
        f" {letter}",
        f"\n{letter}",
    )


def _batched_sequence_logprob_scores(
    *,
    prompt_ids: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
    model,
    device: str,
    pad_token_id: int,
) -> list[float]:
    torch = _require_torch()
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if not candidate_token_ids:
        raise ValueError("candidate_token_ids must be non-empty")

    scores: list[float] = []
    prompt_length = len(prompt_ids)
    for candidate_ids in candidate_token_ids:
        sequence = list(prompt_ids) + list(candidate_ids)
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        total_score = 0.0
        for token_offset, token_id in enumerate(candidate_ids):
            logits_index = prompt_length - 1 + token_offset
            total_score += float(log_probs[0, logits_index, token_id].item())
        scores.append(total_score)
    return scores


def _load_causal_model_bundle(
    *,
    model_name: str,
    checkpoint_dir: Path | None = None,
):
    torch = _require_torch()
    AutoModelForCausalLM, AutoTokenizer, _Trainer, _TrainingArguments, _default_data_collator = (
        _require_transformers()
    )
    resolved_checkpoint = checkpoint_dir if checkpoint_dir is not None else Path(model_name)
    tokenizer_source = _resolve_tokenizer_source(resolved_checkpoint, model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    _ensure_tokenizer_padding(tokenizer)
    if resolved_checkpoint.exists() and (resolved_checkpoint / "adapter_config.json").exists():
        AutoPeftModelForCausalLM, _LoraConfig, _TaskType, _get_peft_model = _require_peft()
        model = AutoPeftModelForCausalLM.from_pretrained(
            resolved_checkpoint.as_posix(),
            torch_dtype=torch.float32,
        )
    else:
        model_source = resolved_checkpoint.as_posix() if resolved_checkpoint.exists() else model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.float32,
        )
    return tokenizer, model


def evaluate_causal_slm(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    output_path: Path | str | None,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
    decoding_mode: str = "generate",
    tokenizer=None,
    model=None,
    device: str | None = None,
) -> tuple[PredictionRecord, ...]:
    torch = _require_torch()
    if tokenizer is None or model is None:
        tokenizer, model = _load_causal_model_bundle(model_name=model_name)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()

    records: list[PredictionRecord] = []
    for example in examples:
        chat_example = build_causal_chat_example(
            example,
            tokenizer=tokenizer,
            prompt_version=prompt_version,
        )
        encoded = tokenizer(
            chat_example["generation_prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = _move_batch_to_device(encoded, resolved_device)
        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(temperature > 0),
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        started = time.perf_counter()
        if decoding_mode == "loglikelihood":
            parsed_choice, choice_scores = _score_choice_letters(encoded, tokenizer, model, resolved_device)
            response_text = json.dumps(
                {
                    "selected_choice": parsed_choice,
                    "choice_scores": choice_scores,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            parse_status = "not_applicable"
        else:
            if getattr(tokenizer, "pad_token_id", None) is not None:
                generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)
            response_text = _decode_generated_text(outputs, encoded, tokenizer)
            parse_result = parse_multiple_choice_response_detail(response_text)
            parsed_choice = parse_result.parsed_choice
            parse_status = parse_result.status
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_option_id = chat_example["letter_to_option_id"].get(parsed_choice) if parsed_choice is not None else None
        records.append(
            PredictionRecord(
                run_id=run_id,
                phase="phase2",
                provider="slm",
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
                parse_status=parse_status,
                contract_version=BENCHMARK_CONTRACT_VERSION,
                parser_version=chat_example["prompt_metadata"]["parser_version"],
                shuffle_key=chat_example["prompt_metadata"]["shuffle_key"],
                shuffle_seed=chat_example["prompt_metadata"]["shuffle_seed"],
                metadata={
                    "architecture": "causal_lm",
                    "option_mapping": chat_example["letter_to_option_id"],
                    **({"choice_scores": choice_scores} if decoding_mode == "loglikelihood" else {}),
                },
            )
        )
    if output_path is not None:
        write_prediction_records(tuple(records), output_path)
    return tuple(records)


def _build_causal_rows(
    examples: Sequence[RecipeExample],
    tokenizer,
    *,
    max_length: int,
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for example in examples:
        chat_example = build_causal_chat_example(
            example,
            tokenizer=tokenizer,
            prompt_version=prompt_version,
        )
        prompt_tokens = tokenizer(
            chat_example["generation_prompt"],
            truncation=True,
            max_length=max_length,
        )
        full_tokens = tokenizer(
            chat_example["training_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids = list(full_tokens["input_ids"])
        labels = list(input_ids)
        attention_mask = list(full_tokens.get("attention_mask", [1] * len(input_ids)))
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        for index in range(prompt_length):
            labels[index] = -100
        for index, is_active in enumerate(attention_mask):
            if not is_active:
                labels[index] = -100
        if not any(label != -100 for label in labels):
            raise ValueError(
                f"max_length={max_length} leaves no supervised target tokens for {example.example_id}"
            )
        row = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        rows.append(row)
        metadata.append(
            {
                "example_id": example.example_id,
                "gold_letter": chat_example["gold_letter"],
                "letter_to_option_id": chat_example["letter_to_option_id"],
            }
        )
    return rows, metadata


def _build_causal_trainer(
    *,
    model_name: str,
    checkpoint_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    tokenizer,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_train_epochs: float,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    AutoModelForCausalLM, _AutoTokenizer, Trainer, TrainingArguments, default_data_collator = (
        _require_transformers()
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if use_lora:
        _AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model = _require_peft()
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            ),
        )
    training_kwargs = {
        "output_dir": checkpoint_dir.as_posix(),
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": [],
    }
    training_args = TrainingArguments(
        **_add_eval_strategy_kwargs(TrainingArguments, training_kwargs)
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=_RowDataset(train_rows),
        eval_dataset=_RowDataset(eval_rows),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )


def _add_eval_strategy_kwargs(training_arguments_cls, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    init_parameters = training_arguments_cls.__init__.__code__.co_varnames
    resolved = dict(kwargs)
    if "evaluation_strategy" in init_parameters:
        resolved["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in init_parameters:
        resolved["eval_strategy"] = "epoch"
    return resolved


def train_causal_slm(
    *,
    train_examples: Sequence[RecipeExample],
    validation_examples: Sequence[RecipeExample],
    test_examples: Sequence[RecipeExample],
    run_id: str,
    output_dir: Path | str,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
    learning_rate: float = 2e-4,
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    num_train_epochs: float = 3.0,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    decoding_mode: str = "generate",
):
    _AutoModelForCausalLM, AutoTokenizer, _Trainer, _TrainingArguments, _default_data_collator = (
        _require_transformers()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_tokenizer_padding(tokenizer)
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_rows, _train_meta = _build_causal_rows(
        train_examples,
        tokenizer,
        max_length=max_length,
        prompt_version=prompt_version,
    )
    validation_rows, _validation_meta = _build_causal_rows(
        validation_examples,
        tokenizer,
        max_length=max_length,
        prompt_version=prompt_version,
    )
    trainer = _build_causal_trainer(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        train_rows=train_rows,
        eval_rows=validation_rows,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    trainer.train()
    trainer.save_model(checkpoint_dir.as_posix())
    tokenizer.save_pretrained(checkpoint_dir.as_posix())
    best_checkpoint_dir, checkpoint_scores = _select_best_causal_checkpoint(
        checkpoint_root=checkpoint_dir,
        validation_examples=validation_examples,
        run_id=run_id,
        model_name=model_name,
        prompt_version=prompt_version,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        decoding_mode=decoding_mode,
    )
    validation_records = evaluate_finetuned_causal_model(
        examples=validation_examples,
        run_id=run_id,
        split="validation",
        checkpoint_dir=best_checkpoint_dir,
        output_path=checkpoint_dir.parent / "validation_predictions.jsonl",
        model_name=model_name,
        prompt_version=prompt_version,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        decoding_mode=decoding_mode,
    )
    test_records = evaluate_finetuned_causal_model(
        examples=test_examples,
        run_id=run_id,
        split="test",
        checkpoint_dir=best_checkpoint_dir,
        output_path=checkpoint_dir.parent / "test_predictions.jsonl",
        model_name=model_name,
        prompt_version=prompt_version,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        decoding_mode=decoding_mode,
    )
    (checkpoint_dir.parent / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "best_checkpoint_dir": best_checkpoint_dir.as_posix(),
                "checkpoints": checkpoint_scores,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "trainer": trainer,
        "validation_records": validation_records,
        "test_records": test_records,
        "checkpoint_dir": checkpoint_dir,
        "best_checkpoint_dir": best_checkpoint_dir,
        "checkpoint_scores": checkpoint_scores,
    }


def evaluate_finetuned_causal_model(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    checkpoint_dir: Path | str,
    output_path: Path | str | None,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
    decoding_mode: str = "generate",
):
    tokenizer, model = _load_causal_model_bundle(
        model_name=model_name,
        checkpoint_dir=Path(checkpoint_dir),
    )
    return evaluate_causal_slm(
        examples=examples,
        run_id=run_id,
        split=split,
        output_path=output_path,
        model_name=model_name,
        prompt_version=prompt_version,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        decoding_mode=decoding_mode,
        tokenizer=tokenizer,
        model=model,
    )


def _resolve_tokenizer_source(checkpoint_dir: Path, *, model_name: str) -> str:
    for candidate in (checkpoint_dir, checkpoint_dir.parent):
        if any((candidate / file_name).exists() for file_name in ("tokenizer.json", "tokenizer_config.json")):
            return candidate.as_posix()
    return model_name


def _score_choice_letters(
    encoded_prompt: Mapping[str, Any],
    tokenizer,
    model,
    device: str,
) -> tuple[str, dict[str, float]]:
    prompt_ids_tensor = encoded_prompt["input_ids"][0]
    prompt_ids = (
        prompt_ids_tensor.tolist() if hasattr(prompt_ids_tensor, "tolist") else list(prompt_ids_tensor)
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    variant_letters: list[str] = []
    variant_token_ids: list[Sequence[int]] = []
    for letter in LETTER_MAP:
        for candidate_text in _candidate_variants_for_letter(letter):
            token_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
            if token_ids:
                variant_letters.append(letter)
                variant_token_ids.append(token_ids)

    scores = _batched_sequence_logprob_scores(
        prompt_ids=prompt_ids,
        candidate_token_ids=variant_token_ids,
        model=model,
        device=device,
        pad_token_id=pad_token_id,
    )
    letter_scores = {letter: float("-inf") for letter in LETTER_MAP}
    for letter, score in zip(variant_letters, scores):
        if score > letter_scores[letter]:
            letter_scores[letter] = score
    best_letter = max(letter_scores.items(), key=lambda item: item[1])[0]
    return best_letter, letter_scores


def _list_checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    checkpoint_dirs = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ] if checkpoint_root.exists() else []
    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.split("-", 1)[-1]
        return (int(suffix) if suffix.isdigit() else -1, path.name)
    ordered = sorted(checkpoint_dirs, key=sort_key)
    if checkpoint_root.exists():
        ordered.append(checkpoint_root)
    return ordered or [checkpoint_root]


def _select_best_causal_checkpoint(
    *,
    checkpoint_root: Path,
    validation_examples: Sequence[RecipeExample],
    run_id: str,
    model_name: str,
    prompt_version: str,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    decoding_mode: str,
) -> tuple[Path, list[dict[str, Any]]]:
    checkpoint_scores: list[dict[str, Any]] = []
    best_checkpoint_dir = checkpoint_root
    best_accuracy = float("-inf")
    for checkpoint_dir in _list_checkpoint_dirs(checkpoint_root):
        records = evaluate_finetuned_causal_model(
            examples=validation_examples,
            run_id=run_id,
            split="validation",
            checkpoint_dir=checkpoint_dir,
            output_path=None,
            model_name=model_name,
            prompt_version=prompt_version,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            decoding_mode=decoding_mode,
        )
        metrics = summarize_prediction_metrics(
            records,
            type("Dataset", (), {"examples": tuple(validation_examples)})(),
        )
        checkpoint_scores.append(
            {
                "checkpoint_dir": checkpoint_dir.as_posix(),
                "accuracy": metrics["accuracy"],
                "correct_count": metrics["correct_count"],
                "parse_failure_count": metrics["parse_failure_count"],
            }
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_checkpoint_dir = checkpoint_dir
    return best_checkpoint_dir, checkpoint_scores
