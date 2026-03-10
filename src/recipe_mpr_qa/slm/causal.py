from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.records import PredictionRecord, write_prediction_records
from recipe_mpr_qa.llm.prompts import (
    CAUSAL_SLM_PROMPT_SPEC,
    build_causal_multiple_choice_prompt,
    parse_multiple_choice_response,
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
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].capitalize()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n\n".join(rendered)


def build_causal_chat_example(example: RecipeExample, tokenizer=None) -> dict[str, Any]:
    prompt_text, letter_to_option_id = build_causal_multiple_choice_prompt(
        query=example.query,
        options=example.options,
        prompt_spec=CAUSAL_SLM_PROMPT_SPEC,
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


def _load_causal_model_bundle(
    *,
    model_name: str,
    checkpoint_dir: Path | None = None,
):
    AutoModelForCausalLM, AutoTokenizer, _Trainer, _TrainingArguments, _default_data_collator = (
        _require_transformers()
    )
    resolved_checkpoint = checkpoint_dir if checkpoint_dir is not None else Path(model_name)
    tokenizer_source = resolved_checkpoint.as_posix() if resolved_checkpoint.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    _ensure_tokenizer_padding(tokenizer)
    if resolved_checkpoint.exists() and (resolved_checkpoint / "adapter_config.json").exists():
        AutoPeftModelForCausalLM, _LoraConfig, _TaskType, _get_peft_model = _require_peft()
        model = AutoPeftModelForCausalLM.from_pretrained(resolved_checkpoint.as_posix())
    else:
        model_source = resolved_checkpoint.as_posix() if resolved_checkpoint.exists() else model_name
        model = AutoModelForCausalLM.from_pretrained(model_source)
    return tokenizer, model


def evaluate_causal_slm(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    output_path: Path | str,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
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
        chat_example = build_causal_chat_example(example, tokenizer=tokenizer)
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
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": bool(temperature > 0),
        }
        if getattr(tokenizer, "pad_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response_text = _decode_generated_text(outputs, encoded, tokenizer)
        parsed_choice = parse_multiple_choice_response(response_text)
        predicted_option_id = (
            chat_example["letter_to_option_id"].get(parsed_choice) if parsed_choice is not None else None
        )
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
                metadata={
                    "architecture": "causal_lm",
                    "option_mapping": chat_example["letter_to_option_id"],
                },
            )
        )
    write_prediction_records(tuple(records), output_path)
    return tuple(records)


def _build_causal_rows(
    examples: Sequence[RecipeExample],
    tokenizer,
    *,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for example in examples:
        chat_example = build_causal_chat_example(example, tokenizer=tokenizer)
        prompt_tokens = tokenizer(
            chat_example["generation_prompt"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        full_tokens = tokenizer(
            chat_example["training_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids = list(full_tokens["input_ids"])
        labels = list(input_ids)
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        for index in range(prompt_length):
            labels[index] = -100
        row = {
            "input_ids": input_ids,
            "attention_mask": list(full_tokens.get("attention_mask", [1] * len(input_ids))),
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
    training_args = TrainingArguments(
        output_dir=checkpoint_dir.as_posix(),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=_RowDataset(train_rows),
        eval_dataset=_RowDataset(eval_rows),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )


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
):
    _AutoModelForCausalLM, AutoTokenizer, _Trainer, _TrainingArguments, _default_data_collator = (
        _require_transformers()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_tokenizer_padding(tokenizer)
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_rows, _train_meta = _build_causal_rows(train_examples, tokenizer, max_length=max_length)
    validation_rows, _validation_meta = _build_causal_rows(
        validation_examples,
        tokenizer,
        max_length=max_length,
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
    validation_records = evaluate_causal_slm(
        examples=validation_examples,
        run_id=run_id,
        split="validation",
        output_path=checkpoint_dir.parent / "validation_predictions.jsonl",
        model_name=model_name,
        prompt_version=prompt_version,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tokenizer=tokenizer,
        model=trainer.model,
    )
    test_records = evaluate_causal_slm(
        examples=test_examples,
        run_id=run_id,
        split="test",
        output_path=checkpoint_dir.parent / "test_predictions.jsonl",
        model_name=model_name,
        prompt_version=prompt_version,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tokenizer=tokenizer,
        model=trainer.model,
    )
    return {
        "trainer": trainer,
        "validation_records": validation_records,
        "test_records": test_records,
        "checkpoint_dir": checkpoint_dir,
    }


def evaluate_finetuned_causal_model(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    checkpoint_dir: Path | str,
    output_path: Path | str,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    prompt_version: str = CAUSAL_SLM_PROMPT_SPEC.version,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
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
        tokenizer=tokenizer,
        model=model,
    )
