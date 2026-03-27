from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.loaders import get_split_examples, load_dataset, load_split_manifest
from recipe_mpr_qa.data.models import DatasetValidationError
from recipe_mpr_qa.formats import (
    DEFAULT_PROMPT_SPEC,
    OPTION_ORDER_SHUFFLE_SEED,
    build_multiple_choice_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def seed_everything(seed: int):
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RunConfig:
    model_name: str
    data_path: str
    split_manifest_path: str
    output_dir: str
    augmented_train_path: str | None = None

    seed: int = 42
    max_seq_length: int = 512

    num_train_epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2

    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    completion_only_loss: bool = True
    report_to: str = "none"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small generative LM on Recipe-MPR using prompt-completion SFT."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HF model ID, e.g. HuggingFaceTB/SmolLM2-135M-Instruct or SmolLM2-360M-Instruct",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"),
        help="Path to processed dataset JSONL",
    )
    parser.add_argument(
        "--split-manifest-path",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "primary_split.json"),
        help="Path to split manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "smollm2_recipe_mpr"),
        help="Where to save checkpoints and config",
    )
    parser.add_argument(
        "--augmented-train-path",
        type=str,
        default=None,
        help=(
            "Optional path to a pre-generated augmented train JSONL artifact in "
            "RecipeExample format. This script reads an existing file; it does "
            "not create augmentation automatically."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=512)

    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch",
        choices=["no", "epoch", "steps"],
    )
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=str2bool, default=False)

    parser.add_argument("--use-lora", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of module names to adapt with LoRA",
    )

    parser.add_argument("--completion-only-loss", type=str2bool, default=True)
    parser.add_argument("--report-to", type=str, default="none")
    return parser


def parse_args(argv: Sequence[str] | None = None):
    return build_arg_parser().parse_args(argv)


def namespace_to_run_config(args) -> RunConfig:
    return RunConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        split_manifest_path=args.split_manifest_path,
        output_dir=args.output_dir,
        augmented_train_path=args.augmented_train_path,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        completion_only_loss=args.completion_only_loss,
        report_to=args.report_to,
    )


def example_to_prompt_completion(example):
    prompt, letter_to_id = build_multiple_choice_prompt(
        query=example.query,
        options=example.options,
        shuffle_key=example.example_id,
    )

    gold_letter = next(
        letter for letter, option_id in letter_to_id.items() if option_id == example.answer_option_id
    )

    return {
        "prompt": prompt,
        "completion": gold_letter,
        "example_id": example.example_id,
        "answer_option_id": example.answer_option_id,
        "query_type_flags": dict(example.query_type_flags),
    }


def _load_augmented_train_examples(augmented_train_path: str, train_examples):
    augmented_dataset = load_dataset(augmented_train_path)
    train_example_ids = {example.example_id for example in train_examples}
    augmented_examples = augmented_dataset.examples
    conflicting_ids = sorted(
        train_example_ids.intersection(example.example_id for example in augmented_examples)
    )
    if conflicting_ids:
        raise DatasetValidationError(
            f"augmented train examples must not reuse original example ids: {conflicting_ids[:5]}"
        )
    for example in augmented_examples:
        parent_example_id = example.source_metadata.get("parent_example_id")
        if parent_example_id not in train_example_ids:
            raise DatasetValidationError(
                f"augmented example {example.example_id} must reference a train parent_example_id"
            )
    return augmented_examples


def build_hf_datasets(
    data_path: str,
    split_manifest_path: str,
    augmented_train_path: str | None = None,
):
    from datasets import Dataset

    dataset = load_dataset(data_path)
    manifest = load_split_manifest(split_manifest_path)

    train_examples = get_split_examples(dataset, manifest, "train")
    val_examples = get_split_examples(dataset, manifest, "validation")
    test_examples = get_split_examples(dataset, manifest, "test")
    if augmented_train_path is not None:
        train_examples = train_examples + _load_augmented_train_examples(
            augmented_train_path,
            train_examples,
        )

    train_rows = [example_to_prompt_completion(example) for example in train_examples]
    val_rows = [example_to_prompt_completion(example) for example in val_examples]
    test_rows = [example_to_prompt_completion(example) for example in test_examples]

    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(val_rows),
        Dataset.from_list(test_rows),
    )


def run_finetune(cfg: RunConfig) -> dict[str, object]:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    os.makedirs(cfg.output_dir, exist_ok=True)
    seed_everything(cfg.seed)

    print("=" * 80)
    print("Recipe-MPR fine-tuning run")
    print("=" * 80)
    print(json.dumps(asdict(cfg), indent=2))

    train_ds, val_ds, test_ds = build_hf_datasets(
        data_path=cfg.data_path,
        split_manifest_path=cfg.split_manifest_path,
        augmented_train_path=cfg.augmented_train_path,
    )

    print(f"Train examples: {len(train_ds)}")
    print(f"Val examples:   {len(val_ds)}")
    print(f"Test examples:  {len(test_ds)}")
    print("-" * 80)
    print("Sample train row:")
    print(train_ds[0])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    peft_config = None
    if cfg.use_lora:
        target_modules = [item.strip() for item in cfg.lora_target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        max_length=cfg.max_seq_length,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        completion_only_loss=cfg.completion_only_loss,
        report_to=cfg.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if cfg.use_lora:
        trainer.model.print_trainable_parameters()

    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    run_config_path = os.path.join(cfg.output_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    log_history_path = os.path.join(cfg.output_dir, "log_history.json")
    with open(log_history_path, "w", encoding="utf-8") as handle:
        json.dump(trainer.state.log_history, handle, indent=2)

    trainer_state_path = os.path.join(cfg.output_dir, "trainer_state_summary.json")
    trainer_state = {
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "log_history": trainer.state.log_history,
    }
    with open(trainer_state_path, "w", encoding="utf-8") as handle:
        json.dump(trainer_state, handle, indent=2, default=str)

    print(f"Saved log history to: {log_history_path}")
    print(f"Saved trainer state summary to: {trainer_state_path}")

    print("=" * 80)
    print(f"Training complete. Final artifacts saved to: {final_dir}")
    print("=" * 80)

    return {
        "config": asdict(cfg),
        "output_dir": cfg.output_dir,
        "final_dir": final_dir,
        "run_config_path": run_config_path,
        "log_history_path": log_history_path,
        "trainer_state_path": trainer_state_path,
        "trainer_state": trainer_state,
        "dataset_sizes": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
        "model": {
            "name": cfg.model_name,
            "use_lora": cfg.use_lora,
            "backend": "huggingface",
        },
        "prompt": {
            "version": DEFAULT_PROMPT_SPEC.version,
            "option_order": "deterministic_per_example_shuffle",
            "shuffle_seed": OPTION_ORDER_SHUFFLE_SEED,
        },
    }


def run_training_from_arg_list(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run_finetune(namespace_to_run_config(parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run_training_from_arg_list(argv)


if __name__ == "__main__":
    main()
