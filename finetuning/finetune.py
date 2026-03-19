import os
import sys
import json
import random
import argparse
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


# ----------------------------
# Path setup so this script works from finetuning/
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from recipe_mpr_qa.formats import build_multiple_choice_prompt
from recipe_mpr_qa.data.loaders import (
    load_dataset,
    load_split_manifest,
    get_split_examples,
)


# ----------------------------
# Helpers
# ----------------------------
def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {x}")


def seed_everything(seed: int):
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

    # For Llama-style small models, these are common defaults.
    # can expand this list later if we want to compare adapter coverage.
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    completion_only_loss: bool = True
    report_to: str = "none"


def parse_args():
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
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
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

    return parser.parse_args()


# ----------------------------
# Dataset conversion
# ----------------------------
def example_to_prompt_completion(example):
    """
    Convert one Recipe-MPR example into the exact prompt/completion form
    expected by your baseline evaluator.

    prompt: the multiple-choice prompt text
    completion: the correct answer letter (A-E)
    """
    prompt, letter_to_id = build_multiple_choice_prompt(
        query=example.query,
        options=example.options,
    )

    gold_letter = next(
        letter for letter, option_id in letter_to_id.items()
        if option_id == example.answer_option_id
    )

    return {
        "prompt": prompt,
        "completion": gold_letter,
        "example_id": example.example_id,
        "answer_option_id": example.answer_option_id,
        "query_type_flags": dict(example.query_type_flags),
    }


def build_hf_datasets(data_path: str, split_manifest_path: str):
    dataset = load_dataset(data_path)
    manifest = load_split_manifest(split_manifest_path)

    train_examples = get_split_examples(dataset, manifest, "train")
    val_examples = get_split_examples(dataset, manifest, "validation")
    test_examples = get_split_examples(dataset, manifest, "test")

    train_rows = [example_to_prompt_completion(ex) for ex in train_examples]
    val_rows = [example_to_prompt_completion(ex) for ex in val_examples]
    test_rows = [example_to_prompt_completion(ex) for ex in test_examples]

    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(val_rows),
        Dataset.from_list(test_rows),
    )


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    cfg = RunConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        split_manifest_path=args.split_manifest_path,
        output_dir=args.output_dir,
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

    os.makedirs(cfg.output_dir, exist_ok=True)
    seed_everything(cfg.seed)

    print("=" * 80)
    print("Recipe-MPR fine-tuning run")
    print("=" * 80)
    print(json.dumps(asdict(cfg), indent=2))

    train_ds, val_ds, test_ds = build_hf_datasets(
        data_path=cfg.data_path,
        split_manifest_path=cfg.split_manifest_path,
    )

    print(f"Train examples: {len(train_ds)}")
    print(f"Val examples:   {len(val_ds)}")
    print(f"Test examples:  {len(test_ds)}")
    print("-" * 80)
    print("Sample train row:")
    print(train_ds[0])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Many causal LMs do not define a pad token. Setting pad_token to eos_token
    # is a common practical choice for training/inference with padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    peft_config = None
    if cfg.use_lora:
        target_modules = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
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

    # Save final adapter/model and tokenizer
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save the exact run config for later reporting/reproducibility
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Save trainer log history
    log_history_path = os.path.join(cfg.output_dir, "log_history.json")
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Save a small trainer state summary too
    trainer_state_path = os.path.join(cfg.output_dir, "trainer_state_summary.json")
    with open(trainer_state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "log_history": trainer.state.log_history,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Saved log history to: {log_history_path}")
    print(f"Saved trainer state summary to: {trainer_state_path}")

    print("=" * 80)
    print(f"Training complete. Final artifacts saved to: {final_dir}")
    print("=" * 80)





if __name__ == "__main__":
    main()