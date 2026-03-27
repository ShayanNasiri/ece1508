from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
from recipe_mpr_qa.data.loaders import get_split_examples, load_dataset, load_split_manifest
from recipe_mpr_qa.formats import (
    DEFAULT_PROMPT_SPEC,
    OPTION_ORDER_SHUFFLE_SEED,
    build_multiple_choice_prompt,
    parse_multiple_choice_response,
)
from recipe_mpr_qa.llm.hf_client import HFClient
from recipe_mpr_qa.llm.ollama_client import OllamaClient

from recipe_mpr_qa.evaluation.utils import compute_accuracy


def _model_display_name(model: str) -> str:
    """Return a human-readable, filename-safe model identifier."""
    adapter_cfg = Path(model) / "adapter_config.json"
    if adapter_cfg.is_file():
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base = cfg.get("base_model_name_or_path", model)
        short = Path(base).name if "/" not in base else base.split("/")[-1]
        return f"{short}_finetuned"
    return model.replace(":", "_").replace("/", "_")


def build_result_row(*, index, example, parsed_letter, letter_to_id, raw_response):
    predicted_id = letter_to_id.get(parsed_letter) if parsed_letter else None
    id_to_letter = {value: key for key, value in letter_to_id.items()}
    correct_letter = id_to_letter.get(example.answer_option_id, "?")
    is_correct = predicted_id == example.answer_option_id
    parse_failure = parsed_letter is None

    row = {
        "index": index,
        "example_id": example.example_id,
        "query": example.query,
        "correct_answer_id": example.answer_option_id,
        "correct_letter": correct_letter,
        "predicted_letter": parsed_letter,
        "predicted_id": predicted_id,
        "is_correct": is_correct,
        "parse_failure": parse_failure,
        "raw_response": raw_response.replace("\n", " ").strip(),
    }

    if not is_correct:
        row["options"] = {
            letter: next(opt.text for opt in example.options if opt.option_id == option_id)
            for letter, option_id in letter_to_id.items()
        }

    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multiple-choice evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model name: Ollama model (e.g. deepseek-r1:7b) or HF model ID "
            "(e.g. HuggingFaceTB/SmolLM2-135M-Instruct)"
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "huggingface"],
        help="Inference backend: 'ollama' (default) or 'huggingface'",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../data/processed/recipe_mpr_qa.jsonl",
        help="Path to prepared dataset JSONL",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default="../data/processed/primary_split.json",
        help="Path to split manifest JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/<Model>_<Split>_<N>.json)",
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="generative",
        choices=["generative", "loglikelihood"],
        help="Evaluation mode: 'generative' (default) or 'loglikelihood' (scores A-E by logits)",
    )
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    eval_mode = getattr(args, "eval_mode", "generative")

    if eval_mode == "loglikelihood" and args.backend != "huggingface":
        raise ValueError("loglikelihood eval mode requires the huggingface backend")

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = {}

    temperature = config.get("temperature", 0)

    if args.backend == "huggingface":
        client = HFClient()
    else:
        ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")
        client = OllamaClient(base_url=ollama_url)

    dataset = load_dataset(args.data)
    manifest = load_split_manifest(args.split_manifest)
    examples = get_split_examples(dataset, manifest, args.split)

    if args.limit is not None:
        examples = examples[: args.limit]

    if args.output is None:
        safe_model = _model_display_name(args.model)
        args.output = f"results/{safe_model}_{args.split}_{len(examples)}.json"

    predictions = []
    ground_truth = []
    result_rows = []

    print(f"Model:    {args.model}")
    print(f"Backend:  {args.backend}")
    print(f"Eval mode: {eval_mode}")
    print(f"Split:    {args.split} ({len(examples)} examples)")
    print(f"Temperature: {temperature}")
    print("-" * 60)

    for index, example in enumerate(tqdm(examples, desc="Evaluating")):
        prompt, letter_to_id = build_multiple_choice_prompt(
            query=example.query,
            options=example.options,
            shuffle_key=example.example_id,
        )

        if eval_mode == "loglikelihood":
            letters = list(letter_to_id.keys())
            try:
                parsed_letter = client.query_loglikelihood(args.model, prompt, letters)
            except RuntimeError as exc:
                print(f"\n  Query {index} failed: {exc}")
                parsed_letter = None
            raw_response = f"[loglikelihood] selected {parsed_letter}"
        else:
            try:
                raw_response = client.query(args.model, prompt, temperature=temperature)
            except RuntimeError as exc:
                print(f"\n  Query {index} failed: {exc}")
                raw_response = ""

            parsed_letter = parse_multiple_choice_response(raw_response)
            if parsed_letter is None:
                options_map = {
                    letter: next(opt.text for opt in example.options if opt.option_id == option_id)
                    for letter, option_id in letter_to_id.items()
                }
                parsed_letter = parse_multiple_choice_response(raw_response, options=options_map)

        predicted_id = letter_to_id.get(parsed_letter) if parsed_letter else None

        predictions.append(predicted_id)
        ground_truth.append(example.answer_option_id)
        result_rows.append(
            build_result_row(
                index=index,
                example=example,
                parsed_letter=parsed_letter,
                letter_to_id=letter_to_id,
                raw_response=raw_response,
            )
        )

    dataset_for_metrics = [{"query_type": dict(example.query_type_flags)} for example in examples]
    metrics = compute_accuracy(predictions, ground_truth, dataset_for_metrics)

    print("\n" + "=" * 60)
    print(f"RESULTS - {args.model} ({args.split} split)")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['overall']:.4f} ({metrics['total_correct']}/{metrics['total']})")
    print(f"Parse Failures:   {metrics['parse_failures']}")
    print("-" * 40)

    for query_type in QUERY_TYPE_NAMES:
        info = metrics[query_type]
        print(f"  {query_type:15s}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

    output_data = {
        "model": args.model,
        "split": args.split,
        "eval_mode": eval_mode,
        "metrics": metrics,
        "results": result_rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    return {
        "model": args.model,
        "backend": args.backend,
        "split": args.split,
        "temperature": temperature,
        "example_count": len(examples),
        "output_path": args.output,
        "metrics": metrics,
        "results": output_data,
        "prompt": {
            "version": DEFAULT_PROMPT_SPEC.version,
            "option_order": "deterministic_per_example_shuffle",
            "shuffle_seed": OPTION_ORDER_SHUFFLE_SEED,
        },
    }


def run_evaluation_from_arg_list(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run_evaluation(build_arg_parser().parse_args(argv))


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run_evaluation_from_arg_list(argv)


if __name__ == "__main__":
    main()
