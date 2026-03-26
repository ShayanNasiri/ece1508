import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from ollama_client import OllamaClient
from hf_client import HFClient
from recipe_mpr_qa.formats import build_multiple_choice_prompt, parse_multiple_choice_response
from recipe_mpr_qa.data.constants import QUERY_TYPE_NAMES
from recipe_mpr_qa.data.loaders import load_dataset, load_split_manifest, get_split_examples
from utils import compute_accuracy


def _model_display_name(model: str) -> str:
    """Return a human-readable, filename-safe model identifier.

    For local PEFT adapter paths (contain adapter_config.json), reads the base
    model name from the config and appends '_finetuned'.
    For HF Hub IDs, returns the ID with unsafe characters replaced.
    """
    adapter_cfg = Path(model) / "adapter_config.json"
    if adapter_cfg.is_file():
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base = cfg.get("base_model_name_or_path", model)
        short = Path(base).name if "/" not in base else base.split("/")[-1]
        return f"{short}_finetuned"
    return model.replace(":", "_").replace("/", "_")


def build_result_row(*, index, example, parsed_letter, letter_to_id, raw_response):
    """Build a single result row dict for the output JSON."""
    predicted_id = letter_to_id.get(parsed_letter) if parsed_letter else None
    id_to_letter = {v: k for k, v in letter_to_id.items()}
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

    # Include option texts for wrong/failed answers to support dashboard inspection
    if not is_correct:
        row["options"] = {
            letter: next(opt.text for opt in example.options if opt.option_id == oid)
            for letter, oid in letter_to_id.items()
        }

    return row


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Multiple-choice evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name: Ollama model (e.g. deepseek-r1:7b) or HF model ID (e.g. HuggingFaceTB/SmolLM2-135M-Instruct)")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "huggingface"],
                        help="Inference backend: 'ollama' (default, local server) or 'huggingface' (transformers, for cluster)")
    parser.add_argument("--data", type=str, default="../data/processed/recipe_mpr_qa.jsonl",
                        help="Path to prepared dataset JSONL")
    parser.add_argument("--split-manifest", type=str, default="../data/processed/primary_split.json",
                        help="Path to split manifest JSON")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/<Model>_<Split>_<N>.json)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of questions to evaluate (default: all)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Load config for defaults
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}

    temperature = config.get("temperature", 0)

    if args.backend == "huggingface":
        client = HFClient()
    else:
        ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")
        client = OllamaClient(base_url=ollama_url)

    # Load prepared dataset and split
    dataset = load_dataset(args.data)
    manifest = load_split_manifest(args.split_manifest)
    examples = get_split_examples(dataset, manifest, args.split)

    if args.limit is not None:
        examples = examples[:args.limit]

    # Auto-generate output path if not provided: results/<Model>_<Split>_<N>.json
    if args.output is None:
        safe_model = _model_display_name(args.model)
        args.output = f"results/{safe_model}_{args.split}_{len(examples)}.json"

    predictions = []
    ground_truth = []
    result_rows = []

    print(f"Model:    {args.model}")
    print(f"Backend:  {args.backend}")
    print(f"Split:    {args.split} ({len(examples)} examples)")
    print(f"Temperature: {temperature}")
    print("-" * 60)

    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        prompt, letter_to_id = build_multiple_choice_prompt(
            query=example.query,
            options=example.options,
            # Match the fine-tuning prompt order and avoid answer-position leakage.
            shuffle_key=example.example_id,
        )

        try:
            raw_response = client.query(args.model, prompt, temperature=temperature)
        except RuntimeError as e:
            print(f"\n  Query {i} failed: {e}")
            raw_response = ""

        parsed_letter = parse_multiple_choice_response(raw_response)
        if parsed_letter is None:
            options_map = {
                letter: next(opt.text for opt in example.options if opt.option_id == oid)
                for letter, oid in letter_to_id.items()
            }
            parsed_letter = parse_multiple_choice_response(raw_response, options=options_map)
        predicted_id = letter_to_id.get(parsed_letter) if parsed_letter else None

        predictions.append(predicted_id)
        ground_truth.append(example.answer_option_id)

        result_rows.append(build_result_row(
            index=i, example=example, parsed_letter=parsed_letter,
            letter_to_id=letter_to_id, raw_response=raw_response,
        ))

    # Build dataset-like dicts for compute_accuracy (expects query_type field)
    dataset_for_metrics = [
        {"query_type": dict(ex.query_type_flags)} for ex in examples
    ]
    metrics = compute_accuracy(predictions, ground_truth, dataset_for_metrics)

    print("\n" + "=" * 60)
    print(f"RESULTS — {args.model} ({args.split} split)")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['overall']:.4f} ({metrics['total_correct']}/{metrics['total']})")
    print(f"Parse Failures:   {metrics['parse_failures']}")
    print("-" * 40)

    for qt in QUERY_TYPE_NAMES:
        info = metrics[qt]
        print(f"  {qt:15s}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

    # Save combined results + metrics as JSON
    output_data = {
        "model": args.model,
        "split": args.split,
        "metrics": metrics,
        "results": result_rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
