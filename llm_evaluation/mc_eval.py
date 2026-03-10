import argparse
import json
import os

from tqdm import tqdm

from ollama_client import OllamaClient
from prompts import build_mc_prompt, parse_mc_response
from utils import load_dataset, compute_accuracy, save_results


def main():
    parser = argparse.ArgumentParser(description="Multiple-choice evaluation via Ollama")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name (e.g. deepseek-r1:7b)")
    parser.add_argument("--data", type=str, required=True, help="Path to 500QA.json")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path for results")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    args = parser.parse_args()

    # Load config for defaults
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}

    ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")
    temperature = config.get("temperature", 0)

    client = OllamaClient(base_url=ollama_url)
    dataset = load_dataset(args.data)

    predictions = []
    ground_truth = []
    result_rows = []

    print(f"Model: {args.model}")
    print(f"Dataset: {len(dataset)} queries")
    print(f"Ollama URL: {ollama_url}")
    print(f"Temperature: {temperature}")
    print("-" * 60)

    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        query = item["query"]
        options = item["options"]
        answer_id = item["answer"]

        prompt, letter_to_id = build_mc_prompt(query, options)

        try:
            raw_response = client.query(args.model, prompt, temperature=temperature)
        except RuntimeError as e:
            print(f"\n  Query {i} failed: {e}")
            raw_response = ""

        parsed_letter = parse_mc_response(raw_response)
        predicted_id = letter_to_id.get(parsed_letter) if parsed_letter else None

        predictions.append(predicted_id)
        ground_truth.append(answer_id)

        # Find the correct letter for logging
        id_to_letter = {v: k for k, v in letter_to_id.items()}
        correct_letter = id_to_letter.get(answer_id, "?")

        result_rows.append({
            "index": i,
            "query": query,
            "correct_answer_id": answer_id,
            "correct_letter": correct_letter,
            "predicted_letter": parsed_letter,
            "predicted_id": predicted_id,
            "is_correct": predicted_id == answer_id,
            "raw_response": raw_response.replace("\n", " ").strip(),
        })

    # Compute and display metrics
    metrics = compute_accuracy(predictions, ground_truth, dataset)

    print("\n" + "=" * 60)
    print(f"RESULTS — {args.model}")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['overall']:.4f} ({metrics['total_correct']}/{metrics['total']})")
    print(f"Parse Failures:   {metrics['parse_failures']}")
    print("-" * 40)

    for qt in ["Specific", "Commonsense", "Negated", "Analogical", "Temporal"]:
        info = metrics[qt]
        print(f"  {qt:15s}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results(result_rows, args.output)
    print(f"\nResults saved to {args.output}")

    # Also save metrics summary as JSON alongside the CSV
    metrics_path = args.output.replace(".csv", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, **metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
