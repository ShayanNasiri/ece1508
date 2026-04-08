"""Generate a bar chart of all evaluation results from JSON files."""

import json
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "results_bar_chart.png")

# Map JSON filenames to clean display names and category
FILE_CONFIG = {
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_test_75.json": {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "category": "zero-shot",
        "params": "7B",
    },
    "Qwen_Qwen2.5-3B-Instruct_test_75.json": {
        "name": "Qwen2.5-3B-Instruct",
        "category": "zero-shot",
        "params": "3B",
    },
    "SmolLM2-1.7B-Instruct_finetuned_synthetic_test_75.json": {
        "name": "SmolLM2-1.7B-Instruct",
        "category": "finetuned + synthetic",
        "params": "1.7B",
    },
    "SmolLM2-1.7B-Instruct_finetuned_test_75.json": {
        "name": "SmolLM2-1.7B-Instruct",
        "category": "finetuned",
        "params": "1.7B",
    },
    "HuggingFaceTB_SmolLM2-1.7B-Instruct_test_75.json": {
        "name": "SmolLM2-1.7B-Instruct",
        "category": "zero-shot",
        "params": "1.7B",
    },
    "HuggingFaceTB_SmolLM2-360M-Instruct_test_75.json": {
        "name": "SmolLM2-360M-Instruct",
        "category": "zero-shot",
        "params": "360M",
    },
    "SmolLM2-360M-Instruct_finetuned_runH_test_75.json": {
        "name": "SmolLM2-360M-Instruct",
        "category": "finetuned + synthetic",
        "params": "360M",
    },
    "SmolLM2-360M-Instruct_finetuned_test_75.json": {
        "name": "SmolLM2-360M-Instruct",
        "category": "finetuned",
        "params": "360M",
    },
    "SmolLM2-135M-Instruct_finetuned_test_75.json": {
        "name": "SmolLM2-135M-Instruct",
        "category": "finetuned",
        "params": "135M",
    },
    "HuggingFaceTB_SmolLM2-135M-Instruct_test_75.json": {
        "name": "SmolLM2-135M-Instruct",
        "category": "zero-shot",
        "params": "135M",
    },
    "smollm2_base_loglikelihood_test_75.json": {
        "name": "SmolLM2-135M (loglikelihood)",
        "category": "zero-shot",
        "params": "135M",
    },
}

# Exclude these from the chart (catastrophic failures / excluded results)
EXCLUDE = {
    "Qwen2.5-3B-Instruct_finetuned_test_75.json",      # v1 catastrophic overfit (0%)
    "Qwen2.5-3B-Instruct_finetuned_v2_test_75.json",    # v2 still degraded (53.3%)
}

CATEGORY_COLORS = {
    "zero-shot": "#4A90D9",
    "finetuned": "#E8833A",
    "finetuned + synthetic": "#6AB04C",
}

CATEGORY_LABELS = {
    "zero-shot": "Zero-shot",
    "finetuned": "Finetuned (train only)",
    "finetuned + synthetic": "Finetuned (train + synthetic)",
}


# Groups define the display order: best models first (top of chart), worst last (bottom).
# Within each group: zero-shot -> finetuned -> finetuned+synthetic.
MODEL_GROUPS = [
    ("DeepSeek-R1-Distill-Qwen-7B (7B)", [
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_test_75.json",
    ]),
    ("Qwen2.5-3B-Instruct (3B)", [
        "Qwen_Qwen2.5-3B-Instruct_test_75.json",
    ]),
    ("SmolLM2-1.7B-Instruct (1.7B)", [
        "SmolLM2-1.7B-Instruct_finetuned_synthetic_test_75.json",
        "SmolLM2-1.7B-Instruct_finetuned_test_75.json",
        "HuggingFaceTB_SmolLM2-1.7B-Instruct_test_75.json",
    ]),
    ("SmolLM2-360M-Instruct (360M)", [
        "SmolLM2-360M-Instruct_finetuned_runH_test_75.json",
        "SmolLM2-360M-Instruct_finetuned_test_75.json",
        "HuggingFaceTB_SmolLM2-360M-Instruct_test_75.json",
    ]),
    ("SmolLM2-135M-Instruct (135M)", [
        "SmolLM2-135M-Instruct_finetuned_test_75.json",
        "HuggingFaceTB_SmolLM2-135M-Instruct_test_75.json",
    ]),
]


def load_results():
    entries = []
    group_boundaries = []  # (y_start, y_end, group_label)
    idx = 0
    for group_label, filenames in MODEL_GROUPS:
        group_start = idx
        for filename in filenames:
            config = FILE_CONFIG.get(filename)
            if config is None or filename in EXCLUDE:
                continue
            filepath = os.path.join(RESULTS_DIR, filename)
            if not os.path.exists(filepath):
                continue
            with open(filepath) as f:
                data = json.load(f)
            metrics = data["metrics"]
            accuracy = metrics["total_correct"] / metrics["total"] * 100
            entries.append({
                "label": CATEGORY_LABELS[config["category"]],
                "accuracy": accuracy,
                "correct": metrics["total_correct"],
                "total": metrics["total"],
                "category": config["category"],
                "parse_failures": metrics["parse_failures"],
                "group": group_label,
            })
            idx += 1
        if idx > group_start:
            group_boundaries.append((group_start, idx - 1, group_label))
    return entries, group_boundaries


def plot_results(entries, group_boundaries):
    # Add spacing between groups
    gap = 0.7
    y_positions = []
    current_y = 0
    prev_group = None
    for e in entries:
        if prev_group is not None and e["group"] != prev_group:
            current_y += gap
        y_positions.append(current_y)
        prev_group = e["group"]
        current_y += 1

    y_positions = np.array(y_positions)
    fig, ax = plt.subplots(figsize=(12, 7))

    accuracies = [e["accuracy"] for e in entries]
    colors = [CATEGORY_COLORS[e["category"]] for e in entries]

    # Build combined y-axis labels: "ModelGroup — Variant"
    ylabels = []
    for e in entries:
        ylabels.append(f"{e['group']}  —  {e['label']}")

    bars = ax.barh(y_positions, accuracies, color=colors, edgecolor="white", height=0.65)

    # Add accuracy labels on bars
    for bar, entry in zip(bars, entries):
        width = bar.get_width()
        label_text = f"{entry['accuracy']:.1f}% ({entry['correct']}/{entry['total']})"
        if width > 35:
            ax.text(width - 1.5, bar.get_y() + bar.get_height() / 2,
                    label_text, ha="right", va="center", fontsize=8.5,
                    fontweight="bold", color="white")
        else:
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                    label_text, ha="left", va="center", fontsize=8.5,
                    fontweight="bold", color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Recipe-MPR QA — Model Evaluation Results (75-question test set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    # 20% random baseline
    ax.axvline(x=20, color="#999999", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(21, y_positions[-1] + 0.4, "Random baseline (20%)",
            fontsize=8, color="#999999", va="top")

    # Horizontal separator lines between groups
    for i in range(len(group_boundaries) - 1):
        _, end, _ = group_boundaries[i]
        next_start, _, _ = group_boundaries[i + 1]
        sep_y = (y_positions[end] + y_positions[next_start]) / 2
        ax.axhline(y=sep_y, color="#CCCCCC", linestyle="-", linewidth=0.8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["zero-shot"], label="Zero-shot"),
        Patch(facecolor=CATEGORY_COLORS["finetuned"], label="Finetuned (train only)"),
        Patch(facecolor=CATEGORY_COLORS["finetuned + synthetic"],
              label="Finetuned (train + synthetic)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Chart saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    entries, group_boundaries = load_results()
    print(f"Loaded {len(entries)} results:")
    for e in entries:
        print(f"  {e['accuracy']:5.1f}%  {e['group']} — {e['label']}")
    print()
    plot_results(entries, group_boundaries)
