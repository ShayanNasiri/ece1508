import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


# ----------------------------
# Extract losses from log history
# ----------------------------
def extract_losses(log_history, use_epoch=False):
    train_x, train_losses = [], []
    val_x, val_losses = [], []

    step = 0

    for log in log_history:
        # Training loss
        if "loss" in log:
            x_val = log["epoch"] if use_epoch and "epoch" in log else step
            train_x.append(x_val)
            train_losses.append(log["loss"])
            step += 1

        # Validation loss
        if "eval_loss" in log:
            x_val = log["epoch"] if use_epoch and "epoch" in log else step
            val_x.append(x_val)
            val_losses.append(log["eval_loss"])

    return train_x, train_losses, val_x, val_losses


# ----------------------------
# Load trainer_state.json
# ----------------------------
def load_log_history(run_dir):
    run_dir = Path(run_dir)

    # 1) custom saved log history
    log_history_path = run_dir / "log_history.json"
    if log_history_path.exists():
        with open(log_history_path, "r") as f:
            return json.load(f)

    # 2) trainer state summary
    trainer_state_summary = run_dir / "trainer_state_summary.json"
    if trainer_state_summary.exists():
        with open(trainer_state_summary, "r") as f:
            state = json.load(f)
        return state["log_history"]

    # 3) standard trainer_state.json directly
    direct_state = run_dir / "trainer_state.json"
    if direct_state.exists():
        with open(direct_state, "r") as f:
            state = json.load(f)
        return state["log_history"]

    # 4) search checkpoint-* subfolders
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1
    )

    for ckpt in reversed(checkpoints):
        state_path = ckpt / "trainer_state.json"
        if state_path.exists():
            print(f"Using trainer state from {state_path}")
            with open(state_path, "r") as f:
                state = json.load(f)
            return state["log_history"]

    raise FileNotFoundError(
        f"Could not find log_history.json, trainer_state_summary.json, "
        f"trainer_state.json, or any checkpoint trainer_state.json in {run_dir}"
    )


# ----------------------------
# Plot a single run
# ----------------------------
def plot_single(run_dir, use_epoch=False, save_path=None):
    log_history = load_log_history(run_dir)

    train_x, train_losses, val_x, val_losses = extract_losses(
        log_history, use_epoch=use_epoch
    )

    plt.figure()

    plt.plot(train_x, train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_x, val_losses, label="Validation Loss")

    xlabel = "Epoch" if use_epoch else "Step"
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(f"Loss Curve: {run_dir}")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


# ----------------------------
# Plot multiple runs (comparison)
# ----------------------------
def plot_multiple(run_dirs, labels=None, use_epoch=False, save_path=None):
    plt.figure()

    for i, run_dir in enumerate(run_dirs):
        log_history = load_log_history(run_dir)
        train_x, train_losses, val_x, val_losses = extract_losses(
            log_history, use_epoch=use_epoch
        )

        label_prefix = labels[i] if labels else Path(run_dir).name

        plt.plot(train_x, train_losses, linestyle="--", label=f"{label_prefix} (train)")

        if val_losses:
            plt.plot(val_x, val_losses, label=f"{label_prefix} (val)")

    xlabel = "Epoch" if use_epoch else "Step"
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Comparison")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot training/validation loss curves")

    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to run directories (containing trainer_state.json)",
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels for each run",
    )

    parser.add_argument(
        "--use-epoch",
        action="store_true",
        help="Plot against epoch instead of step",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save figure (optional)",
    )

    args = parser.parse_args()

    if len(args.runs) == 1:
        plot_single(
            args.runs[0],
            use_epoch=args.use_epoch,
            save_path=args.save,
        )
    else:
        plot_multiple(
            args.runs,
            labels=args.labels,
            use_epoch=args.use_epoch,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()