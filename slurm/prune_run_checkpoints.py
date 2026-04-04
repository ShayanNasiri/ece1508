#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune checkpoint directories down to the best checkpoint.")
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.artifacts_root / args.run_id
    manifest_path = run_dir / "slm" / "checkpoint_manifest.json"
    checkpoints_root = run_dir / "slm" / "checkpoints"
    if not manifest_path.is_file() or not checkpoints_root.is_dir():
        return 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    best_checkpoint_dir = Path(payload.get("best_checkpoint_dir", ""))
    keep_dir_name = best_checkpoint_dir.name if best_checkpoint_dir.name.startswith("checkpoint-") else None

    for child in checkpoints_root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-") and child.name != keep_dir_name:
            shutil.rmtree(child)

    retained = []
    for item in payload.get("checkpoints", []):
        checkpoint_dir = item.get("checkpoint_dir")
        if not checkpoint_dir:
            continue
        if Path(checkpoint_dir).exists():
            retained.append(item)
    payload["checkpoints"] = retained
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
