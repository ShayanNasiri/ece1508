from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.benchmark.provenance import BenchmarkRunManifest

DEFAULT_BENCHMARK_REGISTRY_PATH = Path("artifacts/benchmark/registry.jsonl")


def read_benchmark_registry(
    registry_path: Path | str = DEFAULT_BENCHMARK_REGISTRY_PATH,
) -> tuple[BenchmarkRunManifest, ...]:
    resolved_path = Path(registry_path)
    if not resolved_path.exists():
        return ()
    lines = [line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(BenchmarkRunManifest.from_dict(json.loads(line)) for line in lines)


def register_benchmark_run(
    manifest: BenchmarkRunManifest,
    *,
    registry_path: Path | str = DEFAULT_BENCHMARK_REGISTRY_PATH,
) -> tuple[BenchmarkRunManifest, ...]:
    resolved_path = Path(registry_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {item.run_id: item for item in read_benchmark_registry(resolved_path)}
    existing[manifest.run_id] = manifest
    ordered = tuple(existing[key] for key in sorted(existing))
    resolved_path.write_text(
        "\n".join(
            json.dumps(item.to_dict(), ensure_ascii=True, separators=(",", ":")) for item in ordered
        )
        + "\n",
        encoding="utf-8",
    )
    return ordered


def summarize_registry_statuses(
    manifests: Sequence[BenchmarkRunManifest],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for manifest in manifests:
        counts[manifest.status] = counts.get(manifest.status, 0) + 1
    return dict(sorted(counts.items()))
