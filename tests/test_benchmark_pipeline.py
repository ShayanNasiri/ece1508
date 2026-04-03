from __future__ import annotations

import json
from pathlib import Path

from recipe_mpr_qa.benchmark import (
    build_benchmark_contract,
    build_benchmark_table_rows,
    build_run_manifest,
    read_benchmark_registry,
    register_benchmark_run,
    render_sbatch_script,
    write_benchmark_table,
    write_run_manifest,
)
from recipe_mpr_qa.benchmark.slurm import SlurmJobSpec


def test_benchmark_manifest_registry_and_report_round_trip(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "prediction_metrics": {
                    "accuracy": 0.72,
                    "correct_count": 54,
                    "example_count": 75,
                    "parse_failure_count": 1,
                    "parse_failure_rate": 1 / 75,
                    "accuracy_ci95_low": 0.61,
                    "accuracy_ci95_high": 0.81,
                }
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n", encoding="utf-8")
    contract = build_benchmark_contract(
        dataset_path=dataset_path,
        split_manifest_path=split_path,
        prompt_version="recipe-mpr-mc-v2",
        parser_version="recipe-mpr-mc-parser-v2",
        option_shuffle_seed=1508,
        option_shuffle_strategy="deterministic_per_example",
        split_name="test",
    )
    manifest = build_run_manifest(
        run_id="run-001",
        component="llm",
        status="completed",
        contract=contract,
        dataset_path=dataset_path,
        split_manifest_path=split_path,
        config_payload={"model": "test"},
        model={
            "name": "demo-model",
            "provider": "ollama",
            "interface": "generative",
            "decoding_mode": "generate",
        },
        artifact_paths={"summary": summary_path.as_posix()},
        metrics={"accuracy": 0.72},
    )
    manifest_path = tmp_path / "benchmark_manifest.json"
    registry_path = tmp_path / "registry.jsonl"

    write_run_manifest(manifest, manifest_path)
    register_benchmark_run(manifest, registry_path=registry_path)
    rows = build_benchmark_table_rows((manifest,))
    output_path = tmp_path / "table.json"
    write_benchmark_table(rows, output_path=output_path)

    assert manifest_path.exists()
    assert len(read_benchmark_registry(registry_path)) == 1
    assert rows[0]["accuracy"] == 0.72
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["run_id"] == "run-001"


def test_render_sbatch_script_includes_core_fields() -> None:
    script = render_sbatch_script(
        SlurmJobSpec(
            job_name="demo",
            command=("python", "train.py"),
            workdir="/tmp/project",
            output_path="/tmp/logs/demo.out",
            error_path="/tmp/logs/demo.err",
        )
    )

    assert "#SBATCH --job-name=demo" in script
    assert "cd /tmp/project" in script
    assert "python train.py" in script
