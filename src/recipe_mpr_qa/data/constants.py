from pathlib import Path

QUERY_TYPE_NAMES = (
    "Specific",
    "Commonsense",
    "Negated",
    "Analogical",
    "Temporal",
)

DEFAULT_RAW_DATASET_PATH = Path("data/500QA.json")
DEFAULT_PROCESSED_DATASET_PATH = Path("data/processed/recipe_mpr_qa.jsonl")
DEFAULT_SPLIT_MANIFEST_PATH = Path("data/processed/primary_split.json")
DEFAULT_SPLIT_SEED = 1508
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
