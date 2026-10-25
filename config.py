from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
DATASETS_DIR = DATASETS_ROOT / "dataset_new"
DATA_DIR = DATASETS_DIR / "sec"
EVALS_DIR = PROJECT_ROOT / "evals"
BENCHMARKS_JSON = PROJECT_ROOT / "benchmarks.json"
CAPABILITIES_PATH = PROJECT_ROOT / "metadata" / "capabilities.json"
CASE_TAXONOMY_PATH = PROJECT_ROOT / "metadata" / "case_taxonomy.json"
MODELS_TOML = PROJECT_ROOT / "models.toml"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
