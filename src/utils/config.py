from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs():
    return {
        "product": load_yaml(CONFIG_DIR / "product_config.yaml"),
        "anomaly": load_yaml(CONFIG_DIR / "anomaly_config.yaml"),
        "model": load_yaml(CONFIG_DIR / "model_config.yaml"),
        "rules": load_yaml(CONFIG_DIR / "rules_config.yaml"),
    }
