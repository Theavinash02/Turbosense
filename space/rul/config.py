import os
from pathlib import Path

import yaml


def default_config_path() -> Path:
    """Return $RUL_CONFIG if set, else configs/default.yaml in the working directory, else the repo's copy."""
    if os.environ.get("RUL_CONFIG"):
        return Path(os.environ["RUL_CONFIG"])
    local = Path("configs") / "default.yaml"
    return local if local.exists() else Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Return the settings from a YAML file, with every path turned into a Path."""
    cfg = yaml.safe_load(Path(path or default_config_path()).read_text())
    cfg["paths"] = {k: Path(v) for k, v in cfg["paths"].items()}
    return cfg
