from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("YAML 설정을 쓰려면 PyYAML이 필요합니다: pip install pyyaml") from e

    p = Path(path) if path else DEFAULT_CONFIG_PATH
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"설정 파일이 없습니다: {p}")

    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("experiment_config.yaml 최상위는 mapping 이어야 합니다.")

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    for key in ("data", "split", "ranking", "epochs"):
        if key not in cfg:
            raise KeyError(f"experiment_config.yaml에 '{key}' 섹션이 필요합니다.")
    for k in ("test_size", "val_size", "random_state"):
        if k not in cfg["split"]:
            raise KeyError(f"split.{k}")
    if "group_size" not in cfg["ranking"]:
        raise KeyError("ranking.group_size")
    for k in ("l2r", "anomaly"):
        if k not in cfg["epochs"]:
            raise KeyError(f"epochs.{k}")
    if "csv" not in cfg["data"]:
        raise KeyError("data.csv")
