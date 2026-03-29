from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"

ALLOWED_TEST_MODES = frozenset({"train_thresholds", "test_oracle_ratio"})
ALLOWED_QID_MODES = frozenset({"global", "anomaly_id", "timestamp_hour_1h"})
ALLOWED_SPLIT_MODES = frozenset({"stratified_shuffle", "time_ordered"})


def resolve_test_mode(cfg: dict[str, Any], cli_override: str | None) -> str:
    """CLI --test-mode가 있으면 그것을 쓰고, 없으면 cfg['evaluation']['test_mode']."""
    if cli_override is not None:
        if cli_override not in ALLOWED_TEST_MODES:
            raise ValueError(f"test_mode는 {sorted(ALLOWED_TEST_MODES)} 중 하나여야 합니다: {cli_override!r}")
        return cli_override
    tm = cfg["evaluation"]["test_mode"]
    if tm not in ALLOWED_TEST_MODES:
        raise ValueError(
            f"experiment_config.yaml evaluation.test_mode는 {sorted(ALLOWED_TEST_MODES)} 중 하나여야 합니다: {tm!r}"
        )
    return tm


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

    _apply_config_defaults(cfg)
    _validate(cfg)
    return cfg


def _apply_config_defaults(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg.get("features"), dict):
        cfg["features"] = {}
    cfg["features"].setdefault("include_categorical_columns", True)
    ev = cfg.get("evaluation")
    if isinstance(ev, dict):
        ev.setdefault("ordinal_severity_metrics", False)
    rk = cfg.get("ranking")
    if isinstance(rk, dict):
        rk.setdefault("qid_mode", "global")
        rk.setdefault("global_qid", 0)
    sp = cfg.get("split")
    if isinstance(sp, dict):
        sp.setdefault("mode", "stratified_shuffle")


def _validate(cfg: dict[str, Any]) -> None:
    for key in ("data", "split", "ranking", "epochs", "evaluation"):
        if key not in cfg:
            raise KeyError(f"experiment_config.yaml에 '{key}' 섹션이 필요합니다.")
    for k in ("test_size", "val_size", "random_state"):
        if k not in cfg["split"]:
            raise KeyError(f"split.{k}")
    sm = cfg["split"].get("mode", "stratified_shuffle")
    if sm not in ALLOWED_SPLIT_MODES:
        raise ValueError(f"split.mode는 {sorted(ALLOWED_SPLIT_MODES)} 중 하나여야 합니다: {sm!r}")
    if "group_size" not in cfg["ranking"]:
        raise KeyError("ranking.group_size")
    qm = cfg["ranking"].get("qid_mode", "global")
    if qm not in ALLOWED_QID_MODES:
        raise ValueError(f"ranking.qid_mode는 {sorted(ALLOWED_QID_MODES)} 중 하나여야 합니다: {qm!r}")
    gq = cfg["ranking"].get("global_qid", 0)
    if not isinstance(gq, int):
        raise TypeError(f"ranking.global_qid는 정수여야 합니다: {type(gq).__name__}")
    for k in ("l2r", "anomaly"):
        if k not in cfg["epochs"]:
            raise KeyError(f"epochs.{k}")
    if "csv" not in cfg["data"]:
        raise KeyError("data.csv")
    if "test_mode" not in cfg["evaluation"]:
        raise KeyError("evaluation.test_mode")
    if not isinstance(cfg["features"], dict):
        raise TypeError("features 섹션은 mapping 이어야 합니다.")
    if not isinstance(cfg["features"]["include_categorical_columns"], bool):
        raise TypeError(
            f"features.include_categorical_columns는 bool 이어야 합니다: {type(cfg['features']['include_categorical_columns']).__name__}"
        )
    if not isinstance(cfg["evaluation"]["ordinal_severity_metrics"], bool):
        raise TypeError(
            f"evaluation.ordinal_severity_metrics는 bool 이어야 합니다: {type(cfg['evaluation']['ordinal_severity_metrics']).__name__}"
        )
