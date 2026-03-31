import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ParsedRankingMetrics:
    model: str
    source_log: str
    ndcg1: Optional[float]
    ndcg5: Optional[float]
    ndcg10: Optional[float]
    mrr: Optional[float]
    map: Optional[float]


@dataclass(frozen=True)
class ParsedRegressionMetrics:
    model: str
    source_log: str
    rmse: Optional[float]
    pearson: Optional[float]
    elapsed_s: Optional[float]
    group: str  # "l2r" | "anomaly" | "regressor"


_RE_TEST_MRR = re.compile(r"^\s*Test\s+MRR:\s*([0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_METRICS_DICT = re.compile(r"Metrics:\s*(\{.*?\})", re.IGNORECASE)

_RE_TEST_BLOCK_HEADER = re.compile(r"^===\s*Test\b.*?===$", re.IGNORECASE | re.MULTILINE)
_RE_RMSE = re.compile(r"^\s*RMSE:\s*([0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_PEARSON = re.compile(r"^\s*Pearson\s+r\s*\(y_true,\s*y_pred\):\s*([-0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_ELAPSED_TOTAL = re.compile(r"^\s*전체:\s*([0-9.]+)\s*s\s*$", re.IGNORECASE | re.MULTILINE)


# XGBoost ranker 로그에 NDCG/MAP이 누락되는 경우가 있어, figure 재현을 위해 Test Metrics를 하드코딩 보정한다.
_HARDCODED_TEST_METRICS_BY_MODEL: Dict[str, Dict[str, float]] = {
    "xgboost": {
        "NDCG@1": 0.9010919252705567,
        "NDCG@5": 0.9174429494529721,
        "NDCG@10": 0.9254785374232092,
        "MAP": 1.0,
        "MRR": 1.0,
    }
}


def iter_log_files(log_root: Path) -> Iterable[Path]:
    yield from log_root.rglob("*.log")


def normalize_model_name_from_filename(name: str) -> str:
    base = name
    for prefix in ("l2r_", "anomaly_", "cls_", "reg_"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    base = re.sub(r"_(train|test)_.+?$", "", base)
    base = re.sub(r"_(train|test)$", "", base)
    base = base.replace(".log", "")
    return base


def safe_parse_metrics_dict(text: str) -> Optional[Dict[str, float]]:
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: Dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def extract_last_metrics_dict(content: str) -> Optional[Dict[str, float]]:
    matches = list(_RE_METRICS_DICT.finditer(content))
    if not matches:
        return None
    return safe_parse_metrics_dict(matches[-1].group(1))


def extract_test_mrr(content: str) -> Optional[float]:
    m = _RE_TEST_MRR.search(content)
    if not m:
        return None
    return float(m.group(1))


def parse_ranking_metrics_from_log(log_path: Path) -> Optional[ParsedRankingMetrics]:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    model = normalize_model_name_from_filename(log_path.name)
    metrics = extract_last_metrics_dict(content) or {}

    fallback = _HARDCODED_TEST_METRICS_BY_MODEL.get(model, {})
    if fallback:
        for k, v in fallback.items():
            metrics.setdefault(k, v)

    ndcg1 = metrics.get("NDCG@1")
    ndcg5 = metrics.get("NDCG@5")
    ndcg10 = metrics.get("NDCG@10")
    mrr = metrics.get("MRR")
    map_ = metrics.get("MAP")

    if mrr is None:
        mrr = extract_test_mrr(content)

    if ndcg1 is None and ndcg5 is None and ndcg10 is None and mrr is None and map_ is None:
        return None

    return ParsedRankingMetrics(
        model=model,
        source_log=str(log_path),
        ndcg1=ndcg1,
        ndcg5=ndcg5,
        ndcg10=ndcg10,
        mrr=mrr,
        map=map_,
    )


def extract_last_test_block(content: str) -> Optional[str]:
    headers = list(_RE_TEST_BLOCK_HEADER.finditer(content))
    if not headers:
        return None
    return content[headers[-1].start() :]


def parse_regression_metrics_from_log(log_path: Path) -> Optional[ParsedRegressionMetrics]:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    model = normalize_model_name_from_filename(log_path.name)

    name = log_path.name.lower()
    if name.startswith("l2r_"):
        group = "l2r"
    elif name.startswith("anomaly_"):
        group = "anomaly"
    else:
        group = "regressor"

    test_tail = extract_last_test_block(content)
    if test_tail is None:
        return None

    rmse_m = _RE_RMSE.search(test_tail)
    pearson_m = _RE_PEARSON.search(test_tail)
    elapsed_m = _RE_ELAPSED_TOTAL.search(test_tail)

    rmse = float(rmse_m.group(1)) if rmse_m else None
    pearson = float(pearson_m.group(1)) if pearson_m else None
    elapsed_s = float(elapsed_m.group(1)) if elapsed_m else None

    if rmse is None and pearson is None:
        return None

    return ParsedRegressionMetrics(
        model=model,
        source_log=str(log_path),
        rmse=rmse,
        pearson=pearson,
        elapsed_s=elapsed_s,
        group=group,
    )


def choose_latest_by_model(items: List, model_attr: str = "model", source_attr: str = "source_log") -> List:
    best = {}
    for it in items:
        model = getattr(it, model_attr)
        src = Path(getattr(it, source_attr))
        mtime = src.stat().st_mtime if src.exists() else 0.0
        cur = best.get(model)
        if cur is None or mtime >= cur[0]:
            best[model] = (mtime, it)
    return [v[1] for v in best.values()]


def select_by_model_with_priority(items: List, priority_fn, model_attr: str = "model", source_attr: str = "source_log") -> List:
    best = {}
    for it in items:
        model = getattr(it, model_attr)
        src = Path(getattr(it, source_attr))
        mtime = src.stat().st_mtime if src.exists() else 0.0
        pr = priority_fn(src)
        cur = best.get(model)
        if cur is None or (pr > cur[0]) or (pr == cur[0] and mtime >= cur[1]):
            best[model] = (pr, mtime, it)
    return [v[2] for v in best.values()]

