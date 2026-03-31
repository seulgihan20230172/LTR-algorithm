import argparse
import ast
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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
_RE_VALIDATION_MRR = re.compile(r"^\s*Validation\s+MRR:\s*([0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_METRICS_DICT = re.compile(r"Metrics:\s*(\{.*?\})", re.IGNORECASE)
_RE_XGB_NDCG32 = re.compile(r"validation_0-ndcg@32:([0-9.]+)", re.IGNORECASE)

_RE_TEST_BLOCK_HEADER = re.compile(r"^===\s*Test\b.*?===$", re.IGNORECASE | re.MULTILINE)
_RE_RMSE = re.compile(r"^\s*RMSE:\s*([0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_PEARSON = re.compile(r"^\s*Pearson\s+r\s*\(y_true,\s*y_pred\):\s*([-0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_ELAPSED_TOTAL = re.compile(r"^\s*전체:\s*([0-9.]+)\s*s\s*$", re.IGNORECASE | re.MULTILINE)

# XGBoost ranker 로그에 NDCG/MAP이 누락되는 경우가 있어, paper figure 재현을 위해 Test Metrics를 하드코딩 보정한다.
# (사용자가 제공한 값 / test metrics 사용)
_HARDCODED_TEST_METRICS_BY_MODEL: Dict[str, Dict[str, float]] = {
    "xgboost": {
        "NDCG@1": 0.9010919252705567,
        "NDCG@5": 0.9174429494529721,
        "NDCG@10": 0.9254785374232092,
        "MAP": 1.0,
        "MRR": 1.0,
    }
}


def _iter_log_files(log_root: Path) -> Iterable[Path]:
    yield from log_root.rglob("*.log")


def _normalize_model_name_from_filename(name: str) -> str:
    # Example: l2r_ranknet_test_oracle_ratio.log -> ranknet
    #          anomaly_vanilla_ae_train_thresholds.log -> vanilla_ae
    #          reg_xgboost_regressor_train_score_relevance_0_3.log -> xgboost_regressor
    base = name
    for prefix in ("l2r_", "anomaly_", "cls_", "reg_"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    base = re.sub(r"_(train|test)_.+?$", "", base)
    base = re.sub(r"_(train|test)$", "", base)
    base = base.replace(".log", "")
    return base


def _safe_parse_metrics_dict(text: str) -> Optional[Dict[str, float]]:
    # Logs contain python-literal dicts like:
    # {'NDCG@1': 0.41, 'NDCG@10': 0.55, 'MAP': 0.81, 'MRR': 0.96}
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


def _extract_last_metrics_dict(content: str) -> Optional[Dict[str, float]]:
    matches = list(_RE_METRICS_DICT.finditer(content))
    if not matches:
        return None
    last = matches[-1].group(1)
    return _safe_parse_metrics_dict(last)


def _extract_test_mrr(content: str) -> Optional[float]:
    m = _RE_TEST_MRR.search(content)
    if not m:
        return None
    return float(m.group(1))


def _extract_xgb_validation_ndcg32(content: str) -> Optional[float]:
    vals = [float(m.group(1)) for m in _RE_XGB_NDCG32.finditer(content)]
    return vals[-1] if vals else None


def parse_ranking_metrics_from_log(log_path: Path) -> Optional[ParsedRankingMetrics]:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    model = _normalize_model_name_from_filename(log_path.name)

    metrics = _extract_last_metrics_dict(content) or {}

    # Fallback hardcoded metrics (when logs don't print full dict).
    fallback = _HARDCODED_TEST_METRICS_BY_MODEL.get(model, {})
    if fallback:
        for k, v in fallback.items():
            metrics.setdefault(k, v)

    # NDCG@1/5/10
    ndcg1 = metrics.get("NDCG@1")
    ndcg5 = metrics.get("NDCG@5")
    ndcg10 = metrics.get("NDCG@10")

    mrr = metrics.get("MRR")
    map_ = metrics.get("MAP")

    # Fallback: some logs only contain final MRR section.
    if mrr is None:
        mrr = _extract_test_mrr(content)

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


def _extract_last_test_block(content: str) -> Optional[str]:
    headers = list(_RE_TEST_BLOCK_HEADER.finditer(content))
    if not headers:
        return None
    start = headers[-1].start()
    return content[start:]


def parse_regression_metrics_from_log(log_path: Path) -> Optional[ParsedRegressionMetrics]:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    model = _normalize_model_name_from_filename(log_path.name)

    name = log_path.name.lower()
    if name.startswith("l2r_"):
        group = "l2r"
    elif name.startswith("anomaly_"):
        group = "anomaly"
    else:
        # reg_*, cls_* 등은 비교군(regressor/ML baseline)으로 묶음
        group = "regressor"

    test_tail = _extract_last_test_block(content)
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


def _plot_ranking_panels(
    out_path: Path,
    title: str,
    models: List[str],
    ndcg1: List[Optional[float]],
    ndcg5: List[Optional[float]],
    ndcg10: List[Optional[float]],
    mrr: List[Optional[float]],
    map_: List[Optional[float]],
):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    # Korean font (Windows)
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for candidate in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
            if candidate in available:
                plt.rcParams["font.family"] = candidate
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    def _arr(xs: List[Optional[float]]) -> np.ndarray:
        return np.array([np.nan if v is None else float(v) for v in xs], dtype=float)

    x = np.arange(len(models))
    w = 0.6

    fig_w = max(11.5, 0.65 * len(models) + 6.0)
    fig = plt.figure(figsize=(fig_w, 7.0), dpi=160)
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1, 1, 1], wspace=0.28, hspace=0.15)

    ax_ndcg1 = fig.add_subplot(gs[0, 0])
    ax_ndcg5 = fig.add_subplot(gs[1, 0], sharex=ax_ndcg1)
    ax_ndcg10 = fig.add_subplot(gs[2, 0], sharex=ax_ndcg1)

    ax_mrr = fig.add_subplot(gs[:, 1])
    ax_map = fig.add_subplot(gs[:, 2])

    nd1 = _arr(ndcg1)
    nd5 = _arr(ndcg5)
    nd10 = _arr(ndcg10)
    m = _arr(mrr)
    mp = _arr(map_)

    ymax = np.nanmax(np.concatenate([nd1, nd5, nd10])) if np.any(~np.isnan(np.concatenate([nd1, nd5, nd10]))) else np.nan

    for ax, vals, subtitle in [
        (ax_ndcg1, nd1, "NDCG@1"),
        (ax_ndcg5, nd5, "NDCG@5"),
        (ax_ndcg10, nd10, "NDCG@10"),
    ]:
        ax.bar(x, vals, w, color="#4C78A8")
        ax.set_title(subtitle)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if not math.isnan(float(ymax)):
            ax.axhline(float(ymax), color="red", linestyle=":", linewidth=2.0, alpha=0.9)

    ax_ndcg10.set_xticks(x)
    ax_ndcg10.set_xticklabels(models, rotation=30, ha="right")
    for ax in (ax_ndcg1, ax_ndcg5):
        ax.tick_params(axis="x", labelbottom=False)

    ax_mrr.bar(x, m, w, color="#F58518")
    ax_mrr.set_title("MRR")
    ax_mrr.set_xticks(x)
    ax_mrr.set_xticklabels(models, rotation=30, ha="right")
    ax_mrr.grid(axis="y", linestyle="--", alpha=0.35)

    ax_map.bar(x, mp, w, color="#54A24B")
    ax_map.set_title("MAP")
    ax_map.set_xticks(x)
    ax_map.set_xticklabels(models, rotation=30, ha="right")
    ax_map.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_l2r_ranking_compact(
    out_path: Path,
    title: str,
    ranking: List[ParsedRankingMetrics],
):
    """
    모델별로 NDCG(1/5/10) 3개 막대를 '서로 붙여' 그리고,
    그 3개 중 최댓값 높이로 채움 없는 붉은 점선 외곽 막대(박스)를 씌운다.
    같은 모델 안에서 NDCG 블록 + MRR + MAP 막대가 붙어 보이도록 배치한다.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    import numpy as np

    # 스타일/비율은 create_threshold_graph.py에 맞춤
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40
    plt.rcParams["axes.unicode_minus"] = False

    models = [r.model for r in ranking]
    nd1 = np.array([np.nan if r.ndcg1 is None else float(r.ndcg1) for r in ranking], dtype=float)
    nd5 = np.array([np.nan if r.ndcg5 is None else float(r.ndcg5) for r in ranking], dtype=float)
    nd10 = np.array([np.nan if r.ndcg10 is None else float(r.ndcg10) for r in ranking], dtype=float)
    mrr = np.array([np.nan if r.mrr is None else float(r.mrr) for r in ranking], dtype=float)
    map_ = np.array([np.nan if r.map is None else float(r.map) for r in ranking], dtype=float)

    # Layout: per-model block = [NDCG(outer box width=rect_w), MRR, MAP] placed adjacent.
    # Bars within each model are "붙어서" 보이도록 간격을 최소화한다.
    rect_w = 1.5  # overall bar width for NDCG outer box and MRR/MAP bars (막대폭 크게 증가)
    gap = 0.0  # 같은 모델 내 막대는 붙임
    model_span = rect_w * 3 + gap * 2
    # 폰트가 큰 편이라 모델 간 간격을 넉넉히 확보
    block_gap = 2.10
    base = np.arange(len(models)) * (model_span + block_gap)

    ndcg_left = base - model_span / 2.0
    mrr_left = ndcg_left + rect_w + gap
    map_left = mrr_left + rect_w + gap

    nd_w = rect_w / 3.0  # NDCG@1/@5/@10 bars fill the outer box exactly
    nd_pos = [
        ndcg_left + nd_w * 0.5,
        ndcg_left + nd_w * 1.5,
        ndcg_left + nd_w * 2.5,
    ]
    mrr_pos = mrr_left + rect_w * 0.5
    map_pos = map_left + rect_w * 0.5

    # 가로축-글씨 비율 유지: width=16, font=40 고정, 세로만 축소
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    # NDCG@1/5/10 (붙은 막대, outer box 안에 딱 맞춤)
    ax.bar(nd_pos[0], nd1, nd_w, label="NDCG@1", color="#4C78A8")
    ax.bar(nd_pos[1], nd5, nd_w, label="NDCG@5", color="#72B7B2")
    ax.bar(nd_pos[2], nd10, nd_w, label="NDCG@10", color="#A0CBE8")

    # MRR / MAP
    ax.bar(mrr_pos, mrr, rect_w, label="MRR", color="#F58518")
    ax.bar(map_pos, map_, rect_w, label="MAP", color="#54A24B")

    # Red dashed outline bar for max(NDCG@1,@5,@10) per model
    # Avoid RuntimeWarning for models that have all-NaN NDCG values.
    stacked = np.vstack([nd1, nd5, nd10])
    nd_max = np.array(
        [np.nan if np.all(np.isnan(stacked[:, i])) else float(np.nanmax(stacked[:, i])) for i in range(stacked.shape[1])],
        dtype=float,
    )
    for i in range(len(models)):
        if math.isnan(float(nd_max[i])):
            continue
        rect = Rectangle(
            (float(ndcg_left[i]), 0.0),
            float(rect_w),
            float(nd_max[i]),
            fill=False,
            edgecolor="red",
            linewidth=2.0,
            linestyle=":",
            alpha=0.95,
        )
        ax.add_patch(rect)

    # X ticks centered per model block (가로로 표기)
    ax.set_xticks(base)
    ax.set_xticklabels(models, rotation=0, ha="center")

    # 제목 제거(요청사항)
    ax.set_ylabel("Score", fontsize=40)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0.0)

    ax.tick_params(axis="both", labelsize=40)

    # 범례를 아래로, 2줄로(윗줄 NDCG 3개 / 아랫줄 MRR, MAP)
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    # Matplotlib legend는 column-major로 채우므로, 2행×3열에서 아래 모양이 되도록 재배치:
    #   NDCG@1             NDCG@5          NDCG@10
    #                     MRR              MAP
    # 즉 row2 col1은 빈칸.
    from matplotlib.patches import Rectangle as _Rect

    dummy = _Rect((0, 0), 1, 1, fill=False, edgecolor="none", label=" ")
    # column-major fill, 2 rows x 3 cols:
    # col1: row1=item1, row2=item2
    # col2: row1=item3, row2=item4
    # col3: row1=item5, row2=item6
    # 2행×3열( column-major )에서 아래처럼 보이도록:
    #   NDCG@1             NDCG@5          NDCG@10
    #   MRR                MAP
    # (마지막 칸은 빈칸)
    legend_handles = [
        label_to_handle.get("NDCG@1"),
        label_to_handle.get("MRR"),
        label_to_handle.get("NDCG@5"),
        label_to_handle.get("MAP"),
        label_to_handle.get("NDCG@10"),
        dummy,
    ]
    legend_handles = [h for h in legend_handles if h is not None]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=True,
        fontsize=40,
        handlelength=1.8,
        columnspacing=1.6,
    )

    # 왼쪽 여백을 조금 더 확보(오른쪽 축/라벨 구간만큼)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.14, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _choose_latest_by_model(items: Iterable, model_attr: str = "model", source_attr: str = "source_log"):
    # Pick the most recent log (by filesystem mtime) per model.
    best = {}
    for it in items:
        model = getattr(it, model_attr)
        src = Path(getattr(it, source_attr))
        mtime = src.stat().st_mtime if src.exists() else 0.0
        cur = best.get(model)
        if cur is None:
            best[model] = (mtime, it)
        else:
            if mtime >= cur[0]:
                best[model] = (mtime, it)
    return [v[1] for v in best.values()]


def _select_by_model_with_priority(
    items: Iterable,
    priority_fn,
    model_attr: str = "model",
    source_attr: str = "source_log",
):
    """
    For each model, select item with highest priority, breaking ties by mtime.
    Higher priority value wins.
    """
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


def _plot_bar_chart(
    out_path: Path,
    title: str,
    metric_names: List[str],
    rows: List[Tuple[str, List[Optional[float]]]],
    ylabel: str,
):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    # Try to use a Korean-capable font on Windows to avoid missing glyphs.
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for candidate in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
            if candidate in available:
                plt.rcParams["font.family"] = candidate
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    labels = [r[0] for r in rows]
    values = np.array([[np.nan if v is None else float(v) for v in r[1]] for r in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.22 if len(metric_names) >= 3 else 0.35

    fig_w = max(10.0, 0.7 * len(labels) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), dpi=160)

    offsets = (np.arange(len(metric_names)) - (len(metric_names) - 1) / 2.0) * width
    for i, mname in enumerate(metric_names):
        ax.bar(x + offsets[i], values[:, i], width, label=mname)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=min(3, len(metric_names)))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_best_rmse_pearson_time(
    out_path: Path,
    title: str,
    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]],  # label, rmse, pearson, time(s)
):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    # Korean font (Windows)
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for candidate in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
            if candidate in available:
                plt.rcParams["font.family"] = candidate
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    labels = [r[0] for r in rows]
    rmse = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=float)
    pearson = np.array([np.nan if r[2] is None else float(r[2]) for r in rows], dtype=float)
    tsec = np.array([np.nan if r[3] is None else float(r[3]) for r in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.22

    fig_w = max(10.0, 0.9 * len(labels) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), dpi=160)
    ax2 = ax.twinx()

    ax.bar(x - width, rmse, width, label="RMSE (↓)", color="#4C78A8")
    ax.bar(x, pearson, width, label="Pearson r (↑)", color="#F58518")
    ax2.bar(x + width, tsec, width, label="소요 시간(s) (↓)", color="#54A24B", alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel("RMSE / Pearson")
    ax2.set_ylabel("Time (s)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=3)

    plt.subplots_adjust(top=0.92, bottom=0.22, left=0.10, right=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parse severity experiment logs and draw bar charts.")
    parser.add_argument(
        "--log_root",
        type=str,
        default=str(Path("severity") / "experiment_logs"),
        help="Root directory containing *.log files (recursively scanned).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("CISC-S'26_graph") / "out"),
        help="Output directory for figures and CSVs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_root = (repo_root / args.log_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_files = list(_iter_log_files(log_root))

    ranking_candidates: List[ParsedRankingMetrics] = []
    regression_candidates: List[ParsedRegressionMetrics] = []

    for p in log_files:
        # Ranking metrics (NDCG/MRR/MAP) are most reliable in test_oracle_ratio logs.
        if "l2r_" in p.name and (
            "test_oracle_ratio" in p.name or "train_thresholds" in p.name or "train_score" in p.name
        ):
            r = parse_ranking_metrics_from_log(p)
            if r is not None:
                ranking_candidates.append(r)

        # CVSS 회귀/상관 지표(RMSE/Pearson)는 train_score_relevance_0_3 로그의 Test 블록에 존재.
        if "train_score_relevance_0_3" in p.name:
            rr = parse_regression_metrics_from_log(p)
            if rr is not None:
                regression_candidates.append(rr)

    def _ranking_priority(p: Path) -> int:
        name = p.name.lower()
        if "test_oracle_ratio" in name:
            return 3
        if "train_thresholds" in name:
            return 2
        if "train_score" in name:
            return 1
        return 0

    ranking = _select_by_model_with_priority(ranking_candidates, _ranking_priority)
    regression = _choose_latest_by_model(regression_candidates)

    # ---- Figure 1: L2R 모델끼리 Ranking 지표 비교 (NDCG@1/5/10 + MRR + MAP) ----
    allowed_l2r_models = {"listnet", "listmle", "ranknet", "lambdamart", "xgboost"}
    ranking_l2r = [r for r in ranking if r.model in allowed_l2r_models]
    ranking_l2r.sort(key=lambda x: x.model)

    _plot_l2r_ranking_compact(
        out_path=out_dir / "cvss_l2r_models_rmse_pearson_bar.png",
        title="L2R 성능지표 비교 (NDCG@1/5/10 + MRR + MAP, Test)",
        ranking=ranking_l2r,
    )

    # ---- Figure 2: CVSS (Test) - 그룹별 최고 모델 비교 ----
    def _score_key(x: ParsedRegressionMetrics) -> Tuple[float, float]:
        # RMSE 낮을수록 우선, 동률이면 Pearson 높을수록 우선.
        rmse = float("inf") if x.rmse is None else x.rmse
        pearson = float("-inf") if x.pearson is None else x.pearson
        return (rmse, -pearson)

    def _best_of(group: str, pool: List[ParsedRegressionMetrics]) -> Optional[ParsedRegressionMetrics]:
        cand = [x for x in pool if x.group == group]
        if not cand:
            return None
        return sorted(cand, key=_score_key)[0]

    best_l2r = _best_of("l2r", regression)
    best_anomaly = _best_of("anomaly", regression)
    best_regressor = _best_of("regressor", regression)

    rows2: List[Tuple[str, List[Optional[float]]]] = []
    for label, item in [
        ("Best L2R", best_l2r),
        ("Best Anomaly", best_anomaly),
        ("Best Regressor", best_regressor),
    ]:
        if item is None:
            continue
        rows2.append((f"{label}\n({item.model})", item.rmse, item.pearson, item.elapsed_s))

    _plot_best_rmse_pearson_time(
        out_path=out_dir / "cvss_best_l2r_vs_best_control_rmse_pearson_bar.png",
        title="CVSS 예측 성능 비교 (그룹별 최고, Test)",
        rows=rows2,
    )

    # Also save extracted tables as TSV for quick copy/paste into papers.
    def _fmt(v: Optional[float]) -> str:
        return "" if v is None else f"{v:.6f}"

    (out_dir / "regression_metrics.tsv").write_text(
        "model\tgroup\trmse\tpearson\telapsed_s\tsource_log\n"
        + "\n".join(
            f"{r.model}\t{r.group}\t{_fmt(r.rmse)}\t{_fmt(r.pearson)}\t{_fmt(r.elapsed_s)}\t{r.source_log}"
            for r in regression
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] parsed logs: {len(log_files)}")
    print(f"[OK] CVSS models parsed: {len(regression)}")
    print(f"[OK] fig1: {out_dir / 'cvss_l2r_models_rmse_pearson_bar.png'}")
    print(f"[OK] fig2: {out_dir / 'cvss_best_l2r_vs_best_control_rmse_pearson_bar.png'}")
    print(f"[OK] table: {out_dir / 'regression_metrics.tsv'}")


if __name__ == "__main__":
    main()

