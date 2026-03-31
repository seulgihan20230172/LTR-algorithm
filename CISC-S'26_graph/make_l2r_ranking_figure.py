import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

from graph_common import (
    iter_log_files,
    parse_ranking_metrics_from_log,
    select_by_model_with_priority,
)

def _map_series(values: np.ndarray, *, y_min: float, low_max: float, high_min: float, low_end: float, high_start: float) -> np.ndarray:
    """
    Piecewise-linear mapping into a compressed display y coordinate.
    - values <= low_max: map into [y_min, low_end]
    - values >= high_min: map into [high_start, 1.0]
    - between: clamp to low_end (gap region)
    """
    v = np.array(values, dtype=float)
    out = np.full_like(v, np.nan, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return out

    high_max = float(np.nanmax(finite))
    low_denom = max(1e-12, low_max - y_min)
    high_denom = max(1e-12, high_max - high_min)
    high_span = max(1e-12, 1.0 - high_start)

    for i, val in enumerate(v):
        if not np.isfinite(val):
            continue
        if val <= low_max:
            out[i] = y_min + (val - y_min) / low_denom * (low_end - y_min)
        elif val >= high_min:
            out[i] = high_start + (val - high_min) / high_denom * high_span
        else:
            out[i] = low_end
    return out


def _add_break_symbol(ax, break_y: float, x_min: float, x_max: float):
    """Double wavy break (two lines) with white fill between, across [x_min, x_max]."""
    x_data = np.linspace(x_min, x_max, 500)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    if y_range <= 0:
        return

    wave_pattern = 0.012 * y_range * np.sin(15 * np.pi * (x_data - x_min) / (x_max - x_min))
    y_data1 = break_y + 0.024 * y_range + wave_pattern
    y_data2 = break_y + 0.008 * y_range + wave_pattern

    polygon_points = np.vstack(
        [
            np.column_stack([x_data, y_data1]),
            np.column_stack([x_data[::-1], y_data2[::-1]]),
        ]
    )
    poly = Polygon(polygon_points, facecolor="white", edgecolor="none", zorder=13, transform=ax.transData)
    ax.add_patch(poly)

    ax.plot(x_data, y_data1, "k-", linewidth=2.4, clip_on=False, zorder=15)
    ax.plot(x_data, y_data2, "k-", linewidth=2.4, clip_on=False, zorder=15)
    ax.plot([x_data[0], x_data[0]], [y_data1[0], y_data2[0]], "k-", linewidth=3.4, clip_on=False, zorder=15)
    ax.plot([x_data[-1], x_data[-1]], [y_data1[-1], y_data2[-1]], "k-", linewidth=3.4, clip_on=False, zorder=15)


def _ranking_priority(p: Path) -> int:
    name = p.name.lower()
    if "test_oracle_ratio" in name:
        return 3
    if "train_thresholds" in name:
        return 2
    if "train_score" in name:
        return 1
    return 0


def plot_l2r_ranking_compact(out_path: Path, ranking):
    # threshold_graph와 동일한 width/font 비율(가로 16, 폰트 40). 세로는 낮게 유지.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40
    plt.rcParams["axes.unicode_minus"] = False

    models = [r.model for r in ranking]
    nd1 = np.array([np.nan if r.ndcg1 is None else float(r.ndcg1) for r in ranking], dtype=float)
    nd5 = np.array([np.nan if r.ndcg5 is None else float(r.ndcg5) for r in ranking], dtype=float)
    nd10 = np.array([np.nan if r.ndcg10 is None else float(r.ndcg10) for r in ranking], dtype=float)
    mrr = np.array([np.nan if r.mrr is None else float(r.mrr) for r in ranking], dtype=float)
    map_ = np.array([np.nan if r.map is None else float(r.map) for r in ranking], dtype=float)

    # make_severity_graphs.py(원본)과 동일 파라미터로 맞춤
    rect_w = 1.5
    gap = 0.0
    model_span = rect_w * 3 + gap * 2
    block_gap = 2.10
    base = np.arange(len(models)) * (model_span + block_gap)

    ndcg_left = base - model_span / 2.0
    mrr_left = ndcg_left + rect_w + gap
    map_left = mrr_left + rect_w + gap

    nd_w = rect_w / 3.0
    nd_pos = [ndcg_left + nd_w * 0.5, ndcg_left + nd_w * 1.5, ndcg_left + nd_w * 2.5]
    mrr_pos = mrr_left + rect_w * 0.5
    map_pos = map_left + rect_w * 0.5

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    # y-axis: 0.90~1.00(표시) 구간만 보여주되, 막대는 0.90 기준선에서 시작하게 해서
    # "1.00을 넘는 것처럼" 보이는 착시를 방지한다.
    y_min = 0.90
    xgb_idx = next((i for i, m in enumerate(models) if m == "xgboost"), None)
    low_max = None
    high_min = None
    break_pos = None
    if xgb_idx is not None and np.isfinite(nd1[xgb_idx]) and np.isfinite(nd5[xgb_idx]) and np.isfinite(nd10[xgb_idx]):
        low_max = float(np.nanmax([nd1[xgb_idx], nd5[xgb_idx], nd10[xgb_idx]]))
        other_nd = np.concatenate([np.delete(nd1, xgb_idx), np.delete(nd5, xgb_idx), np.delete(nd10, xgb_idx)])
        other_nd = other_nd[np.isfinite(other_nd)]
        high_min = float(np.nanmin(other_nd)) if other_nd.size else low_max

        low_end = 0.94
        high_start = 0.965
        break_pos = (low_end + high_start) / 2.0 - 0.004

        nd1p = _map_series(nd1, y_min=y_min, low_max=low_max, high_min=high_min, low_end=low_end, high_start=high_start)
        nd5p = _map_series(nd5, y_min=y_min, low_max=low_max, high_min=high_min, low_end=low_end, high_start=high_start)
        nd10p = _map_series(nd10, y_min=y_min, low_max=low_max, high_min=high_min, low_end=low_end, high_start=high_start)
        mrrp = _map_series(mrr, y_min=y_min, low_max=low_max, high_min=high_min, low_end=low_end, high_start=high_start)
        mapp = _map_series(map_, y_min=y_min, low_max=low_max, high_min=high_min, low_end=low_end, high_start=high_start)
    else:
        nd1p, nd5p, nd10p, mrrp, mapp = nd1, nd5, nd10, mrr, map_

    # Bars
    ax.bar(nd_pos[0], nd1p - y_min, nd_w, bottom=y_min, label="NDCG@1", color="#4C78A8")
    ax.bar(nd_pos[1], nd5p - y_min, nd_w, bottom=y_min, label="NDCG@5", color="#72B7B2")
    ax.bar(nd_pos[2], nd10p - y_min, nd_w, bottom=y_min, label="NDCG@10", color="#A0CBE8")
    ax.bar(mrr_pos, mrrp - y_min, rect_w, bottom=y_min, label="MRR", color="#F58518")
    ax.bar(map_pos, mapp - y_min, rect_w, bottom=y_min, label="MAP", color="#54A24B")

    # Red dashed outline for max NDCG per model
    # (막대와 1:1로 맞추기 위해 "이미 그려진 display 값"의 최대를 직접 사용)
    nd_stack_disp = np.vstack([nd1p, nd5p, nd10p])
    nd_max_disp = np.array(
        [np.nan if np.all(np.isnan(nd_stack_disp[:, i])) else float(np.nanmax(nd_stack_disp[:, i])) for i in range(nd_stack_disp.shape[1])],
        dtype=float,
    )

    for i in range(len(models)):
        if not np.isfinite(nd_max_disp[i]):
            continue
        ax.add_patch(
            Rectangle(
                (float(ndcg_left[i]), float(y_min)),
                float(rect_w),
                float(nd_max_disp[i] - y_min),
                fill=False,
                edgecolor="red",
                linewidth=2.0,
                linestyle=":",
                alpha=0.95,
            )
        )

    ax.set_xticks(base)
    ax.set_xticklabels(models, rotation=0, ha="center")
    ax.set_ylabel("Score", fontsize=40)
    ax.tick_params(axis="both", labelsize=40)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 상단 경계에 붙지 않도록 약간의 헤드룸을 둔다(라벨은 1.00 유지)
    ax.set_ylim(0.90, 1.01)
    # xlim을 명시해서 물결이 "그래프 박스 폭" 전체를 덮도록 함
    ax.set_xlim(float(ndcg_left[0]) - 0.6, float(map_left[-1] + rect_w) + 0.6)
    if break_pos is not None:
        x_min, x_max = ax.get_xlim()
        _add_break_symbol(ax, break_pos, x_min=float(x_min), x_max=float(x_max))

        # 1.00 라벨이 상단 경계에 닿지 않도록 tick 위치만 살짝 아래로
        top_pos = 1.005
        tick_pos = [0.90, 0.94, 0.965, top_pos]
        tick_lab = [0.90, low_max, high_min, 1.00]
        dedup_pos = []
        dedup_lab = []
        for p, l in zip(tick_pos, tick_lab):
            if l is None:
                continue
            if dedup_lab and abs(float(l) - float(dedup_lab[-1])) < 1e-10:
                continue
            dedup_pos.append(p)
            dedup_lab.append(l)
        ax.set_yticks(dedup_pos)
        ax.set_yticklabels([f"{float(v):.2f}" for v in dedup_lab])
    else:
        ax.set_yticks([0.90, 0.95, 0.995])
        ax.set_yticklabels(["0.90", "0.95", "1.00"])

    # Two-line legend, centered box; keep current layout that you iterated on
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    dummy = Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", label=" ")
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

    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.14, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log_root",
        type=str,
        nargs="*",
        default=[
            str(Path("severity") / "experiment_logs" / "all_20260330_214454"),
            str(Path("severity") / "experiment_logs" / "from_lambdamart_20260331_092138"),
        ],
        help="One or more log roots to search.",
    )
    p.add_argument("--out", type=str, default=str(Path("CISC-S'26_graph") / "out" / "cvss_l2r_models_rmse_pearson_bar.png"))
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = (repo_root / args.out).resolve()

    ranking_candidates = []
    for lr in args.log_root:
        log_root = (repo_root / lr).resolve()
        for lf in iter_log_files(log_root):
            if "l2r_" in lf.name and ("test_oracle_ratio" in lf.name or "train_thresholds" in lf.name or "train_score" in lf.name):
                m = parse_ranking_metrics_from_log(lf)
                if m is not None:
                    ranking_candidates.append(m)

    ranking = select_by_model_with_priority(ranking_candidates, _ranking_priority)
    allowed = {"listnet", "listmle", "ranknet", "lambdamart", "xgboost"}
    ranking = sorted([r for r in ranking if r.model in allowed], key=lambda x: x.model)

    plot_l2r_ranking_compact(out_path, ranking)


if __name__ == "__main__":
    main()

