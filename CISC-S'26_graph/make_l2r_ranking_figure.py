import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from graph_common import (
    iter_log_files,
    parse_ranking_metrics_from_log,
    select_by_model_with_priority,
)


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

    # Bars
    ax.bar(nd_pos[0], nd1, nd_w, label="NDCG@1", color="#4C78A8")
    ax.bar(nd_pos[1], nd5, nd_w, label="NDCG@5", color="#72B7B2")
    ax.bar(nd_pos[2], nd10, nd_w, label="NDCG@10", color="#A0CBE8")
    ax.bar(mrr_pos, mrr, rect_w, label="MRR", color="#F58518")
    ax.bar(map_pos, map_, rect_w, label="MAP", color="#54A24B")

    # Red dashed outline for max NDCG per model
    stacked = np.vstack([nd1, nd5, nd10])
    nd_max = np.array(
        [np.nan if np.all(np.isnan(stacked[:, i])) else float(np.nanmax(stacked[:, i])) for i in range(stacked.shape[1])],
        dtype=float,
    )
    for i in range(len(models)):
        if math.isnan(float(nd_max[i])):
            continue
        ax.add_patch(
            Rectangle(
                (float(ndcg_left[i]), 0.0),
                float(rect_w),
                float(nd_max[i]),
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
    ax.set_ylim(bottom=0.0)

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
    p.add_argument("--log_root", type=str, default=str(Path("severity") / "experiment_logs"))
    p.add_argument("--out", type=str, default=str(Path("CISC-S'26_graph") / "out" / "cvss_l2r_models_rmse_pearson_bar.png"))
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_root = (repo_root / args.log_root).resolve()
    out_path = (repo_root / args.out).resolve()

    ranking_candidates = []
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

