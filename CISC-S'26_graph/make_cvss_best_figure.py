import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from graph_common import iter_log_files, parse_regression_metrics_from_log, choose_latest_by_model


def plot_best_rmse_pearson_time(out_path: Path, rows):
    # 독립 스타일: 다른 그래프와 rcParams 충돌 방지
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40
    plt.rcParams["axes.unicode_minus"] = False

    labels = [r[0] for r in rows]
    rmse = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=float)
    pearson = np.array([np.nan if r[2] is None else float(r[2]) for r in rows], dtype=float)
    tsec = np.array([np.nan if r[3] is None else float(r[3]) for r in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax2 = ax.twinx()

    ax.bar(x - width, rmse, width, label="RMSE (↓)", color="#4C78A8")
    ax.bar(x, pearson, width, label="Pearson r (↑)", color="#F58518")
    ax2.bar(x + width, tsec, width, label="Time (s) (↓)", color="#54A24B", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("RMSE / Pearson")
    ax2.set_ylabel("Time (s)")

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # legend box under plot, like threshold graph
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=True,
        fontsize=40,
    )

    plt.subplots_adjust(top=0.95, bottom=0.18, left=0.12, right=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_root", type=str, default=str(Path("severity") / "experiment_logs"))
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("CISC-S'26_graph") / "out" / "cvss_best_l2r_vs_best_control_rmse_pearson_bar.png"),
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_root = (repo_root / args.log_root).resolve()
    out_path = (repo_root / args.out).resolve()

    candidates = []
    for lf in iter_log_files(log_root):
        if "train_score_relevance_0_3" in lf.name:
            m = parse_regression_metrics_from_log(lf)
            if m is not None:
                candidates.append(m)
    metrics = choose_latest_by_model(candidates)

    def score_key(x):
        rmse = float("inf") if x.rmse is None else x.rmse
        pearson = float("-inf") if x.pearson is None else x.pearson
        return (rmse, -pearson)

    def best_of(group: str):
        cand = [x for x in metrics if x.group == group]
        if not cand:
            return None
        return sorted(cand, key=score_key)[0]

    best_l2r = best_of("l2r")
    best_anom = best_of("anomaly")
    best_reg = best_of("regressor")

    rows = []
    for label, item in [
        ("Best L2R", best_l2r),
        ("Best Anomaly", best_anom),
        ("Best Regressor", best_reg),
    ]:
        if item is None:
            continue
        rows.append((f"{label}\n({item.model})", item.rmse, item.pearson, item.elapsed_s))

    plot_best_rmse_pearson_time(out_path, rows)


if __name__ == "__main__":
    main()

