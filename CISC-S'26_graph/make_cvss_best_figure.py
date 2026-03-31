import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from graph_common import iter_log_files, parse_regression_metrics_from_log, choose_latest_by_model


def _map_series(values: np.ndarray, *, low_max: float, high_min: float, low_end: float, high_start: float, high_span: float) -> np.ndarray:
    """
    Piecewise-linear mapping into a compressed display y coordinate (reference-style).
    - values <= low_max: scale into [0, low_end]
    - values >= high_min: scale into [high_start, high_start+high_span]
    - between: clamp to low_end (gap region)
    """
    v = np.array(values, dtype=float)
    out = np.full_like(v, np.nan, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return out
    high_max = float(np.nanmax(finite))
    denom = max(1e-9, high_max - high_min)
    for i, val in enumerate(v):
        if not np.isfinite(val):
            continue
        if val <= low_max:
            out[i] = (val / low_max) * low_end if low_max > 0 else 0.0
        elif val >= high_min:
            out[i] = high_start + (val - high_min) / denom * high_span
        else:
            out[i] = low_end
    return out


def _add_break_symbol(ax, break_y: float, x_min: float, x_max: float):
    """Add a double wavy break symbol (two lines) with white fill between, across [x_min, x_max]."""
    from matplotlib.patches import Polygon

    x_data = np.linspace(x_min, x_max, 500)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    if y_range <= 0:
        return

    # Thinner white band between the two waves (user request):
    # - reduce vertical offsets
    # - reduce wave amplitude
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

    # thinner break marker (user request)
    ax.plot(x_data, y_data1, "k-", linewidth=2.4, clip_on=False, zorder=15)
    ax.plot(x_data, y_data2, "k-", linewidth=2.4, clip_on=False, zorder=15)
    ax.plot([x_data[0], x_data[0]], [y_data1[0], y_data2[0]], "k-", linewidth=3.4, clip_on=False, zorder=15)
    ax.plot([x_data[-1], x_data[-1]], [y_data1[-1], y_data2[-1]], "k-", linewidth=3.4, clip_on=False, zorder=15)


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

    # Reference-style "single axis" broken scale: compress values into one y-scale and draw one break symbol.
    idx_l2r = next((i for i, s in enumerate(labels) if s.startswith("Best L2R")), None)
    idx_ae = next((i for i, s in enumerate(labels) if s.startswith("Best AutoEncoder")), None)

    rmse_rest = np.delete(rmse, idx_ae) if idx_ae is not None and rmse.size > 1 else rmse
    time_rest = np.delete(tsec, idx_l2r) if idx_l2r is not None and tsec.size > 1 else tsec

    left_pool = np.concatenate([rmse_rest[np.isfinite(rmse_rest)], pearson[np.isfinite(pearson)]])
    low_max_left = float(np.nanmax(left_pool)) if left_pool.size else 1.0
    high_min_left = float(rmse[idx_ae]) if idx_ae is not None and np.isfinite(rmse[idx_ae]) else max(low_max_left * 3.0, low_max_left + 1.0)

    low_max_right = float(np.nanmax(time_rest[np.isfinite(time_rest)])) if np.any(np.isfinite(time_rest)) else 1.0
    high_min_right = float(tsec[idx_l2r]) if idx_l2r is not None and np.isfinite(tsec[idx_l2r]) else max(low_max_right * 3.0, low_max_right + 1.0)

    # Display scale parameters (0~40 범위에서 표시되도록 구성)
    # - low 구간: 0 .. low_end
    # - high 구간: high_start .. high_start+high_span (<= 40)
    low_end = 18.0
    high_start = 28.0
    high_span = 10.0
    break_pos = (low_end + high_start) / 2.0

    y_rmse = _map_series(rmse, low_max=low_max_left, high_min=high_min_left, low_end=low_end, high_start=high_start, high_span=high_span)
    y_pearson = _map_series(pearson, low_max=low_max_left, high_min=high_min_left, low_end=low_end, high_start=high_start, high_span=high_span)
    y_time = _map_series(tsec, low_max=low_max_right, high_min=high_min_right, low_end=low_end, high_start=high_start, high_span=high_span)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.0, 30.0)

    ax.bar(x - width, y_rmse, width, label="RMSE (↓)", color="#3B4CC0")
    ax.bar(x, y_pearson, width, label="Pearson r (↑)", color="#D1495B")
    ax.bar(x + width, y_time, width, label="Time (s) (↓)", color="#2A9D8F", alpha=0.90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("RMSE / Pearson")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Draw one long break symbol between "rest" and "outliers"
    _add_break_symbol(ax, break_pos, x_min=-0.5, x_max=len(labels) - 0.5)

    # Left y-axis: show ticks as ORIGINAL (uncompressed) values.
    left_max = float(np.nanmax(np.concatenate([rmse[np.isfinite(rmse)], pearson[np.isfinite(pearson)]]))) if (
        np.any(np.isfinite(rmse)) or np.any(np.isfinite(pearson))
    ) else high_min_left
    left_tick_pos = [0.0, low_end, high_start, high_start + high_span]
    left_tick_lab = [0.0, low_max_left, high_min_left, left_max]
    # Deduplicate labels if equal (prevents weird repeats)
    dedup_pos = []
    dedup_lab = []
    for p, l in zip(left_tick_pos, left_tick_lab):
        if dedup_lab and abs(float(l) - float(dedup_lab[-1])) < 1e-9:
            continue
        dedup_pos.append(p)
        dedup_lab.append(l)
    ax.set_yticks(dedup_pos)
    ax.set_yticklabels([f"{v:.2f}".rstrip("0").rstrip(".") for v in dedup_lab])

    # Right y-axis for Time: use display positions but label with original values.
    ax_time = ax.twinx()
    ax_time.set_ylim(ax.get_ylim())
    ax_time.set_ylabel("Time (s)")
    time_max = float(np.nanmax(tsec[np.isfinite(tsec)])) if np.any(np.isfinite(tsec)) else high_min_right

    # Avoid duplicate labels (e.g., if high_min_right == time_max)
    tick_pos = [0.0, low_end, high_start, high_start + high_span]
    tick_lab = [int(0), int(low_max_right), int(high_min_right), int(time_max)]
    dedup_pos = []
    dedup_lab = []
    for p, l in zip(tick_pos, tick_lab):
        if dedup_lab and l == dedup_lab[-1]:
            continue
        dedup_pos.append(p)
        dedup_lab.append(l)
    ax_time.set_yticks(dedup_pos)
    ax_time.set_yticklabels([f"{l}" for l in dedup_lab])

    # Legend under plot
    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_leg,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=40,
        bbox_to_anchor=(0.5, 0.03),
        bbox_transform=fig.transFigure,
    )

    plt.subplots_adjust(top=0.95, bottom=0.14, left=0.12, right=0.88)
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
        ("Best AutoEncoder", best_anom),
        ("Best Regressor", best_reg),
    ]:
        if item is None:
            continue
        rows.append((f"{label}\n({item.model})", item.rmse, item.pearson, item.elapsed_s))

    plot_best_rmse_pearson_time(out_path, rows)


if __name__ == "__main__":
    main()

