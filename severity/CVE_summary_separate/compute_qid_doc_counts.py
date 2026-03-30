from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _compute_cve_calendar_day(df: pd.DataFrame) -> pd.Series:
    if "mod_date" not in df.columns or "pub_date" not in df.columns:
        raise ValueError("CSV에 mod_date, pub_date 컬럼이 필요합니다.")
    md = pd.to_datetime(df["mod_date"], errors="coerce").dt.normalize()
    pd_ = pd.to_datetime(df["pub_date"], errors="coerce").dt.normalize()
    d = pd.concat([md, pd_], axis=1).max(axis=1)
    if d.isna().any():
        bad = int(d.isna().sum())
        raise ValueError(f"mod_date/pub_date 파싱 실패 행이 있습니다: {bad}건")
    return d


def _summary_stats(counts: np.ndarray) -> dict[str, float]:
    c = np.asarray(counts, dtype=np.int64)
    p = np.percentile(c, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]).astype(float)
    return {
        "n_qid": float(len(c)),
        "n_docs_total": float(np.sum(c)),
        "mean": float(np.mean(c)),
        "std": float(np.std(c)),
        "min": float(p[0]),
        "p01": float(p[1]),
        "p05": float(p[2]),
        "p10": float(p[3]),
        "p25": float(p[4]),
        "p50": float(p[5]),
        "p75": float(p[6]),
        "p90": float(p[7]),
        "p95": float(p[8]),
        "p99": float(p[9]),
        "max": float(p[10]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="CVE CSV에서 cve_calendar_day(qid)별 문서 수 분포 계산")
    p.add_argument(
        "--csv",
        type=str,
        default="severity/CVE_summary_separate/cve-datasets/cve_with_meaning70.csv",
        help="입력 CSV (프로젝트 루트 기준 상대경로 가능)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 .log 경로 (미지정 시 CVE_summary_separate/ 아래 자동 생성)",
    )
    p.add_argument("--topk", type=int, default=30, help="문서 수 상위 일자 top-k 출력")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", usecols=["mod_date", "pub_date"])
    day = _compute_cve_calendar_day(df)
    vc = day.value_counts().sort_index()
    counts = vc.to_numpy(dtype=np.int64)
    s = _summary_stats(counts)

    out_path = Path(args.out).resolve() if args.out else csv_path.parent.parent / "qid_doc_counts_cve_calendar_day.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# qid_mode=cve_calendar_day (날짜=normalize(max(mod_date,pub_date)))\n")
        f.write(f"# csv: {csv_path}\n")
        f.write(f"# generated: {date.today().isoformat()}\n\n")
        f.write("[요약 통계]\n")
        f.write(
            "n_qid={n_qid:.0f} n_docs_total={n_docs_total:.0f} mean={mean:.2f} std={std:.2f} "
            "min={min:.0f} p50={p50:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f} max={max:.0f}\n".format(**s)
        )
        f.write(
            "percentiles: p01={p01:.0f} p05={p05:.0f} p10={p10:.0f} p25={p25:.0f} "
            "p50={p50:.0f} p75={p75:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f}\n".format(**s)
        )

        f.write("\n[상위 일자]\n")
        topk = max(1, int(args.topk))
        top = vc.sort_values(ascending=False).head(topk)
        for d, n in top.items():
            f.write(f"{pd.Timestamp(d).date().isoformat()}\t{int(n)}\n")

        f.write("\n[최하위 일자]\n")
        bottom = vc.sort_values(ascending=True).head(min(topk, len(vc)))
        for d, n in bottom.items():
            f.write(f"{pd.Timestamp(d).date().isoformat()}\t{int(n)}\n")

    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()

