#!/usr/bin/env python3
"""
severity 실험 로그(.log)에서 Test 지표·MRR·전체 시간을 뽑아 Markdown 표로 저장한다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _find_test_block(text: str) -> str | None:
    markers = ["=== Test (mode=", "=== Test ===", "[Test]"]
    best = -1
    for m in markers:
        i = text.rfind(m)
        if i > best:
            best = i
    if best < 0:
        return None
    return text[best:]


def _pick_float(pattern: str, block: str) -> float | None:
    m = re.search(pattern, block, re.MULTILINE)
    if not m:
        return None
    return float(m.group(1))


def parse_log_text(text: str) -> dict[str, float | str | None]:
    test_block = _find_test_block(text) or text
    acc = _pick_float(r"^Accuracy:\s*([0-9.]+)\s*$", test_block)
    if acc is None:
        acc = _pick_float(r"Accuracy:\s*([0-9.]+)", test_block)
    macro_f1 = _pick_float(r"^Macro F1:\s*([0-9.]+)\s*$", test_block)
    if macro_f1 is None:
        macro_f1 = _pick_float(r"Macro F1:\s*([0-9.]+)", test_block)

    val_mrr = _pick_float(r"Validation MRR:\s*([0-9.eE+-]+)", text)
    test_mrr = _pick_float(r"Test MRR:\s*([0-9.eE+-]+)", text)
    total_s = _pick_float(r"전체:\s*([0-9.]+)\s*s", text)

    status = "ok"
    if re.search(r"Traceback|ModuleNotFoundError|RuntimeError|Error:", text):
        status = "error"

    return {
        "status": status,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "val_mrr": val_mrr,
        "test_mrr": test_mrr,
        "total_sec": total_s,
    }


def fmt_metric(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}" if abs(x) < 10 else f"{x:.4f}"


def fmt_sec(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", type=str, help=".log 파일이 있는 디렉터리")
    p.add_argument("--out", type=str, default=None, help="출력 md 경로 (기본: log_dir/SUMMARY.md)")
    args = p.parse_args()
    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else log_dir / "SUMMARY.md"
    logs = sorted(log_dir.glob("*.log"))
    rows: list[tuple[str, dict]] = []
    for lp in logs:
        try:
            text = lp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows.append((lp.name, parse_log_text(text)))

    lines: list[str] = []
    lines.append("# Severity 실험 요약\n")
    lines.append(f"- 로그 디렉터리: `{log_dir.as_posix()}`\n")
    lines.append(f"- 로그 파일 수: {len(rows)}\n")
    lines.append("\n| 실험(로그 파일) | 상태 | Test Acc | Test Macro F1 | Val MRR | Test MRR | 전체(s) |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for name, d in rows:
        lines.append(
            f"| `{name}` | {d['status']} | {fmt_metric(d['test_accuracy'])} | {fmt_metric(d['test_macro_f1'])} | "
            f"{fmt_metric(d['val_mrr'])} | {fmt_metric(d['test_mrr'])} | {fmt_sec(d['total_sec'])} |\n"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[요약 저장] {out_path.resolve()}")


if __name__ == "__main__":
    main()
