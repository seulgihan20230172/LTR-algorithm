import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from severity.regression.models import build_regression_model
from severity.experiment_config import DEFAULT_CONFIG_PATH, load_experiment_config
from severity.severity_rank_controlgroup import (
    LABEL_ORDER_DESC,
    TARGET_COL,
    TeeIO,
    allocate_counts,
    apply_test_mode,
    assign_top_scores,
    boundaries_from_train_sorted,
    class_fractions,
    default_log_path,
    evaluate_ranking_all,
    fit_transform_xy,
    make_qid,
    prepare_splits,
    report_metrics,
)


def run(csv_path, model_name, test_mode, test_size, val_size, random_state, group_size):
    t_total_start = time.perf_counter()
    x_train, x_val, x_test, y_train, y_val, y_test, yr_train, yr_val, yr_test = prepare_splits(
        csv_path, test_size, val_size, random_state
    )

    t_train_val_start = time.perf_counter()
    xt, xv, xs, _, _ = fit_transform_xy(x_train, x_val, x_test)

    model = build_regression_model(model_name, random_state=random_state)
    model.fit(xt, yr_train, xv, yr_val)
    t_train_val_end = time.perf_counter()

    t_eval_start = time.perf_counter()
    s_train = model.score(xt)
    s_val = model.score(xv)
    s_test = model.score(xs)

    n_tr, n_va, n_te = len(y_train), len(y_val), len(y_test)
    qid_val = make_qid(n_va, group_size)
    qid_test = make_qid(n_te, group_size)

    fr_train = class_fractions(y_train)
    counts_train = np.array([y_train.value_counts().get(lbl, 0) for lbl in LABEL_ORDER_DESC], dtype=int)
    counts_val = allocate_counts(n_va, fr_train)
    pred_val = assign_top_scores(s_val, counts_val)
    print(f"\n[검증] train 비율로 배정한 분류 정확도(참고): {(pred_val == y_val.values).mean():.4f}")

    train_desc = np.sort(s_train)[::-1]
    th, train_sorted_asc = boundaries_from_train_sorted(train_desc, counts_train)

    pred_train = assign_top_scores(s_train, counts_train)
    print("\n[Train] 실제 Severity와 비교 (train 비율 배정)")
    report_metrics(y_train.values, pred_train, "Train (비율 배정)")

    pred_test = apply_test_mode(test_mode, s_test, y_test, th)

    print(f"\nTrain에서 추정한 score 경계(내림차순 상위부터 Critical→…): {np.array2string(th, precision=6)}")
    print(f"(참고) train score min/max: {train_sorted_asc.min():.6f} / {train_sorted_asc.max():.6f}")

    print("\n[Test] 실제 Severity와 비교")
    t_test_sev_start = time.perf_counter()
    report_metrics(y_test.values, pred_test, f"Test (mode={test_mode})")
    t_test_sev_end = time.perf_counter()

    m_val = evaluate_ranking_all(yr_val, s_val, qid_val)
    m_test = evaluate_ranking_all(yr_test, s_test, qid_test)

    out = x_test.copy()
    out[TARGET_COL] = y_test.values
    out["anomaly_score"] = s_test
    out["pred_severity"] = pred_test
    outp = Path(csv_path).with_name(Path(csv_path).stem + f"_reg_{model_name}_{test_mode}.csv")
    out.to_csv(outp, index=False)

    t_eval_end = time.perf_counter()
    t_total_end = time.perf_counter()

    print(f"\n저장: {outp}")
    print("\n=== 소요 시간 ===")
    print(f"CSV 로드·train/val/test 분할: {t_train_val_start - t_total_start:.3f} s")
    print(f"전처리(인코딩·스케일) + 학습(train∪val): {t_train_val_end - t_train_val_start:.3f} s")
    print(f"추론·경계·평가·저장(전수): {t_eval_end - t_eval_start:.3f} s")
    print(f"  └ Test severity 정답 비교(report_metrics)만: {t_test_sev_end - t_test_sev_start:.3f} s")
    print(f"전체: {t_total_end - t_total_start:.3f} s")
    print("\n=== 랭킹 지표 MRR (relevance=0..3) ===")
    print(f"Validation MRR: {m_val['MRR']:.6f}")
    print(f"Test MRR:       {m_test['MRR']:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"실험 YAML 경로 (기본: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument("--model", choices=["xgboost_regressor"], default="xgboost_regressor")
    p.add_argument("--test-mode", choices=["train_thresholds", "test_oracle_ratio"], default="train_thresholds")
    '''
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--group-size", type=int, default=64)
    '''
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    csv_path = cfg["data"]["csv"]
    test_size = float(cfg["split"]["test_size"])
    val_size = float(cfg["split"]["val_size"])
    random_state = int(cfg["split"]["random_state"])
    group_size = int(cfg["ranking"]["group_size"])

    log_f = None
    old_out, old_err = sys.stdout, sys.stderr
    if not args.no_log:
        log_path = Path(args.log) if args.log else default_log_path("train_severity_reg", args.model, args.test_mode)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")
        sys.stdout = TeeIO(old_out, log_f)
        sys.stderr = TeeIO(old_err, log_f)
        print(f"[로그 파일] {log_path.resolve()}", flush=True)

    try:
        run(csv_path, args.model, args.test_mode, test_size, val_size, random_state, group_size)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()


