import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from severity.classification.models import build_classification_model
from severity.experiment_config import DEFAULT_CONFIG_PATH, load_experiment_config, resolve_test_mode
from severity.feature_importance_log import write_feature_importance_log
from severity.severity_rank_controlgroup import (
    TeeIO,
    allocate_counts,
    apply_test_mode,
    assign_by_thresholds,
    assign_top_scores,
    boundaries_from_train_sorted,
    class_fractions,
    default_log_path,
    evaluate_ranking_all,
    fit_transform_xy,
    prepare_splits,
    print_per_class_recall_by_true_label,
    report_metrics,
    severity_from_train_minmax_relevance,
)
from severity.severity_schema import LABEL_ORDER_DESC, TARGET_COL


def run(
    csv_path,
    model_name,
    test_mode,
    test_size,
    val_size,
    random_state,
    group_size,
    *,
    include_categorical_columns: bool,
    ordinal_severity_metrics: bool,
    qid_mode: str,
    global_qid: int,
    split_mode: str,
):
    t_total_start = time.perf_counter()
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        yr_train,
        yr_val,
        yr_test,
        _qid_train,
        qid_val,
        qid_test,
    ) = prepare_splits(
        csv_path,
        test_size,
        val_size,
        random_state,
        include_categorical_columns=include_categorical_columns,
        qid_mode=qid_mode,
        global_qid=global_qid,
        split_mode=split_mode,
    )

    t_train_val_start = time.perf_counter()
    xt, xv, xs, pre, _ = fit_transform_xy(x_train, x_val, x_test)

    model = build_classification_model(model_name, random_state=random_state)
    model.fit(xt, y_train.values, xv, y_val.values)
    fi_path = write_feature_importance_log(
        pre,
        model,
        prefix="train_severity_cls",
        model_name=model_name,
        test_mode=test_mode,
        X_reference=xt,
        random_state=random_state,
    )
    print(f"\n[피처 중요도 로그] {fi_path.resolve()}", flush=True)
    t_train_val_end = time.perf_counter()

    t_eval_start = time.perf_counter()
    s_train = model.score(xt)
    s_val = model.score(xv)
    s_test = model.score(xs)

    n_va = len(y_val)

    fr_train = class_fractions(y_train)
    counts_train = np.array([y_train.value_counts().get(lbl, 0) for lbl in LABEL_ORDER_DESC], dtype=int)
    train_desc = np.sort(s_train)[::-1]
    th, train_sorted_asc = boundaries_from_train_sorted(train_desc, counts_train)

    if test_mode == "train_thresholds":
        pred_val = assign_by_thresholds(s_val, th)
        print(f"\n[검증] train score 경계(threshold) 기준 분류 정확도(참고): {(pred_val == y_val.values).mean():.4f}")
        pred_train = assign_by_thresholds(s_train, th)
        train_metrics_name = "Train (train_thresholds)"
    elif test_mode == "train_score_relevance_0_3":
        pred_val = severity_from_train_minmax_relevance(s_val, s_train)
        print(
            f"\n[검증] train min~max→[0,3] relevance 반올림 기준 분류 정확도(참고): "
            f"{(pred_val == y_val.values).mean():.4f}"
        )
        pred_train = severity_from_train_minmax_relevance(s_train, s_train)
        train_metrics_name = "Train (train_score_relevance_0_3)"
    else:
        counts_val = allocate_counts(n_va, fr_train)
        pred_val = assign_top_scores(s_val, counts_val)
        print(f"\n[검증] train 비율로 배정한 분류 정확도(참고): {(pred_val == y_val.values).mean():.4f}")
        pred_train = assign_top_scores(s_train, counts_train)
        train_metrics_name = "Train (비율 배정)"

    print_per_class_recall_by_true_label(
        y_val.values,
        pred_val,
        title="[검증] 클래스별 리콜·오답·FP 요약",
    )
    print("  ※ Train/Test의 classification_report recall 열은 위 ‘맞춤’과 대응합니다. precision·F1은 같은 표에서 확인.")

    print("\n[Train] 실제 Severity와 비교")
    report_metrics(
        y_train.values,
        pred_train,
        train_metrics_name,
        ordinal_severity_metrics=ordinal_severity_metrics,
    )

    pred_test = apply_test_mode(test_mode, s_test, y_test, th, s_train=s_train)

    if test_mode == "train_score_relevance_0_3":
        lo, hi = float(np.min(s_train)), float(np.max(s_train))
        print(
            f"\n[train_score_relevance_0_3] train 점수 min~max를 [0,3] relevance로 선형 매핑 후 반올림 "
            f"(train min={lo:.6f}, train max={hi:.6f})"
        )
    else:
        print(f"\nTrain에서 추정한 score 경계(내림차순 상위부터 Critical→…): {np.array2string(th, precision=6)}")
        print(f"(참고) train score min/max: {train_sorted_asc.min():.6f} / {train_sorted_asc.max():.6f}")

    print("\n[Test] 실제 Severity와 비교")
    t_test_sev_start = time.perf_counter()
    report_metrics(
        y_test.values,
        pred_test,
        f"Test (mode={test_mode})",
        ordinal_severity_metrics=ordinal_severity_metrics,
    )
    t_test_sev_end = time.perf_counter()

    m_val = evaluate_ranking_all(yr_val, s_val, qid_val)
    m_test = evaluate_ranking_all(yr_test, s_test, qid_test)

    out = x_test.copy()
    out[TARGET_COL] = y_test.values
    out["anomaly_score"] = s_test
    out["pred_severity"] = pred_test
    outp = Path(csv_path).with_name(Path(csv_path).stem + f"_cls_{model_name}_{test_mode}.csv")
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
    p.add_argument("--model", choices=["random_forest", "lightgbm"], default="random_forest")
    #p.add_argument("--test-mode", choices=["train_thresholds", "test_oracle_ratio"], default="train_thresholds")
    '''
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--group-size", type=int, default=64)
    '''
    p.add_argument(
        "--test-mode",
        choices=["train_thresholds", "test_oracle_ratio", "train_score_relevance_0_3"],
        default=None,
        help="미지정 시 experiment_config.yaml의 evaluation.test_mode 사용.",
    )
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    test_mode = resolve_test_mode(cfg, args.test_mode)
    csv_path = cfg["data"]["csv"]
    test_size = float(cfg["split"]["test_size"])
    val_size = float(cfg["split"]["val_size"])
    random_state = int(cfg["split"]["random_state"])
    group_size = int(cfg["ranking"]["group_size"])
    log_f = None
    old_out, old_err = sys.stdout, sys.stderr
    if not args.no_log:
        log_path = Path(args.log) if args.log else default_log_path("train_severity_cls", args.model, test_mode)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")
        sys.stdout = TeeIO(old_out, log_f)
        sys.stderr = TeeIO(old_err, log_f)
        print(f"[로그 파일] {log_path.resolve()}", flush=True)

    try:
        run(
            csv_path,
            args.model,
            test_mode,
            test_size,
            val_size,
            random_state,
            group_size,
            include_categorical_columns=bool(cfg["features"]["include_categorical_columns"]),
            ordinal_severity_metrics=bool(cfg["evaluation"]["ordinal_severity_metrics"]),
            qid_mode=str(cfg["ranking"]["qid_mode"]),
            global_qid=int(cfg["ranking"]["global_qid"]),
            split_mode=str(cfg["split"]["mode"]),
        )
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()


