"""
L2R 랭킹 점수로 anomaly score를 만든 뒤, experiment_config의 test_mode로 심각도를 배정·평가한다.
피처 분리·전처리는 prepare_splits / build_preprocessor(다른 train_severity_* 와 동일 LEAKAGE 규약).
범주형은 OrdinalEncoder(train_severity_model_ML.py 의 OneHot 과 다름).
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from severity.experiment_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_experiment_config,
    resolve_test_mode,
)
from severity.severity_rank_controlgroup import (  # noqa: E402
    apply_test_mode,
    build_preprocessor,
    prepare_splits,
    print_per_class_recall_by_true_label,
    report_metrics,
    report_metrics_cvss_numeric,
    severity_from_train_minmax_relevance,
    sort_ltr_rows_by_qid,
)
from severity.feature_importance_log import write_feature_importance_log  # noqa: E402
from severity.CVE_summary_separate.cve_schema import TARGET_COL_CVSS, relevance_for_xgb_ranker
from severity.severity_schema import LABEL_ORDER_DESC, TARGET_COL  # noqa: E402

L2R_DIR = ROOT / "L2R"
L2R_SAVE_CHECKPOINT = L2R_DIR / "save_checkpoint"
L2R_SAVE_SUBDIRS = (
    "listnet",
    "listmle",
    "xgboost",
    "lambdarank",
    "ranknet",
    "lambdamart",
)


def ensure_l2r_save_checkpoint_dirs() -> Path:
    """L2R/save_checkpoint 및 하위 모델 폴더를 만든다 (L2R 코드의 상대 경로 저장과 호환)."""
    L2R_SAVE_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    for sub in L2R_SAVE_SUBDIRS:
        (L2R_SAVE_CHECKPOINT / sub).mkdir(parents=True, exist_ok=True)
    return L2R_SAVE_CHECKPOINT


class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            if hasattr(s, "flush"):
                s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            if hasattr(s, "flush"):
                s.flush()

    def isatty(self) -> bool:
        return False
if str(L2R_DIR) not in sys.path:
    sys.path.insert(0, str(L2R_DIR))

from ListNet import train_listnet  # noqa: E402
from ListMLE import train_listmle  # noqa: E402
from metrics import evaluate_all  # noqa: E402
from data_utils import scale_data  # noqa: E402

try:
    from XGBoost_Rank import train_xgb  # noqa: E402
except Exception:
    train_xgb = None

try:
    from RankNet import RankNet  # noqa: E402
except Exception:
    RankNet = None  # type: ignore[misc, assignment]

try:
    import importlib

    _lambda_rank_mod = importlib.import_module("lambdaRank")
    LambdaRank = _lambda_rank_mod.LambdaRank  # noqa: E402
except Exception:
    LambdaRank = None  # type: ignore[misc, assignment]

try:
    from LambdaMART import LambdaMART  # noqa: E402
except Exception:
    LambdaMART = None  # type: ignore[misc, assignment]

def fit_transform_xy(x_train, x_val, x_test):
    pre = build_preprocessor(x_train)
    xt = pre.fit_transform(x_train)
    xv = pre.transform(x_val)
    xs = pre.transform(x_test)
    if hasattr(xt, "toarray"):
        xt, xv, xs = xt.toarray(), xv.toarray(), xs.toarray()
    return np.asarray(xt, dtype=np.float32), np.asarray(xv, dtype=np.float32), np.asarray(xs, dtype=np.float32), pre


# L2R/ 네이티브 .npy 규약과 동일해야 함: L2R/data_utils.load_data (y=[:,0], qid=[:,1], X=[:,2:])
# RankNet·lambdaRank·LambdaMART 등이 암묵적으로 이 순서로 슬라이스함.
L2R_COL_RELEVANCE = 0
L2R_COL_QID = 1
L2R_COL_FEATURE_START = 2


def l2r_train_matrix(y_rel: np.ndarray, qid: np.ndarray, X: np.ndarray) -> np.ndarray:
    """RankNet / LambdaRank / LambdaMART / (내부 지표용) stacked 행렬. 컬럼 순서 바꾸지 말 것."""
    y_rel = np.asarray(y_rel, dtype=np.float32).ravel()
    qid = np.asarray(qid, dtype=np.float32).ravel()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or len(y_rel) != len(X) or len(qid) != len(X):
        raise ValueError("l2r_train_matrix: y_rel, qid 길이와 X 행 수가 같아야 합니다.")
    n, d = X.shape
    out = np.empty((n, L2R_COL_FEATURE_START + d), dtype=np.float32)
    out[:, L2R_COL_RELEVANCE] = y_rel
    out[:, L2R_COL_QID] = qid
    out[:, L2R_COL_FEATURE_START:] = X
    return out


def l2r_infer_matrix(X: np.ndarray) -> np.ndarray:
    """LambdaRank/LambdaMART predict가 기대하는 전폭 행렬(relevance·qid는 더미, 피처는 [:,2:])."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("l2r_infer_matrix: X는 2차원이어야 합니다.")
    n, d = X.shape
    out = np.zeros((n, L2R_COL_FEATURE_START + d), dtype=np.float32)
    out[:, L2R_COL_FEATURE_START:] = X
    return out


def train_ranknet_local(training_data: np.ndarray, epochs: int, lr: float = 0.01):
    if RankNet is None:
        raise RuntimeError("RankNet 모듈을 불러오지 못했습니다.")
    n_feature = int(training_data.shape[1] - 2)
    trainer = RankNet(n_feature, 512, 256, epochs, lr, plot=False)
    trainer.fit(training_data)
    return trainer


def train_lambdarank_local(training_data: np.ndarray, epochs: int, lr: float = 0.001):
    if LambdaRank is None:
        raise RuntimeError("lambdaRank 모듈을 불러오지 못했습니다.")
    n_feature = int(training_data.shape[1] - 2)
    trainer = LambdaRank(training_data, n_feature, 512, 256, epochs, lr)
    trainer.fit()
    return trainer


def train_lambdamart_local(training_data: np.ndarray, n_trees: int, lr: float = 0.1):
    if LambdaMART is None:
        raise RuntimeError("LambdaMART 모듈을 불러오지 못했습니다.")
    trainer = LambdaMART(training_data, number_of_trees=n_trees, lr=lr)
    trainer.fit()
    return trainer


def predict_ranknet_scores(trainer: object, X: np.ndarray) -> np.ndarray:
    inner = trainer.model
    inner.eval()
    with torch.no_grad():
        out = inner.model(torch.tensor(X, dtype=torch.float32))
        return out.numpy().astype(np.float64).ravel()


def class_fractions(y: pd.Series) -> np.ndarray:
    vc = y.value_counts()
    fr = np.array([float(vc.get(lbl, 0)) / len(y) for lbl in LABEL_ORDER_DESC], dtype=np.float64)
    return fr


def allocate_counts(n: int, fr: np.ndarray) -> np.ndarray:
    fr = np.asarray(fr, dtype=np.float64)
    fr = fr / fr.sum()
    raw = np.floor(n * fr).astype(int)
    rem = n - int(raw.sum())
    for _ in range(rem):
        deficit = n * fr - raw.astype(np.float64)
        raw[int(np.argmax(deficit))] += 1
    return raw


def assign_top_scores(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    labels = np.empty(len(scores), dtype=object)
    i = 0
    for lbl, cnt in zip(LABEL_ORDER_DESC, counts):
        labels[order[i : i + cnt]] = lbl
        i += cnt
    return labels


def boundaries_from_train_sorted(train_scores_desc: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cum = np.cumsum([0] + list(counts))
    th = []
    for k in range(3):
        a = train_scores_desc[cum[k + 1] - 1]
        b = train_scores_desc[cum[k + 1]]
        th.append((a + b) / 2.0)
    th = np.array(th, dtype=np.float64)
    train_sorted_asc = train_scores_desc[::-1].copy()
    return th, train_sorted_asc


def assign_by_thresholds(scores: np.ndarray, thresholds_desc: np.ndarray) -> np.ndarray:
    t_ch, t_hm, t_ml = thresholds_desc[0], thresholds_desc[1], thresholds_desc[2]
    out = np.empty(len(scores), dtype=object)
    out[scores >= t_ch] = "Critical"
    out[(scores < t_ch) & (scores >= t_hm)] = "High"
    out[(scores < t_hm) & (scores >= t_ml)] = "Medium"
    out[scores < t_ml] = "Low"
    return out


def train_listnet_local(X, y_rel, qid, X_val, y_val_rel, qid_val, epochs: int, lr: float, patience: int):
    return train_listnet(
        X,
        y_rel,
        qid,
        X_val,
        y_val_rel,
        qid_val,
        epochs=epochs,
        lr=lr,
        patience=epochs,
    )


def train_listmle_local(X, y_rel, qid, X_val, y_val_rel, qid_val, epochs: int, lr: float, patience: int):
    return train_listmle(
        X,
        y_rel,
        qid,
        X_val,
        y_val_rel,
        qid_val,
        epochs=epochs,
        lr=lr,
        patience=epochs,
    )


def predict_torch(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)

'''
def report_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    labels = [l for l in LABEL_ORDER_DESC if l in np.unique(y_true) or l in np.unique(y_pred)]
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro precision: {precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print(f"Macro recall: {recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print(f"Weighted precision: {precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0):.4f}")
    print(f"Weighted recall: {recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))
'''


def run(
    csv_path: str,
    model_name: str,
    test_mode: str,
    test_size: float,
    val_size: float,
    random_state: int,
    epochs: int,
    *,
    include_categorical_columns: bool,
    ordinal_severity_metrics: bool,
    qid_mode: str,
    global_qid: int,
    split_mode: str,
    label_mode: str = "severity",
) -> None:
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
        qid_train,
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
        label_mode=label_mode,
    )

    t_train_val_start = time.perf_counter()
    xt, xv, xs, pre = fit_transform_xy(x_train, x_val, x_test)
    xt, xv, xs = scale_data(xt, xv, xs)
    # 메모리 절약: 스케일링 결과(기본 float64)를 float32로 고정
    xt = np.asarray(xt, dtype=np.float32)
    xv = np.asarray(xv, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)

    if label_mode != "cvss":
        fr_train = class_fractions(y_train)
        counts_train = np.array([y_train.value_counts().get(lbl, 0) for lbl in LABEL_ORDER_DESC], dtype=int)
    else:
        fr_train = None
        counts_train = np.zeros(4, dtype=int)

    ensure_l2r_save_checkpoint_dirs()
    cwd = os.getcwd()
    model = None
    train_mat = l2r_train_matrix(yr_train, qid_train, xt)
    # train 행 순서는 time_ordered/stratified 분할에 따라 CSV 시간 순과 다를 수 있음
    with np.printoptions(suppress=True, precision=4, linewidth=200):
        print(train_mat[0:4, :])
    try:
        os.chdir(L2R_DIR)
        if model_name == "bm25":
            pass
        elif model_name == "listnet":
            model = train_listnet_local(
                xt, yr_train, qid_train, xv, yr_val, qid_val, epochs=epochs, lr=0.001, patience=epochs
            )
        elif model_name == "listmle":
            model = train_listmle_local(
                xt, yr_train, qid_train, xv, yr_val, qid_val, epochs=epochs, lr=0.001, patience=epochs
            )
        elif model_name == "xgboost":
            if train_xgb is None:
                raise RuntimeError("XGBoost_Rank import 실패. xgboost 설치 여부를 확인하세요.")
            xt_r, yr_tr, q_tr = sort_ltr_rows_by_qid(xt, yr_train, qid_train)
            xv_r, yr_vr, q_vr = sort_ltr_rows_by_qid(xv, yr_val, qid_val)

            yr_tr_fit = relevance_for_xgb_ranker(yr_tr, label_mode=label_mode)
            yr_vr_fit = relevance_for_xgb_ranker(yr_vr, label_mode=label_mode)
            model = train_xgb(xt_r, yr_tr_fit, q_tr, xv_r, yr_vr_fit, q_vr)

            print(yr_tr.min(), yr_tr.max(), yr_tr.dtype)
            print(yr_vr.min(), yr_vr.max(), yr_vr.dtype)
            model = train_xgb(xt_r, yr_tr, q_tr, xv_r, yr_vr, q_vr)

        elif model_name == "ranknet":
            model = train_ranknet_local(train_mat, epochs=epochs, lr=0.01)
        elif model_name == "lambdarank":
            model = train_lambdarank_local(train_mat, epochs=epochs, lr=0.001)
        elif model_name == "lambdamart":
            model = train_lambdamart_local(train_mat, n_trees=max(1, int(epochs)), lr=0.1)
        else:
            raise ValueError(f"지원하지 않는 model: {model_name}")
    finally:
        os.chdir(cwd)
    t_train_val_end = time.perf_counter()

    if model_name != "bm25":
        fi_path = write_feature_importance_log(
            pre,
            model,
            prefix="train_severity_l2r",
            model_name=model_name,
            test_mode=test_mode,
            X_reference=xt,
            random_state=random_state,
        )
        print(f"\n[피처 중요도 로그] {fi_path.resolve()}", flush=True)

    t_eval_start = time.perf_counter()
    if model_name == "bm25":
        s_train = xt[:, 0].astype(np.float64)
        s_val = xv[:, 0].astype(np.float64)
        s_test = xs[:, 0].astype(np.float64)
    elif model_name in ("listnet", "listmle"):
        s_train = predict_torch(model, xt)
        s_val = predict_torch(model, xv)
        s_test = predict_torch(model, xs)
    elif model_name == "ranknet":
        s_train = predict_ranknet_scores(model, xt)
        s_val = predict_ranknet_scores(model, xv)
        s_test = predict_ranknet_scores(model, xs)
    elif model_name in ("lambdarank", "lambdamart"):
        s_train = np.asarray(model.predict(l2r_infer_matrix(xt)), dtype=np.float64)
        s_val = np.asarray(model.predict(l2r_infer_matrix(xv)), dtype=np.float64)
        s_test = np.asarray(model.predict(l2r_infer_matrix(xs)), dtype=np.float64)
    elif model_name == "xgboost":
        s_train = model.predict(xt).astype(np.float64)
        s_val = model.predict(xv).astype(np.float64)
        s_test = model.predict(xs).astype(np.float64)
    else:
        raise ValueError(f"지원하지 않는 model: {model_name}")

    if label_mode == "cvss":
        th = np.zeros(3, dtype=np.float64)
        train_sorted_asc = np.array([0.0, 1.0], dtype=np.float64)
        pred_val = apply_test_mode(
            test_mode, s_val, y_val, th, s_train=s_train, label_mode="cvss", y_train=y_train
        )
        pred_train_assign = apply_test_mode(
            test_mode, s_train, y_train, th, s_train=s_train, label_mode="cvss", y_train=y_train
        )
        print(
            f"\n[검증] L2R 점수→CVSS 매핑 MAE(참고): "
            f"{np.mean(np.abs(pred_val - y_val.values.astype(np.float64))):.4f}"
        )
        train_metrics_name = "Train (L2R score→CVSS)"
        print("\n[검증] Validation (CVSS)")
        report_metrics_cvss_numeric(y_val.values, pred_val, "Validation")
        print("\n[Train] CVSS 예측 vs 정답")
        report_metrics_cvss_numeric(y_train.values, pred_train_assign, train_metrics_name)
        pred_test = apply_test_mode(
            test_mode, s_test, y_test, th, s_train=s_train, label_mode="cvss", y_train=y_train
        )
        print(
            f"\n[CVSS] train L2R 점수 min/max: {float(np.min(s_train)):.6f} / {float(np.max(s_train)):.6f} "
            f"(train CVSS 범위로 선형 매핑)"
        )
        print("\n[Test] CVSS 예측 vs 정답")
        t_test_severity_start = time.perf_counter()
        report_metrics_cvss_numeric(y_test.values, pred_test, f"Test (mode={test_mode})")
        t_test_severity_end = time.perf_counter()
    else:
        train_desc = np.sort(s_train)[::-1]
        th, train_sorted_asc = boundaries_from_train_sorted(train_desc, counts_train)

        if test_mode == "train_thresholds":
            pred_val = assign_by_thresholds(s_val, th)
            pred_train_assign = assign_by_thresholds(s_train, th)
            acc_val = accuracy_score(y_val.values, pred_val)
            print(f"\n[검증] train score 경계(threshold) 기준 분류 정확도(참고): {acc_val:.4f}")
            train_metrics_name = "Train (train_thresholds)"
        elif test_mode == "train_score_relevance_0_3":
            pred_val = severity_from_train_minmax_relevance(s_val, s_train)
            pred_train_assign = severity_from_train_minmax_relevance(s_train, s_train)
            acc_val = accuracy_score(y_val.values, pred_val)
            print(
                f"\n[검증] train min~max→[0,3] relevance 반올림 기준 분류 정확도(참고): {acc_val:.4f}"
            )
            train_metrics_name = "Train (train_score_relevance_0_3)"
        else:
            counts_val = allocate_counts(len(y_val), fr_train)
            pred_val = assign_top_scores(s_val, counts_val)
            acc_val = accuracy_score(y_val.values, pred_val)
            print(f"\n[검증] train 비율로 배정한 분류 정확도(early stopping용 참고): {acc_val:.4f}")
            pred_train_assign = assign_top_scores(s_train, counts_train)
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
            pred_train_assign,
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
        t_test_severity_start = time.perf_counter()
        report_metrics(
            y_test.values,
            pred_test,
            f"Test (mode={test_mode})",
            ordinal_severity_metrics=ordinal_severity_metrics,
        )
        t_test_severity_end = time.perf_counter()

    m_val = evaluate_all(yr_val, s_val, qid_val)
    m_test = evaluate_all(yr_test, s_test, qid_test)

    out = x_test.copy()
    if label_mode == "cvss":
        out[TARGET_COL_CVSS] = yr_test
        out["pred_cvss"] = pred_test
    else:
        out[TARGET_COL] = y_test.values
        out["pred_severity"] = pred_test
    out["anomaly_score"] = s_test
    outp = Path(csv_path).with_name(Path(csv_path).stem + f"_l2r_{model_name}_{test_mode}.csv")
    out.to_csv(outp, index=False)
    t_eval_end = time.perf_counter()
    t_total_end = time.perf_counter()

    dt_io = t_train_val_start - t_total_start
    dt_train_val = t_train_val_end - t_train_val_start
    dt_eval = t_eval_end - t_eval_start
    dt_test_severity = t_test_severity_end - t_test_severity_start
    dt_total = t_total_end - t_total_start

    print(f"\n저장: {outp}")
    print("\n=== 소요 시간 ===")
    print(f"CSV 로드·train/val/test 분할: {dt_io:.3f} s")
    print(f"전처리(인코딩·스케일) + 학습(train∪val L2R): {dt_train_val:.3f} s")
    print(f"추론·경계·Train·Val·Test 평가·저장(전수): {dt_eval:.3f} s")
    print(f"  └ Test severity 정답 비교(report_metrics)만: {dt_test_severity:.3f} s")
    print(f"전체: {dt_total:.3f} s")
    rel_note = "relevance=CVSS 수치" if label_mode == "cvss" else "relevance=0..3"
    print(f"\n=== 랭킹 지표 MRR (L2R/metrics, {rel_note}) ===")
    print(f"Validation MRR: {m_val['MRR']:.6f}")
    print(f"Test MRR:       {m_test['MRR']:.6f}")


def _default_log_path(model: str, test_mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SEVERITY_DIR / f"train_severity_l2r_{model}_{test_mode}_{ts}.log"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"실험 YAML 경로 (기본: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="NAME",
        help="experiment_config.yaml 의 profiles 키 (예: logging, cve). 미지정 시 YAML active_profile.",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=[
            "listnet",
            "listmle",
            "xgboost",
            "ranknet",
            "lambdarank",
            "lambdamart",
            "bm25",
        ],
        default="listnet",
        help="L2R/ 폴더 모델. bm25는 학습 없이 스케일된 피처의 첫 열을 점수로 사용(L2R/BM25.py와 동일 취지).",
    )
    p.add_argument(
        "--test-mode",
        type=str,
        choices=["train_thresholds", "test_oracle_ratio", "train_score_relevance_0_3"],
        default=None,
        help="미지정 시 experiment_config.yaml의 evaluation.test_mode 사용.",
    )
    '''
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2, help="train+val+test=원본에서, val은 (1-test-size)*val-size")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=35)
    '''
    p.add_argument(
        "--log",
        type=str,
        default=None,
        metavar="PATH",
        help="로그 파일 경로. 미지정 시 severity 폴더에 자동 생성",
    )
    p.add_argument("--no-log", action="store_true", help="파일 로그를 쓰지 않음")
    args = p.parse_args()

    cfg_resolved = Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH.resolve()
    cfg = load_experiment_config(args.config, profile=args.profile)
    test_mode = resolve_test_mode(cfg, args.test_mode)
    csv_path = cfg["data"]["csv"]
    test_size = float(cfg["split"]["test_size"])
    val_size = float(cfg["split"]["val_size"])
    random_state = int(cfg["split"]["random_state"])
    epochs = int(cfg["epochs"]["l2r"])

    log_f = None
    old_out, old_err = sys.stdout, sys.stderr
    if not args.no_log:
        log_path = Path(args.log) if args.log else _default_log_path(args.model, test_mode)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")
        log_f.write(f"# train_severity_l2r_rank {datetime.now().isoformat()}\n")
        log_f.write(f"# model={args.model} test_mode={test_mode} csv={csv_path}\n")
        log_f.write(f"# config={cfg_resolved}\n\n")
        log_f.flush()
        sys.stdout = _TeeIO(old_out, log_f)
        sys.stderr = _TeeIO(old_err, log_f)
        print(f"[로그 파일] {log_path.resolve()}", flush=True)

    try:
        run(
            csv_path,
            args.model,
            test_mode,
            test_size,
            val_size,
            random_state,
            epochs,
            include_categorical_columns=bool(cfg["features"]["include_categorical_columns"]),
            ordinal_severity_metrics=bool(cfg["evaluation"]["ordinal_severity_metrics"]),
            qid_mode=str(cfg["ranking"]["qid_mode"]),
            global_qid=int(cfg["ranking"]["global_qid"]),
            split_mode=str(cfg["split"]["mode"]),
            label_mode=str(cfg["data"].get("label_mode", "severity")),
        )
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        if log_f is not None:
            log_f.close()
