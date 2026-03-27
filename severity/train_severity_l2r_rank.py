"""
L2R 랭킹 점수로 anomaly score를 만든 뒤, train의 4클래스 비율로 심각도를 배정·평가한다.
범주형 피처는 메모리 한계로 OrdinalEncoder 처리(train_severity_model_ML.py의 OneHot과 다름).
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
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = Path(__file__).resolve().parent
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

from ListNet import ListNet, listnet_loss  # noqa: E402
from ListMLE import listmle_loss  # noqa: E402
from metrics import evaluate_all  # noqa: E402
from data_utils import group_by_qid, scale_data  # noqa: E402

try:
    from XGBoost_Rank import train_xgb  # noqa: E402
except Exception:
    train_xgb = None

TARGET_COL = "Severity"
LEAKAGE_COLS = [
    "Anomaly_Type",
    "Severity",
    "Status",
    "Source",
    "Alert_Method",
    "Timestamp",
    "Anomaly_ID",
]
LABEL_ORDER_DESC = ["Critical", "High", "Medium", "Low"]
RELEVANCE = {"Low": 0.0, "Medium": 1.0, "High": 2.0, "Critical": 3.0}


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].copy()
    x = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    cat_cols = x.select_dtypes(include=["object"]).columns.tolist()
    num_cols = x.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def fit_transform_xy(x_train, x_val, x_test):
    pre = build_preprocessor(x_train)
    xt = pre.fit_transform(x_train)
    xv = pre.transform(x_val)
    xs = pre.transform(x_test)
    if hasattr(xt, "toarray"):
        xt, xv, xs = xt.toarray(), xv.toarray(), xs.toarray()
    return np.asarray(xt, dtype=np.float32), np.asarray(xv, dtype=np.float32), np.asarray(xs, dtype=np.float32), pre


def make_qid(n: int, group_size: int) -> np.ndarray:
    return (np.arange(n, dtype=np.int64) // group_size).astype(np.int64)


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


def train_listmle_local(X, y_rel, qid, X_val, y_val_rel, qid_val, epochs: int, lr: float, patience: int):
    model = ListNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    groups = group_by_qid(qid)
    best_state = None
    best_score = -1.0
    wait = 0
    for epoch in range(epochs):
        model.train()
        for g in groups.values():
            xg = torch.tensor(X[g], dtype=torch.float32)
            yg = torch.tensor(y_rel[g], dtype=torch.float32)
            pred = model(xg)
            loss = listmle_loss(pred, yg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        metrics = evaluate_all(y_val_rel, pred_val, qid_val)
        score = metrics["NDCG@10"]
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_listnet_local(X, y_rel, qid, X_val, y_val_rel, qid_val, epochs: int, lr: float, patience: int):
    model = ListNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    groups = group_by_qid(qid)
    best_state = None
    best_score = -1.0
    wait = 0
    for epoch in range(epochs):
        model.train()
        for g in groups.values():
            xg = torch.tensor(X[g], dtype=torch.float32)
            yg = torch.tensor(y_rel[g], dtype=torch.float32)
            pred = model(xg)
            loss = listnet_loss(pred, yg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        metrics = evaluate_all(y_val_rel, pred_val, qid_val)
        score = metrics["NDCG@10"]
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


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


def run(
    csv_path: str,
    model_name: str,
    test_mode: str,
    test_size: float,
    val_size: float,
    random_state: int,
    group_size: int,
    epochs: int,
) -> None:
    t_total_start = time.perf_counter()
    df = pd.read_csv(csv_path)
    x, y = split_features(df)
    y_rel = y.map(RELEVANCE).astype(np.float32).values

    x_temp, x_test, y_temp, y_test, yr_temp, yr_test = train_test_split(
        x,
        y,
        y_rel,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    val_ratio = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val, yr_train, yr_val = train_test_split(
        x_temp,
        y_temp,
        yr_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    t_train_val_start = time.perf_counter()
    xt, xv, xs, _ = fit_transform_xy(x_train, x_val, x_test)
    xt, xv, xs = scale_data(xt, xv, xs)

    n_tr, n_va, n_te = len(y_train), len(y_val), len(y_test)
    qid_train = make_qid(n_tr, group_size)
    qid_val = make_qid(n_va, group_size)
    qid_test = make_qid(n_te, group_size)

    fr_train = class_fractions(y_train)
    counts_train = np.array([y_train.value_counts().get(lbl, 0) for lbl in LABEL_ORDER_DESC], dtype=int)

    ensure_l2r_save_checkpoint_dirs()
    cwd = os.getcwd()
    try:
        os.chdir(L2R_DIR)
        if model_name == "listnet":
            model = train_listnet_local(
                xt, yr_train, qid_train, xv, yr_val, qid_val, epochs=epochs, lr=0.001, patience=8
            )
        elif model_name == "listmle":
            model = train_listmle_local(
                xt, yr_train, qid_train, xv, yr_val, qid_val, epochs=min(epochs, 50), lr=0.001, patience=8
            )
        elif model_name == "xgboost":
            if train_xgb is None:
                raise RuntimeError("XGBoost_Rank import 실패. xgboost 설치 여부를 확인하세요.")
            _, group = np.unique(qid_train, return_counts=True)
            _, group_val = np.unique(qid_val, return_counts=True)
            xgb_model = train_xgb(xt, yr_train, qid_train, xv, yr_val, qid_val)
            model = xgb_model
        else:
            raise ValueError(f"지원하지 않는 model: {model_name}")
    finally:
        os.chdir(cwd)
    t_train_val_end = time.perf_counter()

    t_eval_start = time.perf_counter()
    if model_name in ("listnet", "listmle"):
        s_train = predict_torch(model, xt)
        s_val = predict_torch(model, xv)
        s_test = predict_torch(model, xs)
    else:
        s_train = model.predict(xt).astype(np.float64)
        s_val = model.predict(xv).astype(np.float64)
        s_test = model.predict(xs).astype(np.float64)

    counts_val = allocate_counts(n_va, fr_train)
    pred_val = assign_top_scores(s_val, counts_val)
    acc_val = accuracy_score(y_val.values, pred_val)
    print(f"\n[검증] train 비율로 배정한 분류 정확도(early stopping용 참고): {acc_val:.4f}")

    train_desc = np.sort(s_train)[::-1]
    th, train_sorted_asc = boundaries_from_train_sorted(train_desc, counts_train)

    pred_train_assign = assign_top_scores(s_train, counts_train)
    print("\n[Train] 실제 Severity와 비교 (train 비율 배정)")
    report_metrics(y_train.values, pred_train_assign, "Train (비율 배정)")

    if test_mode == "train_thresholds":
        pred_test = assign_by_thresholds(s_test, th)
    else:
        if test_mode == "test_oracle_ratio":
            counts_te = np.array([y_test.value_counts().get(lbl, 0) for lbl in LABEL_ORDER_DESC], dtype=int)
        else:
            raise ValueError(test_mode)
        pred_test = assign_top_scores(s_test, counts_te)

    print(f"\nTrain에서 추정한 score 경계(내림차순 상위부터 Critical→…): {np.array2string(th, precision=6)}")
    print(f"(참고) train score min/max: {train_sorted_asc.min():.6f} / {train_sorted_asc.max():.6f}")

    print("\n[Test] 실제 Severity와 비교")
    t_test_severity_start = time.perf_counter()
    report_metrics(y_test.values, pred_test, f"Test (mode={test_mode})")
    t_test_severity_end = time.perf_counter()

    m_val = evaluate_all(yr_val, s_val, qid_val)
    m_test = evaluate_all(yr_test, s_test, qid_test)

    out = x_test.copy()
    out[TARGET_COL] = y_test.values
    out["anomaly_score"] = s_test
    out["pred_severity"] = pred_test
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
    print("\n=== 랭킹 지표 MRR (L2R/metrics, relevance=0..3) ===")
    print(f"Validation MRR: {m_val['MRR']:.6f}")
    print(f"Test MRR:       {m_test['MRR']:.6f}")


def _default_log_path(model: str, test_mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SEVERITY_DIR / f"train_severity_l2r_{model}_{test_mode}_{ts}.log"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="severity/logging_monitoring_anomalies.csv")
    p.add_argument("--model", type=str, choices=["listnet", "listmle", "xgboost"], default="listnet")
    p.add_argument(
        "--test-mode",
        type=str,
        choices=["train_thresholds", "test_oracle_ratio"],
        default="train_thresholds",
        help="train_thresholds: train에서 학습한 score 경계로 test 분할. "
        "test_oracle_ratio: test에서 실제 4class 개수 비율로 상위부터 배정",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2, help="train+val+test=원본에서, val은 (1-test-size)*val-size")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument(
        "--log",
        type=str,
        default=None,
        metavar="PATH",
        help="로그 파일 경로. 미지정 시 severity 폴더에 자동 생성",
    )
    p.add_argument("--no-log", action="store_true", help="파일 로그를 쓰지 않음")
    args = p.parse_args()

    log_f = None
    old_out, old_err = sys.stdout, sys.stderr
    if not args.no_log:
        log_path = Path(args.log) if args.log else _default_log_path(args.model, args.test_mode)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")
        log_f.write(f"# train_severity_l2r_rank {datetime.now().isoformat()}\n")
        log_f.write(f"# model={args.model} test_mode={args.test_mode} csv={args.csv}\n\n")
        log_f.flush()
        sys.stdout = _TeeIO(old_out, log_f)
        sys.stderr = _TeeIO(old_err, log_f)
        print(f"[로그 파일] {log_path.resolve()}", flush=True)

    try:
        run(
            args.csv,
            args.model,
            args.test_mode,
            args.test_size,
            args.val_size,
            args.random_state,
            args.group_size,
            args.epochs,
        )
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        if log_f is not None:
            log_f.close()
