"""
심각도 랭킹 실험 공통 유틸(분할, 전처리, 점수→심각도, 랭킹 지표, 로깅).
이 모듈이 단일 소스이다.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = Path(__file__).resolve().parent

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


class TeeIO:
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


def default_log_path(prefix: str, model: str, test_mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SEVERITY_DIR / f"{prefix}_{model}_{test_mode}_{ts}.log"


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
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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
    scaler = StandardScaler()
    xt = scaler.fit_transform(xt)
    xv = scaler.transform(xv)
    xs = scaler.transform(xs)
    return (
        np.asarray(xt, dtype=np.float32),
        np.asarray(xv, dtype=np.float32),
        np.asarray(xs, dtype=np.float32),
        pre,
        scaler,
    )


def prepare_splits(csv_path: str, test_size: float, val_size: float, random_state: int):
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
    return x_train, x_val, x_test, y_train, y_val, y_test, yr_train, yr_val, yr_test


def make_qid(n: int, group_size: int) -> np.ndarray:
    return (np.arange(n, dtype=np.int64) // group_size).astype(np.int64)


def group_by_qid(qid: np.ndarray) -> dict:
    groups = {}
    for i, q in enumerate(qid):
        groups.setdefault(int(q), []).append(i)
    return groups


def class_fractions(y: pd.Series) -> np.ndarray:
    vc = y.value_counts()
    return np.array([float(vc.get(lbl, 0)) / len(y) for lbl in LABEL_ORDER_DESC], dtype=np.float64)


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
    return np.array(th, dtype=np.float64), train_scores_desc[::-1].copy()


def assign_by_thresholds(scores: np.ndarray, thresholds_desc: np.ndarray) -> np.ndarray:
    t_ch, t_hm, t_ml = thresholds_desc[0], thresholds_desc[1], thresholds_desc[2]
    out = np.empty(len(scores), dtype=object)
    out[scores >= t_ch] = "Critical"
    out[(scores < t_ch) & (scores >= t_hm)] = "High"
    out[(scores < t_hm) & (scores >= t_ml)] = "Medium"
    out[scores < t_ml] = "Low"
    return out


def dcg_k(y, k):
    y = np.asarray(y)[:k]
    return np.sum((2**y - 1) / np.log2(np.arange(2, len(y) + 2)))


def ndcg_k(y_true, y_pred, k):
    idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[idx]
    dcg = dcg_k(y_sorted, k)
    idcg = dcg_k(sorted(y_true, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(y_true, y_pred):
    idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[idx]
    for i, y in enumerate(y_sorted):
        if y > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(y_true, y_pred):
    idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[idx]
    hits, score = 0, 0.0
    for i, y in enumerate(y_sorted):
        if y > 0:
            hits += 1
            score += hits / (i + 1)
    return score / hits if hits > 0 else 0.0


def evaluate_ranking_all(y_rel: np.ndarray, pred_score: np.ndarray, qid: np.ndarray) -> dict:
    groups = group_by_qid(qid)
    res = {"NDCG@1": [], "NDCG@5": [], "NDCG@10": [], "MAP": [], "MRR": []}
    for g in groups.values():
        y_g = y_rel[g]
        p_g = pred_score[g]
        res["NDCG@1"].append(ndcg_k(y_g, p_g, 1))
        res["NDCG@5"].append(ndcg_k(y_g, p_g, 5))
        res["NDCG@10"].append(ndcg_k(y_g, p_g, 10))
        res["MAP"].append(average_precision(y_g, p_g))
        res["MRR"].append(mrr(y_g, p_g))
    return {k: float(np.mean(v)) for k, v in res.items()}


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


def apply_test_mode(test_mode: str, s_test: np.ndarray, y_test: pd.Series, th: np.ndarray) -> np.ndarray:
    if test_mode == "train_thresholds":
        return assign_by_thresholds(s_test, th)
    if test_mode == "test_oracle_ratio":
        counts_te = np.array([y_test.value_counts().get(lbl, 0) for lbl in LABEL_ORDER_DESC], dtype=int)
        return assign_top_scores(s_test, counts_te)
    raise ValueError(test_mode)
