"""
심각도 랭킹 실험 공통 유틸(분할, 전처리, 점수→심각도, 랭킹 지표, 로깅).
TARGET_COL·라벨 순서 등 규약은 severity_schema 를 단일 소스로 한다.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from severity.severity_schema import (
    ANOMALY_ID_COL,
    LABEL_ORDER_DESC,
    LEAKAGE_COLS,
    RELEVANCE,
    TARGET_COL,
    TIMESTAMP_COL,
)

ROOT = Path(__file__).resolve().parents[1]
SEVERITY_DIR = Path(__file__).resolve().parent


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


def _categorical_feature_columns(x: pd.DataFrame) -> list[str]:
    """object·pandas category dtype 열만 범주형 피처로 본다."""
    return [c for c in x.columns if is_object_dtype(x[c]) or is_categorical_dtype(x[c])]


def split_features(
    df: pd.DataFrame, *, include_categorical_columns: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].copy()
    x = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    if not include_categorical_columns:
        drop_cols = _categorical_feature_columns(x)
        x = x.drop(columns=drop_cols)
    if x.shape[1] == 0:
        raise ValueError(
            "피처 열이 없습니다. leakage 제외 후 범주형까지 제외하면 숫자형 피처가 남지 않을 수 있습니다."
        )
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    cat_cols = _categorical_feature_columns(x)
    num_cols = [c for c in x.columns if c not in cat_cols]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("전처리할 피처 열이 없습니다.")
    return ColumnTransformer(transformers=transformers)


def _time_ordered_split_sizes(n: int, test_size: float, val_size: float) -> tuple[int, int, int]:
    """가장 이른 시각 → train, 그다음 → val, 가장 늦은 구간 → test. 반환 (n_train, n_val, n_test)."""
    if n < 3:
        raise ValueError("time_ordered 분할은 행이 최소 3개 이상이어야 합니다.")
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 2)
    n_rem = n - n_test
    val_ratio = val_size / (1.0 - test_size)
    n_val = max(1, int(round(n_rem * val_ratio)))
    n_val = min(n_val, n_rem - 1)
    n_train = n_rem - n_val
    assert n_train >= 1 and n_val >= 1 and n_test >= 1 and n_train + n_val + n_test == n
    return n_train, n_val, n_test


def qids_from_timestamp_hour_1h(df: pd.DataFrame) -> np.ndarray:
    """Timestamp를 1시간 단위로 내린 뒤, 시간대를 시각 순으로 0,1,2,… qid를 부여한다."""
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"qid_mode=timestamp_hour_1h 일 때 CSV에 '{TIMESTAMP_COL}' 열이 필요합니다.")
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"'{TIMESTAMP_COL}'에 파싱되지 않는 시각이 있습니다.")
    hour = ts.dt.floor("h")
    unique_hours = sorted(hour.unique())
    hour_to_qid = {h: i for i, h in enumerate(unique_hours)}
    return hour.map(hour_to_qid).astype(np.int64).to_numpy()


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


def prepare_splits(
    csv_path: str,
    test_size: float,
    val_size: float,
    random_state: int,
    *,
    include_categorical_columns: bool = True,
    qid_mode: str = "global",
    global_qid: int = 0,
    split_mode: str = "stratified_shuffle",
):
    """qid: global | anomaly_id | timestamp_hour_1h (1시간 버킷, 시각 순 연속 qid)."""
    """split_mode: stratified_shuffle(Severity 층화+무작위) | time_ordered(Timestamp 오름차순, 과거→train·중간→val·최근→test)."""
    df = pd.read_csv(csv_path)
    if split_mode == "time_ordered":
        if TIMESTAMP_COL not in df.columns:
            raise ValueError(f"split_mode=time_ordered 일 때 CSV에 '{TIMESTAMP_COL}' 열이 필요합니다.")
        ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        if ts.isna().any():
            raise ValueError(f"split_mode=time_ordered: '{TIMESTAMP_COL}'에 파싱되지 않는 시각이 있습니다.")
        df = df.iloc[np.argsort(ts.values, kind="mergesort")].reset_index(drop=True)
    elif split_mode != "stratified_shuffle":
        raise ValueError(
            f"split_mode는 'stratified_shuffle' 또는 'time_ordered' 여야 합니다: {split_mode!r}"
        )

    if qid_mode == "global":
        qid_all = np.full(len(df), int(global_qid), dtype=np.int64)
    elif qid_mode == "anomaly_id":
        if ANOMALY_ID_COL not in df.columns:
            raise ValueError(
                f"qid_mode=anomaly_id 일 때 CSV에 '{ANOMALY_ID_COL}' 열이 필요합니다."
            )
        qid_series = pd.to_numeric(df[ANOMALY_ID_COL], errors="coerce")
        if qid_series.isna().any():
            raise ValueError(f"'{ANOMALY_ID_COL}'에 숫자로 변환되지 않는 값이 있습니다.")
        qid_all = qid_series.astype(np.int64).to_numpy()
    elif qid_mode == "timestamp_hour_1h":
        qid_all = qids_from_timestamp_hour_1h(df)
    else:
        raise ValueError(
            f"qid_mode는 'global', 'anomaly_id', 'timestamp_hour_1h' 중 하나여야 합니다: {qid_mode!r}"
        )
    x, y = split_features(df, include_categorical_columns=include_categorical_columns)
    y_rel = y.map(RELEVANCE).astype(np.float32).values

    if split_mode == "stratified_shuffle":
        x_temp, x_test, y_temp, y_test, yr_temp, yr_test, q_temp, q_test = train_test_split(
            x,
            y,
            y_rel,
            qid_all,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        val_ratio = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val, yr_train, yr_val, qid_train, qid_val = train_test_split(
            x_temp,
            y_temp,
            yr_temp,
            q_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp,
        )
    else:
        n = len(x)
        n_train, n_val, n_test = _time_ordered_split_sizes(n, test_size, val_size)
        i0, i1, i2 = 0, n_train, n_train + n_val
        x_train, x_val, x_test = x.iloc[i0:i1], x.iloc[i1:i2], x.iloc[i2:]
        y_train, y_val, y_test = y.iloc[i0:i1], y.iloc[i1:i2], y.iloc[i2:]
        yr_train = y_rel[i0:i1]
        yr_val = y_rel[i1:i2]
        yr_test = y_rel[i2:]
        qid_train = qid_all[i0:i1]
        qid_val = qid_all[i1:i2]
        q_test = qid_all[i2:]

    return (
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
        q_test,
    )


def sort_ltr_rows_by_qid(X: np.ndarray, y: np.ndarray, qid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XGBoost rank:ndcg 등은 동일 qid 문서가 연속된 행에 있어야 group 크기와 일치한다."""
    qid = np.asarray(qid)
    order = np.lexsort((np.arange(len(qid), dtype=np.int64), qid))
    return X[order], y[order], qid[order]


def make_qid(n: int, group_size: int) -> np.ndarray:
    """(레거시) 인위적 쿼리 그룹. 현재는 prepare_splits 의 qid_mode를 쓴다."""
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


def relevance_vector(y: np.ndarray) -> np.ndarray:
    """심각도 라벨을 RELEVANCE 스케일(Low=0 … Critical=3)로 변환한다."""
    return np.array([RELEVANCE[str(v)] for v in y], dtype=np.float64)


def ordinal_severity_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """순서 반영: Critical에서 High 오류(거리 1)가 Low 오류(거리 3)보다 작은 페널티가 된다.

    반환: (ordinal_mae, ordinal_rmse, within_one_frac) — within_one은 |Δrelevance|≤1 비율.
    """
    t = relevance_vector(y_true)
    p = relevance_vector(y_pred)
    diff = np.abs(t - p)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    w1 = float(np.mean(diff <= 1.0))
    return mae, rmse, w1


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


def report_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    *,
    ordinal_severity_metrics: bool = False,
) -> None:
    labels = [l for l in LABEL_ORDER_DESC if l in np.unique(y_true) or l in np.unique(y_pred)]
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    if ordinal_severity_metrics:
        ord_mae, ord_rmse, within_one = ordinal_severity_errors(y_true, y_pred)
        print(
            f"Ordinal MAE (relevance 0–3): {ord_mae:.4f}  "
            f"(등간격 스케일 가정의 보조 지표; Critical→High가 Critical→Low보다 페널티 작음)"
        )
        print(f"Ordinal RMSE (relevance): {ord_rmse:.4f}")
        print(f"Within-1 severity (|pred-true|≤1 step): {within_one:.4f}")
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
