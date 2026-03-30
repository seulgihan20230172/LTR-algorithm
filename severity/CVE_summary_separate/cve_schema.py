"""CVE CSV(cve_with_meaning*.csv)용 열 이름·누수 제외.

라벨은 CVSS 실수 그대로 사용한다(4단계 문자열로 버킷팅하지 않음).
train_test_split 층화만 분위(qcut)로 이산 라벨을 만든다 — 이는 라벨이 아니라 분할 균형용.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL_CVSS = "cvss"
MOD_DATE_COL = "mod_date"
PUB_DATE_COL = "pub_date"
CVE_ID_COL = "cve_id"
SUMMARY_COL = "summary"

# X에서 제외: 라벨·식별·날짜 원문·요약 원문
LEAKAGE_COLS_CVE: list[str] = [
    TARGET_COL_CVSS,
    CVE_ID_COL,
    MOD_DATE_COL,
    PUB_DATE_COL,
    SUMMARY_COL,
    "Unnamed: 0",
]


def relevance_for_xgb_ranker(yr: np.ndarray, *, label_mode: str) -> np.ndarray:
    """XGBoost ``rank:ndcg`` 학습 시 기대하는 **정수** relevance.

    메트릭·ListNet 등에는 여전히 float CVSS(또는 0~3 relevance)를 쓰고,
    XGBRanker ``fit`` 에 넣는 라벨만 여기서 반올림한다 (순서 관계 유지).
    """
    y = np.asarray(yr, dtype=np.float64).ravel()
    if label_mode == "cvss":
        y = np.clip(y, 0.0, 10.0)
    else:
        y = np.clip(y, 0.0, 3.0)
    return np.rint(y).astype(np.int32)


def stratify_codes_for_split_cvss(y_cvss: pd.Series) -> np.ndarray | None:
    """train_test_split(stratify=)용 분위 코드. 라벨은 여전히 숫자 CVSS이며, 이 값은 분할에만 사용"""
    n = len(y_cvss)
    if n < 20:
        return None
    q = min(10, max(2, n // 20))
    try:
        cats = pd.qcut(y_cvss.astype(float), q=q, duplicates="drop")
        return cats.cat.codes.to_numpy()
    except (ValueError, TypeError):
        return None
