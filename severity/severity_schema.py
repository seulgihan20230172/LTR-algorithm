"""심각도 실험 CSV·라벨 규약(열 이름, 클래스 순서, relevance). train 스크립트와 유틸이 동일 값을 쓰도록 공유한다."""

TARGET_COL = "Severity"
# CSV 열 이름(이상 건 ID). LTR query id와는 별개다. global qid_mode일 때는 랭킹에 쓰이지 않고 피처에서만 leakage로 제거된다.
ANOMALY_ID_COL = "Anomaly_ID"
TIMESTAMP_COL = "Timestamp"
# X(피처)에서 제외: 타깃(Severity), 식별·시각(메타). Anomaly_Type·Status·Source·Alert_Method는 범주 피처로 포함.
LEAKAGE_COLS = [
    TARGET_COL,
    TIMESTAMP_COL,
    ANOMALY_ID_COL,
]
LABEL_ORDER_DESC = ["Critical", "High", "Medium", "Low"]
RELEVANCE = {"Low": 0.0, "Medium": 1.0, "High": 2.0, "Critical": 3.0}
