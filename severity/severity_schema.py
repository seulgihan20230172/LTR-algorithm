"""심각도 실험 CSV·라벨 규약(열 이름, 클래스 순서, relevance). train 스크립트와 유틸이 동일 값을 쓰도록 공유한다."""

TARGET_COL = "Severity"
QID_COL = "Anomaly_ID"  # 랭킹 qid (행 단위 이상 건 식별자)
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
