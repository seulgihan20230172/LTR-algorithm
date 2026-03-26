import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
LABEL_ORDER = ["Low", "Medium", "High"]
LABEL_TO_INT = {"Low": 0, "Medium": 1, "High": 2}
INT_TO_LABEL = {0: "Low", 1: "Medium", 2: "High"}


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].copy()
    x = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    return x, y


def build_pipeline(x: pd.DataFrame) -> Pipeline:
    cat_cols = x.select_dtypes(include=["object"]).columns.tolist()
    num_cols = x.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def severity_score_from_proba(proba: np.ndarray, class_order: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(class_order)}
    low = proba[:, idx["Low"]]
    med = proba[:, idx["Medium"]]
    high = proba[:, idx["High"]]
    return med + 2.0 * high + 0.0 * low


def predict_with_thresholds(score: np.ndarray, t_low_med: float, t_med_high: float) -> np.ndarray:
    pred_int = np.where(score < t_low_med, 0, np.where(score < t_med_high, 1, 2))
    return np.array([INT_TO_LABEL[i] for i in pred_int])


def find_best_thresholds(score: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    unique_scores = np.unique(np.round(score, 6))
    if len(unique_scores) < 3:
        t1 = np.quantile(score, 1 / 3)
        t2 = np.quantile(score, 2 / 3)
        pred = predict_with_thresholds(score, t1, t2)
        return float(t1), float(t2), f1_score(y_true, pred, average="macro")

    candidates = np.quantile(unique_scores, np.linspace(0.05, 0.95, 120))
    candidates = np.unique(np.round(candidates, 6))

    best_f1 = -1.0
    best_t1, best_t2 = 0.5, 1.5
    for t1, t2 in combinations(candidates, 2):
        if t1 >= t2:
            continue
        pred = predict_with_thresholds(score, t1, t2)
        score_f1 = f1_score(y_true, pred, average="macro")
        if score_f1 > best_f1:
            best_f1 = score_f1
            best_t1, best_t2 = float(t1), float(t2)

    return best_t1, best_t2, best_f1


def class_score_ranges(score: np.ndarray, y_true: np.ndarray) -> dict:
    out = {}
    for label in LABEL_ORDER:
        mask = y_true == label
        s = score[mask]
        if len(s) == 0:
            out[label] = {"min": None, "q25": None, "q50": None, "q75": None, "max": None}
        else:
            out[label] = {
                "min": float(np.min(s)),
                "q25": float(np.quantile(s, 0.25)),
                "q50": float(np.quantile(s, 0.50)),
                "q75": float(np.quantile(s, 0.75)),
                "max": float(np.max(s)),
            }
    return out


def run(csv_path: str, test_size: float, random_state: int) -> None:
    df = pd.read_csv(csv_path)
    x, y = split_features(df)

    if not set(LABEL_ORDER).issubset(set(y.unique())):
        raise ValueError(f"Severity 라벨은 {LABEL_ORDER}를 포함해야 합니다. 현재: {sorted(y.unique().tolist())}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipe = build_pipeline(x_train)
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    model = pipe.named_steps["model"]
    class_order = model.classes_.tolist()
    proba_test = pipe.predict_proba(x_test)
    sev_score = severity_score_from_proba(proba_test, class_order)

    t1, t2, tuned_f1 = find_best_thresholds(sev_score, y_test.to_numpy())
    y_pred_by_threshold = predict_with_thresholds(sev_score, t1, t2)

    print("\n=== 기본 분류 성능(RandomForest direct class prediction) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, labels=LABEL_ORDER, digits=4))
    print("[Confusion Matrix] (rows=true, cols=pred; order: Low, Medium, High)")
    print(confusion_matrix(y_test, y_pred, labels=LABEL_ORDER))

    print("\n=== Severity 경계선 식별(연속 severity score 기반) ===")
    print("severity_score = P(Medium) + 2*P(High)")
    print(f"Low/Medium 경계선: score < {t1:.6f} -> Low")
    print(f"Medium/High 경계선: score >= {t2:.6f} -> High")
    print(f"그 사이: Medium")
    print(f"Threshold 기반 Macro F1: {tuned_f1:.4f}")
    print("\n[Threshold 방식 Classification Report]")
    print(classification_report(y_test, y_pred_by_threshold, labels=LABEL_ORDER, digits=4))
    print("[Threshold 방식 Confusion Matrix] (rows=true, cols=pred; order: Low, Medium, High)")
    print(confusion_matrix(y_test, y_pred_by_threshold, labels=LABEL_ORDER))

    ranges = class_score_ranges(sev_score, y_test.to_numpy())
    print("\n=== 실제 라벨별 severity_score 분포 요약(테스트셋) ===")
    for label in LABEL_ORDER:
        r = ranges[label]
        print(
            f"{label}: min={r['min']:.6f}, q25={r['q25']:.6f}, "
            f"q50={r['q50']:.6f}, q75={r['q75']:.6f}, max={r['max']:.6f}"
        )

    out_df = x_test.copy()
    out_df[TARGET_COL] = y_test.values
    out_df["pred_direct"] = y_pred
    out_df["pred_threshold"] = y_pred_by_threshold
    out_df["severity_score"] = sev_score
    out_path = csv_path.replace(".csv", "_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n예측 결과 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="severity/logging_monitoring_anomalies.csv",
        help="학습/평가할 CSV 경로",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="테스트셋 비율")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()
    run(args.csv, args.test_size, args.random_state)
