# ML target (detection layer)


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
CANONICAL_LABEL_ORDER = ["Low", "Medium", "High", "Critical"]


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


def get_label_order(y: pd.Series) -> list[str]:
    observed = set(y.unique().tolist())
    return [lbl for lbl in CANONICAL_LABEL_ORDER if lbl in observed]


def severity_score_from_proba(proba: np.ndarray, class_order: list[str], label_order: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(class_order)}
    weights = np.array([float(label_order.index(lbl)) for lbl in class_order], dtype=float)
    return proba @ weights


def predict_with_thresholds(score: np.ndarray, thresholds: list[float], label_order: list[str]) -> np.ndarray:
    pred_idx = np.searchsorted(np.array(thresholds, dtype=float), score, side="right")
    return np.array([label_order[i] for i in pred_idx])


def find_best_thresholds(score: np.ndarray, y_true: np.ndarray, label_order: list[str]) -> tuple[list[float], float]:
    n_classes = len(label_order)
    needed = n_classes - 1
    unique_scores = np.unique(np.round(score, 6))
    if len(unique_scores) < n_classes:
        thresholds = [
            float(np.quantile(score, i / n_classes))
            for i in range(1, n_classes)
        ]
        pred = predict_with_thresholds(score, thresholds, label_order)
        return thresholds, f1_score(y_true, pred, average="macro")

    candidates = np.quantile(unique_scores, np.linspace(0.05, 0.95, 24))
    candidates = np.unique(np.round(candidates, 6))

    best_f1 = -1.0
    best_thresholds: list[float] = []
    for comb in combinations(candidates, needed):
        thresholds = list(comb)
        pred = predict_with_thresholds(score, thresholds, label_order)
        score_f1 = f1_score(y_true, pred, average="macro")
        if score_f1 > best_f1:
            best_f1 = score_f1
            best_thresholds = [float(t) for t in thresholds]

    return best_thresholds, best_f1


def class_score_ranges(score: np.ndarray, y_true: np.ndarray, label_order: list[str]) -> dict:
    out = {}
    for label in label_order:
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
    label_order = get_label_order(y)

    if len(label_order) < 3:
        raise ValueError(f"Severity 라벨이 너무 적습니다. 현재: {sorted(y.unique().tolist())}")

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
    sev_score = severity_score_from_proba(proba_test, class_order, label_order)

    thresholds, tuned_f1 = find_best_thresholds(sev_score, y_test.to_numpy(), label_order)
    y_pred_by_threshold = predict_with_thresholds(sev_score, thresholds, label_order)

    print("\n=== 기본 분류 성능(RandomForest direct class prediction) ===")
    print(f"Labels: {label_order}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, labels=label_order, digits=4))
    print(f"[Confusion Matrix] (rows=true, cols=pred; order: {', '.join(label_order)})")
    print(confusion_matrix(y_test, y_pred, labels=label_order))

    print("\n=== Severity 경계선 식별(연속 severity score 기반) ===")
    print("severity_score = Σ[P(class) * ordinal_index(class)]")
    for i, t in enumerate(thresholds):
        left = label_order[i]
        right = label_order[i + 1]
        print(f"{left}/{right} 경계선: {t:.6f}")
    print(f"Threshold 기반 Macro F1: {tuned_f1:.4f}")
    print("\n[Threshold 방식 Classification Report]")
    print(classification_report(y_test, y_pred_by_threshold, labels=label_order, digits=4))
    print(f"[Threshold 방식 Confusion Matrix] (rows=true, cols=pred; order: {', '.join(label_order)})")
    print(confusion_matrix(y_test, y_pred_by_threshold, labels=label_order))

    ranges = class_score_ranges(sev_score, y_test.to_numpy(), label_order)
    print("\n=== 실제 라벨별 severity_score 분포 요약(테스트셋) ===")
    for label in label_order:
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
