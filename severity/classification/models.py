import numpy as np
from sklearn.ensemble import RandomForestClassifier

LABEL_TO_LEVEL = {"Low": 0.0, "Medium": 1.0, "High": 2.0, "Critical": 3.0}


class RandomForestScoreModel:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        _ = x_val, y_val
        self.model.fit(x_train, y_train)

    def score(self, x: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(x)
        weights = np.array([LABEL_TO_LEVEL[c] for c in self.model.classes_], dtype=np.float64)
        return proba @ weights


class LightGBMScoreModel:
    def __init__(self, random_state: int = 42):
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise RuntimeError("lightgbm 패키지가 필요합니다. `pip install lightgbm` 후 실행하세요.") from e
        self.model = LGBMClassifier(
            objective="multiclass",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=random_state,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="multi_logloss",
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(x)
        weights = np.array([LABEL_TO_LEVEL[c] for c in self.model.classes_], dtype=np.float64)
        return proba @ weights


def build_classification_model(model_name: str, random_state: int = 42):
    if model_name == "random_forest":
        return RandomForestScoreModel(random_state=random_state)
    if model_name == "lightgbm":
        return LightGBMScoreModel(random_state=random_state)
    raise ValueError(f"지원하지 않는 classification 모델: {model_name}")

