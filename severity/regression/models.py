import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]


def _safe_import_xgb_regressor():
    removed = []
    for p in list(sys.path):
        try:
            rp = Path(p).resolve()
        except Exception:
            continue
        if rp == ROOT or rp == (ROOT / "xgboost"):
            removed.append(p)
            sys.path.remove(p)
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    finally:
        for p in reversed(removed):
            if p not in sys.path:
                sys.path.insert(0, p)


class XGBoostRegressorScoreModel:
    def __init__(self, random_state: int = 42):
        try:
            XGBRegressor = _safe_import_xgb_regressor()
        except Exception as e:
            raise RuntimeError("xgboost 패키지가 필요합니다. `pip install xgboost` 후 실행하세요.") from e
        self.model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x_train: np.ndarray, y_train_rel: np.ndarray, x_val: np.ndarray, y_val_rel: np.ndarray) -> None:
        self.model.fit(
            x_train,
            y_train_rel,
            eval_set=[(x_val, y_val_rel)],
            verbose=False,
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float64)


class LinearRegressionScoreModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x_train: np.ndarray, y_train_rel: np.ndarray, x_val: np.ndarray, y_val_rel: np.ndarray) -> None:
        # val은 인터페이스 통일용(LinearRegression은 early stopping 없음)
        _ = (x_val, y_val_rel)
        self.model.fit(x_train, y_train_rel)

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float64)


class KNNRegressorScoreModel:
    def __init__(self, *, n_neighbors: int = 5):
        self.model = KNeighborsRegressor(n_neighbors=int(n_neighbors))

    def fit(self, x_train: np.ndarray, y_train_rel: np.ndarray, x_val: np.ndarray, y_val_rel: np.ndarray) -> None:
        _ = (x_val, y_val_rel)
        self.model.fit(x_train, y_train_rel)

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float64)


class DecisionTreeRegressorScoreModel:
    def __init__(self, *, random_state: int = 42):
        self.model = DecisionTreeRegressor(random_state=int(random_state))

    def fit(self, x_train: np.ndarray, y_train_rel: np.ndarray, x_val: np.ndarray, y_val_rel: np.ndarray) -> None:
        _ = (x_val, y_val_rel)
        self.model.fit(x_train, y_train_rel)

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float64)


def build_regression_model(model_name: str, random_state: int = 42):
    if model_name == "linear_regression":
        return LinearRegressionScoreModel()
    if model_name == "knn_regressor":
        return KNNRegressorScoreModel()
    if model_name == "decision_tree_regressor":
        return DecisionTreeRegressorScoreModel(random_state=random_state)
    raise ValueError(f"지원하지 않는 regression 모델: {model_name}")

