"""트리·부스팅: feature_importances_ + (선택) SHAP TreeExplainer 기반 평균 |SHAP| 요약."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer

SEVERITY_DIR = Path(__file__).resolve().parent


def subsample_reference_matrix(
    X: np.ndarray, max_rows: int, random_state: int = 42
) -> np.ndarray:
    """SHAP·설명용으로 전처리된 행렬에서 최대 max_rows행만 무작위 추출."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= max_rows:
        return X
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx]


def _importances_from_model(model_obj: object) -> tuple[np.ndarray | None, str]:
    inner = getattr(model_obj, "model", model_obj)
    if hasattr(inner, "feature_importances_"):
        return np.asarray(inner.feature_importances_, dtype=np.float64), ""
    return None, "이 모델은 sklearn 스타일 feature_importances_를 제공하지 않습니다 (AE·L2R·IsolationForest 등)."


def _shap_mean_abs_per_feature(inner: object, X_ref: np.ndarray) -> tuple[np.ndarray | None, str]:
    try:
        import shap
    except ImportError:
        return None, "shap 미설치: pip install shap"

    X_ref = np.asarray(X_ref, dtype=np.float64)
    try:
        explainer = shap.TreeExplainer(inner)
        sv = explainer.shap_values(X_ref)
    except Exception as e:
        return None, f"SHAP TreeExplainer 미지원 또는 오류: {e!r}"

    if isinstance(sv, list):
        if len(sv) == 0:
            return None, "SHAP shap_values가 빈 리스트입니다."
        ma = np.zeros(sv[0].shape[1], dtype=np.float64)
        for s in sv:
            ma += np.abs(np.asarray(s, dtype=np.float64)).mean(axis=0)
        ma /= len(sv)
    else:
        ma = np.abs(np.asarray(sv, dtype=np.float64)).mean(axis=0)
    return ma, ""


def write_feature_importance_log(
    pre: ColumnTransformer,
    model_obj: object,
    *,
    prefix: str,
    model_name: str,
    test_mode: str,
    top_k: int = 25,
    X_reference: np.ndarray | None = None,
    shap_max_rows: int = 2000,
    random_state: int = 42,
) -> Path:
    """전처리 후 피처 이름, feature_importances_, (가능 시) SHAP 평균 |값| 상위 top_k를 .log로 남긴다.

    파일명에 타임스탬프를 붙이지 않으며, 같은 prefix/model_name/test_mode로 다시 실행하면 동일 파일을 덮어쓴다.
    """
    path = SEVERITY_DIR / f"feature_importance_{prefix}_{model_name}_{test_mode}.log"
    names = list(pre.get_feature_names_out())
    imp, note = _importances_from_model(model_obj)

    lines = [
        f"# feature importance + SHAP (트리/부스팅 위주)",
        f"# time: {datetime.now().isoformat()}",
        f"# model: {model_name}  test_mode: {test_mode}",
        "#",
        "# [1] feature_importances_: 모델 내장 분할 기여(상대 비교용).",
        "# [2] SHAP: TreeExplainer, 참조 표본에서 피처 j의 평균 |SHAP| (방향 무시, 크기만 순위용).",
        "#     다중 클래스면 클래스별 SHAP 절댓값 평균을 다시 평균.",
        "#",
    ]
    lines.append("# --- 전처리 후 피처 목록 (모델 입력 차원 순서) ---")
    lines.append("idx\tfeature")
    for j, nm in enumerate(names):
        lines.append(f"{j}\t{nm}")
    lines.append("#")

    if imp is None or len(imp) != len(names):
        lines.append("# --- 내장 feature_importances_: 없음 또는 차원 불일치 ---")
        lines.append(note or "피처 수와 중요도 벡터 길이가 맞지 않습니다.")
    else:
        imp = np.maximum(imp, 0.0)
        s = float(imp.sum())
        norm = imp / s if s > 0 else imp
        order = np.argsort(-norm)
        lines.append(f"# --- 내장 importance 상위 {min(top_k, len(names))} (합=1 정규화) ---")
        lines.append("rank\timportance_norm\timportance_raw\tfeature")
        for r, j in enumerate(order[:top_k], start=1):
            lines.append(f"{r}\t{norm[j]:.6f}\t{imp[j]:.6f}\t{names[j]}")

    lines.append("#")
    if X_reference is None:
        lines.append("# SHAP: X_reference 미전달 — 생략.")
    else:
        inner = getattr(model_obj, "model", model_obj)
        Xs = subsample_reference_matrix(X_reference, shap_max_rows, random_state)
        shap_ma, shap_note = _shap_mean_abs_per_feature(inner, Xs)
        if shap_ma is None or len(shap_ma) != len(names):
            lines.append(f"# SHAP: {shap_note}")
        else:
            s2 = float(shap_ma.sum())
            sn = shap_ma / s2 if s2 > 0 else shap_ma
            o2 = np.argsort(-sn)
            lines.append(
                f"# --- SHAP mean|value| 상위 {min(top_k, len(names))} (표본 {Xs.shape[0]}행, 합=1 정규화) ---"
            )
            lines.append("rank\tshap_norm\tshap_mean_abs\tfeature")
            for r, j in enumerate(o2[:top_k], start=1):
                lines.append(f"{r}\t{sn[j]:.6f}\t{shap_ma[j]:.6f}\t{names[j]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
