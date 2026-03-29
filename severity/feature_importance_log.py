"""트리·부스팅 등 feature_importances_를 제공하는 모델용 상대적 피처 중요도 로그."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer

SEVERITY_DIR = Path(__file__).resolve().parent


def _importances_from_model(model_obj: object) -> tuple[np.ndarray | None, str]:
    inner = getattr(model_obj, "model", model_obj)
    if hasattr(inner, "feature_importances_"):
        return np.asarray(inner.feature_importances_, dtype=np.float64), ""
    return None, "이 모델은 sklearn 스타일 feature_importances_를 제공하지 않습니다 (AE·L2R·IsolationForest 등)."


def write_feature_importance_log(
    pre: ColumnTransformer,
    model_obj: object,
    *,
    prefix: str,
    model_name: str,
    test_mode: str,
    top_k: int = 25,
) -> Path:
    """전처리 후 피처 이름과 중요도 상위 top_k를 severity 폴더에 .log로 남긴다."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SEVERITY_DIR / f"feature_importance_{prefix}_{model_name}_{test_mode}_{ts}.log"
    names = list(pre.get_feature_names_out())
    imp, note = _importances_from_model(model_obj)

    lines = [
        f"# feature importance (relative, tree-based models)",
        f"# time: {datetime.now().isoformat()}",
        f"# model: {model_name}  test_mode: {test_mode}",
        "#",
        "# 스케일과 중요도: RF/LightGBM/XGBoost의 importance는 분할·감소 불순도 등 기준이라",
        "# 원본 피처 단위(큰 숫자)에 직접 비례하지 않습니다. 전처리 후에도 순위는 대체로 의미가 있습니다.",
        "# (선형·거리 기반 모델과 달리 트리는 스케일에 상대적으로 덜 민감합니다.)",
        "#",
    ]
    if imp is None or len(imp) != len(names):
        lines.append(note or "피처 수와 중요도 벡터 길이가 맞지 않습니다.")
    else:
        imp = np.maximum(imp, 0.0)
        s = float(imp.sum())
        norm = imp / s if s > 0 else imp
        order = np.argsort(-norm)
        lines.append(f"# 상위 {min(top_k, len(names))} (정규화: 합=1)")
        lines.append("rank\timportance_norm\timportance_raw\tfeature")
        for r, j in enumerate(order[:top_k], start=1):
            lines.append(f"{r}\t{norm[j]:.6f}\t{imp[j]:.6f}\t{names[j]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
