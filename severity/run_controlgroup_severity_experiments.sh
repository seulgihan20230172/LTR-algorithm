#!/usr/bin/env bash
# L2R 제외 control group: 분류 + 이상탐지 + 회귀
# 사용: 프로젝트 루트에서 bash severity/run_controlgroup_severity_experiments.sh
#
# 평가 모드(test_mode) 기본값: CONFIG YAML의 evaluation.test_mode
#   CONFIG=path/to.yaml     — YAML 경로 (기본 severity/experiment_config.yaml)
#   TEST_MODE=...           — (선택) 한 번에 덮어쓰기 (--test-mode 전달)
#   PYTHON=python3          — 인터프리터

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
CONFIG="${CONFIG:-severity/experiment_config.yaml}"
export CONFIG

LOG_SUFFIX=$($PY -c "import os,sys; sys.path.insert(0,'.'); from severity.experiment_config import load_experiment_config; print(load_experiment_config(os.environ['CONFIG'])['evaluation']['test_mode'])")
TM_ARGS=()
if [ -n "${TEST_MODE:-}" ]; then
  TM_ARGS=(--test-mode "$TEST_MODE")
  LOG_SUFFIX="$TEST_MODE"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/severity/experiment_logs/controlgroup_${RUN_ID}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR"

echo "[실험 배치] controlgroup_${RUN_ID}"
echo "[설정 파일] ${CONFIG}"
echo "[test_mode] ${LOG_SUFFIX} (YAML의 evaluation.test_mode; TEST_MODE 환경변수로 덮어쓰기 가능)"
echo "[로그 디렉터리] ${LOGDIR}"

run_one() {
  local name="$1"
  shift
  echo ""
  echo "========== ${name} =========="
  if ! "$PY" "$@"; then
    echo "[FAIL] ${name}" >> "${OUT}/FAILED.txt"
    echo "[FAIL] ${name}" >&2
  fi
}

: > "${OUT}/FAILED.txt"

# --- Classification ---
for m in random_forest lightgbm; do
  run_one "cls_${m}" severity/train_severity_classification_rank.py \
    --config "$CONFIG" --model "$m" "${TM_ARGS[@]}" \
    --log "${LOGDIR}/cls_${m}_${LOG_SUFFIX}.log"
done

# --- Anomaly ---
for m in vanilla_ae denoising_ae vae sequence_ae deep_stacked_ae isolation_forest; do
  run_one "anomaly_${m}" severity/train_severity_anomaly_rank.py \
    --config "$CONFIG" --model "$m" "${TM_ARGS[@]}" \
    --log "${LOGDIR}/anomaly_${m}_${LOG_SUFFIX}.log"
done

# --- Regression ---
run_one "reg_xgboost_regressor" severity/train_severity_regression_rank.py \
  --config "$CONFIG" --model xgboost_regressor "${TM_ARGS[@]}" \
  --log "${LOGDIR}/reg_xgboost_regressor_${LOG_SUFFIX}.log"

echo ""
echo "[요약 MD 생성]"
"$PY" severity/summarize_severity_logs.py "$LOGDIR" --out "${OUT}/SUMMARY.md"
echo "[완료] ${OUT}/SUMMARY.md"
