#!/usr/bin/env bash
# severity 관련 실험 전체 실행 (L2R + 분류 + 이상탐지 + 회귀)
# 사용: 프로젝트 루트에서 bash severity/run_all_severity_experiments.sh
# 환경변수로 조정 가능: CSV, TEST_MODE, TEST_SIZE, VAL_SIZE, RANDOM_STATE, GROUP_SIZE, L2R_EPOCHS, ANOMALY_EPOCHS, PYTHON

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
CSV="${CSV:-severity/logging_monitoring_anomalies.csv}"
TEST_MODE="${TEST_MODE:-train_thresholds}"
TEST_SIZE="${TEST_SIZE:-0.2}"
VAL_SIZE="${VAL_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
GROUP_SIZE="${GROUP_SIZE:-64}"
L2R_EPOCHS="${L2R_EPOCHS:-35}"
ANOMALY_EPOCHS="${ANOMALY_EPOCHS:-20}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/severity/experiment_logs/all_${RUN_ID}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR"

echo "[실험 배치] all_${RUN_ID}"
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

# --- L2R ---
for m in listnet listmle xgboost; do
  run_one "l2r_${m}" severity/train_severity_l2r_rank.py \
    --csv "$CSV" --model "$m" --test-mode "$TEST_MODE" \
    --test-size "$TEST_SIZE" --val-size "$VAL_SIZE" --random-state "$RANDOM_STATE" \
    --group-size "$GROUP_SIZE" --epochs "$L2R_EPOCHS" \
    --log "${LOGDIR}/l2r_${m}_${TEST_MODE}.log"
done

# --- Classification ---
for m in random_forest lightgbm; do
  run_one "cls_${m}" severity/train_severity_classification_rank.py \
    --csv "$CSV" --model "$m" --test-mode "$TEST_MODE" \
    --test-size "$TEST_SIZE" --val-size "$VAL_SIZE" --random-state "$RANDOM_STATE" \
    --group-size "$GROUP_SIZE" \
    --log "${LOGDIR}/cls_${m}_${TEST_MODE}.log"
done

# --- Anomaly ---
for m in vanilla_ae denoising_ae vae sequence_ae deep_stacked_ae isolation_forest; do
  run_one "anomaly_${m}" severity/train_severity_anomaly_rank.py \
    --csv "$CSV" --model "$m" --test-mode "$TEST_MODE" \
    --test-size "$TEST_SIZE" --val-size "$VAL_SIZE" --random-state "$RANDOM_STATE" \
    --group-size "$GROUP_SIZE" --epochs "$ANOMALY_EPOCHS" \
    --log "${LOGDIR}/anomaly_${m}_${TEST_MODE}.log"
done

# --- Regression ---
run_one "reg_xgboost_regressor" severity/train_severity_regression_rank.py \
  --csv "$CSV" --model xgboost_regressor --test-mode "$TEST_MODE" \
  --test-size "$TEST_SIZE" --val-size "$VAL_SIZE" --random-state "$RANDOM_STATE" \
  --group-size "$GROUP_SIZE" \
  --log "${LOGDIR}/reg_xgboost_regressor_${TEST_MODE}.log"

echo ""
echo "[요약 MD 생성]"
"$PY" severity/summarize_severity_logs.py "$LOGDIR" --out "${OUT}/SUMMARY.md"
echo "[완료] ${OUT}/SUMMARY.md"
