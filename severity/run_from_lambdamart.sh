#!/usr/bin/env bash
# lambdarank 이후 모델(lambdamart)부터 전체 실험 실행

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
OUT="${ROOT}/severity/experiment_logs/from_lambdamart_${RUN_ID}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR"

echo "[실험 배치] from_lambdamart_${RUN_ID}"
echo "[설정 파일] ${CONFIG}"
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

# --- L2R (lambdarank 다음인 lambdamart부터 시작) ---
# 기존 순서: listnet listmle xgboost ranknet lambdarank lambdamart bm25
for m in lambdamart bm25; do
  run_one "l2r_${m}" severity/train_severity_l2r_rank.py \
    --config "$CONFIG" --model "$m" "${TM_ARGS[@]}" \
    --log "${LOGDIR}/l2r_${m}_${LOG_SUFFIX}.log"
done

# --- Classification (이후 동일하게 진행) ---
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
