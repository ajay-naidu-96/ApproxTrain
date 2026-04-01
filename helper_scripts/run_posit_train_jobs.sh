#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DMODEL="${DMODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG_DIR="${ROOT_DIR}/train_logs"
TEST_MODE="${TEST_MODE:-0}"

mkdir -p "${TRAIN_LOG_DIR}"

cd "${ROOT_DIR}/lut"
./lut_gen.sh

cd "${ROOT_DIR}"
make clean
make convam MULTIPLIER=AMSIMULATOR
make denseam MULTIPLIER=AMSIMULATOR
make matmulam MULTIPLIER=AMSIMULATOR

COMMON_ARGS=(
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --d_model "${DMODEL}"
  --num_layers "${NUM_LAYERS}"
  --num_heads "${NUM_HEADS}"
)

if [[ "${TEST_MODE}" == "1" ]]; then
  COMMON_ARGS+=(--test_mode)
fi

run_job() {
  local multiplier="$1"
  local experiment_name="$2"
  local log_file="$3"

  nohup "${PYTHON_BIN}" train_transformer.py \
    --multiplier="${multiplier}" \
    --experiment_name="${experiment_name}" \
    "${COMMON_ARGS[@]}" \
    > "${log_file}" 2>&1 &

  echo "${multiplier} pid=$! log=${log_file}"
}

run_job "pos8e0" "pos8e0_${TIMESTAMP}" "${TRAIN_LOG_DIR}/pos8e0_${TIMESTAMP}.log"
run_job "pos8e1" "pos8e1_${TIMESTAMP}" "${TRAIN_LOG_DIR}/pos8e1_${TIMESTAMP}.log"
