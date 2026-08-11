#!/usr/bin/env bash
# EgoSim framewise Stage 1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/egosim_train_common.sh"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/ar_diffusion_tf_framewise_egosim.yaml}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/checkpoints/egosim_stage1_framewise}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${ROOT_DIR}/output/egosim_stage1_framewise_${RUN_TIMESTAMP}}"
EGOSIM_MODEL_ROOT="${EGOSIM_MODEL_ROOT:-${WORK_ROOT}/model/EgoSim-14B}"
EGOSIM_DATA_PATH="${EGOSIM_DATA_PATH:-${USER_ROOT}/datasets/luyitas/egosim_egodex_egovid/cache/train/egodex}"
MASTER_PORT="${MASTER_PORT:-6061}"
RDZV_ID="${RDZV_ID:-${MA_JOB_NAME:-egosim_stage1_framewise}}"

egosim_require \
  "${PYTHON_BIN}" "${CONFIG_PATH}" \
  "${EGOSIM_MODEL_ROOT}/diffusion_pytorch_model.safetensors" \
  "${EGOSIM_DATA_PATH}"
egosim_prepare_distributed

mkdir -p "${RUN_OUTPUT_DIR}/runtime_configs" "${RUN_OUTPUT_DIR}/wandb" "${LOGDIR}"
RUNTIME_CONFIG="${RUN_OUTPUT_DIR}/runtime_configs/egosim_stage1_node${NODE_RANK}.yaml"
cp "${CONFIG_PATH}" "${RUNTIME_CONFIG}"
egosim_set_yaml data_path "${EGOSIM_DATA_PATH}" "${RUNTIME_CONFIG}"
egosim_set_yaml egosim_model_root "${EGOSIM_MODEL_ROOT}" "${RUNTIME_CONFIG}"

LOG_FILE="${LOG_FILE:-${RUN_OUTPUT_DIR}/egosim_stage1_node${NODE_RANK}_${RUN_TIMESTAMP}.log}"
{
  echo "===== EgoSim Stage-1 Framewise Teacher Forcing ====="
  echo "LOGDIR=${LOGDIR}"
  echo "CONFIG_PATH=${RUNTIME_CONFIG}"
  echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
} | tee "${LOG_FILE}"
egosim_train "${LOG_FILE}" "${RUNTIME_CONFIG}" "${LOGDIR}" "${RUN_OUTPUT_DIR}" "$@"
