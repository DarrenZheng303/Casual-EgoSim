#!/usr/bin/env bash
# EgoSim framewise Stage 2.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/egosim_train_common.sh"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/causal_cd_framewise_egosim.yaml}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/checkpoints/egosim_stage2_cd_framewise}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${ROOT_DIR}/output/egosim_stage2_cd_framewise_${RUN_TIMESTAMP}}"
EGOSIM_MODEL_ROOT="${EGOSIM_MODEL_ROOT:-${WORK_ROOT}/model/EgoSim-14B}"
EGOSIM_DATA_PATH="${EGOSIM_DATA_PATH:-${USER_ROOT}/datasets/luyitas/egosim_egodex_egovid/cache/train/egodex}"
STAGE1_CKPT="${STAGE1_CKPT:-${ROOT_DIR}/checkpoints/egosim_stage1_framewise/checkpoint_model_001000/model.pt}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
SAVE_EVAL_INTERVAL="${SAVE_EVAL_INTERVAL:-100}"
MASTER_PORT="${MASTER_PORT:-6062}"
RDZV_ID="${RDZV_ID:-${MA_JOB_NAME:-egosim_stage2_framewise}}"

egosim_require \
  "${PYTHON_BIN}" "${CONFIG_PATH}" \
  "${EGOSIM_MODEL_ROOT}/diffusion_pytorch_model.safetensors" \
  "${EGOSIM_DATA_PATH}" "${STAGE1_CKPT}"
egosim_prepare_distributed

mkdir -p "${RUN_OUTPUT_DIR}/runtime_configs" "${RUN_OUTPUT_DIR}/wandb" "${LOGDIR}"
RUNTIME_CONFIG="${RUN_OUTPUT_DIR}/runtime_configs/egosim_stage2_node${NODE_RANK}.yaml"
cp "${CONFIG_PATH}" "${RUNTIME_CONFIG}"
egosim_set_yaml data_path "${EGOSIM_DATA_PATH}" "${RUNTIME_CONFIG}"
egosim_set_yaml egosim_model_root "${EGOSIM_MODEL_ROOT}" "${RUNTIME_CONFIG}"
egosim_set_yaml generator_ckpt "${STAGE1_CKPT}" "${RUNTIME_CONFIG}"
egosim_set_yaml teacher_ckpt "${STAGE1_CKPT}" "${RUNTIME_CONFIG}"
egosim_set_yaml max_train_steps "${MAX_TRAIN_STEPS}" "${RUNTIME_CONFIG}"
egosim_set_yaml log_iters "${SAVE_EVAL_INTERVAL}" "${RUNTIME_CONFIG}"

LOG_FILE="${LOG_FILE:-${RUN_OUTPUT_DIR}/egosim_stage2_node${NODE_RANK}_${RUN_TIMESTAMP}.log}"
{
  echo "===== EgoSim Stage-2 Framewise CCD ====="
  echo "LOGDIR=${LOGDIR}"
  echo "CONFIG_PATH=${RUNTIME_CONFIG}"
  echo "STAGE1_CKPT=${STAGE1_CKPT}"
  echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
} | tee "${LOG_FILE}"
egosim_train "${LOG_FILE}" "${RUNTIME_CONFIG}" "${LOGDIR}" "${RUN_OUTPUT_DIR}" "$@"
