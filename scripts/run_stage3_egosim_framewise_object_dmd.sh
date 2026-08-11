#!/usr/bin/env bash
# EgoSim framewise Stage 3 Object-DMD.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/egosim_train_common.sh"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/causal_forcing_dmd_framewise_egosim_object_dmd.yaml}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/checkpoints/egosim_stage3_framewise_object_dmd/${RUN_TIMESTAMP}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${ROOT_DIR}/output/egosim_stage3_framewise_object_dmd_${RUN_TIMESTAMP}}"
EGOSIM_MODEL_ROOT="${EGOSIM_MODEL_ROOT:-${WORK_ROOT}/model/EgoSim-14B}"
EGOSIM_DATA_PATH="${EGOSIM_DATA_PATH:-${USER_ROOT}/datasets/luyitas/egosim_egodex_egovid/cache/train/egodex}"
GENERATOR_CKPT="${GENERATOR_CKPT:-${ROOT_DIR}/checkpoints/egosim_stage2_cd_framewise/checkpoint_model_000500/model.pt}"
REAL_SCORE_CKPT="${REAL_SCORE_CKPT:-${EGOSIM_MODEL_ROOT}/diffusion_pytorch_model.safetensors}"
FAKE_SCORE_CKPT="${FAKE_SCORE_CKPT:-${EGOSIM_MODEL_ROOT}/diffusion_pytorch_model.safetensors}"
MASTER_PORT="${MASTER_PORT:-6062}"
RDZV_ID="${RDZV_ID:-${MA_JOB_NAME:-egosim_stage3_framewise_object_dmd}}"

egosim_require \
  "${PYTHON_BIN}" "${CONFIG_PATH}" "${EGOSIM_DATA_PATH}" \
  "${GENERATOR_CKPT}" "${REAL_SCORE_CKPT}" "${FAKE_SCORE_CKPT}"
egosim_prepare_distributed

mkdir -p "${RUN_OUTPUT_DIR}/runtime_configs" "${RUN_OUTPUT_DIR}/wandb" "${LOGDIR}"
RUNTIME_CONFIG="${RUN_OUTPUT_DIR}/runtime_configs/egosim_stage3_node${NODE_RANK}.yaml"
cp "${CONFIG_PATH}" "${RUNTIME_CONFIG}"
egosim_set_yaml generator_ckpt "${GENERATOR_CKPT}" "${RUNTIME_CONFIG}"
egosim_set_yaml real_score_ckpt "${REAL_SCORE_CKPT}" "${RUNTIME_CONFIG}"
egosim_set_yaml fake_score_ckpt "${FAKE_SCORE_CKPT}" "${RUNTIME_CONFIG}"
egosim_set_yaml data_path "${EGOSIM_DATA_PATH}" "${RUNTIME_CONFIG}"
egosim_set_yaml egosim_model_root "${EGOSIM_MODEL_ROOT}" "${RUNTIME_CONFIG}"

LOG_FILE="${LOG_FILE:-${RUN_OUTPUT_DIR}/egosim_stage3_node${NODE_RANK}_${RUN_TIMESTAMP}.log}"
{
  echo "===== EgoSim Stage-3 Causal 2-Step DMD + Object-DMD (4-Step First Frame) ====="
  echo "LOGDIR=${LOGDIR}"
  echo "CONFIG_PATH=${RUNTIME_CONFIG}"
  echo "GENERATOR_CKPT=${GENERATOR_CKPT}"
  echo "REAL_SCORE_CKPT=${REAL_SCORE_CKPT}"
  echo "FAKE_SCORE_CKPT=${FAKE_SCORE_CKPT}"
  echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
} | tee "${LOG_FILE}"
egosim_train "${LOG_FILE}" "${RUNTIME_CONFIG}" "${LOGDIR}" "${RUN_OUTPUT_DIR}" "$@"
