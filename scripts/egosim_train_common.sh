#!/usr/bin/env bash

WORK_ROOT="${WORK_ROOT:-/home/ma-user/work}"
USER_ROOT="${USER_ROOT:-${WORK_ROOT}/users/zhengshikang}"
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${USER_ROOT}/conda/envs/causal_forcing/bin/python}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${MA_NUM_GPUS:-8}}"
NNODES="${NNODES:-${MA_NUM_HOSTS:-${VC_WORKER_NUM:-1}}}"
NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-0}}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-5}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-200}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

egosim_require() {
  local path
  for path in "$@"; do
    [[ -e "${path}" ]] || { echo "Required path not found: ${path}" >&2; return 1; }
  done
}

egosim_set_yaml() {
  local key="$1" value="$2" file="$3"
  if grep -q "^${key}:" "${file}"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "${file}"
  else
    printf '%s: %s\n' "${key}" "${value}" >> "${file}"
  fi
}

egosim_prepare_distributed() {
  if [[ "${NNODES}" != "1" ]]; then
    [[ -n "${MASTER_ADDR:-}" || -n "${VC_WORKER_HOSTS:-}" ]] || {
      echo "Multi-node training requires MASTER_ADDR or VC_WORKER_HOSTS." >&2
      return 1
    }
    MASTER_ADDR="${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}"
  fi
}

egosim_train() {
  local log_file="$1" config_path="$2" logdir="$3" output_dir="$4"
  shift 4
  local args=(
    --config_path "${config_path}"
    --logdir "${logdir}"
    --wandb-save-dir "${output_dir}/wandb"
    --no_visualize
  )
  [[ "${DISABLE_WANDB}" == "1" ]] && args+=(--disable-wandb)
  args+=("$@")

  cd "${ROOT_DIR}"
  set +e
  if [[ "${NNODES}" == "1" ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      train.py "${args[@]}" 2>&1 | tee -a "${log_file}"
  else
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
      --nproc_per_node="${NPROC_PER_NODE}" --rdzv_id="${RDZV_ID}" \
      --rdzv_backend=static --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      train.py "${args[@]}" 2>&1 | tee -a "${log_file}"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  echo "===== torchrun exit code: ${status} =====" | tee -a "${log_file}"
  return "${status}"
}
