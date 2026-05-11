#!/usr/bin/env bash
# 使用 CUT3R 状态更新跑一次 TUM「长序列」子集 relpose，输出到
#   eval_results/relpose/<EVAL_DATASET>/cut3r/
# 与 TTT3R 结果合并作图: bash eval/relpose/run_tum_trajectory_compare_plot.sh
#
# 用法:
#   bash eval/relpose/run_cut3r_tum_relpose.sh [EVAL_DATASET]
# 默认 EVAL_DATASET=tum_s1_1000
#
# 可选环境变量同其它脚本: CUDA_VISIBLE_DEVICES, MAIN_PORT, WEIGHTS, SIZE, PYTHON
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EVAL_DATASET="${1:-${EVAL_DATASET:-tum_s1_1000}}"
MAIN_PORT="${MAIN_PORT:-29551}"
WEIGHTS="${WEIGHTS:-${ROOT}/src/cut3r_512_dpt_4_64.pth}"
SIZE="${SIZE:-512}"
OUT_DIR="${ROOT}/eval_results/relpose/${EVAL_DATASET}/cut3r"

_DEFAULT_TTT3R_PY="${HOME}/miniconda3/envs/ttt3r/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif [[ -n "${PY_EXEC:-}" ]]; then
  PY="${PY_EXEC}"
elif [[ -x "${_DEFAULT_TTT3R_PY}" ]]; then
  PY="${_DEFAULT_TTT3R_PY}"
else
  PY="python3"
fi

ACCEL="$(dirname "${PY}")/accelerate"
if [[ ! -x "$ACCEL" ]]; then
  ACCEL="accelerate"
fi

echo "CUT3R relpose  EVAL_DATASET=${EVAL_DATASET}  OUT=${OUT_DIR}"
echo "Python: ${PY}"

exec "$ACCEL" launch --num_processes 1 --main_process_port "${MAIN_PORT}" \
  eval/relpose/launch.py \
  --weights "${WEIGHTS}" \
  --output_dir "${OUT_DIR}" \
  --eval_dataset "${EVAL_DATASET}" \
  --size "${SIZE}" \
  --model_update_type cut3r
