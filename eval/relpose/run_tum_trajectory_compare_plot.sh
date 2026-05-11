#!/usr/bin/env bash
# TUM relpose 轨迹对比图（XY 网格）。默认输出:
#   eval_results/relpose/<EVAL_DATASET>/trajectory_compare_xy.png
#
# 环境变量（可选）:
#   EVAL_DATASET=tum_s1_1000
#   POSE_STRIDE=1                    与 launch.py --pose_eval_stride 一致
#   PRED_DIR_A / PRED_DIR_B          显式指定两套结果根目录
#   LABEL_A LABEL_B                  图例
#   OUT                              输出 PNG
#   PYTHON / PY_EXEC                 解释器，默认优先 miniconda env「ttt3r」
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# 与 long TUM 子集长度 1000 对应 metadata 键 tum_s1_1000
EVAL_DATASET="${EVAL_DATASET:-tum_s1_1000}"
POSE_STRIDE="${POSE_STRIDE:-1}"

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

CUT="${ROOT}/eval_results/relpose/${EVAL_DATASET}/cut3r"
TTT="${ROOT}/eval_results/relpose/${EVAL_DATASET}/ttt3r"

if [[ -n "${PRED_DIR_A:-}" ]]; then
  PRED_A="$PRED_DIR_A"
else
  if [[ -d "$CUT" ]]; then
    PRED_A="$CUT"
  elif [[ -d "$TTT" ]]; then
    PRED_A="$TTT"
  else
    echo "错误: 未找到 ${CUT} 或 ${TTT}，请指定 PRED_DIR_A 或先跑评测。" >&2
    exit 1
  fi
fi

if [[ -n "${PRED_DIR_B:-}" ]]; then
  PRED_B="$PRED_DIR_B"
else
  PRED_B=""
  if [[ -d "$CUT" ]] && [[ "$PRED_A" != "$CUT" ]]; then
    PRED_B="$CUT"
  elif [[ -d "$TTT" ]] && [[ "$PRED_A" != "$TTT" ]]; then
    PRED_B="$TTT"
  fi
fi

OUT="${OUT:-${ROOT}/eval_results/relpose/${EVAL_DATASET}/trajectory_compare_xy.png}"
mkdir -p "$(dirname "${OUT}")"

EXTRA=(--pred_dir_a "${PRED_A}")

if [[ -n "${PRED_B}" ]]; then
  LA="${LABEL_A:-CUT3R}"
  LB="${LABEL_B:-TTT3R}"
  echo "绘制双方法: A=${LA} (${PRED_A})  B=${LB} (${PRED_B})"
  EXTRA+=(--pred_dir_b "${PRED_B}" --label_a "${LA}" --label_b "${LB}")
else
  if [[ -n "${LABEL_A:-}" ]]; then
    LA="$LABEL_A"
  elif [[ "$PRED_A" == "$TTT" ]]; then
    LA="TTT3R"
  else
    LA="CUT3R"
  fi
  echo "仅绘制单方法: ${LA} (${PRED_A})"
  if [[ "${PRED_A}" == "${TTT}" ]] && [[ ! -d "${CUT}" ]]; then
    echo "提示: 若需 GT+CUT3R+TTT3R 三线同图，请先跑 CUT3R 评测:" >&2
    echo "  bash eval/relpose/run_cut3r_tum_relpose.sh ${EVAL_DATASET}" >&2
    echo "  完成后再执行本脚本。" >&2
  fi
  EXTRA+=(--label_a "${LA}")
fi

echo "使用 Python: ${PY}"
exec "${PY}" "${ROOT}/eval/relpose/plot_tum_trajectory_comparison_grid.py" \
  --eval_dataset "${EVAL_DATASET}" \
  --pose_eval_stride "${POSE_STRIDE}" \
  "${EXTRA[@]}" \
  --out "${OUT}"
