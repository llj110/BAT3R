#!/usr/bin/env bash
# 在仓库根运行:  bash eval/mv_recon/run_kitti_scannet_ply.sh [scannet|kitti]
# 数据与权重路径见 eval/mv_recon/bat3r_data_paths.py
# 默认: ScanNet 用 color_1000/depth_1000/pose_1000；KITTI long_s1 用 gather 1000；不限制 max_frames=全部帧（见下方）。
# 环境变量: WEIGHTS、OUTPUT_DIR、MAX_FRAMES(可选，不设置则全部用)、KF_EVERY(默认1=不抽帧)、
#  SCANNET_N_FRAMES、KITTI_GATHER_N、KITTI_PRESET、...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 与 eval/mv_recon/run.sh、eval/video_depth/run_kitti.sh 相同权重名
: "${WEIGHTS:=$ROOT/src/cut3r_512_dpt_4_64.pth}"
: "${OUTPUT_DIR:=$ROOT/eval_results/mv_recon/kitti_scannet_ply}"

DATASET="${1:-scannet}"

extra=()
if [[ -n "${MAX_FRAMES:-}" ]]; then extra+=(--max_frames "$MAX_FRAMES"); fi
if [[ -n "${KF_EVERY:-}" ]]; then extra+=(--kf_every "$KF_EVERY"); fi
if [[ "$DATASET" == "scannet" && "${SCANNET_AUG:-0}" == "1" ]]; then extra+=(--input_aug scannet); fi
if [[ -n "${SCANNET_N_FRAMES:-}" ]]; then extra+=(--scannet_n_frames "$SCANNET_N_FRAMES"); fi
if [[ -n "${SCANNET_SCENES:-}" ]]; then extra+=(--scannet_scenes "$SCANNET_SCENES"); fi
if [[ -n "${KITTI_SCENES:-}" ]]; then extra+=(--kitti_scenes "$KITTI_SCENES"); fi
if [[ -n "${KITTI_PRESET:-}" ]]; then extra+=(--kitti_preset "$KITTI_PRESET"); fi
if [[ -n "${KITTI_GATHER_N:-}" ]]; then extra+=(--kitti_gather_n "$KITTI_GATHER_N"); fi
if [[ "${KITTI_USE_APPROX:-0}" == "1" ]]; then extra+=(--kitti_approx_pose); fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT/src:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT/src"
fi

acmd=(
  accelerate launch -m eval.mv_recon.launch_kitti_scannet_ply
  --weights "$WEIGHTS"
  --output_dir "$OUTPUT_DIR"
  --dataset "$DATASET"
  "${extra[@]}"
)

echo "Running: ${acmd[*]}"
exec "${acmd[@]}"
