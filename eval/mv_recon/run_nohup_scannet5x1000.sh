#!/usr/bin/env bash
# 用 nohup 跑 5 个 ScanNet 场景，每场景 1000 帧；仅写 PLY（默认不算 Acc/ICP 指标，快很多）。
# 要指标：在 python 后加 --eval_metrics 或 export EVAL_METRICS=1
# 使用: bash eval/mv_recon/run_nohup_scannet5x1000.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LOG="$ROOT/eval_results/mv_recon/nohup_scannet5x1000.log"
PIDFILE="$ROOT/eval_results/mv_recon/nohup_scannet5x1000.pid"
mkdir -p "$ROOT/eval_results/mv_recon"
{
  echo ""
  echo "======== $(date -Iseconds) 新任务启动（默认仅 PLY，无 ICP/Acc） ========"
} >> "$LOG"
nohup bash -c "
  source /home/hyp/miniconda3/bin/activate ttt3r
  export PYTHONPATH=$ROOT/src:\$PYTHONPATH
  cd $ROOT
  python -m eval.mv_recon.launch_kitti_scannet_ply \
    --weights $ROOT/src/cut3r_512_dpt_4_64.pth \
    --output_dir $ROOT/eval_results/mv_recon/kitti_scannet_ply_nohup \
    --dataset scannet \
    --scannet_n_frames 1000 \
    --scannet_scenes 'scene0708_00,scene0709_00,scene0710_00,scene0711_00'
" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "nohup 已启动 PID=$!  log=$LOG  pidfile=$PIDFILE"
echo "查看: tail -f $LOG"
