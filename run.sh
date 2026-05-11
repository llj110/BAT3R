#!/usr/bin/env bash
# =============================================================================
# run.sh  --  visualization examples for TTT3R outputs
# =============================================================================
PYTHON=/home/hyp/miniconda3/envs/ttt3r/bin/python

# ---------------------------------------------------------------------------
# Mode 1: demo --outdir  (color/ depth/ conf/ camera/)
# Renders point cloud + green camera frustums + trajectory from demo.py output
# ---------------------------------------------------------------------------
# $PYTHON visualize.py --outdir tmp/taylor \
#     --max_pts 2000000 --elev 20 --azim -60 --dpi 200 \
#     --zoom 0.7 --point_size 1.5 --cam_every 5 --cam_size 0.05

# ---------------------------------------------------------------------------
# Mode 2: eval 输出现为 *-mask.ply / *-gt.ply（相机已嵌入 PLY 头），不再生成 scene.npy
# 若需批量可视化，可对 *.ply 调用本仓库内 visualize_ply.py / view_ply_local.py
# ---------------------------------------------------------------------------
# shopt -s nullglob
# for ply in eval_results/video_recon/7scenes_500/ttt3r/7scenes/*-mask.ply; do
#     $PYTHON view_ply_local.py "$ply" ...
# done

# ---------------------------------------------------------------------------
# Mode 3: Open3D headless --ply + --poses
# Renders .ply point cloud with green wireframe camera frustums,
# orange trajectory line, white background, high-res PNG, no display window.
# ---------------------------------------------------------------------------
# $PYTHON visualize.py --ply scene.ply --poses cameras.npy \
#     --save_path scene_vis.png \
#     --cam_size 0.05 --cam_every 1 --width 1920 --height 1080

echo "ALL DONE"
