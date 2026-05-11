#!/usr/bin/env python3
"""
将多次 relpose 评测（如 CUT3R / TTT3R / BAT3R）在 TUM 上的轨迹叠画到同一 XY 平面图（4×3 组图示例）。

对齐方式与 ``evo_utils.plot_trajectory`` / ``eval_metrics`` 一致：
时间戳对齐 → 估计轨迹对 GT 做 Sim(3)（含尺度）对齐，再绘制。

用法示例（在项目根 BAT3R 下）::

    cd /path/to/BAT3R
    python eval/relpose/plot_tum_trajectory_comparison_grid.py \\
        --eval_dataset tum_s1_1000 \\
        --pred_dirs eval_results/relpose/tum_s1_1000/cut3r \\
                   eval_results/relpose/tum_s1_1000/ttt3r \\
                   eval_results/relpose/tum_s1_1000/bat3r \\
        --labels CUT3R TTT3R BAT3R \\
        --pose_eval_stride 1 \\
        --out tum_fr3_comparison_xy.png

若未指定 ``--seq_list``：默认取所有方法目录里都存在 ``<seq>/pred_traj.txt`` 的序列交集。
取前 nrow*ncol 条（可用 ``--max_plots`` 限制）。
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evo.core import sync  # noqa: E402
from evo.tools import file_interface  # noqa: E402

from eval.relpose.evo_utils import load_traj, make_traj  # noqa: E402
from eval.relpose.metadata import dataset_metadata  # noqa: E402


def _align_to_gt(gt_traj, pred_file: str):
    gt = deepcopy(make_traj(gt_traj))
    pred_raw = file_interface.read_tum_trajectory_file(pred_file)
    pred = deepcopy(make_traj(pred_raw))
    if pred.timestamps.shape[0] == gt.timestamps.shape[0]:
        pred.timestamps = gt.timestamps
    gt_s, pred_s = sync.associate_trajectories(gt, pred)
    pred_s.align(gt_s, correct_scale=True)
    return gt_s, pred_s


def _xy(traj):
    p = traj.positions_xyz
    return p[:, 0], p[:, 1]


def _seqs_in_pred_dir(pred_dir: str):
    return sorted(
        d
        for d in os.listdir(pred_dir)
        if os.path.isfile(os.path.join(pred_dir, d, "pred_traj.txt"))
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dataset", type=str, default="tum_s1_1000")
    p.add_argument(
        "--img_path",
        type=str,
        default=None,
        help="覆盖 metadata 中的 img_path（默认用 metadata）",
    )
    p.add_argument(
        "--pred_dirs",
        nargs="+",
        default=None,
        help="多个方法的评测输出根目录（推荐）",
    )
    # 兼容旧参数：--pred_dir_a/--pred_dir_b + --label_a/--label_b
    p.add_argument("--pred_dir_a", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--pred_dir_b", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--label_a", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--label_b", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="与 pred_dirs 一一对应的方法名",
    )
    p.add_argument(
        "--pose_eval_stride",
        type=int,
        default=1,
        help="需与跑 launch.py 时一致，用于下采样 GT",
    )
    p.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="要画的序列名；默认取所有方法目录交集",
    )
    p.add_argument("--nrow", type=int, default=4)
    p.add_argument("--ncol", type=int, default=3)
    p.add_argument(
        "--max_plots",
        type=int,
        default=None,
        help="最多画多少条序列（默认 nrow*ncol）",
    )
    p.add_argument("--out", type=str, default="tum_traj_compare_xy.png")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    meta = dataset_metadata[args.eval_dataset]
    img_path = args.img_path or meta["img_path"]
    traj_format = meta["traj_format"]
    gt_traj_func = meta["gt_traj_func"]

    method_dirs = list(args.pred_dirs) if args.pred_dirs is not None else []
    if not method_dirs and args.pred_dir_a is not None:
        method_dirs.append(args.pred_dir_a)
        if args.pred_dir_b is not None:
            method_dirs.append(args.pred_dir_b)
    if not method_dirs:
        raise SystemExit("请至少提供一个方法目录：--pred_dirs ...")

    for d in method_dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"方法目录不存在: {d}")

    labels = list(args.labels) if args.labels is not None else []
    if not labels and args.label_a is not None:
        labels.append(args.label_a)
        if len(method_dirs) > 1:
            labels.append(args.label_b or os.path.basename(method_dirs[1]))
    if not labels:
        labels = [os.path.basename(d.rstrip("/")) for d in method_dirs]
    if len(labels) != len(method_dirs):
        raise SystemExit("参数数量不匹配：--labels 数量必须与 --pred_dirs 相同")

    if args.seq_list is not None:
        sequences = args.seq_list
    else:
        seq_sets = [set(_seqs_in_pred_dir(d)) for d in method_dirs]
        sequences = sorted(set.intersection(*seq_sets)) if seq_sets else []

    cap = args.max_plots if args.max_plots is not None else args.nrow * args.ncol
    sequences = sequences[:cap]

    n = len(sequences)
    if n == 0:
        raise SystemExit("没有找到可绘制的序列（检查 pred_dir 与 pred_traj.txt）")

    try:
        import seaborn as sns

        sns.set_theme(style="darkgrid")
    except ImportError:
        plt.rcParams["axes.facecolor"] = "#eaeaf2"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.color"] = "white"

    fig_w, fig_h = 3.6 * args.ncol, 3.6 * args.nrow
    fig, axes = plt.subplots(
        args.nrow, args.ncol, figsize=(fig_w, fig_h), squeeze=False
    )

    method_colors = [
        "#6a3598",
        "#e07a6e",
        "#2f7f5f",
        "#3572a5",
        "#a57f35",
        "#8f3c7a",
    ]
    z_gt, z_est = 1, 3

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        seq = sequences[idx]
        gt_file = gt_traj_func(img_path, None, seq)
        gt_tuple = load_traj(
            gt_file, traj_format=traj_format, stride=args.pose_eval_stride
        )
        pred_paths = [os.path.join(d, seq, "pred_traj.txt") for d in method_dirs]
        if not all(os.path.isfile(x) for x in pred_paths):
            ax.text(0.5, 0.5, f"缺轨迹:\n{seq}", ha="center", va="center")
            ax.axis("off")
            continue

        try:
            gt_ref = None
            aligned_preds = []
            for pred_file in pred_paths:
                gt_s, pred_s = _align_to_gt(gt_tuple, pred_file)
                gt_ref = gt_s
                aligned_preds.append(pred_s)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"{seq}\n{e!s}",
                ha="center",
                va="center",
                fontsize=8,
                wrap=True,
            )
            ax.axis("off")
            continue

        xg, yg = _xy(gt_ref)
        ax.plot(
            xg,
            yg,
            "-",
            color="#b0b0b0",
            linewidth=5.5,
            alpha=0.5,
            label="GT",
            zorder=z_gt,
        )
        for m_idx, pred_traj in enumerate(aligned_preds):
            xp, yp = _xy(pred_traj)
            ax.plot(
                xp,
                yp,
                "-",
                color=method_colors[m_idx % len(method_colors)],
                linewidth=1.2,
                label=labels[m_idx],
                zorder=z_est,
            )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(seq.replace("rgbd_dataset_freiburg3_", "fr3/"), fontsize=9)
        ax.legend(loc="best", fontsize=7, framealpha=0.85)

    mode = " + ".join(["GT", *labels])
    fig.suptitle(
        f"{args.eval_dataset}  XY trajectories (Sim(3) aligned, {mode}, stride={args.pose_eval_stride})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.abspath(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}（{n} 条序列）")


if __name__ == "__main__":
    main()
