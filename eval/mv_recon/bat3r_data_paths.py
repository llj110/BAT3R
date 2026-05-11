"""
BAT3R 仓库内路径：long_scannet_s3 / long_kitti_s1（relpose 与 video_depth metadata 一致）；
权重与 eval 脚本相同：src/cut3r_512_dpt_4_64.pth。
"""
from __future__ import annotations

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _j(*parts: str) -> str:
    return os.path.join(_REPO_ROOT, *parts)


# --- 权重（同 eval/mv_recon/run.sh、eval/video_depth/run_kitti.sh）
DEFAULT_MV_RECON_WEIGHTS: str = _j("src", "cut3r_512_dpt_4_64.pth")

# --- ScanNet long：color_N / depth_N / pose_N.txt，默认 1000 与 relpose scannet_s3_1000 一致
SCANNET_LONG_S3: str = _j("data", "long_scannet_s3")
SCANNET_DEFAULT_N: int = 1000
SCANNET_V2: str = _j("data", "scannetv2")

# --- KITTI long (video_depth kitti_s1_N)
KITTI_LONG_CROPPED: str = _j(
    "data", "long_kitti_s1", "depth_selection", "val_selection_cropped"
)
# KITTI long_s1：image_gathered_N / groundtruth_depth_gathered_N
KITTI_DEFAULT_GATHER_N: int = 1000


def kitti_long_image_gathered(n: int) -> str:
    return os.path.join(KITTI_LONG_CROPPED, f"image_gathered_{int(n)}")


def kitti_long_depth_gathered(n: int) -> str:
    return os.path.join(KITTI_LONG_CROPPED, f"groundtruth_depth_gathered_{int(n)}")


KITTI_RELPOSE_IMAGE_GATHERED: str = _j(
    "data", "kitti", "depth_selection", "val_selection_cropped", "image_gathered"
)
KITTI_SEQUENCES: str = _j("data", "kitti", "sequences")
KITTI_POSES: str = _j("data", "kitti", "poses")
KITTI_DEPTH: str = _j("data", "kitti", "depth")

SEVEN_SCENES: str = _j("data", "7scenes")
NRGBD: str = _j("data", "nrgbd")
TUM: str = _j("data", "tum")

PLY_OUTPUT_DEFAULT: str = _j("eval_results", "mv_recon", "kitti_scannet_ply")
