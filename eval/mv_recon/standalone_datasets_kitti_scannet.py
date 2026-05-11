# 独立：ScanNet / KITTI -> mv_recon 风格多视图，供 launch_kitti_scannet_ply 使用。不修改 data.py / launch.py。
from __future__ import annotations

import glob
import os
import os.path as osp
import re
from collections import deque

import cv2
import numpy as np
import PIL.Image
from eval.mv_recon.base import BaseStereoViewDataset
import eval.mv_recon.dataset_utils.cropping as cropping
from dust3r.utils.image import imread_cv2


def _parse_pose_line_4x4(line: str) -> np.ndarray:
    vals = [float(x) for x in line.split()]
    if len(vals) == 16:
        m = np.array(vals, dtype=np.float32).reshape(4, 4)
    elif len(vals) == 12:
        m3 = np.array(vals, dtype=np.float32).reshape(3, 4)
        m = np.eye(4, dtype=np.float32)
        m[:3, :4] = m3
    else:
        raise ValueError(f"pose line 需要 12 或 16 个数，得到 {len(vals)}: {line[:80]}")
    return m


def _load_scannet_poses(posepath: str) -> list[np.ndarray]:
    with open(posepath) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [_parse_pose_line_4x4(ln) for ln in lines]


def _scannet_depth_to_meters(depth_raw: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth_raw.astype(np.float32), 0.0)
    d = d / 1000.0
    d[d > 10.0] = 0.0
    d[d < 1e-3] = 0.0
    return d


# --------------------------------------------------------------------------- ScanNet
class ScanNetPLYDataset(BaseStereoViewDataset):
    """
    期望目录（与 ``datasets_preprocess/long_prepare_scannet.py`` / relpose ``scannet_s3_N`` 一致）：
      {ROOT}/{scene}/
        color_{N}/   frame_0000.jpg ...
        depth_{N}/   frame_0000.png   (mm uint16)
        pose_{N}.txt 每行 12 或 16 个 float（默认 w2c -> inv 为 c2w，见 invert_pose_to_cam2world）
    其中 N 为 ``n_frames``（如 90、100、500，与 `color_90` 即 N=90 一致）。
    """

    def __init__(
        self,
        full_video: bool = True,
        kf_every: int = 1,
        max_frames: int | None = None,
        n_frames: int = 1000,
        scene_ids: list[str] | None = None,
        invert_pose_to_cam2world: bool = True,
        input_aug: str = "none",  # none | scannet
        *args,
        ROOT: str,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.n_frames = int(n_frames)
        super().__init__(*args, **kwargs)
        self.full_video = full_video
        self.kf_every = max(1, int(kf_every))
        self.max_frames = max_frames
        self.invert_pose_to_cam2world = invert_pose_to_cam2world
        self.input_aug = input_aug
        if scene_ids is not None:
            self.scene_list = list(scene_ids)
        else:
            self.scene_list = self._discover_scenes(ROOT, self.n_frames)
        if not self.scene_list:
            raise FileNotFoundError(
                f"在 {ROOT} 下未找到含 color_{self.n_frames}/ depth_{self.n_frames}/ "
                f"pose_{self.n_frames}.txt 的场景"
            )

    def _color_dir_name(self) -> str:
        return f"color_{self.n_frames}"

    def _depth_dir_name(self) -> str:
        return f"depth_{self.n_frames}"

    def _pose_name(self) -> str:
        return f"pose_{self.n_frames}.txt"

    def _discover_scenes(self, base: str, n: int) -> list[str]:
        cdir, ddir, pfile = f"color_{n}", f"depth_{n}", f"pose_{n}.txt"
        out: list[str] = []
        for name in sorted(os.listdir(base)):
            p = osp.join(base, name)
            if not os.path.isdir(p):
                continue
            if not (
                osp.isdir(osp.join(p, cdir))
                and osp.isdir(osp.join(p, ddir))
                and osp.isfile(osp.join(p, pfile))
            ):
                continue
            out.append(name)
        return out

    def __len__(self) -> int:
        return len(self.scene_list)

    def _get_views(self, idx: int, resolution, rng):
        scene_id = self.scene_list[idx]
        sp = osp.join(self.ROOT, scene_id)
        color_dir = osp.join(sp, self._color_dir_name())
        depth_dir = osp.join(sp, self._depth_dir_name())
        posepath = osp.join(sp, self._pose_name())

        imgs = sorted(
            glob.glob(osp.join(color_dir, "*")),
            key=lambda x: int(re.search(r"(\d+)", osp.basename(x)).group(1)),
        )
        imgs = imgs[:: self.kf_every]
        if self.max_frames is not None:
            imgs = imgs[: self.max_frames]

        poses_all = _load_scannet_poses(posepath)
        w0, h0 = 640, 480
        fx, fy, cx, cy = 577.59, 578.73, 318.91, 242.75
        intr0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views: list[dict] = []
        for cpath in imgs:
            base = osp.splitext(osp.basename(cpath))[0]
            m = re.search(r"(\d+)", base)
            fi = int(m.group(1)) if m else 0
            dpath = osp.join(depth_dir, f"frame_{fi:04d}.png")
            if not os.path.isfile(dpath):
                alt = glob.glob(osp.join(depth_dir, f"*{fi:04d}*.png"))
                dpath = alt[0] if alt else dpath
            if not os.path.isfile(dpath):
                continue
            if fi >= len(poses_all):
                break

            rgb = imread_cv2(cpath)
            if rgb is None or rgb.size == 0:
                continue
            depthmap = imread_cv2(dpath, cv2.IMREAD_UNCHANGED)
            depthmap = _scannet_depth_to_meters(depthmap)
            if depthmap.shape[0] != rgb.shape[0] or depthmap.shape[1] != rgb.shape[1]:
                rgb = cv2.resize(rgb, (depthmap.shape[1], depthmap.shape[0]))
            h, w = depthmap.shape[:2]
            sx, sy = w / w0, h / h0
            intrinsics_ = np.array(
                [
                    [fx * sx, 0, cx * sx],
                    [0, fy * sy, cy * sy],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            w2c = poses_all[fi]
            if not np.isfinite(w2c).all():
                continue
            camera_pose = np.linalg.inv(w2c) if self.invert_pose_to_cam2world else w2c
            if not np.isfinite(camera_pose).all():
                continue

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb, depthmap, intrinsics_, resolution, rng=rng, info=cpath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb, depthmap, intrinsics_, (512, 384), rng=rng, info=cpath
                )
                W, H = rgb_image.size
                cx0, cy0 = W // 2, H // 2
                l, t = cx0 - 112, cy0 - 112
                r, b = cx0 + 112, cy0 + 112
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, (l, t, r, b)
                )

            if self.input_aug == "scannet":
                from eval.relpose.relpose_input_aug import adapt_scannet

                arr = np.asarray(rgb_image).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], -1)
                arr = adapt_scannet(arr)
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                rgb_image = PIL.Image.fromarray(arr)

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics,
                    dataset="scannet",
                    label=osp.join(scene_id, f"{fi:04d}"),
                    instance=cpath,
                )
            )

        if not views:
            raise RuntimeError(f"无有效视角: {scene_id}")
        return views


# --------------------------------------------------------------------------- KITTI
def _read_kitti_P2(calib_path: str) -> np.ndarray:
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.split()[1:13]]
                p = np.array(vals, dtype=np.float32).reshape(3, 4)
                return p
    raise FileNotFoundError(f"P2: 未找到于 {calib_path}")


def _P2_to_intrinsic(P2: np.ndarray, w_ref: int, h_ref: int, w: int, h: int) -> np.ndarray:
    sx, sy = w / float(w_ref), h / float(h_ref)
    fu = P2[0, 0] * sx
    fv = P2[1, 1] * sy
    cu = P2[0, 2] * sx
    cv = P2[1, 2] * sy
    return np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)


def _kitti_depth_meters(d16: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(d16.astype(np.float32), 0.0) / 256.0
    d[d < 1e-3] = 0.0
    d[d > 80.0] = 0.0
    return d


def _load_kitti_poses_c2w(
    seq_dir: str, seq: str, pose_dir: str | None, n_frames: int, approx_forward: bool
) -> list[np.ndarray]:
    cam2 = osp.join(seq_dir, "cam2world.txt")
    if os.path.isfile(cam2):
        lines = [ln.strip() for ln in open(cam2) if ln.strip()]
        out: list[np.ndarray] = []
        for ln in lines[:n_frames]:
            out.append(_parse_pose_line_4x4(ln).astype(np.float32))
        while len(out) < n_frames:
            out.append(np.eye(4, dtype=np.float32))
        return out
    fpath = None
    if pose_dir:
        fpath = osp.join(pose_dir, f"{seq}.txt")
        if not os.path.isfile(fpath):
            fpath = osp.join(pose_dir, seq, "pose.txt")
    if fpath and os.path.isfile(fpath):
        lines = [ln.strip() for ln in open(fpath) if ln.strip()]
        out: list[np.ndarray] = []
        for ln in lines[:n_frames]:
            m12 = np.array([float(x) for x in ln.split()], dtype=np.float32)
            m4 = np.eye(4, dtype=np.float32)
            m4[:3, :4] = m12.reshape(3, 4)
            # KITTI odom: 行常见为 *world* 齐次点的 cam = P * world，即 world2cam
            c2w = np.linalg.inv(m4)
            out.append(c2w.astype(np.float32))
        while len(out) < n_frames:
            out.append(np.eye(4, dtype=np.float32))
        return out
    if approx_forward:
        out = []
        t = 0.0
        for i in range(n_frames):
            m = np.eye(4, dtype=np.float32)
            m[0, 3] = t
            out.append(m)
            t += 0.1
        return out
    raise FileNotFoundError(
        f"KITTI 位姿: 在 {seq_dir} 下未找到 cam2world.txt，且未提供有效的 pose_dir/poses 文件。请设 --kitti_approx_pose 或提供 odometry 的 poses。"
    )


class KITTIPLYDataset(BaseStereoViewDataset):
    """
    目录约定：
      kitti_image_root/{seq}/*.png  与
      kitti_depth_root/{seq}/*.png  按文件名一一对应（排序后同序）
    可选 kitti_image_root/{seq}/calib.txt 提取 P2；否则用 1242x375 的默认 P2 并按首帧分辨率缩放内参。

    位姿（二选一，优先前者）:
      1) {seq}/cam2world.txt 每行 16 个 float
      2) --kitti_pose_dir/{seq}.txt 为 KITTI odometry 风格每行 12 float（3x4 world2cam）
      3) --kitti_approx_pose：无真值时沿 X 的近似位姿，仅作调试
    """

    def __init__(
        self,
        full_video: bool = True,
        kf_every: int = 1,
        max_frames: int | None = None,
        kitti_image_root: str = "",
        kitti_depth_root: str = "",
        kitti_pose_dir: str | None = None,
        kitti_approx_pose: bool = False,
        kitti_image_subdir: str = "",
        scene_ids: list[str] | None = None,
        w_ref: int = 1242,
        h_ref: int = 375,
        *args,
        **kwargs,
    ):
        self.kitti_image_root = kitti_image_root
        self.kitti_depth_root = kitti_depth_root
        self.kitti_pose_dir = kitti_pose_dir
        self.kitti_approx_pose = kitti_approx_pose
        self.kitti_image_subdir = (kitti_image_subdir or "").strip()
        self.w_ref = w_ref
        self.h_ref = h_ref
        super().__init__(*args, **kwargs)
        self.full_video = full_video
        self.kf_every = max(1, int(kf_every))
        self.max_frames = max_frames
        if scene_ids is not None:
            self.scene_list = list(scene_ids)
        else:
            self.scene_list = sorted(
                d
                for d in os.listdir(kitti_image_root)
                if os.path.isdir(osp.join(kitti_image_root, d))
            )
        if not self.scene_list:
            raise FileNotFoundError(f"在 {kitti_image_root} 下无序列子目录")

    def __len__(self) -> int:
        return len(self.scene_list)

    def _get_views(self, idx: int, resolution, rng):
        seq = self.scene_list[idx]
        idir = osp.join(self.kitti_image_root, seq)
        if self.kitti_image_subdir:
            idir = osp.join(idir, self.kitti_image_subdir)
        ddir = osp.join(self.kitti_depth_root, seq)
        imgs = sorted(
            glob.glob(osp.join(idir, "*.png")) + glob.glob(osp.join(idir, "*.jpg"))
        )
        imgs = imgs[:: self.kf_every]
        if self.max_frames is not None:
            imgs = imgs[: self.max_frames]

        seq_dir = osp.join(self.kitti_image_root, seq)
        calib_path = osp.join(seq_dir, "calib.txt")
        if not os.path.isfile(calib_path):
            calib_path = osp.join(idir, "calib.txt")
        if os.path.isfile(calib_path):
            P2 = _read_kitti_P2(calib_path)
        else:
            P2 = np.array(
                [
                    7.18856e2,
                    0.0,
                    6.071928e2,
                    4.538225e1,
                    0.0,
                    7.18856e2,
                    1.852157e2,
                    -1.100887e-1,
                    0.0,
                    0.0,
                    1.0,
                    3.123772e-3,
                ],
                dtype=np.float32,
            ).reshape(3, 4)

        poses_c2w = _load_kitti_poses_c2w(
            seq_dir, seq, self.kitti_pose_dir, len(imgs), self.kitti_approx_pose
        )

        views: list[dict] = []
        for k, cpath in enumerate(imgs):
            name = osp.splitext(osp.basename(cpath))[0]
            dpath = osp.join(ddir, name + ".png")
            if not os.path.isfile(dpath):
                dpath_alt = glob.glob(osp.join(ddir, f"{name}*"))
                dpath = dpath_alt[0] if dpath_alt else dpath
            if not os.path.isfile(dpath):
                continue
            rgb = imread_cv2(cpath)
            if rgb is None or rgb.size == 0:
                continue
            d16 = imread_cv2(dpath, cv2.IMREAD_UNCHANGED)
            depthmap = _kitti_depth_meters(d16)
            if depthmap.shape[0] != rgb.shape[0] or depthmap.shape[1] != rgb.shape[1]:
                rgb = cv2.resize(rgb, (depthmap.shape[1], depthmap.shape[0]))
            h, w = depthmap.shape[:2]
            intr0 = _P2_to_intrinsic(P2, self.w_ref, self.h_ref, w, h)
            if k >= len(poses_c2w):
                break
            camera_pose = poses_c2w[k]

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb, depthmap, intr0, resolution, rng=rng, info=cpath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb, depthmap, intr0, (512, 384), rng=rng, info=cpath
                )
                W, H = rgb_image.size
                cx0, cy0 = W // 2, H // 2
                l, t = cx0 - 112, cy0 - 112
                r, b = cx0 + 112, cy0 + 112
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, (l, t, r, b)
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="kitti",
                    label=osp.join(seq, name),
                    instance=cpath,
                )
            )

        if not views:
            raise RuntimeError(f"KITTI 无有效视角: {seq}")
        return views
