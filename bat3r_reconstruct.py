#!/usr/bin/env python3
"""
BAT3R Video Reconstruction → Static PNG Renderer

One-shot: video/image sequence → 3D point cloud + camera frustums → PNG images

Usage:
    python bat3r_reconstruct.py \
        --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/taylor \
        --render_dir ./renders \
        --render_views 8 \
        --render_width 1920 \
        --render_height 1080

For headless server (no display), add:
    xvfb-run -a python bat3r_reconstruct.py ...
"""

import os
import sys
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
import re
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to path (same as demo.py)
# ---------------------------------------------------------------------------
from add_ckpt_path import add_path_to_dust3r

# ---------------------------------------------------------------------------
# Open3D (required for rendering)
# ---------------------------------------------------------------------------
try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d is required. pip install open3d")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Matplotlib fallback
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
#  Camera frustum geometry (from visualize_ply.py)
# ═══════════════════════════════════════════════════════════════════════════
FRUSTUM_COLOR_RGB = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # neon green


def _frustum_line_segments(c2w, scale=0.12, aspect=1.33, fov_deg=50.0):
    """Return (N,2,3) line segment endpoints in world coords."""
    fh = 2.0 * scale * np.tan(np.radians(fov_deg) / 2.0)
    fw = fh * aspect
    z = float(scale)
    corners_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [-fw / 2, -fh / 2, z],
            [fw / 2, -fh / 2, z],
            [fw / 2, fh / 2, z],
            [-fw / 2, fh / 2, z],
        ],
        dtype=np.float64,
    )
    R, t = c2w[:3, :3].astype(np.float64), c2w[:3, 3].astype(np.float64)
    pts = (R @ corners_cam.T).T + t
    idx = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return np.stack([[pts[a], pts[b]] for a, b in idx], axis=0)


def _frustum_lineset_from_segments(cam_segs):
    if cam_segs is None or len(cam_segs) == 0:
        return None
    pts_list = []
    lines_idx = []
    for i, seg in enumerate(cam_segs):
        p0, p1 = seg[0], seg[1]
        pts_list.append(p0)
        pts_list.append(p1)
        bi = 2 * i
        lines_idx.append([bi, bi + 1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts_list, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines_idx, dtype=np.int32)),
    )
    n = len(lines_idx)
    green = np.tile(FRUSTUM_COLOR_RGB.reshape(1, 3), (n, 1))
    ls.colors = o3d.utility.Vector3dVector(green)
    return ls


def _axis_bounds_points_and_segments(pts_cat, cam_segs):
    lo = pts_cat.min(axis=0)
    hi = pts_cat.max(axis=0)
    if cam_segs is not None and len(cam_segs) > 0:
        s = cam_segs.reshape(-1, 3)
        lo = np.minimum(lo, s.min(axis=0))
        hi = np.maximum(hi, s.max(axis=0))
    ranges = hi - lo
    ranges = np.maximum(ranges, 1e-9)
    return lo, hi, ranges


def _orbit_eye_and_up(center, distance, azim_deg, elev_deg, world_up=None):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    az = np.radians(float(azim_deg))
    el = np.radians(float(elev_deg))
    dx = distance * np.cos(el) * np.sin(az)
    dy = distance * np.cos(el) * np.cos(az)
    dz = distance * np.sin(el)
    offset = np.array([dx, dy, dz], dtype=np.float64)
    eye = center + offset
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        world_up = np.asarray(world_up, dtype=np.float64).reshape(3)
    fn = center - eye
    fn = fn / (np.linalg.norm(fn) + 1e-12)
    r = np.cross(fn, world_up)
    if np.linalg.norm(r) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(fn, world_up)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, fn)
    u = u / (np.linalg.norm(u) + 1e-12)
    return eye, u


# ═══════════════════════════════════════════════════════════════════════════
#  Rendering engines
# ═══════════════════════════════════════════════════════════════════════════
def _prepare_render_buffers(
    pts_list, color_list, conf_list, conf_threshold,
    max_points, poses, pose_stride, frustum_scale, frustum_fov_deg, frustum_aspect
):
    """Merge point clouds and optional camera frustum segments."""
    rng = np.random.default_rng(0)
    all_pts = []
    all_cols = []
    for pts, col, cf in zip(pts_list, color_list, conf_list):
        pts_np = pts.squeeze(0) if pts.ndim == 4 else pts
        col_np = col.squeeze(0) if col.ndim == 4 else col
        cf_np = cf.squeeze(0) if cf.ndim == 3 else cf
        m = cf_np > conf_threshold
        pts_m = pts_np[m]
        col_m = np.clip(col_np[m], 0.0, 1.0)
        if len(pts_m) == 0:
            continue
        n = len(pts_m)
        if n > max_points:
            idx = rng.choice(n, size=max_points, replace=False)
            pts_m = pts_m[idx]
            col_m = col_m[idx]
        all_pts.append(pts_m)
        all_cols.append(col_m)

    if not all_pts:
        return None, None, None

    pts_cat = np.concatenate(all_pts, axis=0)
    cols_cat = np.concatenate(all_cols, axis=0)

    cam_segs = None
    if poses is not None and len(poses) > 0:
        ps = poses[:: max(1, int(pose_stride))]
        segs = []
        for c2w in ps:
            segs.append(
                _frustum_line_segments(
                    c2w, scale=frustum_scale, aspect=frustum_aspect, fov_deg=frustum_fov_deg
                )
            )
        cam_segs = np.concatenate(segs, axis=0)

    return pts_cat, cols_cat, cam_segs


def render_open3d_filament(
    pts_list, color_list, conf_list, conf_threshold,
    out_dir, name_prefix,
    poses=None, pose_stride=1,
    max_points=4_000_000, n_views=8, elevation=22.0,
    point_size=1.5, width=1920, height=1080,
    frustum_scale=0.12, frustum_fov_deg=50.0, frustum_aspect=1.33,
    zoom=None, camera_fov_deg=60.0,
):
    """Open3D Filament OffscreenRenderer (no window needed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = _prepare_render_buffers(
        pts_list, color_list, conf_list, conf_threshold,
        max_points, poses, pose_stride, frustum_scale, frustum_fov_deg, frustum_aspect
    )
    pts_cat, cols_cat, cam_segs = prep
    if pts_cat is None:
        print("  [render] No points to render.")
        return False

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts_cat.astype(np.float64))
    pcd_o3d.colors = o3d.utility.Vector3dVector(cols_cat.astype(np.float64))

    line_set = _frustum_lineset_from_segments(cam_segs)
    lo, hi, ranges = _axis_bounds_points_and_segments(pts_cat, cam_segs)
    center = (lo + hi) / 2.0
    rmax = float(np.max(ranges))
    distance = max(rmax * 2.2, 1e-3)
    if zoom is not None and float(zoom) > 1e-6:
        distance = distance / float(zoom)

    stem = re.sub(r"[^\w\-]+", "_", name_prefix)[:120]
    azims = np.linspace(0.0, 360.0, int(n_views), endpoint=False)

    try:
        rendering = o3d.visualization.rendering
    except AttributeError:
        print("  [render:o3d-filament] Open3D has no rendering module; skipping.")
        return False

    try:
        renderer = rendering.OffscreenRenderer(int(width), int(height))
    except Exception as e:
        print(f"  [render:o3d-filament] OffscreenRenderer failed ({e}); try GLFW or matplotlib.")
        return False

    try:
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size)
        renderer.scene.add_geometry("point_cloud", pcd_o3d, mat)

        if line_set is not None:
            lmat = rendering.MaterialRecord()
            try:
                lmat.shader = "unlitLine"
                if hasattr(lmat, "line_width"):
                    lmat.line_width = 1.5
                renderer.scene.add_geometry("camera_lines", line_set, lmat)
            except Exception:
                try:
                    lmat.shader = "defaultUnlit"
                    renderer.scene.add_geometry("camera_lines", line_set, lmat)
                except Exception:
                    print("  [render:o3d-filament] Could not add LineSet; drawing points only.")

        for vi, azim in enumerate(azims):
            eye, up = _orbit_eye_and_up(center, distance, azim, elevation)
            renderer.setup_camera(
                float(camera_fov_deg),
                center.astype(np.float64),
                eye.astype(np.float64),
                up.astype(np.float64),
            )
            img = renderer.render_to_image()
            out_path = out_dir / f"{stem}_view{vi:02d}_az{int(azim + 0.5) % 360:03d}.png"
            o3d.io.write_image(str(out_path), img)
            print(f"  [render:o3d-filament] {out_path}")

        print(f"  [render:o3d-filament] Wrote {len(azims)} image(s) under {out_dir.resolve()}")
        return True
    except Exception as e:
        print(f"  [render:o3d-filament] Render failed ({e}); try GLFW or matplotlib.")
        return False
    finally:
        try:
            del renderer
        except Exception:
            pass


def _o3d_flush_renderer(vis, n=12):
    for _ in range(n):
        vis.poll_events()
        vis.update_renderer()
    time.sleep(0.06)


def _save_o3d_frame(vis, path):
    _o3d_flush_renderer(vis, 10)
    path = str(path)
    try:
        fbuf = vis.capture_screen_float_buffer(True)
        o3d.io.write_image(path, fbuf)
    except Exception:
        vis.capture_screen_image(path)


def render_open3d_visualizer(
    pts_list, color_list, conf_list, conf_threshold,
    out_dir, name_prefix,
    poses=None, pose_stride=1,
    max_points=4_000_000, n_views=8, elevation=22.0,
    point_size=1.5, width=1920, height=1080,
    frustum_scale=0.12, frustum_fov_deg=50.0, frustum_aspect=1.33,
    zoom=None,
):
    """Legacy Open3D Visualizer + GLFW/OSMesa (needs working GL stack)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = _prepare_render_buffers(
        pts_list, color_list, conf_list, conf_threshold,
        max_points, poses, pose_stride, frustum_scale, frustum_fov_deg, frustum_aspect
    )
    pts_cat, cols_cat, cam_segs = prep
    if pts_cat is None:
        print("  [render] No points to render.")
        return False

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts_cat.astype(np.float64))
    pcd_o3d.colors = o3d.utility.Vector3dVector(cols_cat.astype(np.float64))

    line_set = _frustum_lineset_from_segments(cam_segs)

    stem = re.sub(r"[^\w\-]+", "_", name_prefix)[:120]
    azims = np.linspace(0.0, 360.0, int(n_views), endpoint=False)

    vis = None
    try:
        vis = o3d.visualization.Visualizer()
        try:
            win_ok = vis.create_window(
                visible=False, width=int(width), height=int(height), left=50, top=50
            )
        except Exception as e:
            print(f"  [render] Open3D window failed ({e}). Try: xvfb-run -a ... or --render-engine matplotlib")
            return False

        if win_ok is False:
            print("  [render] Open3D failed to create window (headless / OSMesa missing).")
            return False

        opt = vis.get_render_option()
        if opt is None:
            print("  [render] Open3D has no render context.")
            return False

        vis.add_geometry(pcd_o3d)
        if line_set is not None:
            vis.add_geometry(line_set)

        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = float(point_size)
        opt.show_coordinate_frame = False
        if hasattr(opt, "line_width"):
            opt.line_width = 1.0

        _o3d_flush_renderer(vis, 24)

        for vi, azim in enumerate(azims):
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            if ctr is None:
                print("  [render] Open3D ViewControl unavailable.")
                return False
            ctr.rotate(float(azim), float(elevation))
            if zoom is not None:
                try:
                    ctr.set_zoom(float(zoom))
                except Exception:
                    pass
            _o3d_flush_renderer(vis, 6)
            out_path = out_dir / f"{stem}_view{vi:02d}_az{int(azim + 0.5) % 360:03d}.png"
            _save_o3d_frame(vis, out_path)
            print(f"  [render:o3d] {out_path}")

        print(f"  [render:o3d] Wrote {len(azims)} image(s) under {out_dir.resolve()}")
        return True
    finally:
        if vis is not None:
            try:
                vis.destroy_window()
            except Exception:
                pass


def render_matplotlib(
    pts_list, color_list, conf_list, conf_threshold,
    out_dir, name_prefix,
    poses=None, pose_stride=1,
    max_points=4_000_000, n_views=8, elevation=22.0,
    point_size=0.22, dpi=220, fig_size=(11.0, 9.0),
    frustum_scale=0.12, frustum_fov_deg=50.0, frustum_aspect=1.33,
    margin=0.015,
):
    """Matplotlib fallback: white background, point cloud + neon green camera frustums."""
    if not HAS_MPL:
        print("  [render:mpl] matplotlib not available.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = _prepare_render_buffers(
        pts_list, color_list, conf_list, conf_threshold,
        max_points, poses, pose_stride, frustum_scale, frustum_fov_deg, frustum_aspect
    )
    pts_cat, cols_cat, cam_segs = prep
    if pts_cat is None:
        print("  [render] No points to render.")
        return

    lo, hi, ranges = _axis_bounds_points_and_segments(pts_cat, cam_segs)
    pad = float(margin) * ranges
    xmin, xmax = lo[0] - pad[0], hi[0] + pad[0]
    ymin, ymax = lo[1] - pad[1], hi[1] + pad[1]
    zmin, zmax = lo[2] - pad[2], hi[2] + pad[2]
    draw_ranges = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
    draw_ranges = np.maximum(draw_ranges, 1e-9)

    azims = np.linspace(0.0, 360.0, int(n_views), endpoint=False)
    stem = re.sub(r"[^\w\-]+", "_", name_prefix)[:120]

    for vi, azim in enumerate(azims):
        fig = plt.figure(figsize=fig_size, facecolor="white")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        ax.scatter(
            pts_cat[:, 0], pts_cat[:, 1], pts_cat[:, 2],
            s=point_size, c=cols_cat, linewidths=0, alpha=1.0,
            depthshade=False, rasterized=True,
        )
        if cam_segs is not None and len(cam_segs) > 0:
            lc = Line3DCollection(
                list(cam_segs),
                colors=(float(FRUSTUM_COLOR_RGB[0]), float(FRUSTUM_COLOR_RGB[1]),
                        float(FRUSTUM_COLOR_RGB[2]), 1.0),
                linewidths=0.65,
            )
            ax.add_collection3d(lc)

        ax.view_init(elev=elevation, azim=float(azim))
        ax.set_axis_off()
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_visible(False)
        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        try:
            ax.set_box_aspect(tuple(draw_ranges))
        except Exception:
            pass
        try:
            ax.margins(0)
        except Exception:
            pass
        try:
            ax.dist = 7.0
        except Exception:
            pass

        out_path = out_dir / f"{stem}_view{vi:02d}_az{int(azim + 0.5) % 360:03d}.png"
        fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        print(f"  [render:mpl] {out_path}")

    print(f"  [render:mpl] Wrote {len(azims)} image(s) under {out_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo.py inference helpers (copied verbatim)
# ═══════════════════════════════════════════════════════════════════════════
framerate = 30


def inject_camera_poses_into_ply(ply_path, camera_poses):
    import base64
    import io as _io
    ply_path = os.path.abspath(ply_path)
    buf = _io.BytesIO()
    np.save(buf, np.asarray(camera_poses, dtype=np.float64))
    raw = buf.getvalue()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    lines = ["comment TTT3R_CAMERAS_B64_BEGIN"]
    for i in range(0, len(b64), 120):
        lines.append(f"comment TTT3R_CAM_DATA {b64[i : i + 120]}")
    lines.append("comment TTT3R_CAMERAS_B64_END")
    inserted = ("\n".join(lines) + "\n").encode("ascii")
    with open(ply_path, "rb") as f:
        data = f.read()
    idx = data.find(b"end_header")
    if idx < 0:
        return
    new_data = data[:idx] + inserted + data[idx:]
    with open(ply_path, "wb") as f:
        f.write(new_data)


def prepare_input(img_paths, img_mask, size, raymaps=None, raymap_mask=None,
                  revisit=1, update=True, reset_interval=10000):
    from src.dust3r.utils.image import load_images
    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (images[i]["img"].shape[0], 6,
                     images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
            }
            views.append(view)
            if (i + 1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        raise NotImplementedError("raymaps mode not supported in render script")

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views
    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True,
                   ply_out=None, conf_threshold=1.5):
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat(
        [torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0
    )

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask
    ]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask
    ]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    conf_self_ls = [output["conf_self"].cpu() for output in outputs["pred"]]

    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses_tensor = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses_tensor.device)
        reset_poses = torch.where(
            reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses_tensor, identity
        )
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat(
            [identity.unsqueeze(0), cumulative_bases[:-1]], dim=0
        )
        pr_poses_tensor = torch.einsum("bij,bjk->bik", shifted_bases, pr_poses_tensor)
        pr_poses = list(pr_poses_tensor.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        pts3ds_other_ls = []
        for pose, pself in zip(pr_poses, pts3ds_self_ls):
            pts3ds_other_ls.append(geotrf(pose, pself))
        conf_other_ls = conf_self_ls
    else:
        pts3ds_other_ls = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_other_ls = [output["conf"].cpu() for output in outputs["pred"]]

    H, W, _ = pts3ds_self_ls[0].shape[1:]
    pp_device = pts3ds_self_ls[0].device
    pp_single = torch.tensor([W // 2, H // 2], device=pp_device).float()
    focals = []
    for pself in pts3ds_self_ls:
        pp_batch = pp_single.unsqueeze(0)
        f = estimate_focal_knowing_depth(pself, pp_batch, focal_mode="weiszfeld")
        focals.append(f)
    focal = torch.cat(focals, dim=0)
    B = len(pts3ds_self_ls)
    pp = pp_single.unsqueeze(0).repeat(B, 1)

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    os.makedirs(outdir, exist_ok=True)

    # Save merged PLY
    ply_path = ply_out if ply_out is not None else os.path.join(outdir, "video_merged.ply")
    camera_poses = np.stack(
        [p.squeeze(0).detach().cpu().numpy().astype(np.float64) for p in pr_poses],
        axis=0,
    )
    pts_merge_list = []
    col_merge_list = []
    for pts_w, col, cf in zip(pts3ds_other_ls, colors, conf_other_ls):
        pts_np = pts_w.squeeze(0).detach().cpu().numpy()
        col_np = col.squeeze(0).detach().cpu().numpy()
        cf_np = cf.squeeze(0).detach().cpu().numpy()
        m = cf_np > conf_threshold
        pts_merge_list.append(pts_np[m])
        col_merge_list.append(np.clip(col_np[m], 0.0, 1.0))
    if pts_merge_list:
        all_pts = np.concatenate(pts_merge_list, axis=0)
        all_cols = np.concatenate(col_merge_list, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_cols)
        o3d.io.write_point_cloud(ply_path, pcd)
        inject_camera_poses_into_ply(ply_path, camera_poses)
        print(f"Merged PLY saved ({len(all_pts)} pts, {camera_poses.shape[0]} cams): {ply_path}")
    else:
        print("Warning: no points passed conf threshold; PLY not written.")

    return pts3ds_other_ls, colors, conf_other_ls, cam_dict, camera_poses


def parse_seq_path(p, frame_interval=1):
    global framerate
    if os.path.isdir(p):
        all_img_paths = sorted(glob.glob(f"{p}/*"))
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        img_paths = [
            path for path in all_img_paths
            if os.path.splitext(path.lower())[1] in img_extensions
        ]
        if not img_paths:
            raise ValueError(f"No image files found in directory {p}")
        if frame_interval > 1:
            img_paths = img_paths[::frame_interval]
            print(f" - Image sequence: Total: {len(all_img_paths)}, "
                  f"interval: {frame_interval}, process: {len(img_paths)}")
        framerate = 30.0 / frame_interval
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        framerate = video_fps / frame_interval
        frame_indices = list(range(0, total_frames, frame_interval))
        print(f" - Video FPS: {video_fps}, Interval: {frame_interval}, "
              f"Frames: {len(frame_indices)}, FPS: {framerate}")
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(
        description="BAT3R: video/image → point cloud + camera frustums → static PNG"
    )
    # Model / data args (same as demo.py)
    parser.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--seq_path", type=str, required=True,
                        help="Path to video file or image sequence directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--output_dir", type=str, default="./render_tmp")
    parser.add_argument("--model_update_type", type=str, default="bat3r")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=1000000)
    parser.add_argument("--ply_out", type=str, default=None)

    # Rendering args (new)
    parser.add_argument("--render_dir", type=str, required=True,
                        help="Directory to save rendered PNG images")
    parser.add_argument("--render_engine", choices=["filament", "glfw", "matplotlib"],
                        default="filament",
                        help="filament=Open3D offscreen (default); glfw=Open3D window; matplotlib=fallback")
    parser.add_argument("--render_views", type=int, default=8,
                        help="Number of azimuth views around scene (default 8)")
    parser.add_argument("--render_elevation", type=float, default=22.0)
    parser.add_argument("--render_point_size", type=float, default=1.5)
    parser.add_argument("--render_width", type=int, default=1920)
    parser.add_argument("--render_height", type=int, default=1080)
    parser.add_argument("--render_zoom", type=float, default=None,
                        help="Zoom factor: smaller = wider FOV / more scene visible (typical 0.4-0.9)")
    parser.add_argument("--render_dpi", type=int, default=220,
                        help="DPI for matplotlib only")
    parser.add_argument("--max_points", type=int, default=4_000_000,
                        help="Max points to render (random subsample if larger)")
    parser.add_argument("--pose_stride", type=int, default=1,
                        help="Draw every k-th camera frustum")
    parser.add_argument("--frustum_scale", type=float, default=0.12)
    parser.add_argument("--frustum_fov", type=float, default=50.0)
    parser.add_argument("--frustum_aspect", type=float, default=1.33)
    parser.add_argument("--conf_threshold", type=float, default=1.5,
                        help="Confidence threshold for filtering points")
    parser.add_argument("--no_ply", action="store_true",
                        help="Skip PLY file generation, render only")
    return parser.parse_args()


def main():
    args = get_args()

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add ckpt path
    add_path_to_dust3r(args.model_path)

    # Imports (delayed after ckpt path)
    from bat3r.bat3r_inference import inference_recurrent_lighter
    from bat3r.bat3r_model import ARCroco3DStereo

    # Parse input
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval)
    if not img_paths:
        print(f"No images found in {args.seq_path}")
        return
    print(f"Found {len(img_paths)} images.")

    # Prepare views
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        reset_interval=args.reset_interval,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()

    # Inference
    print("Running inference...")
    start = time.time()
    outputs, state_args = inference_recurrent_lighter(views, model, device)
    elapsed = time.time() - start
    fps = len(views) / elapsed
    print(f"Inference done: {elapsed:.2f}s ({fps:.2f} FPS)")

    # Prepare output
    print("Preparing point clouds and camera poses...")
    if args.no_ply:
        ply_path = None
    else:
        ply_path = args.ply_out or os.path.join(args.output_dir, "video_merged.ply")
    pts3ds_other, colors, conf, cam_dict, camera_poses = prepare_output(
        outputs, args.output_dir, 1, True,
        ply_out=ply_path, conf_threshold=args.conf_threshold,
    )

    # Convert to numpy lists for rendering
    pts_np_list = [p.cpu().numpy() for p in pts3ds_other]
    col_np_list = [c.cpu().numpy() for c in colors]
    conf_np_list = [c.cpu().numpy() for c in conf]

    # Determine name prefix from input
    seq_name = Path(args.seq_path).stem

    # Render
    print(f"\nRendering {args.render_views} views to {args.render_dir} ...")
    if args.render_engine == "filament":
        ok = render_open3d_filament(
            pts_np_list, col_np_list, conf_np_list, args.conf_threshold,
            out_dir=args.render_dir,
            name_prefix=seq_name,
            poses=camera_poses,
            pose_stride=args.pose_stride,
            max_points=args.max_points,
            n_views=args.render_views,
            elevation=args.render_elevation,
            point_size=args.render_point_size,
            width=args.render_width,
            height=args.render_height,
            frustum_scale=args.frustum_scale,
            frustum_fov_deg=args.frustum_fov,
            frustum_aspect=args.frustum_aspect,
            zoom=args.render_zoom,
        )
        if not ok:
            print("Filament failed, trying GLFW...")
            ok = render_open3d_visualizer(
                pts_np_list, col_np_list, conf_np_list, args.conf_threshold,
                out_dir=args.render_dir,
                name_prefix=seq_name,
                poses=camera_poses,
                pose_stride=args.pose_stride,
                max_points=args.max_points,
                n_views=args.render_views,
                elevation=args.render_elevation,
                point_size=args.render_point_size,
                width=args.render_width,
                height=args.render_height,
                frustum_scale=args.frustum_scale,
                frustum_fov_deg=args.frustum_fov,
                frustum_aspect=args.frustum_aspect,
                zoom=args.render_zoom,
            )
        if not ok:
            print("Open3D failed, falling back to matplotlib...")
            render_matplotlib(
                pts_np_list, col_np_list, conf_np_list, args.conf_threshold,
                out_dir=args.render_dir,
                name_prefix=seq_name,
                poses=camera_poses,
                pose_stride=args.pose_stride,
                max_points=args.max_points,
                n_views=args.render_views,
                elevation=args.render_elevation,
                point_size=0.22,
                dpi=args.render_dpi,
                frustum_scale=args.frustum_scale,
                frustum_fov_deg=args.frustum_fov,
                frustum_aspect=args.frustum_aspect,
            )
    elif args.render_engine == "glfw":
        ok = render_open3d_visualizer(
            pts_np_list, col_np_list, conf_np_list, args.conf_threshold,
            out_dir=args.render_dir,
            name_prefix=seq_name,
            poses=camera_poses,
            pose_stride=args.pose_stride,
            max_points=args.max_points,
            n_views=args.render_views,
            elevation=args.render_elevation,
            point_size=args.render_point_size,
            width=args.render_width,
            height=args.render_height,
            frustum_scale=args.frustum_scale,
            frustum_fov_deg=args.frustum_fov,
            frustum_aspect=args.frustum_aspect,
            zoom=args.render_zoom,
        )
        if not ok:
            print("GLFW failed, falling back to matplotlib...")
            render_matplotlib(
                pts_np_list, col_np_list, conf_np_list, args.conf_threshold,
                out_dir=args.render_dir,
                name_prefix=seq_name,
                poses=camera_poses,
                pose_stride=args.pose_stride,
                max_points=args.max_points,
                n_views=args.render_views,
                elevation=args.render_elevation,
                point_size=0.22,
                dpi=args.render_dpi,
                frustum_scale=args.frustum_scale,
                frustum_fov_deg=args.frustum_fov,
                frustum_aspect=args.frustum_aspect,
            )
    else:  # matplotlib
        render_matplotlib(
            pts_np_list, col_np_list, conf_np_list, args.conf_threshold,
            out_dir=args.render_dir,
            name_prefix=seq_name,
            poses=camera_poses,
            pose_stride=args.pose_stride,
            max_points=args.max_points,
            n_views=args.render_views,
            elevation=args.render_elevation,
            point_size=0.22,
            dpi=args.render_dpi,
            frustum_scale=args.frustum_scale,
            frustum_fov_deg=args.frustum_fov,
            frustum_aspect=args.frustum_aspect,
        )

    print(f"\nDone! Images saved to: {Path(args.render_dir).resolve()}")


if __name__ == "__main__":
    main()
