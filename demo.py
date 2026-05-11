#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512

Outputs merged PLY at OUTPUT_DIR/video_merged.ply (points + cameras in header); use view_ply_local.py to view locally.
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import datetime
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_multiotsu
from einops import rearrange

try:
    import open3d as o3d
except ImportError:
    o3d = None

# Set random seed for reproducibility.
random.seed(42)


def inject_camera_poses_into_ply(ply_path, camera_poses):
    """在 PLY 头 end_header 前插入 comment，嵌入 np.save 格式的 N×4×4 camera_poses（与 TTT3R/view_ply_local 一致）。"""
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
        print(f"Warning: could not inject cameras into PLY header: {ply_path}")
        return
    new_data = data[:idx] + inserted + data[idx:]
    with open(ply_path, "wb") as f:
        f.write(new_data)

framerate = 30

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="examples/taylor",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port for the point cloud viewer",
    )
    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help="model update type: cut3r or ttt3r",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Frame interval for video processing (e.g., 1 means every frame, 2 means every other frame)",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=1000000,
        help="Only used for demo, reset state for extremely long sequence, chunks are aligned via global camera poses",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample factor for the point cloud viewer",
    )
    parser.add_argument(
        "--ply_out",
        type=str,
        default=None,
        help="Merged video point cloud + cameras in one PLY (default: OUTPUT_DIR/video_merged.ply)",
    )
    return parser.parse_args()


def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True, reset_interval=10000
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

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


def prepare_output(
    outputs,
    outdir,
    revisit=1,
    use_pose=True,
    ply_out=None,
    conf_threshold=1.5,
):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        ply_out (str, optional): Path to write merged PLY (points + cameras in header). Default outdir/video_merged.ply.
        conf_threshold (float): Confidence threshold when merging points into PLY.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    import roma
    # from viser_utils import convert_scene_output_to_glb


    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    # Use lists instead of cat-ed tensors to save memory
    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    # pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]] # Not needed if use_pose=True
    conf_self_ls = [output["conf_self"].cpu() for output in outputs["pred"]]
    
    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # Convert sequence_scale list
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        pts3ds_other_ls = []
        for pose, pself in zip(pr_poses, pts3ds_self_ls):
            # pself is (1, H, W, 3), pose is (1, 4, 4)
            pts3ds_other_ls.append(geotrf(pose, pself))
        conf_other_ls = conf_self_ls
    else:
        pts3ds_other_ls = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
        conf_other_ls = [output["conf"].cpu() for output in outputs["pred"]]

    # Estimate focal length based on depth per frame to save memory
    focals = []
    # Assume all frames have same H, W
    H, W, _ = pts3ds_self_ls[0].shape[1:] 
    
    # Pre-calculate pp for the first frame (assuming constant size)
    pp_device = pts3ds_self_ls[0].device
    pp_single = torch.tensor([W // 2, H // 2], device=pp_device).float()
    
    for pself in pts3ds_self_ls:
        # pself shape: (1, H, W, 3)
        pp_batch = pp_single.unsqueeze(0) # (1, 2)
        f = estimate_focal_knowing_depth(pself, pp_batch, focal_mode="weiszfeld")
        focals.append(f)
    
    focal = torch.cat(focals, dim=0) # (B,)
    
    # Re-construct pp for cam_dict (B, 2)
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

    # ----- 原：逐帧保存 depth / conf / color / camera（已停用，保留备查）-----
    # if os.path.exists(os.path.join(outdir, "depth")):
    #     shutil.rmtree(os.path.join(outdir, "depth"))
    # if os.path.exists(os.path.join(outdir, "conf")):
    #     shutil.rmtree(os.path.join(outdir, "conf"))
    # if os.path.exists(os.path.join(outdir, "color")):
    #     shutil.rmtree(os.path.join(outdir, "color"))
    # if os.path.exists(os.path.join(outdir, "camera")):
    #     shutil.rmtree(os.path.join(outdir, "camera"))
    # os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    #
    # for f_id, (pts, conf, color, pose, f_val) in enumerate(
    #     zip(pts3ds_self_ls, conf_self_ls, colors, pr_poses, focal)
    # ):
    #     depth = pts[..., 2].squeeze(0).cpu().numpy()
    #     conf_np = conf.squeeze(0).cpu().numpy()
    #     color_np = (color.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    #     c2w = pose.squeeze(0).cpu().numpy()
    #     intrins = np.eye(3)
    #     intrins[0, 0] = f_val.item()
    #     intrins[1, 1] = f_val.item()
    #     intrins[0, 2] = pp_single[0].item()
    #     intrins[1, 2] = pp_single[1].item()
    #     np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
    #     np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf_np)
    #     iio.imwrite(os.path.join(outdir, "color", f"{f_id:06d}.png"), color_np)
    #     np.savez(
    #         os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
    #         pose=c2w,
    #         intrinsics=intrins,
    #     )

    # 整段视频：合并世界坐标点云 + 全部相机位姿 → 单个 PLY（相机写入头 comment，可用 view_ply_local 查看）
    if o3d is None:
        print(
            "Warning: open3d not installed; skipping merged PLY. pip install open3d"
        )
    else:
        ply_path = (
            ply_out
            if ply_out is not None
            else os.path.join(outdir, "video_merged.ply")
        )
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
            print(
                f"Merged PLY saved ({len(all_pts)} points, {camera_poses.shape[0]} cameras): {ply_path}"
            )
        else:
            print("Warning: no points passed conf threshold; PLY not written.")

    return pts3ds_other_ls, colors, conf_other_ls, cam_dict

def parse_seq_path(p, frame_interval=1):
    global framerate
    
    if os.path.isdir(p):
        all_img_paths = sorted(glob.glob(f"{p}/*"))
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        img_paths = [path for path in all_img_paths 
                    if os.path.splitext(path.lower())[1] in img_extensions]
        
        if not img_paths:
            raise ValueError(f"No image files found in directory {p}")
        
        if frame_interval > 1:
            img_paths = img_paths[::frame_interval]
            print(f" - Image sequence: Total images: {len(all_img_paths)}, "
                  f"Frame interval: {frame_interval}, Images to process: {len(img_paths)}")
        
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
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}, Processed Framerate: {framerate}"
        )
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


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        reset_interval=args.reset_interval
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type

    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference_recurrent_lighter(views, model, device)

    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    FPS_num = 1 / per_frame_time
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame), FPS: {FPS_num:.2f}."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    ply_out = args.ply_out
    if ply_out is None:
        ply_out = os.path.join(args.output_dir, "video_merged.ply")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs,
        args.output_dir,
        1,
        True,
        ply_out=ply_out,
        conf_threshold=args.vis_threshold,
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Save 2D grid visualization (RGB | depth | confidence per frame).
    # from viser_utils import save_visualization
    # vis_save_path = os.path.join(args.output_dir, "visualization.png")
    # save_visualization(
    #     pts3d_list=pts3ds_to_vis,
    #     color_list=colors_to_vis,
    #     conf_list=conf,
    #     cam_dict=cam_dict,
    #     save_path=vis_save_path,
    #     vis_threshold=args.vis_threshold,
    #     max_frames=16,
    #     cell_size=256,
    # )

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size,
        port = args.port,
        downsample_factor=args.downsample_factor
    )
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
