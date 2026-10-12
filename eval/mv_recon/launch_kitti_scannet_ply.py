# 独立入口：对 ScanNet / KITTI 跑 mv_recon 模型并写出 PLY。不修改 launch.py / data.py。
# 在 BAT3R 仓库根目录执行。路径默认见 eval/mv_recon/bat3r_data_paths.py：
# 权重 src/cut3r_512_dpt_4_64.pth；ScanNet data/long_scannet_s3 + color_N；
# KITTI data/long_kitti_s1/.../image_gathered_N（video_depth kitti_s1_N）。
from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import open3d as o3d
import os.path as osp
import torch
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r
from eval.mv_recon.bat3r_data_paths import (
    DEFAULT_MV_RECON_WEIGHTS,
    KITTI_DEFAULT_GATHER_N,
    KITTI_DEPTH,
    KITTI_POSES,
    KITTI_RELPOSE_IMAGE_GATHERED,
    KITTI_SEQUENCES,
    PLY_OUTPUT_DEFAULT,
    SCANNET_DEFAULT_N,
    SCANNET_LONG_S3,
    kitti_long_depth_gathered,
    kitti_long_image_gathered,
)
from eval.mv_recon.ply_camera_io import inject_camera_poses_into_ply
from eval.mv_recon.standalone_datasets_kitti_scannet import KITTIPLYDataset, ScanNetPLYDataset
from eval.mv_recon.utils import accuracy, completion

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""
regex = re.compile(pattern, re.VERBOSE)


def get_args_parser():
    import argparse

    p = argparse.ArgumentParser("ScanNet & KITTI -> PLY (mv_recon)", add_help=True)
    p.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_MV_RECON_WEIGHTS,
        help="见 bat3r_data_paths.DEFAULT_MV_RECON_WEIGHTS；与 eval 脚本一致",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="（accelerate 会覆盖）")
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--conf_thresh", type=float, default=0.0)
    p.add_argument(
        "--output_dir",
        type=str,
        default=PLY_OUTPUT_DEFAULT,
        help="输出根目录；PLY 在 {output_dir}/{scannet|kitti}/",
    )
    p.add_argument("--size", type=int, default=512, choices=(512, 224))
    p.add_argument("--revisit", type=int, default=1)
    p.add_argument("--freeze", action="store_true")
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="在 kf_every 下采样后最多用多少帧；默认不填=全部使用",
    )
    p.add_argument("--model_update_type", type=str, default="cut3r")
    p.add_argument("--voxel_size", type=float, default=0.0)
    p.add_argument(
        "--dataset",
        type=str,
        default="scannet",
        choices=["scannet", "kitti"],
    )
    p.add_argument(
        "--input_aug",
        type=str,
        default="none",
        choices=["none", "scannet"],
        help="仅 ScanNet：scannet 使用 eval/relpose 的 adapt_scannet",
    )
    p.add_argument(
        "--delete_ply_after_sequence",
        action="store_true",
        help="每序列后删除 *-mask.ply / *-gt.ply",
    )
    p.add_argument(
        "--eval_metrics",
        action="store_true",
        help="写出 PLY 后再跑 ICP/Acc/Comp/法线并写 logs（慢）；默认只导出 PLY",
    )
    p.add_argument(
        "--scannet_root",
        type=str,
        default=SCANNET_LONG_S3,
        help=f"长序列根（relpose scannet_s3_* 对应 long_scannet_s3，默认 {SCANNET_LONG_S3}）",
    )
    p.add_argument(
        "--scannet_n_frames",
        type=int,
        default=SCANNET_DEFAULT_N,
        help="color_N / depth_N / pose_N.txt 中的 N（如 90、100、1000）",
    )
    p.add_argument(
        "--scannet_scenes",
        type=str,
        default="",
        help="逗号分隔；空=自动发现",
    )
    p.add_argument(
        "--scannet_pose_is_cam2world",
        action="store_true",
        help="pose 已是 cam2world 时不加逆；默认 w2c 并逆成 c2w",
    )
    p.add_argument(
        "--kitti_image_root",
        type=str,
        default="",
        help="通常由 kitti_preset 决定；odom 模式见下方 main",
    )
    p.add_argument(
        "--kitti_depth_root",
        type=str,
        default="",
        help="同 kitti_image_root",
    )
    p.add_argument(
        "--kitti_image_subdir",
        type=str,
        default="",
        help="由 kitti_preset 覆写；odom 为 image_2",
    )
    p.add_argument(
        "--kitti_pose_dir",
        type=str,
        default="",
        help="odom=poses 目录；long_s1 默认空，用子目录内 cam2world 或 kitti_approx_pose",
    )
    p.add_argument("--kitti_approx_pose", action="store_true")
    p.add_argument("--kitti_scenes", type=str, default="")
    p.add_argument("--kf_every", type=int, default=1)
    p.add_argument(
        "--kitti_preset",
        type=str,
        default="long_s1",
        choices=["long_s1", "odom", "relpose"],
        help="long_s1=video_depth kitti_s1_N 路径；odom=sequences+image_2；relpose=旧单 gather 目录",
    )
    p.add_argument(
        "--kitti_gather_n",
        type=int,
        default=KITTI_DEFAULT_GATHER_N,
        help="kitti_s1_N 的 N，对应 image_gathered_N / groundtruth_depth_gathered_N",
    )
    return p


def _icp_threshold(name_data: str) -> float:
    if "DTU" in name_data:
        return 100.0
    if name_data == "kitti":
        return 1.0
    return 0.1


def main(args):
    print(f"[paths] weights={args.weights}")
    add_path_to_dust3r(args.weights)
    from bat3r.bat3r_model import ARCroco3DStereo
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError

    if args.dataset == "kitti":
        if args.kitti_preset == "long_s1":
            n = int(args.kitti_gather_n)
            args.kitti_image_root = kitti_long_image_gathered(n)
            args.kitti_depth_root = kitti_long_depth_gathered(n)
            args.kitti_image_subdir = ""
            args.kitti_pose_dir = args.kitti_pose_dir or ""
        elif args.kitti_preset == "relpose":
            args.kitti_image_root = KITTI_RELPOSE_IMAGE_GATHERED
            args.kitti_depth_root = KITTI_DEPTH
            args.kitti_image_subdir = ""
            if not args.kitti_pose_dir:
                args.kitti_pose_dir = ""
        else:
            if not args.kitti_image_root:
                args.kitti_image_root = KITTI_SEQUENCES
            if not args.kitti_depth_root:
                args.kitti_depth_root = KITTI_DEPTH
            if not args.kitti_image_subdir:
                args.kitti_image_subdir = "image_2"
            if not args.kitti_pose_dir:
                args.kitti_pose_dir = KITTI_POSES

    if args.dataset == "kitti":
        print(
            f"[paths] kitti preset={args.kitti_preset} image_root={args.kitti_image_root} "
            f"depth_root={args.kitti_depth_root} pose_dir={args.kitti_pose_dir!r} "
            f"image_subdir={args.kitti_image_subdir!r} gather_n={getattr(args, 'kitti_gather_n', 'n/a')}\n"
        )
    if args.dataset == "scannet":
        print(
            f"[paths] scannet root={args.scannet_root} n_frames={args.scannet_n_frames} "
            f"(color_N/depth_N/pose_N.txt)\n"
        )

    inv_w2c = not args.scannet_pose_is_cam2world
    scn_ids = [s.strip() for s in args.scannet_scenes.split(",") if s.strip()] or None
    kit_ids = [s.strip() for s in args.kitti_scenes.split(",") if s.strip()] or None

    datasets_all = {}
    if args.dataset == "scannet":
        datasets_all["scannet"] = ScanNetPLYDataset(
            split="test",
            ROOT=args.scannet_root,
            resolution=resolution,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
            n_frames=int(args.scannet_n_frames),
            scene_ids=scn_ids,
            invert_pose_to_cam2world=inv_w2c,
            input_aug=args.input_aug,
        )
    else:
        datasets_all["kitti"] = KITTIPLYDataset(
            split="test",
            kitti_image_root=args.kitti_image_root,
            kitti_depth_root=args.kitti_depth_root,
            kitti_pose_dir=args.kitti_pose_dir or None,
            kitti_approx_pose=args.kitti_approx_pose,
            kitti_image_subdir=args.kitti_image_subdir,
            resolution=resolution,
            full_video=True,
            kf_every=args.kf_every,
            max_frames=args.max_frames,
            scene_ids=kit_ids,
        )

    print("\n=== number of scenes ===")
    for name_data, dataset in datasets_all.items():
        for sid in dataset.scene_list:
            print(f"  {name_data}: {sid}")
    print("========================\n")

    accelerator = Accelerator()
    device = accelerator.device

    model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
    model.config.model_update_type = args.model_update_type
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = comp_all = nc1_all = nc2_all = 0.0
            acc_all_med = comp_all_med = nc1_all_med = nc2_all_med = 0.0
            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs, desc=f"{name_data}"):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = {
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    }
                    for view in batch:
                        for name in list(view.keys()):
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], (tuple, list)):
                                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    revisit = args.revisit
                    update = not args.freeze
                    if revisit > 1:
                        new_views = []
                        for r in range(revisit):
                            for i in range(len(batch)):
                                new_view = deepcopy(batch[i])
                                new_view["idx"] = [
                                    (r * len(batch) + i) for _ in range(len(batch[i]["idx"]))
                                ]
                                new_view["instance"] = [
                                    str(r * len(batch) + i) for _ in range(len(batch[i]["instance"]))
                                ]
                                if r > 0 and not update:
                                    new_view["update"] = torch.zeros_like(batch[i]["update"]).bool()
                                new_views.append(new_view)
                        batch = new_views

                    with torch.cuda.amp.autocast(enabled=False):
                        t0 = time.time()
                        output = model(batch)
                        t1 = time.time()
                    preds, batch = output.ress, output.views
                    valid_length = len(preds) // revisit
                    preds = preds[-valid_length:]
                    batch = batch[-valid_length:]

                    fps = len(batch) / (t1 - t0) if t1 > t0 else 0.0
                    fps_all.append(fps)
                    time_all.append(t1 - t0)

                    print(
                        f"Recon {name_data} {data_idx + 1}/{len(dataset)}  FPS: {fps:.2f}"
                    )

                    gt_pts, pred_pts, _, _, _, monitoring = criterion.get_all_pts3d_t(
                        batch, preds
                    )
                    pred_scale, _, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    camera_poses_list = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        camera_poses_list.append(
                            view["camera_pose"][0].detach().cpu().numpy()
                        )

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]
                        pts = pred_pts[j].cpu().numpy()[0]
                        pts_gt = gt_pts[j].cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx, cy = W // 2, H // 2
                        l, t, r, b = cx - 112, cy - 112, cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        pts[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts = geotrf(in_camera1, pts)
                        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts_gt = geotrf(in_camera1, pts_gt)

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]
                    base_name = scene_id.replace("/", "_")
                    camera_poses = np.stack(camera_poses_list, axis=0)

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )

                    if args.voxel_size > 0:
                        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                        pcd_gt = pcd_gt.voxel_down_sample(voxel_size=args.voxel_size)

                    path_mask = osp.join(save_path, f"{base_name}-mask.ply")
                    path_gt = osp.join(save_path, f"{base_name}-gt.ply")
                    o3d.io.write_point_cloud(path_mask, pcd)
                    o3d.io.write_point_cloud(path_gt, pcd_gt)
                    inject_camera_poses_into_ply(path_mask, camera_poses)
                    inject_camera_poses_into_ply(path_gt, camera_poses)
                    print(f"[ply] {base_name} 已写入: {path_mask} , {path_gt}")

                    if args.eval_metrics:
                        threshold = _icp_threshold(name_data)
                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd,
                            pcd_gt,
                            threshold,
                            np.eye(4),
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )
                        transformation = reg_p2p.transformation
                        pcd = pcd.transform(transformation)
                        pcd.estimate_normals()
                        pcd_gt.estimate_normals()
                        gt_normal = np.asarray(pcd_gt.normals)
                        pred_normal = np.asarray(pcd.normals)

                        acc, acc_med, nc1, nc1_med = accuracy(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        comp, comp_med, nc2, nc2_med = completion(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        line = (
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                            f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                        )
                        print(line)
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(line + "\n")

                        acc_all += acc
                        comp_all += comp
                        nc1_all += nc1
                        nc2_all += nc2
                        acc_all_med += acc_med
                        comp_all_med += comp_med
                        nc1_all_med += nc1_med
                        nc2_all_med += nc2_med

                    if args.delete_ply_after_sequence:
                        for suff in ("-mask.ply", "-gt.ply"):
                            pth = osp.join(save_path, f"{base_name}{suff}")
                            if os.path.isfile(pth):
                                os.remove(pth)

                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            if accelerator.is_main_process and args.eval_metrics:
                to_write = ""
                for i in range(8):
                    fp = osp.join(save_path, f"logs_{i}.txt")
                    if not os.path.isfile(fp):
                        break
                    with open(fp, encoding="utf-8") as f_sub:
                        to_write += f_sub.read()
                with open(osp.join(save_path, "logs_all.txt"), "w", encoding="utf-8") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        m = regex.match(line)
                        if m:
                            data = m.groupdict()
                            for k, v in data.items():
                                if k != "scene_id":
                                    metrics[k].append(float(v))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {mn: sum(vs) / len(vs) for mn, vs in metrics.items()}
                    tail = f"mean: " + " | ".join(
                        f"{k}: {float(np.mean(vs)):.3f}" for k, vs in mean_metrics.items()
                    )
                    f.write(to_write + "\n" + tail + "\n")


def _apply_env_overrides(args) -> None:
    """与 run_kitti_scannet_ply.sh 通过环境变量传参时一致；未在命令行写时可读 EXXX。"""
    e = os.environ
    if not (args.scannet_scenes or "").strip() and e.get("SCANNET_SCENES", "").strip():
        args.scannet_scenes = e["SCANNET_SCENES"].strip()
    if not (args.kitti_scenes or "").strip() and e.get("KITTI_SCENES", "").strip():
        args.kitti_scenes = e["KITTI_SCENES"].strip()
    if args.max_frames is None and e.get("MAX_FRAMES", "").strip().isdigit():
        args.max_frames = int(e["MAX_FRAMES"])
    if e.get("KF_EVERY", "").strip().isdigit():
        args.kf_every = int(e["KF_EVERY"])
    if e.get("SCANNET_N_FRAMES", "").strip().isdigit():
        args.scannet_n_frames = int(e["SCANNET_N_FRAMES"])
    if e.get("KITTI_GATHER_N", "").strip().isdigit():
        args.kitti_gather_n = int(e["KITTI_GATHER_N"])
    if e.get("KITTI_PRESET", "").strip():
        args.kitti_preset = e["KITTI_PRESET"].strip()
    if e.get("EVAL_METRICS", "").strip() in ("1", "true", "yes", "Y"):
        args.eval_metrics = True


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    _apply_env_overrides(args)
    main(args)
