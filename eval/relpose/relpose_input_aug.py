"""TUM / ScanNet：relpose 评测用的「全局」输入适配（单次 forward，非 TTA）。

与 DUSt3R ImgNorm 衔接：在 [0,1] 上做确定性变换后再映射回 [-1,1]。
不使用空间卷积/局部 CLAHE，避免破坏匹配几何。

公开接口：
  - resolve_relpose_aug_preset(mode, eval_dataset) -> "tum"|"scannet"|None
  - apply_loaded_images_aug(images, preset)  # 就地改 load_images 的列表
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import numpy as np
import torch

AugMode = Literal["none", "auto", "tum", "scannet"]

# --- 预设常数（可按需微调）---
_TUM_GRAY_CLIP: Tuple[float, float] = (0.82, 1.25)
_TUM_LUM_TARGET = 0.49
_TUM_LUM_GAIN = (0.9, 1.18)

_SCANNET_PCT = (2.5, 97.5)
_SCANNET_GRAY_CLIP: Tuple[float, float] = (0.78, 1.38)
_SCANNET_LUM_TARGET = 0.50
_SCANNET_LUM_GAIN = (0.88, 1.28)


def resolve_relpose_aug_preset(mode: AugMode, eval_dataset: str) -> Optional[str]:
    m = (mode or "none").lower()
    if m == "none":
        return None
    ds = (eval_dataset or "").lower()
    if m == "auto":
        if "tum" in ds:
            return "tum"
        if "scannet" in ds:
            return "scannet"
        return None
    if m in ("tum", "scannet"):
        return m
    return None


def _tensor_to_hwc01(t: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] DUSt3R 归一化 -> float32 HWC [0,1]。"""
    return (
        (t * 0.5 + 0.5)
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _hwc01_to_tensor(x: np.ndarray) -> torch.Tensor:
    """HWC [0,1] float32 -> [1,3,H,W] DUSt3R 归一化。"""
    chw = np.ascontiguousarray(x.transpose(2, 0, 1))
    t = torch.from_numpy(chw[None])
    return (t - 0.5) / 0.5


def _gray_world(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = x.reshape(-1, 3).mean(axis=0)
    gray = float(m.mean())
    if gray < 1e-6:
        return x
    s = gray / (m + 1e-6)
    s = np.clip(s, lo, hi).astype(np.float32, copy=False)
    return np.clip(x * s.reshape(1, 1, 3), 0.0, 1.0)


def _luminance_gain(x: np.ndarray, target: float, g_lo: float, g_hi: float) -> np.ndarray:
    g = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    m = float(g.mean())
    if m < 1e-6:
        return x
    gain = float(np.clip(target / m, g_lo, g_hi))
    return np.clip(x * gain, 0.0, 1.0)


def _percentile_stretch_channels(x: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    y = np.empty_like(x)
    for c in range(3):
        ch = x[:, :, c]
        lo = float(np.percentile(ch, low_pct))
        hi = float(np.percentile(ch, high_pct))
        if hi - lo < 1e-6:
            y[:, :, c] = ch
        else:
            y[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return y


def adapt_tum(hwc: np.ndarray) -> np.ndarray:
    """TUM：灰世界 + 亮度整体缩放（各一步合成，减少中间图）。"""
    y = _gray_world(hwc, *_TUM_GRAY_CLIP)
    y = _luminance_gain(y, _TUM_LUM_TARGET, *_TUM_LUM_GAIN)
    return y


def adapt_scannet(hwc: np.ndarray) -> np.ndarray:
    """ScanNet：分位拉伸 + 灰世界 + 亮度。"""
    y = _percentile_stretch_channels(hwc, *_SCANNET_PCT)
    y = _gray_world(y, *_SCANNET_GRAY_CLIP)
    y = _luminance_gain(y, _SCANNET_LUM_TARGET, *_SCANNET_LUM_GAIN)
    return y


def apply_loaded_images_aug(images: list[dict[str, Any]], preset: str) -> None:
    """就地更新 ``images[i]['img']``（DUSt3R 格式张量）。"""
    if preset not in ("tum", "scannet"):
        return
    fn = adapt_tum if preset == "tum" else adapt_scannet
    for item in images:
        t = item.get("img")
        if not isinstance(t, torch.Tensor):
            continue
        hwc = _tensor_to_hwc01(t)
        y = fn(hwc)
        item["img"] = _hwc01_to_tensor(y)
