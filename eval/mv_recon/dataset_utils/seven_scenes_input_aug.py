"""7-Scenes：评测用 RGB 全局适配（无额外模型 forward）。

7-Scenes 为室内 Kinect 风格 RGB-D，与 TUM 类似存在通道偏色与曝光起伏；整段序列一次
``_get_views`` 内可对多帧做 **曝光增益时间平滑**，减轻递推/多视图重建时的帧间跳变。

在 PIL 图像（crop/resize 之后、ImgNorm 之前）上操作。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import PIL.Image

_GRAY_CLIP: Tuple[float, float] = (0.84, 1.22)
_LUM_TARGET = 0.49
_RAW_GAIN_CLIP: Tuple[float, float] = (0.92, 1.16)
_GAIN_SMOOTH_WIN = 5


def _pil_to_hwc01(im: PIL.Image.Image) -> np.ndarray:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _hwc01_to_pil(x: np.ndarray) -> PIL.Image.Image:
    u8 = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(u8, mode="RGB")


def _gray_world(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = x.reshape(-1, 3).mean(axis=0)
    gray = float(m.mean())
    if gray < 1e-6:
        return x
    s = gray / (m + 1e-6)
    s = np.clip(s, lo, hi).astype(np.float32, copy=False)
    return np.clip(x * s.reshape(1, 1, 3), 0.0, 1.0)


def _median_luminance(hwc: np.ndarray) -> float:
    lum = 0.299 * hwc[:, :, 0] + 0.587 * hwc[:, :, 1] + 0.114 * hwc[:, :, 2]
    return float(np.median(lum))


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    if n <= 1 or win <= 1:
        return x.astype(np.float32, copy=False)
    win = min(win, n)
    if win % 2 == 0:
        win -= 1
    if win < 1:
        return x.astype(np.float32, copy=False)
    pad = win // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float64) / float(win)
    out = np.convolve(xp, k, mode="valid")
    assert len(out) == n
    return out.astype(np.float32)


def apply_seven_scenes_pil_sequence(pils: List[PIL.Image.Image]) -> List[PIL.Image.Image]:
    """对同一序列的多帧 PIL RGB 做全局适配 + 增益时间平滑。空列表原样返回。"""
    if not pils:
        return pils

    hwcs = [_pil_to_hwc01(im) for im in pils]
    after_gw = [_gray_world(x, *_GRAY_CLIP) for x in hwcs]
    meds = np.array([_median_luminance(y) for y in after_gw], dtype=np.float64)
    meds = np.maximum(meds, 1e-6)
    g_lo, g_hi = _RAW_GAIN_CLIP
    tgt = _LUM_TARGET
    raw_gains = np.clip(tgt / meds, g_lo, g_hi)
    smooth_gains = _moving_average_1d(raw_gains, _GAIN_SMOOTH_WIN)

    out: List[PIL.Image.Image] = []
    for k in range(len(pils)):
        y = np.clip(after_gw[k] * float(smooth_gains[k]), 0.0, 1.0)
        out.append(_hwc01_to_pil(y))
    return out
