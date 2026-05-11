# Optional inference-time augmentation (multi-scale / hflip; fuse mean|median|conf-weighted).
# Not a model architecture: runs the same recurrent forward K times (K = |scales|×|hflips|).
# Lives under eval/ so ``dust3r`` can resolve from another repo (e.g. BAT3R on PYTHONPATH).

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from dust3r.inference import inference_recurrent_lighter


def _round_to_multiple(n: float, m: int = 16) -> int:
    """与 ViT patch（通常 16）对齐。"""
    n = int(round(n))
    return max(m, (max(n, m) // m) * m)


def _clone_view(view: dict) -> dict:
    out = {}
    for k, v in view.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(x.clone() if torch.is_tensor(x) else x for x in v)
        else:
            out[k] = deepcopy(v)
    return out


def augment_views(
    views: Sequence[dict],
    scale: float,
    hflip: bool,
    align_multiple: int = 16,
) -> List[dict]:
    if abs(scale - 1.0) < 1e-6 and not hflip:
        return [_clone_view(v) for v in views]

    aug: List[dict] = []
    for view in views:
        v = _clone_view(view)
        img = v["img"]
        _, _, h0, w0 = img.shape
        h1 = _round_to_multiple(h0 * scale, align_multiple)
        w1 = _round_to_multiple(w0 * scale, align_multiple)
        if h1 != h0 or w1 != w0:
            img = F.interpolate(img, size=(h1, w1), mode="bilinear", align_corners=False)
            v["img"] = img
            rm = v["ray_map"]
            if rm is not None and torch.is_tensor(rm) and not bool(torch.isnan(rm).all()):
                rm = F.interpolate(rm, size=(h1, w1), mode="bilinear", align_corners=False)
                v["ray_map"] = rm
            ts = v["true_shape"]
            if torch.is_tensor(ts):
                v["true_shape"] = torch.tensor(
                    [h1, w1], dtype=ts.dtype, device=ts.device
                )
            else:
                v["true_shape"] = np.array([h1, w1], dtype=getattr(ts, "dtype", np.int32))
        if hflip:
            v["img"] = torch.flip(v["img"], dims=[-1])
            rm = v["ray_map"]
            if rm is not None and torch.is_tensor(rm) and not bool(torch.isnan(rm).all()):
                v["ray_map"] = torch.flip(rm, dims=[-1])
        aug.append(v)
    return aug


def _flip_pts3d_lr(pts: torch.Tensor) -> torch.Tensor:
    out = torch.flip(pts, dims=[-2])
    out = out.clone()
    out[..., 0] = -out[..., 0]
    return out


def align_prediction_to_reference(
    pred: Dict[str, torch.Tensor],
    ref_hw: Tuple[int, int],
    aug_hflip: bool,
) -> Dict[str, torch.Tensor]:
    Href, Wref = ref_hw
    out: Dict[str, torch.Tensor] = {}
    for k, t in pred.items():
        if not torch.is_tensor(t):
            out[k] = t
            continue
        if t.dim() == 4 and t.shape[-1] in (3, 4) and k in (
            "pts3d_in_self_view",
            "pts3d_in_other_view",
            "rgb",
        ):
            x = t
            if aug_hflip:
                if k.startswith("pts3d"):
                    x = _flip_pts3d_lr(x)
                elif k == "rgb":
                    x = torch.flip(x, dims=[-2])
            x = x.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(Href, Wref), mode="bilinear", align_corners=False)
            out[k] = x.permute(0, 2, 3, 1)
        elif t.dim() == 3 and k in ("conf_self", "conf"):
            x = t.unsqueeze(1)
            if aug_hflip:
                x = torch.flip(x, dims=[-1])
            x = F.interpolate(x, size=(Href, Wref), mode="bilinear", align_corners=False)
            out[k] = x.squeeze(1)
        elif t.dim() == 2 and k == "camera_pose":
            out[k] = t.clone()
        else:
            out[k] = t.clone()
    return out


def _aggregate_stack(tensors: List[torch.Tensor], agg: str) -> torch.Tensor:
    stacked = torch.stack(tensors, dim=0)
    if agg == "mean":
        return stacked.mean(dim=0)
    if agg == "median":
        return stacked.median(dim=0).values
    raise ValueError(f"Unknown agg={agg}")


def _stack_variant_weights(
    per_variant: List[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    return torch.stack(
        [p["conf_self"].clamp(min=1e-8) for p in per_variant],
        dim=0,
    )


def _merge_weighted_by_conf(
    per_variant: List[Dict[str, torch.Tensor]],
    key: str,
    weights: torch.Tensor,
    *,
    softmax: bool,
) -> torch.Tensor:
    """weights: V,B,H,W；沿 V=0 融合。"""
    stacked = torch.stack([p[key] for p in per_variant], dim=0)
    if softmax:
        w = F.softmax(weights, dim=0)
    else:
        w = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    if stacked.dim() == 5 and stacked.shape[-1] in (3, 4):
        return (w.unsqueeze(-1) * stacked).sum(dim=0)
    if stacked.dim() == 4 and stacked.shape[-1] in (3, 4):
        return (w.unsqueeze(-1) * stacked).sum(dim=0)
    if stacked.dim() == 4:
        return (w * stacked).sum(dim=0)
    if stacked.dim() == 3:
        return (w * stacked).sum(dim=0)
    raise RuntimeError(f"unexpected shape for {key}: {stacked.shape}")


def _merge_frame_preds(
    per_variant: List[Dict[str, torch.Tensor]],
    aggregate: str,
    baseline_index: int,
) -> Dict[str, torch.Tensor]:
    keys = per_variant[0].keys()
    merged: Dict[str, torch.Tensor] = {}
    use_conf_fusion = aggregate in ("conf_weighted", "conf_softmax")
    softmax = aggregate == "conf_softmax"
    weights: Optional[torch.Tensor] = None
    if use_conf_fusion:
        if "conf_self" not in per_variant[0]:
            raise KeyError("conf_weighted / conf_softmax need conf_self in each pred")
        weights = _stack_variant_weights(per_variant)

    for k in keys:
        sample = per_variant[0][k]
        if not torch.is_tensor(sample):
            merged[k] = per_variant[baseline_index][k]
            continue
        if k == "camera_pose" and sample.dim() == 2:
            merged[k] = per_variant[baseline_index][k].clone()
            continue
        if use_conf_fusion and k in (
            "pts3d_in_self_view",
            "pts3d_in_other_view",
            "rgb",
            "conf_self",
            "conf",
        ):
            assert weights is not None
            merged[k] = _merge_weighted_by_conf(
                per_variant, k, weights, softmax=softmax
            )
        else:
            if aggregate in ("mean", "median"):
                merged[k] = _aggregate_stack([p[k] for p in per_variant], aggregate)
            else:
                merged[k] = _aggregate_stack([p[k] for p in per_variant], "mean")
    return merged


@torch.no_grad()
def inference_recurrent_lighter_tta(
    groups: Sequence[dict],
    model,
    device: Union[str, torch.device],
    scales: Sequence[float] = (1.0,),
    hflips: Sequence[bool] = (False, True),
    aggregate: str = "mean",
    baseline_scale: float = 1.0,
    baseline_hflip: bool = False,
    verbose: bool = True,
):
    """
    TTA：``scales`` × ``hflips`` 的笛卡尔积，每路一次 ``inference_recurrent_lighter``，再融合。

    - ``mean`` / ``median``：variant 维上 ``stack`` 后 ``mean`` / ``median``。
    - ``conf_weighted`` / ``conf_softmax``：按各 variant 的 ``conf_self`` 加权（含 5D pts3d）。
    """
    if len(groups) == 0:
        raise ValueError("empty groups")

    ref = groups[0]["img"]
    _, _, href, wref = ref.shape
    ref_hw = (int(href), int(wref))

    variants: List[Tuple[float, bool]] = [
        (float(s), bool(f)) for s in scales for f in hflips
    ]
    if len(variants) == 1 and variants[0] == (1.0, False):
        return inference_recurrent_lighter(
            list(groups), model, device, verbose=verbose
        )

    baseline_index = 0
    for i, (s, hf) in enumerate(variants):
        if abs(s - baseline_scale) < 1e-6 and hf == baseline_hflip:
            baseline_index = i
            break

    per_frame: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(len(groups))]
    state_args_out = None

    for vi, (sc, hf) in enumerate(variants):
        aug = augment_views(groups, scale=sc, hflip=hf)
        res, state_args = inference_recurrent_lighter(
            aug,
            model,
            device,
            verbose=verbose and vi == 0,
        )
        if vi == baseline_index:
            state_args_out = state_args
        preds = res["pred"]
        assert len(preds) == len(groups)
        for fi, pred in enumerate(preds):
            aligned = align_prediction_to_reference(pred, ref_hw, aug_hflip=hf)
            per_frame[fi].append(aligned)

    merged_preds: List[Dict[str, torch.Tensor]] = []
    for fi in range(len(groups)):
        merged_preds.append(
            _merge_frame_preds(per_frame[fi], aggregate, baseline_index)
        )

    out = dict(views=list(groups), pred=merged_preds)
    return out, state_args_out
