import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
)  # noqa
from dust3r.bayesian_ttt import BayesianRecurrentUnit, UpdateController

inf = float("inf")
from accelerate.logging import get_logger

from einops import rearrange
from dust3r.utils.device import to_cpu, to_gpu

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        model_update_type="cut3r",
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.model_update_type = model_update_type
        self.croco_kwargs = croco_kwargs


class GlobalSceneContext(nn.Module):
    """
    Maintains a global context of the scene to rectify transient observations.
    Useful for stabilizing tracking in dynamic environments.
    """
    def __init__(self, feature_dim, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("global_ctx", torch.zeros(1, 1, feature_dim))
        self.initialized = False
        
    def reset(self):
        self.initialized = False
        self.global_ctx = torch.zeros_like(self.global_ctx)

    def update_and_rectify(self, new_feat, gate=None):
        # new_feat: [B, N, D]
        # Compute global mean of current frame
        frame_ctx = new_feat.mean(dim=1, keepdim=True) # [B, 1, D]
        
        if not self.initialized:
            self.global_ctx = frame_ctx.detach()
            self.initialized = True
        
        # Rectify: Pull frame context towards global context
        # This reduces drift caused by dynamic objects dominating a frame
        ctx_diff = self.global_ctx - frame_ctx
        if gate is None:
            strength = 0.5
        else:
            strength = 0.25 + 0.50 * (1.0 - gate.mean().clamp(0.0, 1.0))
        rectified_feat = new_feat + strength * ctx_diff
        
        # Update Global Context (EMA)
        self.global_ctx = self.alpha * self.global_ctx + (1 - self.alpha) * frame_ctx.detach()
        
        return rectified_feat


class PoseDeltaGate(nn.Module):
    def __init__(self, rot_sigma_deg=15.0, trans_sigma=0.25):
        super().__init__()
        self.rot_sigma_deg = nn.Parameter(torch.tensor(float(rot_sigma_deg)))
        self.trans_sigma = nn.Parameter(torch.tensor(float(trans_sigma)))
        self.register_buffer("_prev_pose", None)

    def reset(self):
        self._prev_pose = None

    def _quat_conj(self, q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def _rel_rot_deg(self, q_from, q_to):
        q_from = F.normalize(q_from, p=2, dim=-1)
        q_to = F.normalize(q_to, p=2, dim=-1)
        q_rel = self._quat_mul(self._quat_conj(q_from), q_to)
        w = q_rel[..., 0].abs().clamp(0.0, 1.0)
        angle = 2.0 * torch.acos(w)
        return angle * (180.0 / torch.pi)

    def forward(self, pose, reset=False):
        if (pose is None) or (not isinstance(pose, torch.Tensor)) or pose.numel() == 0:
            return None
        if not torch.isfinite(pose).all():
            # Invalid pose should not drive recurrent updates.
            return torch.zeros((pose.shape[0], 1, 1), device=pose.device, dtype=pose.dtype)
        if reset or (self._prev_pose is None) or (self._prev_pose.shape != pose.shape):
            self._prev_pose = pose.detach()
            return torch.ones((pose.shape[0], 1, 1), device=pose.device, dtype=pose.dtype)
        prev = self._prev_pose
        if (prev is None) or (not torch.isfinite(prev).all()):
            self._prev_pose = pose.detach()
            return torch.ones((pose.shape[0], 1, 1), device=pose.device, dtype=pose.dtype)
        t, t_prev = pose[:, :3], prev[:, :3]
        dt = (t - t_prev).norm(dim=-1)
        trans_sigma = F.softplus(self.trans_sigma) + 1e-6
        trans_gate = torch.exp(-dt / trans_sigma)
        if pose.shape[-1] >= 7:
            rot_sigma = F.softplus(self.rot_sigma_deg) + 1e-6
            rot_deg = self._rel_rot_deg(prev[:, 3:7], pose[:, 3:7])
            rot_gate = torch.exp(-rot_deg / rot_sigma)
            gate = (rot_gate * trans_gate).clamp(0.0, 1.0)
        else:
            gate = trans_gate.clamp(0.0, 1.0)
        self._prev_pose = pose.detach()
        return gate.view(-1, 1, 1)

class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        ) # [1, 1, 768] pose mask token
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # [1, 256, 1536] pose mem
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v, return_attn=False):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C] global_img_feat
        feat_v: [B, 1, C] out_pose_feat
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)

        attention_maps = []
        for blk in self.write_blocks:
            mem, _, self_attn, cross_attn = blk(mem, feat, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return mem

    def inquire(self, query, mem, return_attn=False):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [1, 1, 768 global_img_feat_i + 768 masked_token(pose)]
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            ) # [1, 1, 768]
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        
        # Bayes-TTT: Initialize the Information-Theoretic Recurrent Unit
        self.bayesian_unit = BayesianRecurrentUnit(feature_dim=self.dec_embed_dim, state_dim=self.dec_embed_dim)
        self.global_context = GlobalSceneContext(feature_dim=self.dec_embed_dim)
        self.pose_delta_gate = PoseDeltaGate()
        
        self.update_controller = UpdateController(self.dec_embed_dim)
        self.prev_depth_median = None
        self.prev_output_pose = None
        self.output_pose_smooth_alpha = 0.25

        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim) # init state tokens [768, 1024]
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        ) # [768, 1024]
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, return_attn):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img) # Linear: [1, 576, 1024] -> [1, 576, 768]
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1) # [1, 1 + 576, 768]
            pos_img = torch.cat([pos_pose, pos_img], dim=1) # [1, 1 + 576, 2]
        final_output.append((f_state, f_img))
        attention_maps = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, self_attn_state, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, self_attn_img, cross_attn_img = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, self_attn_state, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, return_attn=return_attn)
                f_img, _, self_attn_img, cross_attn_img = blk_img(*final_output[-1][::-1], pos_img, pos_state, return_attn=return_attn)
            final_output.append((f_state, f_img))
            attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        # [1, 768, 768] [1, 768, 2]
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat) # Linear: [1, 768, 1024] -> [1, 768, 768]
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, return_attn
        )
        new_state_feat = new_state_feat[-1]

        # State Innovation Gating (SIG)：利用 init_state_feat 约束长序列漂移
        # innovation = new_state_feat 相对于 init_state_feat 的逐 token 偏差
        # innovation 越大 → gate 越小 → new_state_feat 越靠近上一帧 state_feat（保守更新）
        # innovation 越小 → gate 越大 → 正常更新（场景稳定区域充分吸收）
        # scannet去掉attention之后效果最好
       
        with torch.no_grad():
            innovation = (new_state_feat - init_state_feat).norm(dim=-1, keepdim=True)  # [B, N, 1]
            ref_scale = state_feat.norm(dim=-1, keepdim=True).mean().clamp(min=1e-3)
            # kitti: 0.5  5.0  (0.05, 1.0)
            # 7scenes: 1.5  7.0  (0.05, 1.0)
            # camera pose estimation  tum: 1.5  7.0  (0.05, 1.0) 
            # sintel: 5.0, 7.0  (0.01, 1.0)
            # scannet: 1.5  5.0  (0.05, 1.0)
            # bonn: 2.0  2.5  (0.3, 1.0)
            sig_gate = torch.sigmoid(- 1.5 * (innovation / ref_scale - 5.0))
            sig_gate = sig_gate.clamp(0.05, 1.0)
        new_state_feat = new_state_feat * sig_gate + state_feat * (1.0 - sig_gate)

        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _sanitize_camera_pose(self, pose):
        if (pose is None) or (not isinstance(pose, torch.Tensor)):
            return pose
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        if pose.shape[-1] >= 7:
            q = pose[..., 3:7]
            q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pose = pose.clone()
            pose[..., 3:7] = q
        return pose

    def _stabilize_camera_pose_temporal(self, pose, reset=False):
        pose = self._sanitize_camera_pose(pose)
        if (pose is None) or (not isinstance(pose, torch.Tensor)):
            return pose
        if (
            reset
            or self.prev_output_pose is None
            or (not isinstance(self.prev_output_pose, torch.Tensor))
            or self.prev_output_pose.shape != pose.shape
        ):
            self.prev_output_pose = pose.detach()
            return pose

        alpha = float(self.output_pose_smooth_alpha)
        prev = self.prev_output_pose
        out = pose.clone()

        # Smooth rotation in quaternion space with sign alignment.
        if pose.shape[-1] >= 7:
            q_prev = prev[..., 3:7]
            q_cur = pose[..., 3:7]
            dot = (q_prev * q_cur).sum(dim=-1, keepdim=True)
            q_cur = torch.where(dot < 0.0, -q_cur, q_cur)
            q_new = (1.0 - alpha) * q_prev + alpha * q_cur
            q_new = q_new / q_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            out[..., 3:7] = q_new

        self.prev_output_pose = out.detach()
        return out

    def _apply_update_mask_stabilizer(self, update_mask):
        # Keep recurrent updates away from fully-off / fully-on extremes.
        return update_mask.clamp(min=0.005, max=0.90)

    # tbptt training encoder: Truncated Backpropagation Through Time
    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    # tbptt training decoder step: Truncated Backpropagation Through Time
    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
            return_attn=False,
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # update global state
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    # training and testing
    def _forward_impl(self, views, ret_state=False):
        # [B, C, H, W] -> [B, H/16*W/16, 1024]
        shape, feat_ls, pos = self._encode_views(views) # [15, 3, 288, 512] -> feat [15, 576, 1024], pos [15, 576, 2]
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0]) # init state feat [1, 768, 768], state_pos [1, 768, 2]
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1) # [1, 256, 1536] init pose mem
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i) # avg pool: [1, 576, 1024] -> [1, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1) # [1, 1, 768] init pose token
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem) 
                    # [1, 1, 768] use [global_img_feat_i, masked_token(pose)] as query, cross-attend mem, get pose_feat_i
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                ) # [1, 1, 2]
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat, # [1, 768, 768]
                state_pos, # [1, 768, 2]
                feat_i, # [1, 576, 1024]
                pos_i, # [1, 576, 2]
                pose_feat_i, # [1, 1, 768] coarse pose token from pose_retriever
                pose_pos_i, # [1, 1, 2]
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=True,
            ) # [1, 768, 768]
            out_pose_feat_i = dec[-1][:, 0:1] # [1, 1, 768] refined pose token from dust3r
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            ) # [1, 256, 1536] use mem as query, cross-attend [global_img_feat_i, out_pose_feat_i], get new_mem
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(), # [1, 576, 1024]
                dec[self.dec_depth * 2 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth * 3 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth].float(), # [1, 1 + 576, 768]
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            if i  == 0:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    # z-score 归一化消除帧间绝对值偏差，使 sigmoid 输出更稳定
                    _sq_mean = state_query_img_key.mean(dim=1, keepdim=True)
                    _sq_std = state_query_img_key.std(dim=1, keepdim=True).clamp(min=1e-4)
                    state_query_img_key = (state_query_img_key - _sq_mean) / _sq_std
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None]
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self.forward_recurrent_lighter(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self.forward_recurrent_lighter(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    # testing: generate rgb xyz condition on raymap
    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem, *args
    ):
        if len(args) == 2:
            state_var, init_state_var = args
        else:
             # Bayes-TTT: Initialize state variance if not provided
             state_var = torch.ones_like(state_feat[..., :1]) * 1e3
             init_state_var = state_var.clone()

        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes) # [1, 6, 384, 512] -> feat [1, 768, 1024], pos [1, 768, 2]

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=True,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        
        # Bayes-TTT Update Logic
        img_mask = view["img_mask"]
        update = view.get("update", None)
        if update is not None:
            update_mask = (
                img_mask & update
            )  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()

        if self.config.model_update_type == "cut3r":
            update_mask1 = update_mask
        elif self.config.model_update_type == "ttt3r":
            cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
            state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
            # z-score 归一化消除帧间绝对值偏差，使 sigmoid 输出更稳定
            _sq_mean = state_query_img_key.mean(dim=1, keepdim=True)
            _sq_std = state_query_img_key.std(dim=1, keepdim=True).clamp(min=1e-4)
            state_query_img_key = (state_query_img_key - _sq_mean) / _sq_std
            
            # Bayes-TTT Update
            conf_map = res.get('conf_self', res.get('conf', None))
            relevance = torch.sigmoid(state_query_img_key)[..., None] # [B, N, 1]

            if conf_map is not None:
                state_feat_T = state_feat.transpose(1, 2)
                new_state_feat_T = new_state_feat.transpose(1, 2)
                state_var_T = state_var.transpose(1, 2)
                relevance_T = relevance.transpose(1, 2)
                if conf_map.dim() == 3:
                    conf_map_T = conf_map.reshape(conf_map.shape[0], -1).unsqueeze(1)
                elif conf_map.dim() == 2:
                    conf_map_T = conf_map.unsqueeze(1)
                else:
                    conf_map_T = conf_map
                innovation_T = new_state_feat_T - state_feat_T
                innovation_norm = innovation_T.abs().mean(dim=1, keepdim=True)
                if "pts3d_in_self_view" in res:
                    z = res["pts3d_in_self_view"][..., 2].reshape(res["pts3d_in_self_view"].shape[0], -1)
                    curr_depth_median = z.median(dim=1).values.mean()
                else:
                    curr_depth_median = None
                prev_med = self.prev_depth_median
                conf_adj, gate = self.update_controller(innovation_norm, conf_map_T, relevance_T, prev_med, curr_depth_median)
                new_state_feat_T, new_state_var_T, info_gain = self.bayesian_unit(
                    state_feat_T, state_var_T, new_state_feat_T, conf_adj, relevance_score=relevance_T
                )
                state_feat = new_state_feat_T.transpose(1, 2)
                state_var = new_state_var_T.transpose(1, 2)
                update_mask1 = update_mask * gate
                if curr_depth_median is not None:
                    self.prev_depth_median = curr_depth_median
            else:
                update_mask1 = update_mask * relevance
        else:
            # Default fallback if update type not set or other
             update_mask1 = update_mask

        gate_pose = self.pose_delta_gate(
            res.get("camera_pose", None),
            reset=bool(view.get("reset") is not None and view["reset"].any().item()),
        )
        if gate_pose is not None and isinstance(update_mask1, torch.Tensor):
            update_mask1 = self._apply_update_mask_stabilizer(update_mask1 * gate_pose)
        new_state_feat = self.global_context.update_and_rectify(new_state_feat, gate=(gate_pose if gate_pose is not None else None))

        update_mask2 = update_mask
        state_feat = new_state_feat * update_mask1 + state_feat * (
            1 - update_mask1
        )  # update global state
        mem = new_mem * update_mask2 + mem * (
            1 - update_mask2
        )  # then update local state
        # 防止长序列 state 数值漂移：用 dec_norm_state 对 state_feat 归一化

        reset_mask = view["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (
                1 - reset_mask
            )
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
            state_var = init_state_var * reset_mask + state_var * (1 - reset_mask)
            
        return res, (state_feat, state_pos, init_state_feat, mem, init_mem, state_var, init_state_var)

    # recurrent testing
    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(-1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                
                # Bayes-TTT: Initialize state variance (inverse confidence)
                state_var = torch.ones_like(state_feat[..., :1]) * 1e3
                init_state_var = state_var.clone()
                self.global_context.reset()
                self.pose_delta_gate.reset()
                self.prev_depth_median = None
                
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem, state_var, init_state_var)
                )

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            reset_cond = bool(view.get("reset") is not None and view["reset"].any().item())
            gate_pose = self.pose_delta_gate(res.get("camera_pose", None), reset=(i == 0 or reset_cond))
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            
            # update with learning rate
            if i == 0:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    
                    # Bayes-TTT Update
                    conf_map = res.get('conf_self', res.get('conf', None))
                    relevance = torch.sigmoid(state_query_img_key)[..., None] # [B, N, 1]

                    if conf_map is not None:
                         # Transpose inputs to [B, D, N] for Bayesian Unit
                         state_feat_T = state_feat.transpose(1, 2)
                         new_state_feat_T = new_state_feat.transpose(1, 2)
                         state_var_T = state_var.transpose(1, 2)
                         relevance_T = relevance.transpose(1, 2)
                         
                         # Compute Innovation for Controller
                         # We duplicate innovation calc here for the Controller to get the Gate
                         state_feat_T = state_feat.transpose(1, 2)
                         new_state_feat_T = new_state_feat.transpose(1, 2)
                         state_var_T = state_var.transpose(1, 2)
                         
                         innovation_T = new_state_feat_T - state_feat_T
                         innovation_norm = innovation_T.abs().mean(dim=1, keepdim=True)
                         
                         # Depth Scale Consistency Check
                         curr_depth_median = None
                         if "pts3d_in_self_view" in res:
                             pts = res["pts3d_in_self_view"]
                             # Robust median depth estimation
                             if pts.numel() > 0:
                                valid_mask = (pts[..., 2] > 1e-3) & (pts[..., 2] < 100.0)
                                if valid_mask.any():
                                    z = pts[..., 2][valid_mask]
                                    curr_depth_median = z.median()
                                else:
                                    curr_depth_median = pts[..., 2].mean()
                         
                         prev_med = self.prev_depth_median
                         
                         # Transpose conf_map for controller
                         # conf_map: [B, H, W] -> flatten to [B, N] to match state_feat [B, N, D]
                         if conf_map.dim() == 3: # [B, H, W]
                             conf_map_T = conf_map.reshape(conf_map.shape[0], -1) # [B, N]
                             # Expand to [B, 1, N] for controller/bayesian unit
                             conf_map_T = conf_map_T.unsqueeze(1) # [B, 1, N]
                         elif conf_map.dim() == 2: # [B, N]
                             conf_map_T = conf_map.unsqueeze(1)
                         else:
                             conf_map_T = conf_map

                         # Call Controller
                         conf_adj, gate_ctrl = self.update_controller(innovation_norm, conf_map_T, relevance_T, prev_med, curr_depth_median)
                         
                         if curr_depth_median is not None:
                             self.prev_depth_median = curr_depth_median
                         
                         # Call Bayesian Unit
                         # Returns: new_mu, new_var, adaptive_gate
                         new_state_feat_T, new_state_var_T, gate_bayes_T = self.bayesian_unit(
                             state_feat_T, state_var_T, new_state_feat_T, conf_adj, relevance_score=relevance_T
                         )
                         
                         # Transpose back to [B, N, D]
                         state_feat = new_state_feat_T.transpose(1, 2)
                         state_var = new_state_var_T.transpose(1, 2)
                         
                         # Compute Final Gate for Pose Memory
                         gate_bayes = gate_bayes_T.transpose(1, 2)
                         gate_ctrl = gate_ctrl.transpose(1, 2)
                         final_gate = gate_bayes * gate_ctrl
                         
                         # Update Mask 1 (Spatial) is 0 because state_feat is already updated
                         update_mask1 = 0.0
                         
                         # Update Mask 2 (Pose Memory) - Gated by Bayesian reliability
                         update_mask2 = update_mask * final_gate
                    else:
                        update_mask1 = update_mask * relevance
                        update_mask2 = update_mask
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            # update_mask2 is defined in branches above
            if gate_pose is not None and isinstance(update_mask1, torch.Tensor):
                soft_pose_gate *= gate_pose
                update_mask1 = update_mask1 * soft_pose_gate
            new_state_feat = self.global_context.update_and_rectify(new_state_feat, gate=(gate_pose if gate_pose is not None else None))
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            # 防止长序列 state 数值漂移：用 dec_norm_state 对 state_feat 归一化
            
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                state_var = init_state_var * reset_mask + state_var * (1 - reset_mask)
            
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem, state_var, init_state_var)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward_recurrent_lighter(self, views, device='cuda', ret_state=False):
        ress = []
        all_state_args = []
        reset_mask = False
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            device = view["img"].device
            reset_mask = bool(view.get("reset") is not None and view["reset"].any().item())
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0 or reset_mask:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                
                # Bayes-TTT: Initialize state variance (inverse confidence)
                # High variance = low confidence = open to updates
                state_var = torch.ones_like(state_feat[..., :1]) * 1e3 
                init_state_var = state_var.clone()
                self.global_context.reset()
                self.pose_delta_gate.reset()
                self.prev_output_pose = None

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1]

            # update mem
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            ).detach()

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            pose_raw = res.get("camera_pose", None) if isinstance(res, dict) else None
            pose_raw = self._sanitize_camera_pose(pose_raw)
            gate_pose = self.pose_delta_gate(pose_raw, reset=(i == 0 or reset_mask))
            if isinstance(res, dict) and ("camera_pose" in res):
                res["camera_pose"] = pose_raw
            res_cpu = to_cpu(res)
            ress.append(res_cpu)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            if i  == 0 or reset_mask:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    # Bayes-TTT Update
                    conf_map = res.get('conf_self', res.get('conf', None))
                    relevance = torch.sigmoid(state_query_img_key)[..., None] # [B, N, 1]

                    if conf_map is not None:
                         # Transpose inputs to [B, D, N] for Bayesian Unit
                         state_feat_T = state_feat.transpose(1, 2)
                         new_state_feat_T = new_state_feat.transpose(1, 2)
                         
                         # state_var: [B, N, 1] -> [B, 1, N] (broadcasts over D)
                         state_var_T = state_var.transpose(1, 2)
                         
                         # conf_map can be [B, H, W], [B, N], or [B, N, 1]
                         if conf_map.dim() == 3:
                             conf_map_T = conf_map.reshape(conf_map.shape[0], -1).unsqueeze(1)
                         elif conf_map.dim() == 2:
                             conf_map_T = conf_map.unsqueeze(1)
                         else:
                             conf_map_T = conf_map
                         
                         # relevance: [B, N, 1] -> [B, 1, N]
                         relevance_T = relevance.transpose(1, 2)
                         
                         # Call Bayesian Unit
                         new_state_feat_T, new_state_var_T, info_gain = self.bayesian_unit(
                             state_feat_T, state_var_T, new_state_feat_T, conf_map_T, relevance_score=relevance_T
                         )
                         
                         # Transpose back to [B, N, D]
                         state_feat = new_state_feat_T.transpose(1, 2)
                         state_var = new_state_var_T.transpose(1, 2)
                         # state_var = state_var.clamp(min=5.0)

                         # Set update_mask1 to 0 to prevent double update
                         update_mask1 = 0.0
                    else:
                        update_mask1 = update_mask * relevance
                else:
                    # Default fallback
                    update_mask1 = update_mask

            if isinstance(update_mask1, torch.Tensor):
                update_mask1 = self._apply_update_mask_stabilizer(update_mask1)

            update_mask2 = update_mask
            if gate_pose is not None and isinstance(update_mask1, torch.Tensor):
                update_mask1 = self._apply_update_mask_stabilizer(update_mask1 * gate_pose)
            if isinstance(update_mask2, torch.Tensor):
                # mem 更新不受 gate_pose 影响，保证大旋转时 pose memory 正常更新
                update_mask2 = self._apply_update_mask_stabilizer(update_mask2)
            # GlobalSceneContext 整流 new_state_feat，抑制动态目标对 state 的污染
            new_state_feat = self.global_context.update_and_rectify(
                new_state_feat, gate=(gate_pose if gate_pose is not None else None)
            )
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            # 防止长序列 state 数值漂移：用 dec_norm_state 对 state_feat 归一化
            state_feat = state_feat.detach()
            mem = mem.detach()
            state_var = state_var.detach()
            
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                state_var = init_state_var * reset_mask + state_var * (1 - reset_mask)
                
                state_feat = state_feat.detach()
                mem = mem.detach()
                state_var = state_var.detach()
            # all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem, state_var, init_state_var))
            all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem, state_var, init_state_var)]
        if ret_state:
            return ress, views, all_state_args
        return ress, views

if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
