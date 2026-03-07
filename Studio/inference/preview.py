'''
ADOBE CONFIDENTIAL
Copyright 2026 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
'''

from __future__ import annotations

import os
import sys
import math
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffsynth import ModelManager, WanVideoClickMapPipeline, save_video
from PIL import Image


RECAM_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BASE_MODELS = [
    str(RECAM_ROOT / "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
    str(RECAM_ROOT / "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
    str(RECAM_ROOT / "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
]

SPEED_S_MIN = 0.125
SPEED_S_MAX = 8.0

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def resolve_ckpt_path(ckpt_path: str) -> str:
    # modified for non-adobe server
    return ckpt_path


def make_cam_traj_from_preset_refspace(
    preset: str,
    step_m: float = 0.25,
    amp_m: float = 1.6,
    zigzag_span_m: float = 0.5,
    loop_radius_m: float = 2.0
) -> torch.Tensor:
    t_list = []
    
    if preset in ["forward", "backward", "left", "right", "up", "down"]:
        dir_map = {
            "forward":  np.array([+1, 0, 0], dtype=np.float64),
            "backward": np.array([-1, 0, 0], dtype=np.float64),
            "right":    np.array([0, 0, +1], dtype=np.float64),
            "left":     np.array([0, 0, -1], dtype=np.float64),
            "up":       np.array([0, +1, 0], dtype=np.float64),
            "down":     np.array([0, -1, 0], dtype=np.float64),
        }
        d = dir_map[preset] * (float(step_m) / 4.0)
        p = np.zeros(3, dtype=np.float64)
        for i in range(81):
            t_list.append(p.copy())
            p += d
            
    elif preset == "s_curve":
        for i in range(81):
            x = (float(step_m) / 4.0) * i
            z = float(amp_m) * np.sin(2.0 * np.pi * i / 80.0)
            t_list.append(np.array([x, 0.0, z], dtype=np.float64))
            
    elif preset == "zigzag_forward":
        for i in range(81):
            x = (float(step_m) / 4.0) * i
            saw = ((i % 20) / 20.0) - 0.5
            z = float(zigzag_span_m) * saw * 2.0
            t_list.append(np.array([x, 0.0, z], dtype=np.float64))
            
    elif preset == "loop":
        R = float(loop_radius_m)
        for i in range(81):
            theta = 2.0 * np.pi * i / 80.0
            x = R * (1.0 - np.cos(theta))
            z = R * np.sin(theta)
            t_list.append(np.array([x, 0.0, z], dtype=np.float64))
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    traj = []
    for k in range(21):
        j = 4 * k
        t_ref = t_list[j]
        M = np.concatenate([np.eye(3, dtype=np.float64), t_ref.reshape(3, 1)], axis=1)
        traj.append(M.reshape(-1))
    
    traj = np.stack(traj, axis=0).astype(np.float32)
    return torch.from_numpy(traj)


def _rescale_cam_traj_identityR(
    cam_traj_21: torch.Tensor,
    mode: str,
    s_target: float
) -> Tuple[torch.Tensor, Optional[float], Optional[float]]:
    if cam_traj_21 is None or mode == "none":
        return cam_traj_21, None, 1.0
    
    M = cam_traj_21.view(-1, 3, 4)
    t = M[:, :, 3]
    
    if t.shape[0] < 2:
        return cam_traj_21, None, 1.0
    
    dt = t[1:] - t[:-1]
    steps = torch.linalg.norm(dt, dim=1)
    
    if steps.numel() == 0:
        return cam_traj_21, None, 1.0
    
    s_local = float(steps.median().item())
    eps = 1e-8
    if not np.isfinite(s_local) or s_local < eps:
        return cam_traj_21, max(s_local, 0.0), 1.0
    
    if mode == "unit_median":
        alpha = 1.0 / s_local
    elif mode == "fixed":
        alpha = float(s_target) / s_local
    else:
        return cam_traj_21, s_local, 1.0
    
    M[:, :, 3] = t * alpha
    
    return M.reshape(-1, 12), s_local, alpha


def load_image_as_video_tensor(
    image_path: Path,
    height: int = 480,
    width: int = 960,
    num_frames: int = 81
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    
    arr = arr.transpose(2, 0, 1)
    
    vid = np.stack([arr] * num_frames, axis=1)
    
    return torch.from_numpy(vid)


def video_tensor_to_uint8(video_tensor: torch.Tensor) -> np.ndarray:
    vid = video_tensor.detach().cpu().numpy()
    
    if vid.ndim == 4 and vid.shape[0] == 3:
        vid = vid.transpose(1, 2, 3, 0)
    
    vid = (vid + 1.0) * 127.5
    vid = np.clip(vid, 0, 255).astype(np.uint8)
    
    return vid


def ensure_click_modules_and_load(dit, ckpt_path: str):
    dim = dit.blocks[0].self_attn.q.weight.shape[0]
    p_dtype = dit.patch_embedding.weight.dtype
    p_device = dit.patch_embedding.weight.device

    ckpt_local = resolve_ckpt_path(ckpt_path)
    state_dict = torch.load(ckpt_local, map_location="cpu")
    sd_keys = set(state_dict.keys())

    def _has(prefix: str) -> bool:
        return any(k == prefix or k.startswith(prefix + ".") for k in sd_keys)

    for i, blk in enumerate(dit.blocks):
        pref = f"blocks.{i}.click_A_encoder"
        if _has(pref + ".weight"):
            if not hasattr(blk, "click_A_encoder"):
                enc = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
                blk.click_A_encoder = enc
                print(f"Created dit.blocks[{i}].click_A_encoder")

    for i, blk in enumerate(dit.blocks):
        pref = f"blocks.{i}.cam_traj_encoder"
        if _has(pref + ".weight"):
            if not hasattr(blk, "cam_traj_encoder"):
                enc = nn.Linear(12, dim, bias=True).to(dtype=p_dtype, device=p_device)
                blk.cam_traj_encoder = enc
                print(f"Created dit.blocks[{i}].cam_traj_encoder")

    for i, blk in enumerate(dit.blocks):
        pref = f"blocks.{i}.global_context_encoder"
        if _has(pref + ".weight"):
            if not hasattr(blk, "global_context_encoder"):
                enc = nn.Linear(16, dim, bias=True).to(dtype=p_dtype, device=p_device)
                blk.global_context_encoder = enc
                print(f"Created dit.blocks[{i}].global_context_encoder")

    if _has("click_token_proj") and not hasattr(dit, "click_token_proj"):
        dit.click_token_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
        print("Created dit.click_token_proj")
    if _has("click_token_scale") and not hasattr(dit, "click_token_scale"):
        dit.click_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
        print("Created dit.click_token_scale")

    if _has("speed_token_proj") and not hasattr(dit, "speed_token_proj"):
        dit.speed_token_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
        print("Created dit.speed_token_proj")
    if _has("speed_token_scale") and not hasattr(dit, "speed_token_scale"):
        dit.speed_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
        print("Created dit.speed_token_scale")

    if _has("dir_token_proj") and not hasattr(dit, "dir_token_proj"):
        dit.dir_token_proj = nn.Linear(3, dim, bias=True).to(dtype=p_dtype, device=p_device)
        print("Created dit.dir_token_proj")
    if _has("dir_token_scale") and not hasattr(dit, "dir_token_scale"):
        dit.dir_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
        print("Created dit.dir_token_scale")

    if _has("click_film") and not hasattr(dit, "click_film"):
        dit.click_film = nn.Linear(1, 2*dim, bias=True).to(dtype=p_dtype, device=p_device)
        print("Created dit.click_film")
    if _has("click_film_gain") and not hasattr(dit, "click_film_gain"):
        dit.click_film_gain = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
        print("Created dit.click_film_gain")

    if _has("traj_scale_proj.weight") and not hasattr(dit, "traj_scale_proj"):
        dit.traj_scale_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
        print("Created dit.traj_scale_proj")

    if _has("latent_A_proj.weight"):
        w = state_dict["latent_A_proj.weight"]
        out_ch = int(w.shape[0])
        in_ch = int(w.shape[1])
        kH, kW = int(w.shape[2]), int(w.shape[3])
        if not (kH == 1 and kW == 1 and in_ch == 1):
            print("[WARN] latent_A_proj in ckpt not 1x1 conv with in_ch==1")
        if not hasattr(dit, "latent_A_proj"):
            conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=True)
            conv.to(dtype=p_dtype, device=p_device)
            dit.latent_A_proj = conv
            print("Created dit.latent_A_proj")

    if _has("latent_A_gate") and not hasattr(dit, "latent_A_gate"):
        dit.latent_A_gate = nn.Parameter(torch.tensor(0.1, dtype=p_dtype, device=p_device))
        print("Created dit.latent_A_gate")

    for blk in dit.blocks:
        if not hasattr(blk, "projector"):
            blk.projector = nn.Linear(dim, dim, bias=True).to(dtype=p_dtype, device=p_device)
            with torch.no_grad():
                blk.projector.weight.copy_(torch.eye(dim, dtype=p_dtype, device=p_device))
                blk.projector.bias.zero_()

    dit.load_state_dict(state_dict, strict=True)

    if hasattr(dit, "click_token_proj"):
        print("click_token_proj |w| mean:", dit.click_token_proj.weight.data.abs().mean().item())
    if hasattr(dit, "speed_token_proj"):
        print("speed_token_proj  |w| mean:", dit.speed_token_proj.weight.data.abs().mean().item())
    if hasattr(dit, "dir_token_proj"):
        print("dir_token_proj    |w| mean:", dit.dir_token_proj.weight.data.abs().mean().item())
    if hasattr(dit, "click_film"):
        print("click_film        |w| mean:", dit.click_film.weight.data.abs().mean().item())
    if hasattr(dit, "latent_A_proj"):
        print("latent_A_proj     |w| mean:", dit.latent_A_proj.weight.data.abs().mean().item())
    if hasattr(dit, "latent_A_gate"):
        print("latent_A_gate     value:", float(dit.latent_A_gate.data.cpu().item()))
    
    has_cam_traj = any(hasattr(blk, "cam_traj_encoder") for blk in dit.blocks)
    print(f"[Model] cam_traj_encoder present: {has_cam_traj}")


class PreviewInference:
    
    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.model_loaded = False
        self.has_cam_traj = False
        
    def load_model(
        self,
        ckpt_path: str,
        base_model_paths: list = None,
    ) -> bool:
        try:
            if base_model_paths is None:
                base_model_paths = DEFAULT_BASE_MODELS
            
            for p in base_model_paths:
                if not os.path.exists(p):
                    print(f"[Preview] WARNING: Base model not found: {p}")
            
            print(f"[Preview] Loading base models...")
            for p in base_model_paths:
                print(f"  - {p}")
            
            model_manager = ModelManager(torch_dtype=self.dtype, device="cpu")
            model_manager.load_models(base_model_paths)
            
            print(f"[Preview] Building pipeline on {self.device}...")
            self.pipe = WanVideoClickMapPipeline.from_model_manager(
                model_manager, device=self.device
            )
            
            print(f"[Preview] Loading checkpoint: {ckpt_path}")
            ensure_click_modules_and_load(self.pipe.dit, ckpt_path)
            
            self.pipe.dit.click_use_token_A = False
            self.pipe.dit.click_use_spatial_film = False
            self.pipe.dit.click_use_dir_token = False
            self.pipe.dit.latent_click_fusion = "none"
            self.pipe.dit.latent_A_centering = "minus1to1"
            
            self.has_cam_traj = any(hasattr(blk, "cam_traj_encoder") for blk in self.pipe.dit.blocks)
            print(f"[Preview] cam_traj_encoder available: {self.has_cam_traj}")
            
            if not self.has_cam_traj:
                print("[Preview] WARNING: Model does not have cam_traj_encoder, trajectory control may not work!")
            
            self.pipe.to(self.device)
            self.pipe.to(dtype=self.dtype)
            
            self.model_loaded = True
            print(f"[Preview] Model loaded successfully!")
            return True
            
        except Exception as e:
            import traceback
            print(f"[Preview] Failed to load model: {e}")
            print(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def generate(
        self,
        input_image_path: Path,
        trajectory: str,
        scale: float = 1.0,
        height: int = 480,
        width: int = 960,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: int = 0,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        if not self.model_loaded or self.pipe is None:
            return False, None, "Model not loaded"
        
        try:
            print(f"[Preview] Generating: trajectory={trajectory}, scale={scale}")
            
            src_video = load_image_as_video_tensor(
                input_image_path, height, width, num_frames
            )
            
            traj_s_curve_amp_m = 1.4
            traj_loop_radius_m = 1.5
            traj_step_m = 0.25
            re_scale_mode = "fixed"
            re_scale_target = 1.0
            
            used_speed_s = float(np.clip(scale, SPEED_S_MIN, SPEED_S_MAX))
            used_speed_z = float(math.log2(used_speed_s))
            
            speed_scalar_tensor = torch.tensor(
                [[used_speed_z]], dtype=self.dtype, device=self.device
            )
            
            print(f"[Preview] Speed: s={used_speed_s:.2f}, z(log2)={used_speed_z:.4f}")
            
            cam_traj_arg = None
            if self.has_cam_traj:
                cam_traj_21 = make_cam_traj_from_preset_refspace(
                    preset=trajectory,
                    step_m=traj_step_m,
                    amp_m=traj_s_curve_amp_m,
                    loop_radius_m=traj_loop_radius_m,
                )
                
                cam_traj_21, s_local, alpha = _rescale_cam_traj_identityR(
                    cam_traj_21, re_scale_mode, re_scale_target
                )
                
                print(f"[Preview] Trajectory rescale: mode={re_scale_mode}, s_local={s_local}, alpha={alpha}")
                
                cam_traj_arg = cam_traj_21.unsqueeze(0).to(
                    device=self.device, dtype=self.dtype
                )
            
            src_video_tensor = src_video.unsqueeze(0).to(
                device=self.device, dtype=self.dtype
            )
            
            print(f"[Preview] Running inference...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            vid_pil_list = self.pipe(
                prompt="",
                negative_prompt=NEGATIVE_PROMPT,
                source_video=src_video_tensor,
                speed_scalar=speed_scalar_tensor,
                cam_traj_condition=cam_traj_arg,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                tiled=False,
                height=height,
                width=width,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            video_frames = []
            for pil_img in vid_pil_list:
                arr = np.array(pil_img, dtype=np.uint8)
                video_frames.append(arr)
            
            video_array = np.stack(video_frames, axis=0)
            
            print(f"[Preview] Generation complete: {video_array.shape}")
            return True, video_array, "Generation successful"
            
        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[Preview] {error_msg}")
            return False, None, error_msg
    
    def unload_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.model_loaded = False
        self.has_cam_traj = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Preview] Model unloaded")


def load_preview_model(
    ckpt_path: str,
    device: str = "cuda:0",
) -> PreviewInference:
    inference = PreviewInference(device=device)
    inference.load_model(ckpt_path)
    return inference
