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
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _split_indices_even_overlap1(T: int, K: int) -> List[Tuple[int, int]]:
    base = T // K
    r = T % K
    segs = []
    acc = 0
    for i in range(K):
        length = base if i < K - 1 else base + r
        s = acc
        e = acc + length
        e = min(e + 1, T)
        segs.append((s, e))
        acc = acc + base if i < K - 1 else acc + (base + r)
    return segs


def _make_m81_mask(seg_start: int, seg_end: int) -> torch.Tensor:
    m = torch.zeros(81, dtype=torch.bfloat16)
    a = max(0, int(seg_start))
    b = min(81, int(seg_end))
    if b > a:
        m[a:b] = 1.0
    return m.unsqueeze(0)


def _video_array_to_CTHW_floatm11(video_array: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(video_array).float() / 255.0
    x = x.permute(3, 0, 1, 2).contiguous()
    x = (x - 0.5) / 0.5
    return x


def _CTHW_floatm11_to_video_array(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = x.permute(1, 2, 3, 0) * 255.0
    return x.round().byte().numpy()


def _hw_resize_video(vid_CTHW: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    C, T, H, W = vid_CTHW.shape
    x = vid_CTHW.permute(1, 0, 2, 3)
    x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False, antialias=True)
    return x.permute(1, 0, 2, 3).contiguous()


def _interp_to_len(vid_CTHW: torch.Tensor, T_out: int) -> torch.Tensor:
    x = vid_CTHW.unsqueeze(0)
    x = F.interpolate(x, size=(T_out, vid_CTHW.shape[2], vid_CTHW.shape[3]),
                      mode="trilinear", align_corners=True)
    return x[0]


def _refine_degrade_blur_alias(vid_CTHW: torch.Tensor,
                                down_h: int, down_w: int,
                                up_h: int, up_w: int) -> torch.Tensor:
    assert vid_CTHW.ndim == 4
    C, T, H, W = vid_CTHW.shape
    
    x = (vid_CTHW * 0.5 + 0.5).clamp(0, 1)
    x = x.permute(1, 0, 2, 3).contiguous()
    
    x = F.interpolate(x, size=(int(down_h), int(down_w)), mode="area")
    x = F.interpolate(x, size=(int(up_h), int(up_w)), mode="bicubic", align_corners=False)
    
    x = x.permute(1, 0, 2, 3).contiguous()
    return (x - 0.5) / 0.5


def resolve_ckpt_path(ckpt_path: str) -> str:
    # modified for non-adobe server
    return ckpt_path


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
        if _has(pref + ".weight") and not hasattr(blk, "click_A_encoder"):
            enc = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
            blk.click_A_encoder = enc
        
        pref = f"blocks.{i}.cam_traj_encoder"
        if _has(pref + ".weight") and not hasattr(blk, "cam_traj_encoder"):
            enc = nn.Linear(12, dim, bias=True).to(dtype=p_dtype, device=p_device)
            blk.cam_traj_encoder = enc
        
        pref = f"blocks.{i}.global_context_encoder"
        if _has(pref + ".weight") and not hasattr(blk, "global_context_encoder"):
            enc = nn.Linear(16, dim, bias=True).to(dtype=p_dtype, device=p_device)
            blk.global_context_encoder = enc
        
        if not hasattr(blk, "projector"):
            blk.projector = nn.Linear(dim, dim, bias=True).to(dtype=p_dtype, device=p_device)
            with torch.no_grad():
                blk.projector.weight.copy_(torch.eye(dim, dtype=p_dtype, device=p_device))
                blk.projector.bias.zero_()
    
    if _has("click_token_proj") and not hasattr(dit, "click_token_proj"):
        dit.click_token_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
    if _has("click_token_scale") and not hasattr(dit, "click_token_scale"):
        dit.click_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
    if _has("speed_token_proj") and not hasattr(dit, "speed_token_proj"):
        dit.speed_token_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
    if _has("speed_token_scale") and not hasattr(dit, "speed_token_scale"):
        dit.speed_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
    if _has("dir_token_proj") and not hasattr(dit, "dir_token_proj"):
        dit.dir_token_proj = nn.Linear(3, dim, bias=True).to(dtype=p_dtype, device=p_device)
    if _has("dir_token_scale") and not hasattr(dit, "dir_token_scale"):
        dit.dir_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
    if _has("click_film") and not hasattr(dit, "click_film"):
        dit.click_film = nn.Linear(1, 2 * dim, bias=True).to(dtype=p_dtype, device=p_device)
    if _has("click_film_gain") and not hasattr(dit, "click_film_gain"):
        dit.click_film_gain = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
    if _has("traj_scale_proj.weight") and not hasattr(dit, "traj_scale_proj"):
        dit.traj_scale_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
    
    if _has("latent_A_proj.weight"):
        w = state_dict["latent_A_proj.weight"]
        out_ch = int(w.shape[0])
        in_ch = int(w.shape[1])
        kH, kW = int(w.shape[2]), int(w.shape[3])
        if not hasattr(dit, "latent_A_proj"):
            conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=True)
            conv.to(dtype=p_dtype, device=p_device)
            dit.latent_A_proj = conv
    
    if _has("latent_A_gate") and not hasattr(dit, "latent_A_gate"):
        dit.latent_A_gate = nn.Parameter(torch.tensor(0.1, dtype=p_dtype, device=p_device))
    
    dit.load_state_dict(state_dict, strict=True)
    print(f"[Refine] Loaded checkpoint weights")


class RefineInference:
    
    OUTPUT_HEIGHT = 720
    OUTPUT_WIDTH = 1440
    DEGRADE_HEIGHT = 480
    DEGRADE_WIDTH = 960
    NUM_FRAMES = 81
    
    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.model_loaded = False
        
    def load_model(self, ckpt_path: str) -> bool:
        original_cwd = os.getcwd()
        
        try:
            print(f"[Refine] Loading model from: {ckpt_path}")
            print(f"[Refine] Device: {self.device}")
            
            refine_root = Path(__file__).resolve().parent.parent.parent
            os.chdir(refine_root)
            print(f"[Refine] Changed working directory to: {refine_root}")
            
            from diffsynth import ModelManager, WanVideoClickMapPipeline
            
            model_manager = ModelManager(torch_dtype=self.dtype, device="cpu")
            model_manager.load_models([
                "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            ])
            
            self.pipe = WanVideoClickMapPipeline.from_model_manager(model_manager, device=self.device)
            
            ensure_click_modules_and_load(self.pipe.dit, ckpt_path)
            
            self.pipe.dit.click_use_token_A = False
            self.pipe.dit.click_use_spatial_film = False
            self.pipe.dit.click_use_dir_token = False
            self.pipe.dit.latent_click_fusion = "none"
            
            has_global_context = any(
                hasattr(blk, "global_context_encoder") 
                for blk in self.pipe.dit.blocks
            )
            self.pipe.use_global_context = has_global_context
            print(f"[Refine] Global context encoder: {has_global_context}")
            
            self.pipe.to(self.device)
            self.pipe.to(dtype=self.dtype)
            
            self.model_loaded = True
            print(f"[Refine] Model loaded successfully!")
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load Refine model: {str(e)}\n{traceback.format_exc()}"
            print(f"[Refine] ✗ {error_msg}")
            return False
        finally:
            os.chdir(original_cwd)
            print(f"[Refine] Restored working directory to: {original_cwd}")
    
    def _extract_global_context(self, src_video: torch.Tensor) -> torch.Tensor:
        global_ctx_video_480_960 = _hw_resize_video(
            src_video, self.DEGRADE_HEIGHT, self.DEGRADE_WIDTH
        )
        
        global_ctx_video_resized = _hw_resize_video(
            global_ctx_video_480_960, self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH
        )
        
        with torch.no_grad():
            L_global = self.pipe.encode_video(
                global_ctx_video_resized.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            )[0].to(device=self.device)
            
            if L_global.ndim == 4:
                L_global_5d = L_global.unsqueeze(0)
            else:
                L_global_5d = L_global
            
            global_ctx_feat = L_global_5d.mean(dim=(3, 4))
            global_ctx_feat = global_ctx_feat.transpose(1, 2).contiguous()
            
            del L_global, L_global_5d
        
        return global_ctx_feat
    
    def generate_segment(
        self,
        src_video_81: torch.Tensor,
        seg_start: int,
        seg_end: int,
        global_ctx_feat: torch.Tensor,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        m81 = _make_m81_mask(seg_start, seg_end).to(device=self.device)
        
        vid = self.pipe(
            prompt="",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=src_video_81.unsqueeze(0).to(device=self.device, dtype=self.dtype),
            speed_scalar=None,
            cam_traj_condition=None,
            m81=m81,
            enable_refine=True,
            use_full=False,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=0,
            tiled=False,
            height=self.OUTPUT_HEIGHT,
            width=self.OUTPUT_WIDTH,
            global_context_feat=global_ctx_feat,
        )
        
        return vid
    
    def generate(
        self,
        input_video_path: Path,
        num_segments: int = 8,
        trajectory: str = "forward",
        scale: float = 1.0,
        height: int = 720,
        width: int = 1440,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        temp_output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        if not self.model_loaded:
            return False, None, "Model not loaded"
        
        try:
            import imageio.v2 as imageio
            
            print(f"[Refine] Starting refinement: segments={num_segments}")
            print(f"[Refine] Input video: {input_video_path}")
            
            reader = imageio.get_reader(str(input_video_path))
            frames = []
            for frame in reader:
                frames.append(frame)
            reader.close()
            
            video_array = np.stack(frames, axis=0)
            T_input = video_array.shape[0]
            
            print(f"[Refine] Input shape: {video_array.shape}")
            
            src_video = _video_array_to_CTHW_floatm11(video_array)
            
            if T_input != 81:
                print(f"[Refine] Interpolating {T_input} -> 81 frames")
                src_video = _interp_to_len(src_video, 81)
            
            src_degraded = _refine_degrade_blur_alias(
                src_video,
                down_h=self.DEGRADE_HEIGHT,
                down_w=self.DEGRADE_WIDTH,
                up_h=self.OUTPUT_HEIGHT,
                up_w=self.OUTPUT_WIDTH,
            )
            
            print(f"[Refine] Degraded video shape: {tuple(src_degraded.shape)}")
            
            global_ctx_feat = None
            if hasattr(self.pipe, 'use_global_context') and self.pipe.use_global_context:
                print(f"[Refine] Extracting global context...")
                global_ctx_feat = self._extract_global_context(src_degraded)
                print(f"[Refine] Global context shape: {tuple(global_ctx_feat.shape)}")
            
            segments = _split_indices_even_overlap1(81, num_segments)
            print(f"[Refine] Segments: {segments}")
            
            generated_segments = []
            
            for seg_idx, (s81, e81) in enumerate(segments):
                print(f"[Refine] Processing segment {seg_idx + 1}/{num_segments}: frames [{s81}, {e81})")
                
                gen_81 = self.generate_segment(
                    src_video_81=src_degraded,
                    seg_start=s81,
                    seg_end=e81,
                    global_ctx_feat=global_ctx_feat,
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_inference_steps,
                )
                
                if isinstance(gen_81, list):
                    pil_tensors = []
                    for img in gen_81:
                        arr = np.array(img)
                        t = torch.from_numpy(arr).permute(2, 0, 1).float()
                        t = (t / 127.5) - 1.0
                        pil_tensors.append(t)
                    gen_tensor = torch.stack(pil_tensors, dim=1)
                else:
                    gen_tensor = gen_81
                
                generated_segments.append(gen_tensor)
                
                if temp_output_dir is not None:
                    seg_array = _CTHW_floatm11_to_video_array(gen_tensor)
                    seg_path = temp_output_dir / f"c_seg{seg_idx:02d}_generated_81.mp4"
                    imageio.mimsave(
                        str(seg_path),
                        seg_array,
                        fps=30,
                        codec='libx264',
                        pixelformat='yuv420p',
                        quality=8,
                    )
                    print(f"[Refine] Saved segment: {seg_path}")
            
            print(f"[Refine] Stitching {len(generated_segments)} segments...")
            
            from app_utils.stitch import stitch_segments_with_crossfade
            
            stitched = stitch_segments_with_crossfade(generated_segments, alpha=0.5)
            
            output_array = _CTHW_floatm11_to_video_array(stitched)
            
            print(f"[Refine] Output shape: {output_array.shape}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True, output_array, "Refinement completed successfully"
            
        except Exception as e:
            import traceback
            error_msg = f"Refinement error: {str(e)}\n{traceback.format_exc()}"
            print(f"[Refine] ✗ {error_msg}")
            return False, None, error_msg
    
    def unload_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Refine] Model unloaded")
