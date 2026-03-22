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
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

SELF_FORCING_ROOT = Path(__file__).resolve().parent.parent.parent / "Self-Forcing"


def make_cam_traj_from_preset_refspace(
    preset: str,
    step_m: float = 1.0,
    amp_m: float = 1.6,
    zigzag_span_m: float = 0.8,
) -> torch.Tensor:
    t_list = []
    
    if preset in ["forward", "backward", "left", "right"]:
        dir_map = {
            "forward":  np.array([+1, 0, 0], dtype=np.float64),
            "backward": np.array([-1, 0, 0], dtype=np.float64),
            "right":    np.array([0, 0, +1], dtype=np.float64),
            "left":     np.array([0, 0, -1], dtype=np.float64),
        }
        d = dir_map[preset] * (float(step_m) / 4.0)
        p = np.zeros(3, dtype=np.float64)
        for i in range(81):
            t_list.append(p.copy())
            p += d
    else:
        raise ValueError(f"Unsupported preset for Self-Forcing: {preset}. "
                        f"Self-Forcing only supports: forward, backward, left, right")
    
    traj = []
    for k in range(21):
        j = 4 * k
        t_ref = t_list[j]
        M = np.concatenate([np.eye(3, dtype=np.float64), t_ref.reshape(3, 1)], axis=1)
        traj.append(M.reshape(-1))
    traj = np.stack(traj, axis=0).astype(np.float32)
    return torch.from_numpy(traj)


def _parse_re_scale_pose(s: str):
    s = (s or "none").strip().lower()
    if s == "none":
        return "none", None
    if s == "unit_median":
        return "unit_median", 1.0
    if s.startswith("fixed:"):
        try:
            v = float(s.split("fixed:", 1)[1])
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError
            return "fixed", v
        except Exception:
            raise ValueError(f"Bad re_scale_pose value: {s}. Expect 'fixed:<positive-float>'.")
    raise ValueError(f"Unknown re_scale_pose value: {s}")


def _rescale_cam_traj_identityR(
    cam_traj_21: torch.Tensor,
    mode: str,
    s_target: float
) -> Tuple[torch.Tensor, Optional[float], Optional[float]]:
    if mode == "none":
        return cam_traj_21, None, None
    
    traj = cam_traj_21.clone()
    t_all = traj[:, [3, 7, 11]]
    
    dt = t_all[1:] - t_all[:-1]
    step_norms = torch.norm(dt, dim=1)
    
    s_local = float(step_norms.median().item())
    if s_local < 1e-8:
        s_local = 1e-8
    
    if mode == "unit_median":
        alpha = 1.0 / s_local
    elif mode == "fixed":
        alpha = float(s_target) / s_local
    else:
        return traj, None, None
    
    traj[:, 3] *= alpha
    traj[:, 7] *= alpha
    traj[:, 11] *= alpha
    
    return traj, s_local, alpha


def load_image_as_static_video(
    image_path: Path,
    height: int = 480,
    width: int = 960,
    num_frames: int = 81
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    
    frame = torch.from_numpy(arr).float() / 255.0
    frame = frame.permute(2, 0, 1)
    
    frame = frame.unsqueeze(0)
    frame = _batch_resize_with_padding(frame, height, width)
    
    frame = (frame - 0.5) / 0.5
    frame = frame[0]
    
    static_video = frame.unsqueeze(1).repeat(1, num_frames, 1, 1)
    
    return static_video


def _batch_resize_with_padding(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
    pad_value: float = 0.5
) -> torch.Tensor:
    B, C, H, W = frames.shape
    
    scale_h = target_h / H
    scale_w = target_w / W
    scale = min(scale_h, scale_w)
    
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    
    if new_h != H or new_w != W:
        frames = F.interpolate(
            frames, size=(new_h, new_w), 
            mode="bilinear", align_corners=False, antialias=True
        )
    
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if pad_h > 0 or pad_w > 0:
        frames = F.pad(
            frames, 
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant", value=pad_value
        )
    
    return frames


class SelfForcingInference:
    
    SUPPORTED_TRAJECTORIES = {"forward", "backward", "left", "right"}
    
    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        
        self.generator = None
        self.text_encoder = None
        self.vae = None
        self.inference_pipeline = None
        
        self.model_loaded = False
        
        self.config = None
        self.config_path = str(SELF_FORCING_ROOT / "configs" / "self_forcing_dmd_omniroam.yaml")
        
        self.height = 480
        self.width = 960
        self.num_frames = 81
        self.re_scale_mode = "fixed"
        self.re_scale_target = 1.0
        self.traj_step_m = 1.0
        self.traj_zigzag_span_m = 0.8
        
    def load_model(self, ckpt_path: str) -> bool:
        try:
            print(f"[SelfForcing] Loading model from {ckpt_path}")
            print(f"[SelfForcing] Config path: {self.config_path}")
            
            if not torch.distributed.is_initialized():
                import socket
                
                def find_free_port():
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('', 0))
                        s.listen(1)
                        port = s.getsockname()[1]
                    return port
                
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(find_free_port())
                
                torch.distributed.init_process_group(
                    backend='gloo',
                    init_method='env://',
                    world_size=1,
                    rank=0
                )
                print(f"[SelfForcing] Initialized single-process distributed environment")
            
            config = OmegaConf.load(self.config_path)
            default_config_path = str(SELF_FORCING_ROOT / "configs" / "default_config.yaml")
            default_config = OmegaConf.load(default_config_path)
            self.config = OmegaConf.merge(default_config, config)
            
            sf_path = str(SELF_FORCING_ROOT)
            
            print(f"[SelfForcing] SELF_FORCING_ROOT = {SELF_FORCING_ROOT}")
            
            if sf_path not in sys.path:
                sys.path.insert(0, sf_path)
            
            original_cwd = os.getcwd()
            
            os.chdir(SELF_FORCING_ROOT)
            
            try:
                from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper, WanTextEncoder
                from pipeline.custom_causal_inference import CustomCausalInferencePipeline
                
                print("[SelfForcing] Successfully imported Self-Forcing modules")
                
                device = torch.device(self.device)
                
                print("[SelfForcing] Loading VAE...")
                self.vae = WanVAEWrapper()
                self.vae = self.vae.to(device=device, dtype=self.dtype)
                self.vae.eval()
                
                print("[SelfForcing] Loading Text Encoder...")
                self.text_encoder = WanTextEncoder()
                self.text_encoder = self.text_encoder.to(device=device)
                self.text_encoder.eval()
                
                print("[SelfForcing] Loading Generator...")
                model_name = self.config.get("fake_name", "Wan2.1-T2V-1.3B")
                self.generator = WanDiffusionWrapper(
                    model_name=model_name,
                    timestep_shift=self.config.get("timestep_shift", 8.0),
                    is_causal=self.config.get("causal", True),
                    local_attn_size=self.config.get("local_attn_size", -1),
                    sink_size=self.config.get("sink_size", 0)
                )
                
                if self.config.get("use_custom_teacher", False):
                    print("[SelfForcing] Adding custom condition modules...")
                    dim = self.generator.model.dim
                    param_dtype = next(self.generator.model.parameters()).dtype
                    param_device = next(self.generator.model.parameters()).device
                    
                    self.generator.model.speed_token_proj = torch.nn.Linear(1, dim, bias=True).to(
                        dtype=param_dtype, device=param_device
                    )
                    torch.nn.init.normal_(self.generator.model.speed_token_proj.weight, mean=0.0, std=1e-2)
                    torch.nn.init.zeros_(self.generator.model.speed_token_proj.bias)
                    self.generator.model.speed_token_scale = torch.nn.Parameter(
                        torch.tensor([1e-1], dtype=param_dtype, device=param_device)
                    )
                    
                    num_blocks = len(self.generator.model.blocks)
                    for i in range(num_blocks):
                        block = self.generator.model.blocks[i]
                        block.cam_traj_encoder = torch.nn.Linear(12, dim, bias=True).to(
                            dtype=param_dtype, device=param_device
                        )
                        torch.nn.init.normal_(block.cam_traj_encoder.weight, mean=0.0, std=1e-2)
                        torch.nn.init.zeros_(block.cam_traj_encoder.bias)
                        block.projector = torch.nn.Linear(dim, dim, bias=True).to(
                            dtype=param_dtype, device=param_device
                        )
                        torch.nn.init.normal_(block.projector.weight, mean=0.0, std=1e-2)
                        torch.nn.init.zeros_(block.projector.bias)
                    
                    print(f"[SelfForcing]   ✓ Added condition modules to {num_blocks} blocks")
                
                print(f"[SelfForcing] Loading checkpoint from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location="cpu")
                
                def remove_fsdp_prefix(state_dict):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace("._fsdp_wrapped_module", "").replace("_fsdp_wrapped_module.", "")
                        new_state_dict[new_key] = value
                    return new_state_dict
                
                if "generator" in state_dict:
                    print("[SelfForcing] Using generator parameters")
                    gen_state = remove_fsdp_prefix(state_dict["generator"])
                    missing, unexpected = self.generator.load_state_dict(gen_state, strict=False)
                    if missing:
                        print(f"[SelfForcing] Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"[SelfForcing] Unexpected keys: {len(unexpected)}")
                elif "model" in state_dict:
                    print("[SelfForcing] Using model parameters")
                    self.generator.load_state_dict(state_dict["model"], strict=True)
                else:
                    print("[SelfForcing] Loading state dict directly")
                    self.generator.load_state_dict(state_dict, strict=True)
                
                self.generator = self.generator.to(device=device, dtype=self.dtype)
                self.generator.eval()
                
                print("[SelfForcing] Creating inference pipeline...")
                from types import SimpleNamespace
                
                pipeline_args = SimpleNamespace(
                    denoising_step_list=self.config.get("denoising_step_list", [1000, 750, 500, 250]),
                    warp_denoising_step=self.config.get("warp_denoising_step", True),
                    model_kwargs=self.config.get("model_kwargs", {"timestep_shift": 5.0}),
                    num_frame_per_block=self.config.get("num_frame_per_block", 3),
                    independent_first_frame=self.config.get("independent_first_frame", False),
                    context_noise=self.config.get("context_noise", 0),
                    height=self.height,
                    width=self.width
                )
                
                self.inference_pipeline = CustomCausalInferencePipeline(
                    args=pipeline_args,
                    device=device,
                    generator=self.generator,
                    text_encoder=self.text_encoder,
                    vae=self.vae
                )
                
                self.model_loaded = True
                print("[SelfForcing] ✓ Model loaded successfully!")
                return True
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"[SelfForcing] ✗ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
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
        seed: int = None,
        **kwargs,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        if not self.model_loaded or self.inference_pipeline is None:
            return False, None, "Model not loaded"
        
        if trajectory not in self.SUPPORTED_TRAJECTORIES:
            return False, None, (
                f"Trajectory '{trajectory}' not supported in Self-Forcing mode. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_TRAJECTORIES))}"
            )
        
        try:
            if seed is None:
                seed = random.randint(0, 2**31 - 1)
            print(f"[SelfForcing] Generating: trajectory={trajectory}, seed={seed}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            device = torch.device(self.device)
            
            print("[SelfForcing] Loading input image as static video...")
            static_video = load_image_as_static_video(
                input_image_path, height, width, num_frames
            )
            
            print(f"[SelfForcing] Generating preset trajectory: {trajectory}")
            trajectory_tensor = make_cam_traj_from_preset_refspace(
                preset=trajectory,
                step_m=self.traj_step_m,
                zigzag_span_m=self.traj_zigzag_span_m
            )
            
            trajectory_tensor, s_local, alpha = _rescale_cam_traj_identityR(
                trajectory_tensor, self.re_scale_mode, self.re_scale_target
            )
            if s_local is not None:
                print(f"[SelfForcing] Trajectory rescaled: s_local={s_local:.4f}, alpha={alpha:.4f}")
            
            static_video = static_video.unsqueeze(0).to(device=device, dtype=self.dtype)
            trajectory_tensor = trajectory_tensor.unsqueeze(0).to(device=device, dtype=self.dtype)
            
            print("[SelfForcing] Encoding static video to latent...")
            with torch.no_grad():
                input_latent = self.vae.encode_to_latent(static_video)
                input_latent = input_latent.to(dtype=self.dtype)
                print(f"[SelfForcing]   input_latent shape: {tuple(input_latent.shape)}")
            
            initial_latent = input_latent[:, -3:, :, :, :]
            print(f"[SelfForcing] Initial latent (last 3 frames): {tuple(initial_latent.shape)}")
            
            H_lat, W_lat = height // 8, width // 8
            noise = torch.randn(
                [1, 21, 16, H_lat, W_lat],
                device=device, dtype=self.dtype
            )
            
            speed_scalar = torch.tensor([[1.0]], device=device, dtype=self.dtype)
            
            print("[SelfForcing] Running causal inference...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            with torch.no_grad():
                video, generated_latent = self.inference_pipeline.inference(
                    noise=noise,
                    text_prompts=["panoramic video"],
                    cam_traj=trajectory_tensor,
                    speed_scalar=speed_scalar,
                    initial_latent=initial_latent,
                    return_latents=True
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            print("[SelfForcing] Interpolating 21 -> 81 frames...")
            video_21 = rearrange(video, 'b t c h w -> b c t h w')
            video_81 = F.interpolate(
                video_21, size=(81, video_21.shape[3], video_21.shape[4]),
                mode='trilinear', align_corners=False
            )
            
            video_out = video_81[0]
            video_out = rearrange(video_out, 'c t h w -> t h w c')
            video_out = video_out.float().cpu()
            video_out = video_out.clamp(0, 1)
            video_out = (video_out * 255).round().to(torch.uint8).numpy()
            
            print(f"[SelfForcing] Generation complete: {video_out.shape}")
            return True, video_out, "Generation successful"
            
        except Exception as e:
            import traceback
            error_msg = f"Generation error: {str(e)}\n{traceback.format_exc()}"
            print(f"[SelfForcing] ✗ {error_msg}")
            return False, None, error_msg
    
    def unload_model(self):
        if self.generator is not None:
            del self.generator
            self.generator = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.inference_pipeline is not None:
            del self.inference_pipeline
            self.inference_pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        print("[SelfForcing] Model unloaded")
