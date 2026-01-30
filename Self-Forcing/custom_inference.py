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

import argparse
import torch
import os
import json
import numpy as np
from typing import Tuple, Optional
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from PIL import Image

from pipeline.custom_causal_inference import CustomCausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper, WanTextEncoder
from utils.misc import set_seed
from torchvision.io import write_video


# =========================
# Trajectory Generation & Processing
# =========================

def make_cam_traj_from_preset_refspace(
    preset: str,
    step_m: float = 1.0,
) -> torch.Tensor:
    t_list = []
    
    if preset in ["forward","backward","left","right"]:
        dir_map = {
            "forward":  np.array([+1, 0, 0], dtype=np.float64),
            "backward": np.array([-1, 0, 0], dtype=np.float64),
            "right":    np.array([0, 0, +1], dtype=np.float64),
            "left":     np.array([0, 0, -1], dtype=np.float64),
        }
        # Every 4 frames = one latent timestep; step_m is displacement per latent timestep
        d = dir_map[preset] * (float(step_m) / 4.0)
        p = np.zeros(3, dtype=np.float64)
        for i in range(81):
            t_list.append(p.copy())
            p += d
    else:
        raise ValueError(f"Unknown preset: {preset}. Supported: forward, backward, left, right")
    
    # Sample every 4th frame to get 21 frames, pack as [I|t]
    traj = []
    for k in range(21):
        j = 4 * k
        t_ref = t_list[j]
        M = np.concatenate([np.eye(3, dtype=np.float64), t_ref.reshape(3,1)], axis=1)  # (3,4)
        traj.append(M.reshape(-1))  # 12 values
    traj = np.stack(traj, axis=0).astype(np.float32)  # (21,12)
    return torch.from_numpy(traj)


def _parse_re_scale_pose(s: str) -> Tuple[str, Optional[float]]:
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
            raise ValueError(f"Bad --re_scale_pose value: {s}. Expect 'fixed:<positive-float>'.")
    raise ValueError(f"Unknown --re_scale_pose value: {s}")


def _rescale_cam_traj_identityR(
    cam_traj_21: torch.Tensor, 
    mode: str, 
    s_target: Optional[float]
) -> Tuple[torch.Tensor, Optional[float], float]:
    if cam_traj_21 is None or mode == "none":
        return cam_traj_21, None, 1.0
    
    # Extract translation vectors: t is at columns [3, 7, 11]
    M = cam_traj_21.view(21, 3, 4)  # (21, 3, 4) -> [I|t] per (3,4) block
    t = M[:, :, 3]  # (21, 3) - translations
    
    if t.shape[0] < 2:
        return cam_traj_21, None, 1.0
    
    # Compute median step size
    dt = t[1:] - t[:-1]  # (20, 3)
    step = torch.linalg.norm(dt, dim=1)  # (20,)
    
    if step.numel() == 0:
        return cam_traj_21, None, 1.0
    
    s_local = torch.median(step).item()
    
    eps = 1e-8
    if not np.isfinite(s_local) or s_local < eps:
        return cam_traj_21, max(s_local, 0.0), 1.0
    
    # Determine target scale
    if mode == "unit_median":
        s_tgt = 1.0
    elif mode == "fixed":
        s_tgt = float(s_target)
    else:
        return cam_traj_21, s_local, 1.0
    
    # Scale translations
    alpha = s_tgt / s_local
    t_scaled = t * alpha
    M[:, :, 3] = t_scaled
    
    return M.reshape(21, 12), s_local, alpha


# =========================
# Image Loading and Processing
# =========================


def _batch_resize_with_padding(batch_imgs, target_height, target_width):
    B, C, H, W = batch_imgs.shape
    
    # Scale by longest dimension
    tgt_long = max(target_height, target_width)
    src_long = max(H, W)
    scale = tgt_long / float(src_long)
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    
    if new_h != H or new_w != W:
        batch_imgs = F.interpolate(
            batch_imgs, size=(new_h, new_w),
            mode="bilinear", align_corners=False, antialias=True
        )
    
    # Pad/crop height
    if new_h < target_height:
        pad_total = target_height - new_h
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        batch_imgs = F.pad(batch_imgs, (0, 0, pad_top, pad_bot), value=0.0)
    elif new_h > target_height:
        top = (new_h - target_height) // 2
        batch_imgs = batch_imgs[:, :, top:top + target_height, :]
    
    # Pad/crop width
    if new_w < target_width:
        pad_total = target_width - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        batch_imgs = F.pad(batch_imgs, (pad_left, pad_right, 0, 0), value=0.0)
    elif new_w > target_width:
        left = (new_w - target_width) // 2
        batch_imgs = batch_imgs[:, :, :, left:left + target_width]
    
    return batch_imgs


def save_mp4_from_tensor(video_tensor: torch.Tensor, output_path: str, fps: int = 16):
    # Handle different input formats
    if video_tensor.dim() == 4:
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            video_tensor = rearrange(video_tensor, 'c t h w -> t h w c')
        # else assume (T, H, W, C)
    
    # Ensure on CPU
    video_tensor = video_tensor.cpu()
    
    # Normalize to [0, 1] if in [-1, 1]
    if video_tensor.min() < -0.1:
        video_tensor = (video_tensor + 1.0) / 2.0
    
    # Convert to uint8
    video_tensor = (255.0 * video_tensor).clamp(0, 255).to(torch.uint8)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_video(output_path, video_tensor, fps=fps)


def save_cam_traj_json(traj_21x12: torch.Tensor, out_json_path: str, rescale_info: dict = None):
    # Convert 21x12 trajectory to 81x3 positions
    # Each row of 21x12 is [I|t] = [1,0,0,x, 0,1,0,y, 0,0,1,z]
    # Extract positions: columns [3, 7, 11]
    traj_21x12 = traj_21x12.cpu().numpy()
    M = traj_21x12.reshape(21, 3, 4)  # (21, 3, 4)
    pos_21 = M[:, :, 3]  # (21, 3) - positions at every 4th frame
    
    # Interpolate to 81 frames (linear interpolation)
    from scipy.interpolate import interp1d
    frame_indices_21 = np.arange(0, 81, 4)  # [0, 4, 8, ..., 80]
    frame_indices_81 = np.arange(81)
    
    # Interpolate each dimension
    pos_81 = np.zeros((81, 3), dtype=np.float32)
    for dim in range(3):
        f = interp1d(frame_indices_21, pos_21[:, dim], kind='linear', fill_value='extrapolate')
        pos_81[:, dim] = f(frame_indices_81)
    
    # Build JSON structure
    frames = []
    for i in range(81):
        t = pos_81[i].tolist()  # [x, y, z]
        # [I|t] = [[1,0,0,x], [0,1,0,y], [0,0,1,z]]
        matrix_3x4 = [
            [1.0, 0.0, 0.0, t[0]],
            [0.0, 1.0, 0.0, t[1]],
            [0.0, 0.0, 1.0, t[2]]
        ]
        frames.append({
            "frame_index": i,
            "position": t,
            "matrix_3x4": matrix_3x4
        })
    
    data = {
        "_comment": [
            "Camera trajectory in reference camera coordinate system",
            "Coordinate system: +X=forward, +Y=up, +Z=right",
            "Each frame contains:",
            "  - position: [x, y, z] translation in reference camera space",
            "  - matrix_3x4: 3x4 transformation matrix [I|t] where I is identity rotation, t is translation",
            "  - Flattened as [1,0,0,x, 0,1,0,y, 0,0,1,z] for model input",
            "Total 81 frames corresponding to the generated video"
        ],
        "num_frames": 81,
        "frames": frames
    }
    
    if rescale_info:
        data["rescale_info"] = rescale_info
    
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(data, f, indent=2)


# =========================
# Main Inference
# =========================

def parse_args():
    parser = argparse.ArgumentParser("Custom DMD Inference with Trajectory Conditioning")
    
    # Model & checkpoint
    parser.add_argument("--config_path", type=str, required=True, 
                        help="Path to config file (e.g., configs/self_forcing_dmd_omniroam.yaml)")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to checkpoint (e.g., logs/checkpoint_model_006000/model.pt)")
    parser.add_argument("--use_ema", action="store_true", 
                        help="Use EMA parameters (generator_ema instead of generator)")
    
    # Data source (local only)
    parser.add_argument("--local_folder", type=str, required=True,
                        help="Local folder containing panorama images. "
                             "Supported formats: .jpg, .jpeg, .png, .bmp")
    
    # Trajectory settings
    parser.add_argument("--traj_preset", type=str, default="forward",
                        choices=["forward","backward","left","right"],
                        help="Preset trajectory for camera movement")
    parser.add_argument("--traj_step_m", type=float, default=1.0,
                        help="Step size in meters per latent timestep")
    parser.add_argument("--re_scale_pose", type=str, default="fixed:1.0",
                        help="Trajectory rescale: none | unit_median | fixed:<float>")
    
    # Inference settings
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default="panoramic video",
                        help="Text prompt for generation")
    parser.add_argument("--output_folder", type=str, default="./custom_inference_output")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of videos to generate (limited by images in folder)")
    parser.add_argument("--seed", type=int, default=2025)
    
    # Speed control
    parser.add_argument("--speed_scalar", type=float, default=1.0,
                        help="Speed scalar (1.0 = normal speed)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.distributed.is_initialized():
        import os
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
        print(f"Initialized single-process distributed environment on port {os.environ['MASTER_PORT']}")
    
    # Load config
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    print("\n" + "="*80)
    print("Initializing Model Components")
    print("="*80)
    
    model_name = config.get("fake_name", "Wan2.1-T2V-1.3B")
    print(f"Model: {model_name}")
    print(f"Note: VAE and Text Encoder use hardcoded paths from wan_models/{model_name}/")
    
    # 1. VAE
    print("Loading VAE...")
    vae = WanVAEWrapper()  # No model_name parameter
    vae = vae.to(device=device, dtype=torch.bfloat16)
    vae.eval()
    
    # 2. Text Encoder
    print("Loading Text Encoder...")
    text_encoder = WanTextEncoder()  # No model_name parameter
    text_encoder = text_encoder.to(device=device)
    text_encoder.eval()
    
    # 3. Generator
    print("Loading Generator...")
    generator = WanDiffusionWrapper(
        model_name=model_name,
        timestep_shift=config.get("timestep_shift", 8.0),
        is_causal=config.get("causal", True),
        local_attn_size=config.get("local_attn_size", -1),
        sink_size=config.get("sink_size", 0)
    )
    
    use_custom_teacher = config.get("use_custom_teacher", False)
    if use_custom_teacher:
        print("Adding custom condition modules to generator (camera trajectory + speed)...")
        dim = generator.model.dim
        param_dtype = next(generator.model.parameters()).dtype
        param_device = next(generator.model.parameters()).device
        
        # Speed token modules
        generator.model.speed_token_proj = torch.nn.Linear(1, dim, bias=True).to(dtype=param_dtype, device=param_device)
        torch.nn.init.normal_(generator.model.speed_token_proj.weight, mean=0.0, std=1e-2)
        torch.nn.init.zeros_(generator.model.speed_token_proj.bias)
        generator.model.speed_token_scale = torch.nn.Parameter(torch.tensor([1e-1], dtype=param_dtype, device=param_device))
        
        # Camera trajectory encoder and projector for each block
        num_blocks = len(generator.model.blocks)
        for i in range(num_blocks):
            block = generator.model.blocks[i]
            # Trajectory encoder: 12-dim -> dim
            block.cam_traj_encoder = torch.nn.Linear(12, dim, bias=True).to(dtype=param_dtype, device=param_device)
            torch.nn.init.normal_(block.cam_traj_encoder.weight, mean=0.0, std=1e-2)
            torch.nn.init.zeros_(block.cam_traj_encoder.bias)
            # Projector for fusion
            block.projector = torch.nn.Linear(dim, dim, bias=True).to(dtype=param_dtype, device=param_device)
            torch.nn.init.normal_(block.projector.weight, mean=0.0, std=1e-2)
            torch.nn.init.zeros_(block.projector.bias)
        
        print(f"  ✓ Added speed_token_proj, speed_token_scale")
        print(f"  ✓ Added cam_traj_encoder + projector to {num_blocks} blocks")
    
    print(f"\nLoading checkpoint from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")

    def remove_fsdp_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove all occurrences of '_fsdp_wrapped_module.'
            new_key = key.replace("._fsdp_wrapped_module", "").replace("_fsdp_wrapped_module.", "")
            new_state_dict[new_key] = value
        return new_state_dict
    
    if args.use_ema and "generator_ema" in state_dict:
        print("Using EMA parameters")
        generator_state = remove_fsdp_prefix(state_dict["generator_ema"])
        missing_keys, unexpected_keys = generator.load_state_dict(generator_state, strict=False)
        print(f"\n⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:10] if len(missing_keys) > 10 else missing_keys}")
        print(f"\n⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10] if len(unexpected_keys) > 10 else unexpected_keys}")
    elif "generator" in state_dict:
        print("Using generator parameters")
        missing_keys, unexpected_keys = generator.load_state_dict(state_dict["generator"], strict=False)
        print(f"\n⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:10] if len(missing_keys) > 10 else missing_keys}")
        print(f"\n⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10] if len(unexpected_keys) > 10 else unexpected_keys}")
    elif "model" in state_dict:
        print("Using model parameters")
        generator.load_state_dict(state_dict["model"], strict=True)
    else:
        print("Loading state dict directly")
        generator.load_state_dict(state_dict, strict=True)
    
    generator = generator.to(device=device, dtype=torch.bfloat16)
    generator.eval()
    
    print("\nCreating CustomCausalInferencePipeline...")
    
    from types import SimpleNamespace
    pipeline_args = SimpleNamespace(
        denoising_step_list=config.get("denoising_step_list", [1000, 750, 500, 250]),
        warp_denoising_step=config.get("warp_denoising_step", True),
        model_kwargs=config.get("model_kwargs", {"timestep_shift": 5.0}),
        num_frame_per_block=config.get("num_frame_per_block", 3),
        independent_first_frame=config.get("independent_first_frame", False),
        context_noise=config.get("context_noise", 0),
        height=args.height,
        width=args.width
    )
    
    inference_pipeline = CustomCausalInferencePipeline(
        args=pipeline_args,
        device=device,
        generator=generator,
        text_encoder=text_encoder,
        vae=vae
    )
    
    print("✓ Model initialization complete")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Parse rescale mode
    re_scale_mode, re_scale_target = _parse_re_scale_pose(args.re_scale_pose)
    print(f"Trajectory rescale: mode={re_scale_mode}, target={re_scale_target}")
    
    local_folder_path = args.local_folder
    print(f"\nLoading panorama images from local folder: {local_folder_path}")
    
    if not os.path.isdir(local_folder_path):
        raise RuntimeError(f"Local folder does not exist: {local_folder_path}")
    
    img_files = [f for f in os.listdir(local_folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(img_files) == 0:
        raise RuntimeError(f"No image files found in {local_folder_path}. "
                         f"Supported formats: .jpg, .jpeg, .png, .bmp")
    
    img_files = sorted(img_files)
    print(f"Found {len(img_files)} image(s) in folder")
    
    if args.num_samples < len(img_files):
        img_files = img_files[:args.num_samples]
        print(f"Limited to first {args.num_samples} image(s)")
    else:
        print(f"Will process all {len(img_files)} image(s)")
    
    print("\nImages to process:")
    for i, img_file in enumerate(img_files, 1):
        print(f"  {i}. {img_file}")
    
    print("\n" + "="*80)
    print(f"Starting Inference ({len(img_files)} samples)")
    print("="*80 + "\n")
    
    for idx in tqdm(range(len(img_files)), desc="Generating videos"):
        try:
            print(f"\n{'='*80}")
            
            img_file = img_files[idx]
            print(f"[{idx+1}/{len(img_files)}] Processing local image: {img_file}")
            
            # Load image
            img_path = os.path.join(args.local_folder, img_file)
            img = Image.open(img_path).convert("RGB")
            
            # Convert to tensor
            arr = np.array(img, dtype=np.uint8)
            frame = torch.from_numpy(arr).float() / 255.0
            frame = frame.permute(2, 0, 1)
            
            frame = frame.unsqueeze(0)
            frame = _batch_resize_with_padding(frame, args.height, args.width)
            frame = (frame - 0.5) / 0.5
            frame = frame[0]
            
            static_video = frame.unsqueeze(1).repeat(1, 81, 1, 1)
            
            print(f"  Generating preset trajectory: {args.traj_preset}")
            trajectory = make_cam_traj_from_preset_refspace(
                args.traj_preset,
                step_m=args.traj_step_m
            )
            
            safe_name = os.path.splitext(img_file)[0]
            output_subdir = os.path.join(args.output_folder, safe_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            trajectory_before_rescale = trajectory.clone()
            trajectory, s_local, alpha = _rescale_cam_traj_identityR(
                trajectory, re_scale_mode, re_scale_target
            )
            rescale_info = None
            if s_local is not None:
                print(f"  Trajectory rescaled: s_local={s_local:.4f}, alpha={alpha:.4f}")
                rescale_info = {
                    "mode": re_scale_mode,
                    "s_local": float(s_local),
                    "alpha": float(alpha),
                    "target": float(re_scale_target) if re_scale_target is not None else None
                }
            
            static_video = static_video.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            trajectory = trajectory.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            
            print("  Encoding static video to latent...")
            with torch.no_grad():
                input_latent = vae.encode_to_latent(static_video)
                input_latent = input_latent.to(dtype=torch.bfloat16)
                print(f"    └─ input_latent shape: {tuple(input_latent.shape)}, dtype: {input_latent.dtype}")
            
            initial_latent = input_latent[:, -3:, :, :, :]
            print(f"  Extracted initial_latent (last 3 frames): {tuple(initial_latent.shape)}, dtype: {initial_latent.dtype}")
            
            print("  Encoding text prompt...")
            with torch.no_grad():
                conditional_dict = text_encoder(text_prompts=[args.prompt])
                conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(dtype=torch.bfloat16)
                conditional_dict["cam_traj"] = trajectory
                conditional_dict["speed_scalar"] = torch.tensor([[args.speed_scalar]], 
                                                                device=device, dtype=torch.bfloat16)
            
            H, W = args.height, args.width
            noise = torch.randn(
                [1, 21, 16, H // 8, W // 8],
                device=device, dtype=torch.bfloat16
            )
            
            print(f"\n  [DEBUG] Inputs before inference:")
            print(f"    ├─ noise shape: {noise.shape}, dtype: {noise.dtype}")
            print(f"    ├─ noise stats: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")
            print(f"    ├─ initial_latent shape: {initial_latent.shape}, dtype: {initial_latent.dtype}")
            print(f"    ├─ initial_latent stats: mean={initial_latent.mean().item():.4f}, std={initial_latent.std().item():.4f}")
            print(f"    ├─ trajectory shape: {trajectory.shape}")
            print(f"    ├─ trajectory sample [0,0,:]: {trajectory[0,0,:].tolist()}")
            print(f"    └─ speed_scalar: {conditional_dict['speed_scalar'].item():.2f}")
            
            print("\n  Generating video with CustomCausalInferencePipeline...")
            
            with torch.no_grad():
                video, generated_latent = inference_pipeline.inference(
                    noise=noise,
                    text_prompts=[args.prompt],
                    cam_traj=trajectory,
                    speed_scalar=conditional_dict["speed_scalar"],
                    initial_latent=initial_latent,
                    return_latents=True
                )
            
            print(f"\n  [DEBUG] Outputs after inference:")
            print(f"    ├─ video shape: {tuple(video.shape)}")
            print(f"    ├─ video stats: mean={video.mean().item():.4f}, std={video.std().item():.4f}")
            print(f"    ├─ generated_latent shape: {tuple(generated_latent.shape)}")
            print(f"    ├─ generated_latent stats: mean={generated_latent.mean().item():.4f}, std={generated_latent.std().item():.4f}")
            print(f"    └─ generated_latent min/max: [{generated_latent.min().item():.4f}, {generated_latent.max().item():.4f}]")
            
            if generated_latent.std().item() > 1.5:
                print(f"    ⚠️  WARNING: Generated latent has very high std ({generated_latent.std().item():.4f})")
                print(f"    ⚠️  Normal latent std is typically < 1.0. This suggests denoising may not be working!")

            
            print(f"  Saving outputs to {output_subdir}...")
            
            video_21 = rearrange(video, 'b t c h w -> b c t h w')
            
            video_81 = F.interpolate(
                video_21, size=(81, video_21.shape[3], video_21.shape[4]),
                mode='trilinear', align_corners=False
            )
            video_81 = rearrange(video_81[0], 'c t h w -> c t h w')
            
            generated_path = os.path.join(output_subdir, 'generated.mp4')
            save_mp4_from_tensor(video_81, generated_path, fps=30)
            print(f"    ✓ Saved generated.mp4")
            
            input_path = os.path.join(output_subdir, 'input.mp4')
            save_mp4_from_tensor(static_video[0], input_path, fps=30)
            print(f"    ✓ Saved input.mp4")
            
            trajectory_path = os.path.join(output_subdir, 'trajectory.json')
            save_cam_traj_json(trajectory_before_rescale, trajectory_path, rescale_info)
            print(f"    ✓ Saved trajectory.json")
            
            print(f"  ✓ All outputs saved to: {output_subdir}")
            print("="*80)
            
        except Exception as e:
            print(f"\n✗ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            print("="*80)
            continue
    
    print(f"\n{'='*80}")
    print(f"Inference Complete!")
    print(f"Results saved to: {args.output_folder}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
