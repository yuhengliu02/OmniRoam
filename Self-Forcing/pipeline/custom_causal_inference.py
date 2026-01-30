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

from typing import List, Optional
import torch
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CustomCausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        print("\n" + "="*80)
        print("Initializing CustomCausalInferencePipeline")
        print("  - Supports: cam_traj (B, 21, 12), speed_scalar (B, 1)")
        print("  - Uses FULL trajectory for all blocks (matching training)")
        print("="*80 + "\n")
        
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize hyperparameters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            # Keep on CPU for indexing, then move to device
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list].to(device)
            print(f"[CustomPipeline] Applied warp_denoising_step: {self.denoising_step_list.tolist()}")
        else:
            # Move to device after initialization
            self.denoising_step_list = self.denoising_step_list.to(device)
            print(f"[CustomPipeline] Denoising steps: {self.denoising_step_list.tolist()}")

        self.num_transformer_blocks = 30
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.context_noise = getattr(args, "context_noise", 0)
        
        # Dynamically compute frame_seq_length based on resolution
        height = getattr(args, "height", 480)
        width = getattr(args, "width", 960)
        # For WAN2.1: patch_size = (1, 2, 2), so each latent frame -> (H/2) * (W/2) tokens
        self.frame_seq_length = (height // 8 // 2) * (width // 8 // 2)
        kv_cache_size = 24 * self.frame_seq_length  # 24 frames max
        
        print(f"[CustomPipeline] Configuration:")
        print(f"  - num_frame_per_block: {self.num_frame_per_block}")
        print(f"  - frame_seq_length: {self.frame_seq_length} (latent {height//8}x{width//8})")
        print(f"  - kv_cache_size: {kv_cache_size}")
        print(f"  - context_noise: {self.context_noise}")

        self.kv_cache_size = kv_cache_size
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.args = args

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,  # (B, 21, 16, H, W)
        text_prompts: List[str],
        cam_traj: Optional[torch.Tensor] = None,  # (B, 21, 12) - FULL trajectory
        speed_scalar: Optional[torch.Tensor] = None,  # (B, 1)
        initial_latent: Optional[torch.Tensor] = None,  # (B, 3, 16, H, W) or None
        return_latents: bool = False,
    ) -> torch.Tensor:
        """
        Perform causal inference with trajectory conditioning.
        
        Args:
            noise: Input noise (B, num_frames, 16, H_lat, W_lat)
            text_prompts: List of text prompts
            cam_traj: Camera trajectory (B, 21, 12) - FULL trajectory, not block-specific!
            speed_scalar: Speed conditioning (B, 1)
            initial_latent: Initial context frames (B, 3, 16, H_lat, W_lat) for DMD
            return_latents: Whether to return latent representations
            
        Returns:
            video: Generated video (B, num_frames, 3, H, W) in [0, 1]
            latents (optional): Latent representations
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # Validate inputs
        assert num_frames % self.num_frame_per_block == 0, \
            f"num_frames ({num_frames}) must be divisible by num_frame_per_block ({self.num_frame_per_block})"
        
        num_blocks = num_frames // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        
        print(f"\n[CustomPipeline.inference] Starting generation:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - num_frames: {num_frames} ({num_blocks} blocks)")
        print(f"  - num_input_frames: {num_input_frames}")
        print(f"  - noise shape: {tuple(noise.shape)}")
        if cam_traj is not None:
            print(f"  - cam_traj shape: {tuple(cam_traj.shape)}")
        if speed_scalar is not None:
            print(f"  - speed_scalar: {speed_scalar.item():.2f}")
        
        # Step 1: Encode text
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        
        # Convert text embeddings to match generator dtype (bfloat16)
        if "prompt_embeds" in conditional_dict:
            conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(dtype=noise.dtype)
        
        # Step 2: Add trajectory conditions (IMPORTANT: Use FULL trajectory!)
        if cam_traj is not None:
            # Keep the FULL 21-frame trajectory - don't slice it per block!
            # This matches how the model was trained
            conditional_dict["cam_traj"] = cam_traj
            print(f"  ✓ Added FULL trajectory to conditional_dict: {tuple(cam_traj.shape)}")
        
        if speed_scalar is not None:
            conditional_dict["speed_scalar"] = speed_scalar
            print(f"  ✓ Added speed_scalar to conditional_dict: {speed_scalar.item():.2f}")
        
        # Prepare output tensor
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        
        # Step 3: Initialize KV cache
        if self.kv_cache1 is None:
            print(f"  [KV Cache] Initializing new cache...")
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            print(f"  [KV Cache] Resetting existing cache...")
            # Reset cache for new generation
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
        
        # Step 4: Pre-fill KV cache with initial_latent (if provided)
        current_start_frame = 0
        if initial_latent is not None:
            print(f"\n  [KV Pre-fill] Using initial_latent: {tuple(initial_latent.shape)}")
            num_init_frames = initial_latent.shape[1]
            
            # Create identity trajectory for static initial frames
            prefill_conditional_dict = conditional_dict.copy()
            if cam_traj is not None:
                identity_traj = torch.zeros(batch_size, num_init_frames, 12, 
                                           device=cam_traj.device, dtype=cam_traj.dtype)
                identity_traj[:, :, 0] = 1.0  # R[0,0]
                identity_traj[:, :, 4] = 1.0  # R[1,1]
                identity_traj[:, :, 8] = 1.0  # R[2,2]
                prefill_conditional_dict["cam_traj"] = identity_traj
                print(f"    - Created identity trajectory for {num_init_frames} static frames")
            
            timestep = torch.zeros([batch_size, num_init_frames], device=noise.device, dtype=torch.int64)
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=prefill_conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length  # KV cache at position [0,1,2]
                )
            
            # ===== NEW: Write initial_latent to output for alignment =====
            output[:, current_start_frame:current_start_frame + num_init_frames] = initial_latent
            # =============================================================
            
            print(f"    ✓ KV cache pre-filled with {num_init_frames} frames")
            
            # ===== Increment current_start_frame for append mode (matching training) =====
            current_start_frame += num_init_frames  # Now = 3
            print(f"    ℹ️  current_start_frame incremented to {current_start_frame} (append mode)")
            # =============================================================================
        
        # Step 5: Multi-step denoising (block by block)
        print(f"\n  [Denoising] Processing {num_blocks} blocks...")
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for block_idx, current_num_frames in enumerate(all_num_frames):
            print(f"    Block {block_idx+1}/{num_blocks} ({current_num_frames} frames):")
            
            noisy_input = noise[:, block_idx * self.num_frame_per_block:(block_idx + 1) * self.num_frame_per_block]
            # ===========================================================================
            
            block_conditional_dict = conditional_dict.copy()
            if cam_traj is not None:
                # Get trajectory for current block's frames (aligned with noise indexing)
                traj_start = block_idx * self.num_frame_per_block
                traj_end = traj_start + self.num_frame_per_block
                block_conditional_dict["cam_traj"] = cam_traj[:, traj_start:traj_end, :]
                print(f"      └─ block cam_traj: [{traj_start}:{traj_end}], shape: {block_conditional_dict['cam_traj'].shape}")
            
            # Multi-step denoising for this block
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_conditional_dict,  # Uses FULL trajectory!
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                
                # Add noise for next step (except last step)
                if step_idx < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[step_idx + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, (batch_size, current_num_frames))
                else:
                    noisy_input = denoised_pred
            
            # Record output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            # Update KV cache with clean prediction (context noise)
            context_timestep = torch.ones_like(timestep) * self.context_noise
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=block_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            
            current_start_frame += current_num_frames
            print(f"      ✓ Block {block_idx+1} complete")
        
        if num_input_frames > 0:
            print(f"\n  [Post-processing] Keeping last 21 frames (frames {num_input_frames}:24)")
            output = output[:, num_input_frames:, :, :, :]  # [:, 3:24] = 21 frames
            print(f"    └─ Output shape after trimming: {tuple(output.shape)}")
        
        print(f"\n  [Decoding] Converting latents to pixels...")
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        print(f"  ✓ Generation complete: {tuple(video.shape)}\n")
        
        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize KV cache for the Wan model."""
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize cross-attention cache for the Wan model."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

