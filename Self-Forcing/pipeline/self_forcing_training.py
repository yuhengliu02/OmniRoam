from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 debug: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = None
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = None
        self.num_max_frames = num_max_frames
        self.debug = debug

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            cam_traj: Optional[torch.Tensor] = None,
            speed_scalar: Optional[torch.Tensor] = None,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        if self.frame_seq_length is None:
            self.frame_seq_length = (height // 2) * (width // 2)
            self.kv_cache_size = self.num_max_frames * self.frame_seq_length
            print(f"[Pipeline] Dynamically set frame_seq_length = {self.frame_seq_length} "
                  f"(latent {height}x{width}, kv_cache_size={self.kv_cache_size})")
        
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
        if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"\n{'='*80}")
            print(f"[DEBUG][Pipeline.Init] Output tensor configuration")
            print(f"  ├─ num_frames (noise): {num_frames}")
            print(f"  ├─ num_input_frames (initial_latent): {num_input_frames}")
            print(f"  ├─ num_output_frames (total): {num_output_frames}")
            print(f"  ├─ num_blocks: {num_blocks}")
            print(f"  └─ Expected: output[:, 0:{num_output_frames}] will be populated")
            print(f"{'='*80}\n")
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        current_start_frame = 0
        
        if initial_latent is not None:
            num_init_frames = initial_latent.shape[1]
            timestep = torch.ones([batch_size, num_init_frames], device=noise.device, dtype=torch.int64) * 0
            
            if self.debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"\n{'='*80}")
                print(f"[DEBUG][Pipeline.KV_PreFill] Initializing KV cache with initial_latent")
                print(f"  ├─ initial_latent shape: {tuple(initial_latent.shape)}")
                print(f"  ├─ initial_latent stats: mean={initial_latent.mean().item():.4f}, std={initial_latent.std().item():.4f}")
                print(f"  ├─ num_init_frames: {num_init_frames}")
                print(f"  └─ Purpose: Pre-fill KV cache from static video's last {num_init_frames} frames")
            
            prefill_conditional_dict = conditional_dict.copy()
            
            if cam_traj is not None:
                identity_traj = torch.zeros(batch_size, num_init_frames, 12, 
                                           device=cam_traj.device, dtype=cam_traj.dtype)
                for i in range(num_init_frames):
                    identity_traj[:, i, 0] = 1.0
                    identity_traj[:, i, 4] = 1.0
                    identity_traj[:, i, 8] = 1.0
                
                prefill_conditional_dict["cam_traj"] = identity_traj
                
                if self.debug and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"  ├─ Created identity trajectory for pre-fill:")
                    print(f"  │  ├─ Shape: {tuple(identity_traj.shape)}")
                    print(f"  │  ├─ Identity matrix diagonal: [{identity_traj[0,0,0]:.1f}, {identity_traj[0,0,4]:.1f}, {identity_traj[0,0,8]:.1f}]")
                    print(f"  │  └─ Translation: [{identity_traj[0,0,3]:.1f}, {identity_traj[0,0,7]:.1f}, {identity_traj[0,0,11]:.1f}]")
            
            if speed_scalar is not None:
                prefill_conditional_dict["speed_scalar"] = speed_scalar
                if self.debug and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"  ├─ speed_scalar: {speed_scalar.item():.2f}")
            # ================================================
            
            # Cache initial latent through generator to populate KV cache
            # These frames are NOT part of the output (they are context from input video)
            if self.debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  ├─ Running generator forward to populate KV cache...")
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=prefill_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            
            output[:, current_start_frame:current_start_frame + num_init_frames] = initial_latent
            
            if self.debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  └─ ✓ KV cache pre-filled successfully")
                print(f"{'='*80}\n")
            
            current_start_frame += num_init_frames
            # ===============================================================================
            
            if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"\n[DEBUG][Pipeline.PostPreFill] After KV cache pre-fill")
                print(f"  ├─ current_start_frame: {current_start_frame} (incremented for append mode)")
                print(f"  └─ Next block will write to: output[:, {current_start_frame}:{current_start_frame + self.num_frame_per_block}]\n")
            # ================================================

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"\n{'='*80}")
            print(f"[DEBUG][Pipeline.Denoising] Configuration")
            print(f"  ├─ num_blocks: {num_blocks}")
            print(f"  ├─ num_denoising_steps: {num_denoising_steps}")
            print(f"  ├─ exit_flags: {exit_flags}")
            print(f"  ├─ last_step_only: {self.last_step_only}")
            print(f"  ├─ same_step_across_blocks: {self.same_step_across_blocks}")
            print(f"  └─ denoising_step_list: {[float(x) for x in self.denoising_step_list.tolist()]}")
            print(f"{'='*80}\n")
        # ================================================

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, block_index * self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]
            block_conditional_dict = conditional_dict.copy()
            if cam_traj is not None and current_num_frames == self.num_frame_per_block:
                traj_start = block_index * self.num_frame_per_block
                traj_end = traj_start + self.num_frame_per_block
                block_traj = cam_traj[:, traj_start:traj_end, :]
                block_conditional_dict["cam_traj"] = block_traj
            
            if speed_scalar is not None:
                block_conditional_dict["speed_scalar"] = speed_scalar
            # ================================================

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                
                if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"  [Block {block_index}] Step {index}/{num_denoising_steps-1}: timestep={float(current_timestep):.1f}, exit_flag={exit_flag}")
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=block_conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    
                    if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"  [Block {block_index}] Exiting denoising loop at step {index} (exit_flag=True)")
                    break

            # Step 3.2: record the model's output
            if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"  [Block {block_index}] Writing to output")
                print(f"    ├─ current_start_frame: {current_start_frame}")
                print(f"    ├─ Writing to: output[:, {current_start_frame}:{current_start_frame + current_num_frames}]")
                print(f"    ├─ denoised_pred shape: {tuple(denoised_pred.shape)}")
            # ================================================
            
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"    └─ ✓ Written successfully\n")

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=block_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        # ===== DEBUG: Print final output info =====
        if self.debug and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"\n{'='*80}")
            print(f"[DEBUG][Pipeline.Final] Returning output")
            print(f"  ├─ output shape: {tuple(output.shape)}")
            print(f"  ├─ output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
            # Count non-zero frames
            non_zero_frames = (output.abs().sum(dim=[2,3,4]) > 1e-6).sum(dim=1)
            print(f"  ├─ Non-zero frames per batch: {non_zero_frames.tolist()}")
            print(f"  └─ Final current_start_frame: {current_start_frame}")
            print(f"{'='*80}\n")

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
