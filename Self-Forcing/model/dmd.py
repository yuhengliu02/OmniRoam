from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.distributed as dist

from model.base import SelfForcingModel

def print_gpu_memory(message, rank=0):
    """Print GPU memory usage with a custom message"""
    if dist.get_rank() == rank:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"[GPU Memory] {message}")
        print(f"  ├─ Allocated: {allocated:.2f} GB")
        print(f"  ├─ Reserved:  {reserved:.2f} GB")
        print(f"  └─ Peak:      {max_allocated:.2f} GB")


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.inference_pipeline: SelfForcingTrainingPipeline = None

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        input_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        
        use_custom_teacher = getattr(self.args, "use_custom_teacher", False)
        debug = getattr(self.args, "debug", False)
        
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG][DMD.KL_Grad] Computing KL gradient (teacher vs critic)")
            print(f"  ├─ noisy_image_or_video shape: {tuple(noisy_image_or_video.shape)}")
            print(f"  ├─ use_custom_teacher: {use_custom_teacher}")
            if input_latent is not None:
                print(f"  └─ input_latent shape: {tuple(input_latent.shape)}")
        
        if use_custom_teacher and input_latent is not None:
            B, F, C, H, W = noisy_image_or_video.shape
            full_input = torch.cat([noisy_image_or_video, input_latent], dim=1)
            full_timestep = timestep.repeat(1, 2)
            
            debug = getattr(self.args, "debug", False)
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"\n[DEBUG][DMD.KL_Grad] Building 42-frame input for teacher/critic")
                print(f"  ├─ Concatenation: [noisy_target_21 | clean_input_21]")
                print(f"  ├─ noisy_target shape: {tuple(noisy_image_or_video.shape)}")
                print(f"  ├─ clean_input shape: {tuple(input_latent.shape)}")
                print(f"  ├─ full_input shape: {tuple(full_input.shape)}")
                print(f"  └─ full_timestep shape: {tuple(full_timestep.shape)}")
            
            teacher_input = full_input
            teacher_timestep = full_timestep
            critic_input = full_input
            critic_timestep = full_timestep
        else:
            teacher_input = noisy_image_or_video
            teacher_timestep = timestep
            critic_input = noisy_image_or_video
            critic_timestep = timestep
        
        debug = getattr(self.args, "debug", False)
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n[DEBUG][DMD.KL_Grad] Critic (fake_score) forward")
            print(f"  ├─ Input shape: {tuple(critic_input.shape)}")
            print(f"  ├─ Timestep shape: {tuple(critic_timestep.shape)}")
            if 'cam_traj' in conditional_dict:
                traj = conditional_dict['cam_traj']
                print(f"  └─ cam_traj shape: {tuple(traj.shape)}")
        
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=critic_input,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )
        
        print_gpu_memory(f"        After critic conditional forward", rank=0)

        if self.fake_guidance_scale != 0.0:
            debug = getattr(self.args, "debug", False)
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  ├─ Running critic unconditional forward (guidance_scale={self.fake_guidance_scale})")
            
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=critic_input,
                conditional_dict=unconditional_dict,
                timestep=critic_timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
            
            print_gpu_memory(f"        After critic unconditional forward", rank=0)
        else:
            pred_fake_image = pred_fake_image_cond
        
        debug = getattr(self.args, "debug", False)
        if use_custom_teacher and input_latent is not None:
            pred_fake_image = pred_fake_image[:, :21]
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  └─ Extracted target prediction (first 21 frames): {tuple(pred_fake_image.shape)}")
        
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n[DEBUG][DMD.KL_Grad] Teacher (real_score) forward")
            print(f"  ├─ Input shape: {tuple(teacher_input.shape)}")
            print(f"  └─ Guidance scale: {self.real_guidance_scale}")

        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=teacher_input,
            conditional_dict=conditional_dict,
            timestep=teacher_timestep
        )
        
        print_gpu_memory(f"        After teacher conditional forward", rank=0)
        
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=teacher_input,
            conditional_dict=unconditional_dict,
            timestep=teacher_timestep
        )
        
        print_gpu_memory(f"        After teacher unconditional forward", rank=0)

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale
        
        debug = getattr(self.args, "debug", False)
        if use_custom_teacher and input_latent is not None:
            pred_real_image = pred_real_image[:, :21]
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  └─ Extracted target prediction (first 21 frames): {tuple(pred_real_image.shape)}")

        grad = (pred_fake_image - pred_real_image)

        if normalization:
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n[DEBUG][DMD.KL_Grad] Gradient computation")
            print(f"  ├─ grad = pred_fake - pred_real")
            print(f"  ├─ Grad norm (raw): {torch.mean(torch.abs(grad)).item():.6f}")
            print(f"  ├─ Normalization: {normalization}")
            if normalization:
                print(f"  ├─ Normalizer mean: {normalizer.mean().item():.6f}")
            print(f"  └─ Final grad norm: {torch.mean(torch.abs(grad)).item():.6f}")
            print(f"{'='*80}\n")

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        input_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
            - input_latent: [B, 21, 16, H, W] static video latent for custom teacher (NEW)
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                input_latent=input_latent
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        input_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: first frame latent for i2v
            - input_latent: [B, 21, 16, H, W] static video latent for custom teacher (NEW)
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if input_latent is not None and initial_latent is None:
            initial_latent = input_latent[:, -3:, :, :, :]
            
            debug = getattr(self.args, "debug", False)
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"\n{'='*80}")
                print(f"[DEBUG][Generator.Loss] Step 1: Convert input_latent to initial_latent")
                print(f"  ├─ input_latent shape: {tuple(input_latent.shape)}")
                print(f"  ├─ initial_latent shape (last 3 frames): {tuple(initial_latent.shape)}")
                print(f"  ├─ initial_latent stats: mean={initial_latent.mean().item():.4f}, std={initial_latent.std().item():.4f}")
                print(f"  └─ ✓ Conversion successful - will be used for KV cache pre-fill")
        
        debug = getattr(self.args, "debug", False)
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n[DEBUG][Generator.Loss] Step 2: Calling _run_generator")
            print(f"  ├─ image_or_video_shape: {image_or_video_shape}")
            print(f"  ├─ initial_latent: {tuple(initial_latent.shape) if initial_latent is not None else None}")
            print(f"  ├─ conditional_dict keys: {list(conditional_dict.keys())}")
            if 'cam_traj' in conditional_dict:
                traj = conditional_dict['cam_traj']
                print(f"  ├─ cam_traj shape: {tuple(traj.shape)}")
                M = traj.view(traj.shape[0], traj.shape[1], 3, 4)
                t = M[:, :, :, 3]
                dt = t[:, 1:] - t[:, :-1]
                step_sizes = torch.linalg.norm(dt, dim=2)
                print(f"  ├─ trajectory translation norms: min={t.norm(dim=2).min().item():.4f}, max={t.norm(dim=2).max().item():.4f}")
                print(f"  ├─ trajectory step sizes: min={step_sizes.min().item():.4f}, max={step_sizes.max().item():.4f}, median={step_sizes.median().item():.4f}")
            if 'speed_scalar' in conditional_dict:
                print(f"  ├─ speed_scalar: {conditional_dict['speed_scalar'].item():.2f}")
            print(f"  └─ Entering backward simulation...")
        
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )
        
        debug = getattr(self.args, "debug", False)
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n[DEBUG][Generator.Loss] Step 3: Generation complete")
            print(f"  ├─ pred_image shape: {tuple(pred_image.shape)}")
            print(f"  ├─ pred_image stats: mean={pred_image.mean().item():.4f}, std={pred_image.std().item():.4f}")
            print(f"  ├─ gradient_mask: {'present' if gradient_mask is not None else 'None'}")
            print(f"  ├─ denoised_timestep_from: {denoised_timestep_from}")
            print(f"  └─ denoised_timestep_to: {denoised_timestep_to}")

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            input_latent=input_latent
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        input_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: first frame latent for i2v
            - input_latent: [B, 21, 16, H, W] static video latent for custom teacher (NEW)
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        
        if input_latent is not None and initial_latent is None:
            initial_latent = input_latent[:, -3:, :, :, :]
            
            debug = getattr(self.args, "debug", False)
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"\n{'='*80}")
                print(f"[DEBUG][Critic.Loss] Step 1: Convert input_latent to initial_latent")
                print(f"  ├─ input_latent shape: {tuple(input_latent.shape)}")
                print(f"  └─ initial_latent shape (last 3 frames): {tuple(initial_latent.shape)}")
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )
        
        debug = getattr(self.args, "debug", False)
        if debug and dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n[DEBUG][Critic.Loss] Step 2: Generator produced fake samples")
            print(f"  └─ generated_image shape: {tuple(generated_image.shape)}")

        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        use_custom_teacher = getattr(self.args, "use_custom_teacher", False)
        
        debug = getattr(self.args, "debug", False)
        if use_custom_teacher and input_latent is not None:
            B, F, C, H, W = noisy_generated_image.shape
            assert F == 21, f"Expected 21 frames, got {F}"
            assert input_latent.shape[1] == 21, f"Expected 21 input frames, got {input_latent.shape[1]}"
            
            critic_input = torch.cat([noisy_generated_image, input_latent], dim=1)
            critic_input_timestep = critic_timestep.repeat(1, 2)
            
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"\n[DEBUG][Critic.Loss] Step 3: Building 42-frame input for critic")
                print(f"  ├─ Concatenation: [noisy_generated_21 | clean_input_21]")
                print(f"  ├─ critic_input shape: {tuple(critic_input.shape)}")
                print(f"  └─ critic_input_timestep shape: {tuple(critic_input_timestep.shape)}")
        else:
            critic_input = noisy_generated_image
            critic_input_timestep = critic_timestep

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=critic_input,
            conditional_dict=conditional_dict,
            timestep=critic_input_timestep
        )
        
        debug = getattr(self.args, "debug", False)
        if use_custom_teacher and input_latent is not None:
            pred_fake_image = pred_fake_image[:, :21]
            
            if debug and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  └─ Extracted target prediction: {tuple(pred_fake_image.shape)}")

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
