import torch.nn.functional as F
from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(args, device)

        # Step 1: Initialize all models

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        
        # ===== Add custom modules to student (same as teacher) =====
        print("[ODE] Adding custom condition modules to student generator...")
        dim = self.generator.model.dim  # 1536 for 1.3B
        param_dtype = next(self.generator.model.parameters()).dtype
        param_device = next(self.generator.model.parameters()).device
        
        # Speed token projection (1 -> dim) and scale
        self.generator.model.speed_token_proj = torch.nn.Linear(1, dim, bias=True).to(dtype=param_dtype, device=param_device)
        torch.nn.init.normal_(self.generator.model.speed_token_proj.weight, mean=0.0, std=1e-2)
        torch.nn.init.zeros_(self.generator.model.speed_token_proj.bias)
        # FSDP requires 1D tensor instead of scalar
        self.generator.model.speed_token_scale = torch.nn.Parameter(torch.tensor([1e-1], dtype=param_dtype, device=param_device))
        
        # Camera trajectory encoder (12 -> dim) and projector (dim -> dim) for each block
        for blk in self.generator.model.blocks:
            blk.cam_traj_encoder = torch.nn.Linear(12, dim, bias=True).to(dtype=param_dtype, device=param_device)
            with torch.no_grad():
                blk.cam_traj_encoder.weight.zero_()
                if blk.cam_traj_encoder.bias is not None:
                    blk.cam_traj_encoder.bias.zero_()
            
            blk.projector = torch.nn.Linear(dim, dim, bias=True).to(dtype=param_dtype, device=param_device)
            with torch.no_grad():
                blk.projector.weight.copy_(torch.eye(dim, dtype=param_dtype, device=param_device))
                blk.projector.bias.zero_()
        
        print(f"[ODE] Added custom modules: speed_token_proj, speed_token_scale, cam_traj_encoder, projector to {len(self.generator.model.blocks)} blocks")
        # ===========================================================
        
        if getattr(args, "generator_ckpt", False):
            print(f"[ODE] Loading pretrained generator from {args.generator_ckpt}")
            checkpoint = torch.load(args.generator_ckpt, map_location="cpu")
            
            # 兼容两种checkpoint格式：
            # 1. ReCamMaster teacher格式：直接是state_dict
            # 2. CausVid格式：{'generator': state_dict}
            if isinstance(checkpoint, dict) and 'generator' in checkpoint:
                state_dict = checkpoint['generator']
                print("  → Loaded from CausVid format (checkpoint['generator'])")
            else:
                state_dict = checkpoint
                print("  → Loaded from teacher format (direct state_dict)")
            
            # 添加 'model.' 前缀到teacher checkpoint的keys
            # 因为WanDiffusionWrapper.model需要这个前缀
            new_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('model.'):
                    # 添加model.前缀（现在所有teacher的模块都应该能正确匹配了）
                    new_key = f'model.{key}'
                    
                    # Special handling: teacher's speed_token_scale is scalar, but student needs 1D tensor for FSDP
                    if key == 'speed_token_scale' and value.ndim == 0:
                        value = value.unsqueeze(0)  # [] -> [1]
                        print(f"  → Reshaped speed_token_scale from scalar to 1D tensor for FSDP compatibility")
                    
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # 加载权重（非严格模式，因为causal/bidirectional架构差异）
            missing_keys, unexpected_keys = self.generator.load_state_dict(
                new_state_dict, strict=False
            )
            
            # Count custom module keys that were successfully loaded
            custom_keys_loaded = sum(1 for k in new_state_dict.keys() if any(x in k for x in ['speed_token', 'cam_traj_encoder', 'projector']))
            print(f"  → Loaded {len(new_state_dict) - len(unexpected_keys)} / {len(new_state_dict)} keys from teacher")
            print(f"  → Custom condition modules loaded: {custom_keys_loaded} keys")
            
            if missing_keys:
                print(f"  → Missing keys ({len(missing_keys)}) - expected for causal/bidirectional differences:")
                for i, key in enumerate(missing_keys[:5]):
                    print(f"      - {key}")
                if len(missing_keys) > 5:
                    print(f"      ... and {len(missing_keys) - 5} more")
            
            if unexpected_keys:
                print(f"  → Unexpected keys ({len(unexpected_keys)}):")
                for i, key in enumerate(unexpected_keys[:5]):
                    print(f"      - {key}")
                if len(unexpected_keys) > 5:
                    print(f"      ... and {len(unexpected_keys) - 5} more")

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

    def _initialize_models(self, args, device):
        """
        Override parent to skip unnecessary model loading for ODE regression.
        ODE regression only needs the generator.
        """
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        
        # Need scheduler from generator for timestep list
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        # Index the denoising_step_list on CPU, then move to device
        timestep = self.denoising_step_list[index.cpu()].to(self.device)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]

        return noisy_input, timestep

    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
        trajectory: torch.Tensor = None,  # [B, 21, 12]
        speed: torch.Tensor = None,  # [B, 1]
        input_latent: torch.Tensor = None  # NEW: [B, 21, 16, H, W] full input video latent
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
                May also contain cam_traj and speed_scalar.
            - trajectory: [B, F, 12] camera trajectory
            - speed: [B, 1] or scalar, speed condition
            - input_latent: [B, 21, 16, H, W] full input video latent (NEW)
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)
        
        # NEW: Use last 3 frames of input_latent as initial latent for autoregressive generation
        initial_latent = None
        if input_latent is not None:
            # Extract last 3 frames: [B, 21, 16, H, W] -> [B, 3, 16, H, W]
            initial_latent = input_latent[:, -3:, :, :, :]
            # Add to conditional_dict for generator
            conditional_dict = {**conditional_dict, "initial_latent": initial_latent}
        
        # Trajectory and speed are already in conditional_dict (added in trainer/ode.py)

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        return loss, log_dict
