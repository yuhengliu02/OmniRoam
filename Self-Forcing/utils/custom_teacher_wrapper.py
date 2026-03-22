import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import sys


class CustomTeacherWrapper(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        finetuned_ckpt_path: str,
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_name="Teacher"
    ):
        super().__init__()
        
        self.model_name = model_name
        print(f"[Custom{model_name}] Initializing {model_name.lower()} model...")
        print(f"[Custom{model_name}] Base model path: {base_model_path}")
        print(f"[Custom{model_name}] Finetuned checkpoint: {finetuned_ckpt_path}")
        
        omniroam_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if omniroam_root not in sys.path:
            sys.path.insert(0, omniroam_root)
        
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.utils import load_state_dict

        print(f"[Custom{model_name}] Loading base WAN2.1 1.3B model from {base_model_path}")
        
        model_config = {
            "dim": 1536,
            "in_dim": 16,
            "ffn_dim": 8960,
            "out_dim": 16,
            "text_dim": 4096,
            "freq_dim": 256,
            "eps": 1e-6,
            "patch_size": (1, 2, 2),
            "num_heads": 12,
            "num_layers": 30,
            "has_image_input": False,
        }
        
        self.model = WanModel(**model_config)

        base_ckpt_path = os.path.join(base_model_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(base_ckpt_path):
            print(f"[Custom{model_name}] Loading base weights from {base_ckpt_path}")
            base_state_dict = load_state_dict(base_ckpt_path)
            missing, unexpected = self.model.load_state_dict(base_state_dict, strict=False)
            print(f"[Custom{model_name}] Base model loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print(f"[Custom{model_name}] Warning: Base checkpoint not found at {base_ckpt_path}")
        
        self.model.eval()
        
        dim = self.model.dim  # 1536 for 1.3B
        param_dtype = next(self.model.parameters()).dtype
        param_device = next(self.model.parameters()).device
        
        print(f"[Custom{model_name}] Model dimension: {dim}")
        print(f"[Custom{model_name}] Number of blocks: {len(self.model.blocks)}")
        
        print(f"[Custom{model_name}] Adding speed_token_proj and speed_token_scale...")
        self.model.speed_token_proj = nn.Linear(1, dim, bias=True).to(dtype=param_dtype, device=param_device)
        nn.init.normal_(self.model.speed_token_proj.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.model.speed_token_proj.bias)
        self.model.speed_token_scale = nn.Parameter(torch.tensor([1e-1], dtype=param_dtype, device=param_device))
        
        print(f"[Custom{model_name}] Adding cam_traj_encoder to each block...")
        for blk in self.model.blocks:
            blk.cam_traj_encoder = nn.Linear(12, dim, bias=True).to(dtype=param_dtype, device=param_device)
            with torch.no_grad():
                blk.cam_traj_encoder.weight.zero_()
                if blk.cam_traj_encoder.bias is not None:
                    blk.cam_traj_encoder.bias.zero_()
        
        print(f"[Custom{model_name}] Adding projector to each block...")
        for blk in self.model.blocks:
            blk.projector = nn.Linear(dim, dim, bias=True).to(dtype=param_dtype, device=param_device)
            with torch.no_grad():
                blk.projector.weight.copy_(torch.eye(dim, dtype=param_dtype, device=param_device))
                blk.projector.bias.zero_()
        
        print(f"[Custom{model_name}] Loading finetuned checkpoint from {finetuned_ckpt_path}")
        if not os.path.exists(finetuned_ckpt_path):
            raise FileNotFoundError(f"Finetuned checkpoint not found: {finetuned_ckpt_path}")
        
        state_dict = torch.load(finetuned_ckpt_path, map_location="cpu", weights_only=False)
        
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        if "speed_token_scale" in state_dict and state_dict["speed_token_scale"].ndim == 0:
            print(f"[Custom{model_name}] Converting speed_token_scale from scalar to 1D tensor for FSDP compatibility")
            state_dict["speed_token_scale"] = state_dict["speed_token_scale"].unsqueeze(0)
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        print(f"[Custom{model_name}] Loaded checkpoint - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if len(missing_keys) > 0:
            print(f"[Custom{model_name}] First 10 missing keys: {missing_keys[:10]}")
        if len(unexpected_keys) > 0:
            print(f"[Custom{model_name}] First 10 unexpected keys: {unexpected_keys[:10]}")
        
        assert hasattr(self.model, "speed_token_proj"), "speed_token_proj not loaded"
        assert hasattr(self.model, "speed_token_scale"), "speed_token_scale not loaded"
        assert self.model.speed_token_scale.ndim == 1, f"speed_token_scale must be 1D for FSDP, got shape {self.model.speed_token_scale.shape}"
        assert hasattr(self.model.blocks[0], "cam_traj_encoder"), "cam_traj_encoder not loaded"
        assert hasattr(self.model.blocks[0], "projector"), "projector not loaded"
        print(f"[Custom{model_name}] ✓ All custom modules verified (speed_token_proj, speed_token_scale, cam_traj_encoder, projector)")
        
        from utils.scheduler import FlowMatchScheduler
        self.scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        
        self.model = self.model.to(dtype=torch_dtype, device=device)
        self.uniform_timestep = True
        
        self.use_gradient_checkpointing = False
        
        print(f"[Custom{model_name}] ✓ Initialization complete - Ready for use")
    
    def _convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: [B, C, H, W]
        xt: [B, C, H, W]
        timestep: [B]
        
        Formula: x0 = xt - sigma_t * flow_pred
        """
        original_dtype = flow_pred.dtype
        flow_pred = flow_pred.double()
        xt = xt.double()
        sigmas = self.scheduler.sigmas.double().to(flow_pred.device)
        timesteps = self.scheduler.timesteps.double().to(flow_pred.device)
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.double().unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)
    
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher forward pass.
        
        Args:
            noisy_image_or_video: [B, F, 16, H, W]
                - F=21: will be repeated to 42 internally
                - F=42: used as-is (already [target_21 | input_21])
            conditional_dict: contains
                - prompt_embeds: [B, 512, 4096]
                - cam_traj: [B, F_traj, 12] where F_traj matches F or 21
                - speed_scalar: [B, 1] (always 1.0)
            timestep: [B] or [B, F] - will extract first value per batch
        
        Returns:
            flow_pred: [B, F, 16, H, W]
            x0_pred: [B, F, 16, H, W]
        """
        B, F, C, H, W = noisy_image_or_video.shape
        
        # Extract conditions
        prompt_embeds = conditional_dict["prompt_embeds"]
        cam_traj = conditional_dict.get("cam_traj", None)
        speed_scalar = conditional_dict.get("speed_scalar", None)
        
        if timestep.dim() == 2:
            uniform_timestep = timestep[:, 0]  # [B]
        else:
            uniform_timestep = timestep  # Already [B]
        
        original_F = F
        if F == 21:
            noisy_image_or_video = noisy_image_or_video.repeat(1, 2, 1, 1, 1)
            if cam_traj is not None and cam_traj.shape[1] == 21:
                cam_traj = cam_traj.repeat(1, 2, 1)
            F = 42
        elif F != 42:
            raise ValueError(f"Teacher expects F=21 or F=42, got F={F}")
        
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        
        flow_pred = self.model(
            x=x,
            timestep=uniform_timestep,
            context=prompt_embeds,
            cam_traj=cam_traj,
            speed_scalar=speed_scalar,
            click_map=None,
            dir_vector=None,
            global_context_feat=None,
            traj_scale_token=None,
            use_gradient_checkpointing=self.use_gradient_checkpointing,  # Use instance flag
            use_gradient_checkpointing_offload=False
        )

        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        
        timestep_per_frame = uniform_timestep.unsqueeze(1).expand(-1, F)
        
        x0_pred = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep_per_frame.flatten(0, 1)
        ).unflatten(0, (B, F))
        
        return flow_pred, x0_pred
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency during training."""
        print(f"[Custom{self.model_name}] Enabling gradient checkpointing...")
        
        self.use_gradient_checkpointing = True
        print(f"[Custom{self.model_name}] ✓ Gradient checkpointing enabled (will use forward parameter)")
    
    def get_scheduler(self):
        """Return the flow matching scheduler"""
        return self.scheduler
    
    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

