from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import torch.nn.functional as F

class WanVideoClickMapPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoClickMapPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    def _downsample_m81_to_m21(self, m81: torch.Tensor) -> torch.Tensor:
        if m81.ndim == 1:
            m81 = m81.unsqueeze(0)
        x = m81.to(self.device, dtype=self.torch_dtype).unsqueeze(1)

        x = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
        x = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        m21 = x.squeeze(1)
        return m21


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        source_video=None,
        input_video=None,
        m81: Optional[torch.Tensor] = None,
        enable_refine: bool = False,
        enable_refine_ar: bool = False,
        use_full: bool = False,
        global_context_feat: Optional[torch.Tensor] = None,
        speed_scalar: Optional[torch.Tensor] = None,
        cam_traj_condition: Optional[torch.Tensor] = None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)

        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        self.load_models_to_device(['vae'])
        source_video = source_video.to(dtype=self.torch_dtype, device=self.device)
        source_latents = self.encode_video(source_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

        right_stack = source_latents

        if enable_refine_ar:
            right_stack = source_latents

        elif enable_refine:
            if m81 is None:
                raise ValueError("enable_refine=True must provide m81")
            m81 = m81 if torch.is_tensor(m81) else torch.as_tensor(m81)
            m21 = self._downsample_m81_to_m21(m81)
            B, C, Tlat, HH, WW = source_latents.shape
            if m21.ndim == 1:
                m21 = m21.unsqueeze(0)
            if m21.shape[0] != B:
                if m21.shape[0] == 1:
                    m21 = m21.expand(B, -1)
                else:
                    raise ValueError(f"m21 batch mismatch with source_latents: {m21.shape[0]} vs {B}")
            m21_mask = m21.to(dtype=self.torch_dtype, device=self.device).view(B, 1, Tlat, 1, 1)
            L_full = source_latents
            L_mask = L_full * m21_mask

            right_stack = L_mask if not use_full else torch.cat([L_mask, source_latents], dim=2)


        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        extra_input = self.prepare_extra_input(latents)
        tea_cache_posi = {"tea_cache": None}
        tea_cache_nega = {"tea_cache": None}

        self.load_models_to_device(["dit"])
        tgt_len = latents.shape[2]

        if speed_scalar is not None:
            if not torch.is_tensor(speed_scalar):
                speed_scalar = torch.tensor([[float(speed_scalar)]], dtype=self.torch_dtype, device=self.device)
            else:
                speed_scalar = speed_scalar.to(dtype=self.torch_dtype, device=self.device)
                if speed_scalar.ndim == 1:
                    speed_scalar = speed_scalar.unsqueeze(1)

        cam_traj = None
        if cam_traj_condition is not None:
            cam_traj = cam_traj_condition
            if not torch.is_tensor(cam_traj):
                cam_traj = torch.as_tensor(cam_traj, dtype=self.torch_dtype, device=self.device)
            else:
                cam_traj = cam_traj.to(dtype=self.torch_dtype, device=self.device)
            if cam_traj.ndim == 2:
                cam_traj = cam_traj.unsqueeze(0)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            lat_in = torch.cat([latents, right_stack], dim=2)

            global_ctx_feat_final = None
            if enable_refine and hasattr(self, 'use_global_context') and getattr(self, 'use_global_context', False):
                if global_context_feat is not None:
                    global_ctx_feat_final = global_context_feat.to(dtype=self.torch_dtype, device=self.device)
                else:
                    if source_latents.ndim == 5:
                        global_ctx_feat_final = source_latents.mean(dim=(3, 4))
                        global_ctx_feat_final = global_ctx_feat_final.transpose(1, 2)
            noise_pos = self.dit(
                lat_in, timestep=timestep, context=prompt_emb_posi["context"],
                speed_scalar=speed_scalar,
                cam_traj=cam_traj,
                global_context_feat=global_ctx_feat_final,
                **extra_input, **tea_cache_posi
            )

            if cfg_scale != 1.0:
                noise_nega = self.dit(
                    lat_in, timestep=timestep, context=prompt_emb_nega["context"],
                    speed_scalar=speed_scalar,
                    cam_traj=cam_traj,
                    global_context_feat=global_ctx_feat_final,
                    **extra_input, **tea_cache_nega
                )
                noise_pred = noise_nega + cfg_scale * (noise_pos - noise_nega)
            else:
                noise_pred = noise_pos

            latents = self.scheduler.step(
                noise_pred[:, :, :tgt_len, ...],
                self.scheduler.timesteps[progress_id],
                lat_in[:, :, :tgt_len, ...]
            )

        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        return frames