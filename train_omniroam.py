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

import os
import re
import json
import argparse
import random
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.utils.data
from einops import rearrange
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch.nn.functional as F

from diffsynth import ModelManager, WanVideoClickMapPipeline

from concurrent.futures import ThreadPoolExecutor, as_completed

from glob import glob

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor
import wandb

import math

def pad4(n: int) -> str:
    return f"{n:04d}"

class OnlineFramesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_test_split_json: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        steps_per_epoch: int = 0,
        batch_fetch: int = 16,
        debug: bool = False,

        enable_speed_control: bool = False,
        speed_min: float = 1.0,
        speed_max: float = 8.0,
        speed_fixed: float = None,
        speed_neutral_log_halfwidth: float = 0.5,
        
        speed_two_bucket: bool = True,
        speed_bucket_prob_one: float = 0.5,
        speed_bucket_min_fast: float = 1.1,
        speed_bucket_max_fast: float = 8.0,
        round_speed_one_decimal: bool = True,

        cam_traj_condition: bool = False,
        re_scale_pose_raw: str = "none",

        traj_filter_enable: bool = True,
        traj_tnorm_mean_max: float = 50.0,
        traj_tnorm_median_max: float = 50.0,
        traj_tnorm_any_max: float = 120.0,
        traj_require_finite: bool = True,
        
        interiorgs_data_root: str = "data/InteriorGS-360video",
        interiorgs_frames_subdir: str = "pano_camera0",
        interiorgs_max_frames: int = 800,
        interiorgs_frame_ext: str = "png",
        
        refine_mode: bool = False,
        refine_speed_min: float = 1.1,
        refine_speed_max: float = 8.0,
        refine_window_policy: str = "random",
        refine_degrade_down_h: int = 240,
        refine_degrade_down_w: int = 480,
        refine_use_global_context: bool = False,
    ):
        with open(train_test_split_json, "r") as f:
            split = json.load(f)
        
        train_data = split["train"]
        if isinstance(train_data, list):
            self.video_ids = train_data
        else:
            raise RuntimeError(f"Expected 'train' to be a list of video IDs, got {type(train_data)}")
        
        if len(self.video_ids) == 0:
            raise RuntimeError(f"No videos found in {train_test_split_json}")
        
        self.interiorgs_data_root = os.path.abspath(interiorgs_data_root)
        self.interiorgs_frames_subdir = interiorgs_frames_subdir.strip("/")
        self.interiorgs_max_frames = int(interiorgs_max_frames)
        self.interiorgs_frame_ext = str(interiorgs_frame_ext).lower().lstrip(".")
        
        self._ns_colmap_cache: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.batch_fetch = max(1, int(batch_fetch))
        self.debug = debug

        self.steps_per_epoch = steps_per_epoch if steps_per_epoch > 0 else 1_000_000

        self.enable_speed_control = bool(enable_speed_control)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.speed_fixed = (None if speed_fixed is None else float(speed_fixed))
        self.speed_neutral_log_halfwidth = float(speed_neutral_log_halfwidth)
        
        self.speed_two_bucket = bool(speed_two_bucket)
        self.speed_bucket_prob_one = float(speed_bucket_prob_one)
        self.speed_bucket_min_fast = float(speed_bucket_min_fast)
        self.speed_bucket_max_fast = float(speed_bucket_max_fast)
        self.round_speed_one_decimal = bool(round_speed_one_decimal)

        self.cam_traj_condition = bool(cam_traj_condition)

        self._re_scale_mode, self._re_scale_target = self._parse_re_scale_pose(re_scale_pose_raw)

        self.traj_filter_enable = bool(traj_filter_enable)
        self.traj_tnorm_mean_max = float(traj_tnorm_mean_max)
        self.traj_tnorm_median_max = float(traj_tnorm_median_max)
        self.traj_tnorm_any_max = float(traj_tnorm_any_max)
        self.traj_require_finite = bool(traj_require_finite)
        self._traj_drop_count = 0

        self.refine_mode = bool(refine_mode)
        self.refine_speed_min = float(refine_speed_min)
        self.refine_speed_max = float(refine_speed_max)
        self.refine_window_policy = str(refine_window_policy)
        self.refine_degrade_down_h = int(refine_degrade_down_h)
        self.refine_degrade_down_w = int(refine_degrade_down_w)
        self.refine_use_global_context = bool(refine_use_global_context)


    def _get_transforms_json_path(self, video_id: str) -> str:
        return os.path.join(self.interiorgs_data_root, video_id, "transforms.json")
    
    def _get_interiorgs_start_range(self, speed_s: float) -> tuple:
        max_frames = self.interiorgs_max_frames
        if 4.0 <= speed_s < 5.5:
            return (1, 400)
        elif 5.5 <= speed_s < 6.5:
            return (1, 200)
        elif 6.5 <= speed_s <= 8.0:
            return (1, 100)
        else:
            return (1, 710)
    
    def _ns_transform_to_internal_c2w(self, M_ns: np.ndarray) -> np.ndarray:
        M_ns = np.asarray(M_ns, dtype=np.float64)
        Rwc = M_ns[:3, :3]
        Cw  = M_ns[:3, 3]

        fwd =  Rwc[:, 2]
        up  = -Rwc[:, 1]
        rgt =  Rwc[:, 0]

        R_int = np.stack([fwd, up, rgt], axis=1)

        M_int = np.eye(4, dtype=np.float64)
        M_int[:3, :3] = R_int
        M_int[:3, 3]  = Cw
        return M_int
    

    def _build_cam_traj_21_identityR(
        self,
        ns_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
        tg_start_idx: int,
        pick_81: np.ndarray,
        L_real: int,
    ) -> torch.Tensor:
        if pick_81 is not None:
            C81 = []
            for i in range(81):
                idx = int(tg_start_idx + int(pick_81[i]))
                if idx not in ns_map:
                    raise KeyError(f"[cam_traj] missing pose for target idx={idx}")
                _R_wc, C_w = ns_map[idx]
                C81.append(C_w)
            C81 = np.stack(C81, axis=0)
            ref_idx = int(tg_start_idx + int(pick_81[0]))
            R_ref, C_ref = ns_map[ref_idx]
        else:
            Cres = []
            for j in range(L_real):
                idx = int(tg_start_idx + j)
                if idx not in ns_map:
                    raise KeyError(f"[cam_traj] missing pose for target idx={idx} (interp)")
                _R_wc, C_w = ns_map[idx]
                Cres.append(C_w)
            Cres = np.stack(Cres, axis=0)
            t_real = np.linspace(0, L_real - 1, L_real, dtype=np.float64)
            t_81   = np.linspace(0, L_real - 1, 81,     dtype=np.float64)
            C81 = np.stack([np.interp(t_81, t_real, Cres[:,a]) for a in range(3)], axis=1)
            R_ref, C_ref = ns_map[tg_start_idx]

        R_ref_T = R_ref.T
        traj = []
        for k in range(21):
            j81 = 4 * k
            t_w = C81[j81] - C_ref
            t_ref = (R_ref_T @ t_w).astype(np.float64)
            M = np.concatenate([np.eye(3, dtype=np.float64), t_ref.reshape(3,1)], axis=1)
            traj.append(M.reshape(-1))
        traj = np.stack(traj, axis=0).astype(np.float32)
        return torch.from_numpy(traj)

    def _get_ns_colmap_map(self, video_id: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        cache_key = f"interiorgs_{video_id}"
        if cache_key in self._ns_colmap_cache:
            return self._ns_colmap_cache[cache_key]

        json_path = self._get_transforms_json_path(video_id)
        
        try:
            with open(json_path, "r") as f:
                j = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"transforms.json not found for video {video_id}: {json_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON for video {video_id} at {json_path}: {e}")
        
        per_image = j.get("per_image", {})
        
        mp: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        idx_re = re.compile(r"frame[_\-]?0*([0-9]+)")
        
        for img_name, info in per_image.items():
            if not img_name.startswith("pano_camera0/"):
                continue
            m_idx = idx_re.search(img_name)
            if not m_idx:
                continue
            idx = int(m_idx.group(1))
            
            M_ns = info.get("transform_matrix", None)
            if M_ns is None:
                continue
            M_ns = np.array(M_ns, dtype=np.float64)
            if M_ns.shape != (4, 4):
                continue
            

            C_w_raw = M_ns[:3, 3]
            
            C_w_fixed = np.array([
                C_w_raw[0],
                C_w_raw[1],
                C_w_raw[2]
            ], dtype=np.float64)
            
            R_fixed = np.array([
                [0, 0, 1],
                [0, -1, 0],
                [1, 0, 0]
            ], dtype=np.float64)
            
            M_ns_fixed = np.eye(4, dtype=np.float64)
            M_ns_fixed[:3, :3] = R_fixed
            M_ns_fixed[:3, 3] = C_w_fixed
            
            M_int = self._ns_transform_to_internal_c2w(M_ns_fixed)
            R_wc = M_int[:3, :3]
            C_w = M_int[:3, 3]
            
            mp[idx] = (R_wc, C_w)
        
        if not mp:
            raise RuntimeError(f"No pano_camera0 frames found for InteriorGS {video_id}: {json_path}")
        
        self._ns_colmap_cache[cache_key] = mp
        return mp
    

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return v * 0.0
        return v / n

    def _sample_speed(self) -> Tuple[float, float]:
        if not self.enable_speed_control:
            return 1.0, 0.0
        
        if self.speed_two_bucket:
            if np.random.rand() < self.speed_bucket_prob_one:
                s = 1.0
            else:
                low = max(1.0, float(self.speed_bucket_min_fast))
                high = max(low, float(self.speed_bucket_max_fast))
                s = float(np.random.uniform(low=low, high=high))
                if self.round_speed_one_decimal:
                    s = float(np.round(s, 1))
                s = min(max(s, 1.0), high)
            z = float(math.log2(s))
            return s, z
        else:
            zmin, zmax = math.log2(self.speed_min), math.log2(self.speed_max)
            if self.speed_fixed is not None:
                s = float(np.clip(self.speed_fixed, self.speed_min, self.speed_max))
                z = float(math.log2(s))
            else:
                z = float(np.random.uniform(low=zmin, high=zmax))
                s = float(2.0 ** z)

            h = float(self.speed_neutral_log_halfwidth)
            if h > 0.0 and (-h <= z <= h):
                s, z = 1.0, 0.0
            
            return s, z
    

    def _interp_to_len(self, vid_CTHW: torch.Tensor, T_out: int) -> torch.Tensor:
        assert vid_CTHW.ndim == 4
        C, T, H, W = vid_CTHW.shape
        x = vid_CTHW.unsqueeze(0)
        x = F.interpolate(x, size=(T_out, H, W), mode="trilinear", align_corners=True)
        return x[0]
    

    def _uniform_pick_81_from_long(self, L_real: int) -> np.ndarray:
        assert L_real >= 81

        idx = np.floor(np.linspace(0, L_real - 1, 81, dtype=np.float64)).astype(np.int64)
        idx[0] = 0
        idx[-1] = L_real - 1

        if not np.all(idx[1:] > idx[:-1]):
            i = np.arange(81, dtype=np.int64)
            idx = (i * (L_real - 1)) // 80
            idx[0] = 0
            idx[-1] = L_real - 1

        return idx
    
    def _parse_re_scale_pose(self, s: str):
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

    def _rescale_cam_traj_identityR(self, cam_traj_21: torch.Tensor, mode: str, s_target: float):
        if cam_traj_21 is None or mode == "none":
            return cam_traj_21, None, 1.0
        M = cam_traj_21.view(-1, 3, 4)
        t = M[:, :, 3]
        if t.shape[0] < 2:
            return cam_traj_21, None, 1.0
        dt = t[1:] - t[:-1]
        step = torch.linalg.norm(dt, dim=1)
        if step.numel() == 0:
            return cam_traj_21, None, 1.0
        s_local = torch.median(step).item()
        eps = 1e-8
        if not np.isfinite(s_local) or s_local < eps:
            return cam_traj_21, max(s_local, 0.0), 1.0

        if mode == "unit_median":
            s_tgt = 1.0
        elif mode == "fixed":
            s_tgt = float(s_target)
        else:
            return cam_traj_21, s_local, 1.0

        alpha = float(s_tgt / s_local)
        t_scaled = t * alpha
        M[:, :, 3] = t_scaled
        return M.reshape(-1, 12), s_local, alpha
    

    def _cam_traj_stats(self, cam_traj_21: torch.Tensor):
        if cam_traj_21 is None:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "finite_ok": True}

        ct = cam_traj_21
        if isinstance(ct, torch.Tensor) and ct.ndim == 2:
            ct = ct.unsqueeze(0)
        if not isinstance(ct, torch.Tensor):
            return {"mean": float("inf"), "median": float("inf"), "max": float("inf"), "finite_ok": False}

        ct = ct.detach()
        finite_ok = torch.isfinite(ct).all().item()
        M = ct.view(ct.shape[0], -1, 3, 4)
        t = M[:, :, :, 3]
        t_norm = torch.linalg.norm(t.to(torch.float32), dim=2)
        mean_v = t_norm.mean(dim=1).mean().item()
        median_v = t_norm.median(dim=1).values.mean().item()
        max_v = t_norm.max().item()
        return {"mean": float(mean_v), "median": float(median_v), "max": float(max_v), "finite_ok": bool(finite_ok)}

    def _traj_pass(self, stats: dict) -> bool:
        if self.traj_require_finite and not stats.get("finite_ok", True):
            return False
        if stats["mean"] > self.traj_tnorm_mean_max:
            return False
        if stats["median"] > self.traj_tnorm_median_max:
            return False
        if stats["max"] > self.traj_tnorm_any_max:
            return False
        return True
    
    def _make_refine_window(self, s: float) -> Tuple[int, int]:
        k = int(math.ceil(81.0 / max(s, 1e-6)))
        k = max(1, min(81, k))
        if self.refine_window_policy == "center":
            j0 = max(0, (81 - k) // 2)
        else:
            j0 = int(np.random.randint(0, 81 - k + 1))
        return k, j0

    def _degrade_blur_alias(self, vid_CTHW: torch.Tensor,
                              down_h: int, down_w: int,
                              up_h: int, up_w: int) -> torch.Tensor:
        assert vid_CTHW.ndim == 4, f"expected (C,T,H,W), got {tuple(vid_CTHW.shape)}"
        C, T, H, W = vid_CTHW.shape

        x = (vid_CTHW * 0.5 + 0.5).clamp(0, 1)
        x = x.permute(1, 0, 2, 3).contiguous()

        x = torch.nn.functional.interpolate(
            x, size=(int(down_h), int(down_w)), mode="area"
        )
        x = torch.nn.functional.interpolate(
            x, size=(int(up_h), int(up_w)), mode="bicubic", align_corners=False
        )

        x = x.permute(1, 0, 2, 3).contiguous()
        return (x - 0.5) / 0.5



    def __len__(self):
        return self.steps_per_epoch

    def _get_frame_path(self, video_id: str, idx: int) -> str:
        return os.path.join(
            self.interiorgs_data_root, 
            video_id, 
            self.interiorgs_frames_subdir, 
            f"frame_{pad4(idx)}.{self.interiorgs_frame_ext}"
        )

    def _load_frames_tensor(self, video_id: str, start: int, end: int):
        total = end - start + 1
        keys = list(range(start, end + 1))
        paths = [self._get_frame_path(video_id, k) for k in keys]

        t0 = time.time()
        max_workers = max(4, self.batch_fetch)
        imgs = [None] * total

        def fetch_one(i_path):
            i, path = i_path
            from PIL import Image
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return i, img

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_one, (i, path)) for i, path in enumerate(paths)]
            for fut in as_completed(futures):
                i, img = fut.result()
                imgs[i] = img

        t_transform_start = time.time()
        
        tensor_list = []
        for img in imgs:
            img_array = np.array(img, dtype=np.uint8)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            tensor_list.append(img_tensor)
        
        batch_imgs = torch.stack(tensor_list, dim=0)
        
        batch_imgs = self._batch_resize_with_padding(batch_imgs, self.height, self.width)
        
        batch_imgs = (batch_imgs - 0.5) / 0.5
        
        t_transform_end = time.time()
        t_elapsed = time.time() - t0

        vid = rearrange(batch_imgs, "T C H W -> C T H W")
        
        if self.debug:
            print(f"[DEBUG] Transform time: {t_transform_end - t_transform_start:.3f}s")
        
        return vid, paths, t_elapsed

    def _load_frames_by_indices(self, video_id: str, indices: List[int]):
        if len(indices) == 0:
            raise ValueError("indices is empty")

        paths = [self._get_frame_path(video_id, int(k)) for k in indices]
        t0 = time.time()
        max_workers = max(4, self.batch_fetch)
        imgs = [None] * len(indices)

        def fetch_one(i_path):
            i, path = i_path
            from PIL import Image
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return i, img

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_one, (i, path)) for i, path in enumerate(paths)]
            for fut in as_completed(futures):
                i, img = fut.result()
                imgs[i] = img

        tensor_list = []
        for img in imgs:
            arr = np.array(img, dtype=np.uint8)
            t = torch.from_numpy(arr).float() / 255.0
            t = t.permute(2, 0, 1)
            tensor_list.append(t)
        batch_imgs = torch.stack(tensor_list, dim=0)
        batch_imgs = self._batch_resize_with_padding(batch_imgs, self.height, self.width)
        batch_imgs = (batch_imgs - 0.5) / 0.5
        vid = rearrange(batch_imgs, "T C H W -> C T H W")

        t_elapsed = time.time() - t0
        return vid, paths, t_elapsed


    def _check_ns_poses_exist(self, ns_map: Dict[int, Tuple[np.ndarray, np.ndarray]], indices: List[int]) -> bool:
        for idx in indices:
            if int(idx) not in ns_map:
                return False
        return True
    
    def _batch_resize_with_padding(self, batch_imgs, target_height, target_width):
        assert batch_imgs.ndim == 4
        B, C, H, W = batch_imgs.shape

        tgt_long  = max(target_height, target_width)
        tgt_short = min(target_height, target_width)
        src_long  = max(H, W)
        src_short = min(H, W)

        scale = tgt_long / float(src_long)
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))

        if new_h != H or new_w != W:
            batch_imgs = F.interpolate(
                batch_imgs, size=(new_h, new_w),
                mode="bilinear", align_corners=False, antialias=True
            )

        if new_h < target_height:
            pad_total = target_height - new_h
            pad_top   = pad_total // 2
            pad_bot   = pad_total - pad_top
            batch_imgs = F.pad(batch_imgs, (0, 0, pad_top, pad_bot), value=0.0)
            new_h = target_height
        elif new_h > target_height:
            top = (new_h - target_height) // 2
            batch_imgs = batch_imgs[:, :, top:top + target_height, :]
            new_h = target_height

        if new_w < target_width:
            pad_total = target_width - new_w
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            batch_imgs = F.pad(batch_imgs, (pad_left, pad_right, 0, 0), value=0.0)
            new_w = target_width
        elif new_w > target_width:
            left = (new_w - target_width) // 2
            batch_imgs = batch_imgs[:, :, :, left:left + target_width]
            new_w = target_width

        return batch_imgs


    def _load_caption(self, video_id: str):
        return "A panoramic video.", "", 0.0

    def __getitem__(self, idx):
        while True:
            if self.refine_mode:
                vid = random.choice(self.video_ids)
                
                lo = int(math.ceil(self.refine_speed_min * 10))
                hi = int(math.floor(self.refine_speed_max * 10))
                hi = max(hi, lo)
                s = float(np.random.randint(lo, hi + 1)) / 10.0

                L_real = 1 + int(math.ceil(80.0 * s))
                
                start_min, start_max = self._get_interiorgs_start_range(s)
                valid_start_max = min(start_max, self.interiorgs_max_frames - L_real + 1)
                if valid_start_max < start_min:
                    continue
                
                tg_start = int(np.random.randint(start_min, valid_start_max + 1))
                tg_end_needed = tg_start + L_real - 1
                
                try:
                    pick = self._uniform_pick_81_from_long(L_real)
                    abs_81 = [tg_start + int(i) for i in pick.tolist()]

                    acc_81_video, acc_81_urls, _ = self._load_frames_by_indices(vid, abs_81)

                    acc_81_video = self._degrade_blur_alias(
                        vid_CTHW=acc_81_video,
                        down_h=self.refine_degrade_down_h, 
                        down_w=self.refine_degrade_down_w,
                        up_h=self.height, 
                        up_w=self.width
                    )

                    k, j0 = self._make_refine_window(s)
                    j1 = j0 + k - 1

                    r0 = int(pick[j0])
                    r1 = int(pick[j1])

                    sub_abs = list(range(tg_start + r0, tg_start + r1 + 1))
                    sub_1x, sub_urls, _ = self._load_frames_by_indices(vid, sub_abs)

                    target_81_video = self._interp_to_len(sub_1x, 81)

                    m81 = torch.zeros(81, dtype=torch.float32)
                    m81[j0:j1+1] = 1.0

                    sample = {
                        "video_id": vid,
                        "input_video": acc_81_video,
                        "target_video": target_81_video,
                        "text": "A panoramic video.",
                        "m81": m81,
                        "debug_info": {
                            "video_id": vid,
                            "dataset_source": "interiorgs",
                            "refine_speed": float(s),
                            "refine_window_k": int(k),
                            "refine_window_j0": int(j0),
                            "refine_r0_r1": (int(r0), int(r1)),
                            "target_range": (int(tg_start + r0), int(tg_start + r1)),
                            "target_window_urls": sub_urls,
                            "pick_abs_indices_81": abs_81,
                        },
                    }
                    
                    return sample
                
                except Exception as e:
                    continue
            
            vid = random.choice(self.video_ids)
            
            s, z = self._sample_speed()
            
            start_min, start_max = self._get_interiorgs_start_range(s)
            
            max_frames = self.interiorgs_max_frames
            in_start = random.randint(start_min, min(start_max, max_frames - 81 - 1))
            in_end = in_start + 80
            
            tg_start = in_end + 1
            
            if s >= 1.0:
                L_real = 1 + int(math.ceil(80.0 * s))
            else:
                L_real = 1 + max(1, int(math.floor(80.0 * s)))
            
            tg_end_needed = tg_start + L_real - 1
            
            if tg_end_needed > max_frames:
                continue

            try:
                input_video, input_urls, t_in = self._load_frames_tensor(vid, in_start, in_end)
                eff_in_range = (in_start, in_end)

                ns_map = self._get_ns_colmap_map(vid)

                in_end_idx = in_end
                if in_end_idx not in ns_map:
                    raise RuntimeError(f"missing pose for input last frame idx={in_end_idx} of video {vid}")
                R_wc_end, C_end = ns_map[in_end_idx]

                if s >= 1.0:
                    pick = self._uniform_pick_81_from_long(L_real)
                    abs_indices_81 = [tg_start + int(i) for i in pick.tolist()]
                    traj_pose_indices = abs_indices_81
                else:
                    traj_pose_indices = list(range(tg_start, tg_end_needed + 1))

                required_pose_indices = set([in_end_idx])
                if self.cam_traj_condition:
                    required_pose_indices.update(traj_pose_indices)
                

                if not self._check_ns_poses_exist(ns_map, list(required_pose_indices)):
                    raise RuntimeError(
                        f"[Speed] missing poses for indices: "
                        f"{sorted(list(required_pose_indices))[:5]}... (total {len(required_pose_indices)})"
                    )

                if s >= 1.0:
                    target_video, target_urls, t_tg = self._load_frames_by_indices(vid, abs_indices_81)
                    eff_tg_range = (tg_start, tg_end_needed)
                else:
                    long_target_video, long_target_urls, t_tg_long = self._load_frames_tensor(vid, tg_start, tg_end_needed)

                    target_video = self._interp_to_len(long_target_video, 81)
                    target_urls = long_target_urls
                    t_tg = t_tg_long
                    eff_tg_range = (tg_start, tg_end_needed)

                text, caption_url, t_cap = self._load_caption(vid)

                cam_traj_21 = None
                traj_scale_scalar = None

                sample = {
                    "video_id": vid,
                    "input_video": input_video,
                    "target_video": target_video,
                    "text": text,
                    "cam_traj": None,
                    "debug_info": {
                        "video_id": vid,
                        "input_range": (in_start, in_end),
                        "target_range": eff_tg_range,
                        "target_indices": (abs_indices_81 if s >= 1.0 else list(range(tg_start, tg_end_needed + 1))),
                        "input_paths": input_urls,
                        "target_paths": target_urls,
                        "input_load_time_s": t_in,
                        "target_load_time_s": t_tg,
                        "caption_url": caption_url,
                        "caption_time_s": t_cap,
                        "caption_text": text,
                        "colmap_transforms_path": self._get_transforms_json_path(vid),
                        "speed": float(s),
                        "speed_log2": float(z),
                        "real_len_for_target": int(L_real),
                    },
                }

                if s >= 1.0:
                    cam_traj_21 = self._build_cam_traj_21_identityR(
                        ns_map=ns_map,
                        tg_start_idx=eff_tg_range[0],
                        pick_81=np.asarray(pick, dtype=np.int64),
                        L_real=L_real,
                    )
                else:
                    cam_traj_21 = self._build_cam_traj_21_identityR(
                        ns_map=ns_map,
                        tg_start_idx=eff_tg_range[0],
                        pick_81=None,
                        L_real=L_real,
                    )

                if cam_traj_21 is not None:
                    sample["cam_traj"] = cam_traj_21

                if (cam_traj_21 is not None) and (self._re_scale_mode != "none"):
                    ct = sample.get("cam_traj", None)
                    if isinstance(ct, torch.Tensor):
                        ct_scaled, s_local, alpha = self._rescale_cam_traj_identityR(
                            cam_traj_21=ct,
                            mode=self._re_scale_mode,
                            s_target=(self._re_scale_target if self._re_scale_target is not None else 1.0),
                        )
                        sample["cam_traj"] = ct_scaled
                        dbg = sample.get("debug_info", {})
                        dbg["traj_rescale_mode"] = self._re_scale_mode
                        dbg["traj_rescale_s_target"] = (self._re_scale_target if self._re_scale_target is not None else 1.0)
                        dbg["traj_rescale_s_local"] = float(s_local) if s_local is not None else None
                        dbg["traj_rescale_alpha"] = float(alpha)
                        sample["debug_info"] = dbg

                if self.traj_filter_enable and (sample.get("cam_traj", None) is not None):
                    stats = self._cam_traj_stats(sample["cam_traj"])
                    dbg = sample.get("debug_info", {})
                    dbg["cam_traj_t_norm_mean"] = stats["mean"]
                    dbg["cam_traj_t_norm_median"] = stats["median"]
                    dbg["cam_traj_t_norm_max"] = stats["max"]
                    dbg["cam_traj_finite_ok"] = stats["finite_ok"]
                    sample["debug_info"] = dbg
                    if not self._traj_pass(stats):
                        self._traj_drop_count += 1
                        if self.debug:
                            print(f"[TRJ-FILTER] DROP: mean={stats['mean']:.2f}, med={stats['median']:.2f}, "
                                f"max={stats['max']:.2f}, finite={stats['finite_ok']} | video={sample.get('video_id','?')}")
                        continue

                return sample


            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Failed to load data for video {vid} ({in_start}_{in_end} -> {tg_start}_{tg_end_needed}): {e}")
                continue


class LightningModelForTrainOnline(pl.LightningModule):
    def __init__(
        self,
        dit_path: str,
        text_encoder_path: str,
        vae_path: str,
        learning_rate: float = 1e-5,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        resume_ckpt_path: str = None,
        tiled: bool = False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        output_path: str = "./",
        args=None,
    ):
        super().__init__()
        self.args = args
        self.output_path = output_path

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_paths = [text_encoder_path, vae_path]
        if os.path.isfile(dit_path):
            model_paths.append(dit_path)
        else:
            model_paths += dit_path.split(",")
        model_manager.load_models(model_paths)

        self.pipe = WanVideoClickMapPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)


        dit = self.pipe.dit
        dim = dit.blocks[0].self_attn.q.weight.shape[0]
        param_dtype = dit.patch_embedding.weight.dtype
        param_device = dit.patch_embedding.weight.device

        self.use_speed_token = bool(getattr(self.args, "enable_speed_control", False))
        if self.use_speed_token and not hasattr(dit, "speed_token_proj"):
            dit.speed_token_proj = nn.Linear(1, dim, bias=True).to(dtype=param_dtype, device=param_device)
            nn.init.normal_(dit.speed_token_proj.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(dit.speed_token_proj.bias)
            dit.speed_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=param_dtype, device=param_device))


        self.use_cam_traj = bool(getattr(self.args, "cam_traj_condition", False))
        if self.use_cam_traj:
            for blk in dit.blocks:
                if not hasattr(blk, "cam_traj_encoder"):
                    blk.cam_traj_encoder = nn.Linear(12, dim, bias=True).to(dtype=param_dtype, device=param_device)
                    with torch.no_grad():
                        blk.cam_traj_encoder.weight.zero_()
                        if blk.cam_traj_encoder.bias is not None:
                            blk.cam_traj_encoder.bias.zero_()


        for blk in dit.blocks:
            if not hasattr(blk, "projector"):
                blk.projector = nn.Linear(dim, dim, bias=True).to(dtype=param_dtype, device=param_device)
                with torch.no_grad():
                    blk.projector.weight.copy_(torch.eye(dim, dtype=param_dtype, device=param_device))
                    blk.projector.bias.zero_()

        self.refine_mode = bool(getattr(args, "refine_mode", False))
        self.refine_use_global_context = bool(getattr(args, "refine_use_global_context", False))
        
        if self.refine_use_global_context:
            for blk in dit.blocks:
                if not hasattr(blk, "global_context_encoder"):
                    blk.global_context_encoder = nn.Linear(16, dim, bias=True).to(dtype=param_dtype, device=param_device)
                    with torch.no_grad():
                        blk.global_context_encoder.weight.zero_()
                        if blk.global_context_encoder.bias is not None:
                            blk.global_context_encoder.bias.zero_()

        if resume_ckpt_path is not None:
            if not os.path.exists(resume_ckpt_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
            
            print(f"[Resume] Loading checkpoint from: {resume_ckpt_path}")
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print(f"[Resume] Successfully loaded checkpoint")


        self.freeze_parameters()
        for name, p in dit.named_parameters():
            if any(k in name for k in [
                                       "speed_token_proj", 
                                       "speed_token_scale", 
                                       "projector", 
                                       "cam_traj_encoder",
                                       "global_context_encoder",
                                       ]):
                p.requires_grad = True

        for name, p in dit.named_parameters():
            if "self_attn" in name:
                p.requires_grad = True

        trainable_params = 0
        seen = set()
        for _, module in self.pipe.denoising_model().named_modules():
            for p in module.parameters():
                if p.requires_grad and p not in seen:
                    trainable_params += p.numel()
                    seen.add(p)
        print(f"Total number of trainable parameters: {trainable_params}")

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def _encode_prompt_on_device(self, text: str):
        self.pipe.device = self.device
        pe = self.pipe.encode_prompt(text)
        if "context" in pe and isinstance(pe["context"], torch.Tensor):
            pe["context"] = pe["context"].to(device=self.device, dtype=self.pipe.torch_dtype)
        for k, v in list(pe.items()):
            if isinstance(v, torch.Tensor) and k != "context":
                pe[k] = v.to(device=self.device)
        return pe
    

    @staticmethod
    def _repeat_last_frame_temporally(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            C, T, H, W = x.shape
            last = x[:, T-1:T, :, :]
            return last.repeat(1, T, 1, 1)
        elif x.ndim == 5:
            B, C, T, H, W = x.shape
            last = x[:, :, T-1:T, :, :]
            return last.repeat(1, 1, T, 1, 1)
        else:
            raise ValueError(f"Expected 4D or 5D video tensor, got shape {tuple(x.shape)}")
    
    def _downsample_m81_to_m21(self, m81: torch.Tensor) -> torch.Tensor:
        if m81.ndim == 1:
            m81 = m81.unsqueeze(0)

        x = m81.to(self.device, dtype=self.pipe.torch_dtype).unsqueeze(1)

        x = F.avg_pool1d(
            x, kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

        x = F.avg_pool1d(
            x, kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

        m21 = x.squeeze(1)
        return m21
        
    
    def _cam_traj_21_to_xyz81(self, cam_traj_21: torch.Tensor,
                            mode_hint: str = "identityR") -> np.ndarray:
        ct = cam_traj_21
        if isinstance(ct, torch.Tensor):
            ct = ct.detach().to(device="cpu", dtype=torch.float32)
            if ct.ndim == 3:
                ct = ct[0]
        ct = ct.view(21, 3, 4)
        trans = ct[:, :, 3]

        mode = mode_hint
        if mode_hint not in ("identityR",):
            mode = "identityR"

        t21 = trans

        t21_np = t21.numpy().astype(np.float32)
        t_idx21 = np.arange(21, dtype=np.float32) * 4.0
        t_idx81 = np.arange(81, dtype=np.float32)
        xyz = []
        for a in range(3):
            xyz.append(np.interp(t_idx81, t_idx21, t21_np[:, a]))
        traj_xyz81 = np.stack(xyz, axis=1).astype(np.float32)
        return traj_xyz81


    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device

        input_vid  = batch["input_video"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        target_vid = batch["target_video"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        text       = batch["text"]
        dbg        = batch.get("debug_info", None)

        if self.refine_mode:
            with torch.no_grad():
                L_input = self.pipe.encode_video(input_vid, **self.tiler_kwargs)[0].to(self.device)
                TGT = self.pipe.encode_video(target_vid, **self.tiler_kwargs)[0].to(self.device)

            global_context_feat = None
            if self.refine_use_global_context:
                if L_input.ndim == 4:
                    L_input_5d = L_input.unsqueeze(0)
                else:
                    L_input_5d = L_input
                global_context_feat = L_input_5d.mean(dim=(3, 4))
                global_context_feat = global_context_feat.transpose(1, 2)

            del input_vid, target_vid
            torch.cuda.empty_cache()

            if L_input.ndim == 4: L_input = L_input.unsqueeze(0)
            if TGT.ndim     == 4: TGT     = TGT.unsqueeze(0)

            m81 = batch["m81"]
            if m81.ndim == 1: m81 = m81.unsqueeze(0)
            m21 = self._downsample_m81_to_m21(m81)
            m21 = m21.to(self.device, dtype=L_input.dtype).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            L_mask = L_input * m21

            latents = torch.cat([TGT, L_mask], dim=2)

            del L_input, TGT, m21, L_mask
            torch.cuda.empty_cache()

            prompt_emb = self._encode_prompt_on_device(text)

            noise = torch.randn_like(latents, device=self.device)
            timesteps = self.pipe.scheduler.timesteps
            if timesteps.device != self.device:
                timesteps = timesteps.to(self.device)
            timestep_id = torch.randint(0, timesteps.shape[0], (1,), device=self.device)
            timestep = timesteps[timestep_id].to(dtype=self.pipe.torch_dtype)

            extra_in = self.pipe.prepare_extra_input(latents)
            if isinstance(extra_in, dict):
                extra_input = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in extra_in.items()}
            elif torch.is_tensor(extra_in):
                extra_input = extra_in.to(self.device)
            elif isinstance(extra_in, (list, tuple)):
                extra_input = [x.to(self.device) if torch.is_tensor(x) else x for x in extra_in]
            else:
                extra_input = extra_in

            origin_latents = latents.clone()
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep).to(self.device)

            tgt_len = 21
            noisy_latents[:, :, tgt_len:, ...] = origin_latents[:, :, tgt_len:, ...]

            training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

            noise_pred = self.pipe.denoising_model()(
                noisy_latents, timestep=timestep,
                **prompt_emb, **extra_input,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                cam_traj=None,
                global_context_feat=global_context_feat,
            )

            loss = torch.nn.functional.mse_loss(
                noise_pred[:, :, :tgt_len, ...].float(),
                training_target[:, :, :tgt_len, ...].float()
            )
            loss = loss * self.pipe.scheduler.training_weight(timestep)

            self.log("train_loss", loss, prog_bar=True)
            if dbg is not None:
                self.log("debug/refine_speed", torch.tensor(float(dbg.get("refine_speed", 0.0)),
                        device=self.device), prog_bar=False)
                self.log("debug/refine_k", torch.tensor(float(dbg.get("refine_window_k", 0.0)),
                        device=self.device), prog_bar=False)

            return loss

        use_static = False
        if getattr(self.args, "static_image_input", False):
            use_static = True
        else:
            ratio = float(getattr(self.args, "static_input_ratio", 0.0))
            if ratio > 0.0:
                use_static = (torch.rand(1, device=self.device).item() < ratio)

        if use_static:
            input_vid = self._repeat_last_frame_temporally(input_vid)

        self.log("debug/static_input_flag", 1.0 if use_static else 0.0, prog_bar=False)

        def _extract_speed_z(dbg_field):
            if isinstance(dbg_field, (list, tuple)):
                vals = [float(x.get("speed_log2", 0.0)) for x in dbg_field]
            else:
                vals = [float(dbg_field.get("speed_log2", 0.0))]
            t = torch.tensor(vals, dtype=self.pipe.torch_dtype, device=self.device).unsqueeze(1)
            return t

        speed_z = _extract_speed_z(dbg)
        

        t0 = time.time()
        input_latents  = self.pipe.encode_video(input_vid,  **self.tiler_kwargs)[0].to(self.device)
        t_in_enc = time.time() - t0

        t0 = time.time()
        target_latents = self.pipe.encode_video(target_vid, **self.tiler_kwargs)[0].to(self.device)
        t_tg_enc = time.time() - t0

        del input_vid, target_vid
        torch.cuda.empty_cache()

        if input_latents.ndim == 4:
            input_latents = input_latents.unsqueeze(0)
        if target_latents.ndim == 4:
            target_latents = target_latents.unsqueeze(0)

        latents = torch.cat((target_latents, input_latents), dim=2).to(self.device)

        prompt_emb = self._encode_prompt_on_device(text)

        noise = torch.randn_like(latents, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        if timesteps.device != self.device:
            timesteps = timesteps.to(self.device)
        timestep_id = torch.randint(0, timesteps.shape[0], (1,), device=self.device)
        timestep = timesteps[timestep_id].to(dtype=self.pipe.torch_dtype)

        extra_in = self.pipe.prepare_extra_input(latents)
        if isinstance(extra_in, dict):
            extra_input = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in extra_in.items()}
        elif torch.is_tensor(extra_in):
            extra_input = extra_in.to(self.device)
        elif isinstance(extra_in, (list, tuple)):
            extra_input = [x.to(self.device) if torch.is_tensor(x) else x for x in extra_in]
        else:
            extra_input = extra_in

        origin_latents = latents.clone()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep).to(self.device)

        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]

        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        cam_traj = batch.get("cam_traj", None)
        if cam_traj is not None:
            if isinstance(cam_traj, torch.Tensor) and cam_traj.ndim == 2:
                cam_traj = cam_traj.unsqueeze(0)
            cam_traj = cam_traj.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        def _extract_speed_val(dbg_field):
            if isinstance(dbg_field, (list, tuple)):
                vals = [float(x.get("speed", 1.0)) for x in dbg_field]
            else:
                vals = [float(dbg_field.get("speed", 1.0))]
            return vals[0]

        s_val = _extract_speed_val(dbg)

        self.log("debug/speed_now", float(s_val), prog_bar=False)
        self.log("debug/cond_cam_traj_present", 1.0 if cam_traj is not None else 0.0, prog_bar=False)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep,
            speed_scalar=speed_z,
            cam_traj=(cam_traj if self.use_cam_traj else None),
            **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
 
        loss = torch.nn.functional.mse_loss(
            noise_pred[:, :, :tgt_latent_len, ...].float(),
            training_target[:, :, :tgt_latent_len, ...].float()
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)

        self.log("debug/speed_z_abs_mean", speed_z.abs().mean(), prog_bar=False)

        try:
            if cam_traj is not None:
                ct = cam_traj
                if ct.ndim == 2:
                    ct = ct.unsqueeze(0)
                M = ct.view(ct.shape[0], -1, 3, 4)
                t = M[:, :, :, 3]
                t_norm = torch.linalg.norm(t, dim=2)
                t_norm_mean = t_norm.mean(dim=1).mean()
                t_norm_median = t_norm.median(dim=1).values.mean()
                self.log("debug/cam_traj_t_norm_mean", t_norm_mean, prog_bar=False)
                self.log("debug/cam_traj_t_norm_median", t_norm_median, prog_bar=False)
            else:
                self.log("debug/cam_traj_t_norm_mean", torch.tensor(0.0, device=self.device), prog_bar=False)
                self.log("debug/cam_traj_t_norm_median", torch.tensor(0.0, device=self.device), prog_bar=False)
        except Exception as _e:
            pass


        return loss

    def configure_optimizers(self):
        wd = 1e-2
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            name = n.lower()
            is_bias  = name.endswith("bias")
            is_norm  = ("norm" in name) or ("layernorm" in name) or ("ln" in name)
            is_scale = ("scale" in name) or ("gain" in name)
            is_1d    = (p.ndim <= 1)
            if is_bias or is_norm or is_scale or is_1d:
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = [
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=self.learning_rate)


    def on_after_backward(self):
        names_to_check = [
            "speed_token_proj.weight", "speed_token_proj.bias", "speed_token_scale",
            "cam_traj_encoder"
        ]
        for n, p in self.pipe.denoising_model().named_parameters():
            if any(k in n for k in names_to_check):
                g = (p.grad.float().norm().item()
                     if p.grad is not None else 0.0)
                self.log(f"grad/{n}", g, prog_bar=False)



    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            for n, p in self.pipe.denoising_model().named_parameters():
                if "token_" in n and ("proj" in n or "scale" in n):
                    self.log(f"param_abs/{n}", p.detach().float().abs().mean().item(), prog_bar=False)

                if self.use_cam_traj and "cam_traj_encoder" in n:
                    self.log(f"param_abs/{n}", p.detach().float().abs().mean().item(), prog_bar=False)

                if "traj_scale_proj" in n:
                    self.log(f"param_abs/{n}", p.detach().float().abs().mean().item(), prog_bar=False)




    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        current_step = self.global_step

        ckpt_name = f"step{current_step}.ckpt"
        param_ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, param_ckpt_path)
        print(f"[Checkpoint] Saved weights checkpoint: {param_ckpt_path}")

        try:
            for old_full in glob(os.path.join(checkpoint_dir, "step*_full.ckpt")):
                try:
                    os.remove(old_full)
                except FileNotFoundError:
                    pass
        except Exception as e:
            print(f"[Checkpoint] Warning: failed to cleanup old full ckpts: {e}")

        full_ckpt_name = f"step{current_step}_full.ckpt"
        full_ckpt_path = os.path.join(checkpoint_dir, full_ckpt_name)
        torch.save(checkpoint, full_ckpt_path)
        print(f"[Checkpoint] Saved full checkpoint: {full_ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train ReCamMaster (online encode with captions)")
    p.add_argument("--train_test_split_json", type=str, required=True,
                   help="Path to InteriorGS train_test_files.json")
    p.add_argument("--output_path", type=str, default="./")
    
    p.add_argument("--text_encoder_path", type=str, required=True)
    p.add_argument("--vae_path", type=str, required=True)
    p.add_argument("--dit_path", type=str, required=True)
    p.add_argument("--resume_ckpt_path", type=str, default=None)

    p.add_argument("--tiled", action="store_true", default=False)
    p.add_argument("--tile_size_height", type=int, default=34)
    p.add_argument("--tile_size_width", type=int, default=34)
    p.add_argument("--tile_stride_height", type=int, default=18)
    p.add_argument("--tile_stride_width", type=int, default=16)

    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--training_strategy", type=str, default="deepspeed_stage_1",
                   choices=["ddp", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"])
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--steps_per_epoch", type=int, default=500)

    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=960)

    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--checkpoint_every_n_epochs", type=int, default=1)
    p.add_argument("--batch_fetch", type=int, default=16)

    p.add_argument("--ckpt_path", type=str, default=None)

    p.add_argument("--task", type=str, default="preview", choices=["preview", "refine"], required=True)

    p.add_argument("--use_wandb", action="store_true", default=False,
                help="Enable Weights & Biases logging (scalars only)")
    p.add_argument("--wandb_run_name", type=str, default=None,
                help="Optional W&B run name")
    p.add_argument("--wandb_project", type=str, default="",
                help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default="",
                help="W&B entity name")
    p.add_argument("--wandb_api_key_file", type=str, 
                default="configs/wandb.txt",
                help="Path to file containing W&B API key")

    p.add_argument("--enable_speed_control", action="store_true", default=False)
    p.add_argument("--speed_min", type=float, default=1.0)
    p.add_argument("--speed_max", type=float, default=8.0)
    p.add_argument("--speed_fixed", type=float, default=None)
    p.add_argument("--speed_neutral_log_halfwidth", type=float, default=0,  
                   help="Neutral zone half-width in log2 space. E.g., 0.5 means log2 in [-0.5, 0.5] (s in [2^-0.5, 2^0.5]) will snap to 1x.")

    p.add_argument("--static_image_input", action="store_true", default=False,
                help="If set, use the last frame of input video repeated 81 times as model input.")
    

    p.add_argument("--static_input_ratio", type=float, default=0.5,
               help="Probability in [0,1]. With this probability, replace input video with static sequence (last frame repeated). "
                    "If --static_image_input is True, equivalent to ratio=1.0 (all static).")

    p.add_argument("--cam_traj_condition", action="store_true", default=False,
               help="Enable camera trajectory conditioning (ReCamMaster style), expects (21,12) per sample.")
    

    p.add_argument(
        "--re_scale_pose", type=str, default="none",
        help="Rescale camera trajectory translations to a unified target scale. "
             "Options: 'none' | 'unit_median' | 'fixed:<float>' (e.g., fixed:0.5). "
             "Only applies to identityR cam_traj; velocity+scale path is untouched."
    )

    p.add_argument("--traj_filter_enable", action="store_true", default=False,
                   help="Filter extreme/corrupted cam_traj samples during Dataset sampling")
    p.add_argument("--traj_tnorm_mean_max", type=float, default=30.0,
                   help="Max mean of translation norm per step (after rescaling). Samples exceeding this are dropped.")
    p.add_argument("--traj_tnorm_median_max", type=float, default=30.0,
                   help="Max median of translation norm per step (after rescaling). Samples exceeding this are dropped.")
    p.add_argument("--traj_tnorm_any_max", type=float, default=60.0,
                   help="Max of any single translation norm (after rescaling). Samples exceeding this are dropped.")
    p.add_argument("--traj_require_finite", action="store_true", default=False,
                   help="Require cam_traj to be finite (no NaN/Inf)")


    p.add_argument("--speed_two_bucket", action="store_true", default=False,
                   help="Enable two-bucket speed sampling: 50% sample 1.0x, 50% uniform from [1.1, 8.0] (rounded to 1 decimal)")
    p.add_argument("--speed_bucket_prob_one", type=float, default=0.5,
                   help="Probability of sampling 1.0x speed, default 0.5")
    p.add_argument("--speed_bucket_min_fast", type=float, default=1.1,
                   help="Min speed for fast bucket (inclusive), default 1.1")
    p.add_argument("--speed_bucket_max_fast", type=float, default=8.0,
                   help="Max speed for fast bucket (inclusive), default 8.0")
    p.add_argument("--round_speed_one_decimal", action="store_true", default=True,
                   help="Round speed to 1 decimal place")

    p.add_argument("--interiorgs_data_root", type=str,
                   default="data/InteriorGS-360video",
                   help="Local root directory of InteriorGS dataset")
    p.add_argument("--interiorgs_frames_subdir", type=str, default="pano_camera0",
                   help="Subdirectory for InteriorGS frame files")
    p.add_argument("--interiorgs_max_frames", type=int, default=800,
                   help="Max number of frames per scene in InteriorGS")
    p.add_argument("--interiorgs_frame_ext", type=str, default="png",
                   help="Frame file extension for InteriorGS dataset (jpg or png).")
    
    p.add_argument("--refine_speed_min", type=float, default=1.1,
                   help="Min speed in Refine mode")
    p.add_argument("--refine_speed_max", type=float, default=8.0,
                   help="Max speed in Refine mode")
    p.add_argument("--refine_window_policy", type=str, default="random",
                   choices=["random", "center"],
                   help="Window selection policy in Refine mode: random or center")
    p.add_argument("--refine_degrade_down_h", type=int, default=240,
                   help="Degradation downsample height in Refine mode")
    p.add_argument("--refine_degrade_down_w", type=int, default=480,
                   help="Degradation downsample width in Refine mode")
    p.add_argument("--refine_use_global_context", action="store_true", default=False,
                   help="Enable global context feature injection in Refine mode")
    
    return p.parse_args()


def train_online(args):
    dataset = OnlineFramesDataset(
        train_test_split_json=args.train_test_split_json,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps_per_epoch=args.steps_per_epoch,
        batch_fetch=args.batch_fetch,

        enable_speed_control=args.enable_speed_control,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        speed_fixed=args.speed_fixed,
        speed_neutral_log_halfwidth=args.speed_neutral_log_halfwidth,
        
        speed_two_bucket=args.speed_two_bucket,
        speed_bucket_prob_one=args.speed_bucket_prob_one,
        speed_bucket_min_fast=args.speed_bucket_min_fast,
        speed_bucket_max_fast=args.speed_bucket_max_fast,
        round_speed_one_decimal=args.round_speed_one_decimal,

        cam_traj_condition=args.cam_traj_condition,
        re_scale_pose_raw = args.re_scale_pose,

        traj_filter_enable = args.traj_filter_enable,
        traj_tnorm_mean_max = args.traj_tnorm_mean_max,
        traj_tnorm_median_max = args.traj_tnorm_median_max,
        traj_tnorm_any_max = args.traj_tnorm_any_max,
        traj_require_finite = args.traj_require_finite,
        
        interiorgs_data_root = args.interiorgs_data_root,
        interiorgs_frames_subdir = args.interiorgs_frames_subdir,
        interiorgs_max_frames = args.interiorgs_max_frames,
        interiorgs_frame_ext = args.interiorgs_frame_ext,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LightningModelForTrainOnline(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        output_path=args.output_path,
        args=args
    )

    wb_logger = None
    if args.use_wandb:
        _WANDB_API_KEY = None
        try:
            with open(args.wandb_api_key_file, "r") as f:
                _WANDB_API_KEY = f.read().strip()
                if "=" in _WANDB_API_KEY:
                    _WANDB_API_KEY = _WANDB_API_KEY.split("=", 1)[1].strip()
        except Exception as e:
            print(f"[W&B] Failed to read API key from {args.wandb_api_key_file}: {e}")
            print("[W&B] Will try to use existing wandb login or environment variable")
        
        @rank_zero_only
        def _create_wandb_logger():
            if _WANDB_API_KEY:
                try:
                    os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
                    wandb.login(key=_WANDB_API_KEY, relogin=True)
                    print(f"[W&B] Logged in successfully")
                except Exception as e:
                    print(f"[W&B] Login failed: {e}")

            return WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                save_dir=os.path.join(args.output_path, "wandb"),
                log_model=False,
                settings=wandb.Settings(start_method="thread"),
            )

        wb_logger = _create_wandb_logger()

    logger = wb_logger if wb_logger is not None else None

    trainer_kwargs = dict(
        accelerator="gpu",
        devices="auto",
        num_nodes=args.num_nodes,
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ModelCheckpoint(save_top_k=-1, every_n_epochs=args.checkpoint_every_n_epochs),
                   LearningRateMonitor(logging_interval="step"),
                   DeviceStatsMonitor(),
                   ],
        logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
    )


    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, dataloader, ckpt_path=args.ckpt_path)


def train_refine(args):
    args.refine_mode = True
    
    dataset = OnlineFramesDataset(
        train_test_split_json=args.train_test_split_json,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps_per_epoch=args.steps_per_epoch,
        batch_fetch=args.batch_fetch,

        refine_mode=True,
        refine_speed_min=args.refine_speed_min,
        refine_speed_max=args.refine_speed_max,
        refine_window_policy=args.refine_window_policy,
        refine_degrade_down_h=args.refine_degrade_down_h,
        refine_degrade_down_w=args.refine_degrade_down_w,
        refine_use_global_context=args.refine_use_global_context,
        
        interiorgs_data_root = args.interiorgs_data_root,
        interiorgs_frames_subdir = args.interiorgs_frames_subdir,
        interiorgs_max_frames = args.interiorgs_max_frames,
        interiorgs_frame_ext = args.interiorgs_frame_ext,
        
        enable_speed_control=False,
        cam_traj_condition=False,
        traj_filter_enable=False,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LightningModelForTrainOnline(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        output_path=args.output_path,
        args=args
    )

    wb_logger = None
    if args.use_wandb:
        _WANDB_API_KEY = None
        try:
            with open(args.wandb_api_key_file, "r") as f:
                _WANDB_API_KEY = f.read().strip()
                if "=" in _WANDB_API_KEY:
                    _WANDB_API_KEY = _WANDB_API_KEY.split("=", 1)[1].strip()
        except Exception as e:
            print(f"[W&B] Failed to read API key from {args.wandb_api_key_file}: {e}")
            print("[W&B] Will try to use existing wandb login or environment variable")
        
        @rank_zero_only
        def _create_wandb_logger():
            if _WANDB_API_KEY:
                try:
                    os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
                    wandb.login(key=_WANDB_API_KEY, relogin=True)
                    print(f"[W&B] Logged in successfully")
                except Exception as e:
                    print(f"[W&B] Login failed: {e}")

            return WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                save_dir=os.path.join(args.output_path, "wandb"),
                log_model=False,
                settings=wandb.Settings(start_method="thread"),
            )

        wb_logger = _create_wandb_logger()

    logger = wb_logger if wb_logger is not None else None

    trainer_kwargs = dict(
        accelerator="gpu",
        devices="auto",
        num_nodes=args.num_nodes,
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ModelCheckpoint(save_top_k=-1, every_n_epochs=args.checkpoint_every_n_epochs),
                   LearningRateMonitor(logging_interval="step"),
                   DeviceStatsMonitor(),
                   ],
        logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
    )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, dataloader, ckpt_path=args.ckpt_path)




if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    
    if args.task == "preview":
        train_online(args)
    elif args.task == "refine":
        train_refine(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

