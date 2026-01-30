import os
import re
import json
import random
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

def pad4(n: int) -> str:
    """Format frame index with 4-digit zero-padding"""
    return f"{n:04d}"

def parse_emb_name(emb_name: str) -> Tuple[int, int]:
    """Parse embedding name like '0_80' or '81_161' to (start, end)"""
    m = re.match(r"^(\d+)_(\d+)(?:\.tensors\.pth)?$", emb_name)
    if m is None:
        raise ValueError(f"Bad emb name: {emb_name}")
    return int(m.group(1)), int(m.group(2))

class RaCamDataset(Dataset):
    """
    Dataset for Self-Forcing training with OmniRoam teacher model.
    
    Loads:
    - Static 81-frame videos from local InteriorGS-360video dataset
    - Camera trajectories from COLMAP transforms.json
    - Returns video in pixel space (VAE encoding done by trainer)
    """

    def __init__(
        self,
        dataset_root: str = "data/InteriorGS-360video",
        split_json: str = None,
        frames_subdir: str = "pano_camera0",
        max_frames: int = 800,
        frame_ext: str = "png",

        height: int = 480,
        width: int = 960,
        num_frames: int = 81,
        steps_per_epoch: int = 500,
        batch_fetch: int = 16,
        debug: bool = False,

        re_scale_pose: str = "unit_median",
    ):
        print(f"[OmniRoamDataset] Initializing dataset...")

        if not os.path.isabs(dataset_root):
            omniroam_root = Path(__file__).resolve().parent.parent.parent
            self.dataset_root = (omniroam_root / dataset_root).resolve()
        else:
            self.dataset_root = Path(dataset_root).resolve()

        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root not found: {self.dataset_root}")

        print(f"[OmniRoamDataset] Dataset root: {self.dataset_root}")

        if split_json is None:
            raise ValueError("split_json is required")

        split_path = Path(split_json)
        if not split_path.exists():
            raise ValueError(f"Split JSON not found: {split_json}")

        with open(split_path, "r") as f:
            split = json.load(f)

        self.train_dict: Dict[str, dict] = split.get("train", {})
        self.video_ids = list(self.train_dict.keys())

        if len(self.video_ids) == 0:
            raise ValueError(f"No train videos found in {split_json}")

        print(f"[OmniRoamDataset] Loaded {len(self.video_ids)} videos from split")

        self.frames_subdir = frames_subdir.strip("/")
        self.max_frames = int(max_frames)
        self.frame_ext = str(frame_ext).lower().lstrip(".")

        self._colmap_cache: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch > 0 else 1_000_000
        self.batch_fetch = max(1, int(batch_fetch))
        self.debug = debug

        self._re_scale_mode, self._re_scale_target = self._parse_re_scale_pose(re_scale_pose)
        print(f"[OmniRoamDataset] Trajectory rescale: mode={self._re_scale_mode}, target={self._re_scale_target}")

    def __len__(self):
        return self.steps_per_epoch

    def _frame_path(self, video_id: str, idx: int) -> Path:
        """Get local path for a frame"""
        return self.dataset_root / video_id / self.frames_subdir / f"frame_{pad4(idx)}.{self.frame_ext}"

    def _transforms_path(self, video_id: str) -> Path:
        """Get local path for COLMAP transforms.json"""
        return self.dataset_root / video_id / "transforms.json"

    def _ns_transform_to_internal_c2w(self, M_ns: np.ndarray) -> np.ndarray:
        """Convert OpenCV cam-to-world to internal coordinate system"""
        M_ns = np.asarray(M_ns, dtype=np.float64)
        Rwc = M_ns[:3, :3]
        Cw = M_ns[:3, 3]

        fwd = Rwc[:, 2]
        up = -Rwc[:, 1]
        rgt = Rwc[:, 0]

        R_int = np.stack([fwd, up, rgt], axis=1)

        M_int = np.eye(4, dtype=np.float64)
        M_int[:3, :3] = R_int
        M_int[:3, 3] = Cw
        return M_int

    def _rotmat_to_quat_wxyz(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        R = np.asarray(R, dtype=np.float64)
        t = np.trace(R)
        if t > 0.0:
            S = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2,1] - R[1,2]) / S
            y = (R[0,2] - R[2,0]) / S
            z = (R[1,0] - R[0,1]) / S
        else:
            i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
            if i == 0:
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
                x = 0.25 * S
                y = (R[0,1] + R[1,0]) / S
                z = (R[0,2] + R[2,0]) / S
                w = (R[2,1] - R[1,2]) / S
            elif i == 1:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
                y = 0.25 * S
                x = (R[0,1] + R[1,0]) / S
                z = (R[1,2] + R[2,1]) / S
                w = (R[0,2] - R[2,0]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
                z = 0.25 * S
                x = (R[0,2] + R[2,0]) / S
                y = (R[1,2] + R[2,1]) / S
                w = (R[1,0] - R[0,1]) / S
        q = np.array([w, x, y, z], dtype=np.float64)
        q /= (np.linalg.norm(q) + 1e-12)
        return q

    def _quat_wxyz_to_rotmat(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix"""
        w, x, y, z = (q / (np.linalg.norm(q) + 1e-12)).tolist()
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
            [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
            [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
        ], dtype=np.float64)

    def _avg_rotations_quat(self, R_list: List[np.ndarray]) -> np.ndarray:
        """Average multiple rotation matrices using quaternion averaging"""
        assert len(R_list) >= 1
        q0 = self._rotmat_to_quat_wxyz(R_list[0])
        acc = q0.copy()
        for R in R_list[1:]:
            q = self._rotmat_to_quat_wxyz(R)
            if np.dot(q, q0) < 0.0:
                q = -q
            acc += q
        acc /= (np.linalg.norm(acc) + 1e-12)
        return self._quat_wxyz_to_rotmat(acc)

    def _get_colmap_map(self, video_id: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Load COLMAP poses and return dict[frame_idx] = (R_wc, C_w)
        InteriorGS dataset: directly use pano_camera0 rotation
        """
        if video_id in self._colmap_cache:
            return self._colmap_cache[video_id]

        transforms_path = self._transforms_path(video_id)
        if not transforms_path.exists():
            raise RuntimeError(f"transforms.json not found: {transforms_path}")

        with open(transforms_path, "r") as f:
            j = json.load(f)

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
            C_w_fixed = np.array([C_w_raw[0], C_w_raw[1], C_w_raw[2]], dtype=np.float64)

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
            raise RuntimeError(f"No pano_camera0 frames found for {video_id}: {transforms_path}")

        self._colmap_cache[video_id] = mp
        return mp

    def _parse_re_scale_pose(self, s: str) -> Tuple[str, float]:
        """
        Parse re_scale_pose config string.
        Returns: (mode, target_value)
        - 'none' -> ('none', None)
        - 'unit_median' -> ('unit_median', 1.0)
        - 'fixed:0.5' -> ('fixed', 0.5)
        """
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

    def _rescale_cam_traj_identityR(self, cam_traj_21: torch.Tensor) -> torch.Tensor:
        """
        Rescale trajectory to match teacher training.
        cam_traj_21: (21, 12) with [I|t] per row, where t is at columns [3,7,11]
        
        Returns: scaled trajectory (21, 12)
        """
        if cam_traj_21 is None or self._re_scale_mode == "none":
            return cam_traj_21

        M = cam_traj_21.view(21, 3, 4)
        t = M[:, :, 3]

        if t.shape[0] < 2:
            return cam_traj_21

        dt = t[1:] - t[:-1]
        step = torch.linalg.norm(dt, dim=1)

        if step.numel() == 0:
            return cam_traj_21

        s_local = torch.median(step).item()

        eps = 1e-8
        if not np.isfinite(s_local) or s_local < eps:
            return cam_traj_21

        if self._re_scale_mode == "unit_median":
            s_tgt = 1.0
        elif self._re_scale_mode == "fixed":
            s_tgt = float(self._re_scale_target)
        else:
            return cam_traj_21

        alpha = s_tgt / s_local
        t_scaled = t * alpha
        M[:, :, 3] = t_scaled

        if self.debug:
            print(f"[RaCamDataset] Trajectory rescale: s_local={s_local:.4f}, s_tgt={s_tgt:.4f}, alpha={alpha:.4f}")

        return M.reshape(21, 12)

    def _build_cam_traj_21_identityR(
        self,
        ns_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
        tg_start_idx: int,
        num_frames: int = 81
    ) -> torch.Tensor:
        """
        Build 21x12 camera trajectory from 81-frame sequence.
        Exact logic from finetune_online_raymap.py:_build_cam_traj_21_identityR
        
        Returns: (21, 12) tensor with [I|t] flattened per row
        """
        C81 = []
        for i in range(num_frames):
            idx = tg_start_idx + i
            if idx not in ns_map:
                raise KeyError(f"Missing pose for idx={idx}")
            _R_wc, C_w = ns_map[idx]
            C81.append(C_w)
        C81 = np.stack(C81, axis=0)

        R_ref, C_ref = ns_map[tg_start_idx]
        R_ref_T = R_ref.T

        traj = []
        for k in range(21):
            j81 = 4 * k
            t_w = C81[j81] - C_ref
            t_ref = (R_ref_T @ t_w).astype(np.float64)

            M = np.concatenate([np.eye(3, dtype=np.float64), t_ref.reshape(3, 1)], axis=1)
            traj.append(M.reshape(-1))

        traj = np.stack(traj, axis=0).astype(np.float32)
        traj_tensor = torch.from_numpy(traj)

        traj_tensor = self._rescale_cam_traj_identityR(traj_tensor)

        return traj_tensor

    def _load_frames_by_indices(
        self,
        video_id: str,
        indices: List[int]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Load frames from local filesystem by index list.
        Returns: (3, T, H, W) tensor in [-1, 1] range, list of paths
        """
        if len(indices) == 0:
            raise ValueError("indices is empty")

        paths = [self._frame_path(video_id, int(k)) for k in indices]

        max_workers = max(4, self.batch_fetch)
        imgs = [None] * len(indices)

        def fetch_one(i_path):
            i, path = i_path
            if not path.exists():
                raise FileNotFoundError(f"Frame not found: {path}")
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return i, img

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_one, (i, path)) for i, path in enumerate(paths)]
            for fut in as_completed(futures):
                i, img = fut.result()
                imgs[i] = img

        import torch.nn.functional as F
        tensor_list = []
        for img in imgs:
            arr = np.array(img, dtype=np.uint8)
            t = torch.from_numpy(arr).float() / 255.0
            t = t.permute(2, 0, 1)
            tensor_list.append(t)

        batch_imgs = torch.stack(tensor_list, dim=0)

        batch_imgs = self._batch_resize_with_padding(batch_imgs, self.height, self.width)

        batch_imgs = (batch_imgs - 0.5) / 0.5

        from einops import rearrange
        vid = rearrange(batch_imgs, "T C H W -> C T H W")

        return vid, [str(p) for p in paths]

    def _batch_resize_with_padding(self, batch_imgs, target_height, target_width):
        """Resize maintaining aspect ratio, then pad/crop to exact target size"""
        import torch.nn.functional as F

        B, C, H, W = batch_imgs.shape

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

        if new_h < target_height:
            pad_total = target_height - new_h
            pad_top = pad_total // 2
            pad_bot = pad_total - pad_top
            batch_imgs = F.pad(batch_imgs, (0, 0, pad_top, pad_bot), value=0.0)
        elif new_h > target_height:
            top = (new_h - target_height) // 2
            batch_imgs = batch_imgs[:, :, top:top + target_height, :]

        if new_w < target_width:
            pad_total = target_width - new_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            batch_imgs = F.pad(batch_imgs, (pad_left, pad_right, 0, 0), value=0.0)
        elif new_w > target_width:
            left = (new_w - target_width) // 2
            batch_imgs = batch_imgs[:, :, :, left:left + target_width]

        return batch_imgs

    def __getitem__(self, idx):
        """
        Sample one video with trajectory.
        Returns dict ready for Self-Forcing training.
        """
        while True:
            try:
                vid = random.choice(self.video_ids)

                max_safe_start = self.max_frames - 162
                if max_safe_start < 1:
                    if self.debug:
                        print(f"[OmniRoamDataset] Skipping {vid}: max_frames ({self.max_frames}) too small for input+target")
                    continue

                in_start = random.randint(1, max_safe_start)
                in_end = in_start + 80

                tg_start = in_end + 1
                tg_end = tg_start + 80

                colmap_map = self._get_colmap_map(vid)

                target_frames_needed = list(range(tg_start, tg_end + 1))
                if not all(idx in colmap_map for idx in target_frames_needed):
                    if self.debug:
                        print(f"[OmniRoamDataset] Skipping {vid}: missing COLMAP poses for target frames [{tg_start}, {tg_end}]")
                    continue

                indices = list(range(in_start, in_end + 1))
                video, paths = self._load_frames_by_indices(vid, indices)

                last_frame = video[:, -1:, :, :]
                static_video = last_frame.repeat(1, 81, 1, 1)

                trajectory = self._build_cam_traj_21_identityR(
                    ns_map=colmap_map,
                    tg_start_idx=tg_start,
                    num_frames=81
                )

                if not torch.isfinite(trajectory).all():
                    if self.debug:
                        print(f"[OmniRoamDataset] Skipping {vid}: trajectory has NaN/Inf")
                    continue

                return {
                    "video": static_video,
                    "trajectory": trajectory,
                    "speed": 1.0,
                    "prompt": "panoramic video",
                    "video_id": vid,
                    "input_range": (in_start, in_end),
                    "target_range": (tg_start, tg_end)
                }

            except Exception as e:
                if self.debug:
                    print(f"[OmniRoamDataset] Failed to load sample: {e}")
                continue

