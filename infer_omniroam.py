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

import os, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from PIL import Image
import imageio.v2 as imageio

from diffsynth import ModelManager, WanVideoClickMapPipeline, save_video
from Studio.app_utils.stitch import read_video_as_tensor, write_tensor_to_video

import multiprocessing as mp
import math

def run_worker(worker_id: int, device: str, args):
    torch.cuda.set_device(int(device.split(":")[1]) if device.startswith("cuda:") else 0)
    
    if args.enable_refine:
        print(f"[GPU {worker_id}] Refine mode ON. Dir={args.refine_local_dir}")
    else:
        print(f"[GPU {worker_id}] Local image mode ON. Dir={args.local_images_dir}")


    print(f"[GPU {worker_id}] init model on {device}")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoClickMapPipeline.from_model_manager(model_manager, device=device)
    ensure_click_modules_and_load(pipe.dit, args.ckpt_path)

    has_global_context = any(hasattr(blk, "global_context_encoder") for blk in pipe.dit.blocks)
    pipe.use_global_context = has_global_context
    if has_global_context:
        print(f"[GPU {worker_id}] Detected global_context_encoder in model checkpoint.")
    
    if args.enable_refine:
        print(f"[GPU {worker_id}] Refine mode: disabling all conditions")
    else:
        has_cam_traj = any(hasattr(blk, "cam_traj_encoder") for blk in pipe.dit.blocks)
        if args.use_cam_traj and not has_cam_traj:
            print("[WARN] use_cam_traj=True but model has no cam_traj_encoder: auto disabled.")
            args.use_cam_traj = False

        print(f"[GPU {worker_id}] cam_traj={args.use_cam_traj}")
        
        re_scale_mode, re_scale_target = _parse_re_scale_pose(args.re_scale_pose)
        print(f"[GPU {worker_id}] re_scale_pose={args.re_scale_pose} -> mode={re_scale_mode}, target={re_scale_target}")




    pipe.to(device); pipe.to(dtype=torch.bfloat16)

    q = args._shared_queue
    infer_count = 0
    while True:
        try:
            item = q.get_nowait()
        except Exception:
            break

        try:
            if args.enable_refine:
                video_id, mp4_path = item
                print(f"[GPU {worker_id}] Processing local video: {video_id}")
                print(f"[GPU {worker_id}]   Input: {mp4_path}")
                out_dir = os.path.join(args.output_dir, video_id.replace("/", "__"))
                os.makedirs(out_dir, exist_ok=True)
                print(f"[GPU {worker_id}]   Output: {out_dir}")

                raw_T_H_W_3 = _read_local_mp4_as_tensor_01(mp4_path)
                src_full_raw = _video_uint8_to_CTHW_floatm11(raw_T_H_W_3, args.height, args.width)
                save_mp4_simple(src_full_raw, os.path.join(out_dir, "a_input_full_raw.mp4"), fps=30, quality=5)
                
                T_raw = src_full_raw.shape[1]
                if T_raw != 81:
                    print(f"[GPU {worker_id}]   Interpolating {T_raw} frames -> 81 frames")
                    src_full_81 = interp_to_len(src_full_raw, 81)
                else:
                    src_full_81 = src_full_raw
                save_mp4_simple(src_full_81, os.path.join(out_dir, "a_input_81.mp4"), fps=30, quality=5)
                
                global_ctx_feat = None
                if hasattr(pipe, 'use_global_context') and getattr(pipe, 'use_global_context', False):
                    print(f"[GPU {worker_id}] Extracting global context...")
                    global_ctx_video_480_960 = F.interpolate(
                        src_full_81.permute(1, 0, 2, 3),
                        size=(480, 960),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    ).permute(1, 0, 2, 3)
                    global_ctx_video_resized = F.interpolate(
                        global_ctx_video_480_960.permute(1, 0, 2, 3),
                        size=(args.height, args.width),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    ).permute(1, 0, 2, 3)
                    with torch.no_grad():
                        L_global = pipe.encode_video(
                            global_ctx_video_resized.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                        )[0].to(device=device)
                        if L_global.ndim == 4:
                            L_global_5d = L_global.unsqueeze(0)
                        else:
                            L_global_5d = L_global
                        global_ctx_feat = L_global_5d.mean(dim=(3, 4))
                        global_ctx_feat = global_ctx_feat.transpose(1, 2).contiguous()
                        del L_global, L_global_5d
                    print(f"[GPU {worker_id}] Global context extracted: {tuple(global_ctx_feat.shape)}")

                segs_81 = _split_indices_even_overlap1(81, max(1, int(args.refine_num_segments)))
                print(f"[GPU {worker_id}]   Segments: {segs_81}")

                seg_video_paths = []
                seg_lens = []

                for si, (s81, e81) in enumerate(segs_81):
                    seg_raw = src_full_81[:, s81:e81, :, :]
                    seg_len = e81 - s81
                    seg_lens.append(seg_len)
                    save_mp4_simple(seg_raw, os.path.join(out_dir, f"b_seg{si:02d}_input_raw_from81.mp4"), fps=30, quality=5)
                    
                    m81 = _make_m81_mask(s81, e81).to(device=device)

                    vid = pipe(
                        prompt="",
                        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        source_video=src_full_81.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                        speed_scalar=None,
                        cam_traj_condition=None,
                        m81=m81,
                        enable_refine=True,
                        use_full=False,
                        cfg_scale=args.cfg_scale,
                        num_inference_steps=args.num_inference_steps,
                        seed=0, tiled=False,
                        height=args.height, width=args.width,
                        global_context_feat=global_ctx_feat,
                    )
                    seg_video_path = os.path.join(out_dir, f"c_seg{si:02d}_generated_81.mp4")
                    save_video(vid, seg_video_path, fps=30, quality=5)
                    seg_video_paths.append(seg_video_path)

                print(f"[GPU {worker_id}] Loading {len(seg_video_paths)} segment videos for stitching...")
                gen_81_list = []
                for seg_path in seg_video_paths:
                    seg_tensor = read_video_as_tensor(seg_path)
                    gen_81_list.append(seg_tensor)
                
                if args.refine_use_crossfade and len(gen_81_list) > 1:
                    print(f"[GPU {worker_id}] Stitching with cross-fade (alpha={args.refine_crossfade_alpha})...")
                    gen_concat = _stitch_segments_with_crossfade(gen_81_list, seg_lens, alpha=args.refine_crossfade_alpha)
                else:
                    print(f"[GPU {worker_id}] Stitching with simple concatenation...")
                    gen_concat = _concat_time(gen_81_list)
                
                write_tensor_to_video(gen_concat, os.path.join(out_dir, "refined.mp4"), fps=30)
                print(f"[GPU {worker_id}] Done: {video_id} -> {os.path.join(out_dir, 'refined.mp4')}")

                infer_count += 1
                del src_full_raw, src_full_81, gen_81_list, gen_concat
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            else:
                img_path = item
                video_id = os.path.splitext(os.path.basename(img_path))[0]
                print(f"[GPU {worker_id}] [LOCAL] image={img_path} -> id={video_id}")

                out_dir = os.path.join(args.output_dir, video_id.replace("/", "__"))
                os.makedirs(out_dir, exist_ok=True)

                B = 1
                def _make_speed_tensor(z):
                    return torch.full((B, 1), float(z), dtype=torch.bfloat16, device=device)
                used_speed_s = 1.0
                used_speed_z = 0.0
                speed_scalar_tensor = _make_speed_tensor(0.0)
                if args.enable_speed_control:
                    if args.speed_fixed is None:
                        sampled = np.random.uniform(1.1, 8.0)
                        used_speed_s = float(np.clip(np.round(sampled, 2), 0.125, 8.0))
                    else:
                        used_speed_s = float(np.clip(args.speed_fixed, 0.125, 8.0))
                    used_speed_z = float(math.log2(used_speed_s))
                    speed_scalar_tensor = torch.tensor([[used_speed_z]], dtype=torch.bfloat16, device=device)

                prompt_text = ""

                src_video = load_local_image_as_video(
                    img_path, height=args.height, width=args.width, num_frames=args.num_frames
                )
                    
                cam_traj_arg = None
                traj_scale_tensor = None
                pos81 = None
                if args.use_cam_traj:
                    if args.traj_mode in ("gt", "random_gt"):
                        preset = args.traj_preset
                        cam_traj = make_cam_traj_from_preset_refspace(
                            preset=preset,
                            step_m=float(args.traj_step_m),
                            amp_m=float(args.traj_s_curve_amp_m),
                            zigzag_span_m=float(args.traj_zigzag_span_m),
                        )
                        cam_traj, s_local, alpha = _rescale_cam_traj_identityR(
                            cam_traj, re_scale_mode, re_scale_target
                        )

                    else:
                        preset = (args.traj_preset if args.traj_mode == "fixed"
                                else random.choice(["forward","backward","left","right","s_curve","zigzag_forward"]))
                        cam_traj = make_cam_traj_from_preset_refspace(
                            preset=preset,
                            step_m=float(args.traj_step_m),
                            amp_m=float(args.traj_s_curve_amp_m),
                            zigzag_span_m=float(args.traj_zigzag_span_m),
                        )
                        cam_traj, s_local, alpha = _rescale_cam_traj_identityR(
                            cam_traj, re_scale_mode, re_scale_target
                        )
                        print(f"[GPU {worker_id}] [re_scale] mode={re_scale_mode} s_local={s_local} alpha={alpha}")

                    cam_traj_arg = cam_traj.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

                save_mp4_simple(src_video, os.path.join(out_dir, "input.mp4"), fps=30, quality=5)

                vid = pipe(
                    prompt=prompt_text,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    source_video=src_video.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                    speed_scalar=speed_scalar_tensor,
                    cam_traj_condition=cam_traj_arg,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=0, tiled=False,
                    height=args.height, width=args.width,
                )

                save_video(vid, os.path.join(out_dir, "generated.mp4"), fps=30, quality=5)
                print(f"[GPU {worker_id}] [DONE] {os.path.join(out_dir, 'generated.mp4')}")
                infer_count += 1

                del src_video, vid
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {worker_id}] [WARN] skip item due to: {e}")
            continue

    print(f"[GPU {worker_id}] finished. total={infer_count}")


def resolve_ckpt_path(ckpt_path: str) -> str:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
    
    print(f"[CKPT] Loading local checkpoint: {ckpt_path}")
    return ckpt_path

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
            raise ValueError(f"Bad --re_scale_pose value: {s}. Expect 'fixed:<positive-float>'.")
    raise ValueError(f"Unknown --re_scale_pose value: {s}")

def _rescale_cam_traj_identityR(cam_traj_21: torch.Tensor, mode: str, s_target: float):
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

    s_local = step.median().item()
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
    M[:, :, 3] = t * alpha
    return M.reshape(-1, 12), s_local, alpha



def make_cam_traj_from_preset_refspace(preset: str,
                                       step_m: float,
                                       amp_m: float = 0.5,
                                       zigzag_span_m: float = 0.5,
                                       loop_radius_m: float = 2.0) -> torch.Tensor:
    import numpy as _np
    t_list = []
    if preset in ["forward","backward","left","right"]:
        dir_map = {
            "forward":  _np.array([+1, 0, 0], dtype=_np.float64),
            "backward": _np.array([-1, 0, 0], dtype=_np.float64),
            "right":    _np.array([0, 0, +1], dtype=_np.float64),
            "left":     _np.array([0, 0, -1], dtype=_np.float64),
        }
        d = dir_map[preset] * (float(step_m) / 4.0)
        p = _np.zeros(3, dtype=_np.float64)
        for i in range(81):
            t_list.append(p.copy())
            p += d
    elif preset == "s_curve":
        for i in range(81):
            x = (float(step_m)/4.0) * i
            z = float(amp_m) * _np.sin(2.0*_np.pi*i/80.0)
            t_list.append(_np.array([x, 0.0, z], dtype=_np.float64))
    elif preset == "zigzag_forward":
        for i in range(81):
            x = (float(step_m)/4.0) * i
            saw = ((i % 20) / 20.0) - 0.5
            z = float(zigzag_span_m) * saw * 2.0
            t_list.append(_np.array([x, 0.0, z], dtype=_np.float64))
    elif preset == "loop":
        R = float(loop_radius_m)
        for i in range(81):
            theta = 2.0 * _np.pi * i / 80.0
            x = R * (1.0 - _np.cos(theta))
            z = R * _np.sin(theta)
            t_list.append(_np.array([x, 0.0, z], dtype=_np.float64))
    else:
        raise ValueError(f"Unknown preset: {preset}")

    traj = []
    for k in range(21):
        j = 4 * k
        t_ref = t_list[j]
        M = _np.concatenate([_np.eye(3, dtype=_np.float64), t_ref.reshape(3,1)], axis=1)
        traj.append(M.reshape(-1))
    traj = _np.stack(traj, axis=0).astype(_np.float32)
    return torch.from_numpy(traj)


def to_uint8_video_array(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 5:
        x = x[0]
    assert x.ndim == 4 and x.shape[0] in (1, 3), f"expected (C,T,H,W), got {tuple(x.shape)}"
    x = x.detach().cpu().float()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = rearrange(x, "C T H W -> T H W C").mul(255.0).round().byte().numpy()
    return x

def save_mp4_simple(x: torch.Tensor, out_path: str, fps: int = 30, quality: int = 8):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = to_uint8_video_array(x)
    imageio.mimsave(out_path, arr, fps=fps, codec="libx264", quality=quality)

def _preprocess_CHW_to_size(x_chw_float01: torch.Tensor, height: int, width: int,
                            pad_value: float = 127.0/255.0) -> torch.Tensor:
    assert x_chw_float01.ndim == 3 and x_chw_float01.shape[0] == 3
    _, H0, W0 = x_chw_float01.shape
    x = x_chw_float01.unsqueeze(0)

    tgt_long = max(height, width)
    src_long = max(H0, W0)
    scale = (tgt_long / float(src_long)) if src_long > 0 else 1.0
    new_h = max(1, int(round(H0 * scale)))
    new_w = max(1, int(round(W0 * scale)))

    if new_h != H0 or new_w != W0:
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)

    _, _, h, w = x.shape
    if h < height:
        pad_total = height - h
        top = pad_total // 2; bot = pad_total - top
        x = F.pad(x, (0, 0, top, bot), value=pad_value)
        h = height
    elif h > height:
        top = (h - height) // 2
        x = x[:, :, top:top + height, :]

    if w < width:
        pad_total = width - w
        left = pad_total // 2; right = pad_total - left
        x = F.pad(x, (left, right, 0, 0), value=pad_value)
    elif w > width:
        left = (w - width) // 2
        x = x[:, :, :, left:left + width]

    x = (x - 0.5) / 0.5
    return x[0]


def load_local_image_as_video(path: str, height: int, width: int, num_frames: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    chw = _preprocess_CHW_to_size(t, height, width, pad_value=127.0/255.0)
    vid = chw.unsqueeze(1).repeat(1, num_frames, 1, 1)
    return vid


def list_images_in_dir(dir_path: str, exts: str):
    extset = {e.strip().lower() for e in exts.split(",") if e.strip()}
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if not os.path.isfile(p):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in extset:
            files.append(os.path.abspath(p))
    files.sort()
    return files


def _read_local_mp4_as_tensor_01(path: str) -> np.ndarray:
    rdr = imageio.get_reader(path)
    frames = []
    for fr in rdr:
        frames.append(fr)
    rdr.close()
    if len(frames) == 0:
        raise RuntimeError(f"Empty video: {path}")
    return np.stack(frames, axis=0)

def _hw_resize_and_center_crop_to(x_TCHW: torch.Tensor, height: int, width: int) -> torch.Tensor:
    T, C, H0, W0 = x_TCHW.shape
    tgt_long = max(height, width)
    src_long = max(H0, W0)
    scale = (tgt_long / float(src_long)) if src_long > 0 else 1.0
    new_h = max(1, int(round(H0 * scale)))
    new_w = max(1, int(round(W0 * scale)))
    x = x_TCHW
    if new_h != H0 or new_w != W0:
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    if new_h < height:
        pad_total = height - new_h
        top = pad_total // 2; bot = pad_total - top
        x = F.pad(x, (0, 0, top, bot), value=0.5)
        new_h = height
    elif new_h > height:
        top = (new_h - height) // 2
        x = x[:, :, top:top + height, :]
        new_h = height
    if new_w < width:
        pad_total = width - new_w
        left = pad_total // 2; right = pad_total - left
        x = F.pad(x, (left, right, 0, 0), value=0.5)
        new_w = width
    elif new_w > width:
        left = (new_w - width) // 2
        x = x[:, :, :, left:left + width]
        new_w = width
    return x

def _video_uint8_to_CTHW_floatm11(x_T_H_W_3: np.ndarray, height: int, width: int) -> torch.Tensor:
    x = torch.from_numpy(x_T_H_W_3).float() / 255.0
    x = x.permute(0,3,1,2).contiguous()
    x = _hw_resize_and_center_crop_to(x, height, width)
    x = x.permute(1,0,2,3).contiguous()
    x = (x - 0.5) / 0.5
    return x

def interp_to_len(vid_CTHW: torch.Tensor, T_out: int) -> torch.Tensor:
    x = vid_CTHW.unsqueeze(0)
    x = F.interpolate(x, size=(T_out, vid_CTHW.shape[2], vid_CTHW.shape[3]),
                      mode="trilinear", align_corners=True)
    return x[0]

def _split_indices_even_overlap1(T: int, K: int) -> list:
    base = T // K
    r = T % K
    segs = []
    acc = 0
    for i in range(K):
        length = base if i < K-1 else base + r
        s = acc
        e = acc + length
        e = min(e + 1, T)
        segs.append((s, e))
        acc = acc + base if i < K-1 else acc + (base + r)
    return segs

def _concat_time(list_CTHW: list) -> torch.Tensor:
    return torch.cat(list_CTHW, dim=1)

def _stitch_segments_with_crossfade(gen_81_list: list, seg_lens: list, alpha: float = 0.5) -> torch.Tensor:
    assert len(gen_81_list) > 0
    acc = None
    for i, seg in enumerate(gen_81_list):
        if acc is None:
            acc = seg
            continue
        prev_last = acc[:, -1:, :, :]
        curr_first = seg[:, :1, :, :]
        blended = (1.0 - alpha) * prev_last + alpha * curr_first
        acc[:, -1:, :, :] = blended
        acc = torch.cat([acc, seg[:, 1:, :, :]], dim=1)
    return acc

def _make_m81_mask(seg_start: int, seg_end: int) -> torch.Tensor:
    m = torch.zeros(81, dtype=torch.bfloat16)
    a = max(0, int(seg_start)); b = min(81, int(seg_end))
    if b > a:
        m[a:b] = 1.0
    return m.unsqueeze(0)


def ensure_click_modules_and_load(dit, ckpt_path: str):
    dim = dit.blocks[0].self_attn.q.weight.shape[0]
    p_dtype = dit.patch_embedding.weight.dtype
    p_device = dit.patch_embedding.weight.device

    ckpt_local = resolve_ckpt_path(ckpt_path)
    state_dict = torch.load(ckpt_local, map_location="cpu")
    sd_keys = set(state_dict.keys())

    def _has(prefix: str) -> bool:
        return any(k == prefix or k.startswith(prefix + ".") for k in sd_keys)

    for i, blk in enumerate(dit.blocks):
        pref = f"blocks.{i}.cam_traj_encoder"
        if _has(pref + ".weight"):
            if not hasattr(blk, "cam_traj_encoder"):
                enc = nn.Linear(12, dim, bias=True).to(dtype=p_dtype, device=p_device)
                blk.cam_traj_encoder = enc
                print(f"Created dit.blocks[{i}].cam_traj_encoder")

    if _has("speed_token_proj") and not hasattr(dit, "speed_token_proj"):
        dit.speed_token_proj = nn.Linear(1, dim, bias=True).to(dtype=p_dtype, device=p_device)
        print("Created dit.speed_token_proj")
    if _has("speed_token_scale") and not hasattr(dit, "speed_token_scale"):
        dit.speed_token_scale = nn.Parameter(torch.tensor(1e-1, dtype=p_dtype, device=p_device))
        print("Created dit.speed_token_scale")

    for blk in dit.blocks:
        if not hasattr(blk, "projector"):
            blk.projector = nn.Linear(dim, dim, bias=True).to(dtype=p_dtype, device=p_device)
            with torch.no_grad():
                blk.projector.weight.copy_(torch.eye(dim, dtype=p_dtype, device=p_device))
                blk.projector.bias.zero_()

    for i, blk in enumerate(dit.blocks):
        pref = f"blocks.{i}.global_context_encoder"
        if _has(pref + ".weight"):
            if not hasattr(blk, "global_context_encoder"):
                enc = nn.Linear(16, dim, bias=True).to(dtype=p_dtype, device=p_device)
                blk.global_context_encoder = enc
                print(f"Created dit.blocks[{i}].global_context_encoder")

    dit.load_state_dict(state_dict, strict=True)


def parse_args():
    p = argparse.ArgumentParser("Inference with local images for OmniRoam preview stage")
    
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--num_frames", type=int, default=81)

    
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./preview_test")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--devices", type=str, default=None,
                   help="Multi-GPU parallel, e.g., 'cuda:0,cuda:1,cuda:2,cuda:3'; leave empty for single GPU with --device")

    p.add_argument("--enable_speed_control", action="store_true", default=False,
                help="Enable speed control condition")
    p.add_argument("--speed_fixed", type=float, default=None,
                help="Fixed speed s in [1.0, 8.0]; if not set, random with 2 decimals")
    
    p.add_argument("--local_images_dir", type=str, default=None,
                   help="If provided, enables local static image inference mode. Each image in this directory will be repeated to num_frames frames as input video.")
    p.add_argument("--local_image_exts", type=str, default=".jpg,.jpeg,.png,.bmp",
                   help="Comma-separated list of local image extensions")
    

    p.add_argument("--use_cam_traj", action="store_true",
                   help="Enable camera trajectory condition (21x12 [I|t] per-sample). Model ckpt must contain cam_traj_encoder.")
    p.add_argument("--traj_mode", type=str, default="fixed",
                   choices=["gt","random_gt","fixed","random"],
                   help="gt: use target segment real trajectory (requires rig/colmap); random_gt: randomly select continuous GT trajectory from this video (no need to be adjacent to target); fixed: preset straight line; random: random from several presets.")
    p.add_argument("--traj_preset", type=str, default="forward",
                   choices=["forward","backward","left","right","s_curve","zigzag_forward","loop"],
                   help="Preset to use when traj_mode=fixed.")
    p.add_argument("--traj_step_m", type=float, default=0.25,
                   help="Displacement scale per latent timestep, in meters (try to match training data scale).")
    p.add_argument("--traj_s_curve_amp_m", type=float, default=1.6,
                   help="Lateral amplitude for s_curve trajectory (meters).")
    p.add_argument("--traj_zigzag_span_m", type=float, default=0.8,
                   help="Lateral swing amplitude for zigzag_forward trajectory (meters).")
    
    p.add_argument("--traj_loop_radius_m", type=float, default=1.5,
                   help="Loop trajectory circle radius (meters)")
    
    p.add_argument(
        "--re_scale_pose", type=str, default="none",
        help="Rescale camera trajectory translations to a unified target scale. "
             "Options: 'none' | 'unit_median' | 'fixed:<float>' (e.g., fixed:0.5). "
             "Only applies to identityR cam_traj; velocity+scale path is untouched."
    )
    
    p.add_argument("--enable_refine", action="store_true", default=False,
                   help="Enable refine inference. When enabled, no preview conditions are used, only m81 mask for segmented refinement.")
    p.add_argument("--refine_local_dir", type=str, default=None,
                   help="In refine mode, read subdirectories from this directory, each containing generated.mp4 input video.")
    p.add_argument("--refine_num_segments", type=int, default=8,
                   help="Split input evenly into how many segments. For 81 frames with 3 segments = 27 frames each; if not divisible, last segment gets 1~several extra frames.")
    p.add_argument("--refine_degrade_down_h", type=int, default=None,
                   help="Refine degradation target height; None means no degradation.")
    p.add_argument("--refine_degrade_down_w", type=int, default=None,
                   help="Refine degradation target width; None means no degradation.")
    p.add_argument("--refine_use_crossfade", action="store_true", default=False,
                   help="Use cross-fade to stitch video segments (smooth boundary transition) instead of simple concatenation. Recommended for multi-segment.")
    p.add_argument("--refine_crossfade_alpha", type=float, default=0.5,
                   help="Cross-fade blending weight, 0.5 means uniform mix of before/after frames.")
    
    return p.parse_args()


def main():
    args = parse_args()

    if args.enable_refine:
        if not args.refine_local_dir or not os.path.isdir(args.refine_local_dir):
            raise RuntimeError(f"Refine mode requires valid --refine_local_dir: {args.refine_local_dir}")
        
        video_subdirs = []
        for name in sorted(os.listdir(args.refine_local_dir)):
            subdir_path = os.path.join(args.refine_local_dir, name)
            if os.path.isdir(subdir_path):
                generated_mp4 = os.path.join(subdir_path, "generated.mp4")
                if os.path.exists(generated_mp4):
                    video_subdirs.append((name, generated_mp4))
        
        if len(video_subdirs) == 0:
            raise RuntimeError(f"No subdirectories with generated.mp4 found in: {args.refine_local_dir}")
        
        print(f"[Refine] Found {len(video_subdirs)} video subdirectories with generated.mp4")
        input_list = video_subdirs
    else:
        if not args.local_images_dir:
            raise RuntimeError("Preview mode requires --local_images_dir")
        
        if not os.path.isdir(args.local_images_dir):
            raise RuntimeError(f"--local_images_dir does not exist or is not a directory: {args.local_images_dir}")
        
        img_paths = list_images_in_dir(args.local_images_dir, args.local_image_exts)
        if len(img_paths) == 0:
            raise RuntimeError(f"No matching images in directory: {args.local_images_dir} (exts: {args.local_image_exts})")
        
        print(f"[INFO] Found {len(img_paths)} images in {args.local_images_dir}")
        input_list = img_paths

    if args.devices is None:
        args._shared_queue = None
        args.device = args.device if torch.cuda.is_available() else "cpu"
        m = mp.Manager()
        q = m.Queue()

        for item in input_list:
            q.put(item)

        args._shared_queue = q
        run_worker(worker_id=0, device=args.device, args=args)
        return

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(devices) == 0:
        raise ValueError("`--devices` parses to empty. Example: --devices cuda:0,cuda:1,cuda:2,cuda:3")

    mp.set_start_method("spawn", force=True)
    m = mp.Manager()
    q = m.Queue()
    
    for item in input_list:
        q.put(item)
    
    args._shared_queue = q

    procs = []
    for i, dev in enumerate(devices):
        pass_args = argparse.Namespace(**vars(args))
        p = mp.Process(target=run_worker, args=(i, dev, pass_args), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()



if __name__ == "__main__":
    main()
