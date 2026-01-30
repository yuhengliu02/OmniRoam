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
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from pathlib import Path

try:
    import av
except ImportError:
    av = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _resample_to_len(seg: torch.Tensor, L: int) -> torch.Tensor:
    C, T, H, W = seg.shape
    if T == L:
        return seg
    x = seg.unsqueeze(0)
    x = F.interpolate(x, size=(L, H, W), mode='trilinear', align_corners=True)
    return x[0]


def stitch_segments_with_crossfade(
    segments: List[torch.Tensor],
    alpha: float = 0.5,
    target_frames_per_segment: Optional[List[int]] = None,
) -> torch.Tensor:
    if not segments:
        raise ValueError("No segments provided")
    
    if len(segments) == 1:
        return segments[0]
    
    if target_frames_per_segment is None:
        seg_lens = [seg.shape[1] for seg in segments]
    else:
        seg_lens = target_frames_per_segment
    
    resampled = []
    for seg, L in zip(segments, seg_lens):
        if seg.shape[1] != L:
            seg = _resample_to_len(seg, L)
        resampled.append(seg)
    
    acc = None
    for i, seg_L in enumerate(resampled):
        if acc is None:
            acc = seg_L
            continue
        
        prev_last = acc[:, -1:, :, :]
        curr_first = seg_L[:, :1, :, :]
        blended = (1.0 - alpha) * prev_last + alpha * curr_first
        
        acc = torch.cat([acc[:, :-1, :, :], blended, seg_L[:, 1:, :, :]], dim=1)
    
    return acc


def read_video_as_tensor(video_path: str) -> torch.Tensor:
    if av is not None:
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)
        container.close()
        return torch.stack(frames, dim=1)
    elif imageio is not None:
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(img)
        reader.close()
        return torch.stack(frames, dim=1)
    else:
        raise ImportError("Neither av nor imageio is available for video reading")


def write_tensor_to_video(
    tensor: torch.Tensor, 
    output_path: str, 
    fps: int = 30,
    use_h264: bool = True,
) -> bool:
    try:
        C, T, H, W = tensor.shape
        
        if tensor.min() < 0:
            tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
        
        video_array = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        
        if av is not None and use_h264:
            container = av.open(output_path, mode='w')
            stream = container.add_stream('h264', rate=fps)
            stream.width, stream.height = W, H
            stream.pix_fmt = 'yuv420p'
            
            for frame_np in video_array:
                frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)
            
            packet = stream.encode(None)
            if packet:
                container.mux(packet)
            container.close()
        elif imageio is not None:
            imageio.mimsave(
                output_path,
                video_array,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                quality=8,
            )
        else:
            raise ImportError("Neither av nor imageio is available for video writing")
        
        return True
    except Exception as e:
        print(f"[Stitch] Error writing video: {e}")
        return False


def stitch_folder(
    folder_path: str,
    output_filename: str = "stitched.mp4",
    alpha: float = 0.5,
    pattern: str = r"c_seg(\d+)_generated_81\.mp4",
) -> Optional[str]:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[Stitch] Folder not found: {folder_path}")
        return None
    
    video_files = []
    for f in folder.iterdir():
        if f.suffix == '.mp4':
            match = re.search(pattern, f.name)
            if match:
                seg_idx = int(match.group(1))
                video_files.append((seg_idx, f))
    
    if not video_files:
        print(f"[Stitch] No segment files found in {folder_path}")
        return None
    
    video_files.sort(key=lambda x: x[0])
    
    print(f"[Stitch] Found {len(video_files)} segments")
    
    segments = []
    for seg_idx, seg_path in video_files:
        print(f"[Stitch] Reading segment {seg_idx}: {seg_path.name}")
        vid_tensor = read_video_as_tensor(str(seg_path))
        vid_tensor = (vid_tensor - 0.5) / 0.5
        segments.append(vid_tensor)
    
    print(f"[Stitch] Stitching {len(segments)} segments...")
    stitched = stitch_segments_with_crossfade(segments, alpha=alpha)
    
    output_path = folder / output_filename
    success = write_tensor_to_video(stitched, str(output_path))
    
    if success:
        print(f"[Stitch] Saved: {output_path}")
        return str(output_path)
    else:
        return None


def process_folder(folder_path: str, alpha: float = 0.5):
    stitch_folder(folder_path, output_filename='d_stitched_generated.mp4', alpha=alpha)


def main(root_dir: str):
    for subfolder in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            process_folder(subfolder_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root directory containing subfolders")
    parser.add_argument("--folder", type=str, help="Single folder to process")
    parser.add_argument("--alpha", type=float, default=0.5, help="Cross-fade weight")
    args = parser.parse_args()
    
    if args.folder:
        stitch_folder(args.folder, alpha=args.alpha)
    elif args.root_dir:
        main(args.root_dir)
    else:
        print("Please specify --root_dir or --folder")
