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

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cv2


def validate_erp_image(image_path: Path) -> Tuple[bool, str]:
    try:
        if not image_path.exists():
            return False, f"File does not exist: {image_path}"
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if image_path.suffix.lower() not in valid_extensions:
            return False, f"Invalid file type: {image_path.suffix}. Must be one of {valid_extensions}"
        
        with Image.open(image_path) as img:
            width, height = img.size
            
            ratio = width / height
            if not (1.95 <= ratio <= 2.05):
                return False, f"Invalid aspect ratio: {ratio:.2f}. ERP images must have 2:1 ratio (got {width}x{height})"
            
            if width < 256 or height < 128:
                return False, f"Image too small: {width}x{height}. Minimum is 256x128"
            
        return True, "Valid ERP image"
        
    except Exception as e:
        return False, f"Error reading image: {str(e)}"


def validate_erp_image_bytes(image_bytes: bytes) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    try:
        import io
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            
            ratio = width / height
            if not (1.95 <= ratio <= 2.05):
                return False, f"Invalid aspect ratio: {ratio:.2f}. ERP images must have 2:1 ratio (got {width}x{height})", None
            
            if width < 256 or height < 128:
                return False, f"Image too small: {width}x{height}. Minimum is 256x128", None
            
        return True, "Valid ERP image", (width, height)
        
    except Exception as e:
        return False, f"Error reading image: {str(e)}", None


def extract_last_frame(video_path: Path, output_path: Path) -> Tuple[bool, str]:
    try:
        if not video_path.exists():
            return False, f"Video file does not exist: {video_path}"
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, f"Cannot open video: {video_path}"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return False, "Video has no frames"
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "Failed to read last frame"
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=95)
        
        return True, f"Last frame saved to {output_path}"
        
    except Exception as e:
        return False, f"Error extracting last frame: {str(e)}"


def get_video_info(video_path: Path) -> Optional[dict]:
    try:
        if not video_path.exists():
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": 0.0,
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        
        cap.release()
        return info
        
    except Exception:
        return None


def validate_erp_video(video_path: Path) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    try:
        if not video_path.exists():
            return False, f"File does not exist: {video_path}", None
        
        valid_extensions = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
        if video_path.suffix.lower() not in valid_extensions:
            return False, f"Invalid file type: {video_path.suffix}. Must be one of {valid_extensions}", None
        
        info = get_video_info(video_path)
        if info is None:
            return False, "Cannot read video file", None
        
        width = info["width"]
        height = info["height"]
        
        ratio = width / height if height > 0 else 0
        if not (1.95 <= ratio <= 2.05):
            return False, f"Invalid aspect ratio: {ratio:.2f}. ERP videos must have 2:1 ratio (got {width}x{height})", None
        
        if width < 256 or height < 128:
            return False, f"Video too small: {width}x{height}. Minimum is 256x128", None
        
        if info["frame_count"] <= 0:
            return False, "Video has no frames", None
        
        return True, f"Valid ERP video ({info['frame_count']} frames)", (width, height)
        
    except Exception as e:
        return False, f"Error reading video: {str(e)}", None


def validate_erp_video_bytes(video_bytes: bytes, temp_path: Path) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        
        return validate_erp_video(temp_path)
        
    except Exception as e:
        return False, f"Error validating video: {str(e)}", None


def resize_image_for_inference(
    image_path: Path,
    target_width: int = 960,
    target_height: int = 480
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def save_video_tensor(
    video_tensor: np.ndarray,
    output_path: Path,
    fps: int = 30
) -> bool:
    try:
        T, H, W, C = video_tensor.shape
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        
        for t in range(T):
            frame_bgr = cv2.cvtColor(video_tensor[t], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return True
        
    except Exception:
        return False
