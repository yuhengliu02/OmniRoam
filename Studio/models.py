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

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class StageEnum(str, Enum):
    preview = "preview"
    self_forcing = "self_forcing"
    refine = "refine"


class TrajectoryEnum(str, Enum):
    forward = "forward"
    backward = "backward"
    left = "left"
    right = "right"
    up = "up"
    down = "down"
    s_curve = "s_curve"
    zigzag_forward = "zigzag_forward"
    loop = "loop"


class GenerateRequest(BaseModel):
    stage: StageEnum = Field(..., description="Model stage to use")
    trajectory: TrajectoryEnum = Field(..., description="Camera trajectory preset")
    scale: float = Field(
        default=1.0,
        ge=0.125,
        le=8.0,
        description="Speed scale factor (0.125 - 8.0), maps to speed_fixed"
    )
    num_frames: int = Field(
        default=81,
        ge=1,
        le=161,
        description="Number of frames to generate"
    )
    
    @validator("scale")
    def validate_scale(cls, v):
        return round(v, 2)


class UseLastFrameRequest(BaseModel):
    video_filename: str = Field(..., description="Video filename to extract last frame from")


class GpuMemoryInfo(BaseModel):
    used_gb: float
    total_gb: float
    percent: float
    available: bool


class StatusResponse(BaseModel):
    gpu_busy: bool
    model_loaded: bool
    current_task: Optional[str]
    current_phase: Optional[str] = None
    elapsed_seconds: float
    has_input_image: bool = False
    last_generated_video: Optional[str] = None
    generation_completed: bool = False
    newly_generated_video: Optional[str] = None
    gpu_memory: Optional[GpuMemoryInfo] = None


class GenerateResponse(BaseModel):
    status: str
    message: Optional[str] = None


class VideoListResponse(BaseModel):
    videos: List[str]


class UploadResponse(BaseModel):
    status: str
    message: Optional[str] = None
    filename: Optional[str] = None


class ClearResponse(BaseModel):
    status: str
    cleared_count: int = 0


class LogsResponse(BaseModel):
    logs: str


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


class PresetImagesResponse(BaseModel):
    images: List[dict]


class LastFrameResponse(BaseModel):
    status: str
    frame_path: Optional[str] = None
    message: Optional[str] = None
    is_video_input: bool = False
    original_video_filename: Optional[str] = None
