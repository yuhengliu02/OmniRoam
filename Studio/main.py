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

import asyncio
import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse

from models import (
    GenerateRequest,
    GenerateResponse,
    StatusResponse,
    VideoListResponse,
    UploadResponse,
    ClearResponse,
    LogsResponse,
    ErrorResponse,
    StageEnum,
    TrajectoryEnum,
    LastFrameResponse,
    PresetImagesResponse,
)
from state import get_state, RuntimeState
from app_utils.video import (
    validate_erp_image,
    validate_erp_image_bytes,
    validate_erp_video,
    validate_erp_video_bytes,
    extract_last_frame,
    get_video_info,
)
from app_utils.logging import get_logger, get_log_buffer
from app_utils.time import get_timestamp_filename, format_duration


BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
UPLOADS_DIR = BASE_DIR / "uploads"
PRESETS_DIR = BASE_DIR / "presets"

OMNIROAM_ROOT = BASE_DIR.parent
PREVIEW_CKPT_PATH = os.environ.get(
    "OMNIROAM_PREVIEW_CKPT",
    str(OMNIROAM_ROOT / "models" / "OmniRoam" / "Preview" / "preview.ckpt")
)
SELF_FORCING_CKPT_PATH = os.environ.get(
    "OMNIROAM_SELF_FORCING_CKPT",
    str(OMNIROAM_ROOT / "models" / "OmniRoam" / "Self-forcing" / "self-forcing.pt")
)
REFINE_CKPT_PATH = os.environ.get(
    "OMNIROAM_REFINE_CKPT",
    str(OMNIROAM_ROOT / "models" / "OmniRoam" / "Refine" / "refine.ckpt")
)

TEMP_DIR = BASE_DIR / "temp_generated"

# Device
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "cuda:0")
if not DEVICE.startswith("cuda"):
    DEVICE = f"cuda:{DEVICE}" if DEVICE.isdigit() else "cuda:0"

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 960
DEFAULT_NUM_FRAMES = 81
DEFAULT_CFG_SCALE = 5.0
DEFAULT_INFERENCE_STEPS = 50

# Logger
logger = get_logger("omniroam")



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== OmniRoam Interactive System Starting ===")
    
    for d in [OUTPUTS_DIR, LOGS_DIR, UPLOADS_DIR, PRESETS_DIR, TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {d}")
    
    state = get_state()
    
    try:
        logger.info(f"Loading Preview model from: {PREVIEW_CKPT_PATH}")
        logger.info(f"Device: {DEVICE}")
        
        from inference.preview import PreviewInference
        
        preview_model = PreviewInference(device=DEVICE)
        success = preview_model.load_model(PREVIEW_CKPT_PATH)
        
        if success:
            state.register_model("preview", preview_model)
            state.models_loaded = True
            logger.info("Preview model loaded successfully!")
        else:
            logger.error("Failed to load Preview model!")
            state.log_error("Failed to load Preview model at startup")
            
    except Exception as e:
        error_msg = f"Preview model loading error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        state.log_error(error_msg)
    
    logger.info("Self-Forcing model will be loaded on-demand when selected")
    
    logger.info("=== Startup Complete ===")
    
    yield
    
    logger.info("=== Shutting Down ===")
    
    preview_model = state.get_model("preview")
    if preview_model:
        preview_model.unload_model()
    
    self_forcing_model = state.get_model("self_forcing")
    if self_forcing_model:
        self_forcing_model.unload_model()
    
    refine_model = state.get_model("refine")
    if refine_model:
        refine_model.unload_model()
    
    log_buffer = get_log_buffer()
    log_file = LOGS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_buffer.save_to_file(log_file)
    
    logger.info("=== Shutdown Complete ===")


app = FastAPI(
    title="OmniRoam Interactive System",
    description="Single-user, single-GPU video generation backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_video_list() -> List[str]:
    videos = []
    if OUTPUTS_DIR.exists():
        for f in OUTPUTS_DIR.glob("*.mp4"):
            videos.append(f.name)
    videos.sort(reverse=True)
    return videos


def save_video_array(video_array: np.ndarray, output_path: Path, fps: int = 30) -> bool:
    try:
        import imageio.v2 as imageio
        
        imageio.mimsave(
            str(output_path),
            video_array,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            quality=8,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save video: {e}")
        return False


async def run_generation_task(
    state: RuntimeState,
    input_path: Path,
    output_path: Path,
    stage: StageEnum,
    trajectory: TrajectoryEnum,
    scale: float,
    num_frames: int,
):
    try:
        logger.info(f"Starting generation: stage={stage}, traj={trajectory}, scale={scale}")
        
        if stage == StageEnum.preview:
            model = state.get_model("preview")
        elif stage == StageEnum.self_forcing:
            model = state.get_model("self_forcing")
        elif stage == StageEnum.refine:
            model = state.get_model("refine")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        if model is None:
            raise RuntimeError(f"Model for stage '{stage}' not loaded")
        
        if stage == StageEnum.refine:
            num_segments = int(scale)
            
            video_input_path = state.source_video_path if state.source_video_path else input_path
            
            logger.info(f"Refine mode: segments={num_segments}, video_input={video_input_path}")
            
            import uuid
            gen_id = uuid.uuid4().hex[:8]
            temp_gen_dir = TEMP_DIR / f"refine_{gen_id}"
            temp_gen_dir.mkdir(parents=True, exist_ok=True)
            
            success, video_array, message = model.generate(
                input_video_path=video_input_path,
                num_segments=num_segments,
                trajectory=trajectory.value,
                scale=1.0,
                height=720,
                width=1440,
                num_frames=81,
                cfg_scale=DEFAULT_CFG_SCALE,
                num_inference_steps=DEFAULT_INFERENCE_STEPS,
                temp_output_dir=temp_gen_dir,
            )
            
            try:
                import shutil
                shutil.rmtree(temp_gen_dir)
            except Exception:
                pass
            
        else:
            success, video_array, message = model.generate(
                input_image_path=input_path,
                trajectory=trajectory.value,
                scale=scale,
                height=DEFAULT_HEIGHT,
                width=DEFAULT_WIDTH,
                num_frames=num_frames,
                cfg_scale=DEFAULT_CFG_SCALE,
                num_inference_steps=DEFAULT_INFERENCE_STEPS,
            )
        
        if not success:
            raise RuntimeError(message)
        
        state.set_phase("saving", pending_video=output_path.name)
        logger.info("Generation complete, saving video...")
        
        if not save_video_array(video_array, output_path):
            raise RuntimeError("Failed to save video")
        
        last_frame_path = UPLOADS_DIR / f"last_frame_{output_path.stem}.png"
        extract_success, extract_msg = extract_last_frame(output_path, last_frame_path)
        
        if extract_success:
            state.last_frame_path = last_frame_path
            logger.info(f"Last frame saved: {last_frame_path}")
        else:
            logger.warning(f"Failed to extract last frame: {extract_msg}")
        
        state.last_generated_video = output_path
        
        await asyncio.sleep(2)
        
        logger.info(f"Video saved: {output_path}")
        
        state.release_gpu(completed_video=output_path.name)
        return
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        state.log_error(error_msg)
        state.release_gpu()  # Release without completion notification on error


@app.get("/")
async def root():
    return {"status": "ok", "service": "OmniRoam Interactive System"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    state = get_state()
    status = state.get_status_dict()
    return StatusResponse(**status)


@app.post("/acknowledge_completion")
async def acknowledge_completion():
    state = get_state()
    video = state.acknowledge_completion()
    return {"acknowledged": True, "video": video}


@app.post("/upload/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    state = get_state()
    
    content = await file.read()
    filename_lower = file.filename.lower() if file.filename else ""
    
    video_extensions = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    file_ext = Path(filename_lower).suffix
    is_video = file_ext in video_extensions
    is_image = file_ext in image_extensions
    
    if not is_video and not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: images ({image_extensions}) or videos ({video_extensions})"
        )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_video:
        video_filename = f"upload_{timestamp}_{file.filename}"
        video_path = UPLOADS_DIR / video_filename
        
        with open(video_path, "wb") as f:
            f.write(content)
        
        valid, msg, dims = validate_erp_video(video_path)
        if not valid:
            video_path.unlink(missing_ok=True)
            logger.warning(f"Invalid video upload: {msg}")
            raise HTTPException(status_code=400, detail=msg)
        
        frame_filename = f"upload_{timestamp}_lastframe.png"
        frame_path = UPLOADS_DIR / frame_filename
        
        success, extract_msg = extract_last_frame(video_path, frame_path)
        if not success:
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to extract last frame: {extract_msg}")
        
        state.set_input_image(frame_path)
        
        state.source_video_path = video_path
        
        logger.info(f"Video uploaded: {video_filename} ({dims[0]}x{dims[1]}), using last frame as input")
        return UploadResponse(
            status="success",
            message=f"Video uploaded ({dims[0]}x{dims[1]}). Using last frame as input.",
            filename=frame_filename,
        )
    else:
        valid, msg, dims = validate_erp_image_bytes(content)
        if not valid:
            logger.warning(f"Invalid upload: {msg}")
            raise HTTPException(status_code=400, detail=msg)
        
        filename = f"upload_{timestamp}_{file.filename}"
        save_path = UPLOADS_DIR / filename
        
        with open(save_path, "wb") as f:
            f.write(content)
        
        state.set_input_image(save_path)
        state.source_video_path = None  # Clear any previous source video
        
        logger.info(f"Image uploaded: {filename} ({dims[0]}x{dims[1]})")
        return UploadResponse(
            status="success",
            message=f"Image uploaded successfully ({dims[0]}x{dims[1]})",
            filename=filename,
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    state = get_state()
    
    if state.current_input_image_path is None:
        raise HTTPException(
            status_code=400,
            detail="No input image set. Please upload an image first."
        )
    
    if not state.current_input_image_path.exists():
        state.clear_input_image()
        raise HTTPException(
            status_code=400,
            detail="Input image file not found. Please upload again."
        )
    
    if not state.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please wait for server initialization."
        )
    
    if not state.acquire_gpu("generate"):
        raise HTTPException(
            status_code=409,
            detail="GPU is busy. Please wait for current generation to complete."
        )
    
    output_filename = get_timestamp_filename("mp4")
    output_path = OUTPUTS_DIR / output_filename
    
    background_tasks.add_task(
        run_generation_task,
        state=state,
        input_path=state.current_input_image_path,
        output_path=output_path,
        stage=request.stage,
        trajectory=request.trajectory,
        scale=request.scale,
        num_frames=request.num_frames,
    )
    
    logger.info(f"Generation started: {output_filename}")
    return GenerateResponse(status="started", message=f"Generating: {output_filename}")


@app.get("/videos", response_model=VideoListResponse)
async def list_videos():
    videos = get_video_list()
    return VideoListResponse(videos=videos)


@app.get("/video/{filename}")
async def get_video(filename: str):
    video_path = OUTPUTS_DIR / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename,
    )


@app.post("/clear_all", response_model=ClearResponse)
async def clear_all():
    state = get_state()
    
    if state.gpu_busy:
        raise HTTPException(
            status_code=409,
            detail="Cannot clear while generation is in progress."
        )
    
    cleared_count = 0
    if OUTPUTS_DIR.exists():
        for f in OUTPUTS_DIR.glob("*.mp4"):
            try:
                f.unlink()
                cleared_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
    
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.glob("*"):
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
    
    state.clear_input_image()
    state.last_generated_video = None
    state.last_frame_path = None
    
    logger.info(f"Cleared {cleared_count} videos")
    return ClearResponse(status="success", cleared_count=cleared_count)


@app.get("/logs")
async def get_logs():
    log_buffer = get_log_buffer()
    logs = log_buffer.get_all()
    return PlainTextResponse(logs)


@app.get("/logs/download")
async def download_logs():
    log_buffer = get_log_buffer()
    temp_path = LOGS_DIR / f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_buffer.save_to_file(temp_path)
    
    return FileResponse(
        temp_path,
        media_type="text/plain",
        filename=temp_path.name,
    )


@app.post("/use_last_frame", response_model=LastFrameResponse)
async def use_last_frame():
    state = get_state()
    
    if state.last_generated_video is None:
        raise HTTPException(
            status_code=400,
            detail="No generated video available."
        )
    
    if state.last_frame_path is None or not state.last_frame_path.exists():
        last_frame_path = UPLOADS_DIR / f"last_frame_{state.last_generated_video.stem}.png"
        success, msg = extract_last_frame(state.last_generated_video, last_frame_path)
        
        if not success:
            raise HTTPException(status_code=500, detail=msg)
        
        state.last_frame_path = last_frame_path
    
    state.set_input_image(state.last_frame_path)
    
    logger.info(f"Using last frame as input: {state.last_frame_path}")
    return LastFrameResponse(
        status="success",
        frame_path=str(state.last_frame_path.name),
        message="Last frame set as input image."
    )


@app.post("/use_gallery_video/{video_filename}", response_model=LastFrameResponse)
async def use_gallery_video(video_filename: str, model: str = "preview"):
    state = get_state()
    
    if "/" in video_filename or "\\" in video_filename or ".." in video_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    video_path = OUTPUTS_DIR / video_filename
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_filename}"
        )
    
    valid, msg, dims = validate_erp_video(video_path)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    
    if model == "refine":
        video_info = get_video_info(video_path)
        if video_info and video_info.get("frame_count") != 81:
            raise HTTPException(
                status_code=400,
                detail=f"Refine mode requires exactly 81 frames. This video has {video_info.get('frame_count', 'unknown')} frames."
            )
    
    frame_filename = f"gallery_{video_path.stem}_lastframe.png"
    frame_path = UPLOADS_DIR / frame_filename
    
    success, extract_msg = extract_last_frame(video_path, frame_path)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to extract last frame: {extract_msg}")
    
    if model == "refine":
        state.set_input_image(frame_path, is_video_input=True, original_video_path=video_path)
        state.source_video_path = video_path
        logger.info(f"Using gallery video '{video_filename}' as refine input (full video)")
        return LastFrameResponse(
            status="success",
            frame_path=frame_filename,
            message=f"Using full video '{video_filename}' as refine input.",
            is_video_input=True,
            original_video_filename=video_filename
        )
    else:
        state.set_input_image(frame_path, is_video_input=False, original_video_path=video_path)
        state.source_video_path = video_path
        logger.info(f"Using gallery video '{video_filename}' last frame as input")
        return LastFrameResponse(
            status="success",
            frame_path=frame_filename,
            message=f"Using last frame from '{video_filename}' as input."
        )


@app.get("/preset_images", response_model=PresetImagesResponse)
async def get_preset_images():
    images = []
    if PRESETS_DIR.exists():
        for f in PRESETS_DIR.glob("*.png"):
            images.append({"name": f.stem, "path": f.name})
        for f in PRESETS_DIR.glob("*.jpg"):
            images.append({"name": f.stem, "path": f.name})
    
    return PresetImagesResponse(images=images)


@app.post("/select_preset/{filename}", response_model=UploadResponse)
async def select_preset(filename: str):
    state = get_state()
    
    preset_path = PRESETS_DIR / filename
    
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail="Preset image not found")
    
    valid, msg = validate_erp_image(preset_path)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    
    state.set_input_image(preset_path)
    
    logger.info(f"Preset selected: {filename}")
    return UploadResponse(
        status="success",
        message=f"Preset '{filename}' selected as input.",
        filename=filename,
    )


@app.get("/current_input")
async def get_current_input():
    state = get_state()
    
    if state.current_input_image_path is None:
        return {"has_input": False, "filename": None}
    
    return {
        "has_input": True,
        "filename": state.current_input_image_path.name,
        "is_preset": str(state.current_input_image_path).startswith(str(PRESETS_DIR)),
    }


@app.get("/input_image/{filename}")
@app.head("/input_image/{filename}")
async def get_input_image(filename: str):
    upload_path = UPLOADS_DIR / filename
    if upload_path.exists():
        return FileResponse(upload_path)
    
    preset_path = PRESETS_DIR / filename
    if preset_path.exists():
        return FileResponse(preset_path)
    
    raise HTTPException(status_code=404, detail="Image not found")


VRAM_THRESHOLD_PREVIEW_SF_GB = 75.0
VRAM_THRESHOLD_REFINE_GB = 150.0


def get_total_vram_gb() -> float:
    """Get total GPU VRAM in GB."""
    try:
        from state import get_gpu_memory_info
        gpu_info = get_gpu_memory_info()
        return gpu_info.get('total_gb', 0.0)
    except Exception:
        return 0.0


def should_offload_for_new_model(target_model: str = None) -> bool:
    total_vram = get_total_vram_gb()
    
    if target_model == "refine":
        return total_vram < VRAM_THRESHOLD_REFINE_GB
    else:
        return total_vram < VRAM_THRESHOLD_PREVIEW_SF_GB


async def _load_preview_model(state) -> bool:
    from inference.preview import PreviewInference
    
    logger.info("Loading Preview model...")
    preview_model = PreviewInference(device=DEVICE)
    success = preview_model.load_model(PREVIEW_CKPT_PATH)
    
    if success:
        state.register_model("preview", preview_model)
        state.models_loaded = True
        logger.info("Preview model loaded successfully!")
        return True
    else:
        logger.error("Failed to load Preview model!")
        return False


async def _load_self_forcing_model(state) -> bool:
    from inference.self_forcing import SelfForcingInference
    
    logger.info("Loading Self-Forcing model...")
    self_forcing_model = SelfForcingInference(device=DEVICE)
    
    try:
        success = self_forcing_model.load_model(SELF_FORCING_CKPT_PATH)
        
        if success:
            state.register_model("self_forcing", self_forcing_model)
            logger.info("Self-Forcing model loaded successfully!")
            return True
        else:
            logger.error("Failed to load Self-Forcing model!")
            state.set_model_load_failed(
                "self_forcing",
                "Model loading returned False",
                fallback=True
            )
            return False
    except Exception as e:
        logger.error(f"Exception while loading Self-Forcing model: {str(e)}")
        state.set_model_load_failed(
            "self_forcing",
            str(e),
            fallback=True
        )
        return False


async def _unload_preview_model(state) -> None:
    preview_model = state.get_model("preview")
    if preview_model:
        logger.info("Unloading Preview model to free VRAM...")
        preview_model.unload_model()
        state.unregister_model("preview")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Preview model unloaded")


async def _unload_self_forcing_model(state) -> None:
    sf_model = state.get_model("self_forcing")
    if sf_model:
        logger.info("Unloading Self-Forcing model to free VRAM...")
        sf_model.unload_model()
        state.unregister_model("self_forcing")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Self-Forcing model unloaded")


async def _load_refine_model(state) -> bool:
    from inference.refine import RefineInference
    
    logger.info("Loading Refine model...")
    refine_model = RefineInference(device=DEVICE)
    
    try:
        success = refine_model.load_model(REFINE_CKPT_PATH)
        
        if success:
            state.register_model("refine", refine_model)
            logger.info("Refine model loaded successfully!")
            return True
        else:
            logger.error("Failed to load Refine model!")
            state.set_model_load_failed(
                "refine",
                "Model loading returned False",
                fallback=True
            )
            return False
    except Exception as e:
        logger.error(f"Exception while loading Refine model: {str(e)}")
        state.set_model_load_failed(
            "refine",
            str(e),
            fallback=True
        )
        return False


async def _unload_refine_model(state) -> None:
    """Unload refine model to free VRAM."""
    refine_model = state.get_model("refine")
    if refine_model:
        logger.info("Unloading Refine model to free VRAM...")
        refine_model.unload_model()
        state.unregister_model("refine")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Refine model unloaded")


async def _unload_all_models(state) -> None:
    """Unload all loaded models to free VRAM."""
    await _unload_preview_model(state)
    await _unload_self_forcing_model(state)
    await _unload_refine_model(state)


@app.post("/switch_model/{model_name}")
async def switch_model(model_name: str, background_tasks: BackgroundTasks):
    state = get_state()
    
    if model_name not in ["preview", "self_forcing", "refine"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    if state.gpu_busy:
        raise HTTPException(
            status_code=409,
            detail="Cannot switch models while generation is in progress."
        )
    
    if state.model_loading:
        raise HTTPException(
            status_code=409,
            detail=f"Already loading model: {state.model_loading_name}"
        )
    
    background_tasks.add_task(load_model_task, model_name)
    
    state.set_model_loading(True, model_name)
    
    return {
        "status": "loading",
        "message": f"Loading {model_name} model...",
        "model_name": model_name,
    }


async def load_model_task(model_name: str):
    state = get_state()
    
    try:
        state.clear_model_failure()
        
        total_vram = get_total_vram_gb()
        
        logger.info(f"=== Model Loading Request: {model_name} ===")
        logger.info(f"Current state - Preview: {state.preview_loaded}, Self-Forcing: {state.self_forcing_loaded}, Refine: {state.refine_loaded}")
        logger.info(f"Total VRAM: {total_vram:.1f} GB")
        
        model_already_loaded = False
        if model_name == "preview" and state.preview_loaded:
            logger.info("Preview model already loaded, skipping load")
            model_already_loaded = True
        elif model_name == "self_forcing" and state.self_forcing_loaded:
            logger.info("Self-Forcing model already loaded, skipping load")
            model_already_loaded = True
        elif model_name == "refine" and state.refine_loaded:
            logger.info("Refine model already loaded, skipping load")
            model_already_loaded = True
        
        if model_already_loaded:
            state.set_model_loading(False, None)
            return
        
        if total_vram >= VRAM_THRESHOLD_REFINE_GB:
            logger.info(f"VRAM >= 150GB: Loading {model_name} without offloading")
            if model_name == "preview":
                success = await _load_preview_model(state)
            elif model_name == "self_forcing":
                success = await _load_self_forcing_model(state)
            else:
                success = await _load_refine_model(state)
            
            if not success:
                logger.error(f"Failed to load {model_name} model")
                state.set_model_load_failed(model_name, f"Failed to load {model_name}", fallback=False)
        
        elif total_vram >= VRAM_THRESHOLD_PREVIEW_SF_GB:
            logger.info(f"VRAM scenario: 75GB <= VRAM < 150GB")
            
            if model_name == "refine":
                logger.info(f"Loading Refine: Offloading all models first")
                await _unload_all_models(state)
                logger.info(f"After offload - Preview: {state.preview_loaded}, Self-Forcing: {state.self_forcing_loaded}, Refine: {state.refine_loaded}")
                
                success = await _load_refine_model(state)
                
                if not success:
                    logger.error("Failed to load Refine model, falling back to Preview")
                    fallback_success = await _load_preview_model(state)
                    if fallback_success:
                        logger.info("Successfully fell back to Preview model")
                    else:
                        logger.critical("Failed to load Preview model as fallback!")
                        state.set_model_load_failed(
                            "preview",
                            "Preview fallback failed - system unusable",
                            fallback=False
                        )
            
            else:
                if state.refine_loaded:
                    logger.info(f"Refine is loaded, offloading before loading {model_name}")
                    await _unload_refine_model(state)
                    logger.info(f"After offloading Refine - Preview: {state.preview_loaded}, Self-Forcing: {state.self_forcing_loaded}, Refine: {state.refine_loaded}")
                
                logger.info(f"Loading {model_name} model...")
                if model_name == "preview":
                    success = await _load_preview_model(state)
                else:
                    success = await _load_self_forcing_model(state)
                
                if not success:
                    logger.error(f"Failed to load {model_name} model, falling back to Preview")
                    if model_name != "preview" and not state.preview_loaded:
                        fallback_success = await _load_preview_model(state)
                        if not fallback_success:
                            logger.critical("Failed to load Preview model as fallback!")
                            state.set_model_load_failed(
                                "preview",
                                "Preview fallback failed - system unusable",
                                fallback=False
                            )
                else:
                    logger.info(f"Successfully loaded {model_name} model")
        
        else:
            logger.info(f"VRAM < 75GB: Only one model allowed")
            
            if model_name == "refine":
                logger.info("Offloading all models before loading Refine")
                await _unload_all_models(state)
                success = await _load_refine_model(state)
                
                if not success:
                    logger.error("Failed to load Refine model, falling back to Preview")
                    fallback_success = await _load_preview_model(state)
                    if fallback_success:
                        logger.info("Successfully fell back to Preview model")
                    else:
                        logger.critical("Failed to load Preview model as fallback!")
                        state.set_model_load_failed(
                            "preview",
                            "Preview fallback failed - system unusable",
                            fallback=False
                        )
            
            elif model_name == "preview":
                if state.self_forcing_loaded:
                    logger.info("Offloading Self-Forcing before loading Preview")
                    await _unload_self_forcing_model(state)
                if state.refine_loaded:
                    logger.info("Offloading Refine before loading Preview")
                    await _unload_refine_model(state)
                
                logger.info(f"After offload - Preview: {state.preview_loaded}, Self-Forcing: {state.self_forcing_loaded}, Refine: {state.refine_loaded}")
                logger.info(f"Loading Preview model...")
                success = await _load_preview_model(state)
                
                if not success:
                    logger.critical("Failed to load Preview model - system unusable!")
                    state.set_model_load_failed(
                        "preview",
                        "Preview model failed to load - system unusable",
                        fallback=False
                    )
                else:
                    logger.info(f"Successfully loaded Preview model")
            
            else:
                if state.preview_loaded:
                    logger.info("Offloading Preview before loading Self-Forcing")
                    await _unload_preview_model(state)
                if state.refine_loaded:
                    logger.info("Offloading Refine before loading Self-Forcing")
                    await _unload_refine_model(state)
                
                success = await _load_self_forcing_model(state)
                
                if not success:
                    logger.error("Failed to load Self-Forcing model, falling back to Preview")
                    fallback_success = await _load_preview_model(state)
                    if fallback_success:
                        logger.info("Successfully fell back to Preview model")
                    else:
                        logger.critical("Failed to load Preview model as fallback!")
                        state.set_model_load_failed(
                            "preview",
                            "Preview fallback failed - system unusable",
                            fallback=False
                        )
        
        logger.info(f"=== Final state after loading - Preview: {state.preview_loaded}, Self-Forcing: {state.self_forcing_loaded}, Refine: {state.refine_loaded} ===")
    
    except Exception as e:
        error_msg = f"Model loading error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        state.log_error(error_msg)
        state.set_model_load_failed(model_name, str(e), fallback=False)
    
    finally:
        state.set_model_loading(False, None)


@app.get("/model_status")
async def get_model_status():
    """
    Get detailed model loading status.
    """
    state = get_state()
    total_vram = get_total_vram_gb()
    
    return {
        "preview_loaded": state.preview_loaded,
        "self_forcing_loaded": state.self_forcing_loaded,
        "refine_loaded": state.refine_loaded,
        "model_loading": state.model_loading,
        "model_loading_name": state.model_loading_name,
        "total_vram_gb": total_vram,
        "vram_threshold_preview_sf_gb": VRAM_THRESHOLD_PREVIEW_SF_GB,
        "vram_threshold_refine_gb": VRAM_THRESHOLD_REFINE_GB,
        "can_load_preview_sf": total_vram >= VRAM_THRESHOLD_PREVIEW_SF_GB,
        "can_load_refine_with_others": total_vram >= VRAM_THRESHOLD_REFINE_GB,
    }



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    error_msg = f"Unhandled error: {str(exc)}\n{traceback.format_exc()}"
    logger.error(error_msg)
    get_state().log_error(error_msg)
    
    return {"status": "error", "message": str(exc)}



if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )

