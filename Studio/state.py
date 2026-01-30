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

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict
import torch

_pynvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_available = True
except Exception:
    pass


def get_gpu_memory_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            'used_gb': 0.0,
            'total_gb': 0.0,
            'percent': 0.0,
            'available': False,
        }
    
    if _pynvml_available:
        try:
            device_index = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            used_gb = mem_info.used / (1024 ** 3)
            total_gb = mem_info.total / (1024 ** 3)
            percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0.0
            
            return {
                'used_gb': round(used_gb, 2),
                'total_gb': round(total_gb, 2),
                'percent': round(percent, 1),
                'available': True,
            }
        except Exception:
            pass
    
    try:
        device = torch.cuda.current_device()
        used = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        
        used_gb = used / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        percent = (used / total * 100) if total > 0 else 0.0
        
        return {
            'used_gb': round(used_gb, 2),
            'total_gb': round(total_gb, 2),
            'percent': round(percent, 1),
            'available': True,
        }
    except Exception:
        return {
            'used_gb': 0.0,
            'total_gb': 0.0,
            'percent': 0.0,
            'available': False,
        }


@dataclass
class RuntimeState:
    
    current_input_image_path: Optional[Path] = None
    
    gpu_busy: bool = False
    current_task: Optional[str] = None
    current_task_start_time: Optional[float] = None
    current_phase: Optional[str] = None
    
    loaded_models: dict = field(default_factory=dict)
    models_loaded: bool = False
    
    model_loading: bool = False
    model_loading_name: Optional[str] = None
    preview_loaded: bool = False
    self_forcing_loaded: bool = False
    refine_loaded: bool = False
    
    model_load_failed: bool = False
    failed_model_name: Optional[str] = None
    fallback_to_preview: bool = False
    model_error_message: Optional[str] = None
    
    error_log: List[str] = field(default_factory=list)
    
    last_generated_video: Optional[Path] = None
    last_frame_path: Optional[Path] = None
    
    source_video_path: Optional[Path] = None
    
    is_video_input: bool = False
    
    generation_completed: bool = False
    newly_generated_video: Optional[str] = None
    
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def acquire_gpu(self, task_name: str) -> bool:
        with self._lock:
            if self.gpu_busy:
                return False
            self.gpu_busy = True
            self.current_task = task_name
            self.current_task_start_time = time.time()
            self.current_phase = "generating"
            return True
    
    def set_phase(self, phase: str, pending_video: Optional[str] = None) -> None:
        with self._lock:
            self.current_phase = phase
            if pending_video:
                self.newly_generated_video = pending_video
    
    def release_gpu(self, completed_video: Optional[str] = None) -> None:
        with self._lock:
            self.gpu_busy = False
            self.current_task = None
            self.current_task_start_time = None
            self.current_phase = None
            if completed_video:
                self.generation_completed = True
                self.newly_generated_video = completed_video
    
    def acknowledge_completion(self) -> Optional[str]:
        with self._lock:
            video = self.newly_generated_video
            self.generation_completed = False
            self.newly_generated_video = None
            return video
    
    def get_elapsed_seconds(self) -> float:
        if self.current_task_start_time is None:
            return 0.0
        return time.time() - self.current_task_start_time
    
    def log_error(self, error_msg: str) -> None:
        timestamp = datetime.now().isoformat()
        self.error_log.append(f"[{timestamp}] {error_msg}")
    
    def get_logs(self) -> str:
        return "\n".join(self.error_log)
    
    def clear_logs(self) -> None:
        self.error_log = []
    
    def set_input_image(
        self, 
        path: Path, 
        is_video_input: bool = False, 
        original_video_path: Optional[Path] = None
    ) -> None:
        self.current_input_image_path = path
        self.is_video_input = is_video_input
        if original_video_path:
            self.source_video_path = original_video_path
    
    def clear_input_image(self) -> None:
        self.current_input_image_path = None
        self.is_video_input = False
    
    def register_model(self, name: str, model: Any) -> None:
        self.loaded_models[name] = model
        if name == "preview":
            self.preview_loaded = True
        elif name == "self_forcing":
            self.self_forcing_loaded = True
        elif name == "refine":
            self.refine_loaded = True
    
    def unregister_model(self, name: str) -> None:
        if name in self.loaded_models:
            del self.loaded_models[name]
        if name == "preview":
            self.preview_loaded = False
        elif name == "self_forcing":
            self.self_forcing_loaded = False
        elif name == "refine":
            self.refine_loaded = False
    
    def get_model(self, name: str) -> Optional[Any]:
        return self.loaded_models.get(name)
    
    def set_model_loading(self, loading: bool, model_name: Optional[str] = None) -> None:
        with self._lock:
            self.model_loading = loading
            self.model_loading_name = model_name if loading else None
    
    def set_model_load_failed(self, model_name: str, error_msg: str, fallback: bool = False) -> None:
        with self._lock:
            self.model_load_failed = True
            self.failed_model_name = model_name
            self.model_error_message = error_msg
            self.fallback_to_preview = fallback
            self.error_log.append(f"[{model_name}] Load failed: {error_msg}")
    
    def clear_model_failure(self) -> None:
        with self._lock:
            self.model_load_failed = False
            self.failed_model_name = None
            self.model_error_message = None
            self.fallback_to_preview = False
    
    def get_status_dict(self) -> dict:
        gpu_memory = get_gpu_memory_info()
        
        return {
            "gpu_busy": self.gpu_busy,
            "model_loaded": self.models_loaded,
            "current_task": self.current_task,
            "current_phase": self.current_phase,
            "elapsed_seconds": round(self.get_elapsed_seconds(), 2),
            "has_input_image": self.current_input_image_path is not None,
            "last_generated_video": str(self.last_generated_video.name) if self.last_generated_video else None,
            "generation_completed": self.generation_completed,
            "newly_generated_video": self.newly_generated_video,
            "gpu_memory": gpu_memory,
            "model_loading": self.model_loading,
            "model_loading_name": self.model_loading_name,
            "preview_loaded": self.preview_loaded,
            "self_forcing_loaded": self.self_forcing_loaded,
            "refine_loaded": self.refine_loaded,
            "model_load_failed": self.model_load_failed,
            "failed_model_name": self.failed_model_name,
            "fallback_to_preview": self.fallback_to_preview,
            "model_error_message": self.model_error_message,
        }


_state: Optional[RuntimeState] = None


def get_state() -> RuntimeState:
    global _state
    if _state is None:
        _state = RuntimeState()
    return _state


def reset_state() -> None:
    global _state
    _state = RuntimeState()
