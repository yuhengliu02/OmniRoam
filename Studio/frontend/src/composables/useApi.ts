/**
 * API composable for OmniRoam backend communication.
 */

import { ref } from 'vue'

// Base URL - can be configured via environment variable
const API_BASE = import.meta.env.VITE_API_BASE || '/api'

// Types
export interface GpuMemoryInfo {
  used_gb: number
  total_gb: number
  percent: number
  available: boolean
}

export interface SystemStatus {
  gpu_busy: boolean
  model_loaded: boolean
  current_task: string | null
  current_phase: 'generating' | 'saving' | null
  elapsed_seconds: number
  has_input_image: boolean
  last_generated_video: string | null
  generation_completed: boolean
  newly_generated_video: string | null
  gpu_memory: GpuMemoryInfo | null
  // Model loading state
  model_loading: boolean
  model_loading_name: string | null
  preview_loaded: boolean
  self_forcing_loaded: boolean
  refine_loaded: boolean
  model_load_failed: boolean
  failed_model_name: string | null
  fallback_to_preview: boolean
  model_error_message: string | null
}

export interface ModelStatus {
  preview_loaded: boolean
  self_forcing_loaded: boolean
  refine_loaded: boolean
  model_loading: boolean
  model_loading_name: string | null
  total_vram_gb: number
  vram_threshold_preview_sf_gb: number
  vram_threshold_refine_gb: number
  can_load_preview_sf: boolean
  can_load_refine_with_others: boolean
}

export interface GenerateRequest {
  stage: 'preview' | 'self_forcing' | 'refine'
  trajectory: 'forward' | 'right' | 'backward' | 'left' | 'up' | 'down' | 's_curve' | 'zigzag_forward' | 'loop'
  scale: number
  num_frames: number
}

export interface UploadResponse {
  status: string
  message?: string
  filename?: string
}

export interface VideoListResponse {
  videos: string[]
}

export interface PresetImage {
  name: string
  path: string
}

// API functions
export function useApi() {
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }
    
    return response.json()
  }

  // Get system status
  async function getStatus(): Promise<SystemStatus> {
    return fetchJson<SystemStatus>('/status')
  }

  // Upload image
  async function uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch(`${API_BASE}/upload/image`, {
      method: 'POST',
      body: formData,
    })
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }
    
    return response.json()
  }

  // Start generation
  async function startGeneration(request: GenerateRequest): Promise<{ status: string; message?: string }> {
    return fetchJson('/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  // Get video list
  async function getVideos(): Promise<VideoListResponse> {
    return fetchJson<VideoListResponse>('/videos')
  }

  // Get video URL
  function getVideoUrl(filename: string): string {
    return `${API_BASE}/video/${filename}`
  }

  // Clear all
  async function clearAll(): Promise<{ status: string; cleared_count: number }> {
    return fetchJson('/clear_all', { method: 'POST' })
  }

  // Get logs
  async function getLogs(): Promise<string> {
    const response = await fetch(`${API_BASE}/logs`)
    return response.text()
  }

  // Download logs
  function getLogsDownloadUrl(): string {
    return `${API_BASE}/logs/download`
  }

  // Use last frame as input
  async function useLastFrame(): Promise<{ status: string; frame_path?: string; message?: string }> {
    return fetchJson('/use_last_frame', { method: 'POST' })
  }

  // Use gallery video as input (last frame for preview/self_forcing, full video for refine)
  async function useGalleryVideo(filename: string, model: string = 'preview'): Promise<{ 
    status: string; 
    frame_path?: string; 
    message?: string; 
    is_video_input?: boolean; 
    original_video_filename?: string 
  }> {
    return fetchJson(`/use_gallery_video/${encodeURIComponent(filename)}?model=${encodeURIComponent(model)}`, { method: 'POST' })
  }

  // Get preset images
  async function getPresetImages(): Promise<{ images: PresetImage[] }> {
    return fetchJson<{ images: PresetImage[] }>('/preset_images')
  }

  // Select preset image
  async function selectPreset(filename: string): Promise<UploadResponse> {
    return fetchJson(`/select_preset/${filename}`, { method: 'POST' })
  }

  // Get current input info
  async function getCurrentInput(): Promise<{ has_input: boolean; filename?: string; is_preset?: boolean }> {
    return fetchJson('/current_input')
  }

  // Get input image URL
  function getInputImageUrl(filename: string): string {
    return `${API_BASE}/input_image/${filename}`
  }

  // Acknowledge generation completion
  async function acknowledgeCompletion(): Promise<{ acknowledged: boolean; video: string | null }> {
    return fetchJson('/acknowledge_completion', { method: 'POST' })
  }

  // Switch to a different model
  async function switchModel(modelName: 'preview' | 'self_forcing'): Promise<{ status: string; message?: string; model_name?: string }> {
    return fetchJson(`/switch_model/${modelName}`, { method: 'POST' })
  }

  // Get detailed model status
  async function getModelStatus(): Promise<ModelStatus> {
    return fetchJson<ModelStatus>('/model_status')
  }

  return {
    loading,
    error,
    getStatus,
    uploadImage,
    startGeneration,
    getVideos,
    getVideoUrl,
    clearAll,
    getLogs,
    getLogsDownloadUrl,
    useLastFrame,
    useGalleryVideo,
    getPresetImages,
    selectPreset,
    getCurrentInput,
    getInputImageUrl,
    acknowledgeCompletion,
    switchModel,
    getModelStatus,
  }
}

