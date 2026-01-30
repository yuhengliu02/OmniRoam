/**
 * Application state store for OmniRoam.
 */

import { reactive, computed } from 'vue'
import type { SystemStatus } from '@/composables/useApi'

// Types
export type ViewMode = 'erp' | 'perspective'
export type TrajectoryPreset = 'forward' | 'backward' | 'left' | 'right' | 'up' | 'down' | 's_curve' | 'zigzag_forward' | 'loop'
export type ModelStage = 'preview' | 'self_forcing' | 'refine'

export interface AppState {
  // View mode
  viewMode: ViewMode
  
  // Input image (or thumbnail for video input)
  inputImage: {
    filename: string | null
    url: string | null
    isPreset: boolean
    isVideoInput: boolean  // True if input is a full video (for refine mode)
    originalVideoFilename: string | null  // Original video filename for refine mode
  }
  
  // Source video path (for refine mode - full video, not just last frame)
  sourceVideoFilename: string | null
  
  // Selected trajectory
  selectedTrajectory: TrajectoryPreset | null
  
  // Model stages
  previewStage: 'preview' | 'self_forcing'
  refineStage: ModelStage
  
  // Current selected model (unified for all stages)
  selectedModel: 'preview' | 'self_forcing' | 'refine'
  
  // Scale (or segments for refine mode)
  scale: number
  segments: number  // For refine mode (2-8)
  
  // Generation
  isGenerating: boolean
  isSaving: boolean
  generationElapsed: number
  pendingVideoFilename: string | null  // Video waiting to appear in gallery
  
  // Gallery
  videos: string[]
  activeVideo: string | null
  
  // System status
  systemStatus: SystemStatus | null
  
  // A-Frame camera orientation (for perspective indicator)
  cameraYaw: number
  cameraPitch: number
  
  // Model loading state
  modelLoading: boolean
  modelLoadingName: string | null  // 'preview', 'self_forcing', or 'refine'
  previewLoaded: boolean
  selfForcingLoaded: boolean
  refineLoaded: boolean
  
  // Prevent polling from overwriting manual state updates
  preventPollingUpdate: boolean
  modelLoadingDetected: boolean  // Track if we've seen model_loading = true
  
  // Model failure state
  modelLoadFailed: boolean
  failedModelName: string | null
  fallbackToPreview: boolean
  modelErrorMessage: string | null
}

// Create reactive state
const state = reactive<AppState>({
  viewMode: 'erp',
  inputImage: {
    filename: null,
    url: null,
    isPreset: false,
    isVideoInput: false,
    originalVideoFilename: null,
  },
  sourceVideoFilename: null,
  selectedTrajectory: null,
  previewStage: 'preview',
  refineStage: 'preview',
  selectedModel: 'preview',
  scale: 1.0,
  segments: 8,  // Default 8 segments for refine mode
  isGenerating: false,
  isSaving: false,
  generationElapsed: 0,
  pendingVideoFilename: null,
  videos: [],
  activeVideo: null,
  systemStatus: null,
  cameraYaw: 0,
  cameraPitch: 0,
  // Model loading state
  modelLoading: false,
  modelLoadingName: null,
  previewLoaded: true,  // Preview is loaded by default on startup
  selfForcingLoaded: false,
  refineLoaded: false,
  preventPollingUpdate: false,  // Allow polling by default
  modelLoadingDetected: false,  // Track if we've seen model_loading = true
  // Model failure state
  modelLoadFailed: false,
  failedModelName: null,
  fallbackToPreview: false,
  modelErrorMessage: null,
})

// Generation timer
let generationTimer: number | null = null

// Computed properties
const canGenerate = computed(() => {
  // Refine mode doesn't require trajectory selection
  const needsTrajectory = state.selectedModel !== 'refine'
  
  return (
    state.inputImage.filename !== null &&
    (!needsTrajectory || state.selectedTrajectory !== null) &&
    !state.isGenerating &&
    !state.isSaving &&
    !state.modelLoading &&  // Don't allow generation while model is loading
    state.systemStatus?.model_loaded === true
  )
})

const gpuStatus = computed(() => {
  if (!state.systemStatus) return 'unknown'
  // Show 'working' if locally generating/saving OR server says busy
  if (state.isGenerating || state.isSaving || state.systemStatus.gpu_busy) return 'working'
  return 'idle'
})

const modelStatus = computed(() => {
  if (!state.systemStatus) return 'unknown'
  // If failed, show 'failed' status
  if (state.modelLoadFailed) return 'failed'
  // If loading, show 'loading' status
  if (state.modelLoading) return 'loading'
  return state.systemStatus.model_loaded ? 'loaded' : 'not_loaded'
})

// GPU Memory info computed
const gpuMemory = computed(() => {
  return state.systemStatus?.gpu_memory || null
})

const gpuMemoryPercent = computed(() => {
  const mem = state.systemStatus?.gpu_memory
  if (!mem || !mem.available) return 0
  return mem.percent
})

const gpuMemoryColor = computed(() => {
  const percent = gpuMemoryPercent.value
  if (percent < 20) return 'green'
  if (percent < 70) return 'orange'
  return 'red'
})

// Actions
function setViewMode(mode: ViewMode) {
  state.viewMode = mode
}

function setInputImage(
  filename: string, 
  url: string, 
  isPreset: boolean = false,
  isVideoInput: boolean = false,
  originalVideoFilename: string | null = null
) {
  state.inputImage = { filename, url, isPreset, isVideoInput, originalVideoFilename }
}

function clearInputImage() {
  state.inputImage = { filename: null, url: null, isPreset: false, isVideoInput: false, originalVideoFilename: null }
}

function setTrajectory(trajectory: TrajectoryPreset | null) {
  state.selectedTrajectory = trajectory
}

function setPreviewStage(stage: 'preview' | 'self_forcing') {
  state.previewStage = stage
}

function setRefineStage(stage: ModelStage) {
  state.refineStage = stage
}

function setScale(scale: number) {
  state.scale = Math.round(scale * 10) / 10 // Round to 1 decimal
}

function setSegments(segments: number) {
  state.segments = Math.max(2, Math.min(8, Math.round(segments))) // Clamp to 2-8, integers only
}

function setSelectedModel(model: 'preview' | 'self_forcing' | 'refine') {
  state.selectedModel = model
}

function setSourceVideoFilename(filename: string | null) {
  state.sourceVideoFilename = filename
}

function startGeneration() {
  state.isGenerating = true
  state.generationElapsed = 0
  
  // Reset completion flag for new generation
  completionHandled = false
  
  // Start timer
  generationTimer = window.setInterval(() => {
    state.generationElapsed += 0.1
  }, 100)
}

function stopGeneration() {
  state.isGenerating = false
  state.isSaving = false
  
  if (generationTimer !== null) {
    clearInterval(generationTimer)
    generationTimer = null
  }
}

function setSaving(saving: boolean, pendingVideo: string | null = null) {
  state.isSaving = saving
  state.pendingVideoFilename = pendingVideo
  // Keep isGenerating true during saving to maintain locked state
  if (saving) {
    state.isGenerating = false
  }
}

// Track recently completed video to prevent accidental switches
let recentlyCompletedVideo: string | null = null
let recentlyCompletedTime: number = 0

function checkAndCompleteSaving() {
  // Check if pending video is now in the gallery
  if (state.pendingVideoFilename && state.videos.includes(state.pendingVideoFilename)) {
    // Video is now in gallery, complete the saving process
    const completedVideo = state.pendingVideoFilename
    console.log('[AppState] Video appeared in gallery:', completedVideo)
    
    state.isSaving = false
    state.activeVideo = completedVideo
    state.pendingVideoFilename = null
    
    // Track this completion to prevent immediate re-selection
    recentlyCompletedVideo = completedVideo
    recentlyCompletedTime = Date.now()
    
    if (generationTimer !== null) {
      clearInterval(generationTimer)
      generationTimer = null
    }
    
    console.log('[AppState] Active video set to:', state.activeVideo)
    return true
  }
  return false
}

function setVideos(videos: string[]) {
  state.videos = videos
  // Check if pending video is now in the list
  checkAndCompleteSaving()
}

function addVideo(filename: string) {
  if (!state.videos.includes(filename)) {
    state.videos.unshift(filename)
  }
}

function setActiveVideo(filename: string | null) {
  // Prevent rapid switching right after generation completes
  // This guards against any accidental selection changes within 500ms of completion
  if (recentlyCompletedVideo && (Date.now() - recentlyCompletedTime) < 500) {
    if (filename !== recentlyCompletedVideo && filename !== null) {
      console.log('[AppState] Ignoring video switch to', filename, 'within protection period')
      return
    }
  }
  state.activeVideo = filename
}

function clearVideos() {
  state.videos = []
  state.activeVideo = null
}

function clearContent() {
  // Clear both input image and active video
  state.inputImage = { filename: null, url: null, isPreset: false, isVideoInput: false, originalVideoFilename: null }
  state.activeVideo = null
}

// Completion callback
let onGenerationCompleteCallback: ((video: string) => void) | null = null
let completionHandled = false  // Guard against multiple callback triggers

function setOnGenerationComplete(callback: ((video: string) => void) | null) {
  onGenerationCompleteCallback = callback
}

function updateSystemStatus(status: SystemStatus) {
  state.systemStatus = status
  
  // Update model loading state from server
  const currentlyLoading = status.model_loading || false
  state.modelLoading = currentlyLoading
  state.modelLoadingName = status.model_loading_name || null
  
  // Track if we've seen model_loading = true
  if (currentlyLoading) {
    state.modelLoadingDetected = true
  }
  
  // Only update model loaded states if not prevented by manual update
  if (!state.preventPollingUpdate) {
    state.previewLoaded = status.preview_loaded ?? true  // Default true on old API
    state.selfForcingLoaded = status.self_forcing_loaded ?? false
    state.refineLoaded = status.refine_loaded ?? false
  } else {
    // If model loading just finished (and we've seen it start), allow polling to resume
    // Only re-enable if we've actually seen model_loading = true at some point
    if (!currentlyLoading && state.modelLoadingDetected) {
      console.log('[AppState] Model loading finished, re-enabling polling updates')
      state.preventPollingUpdate = false
      state.modelLoadingDetected = false
      // Now update with the actual backend state
      state.previewLoaded = status.preview_loaded ?? true
      state.selfForcingLoaded = status.self_forcing_loaded ?? false
      state.refineLoaded = status.refine_loaded ?? false
    } else if (!currentlyLoading && !state.modelLoadingDetected) {
      // Backend never reported loading=true, this might be a race condition
      // Wait a bit longer (will be corrected on next poll)
      console.log('[AppState] Waiting for model_loading=true signal...')
    }
  }
  
  // Update model failure state
  state.modelLoadFailed = status.model_load_failed ?? false
  state.failedModelName = status.failed_model_name ?? null
  state.fallbackToPreview = status.fallback_to_preview ?? false
  state.modelErrorMessage = status.model_error_message ?? null
  
  // Update phase based on server status
  if (status.current_phase === 'saving' && state.isGenerating) {
    // Transition from generating to saving, set pending video filename
    state.isGenerating = false
    state.isSaving = true
    if (status.newly_generated_video) {
      state.pendingVideoFilename = status.newly_generated_video
    }
  }
  
  // Check for generation completion - trigger callback ONLY ONCE
  if (status.generation_completed && status.newly_generated_video && !completionHandled) {
    // Mark as handled to prevent multiple triggers
    completionHandled = true
    // Trigger completion callback to start loading videos
    if (onGenerationCompleteCallback) {
      onGenerationCompleteCallback(status.newly_generated_video)
    }
  }
  
  // DON'T auto-stop generation based on gpu_busy - let gallery loading control it
  // Only stop if we're generating (not saving) and server says not busy
  if (!status.gpu_busy && state.isGenerating && !state.isSaving) {
    stopGeneration()
  }
  
  // Update elapsed time from server (keep timer running during saving too)
  if (status.gpu_busy || state.isSaving) {
    state.generationElapsed = status.elapsed_seconds
  }
}

function setModelLoading(loading: boolean, modelName: string | null) {
  state.modelLoading = loading
  state.modelLoadingName = modelName
}

function setCameraOrientation(yaw: number, pitch: number) {
  state.cameraYaw = yaw
  state.cameraPitch = pitch
}

// Export
export function useAppState() {
  return {
    state,
    canGenerate,
    gpuStatus,
    modelStatus,
    gpuMemory,
    gpuMemoryPercent,
    gpuMemoryColor,
    setViewMode,
    setInputImage,
    clearInputImage,
    setTrajectory,
    setPreviewStage,
    setRefineStage,
    setScale,
    setSegments,
    setSelectedModel,
    setSourceVideoFilename,
    startGeneration,
    stopGeneration,
    setSaving,
    checkAndCompleteSaving,
    setVideos,
    addVideo,
    setActiveVideo,
    clearVideos,
    clearContent,
    updateSystemStatus,
    setCameraOrientation,
    setOnGenerationComplete,
    setModelLoading,
  }
}

