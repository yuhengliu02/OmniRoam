<script setup lang="ts">
import { ref, computed, watch, onMounted, provide } from 'vue'
import { useAppState, type TrajectoryPreset } from '@/stores/appState'
import { useApi } from '@/composables/useApi'
import VideoPlayer from '@/components/VideoPlayer.vue'
import ImageViewer from '@/components/ImageViewer.vue'
import TrajectoryPresets from '@/components/TrajectoryPresets.vue'
import PresetImages from '@/components/PresetImages.vue'

const { state, setInputImage, setTrajectory, clearContent } = useAppState()

// Trajectories disabled for Self-Forcing model
const SELF_FORCING_DISABLED_TRAJECTORIES: TrajectoryPreset[] = ['s_curve', 'loop']

// All trajectories (for Refine mode)
const ALL_TRAJECTORIES: TrajectoryPreset[] = ['forward', 'backward', 'left', 'right', 's_curve', 'loop']

// Computed disabled trajectories based on selected model
const disabledTrajectories = computed<TrajectoryPreset[]>(() => {
  // Refine mode: disable all trajectories
  if (state.selectedModel === 'refine') {
    return ALL_TRAJECTORIES
  }
  // Self-forcing mode: disable s_curve and loop
  if (state.previewStage === 'self_forcing') {
    return SELF_FORCING_DISABLED_TRAJECTORIES
  }
  return []
})

// Check if there's content to clear
const hasContent = computed(() => {
  return !!state.inputImage.filename || !!state.activeVideo
})

// Clear all content from player
function handleClear() {
  clearContent()
  // Reset playback state
  isPlaying.value = false
  currentTime.value = 0
  duration.value = 0
  currentFrame.value = 0
  totalFrames.value = 0
  frameInput.value = ''
  videoRef.value = null
}

// Compute aspect ratio based on view mode
const playerAspectRatio = computed(() => {
  return state.viewMode === 'perspective' ? '16 / 9' : '2 / 1'
})
const api = useApi()

// Video playback state (shared with VideoPlayer)
const videoRef = ref<HTMLVideoElement | null>(null)
const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const currentFrame = ref(0)
const totalFrames = ref(0)
const frameInput = ref('')
const FPS = 30

// Provide video ref to child components
provide('videoRef', videoRef)
provide('isPlaying', isPlaying)
provide('currentTime', currentTime)
provide('duration', duration)
provide('currentFrame', currentFrame)
provide('totalFrames', totalFrames)

// Check if we have a playable video
const hasVideo = computed(() => !!state.activeVideo)

// Format time as MM:SS
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// Video control functions
function togglePlay() {
  if (!videoRef.value || !hasVideo.value) return
  
  if (videoRef.value.paused) {
    videoRef.value.play()
  } else {
    videoRef.value.pause()
  }
}

function seek(event: Event) {
  const input = event.target as HTMLInputElement
  if (videoRef.value && hasVideo.value) {
    videoRef.value.currentTime = parseFloat(input.value)
  }
}

function jumpToFrame() {
  const frame = parseInt(frameInput.value)
  if (!isNaN(frame) && videoRef.value && hasVideo.value) {
    const time = Math.min(Math.max(frame / FPS, 0), duration.value)
    videoRef.value.currentTime = time
  }
}

// Reset playback state when video is removed
watch(hasVideo, (has) => {
  if (!has) {
    isPlaying.value = false
    currentTime.value = 0
    duration.value = 0
    currentFrame.value = 0
    totalFrames.value = 0
    frameInput.value = ''
    videoRef.value = null
  }
})

const fileInputRef = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const uploadError = ref<string | null>(null)
const isUploading = ref(false)

// Upload handling
function triggerUpload() {
  fileInputRef.value?.click()
}

async function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (input.files?.length) {
    await uploadFile(input.files[0])
  }
  // Reset input
  input.value = ''
}

async function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false
  
  const file = event.dataTransfer?.files[0]
  if (file) {
    await uploadFile(file)
  }
}

async function uploadFile(file: File) {
  uploadError.value = null
  
  const isImage = file.type.startsWith('image/')
  const isVideo = file.type.startsWith('video/')
  
  // Validate file type
  if (!isImage && !isVideo) {
    uploadError.value = 'Please upload an image or video file'
    return
  }
  
  // Quick client-side ratio check for images
  if (isImage) {
    try {
      const img = await loadImage(file)
      const ratio = img.width / img.height
      if (ratio < 1.95 || ratio > 2.05) {
        uploadError.value = `Invalid aspect ratio (${ratio.toFixed(2)}). ERP content must have 2:1 ratio.`
        return
      }
    } catch {
      uploadError.value = 'Failed to read image'
      return
    }
  }
  
  // For videos, we'll let the backend validate the ratio
  // Show appropriate loading message
  isUploading.value = true
  uploadingType.value = isVideo ? 'video' : 'image'
  
  try {
    const response = await api.uploadImage(file)
    if (response.status === 'success' && response.filename) {
      setInputImage(
        response.filename,
        api.getInputImageUrl(response.filename),
        false
      )
    }
  } catch (err: any) {
    uploadError.value = err.message || 'Upload failed'
  } finally {
    isUploading.value = false
    uploadingType.value = null
  }
}

const uploadingType = ref<'image' | 'video' | null>(null)

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

function handleDragOver(event: DragEvent) {
  event.preventDefault()
  isDragging.value = true
}

function handleDragLeave() {
  isDragging.value = false
}

// Content to display
const showUploadBox = computed(() => {
  return !state.inputImage.filename && !state.activeVideo
})

const showImage = computed(() => {
  return state.inputImage.filename && !state.activeVideo
})

const showVideo = computed(() => {
  return !!state.activeVideo
})

const currentVideoUrl = computed(() => {
  if (state.activeVideo) {
    return api.getVideoUrl(state.activeVideo)
  }
  return null
})
</script>

<template>
  <section class="workspace">
    <!-- Video Player Panel -->
    <div class="player-panel panel">
      <div 
        :class="['player-content', state.viewMode === 'erp' ? 'mode-erp' : 'mode-perspective']"
      >
        <!-- Upload Box -->
        <div 
          v-if="showUploadBox"
          :class="['upload-box', { dragging: isDragging }]"
          @click="triggerUpload"
          @drop="handleDrop"
          @dragover="handleDragOver"
          @dragleave="handleDragLeave"
        >
          <input 
            ref="fileInputRef"
            type="file"
            accept="image/*,video/*"
            @change="handleFileSelect"
            hidden
          />
          
          <div class="upload-content">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            
            <div v-if="isUploading" class="upload-text">
              <span v-if="uploadingType === 'video'">Processing video...</span>
              <span v-else>Uploading...</span>
            </div>
            <div v-else class="upload-text">
              <strong>Click to upload</strong> or drag & drop
              <span class="upload-hint">ERP images or videos (2:1 ratio)</span>
              <span v-if="state.selectedModel === 'refine'" class="upload-hint-sub">Refine: Full video will be used (81 frames required)</span>
            </div>
            
            <div v-if="uploadError" class="upload-error">{{ uploadError }}</div>
          </div>
        </div>

        <!-- Image Display (with 360° support in perspective mode) -->
        <ImageViewer 
          v-if="showImage"
          :src="state.inputImage.url!"
          :mode="state.viewMode"
          :is-preset="state.inputImage.isPreset"
        />
        
        <!-- Video Input Badge (for refine mode) -->
        <div v-if="showImage && state.inputImage.isVideoInput" class="video-input-badge">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="23 7 16 12 23 17 23 7"/>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
          </svg>
          <span>Video Input</span>
        </div>

        <!-- Video Player -->
        <VideoPlayer 
          v-if="showVideo"
          :src="currentVideoUrl!"
          :mode="state.viewMode"
        />
      </div>
    </div>

    <!-- Always Visible Video Controls -->
    <div class="video-controls-bar">
      <!-- Play/Pause Button -->
      <button 
        :class="['play-btn', { disabled: !hasVideo, playing: isPlaying }]"
        :disabled="!hasVideo"
        @click="togglePlay"
      >
        <span class="btn-glow"></span>
        <span class="btn-glow-pulse"></span>
        <span class="btn-icon">
          <svg v-if="!isPlaying" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="6 3 20 12 6 21 6 3"/>
          </svg>
          <svg v-else viewBox="0 0 24 24" fill="currentColor">
            <rect x="5" y="4" width="4" height="16"/>
            <rect x="15" y="4" width="4" height="16"/>
          </svg>
        </span>
      </button>

      <!-- Progress Bar -->
      <div class="progress-container">
        <input 
          type="range" 
          class="progress-bar"
          :value="currentTime"
          :max="duration || 100"
          step="0.01"
          :disabled="!hasVideo"
          @input="seek"
        />
        <div class="time-display">
          {{ formatTime(currentTime) }} / {{ formatTime(duration) }}
        </div>
      </div>

      <!-- Frame Controls -->
      <div class="frame-controls">
        <span class="frame-label">frame:</span>
        <span class="frame-current">{{ currentFrame }}</span>
        <span class="frame-separator">/</span>
        <span class="frame-total">{{ totalFrames || '---' }}</span>
        
        <input 
          v-model="frameInput"
          type="number"
          class="frame-input"
          placeholder="#"
          min="0"
          :max="totalFrames"
          :disabled="!hasVideo"
          @keyup.enter="jumpToFrame"
        />
        <button 
          class="frame-jump-btn" 
          :disabled="!hasVideo"
          @click="jumpToFrame"
        >Go</button>
      </div>

      <!-- Clear Button -->
      <button 
        :class="['clear-btn', { disabled: !hasContent }]"
        :disabled="!hasContent"
        @click="handleClear"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        Clear
      </button>
    </div>

    <!-- Trajectory Presets -->
    <div :class="['trajectory-section', { disabled: state.isGenerating || state.modelLoading }]">
      <h3 class="section-title">Camera Trajectory</h3>
      <TrajectoryPresets 
        :selected="state.selectedTrajectory"
        :disabled="state.isGenerating || state.modelLoading"
        :disabled-trajectories="disabledTrajectories"
        @select="setTrajectory"
      />
    </div>

    <!-- Preset Images -->
    <div class="presets-section">
      <h3 class="section-title">Preset Images</h3>
      <PresetImages />
    </div>
  </section>
</template>

<style scoped>
.workspace {
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  overflow: hidden;
}

.player-panel {
  /* Don't use flex-grow, let aspect-ratio control size */
  flex: 0 1 auto;
  width: 100%;
  max-height: 100%;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 
    0 4px 20px rgba(0, 0, 0, 0.3),
    0 0 40px rgba(255, 255, 255, 0.02);
}

/* ERP mode: panel has 2:1 aspect ratio */
.player-panel:has(.mode-erp) {
  aspect-ratio: 2 / 1;
}

/* Perspective mode: panel has 16:9 aspect ratio */
.player-panel:has(.mode-perspective) {
  aspect-ratio: 16 / 9;
}

.player-content {
  /* Fill the entire panel */
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-box {
  position: absolute;
  inset: var(--spacing-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border: 2px dashed var(--color-border);
  border-radius: var(--radius-md);
  transition: all 0.2s ease;
}

.upload-box:hover,
.upload-box.dragging {
  border-color: var(--color-accent);
  background: rgba(88, 101, 242, 0.05);
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
  text-align: center;
}

.upload-icon {
  width: 48px;
  height: 48px;
  color: var(--color-text-muted);
}

.upload-text {
  color: var(--color-text);
  font-size: 14px;
}

.upload-text strong {
  color: var(--color-accent);
}

.upload-hint {
  display: block;
  margin-top: var(--spacing-xs);
  font-size: 12px;
  color: var(--color-text-muted);
}

.upload-hint-sub {
  display: block;
  margin-top: 2px;
  font-size: 11px;
  color: var(--color-text-muted);
  opacity: 0.7;
}

.upload-error {
  color: var(--color-error);
  font-size: 13px;
  padding: var(--spacing-sm) var(--spacing-md);
  background: rgba(240, 71, 71, 0.1);
  border-radius: var(--radius-md);
}

.section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--spacing-sm);
  font-family: var(--font-body);
}

.trajectory-section {
  flex: 0 0 auto;
}

.trajectory-section.disabled {
  opacity: 0.6;
}

.trajectory-section.disabled .section-title::after {
  content: ' (Locked)';
  font-size: 10px;
  color: var(--color-text-muted);
  font-weight: normal;
}

.presets-section {
  flex: 0 0 auto;
}

/* Video Controls Bar */
.video-controls-bar {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-panel);
  border-radius: var(--radius-lg);
  flex-shrink: 0;
  box-shadow: 
    0 4px 20px rgba(0, 0, 0, 0.3),
    0 0 40px rgba(255, 255, 255, 0.02);
}

/* Circular Play Button with #5865f2 Glow */
.play-btn {
  position: relative;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(145deg, #2a2a3a, #1a1a28);
  border: 1px solid rgba(88, 101, 242, 0.3);
  color: white;
  flex-shrink: 0;
  cursor: pointer;
  overflow: hidden;
  transition: all 0.3s ease;
}

.play-btn .btn-icon {
  position: relative;
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.play-btn .btn-icon svg {
  width: 16px;
  height: 16px;
  filter: drop-shadow(0 0 4px rgba(88, 101, 242, 0.5));
}

/* Glow effects */
.play-btn .btn-glow {
  position: absolute;
  inset: -2px;
  border-radius: 50%;
  background: linear-gradient(45deg, 
    rgba(88, 101, 242, 0.4),
    rgba(114, 137, 218, 0.2),
    rgba(88, 101, 242, 0.4)
  );
  opacity: 0;
  transition: opacity 0.3s ease;
  filter: blur(8px);
  z-index: 0;
}

.play-btn .btn-glow-pulse {
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(88, 101, 242, 0.3) 0%, transparent 70%);
  opacity: 0;
  animation: playBtnPulse 2s ease-in-out infinite;
  z-index: 0;
}

@keyframes playBtnPulse {
  0%, 100% { opacity: 0; transform: scale(0.95); }
  50% { opacity: 1; transform: scale(1.1); }
}

/* Enabled/Active state */
.play-btn:not(.disabled) {
  box-shadow: 
    0 0 15px rgba(88, 101, 242, 0.3),
    0 0 30px rgba(88, 101, 242, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.play-btn:not(.disabled) .btn-glow {
  opacity: 0.5;
}

.play-btn:not(.disabled) .btn-glow-pulse {
  opacity: 1;
}

.play-btn:not(.disabled):hover {
  transform: scale(1.05);
  box-shadow: 
    0 0 25px rgba(88, 101, 242, 0.5),
    0 0 50px rgba(88, 101, 242, 0.25),
    inset 0 1px 0 rgba(255, 255, 255, 0.15);
}

.play-btn:not(.disabled):hover .btn-glow {
  opacity: 0.8;
}

/* Playing state - more intense glow */
.play-btn.playing:not(.disabled) {
  border-color: rgba(88, 101, 242, 0.6);
}

.play-btn.playing:not(.disabled) .btn-glow {
  opacity: 0.7;
  animation: playingGlow 1.5s ease-in-out infinite alternate;
}

@keyframes playingGlow {
  from { filter: blur(8px) brightness(1); }
  to { filter: blur(12px) brightness(1.3); }
}

/* Disabled state */
.play-btn.disabled {
  background: linear-gradient(145deg, #222228, #1a1a1e);
  border-color: rgba(255, 255, 255, 0.05);
  cursor: not-allowed;
  opacity: 0.6;
}

.play-btn.disabled .btn-icon svg {
  filter: none;
  opacity: 0.4;
}

.play-btn.disabled .btn-glow,
.play-btn.disabled .btn-glow-pulse {
  display: none;
}

/* Progress Container */
.progress-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.progress-bar {
  width: 100%;
  height: 5px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  cursor: pointer;
}

.progress-bar::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 12px;
  height: 12px;
  background: linear-gradient(145deg, #7289da, #5865f2);
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 0 10px rgba(88, 101, 242, 0.5);
}

.progress-bar::-moz-range-thumb {
  width: 12px;
  height: 12px;
  background: linear-gradient(145deg, #7289da, #5865f2);
  border-radius: 50%;
  cursor: pointer;
  border: none;
  box-shadow: 0 0 10px rgba(88, 101, 242, 0.5);
}

.progress-bar:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.progress-bar:disabled::-webkit-slider-thumb {
  background: #444;
  box-shadow: none;
}

.time-display {
  font-size: 11px;
  color: var(--color-text-muted);
  font-variant-numeric: tabular-nums;
}

/* Frame Controls */
.frame-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 12px;
  color: var(--color-text-muted);
  flex-shrink: 0;
}

.frame-current,
.frame-total {
  font-variant-numeric: tabular-nums;
  min-width: 32px;
  text-align: right;
}

.frame-current {
  color: var(--color-text);
}

.frame-separator {
  opacity: 0.5;
}

.frame-input {
  width: 60px;
  padding: var(--spacing-xs) var(--spacing-sm);
  font-size: 12px;
  text-align: center;
  background: var(--color-input);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text);
}

.frame-input:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.frame-jump-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-panel-hover);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--color-text);
  cursor: pointer;
  transition: background 0.2s ease;
}

.frame-jump-btn:hover:not(:disabled) {
  background: var(--color-border);
}

.frame-jump-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Clear Button */
.clear-btn {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-md);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  font-size: 12px;
  color: var(--color-text-muted);
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.clear-btn svg {
  width: 14px;
  height: 14px;
}

.clear-btn:not(.disabled):hover {
  background: rgba(240, 71, 71, 0.1);
  border-color: rgba(240, 71, 71, 0.3);
  color: #f04747;
}

.clear-btn.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Video Input Badge (for refine mode) */
.video-input-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: rgba(88, 101, 242, 0.9);
  border-radius: 16px;
  z-index: 10;
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.video-input-badge svg {
  width: 14px;
  height: 14px;
  color: white;
}

.video-input-badge span {
  font-size: 11px;
  font-weight: 600;
  color: white;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
</style>

