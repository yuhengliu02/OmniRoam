<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppState, type ModelStage } from '@/stores/appState'
import { useApi, type GenerateRequest } from '@/composables/useApi'
import PerspectiveIndicator from '@/components/PerspectiveIndicator.vue'

const { 
  state, 
  canGenerate,
  gpuStatus,
  modelStatus,
  gpuMemory,
  gpuMemoryPercent,
  gpuMemoryColor,
  setPreviewStage, 
  setRefineStage, 
  setScale,
  setSegments,
  setSelectedModel,
  setTrajectory,
  startGeneration,
  addVideo,
  setActiveVideo,
  setModelLoading,
  updateSystemStatus,
} = useAppState()

const api = useApi()

const isGenerating = ref(false)
const generationError = ref<string | null>(null)

// Unified model selection (all 3 are mutually exclusive)
const selectedModel = ref<'preview' | 'self_forcing' | 'refine'>('preview')

// Trajectories not supported by Self-Forcing
const SELF_FORCING_UNSUPPORTED_TRAJECTORIES = ['s_curve', 'loop']

// Check if model is loading
const isModelLoading = computed(() => state.modelLoading)

// Check if a specific model is loading
const isLoadingPreview = computed(() => state.modelLoading && state.modelLoadingName === 'preview')
const isLoadingSelfForcing = computed(() => state.modelLoading && state.modelLoadingName === 'self_forcing')
const isLoadingRefine = computed(() => state.modelLoading && state.modelLoadingName === 'refine')

// Check if in refine mode
const isRefineMode = computed(() => selectedModel.value === 'refine')

// Check if scale/segments is disabled
const isScaleDisabled = computed(() => {
  return isWorking.value || isModelLoading.value || selectedModel.value === 'self_forcing' || selectedModel.value === 'refine'
})

// Check if sections should be locked (during generation, saving, model loading, or critical failure)
const isSectionsLocked = computed(() => {
  // Lock if currently working or loading
  if (isWorking.value || isModelLoading.value) return true
  
  // Lock if preview model failed (critical failure - system unusable)
  if (state.modelLoadFailed && !state.fallbackToPreview) {
    return true
  }
  
  return false
})

async function selectModel(model: 'preview' | 'self_forcing' | 'refine') {
  // Don't allow selection during generation or model loading
  if (isWorking.value || isModelLoading.value) return
  
  // Check if we need to load the model based on CURRENT frontend state
  console.log(`[Settings] Selecting model: ${model}`)
  console.log(`[Settings] Current state - Preview: ${state.previewLoaded}, Self-Forcing: ${state.selfForcingLoaded}, Refine: ${state.refineLoaded}`)
  
  const needsLoad = (model === 'preview' && !state.previewLoaded) || 
                   (model === 'self_forcing' && !state.selfForcingLoaded) ||
                   (model === 'refine' && !state.refineLoaded)
  
  console.log(`[Settings] needsLoad = ${needsLoad}`)
  
  if (needsLoad) {
    try {
      // Optimistically set loading state
      setModelLoading(true, model)
      
      // IMPORTANT: Immediately update frontend state to reflect expected changes
      // This prevents race conditions from polling delays
      console.log(`[Settings] Proactively updating model states for ${model} load`)
      
      // Prevent polling from overwriting our manual updates until loading completes
      state.preventPollingUpdate = true
      console.log(`[Settings] Disabled polling updates until model loading completes`)
      
      // Determine which models will be offloaded based on VRAM logic
      // (This mirrors the backend logic)
      const totalVram = state.systemStatus?.gpu_memory?.total_gb || 80  // Assume 80GB if unknown
      
      if (model === 'refine') {
        // Loading Refine will offload ALL other models (if VRAM < 150GB)
        if (totalVram < 150) {
          console.log(`[Settings] VRAM < 150GB: Loading Refine will offload all models`)
          state.previewLoaded = false
          state.selfForcingLoaded = false
        }
        // else: >= 150GB, no offloading needed
      } else if (model === 'preview') {
        // Loading Preview
        if (totalVram < 75) {
          // < 75GB: Offload all others
          console.log(`[Settings] VRAM < 75GB: Loading Preview will offload others`)
          state.selfForcingLoaded = false
          state.refineLoaded = false
        } else if (totalVram < 150) {
          // 75-150GB: Only offload Refine if it's loaded
          if (state.refineLoaded) {
            console.log(`[Settings] VRAM < 150GB: Loading Preview will offload Refine`)
            state.refineLoaded = false
          }
        }
        // else: >= 150GB, no offloading needed
      } else if (model === 'self_forcing') {
        // Loading Self-Forcing
        if (totalVram < 75) {
          // < 75GB: Offload all others
          console.log(`[Settings] VRAM < 75GB: Loading Self-Forcing will offload others`)
          state.previewLoaded = false
          state.refineLoaded = false
        } else if (totalVram < 150) {
          // 75-150GB: Only offload Refine if it's loaded
          if (state.refineLoaded) {
            console.log(`[Settings] VRAM < 150GB: Loading Self-Forcing will offload Refine`)
            state.refineLoaded = false
          }
        }
        // else: >= 150GB, no offloading needed
      }
      
      // Now mark the target model as loading (will be set to loaded after API returns)
      console.log(`[Settings] Updated state - Preview: ${state.previewLoaded}, Self-Forcing: ${state.selfForcingLoaded}, Refine: ${state.refineLoaded}`)
      
      // Call API to switch model
      const response = await api.switchModel(model)
      
      if (response.status === 'loading') {
        // Model is loading in background, update selection
        selectedModel.value = model
        setSelectedModel(model)
        
        // Optimistically mark target model as loaded
        // (will be corrected by polling if it fails)
        if (model === 'preview') {
          state.previewLoaded = true
        } else if (model === 'self_forcing') {
          state.selfForcingLoaded = true
        } else if (model === 'refine') {
          state.refineLoaded = true
        }
        
        // Update stage tracking
        if (model === 'preview' || model === 'self_forcing') {
          setPreviewStage(model)
        } else {
          setRefineStage(model)
        }
        
        // If switching to self_forcing, clear unsupported trajectory selection
        if (model === 'self_forcing') {
          if (state.selectedTrajectory && SELF_FORCING_UNSUPPORTED_TRAJECTORIES.includes(state.selectedTrajectory)) {
            setTrajectory(null)
          }
        }
        
        // NOTE: The preventPollingUpdate flag will be automatically cleared by appState.ts
        // when it detects model_loading changed from true to false
      } else if (response.status === 'success') {
        // Model was already loaded, no need to wait for loading to complete
        setModelLoading(false, null)
        state.preventPollingUpdate = false  // Re-enable polling immediately
        selectedModel.value = model
        setSelectedModel(model)
        
        // Mark as loaded
        if (model === 'preview') {
          state.previewLoaded = true
        } else if (model === 'self_forcing') {
          state.selfForcingLoaded = true
        } else if (model === 'refine') {
          state.refineLoaded = true
        }
        
        if (model === 'preview' || model === 'self_forcing') {
          setPreviewStage(model)
        } else {
          setRefineStage(model)
        }
      }
    } catch (err: any) {
      console.error('Model switch error:', err)
      setModelLoading(false, null)
      state.preventPollingUpdate = false  // Re-enable polling on error
      generationError.value = err.message || 'Failed to load model'
    }
  } else {
    // Model already loaded, just switch selection
    selectedModel.value = model
    setSelectedModel(model)
    
    if (model === 'preview' || model === 'self_forcing') {
      setPreviewStage(model)
    } else {
      setRefineStage(model)
    }
    
    // If switching to self_forcing, clear unsupported trajectory selection
    if (model === 'self_forcing') {
      if (state.selectedTrajectory && SELF_FORCING_UNSUPPORTED_TRAJECTORIES.includes(state.selectedTrajectory)) {
        setTrajectory(null)
      }
    }
  }
}

// Format elapsed time
const elapsedFormatted = computed(() => {
  const secs = Math.floor(state.generationElapsed)
  const mins = Math.floor(secs / 60)
  const s = secs % 60
  return `${mins}:${s.toString().padStart(2, '0')}`
})

// Generate button text
const generateBtnText = computed(() => {
  if (state.isSaving) {
    return `Saving Video... ${elapsedFormatted.value}`
  }
  if (state.isGenerating) {
    return `Generating... ${elapsedFormatted.value}`
  }
  return 'Generate'
})

// Check if button should be in "working" state (either generating or saving)
const isWorking = computed(() => {
  return state.isGenerating || state.isSaving
})

// Can download
const canDownload = computed(() => {
  return state.activeVideo !== null
})

async function handleGenerate() {
  if (!canGenerate.value || isGenerating.value) return
  
  isGenerating.value = true
  generationError.value = null
  startGeneration()
  
  try {
    const stage = selectedModel.value as GenerateRequest['stage']
    
    // For refine mode, scale is the number of segments (2-8)
    const scaleOrSegments = selectedModel.value === 'refine' ? state.segments : state.scale
    
    // For refine mode, trajectory is not used, so provide a default value
    const trajectory = selectedModel.value === 'refine' ? 'forward' : state.selectedTrajectory!
    
    const request: GenerateRequest = {
      stage,
      trajectory,
      scale: scaleOrSegments,
      num_frames: 81,
    }
    
    const response = await api.startGeneration(request)
    
    if (response.status === 'started') {
      await pollForCompletion()
    } else {
      throw new Error(response.message || 'Generation failed')
    }
  } catch (err: any) {
    generationError.value = err.message || 'Generation failed'
    console.error('Generation error:', err)
  } finally {
    isGenerating.value = false
  }
}

async function pollForCompletion() {
  const maxPolls = 300
  let polls = 0
  
  while (polls < maxPolls) {
    await new Promise(resolve => setTimeout(resolve, 1000))
    polls++
    
    try {
      const status = await api.getStatus()
      
      if (!status.gpu_busy) {
        const videos = await api.getVideos()
        if (videos.videos.length > 0) {
          const newVideo = videos.videos[0]
          addVideo(newVideo)
          setActiveVideo(newVideo)
        }
        return
      }
    } catch (err) {
      console.error('Poll error:', err)
    }
  }
  
  throw new Error('Generation timed out')
}

function downloadVideo() {
  if (state.activeVideo) {
    const url = api.getVideoUrl(state.activeVideo)
    const a = document.createElement('a')
    a.href = url
    a.download = state.activeVideo
    a.click()
  }
}

async function downloadLogs() {
  window.open(api.getLogsDownloadUrl(), '_blank')
}
</script>

<template>
  <aside class="settings panel">
    <!-- Perspective Indicator -->
    <section class="settings-section">
      <h3 class="section-title">View Orientation</h3>
      <PerspectiveIndicator 
        :active="state.viewMode === 'perspective'"
        :yaw="state.cameraYaw"
        :pitch="state.cameraPitch"
      />
    </section>

    <!-- Preview Stage Models -->
    <section :class="['settings-section', { disabled: isSectionsLocked }]">
      <h3 class="section-title">Preview Stage</h3>
      <div class="model-grid">
        <button 
          :class="['model-card', { active: selectedModel === 'preview', loading: isLoadingPreview }]"
          :disabled="isSectionsLocked"
          @click="selectModel('preview')"
        >
          <div class="model-glow"></div>
          <div class="model-glow-pulse"></div>
          <div class="model-content">
            <template v-if="isLoadingPreview">
              <svg class="model-spinner" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="60" stroke-linecap="round"/>
              </svg>
              <span class="model-loading-text">Loading Model...</span>
            </template>
            <template v-else>
              <span class="model-brand">OmniRoam</span>
              <span class="model-name">Preview</span>
            </template>
          </div>
        </button>
        <button 
          :class="['model-card', { active: selectedModel === 'self_forcing', loading: isLoadingSelfForcing }]"
          :disabled="isSectionsLocked"
          @click="selectModel('self_forcing')"
        >
          <div class="model-glow"></div>
          <div class="model-glow-pulse"></div>
          <div class="model-content">
            <template v-if="isLoadingSelfForcing">
              <svg class="model-spinner" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="60" stroke-linecap="round"/>
              </svg>
              <span class="model-loading-text">Loading Model...</span>
            </template>
            <template v-else>
              <span class="model-brand">OmniRoam</span>
              <span class="model-name">Self-forcing</span>
            </template>
          </div>
        </button>
      </div>
    </section>

    <!-- Scale / Segments Setting -->
    <section :class="['settings-section', { disabled: isScaleDisabled && !isRefineMode }]">
      <h3 :class="['section-title', { 'segments-title': isRefineMode }]">
        <template v-if="isRefineMode">
          Segments
        </template>
        <template v-else>
          Scale
          <span v-if="selectedModel === 'self_forcing'" class="scale-note">(Fixed for Self-forcing)</span>
        </template>
      </h3>
      <div class="scale-control">
        <template v-if="isRefineMode">
          <!-- Segments slider for refine mode: 2-8, integers only -->
          <input 
            type="range"
            :value="state.segments"
            min="2"
            max="8"
            step="1"
            :disabled="isWorking || isModelLoading"
            @input="(e) => setSegments(parseInt((e.target as HTMLInputElement).value))"
          />
          <span class="scale-value segments-value">{{ state.segments }}</span>
        </template>
        <template v-else>
          <!-- Scale slider for preview/self-forcing modes -->
          <input 
            type="range"
            :value="selectedModel === 'self_forcing' ? 1.0 : state.scale"
            min="1.0"
            max="8.0"
            step="0.1"
            :disabled="isScaleDisabled"
            @input="(e) => setScale(parseFloat((e.target as HTMLInputElement).value))"
          />
          <span class="scale-value">{{ selectedModel === 'self_forcing' ? '1.0' : state.scale.toFixed(1) }}</span>
        </template>
      </div>
    </section>

    <!-- Refine Stage -->
    <section :class="['settings-section', { disabled: isSectionsLocked }]">
      <h3 class="section-title">Refine Stage</h3>
      <button 
        :class="['model-card refine-card', { active: selectedModel === 'refine', loading: isLoadingRefine }]"
        :disabled="isSectionsLocked"
        @click="selectModel('refine')"
      >
        <div class="model-glow"></div>
        <div class="model-glow-pulse"></div>
        <div class="model-content horizontal">
          <template v-if="isLoadingRefine">
            <svg class="model-spinner" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="60" stroke-linecap="round"/>
            </svg>
            <span class="model-loading-text">Loading Model...</span>
          </template>
          <template v-else>
            <span class="model-brand">OmniRoam</span>
            <span class="model-name">Refine</span>
          </template>
        </div>
      </button>
    </section>

    <!-- System Status -->
    <section class="settings-section status-section">
      <h3 class="section-title">System Status</h3>
      <div class="status-list">
        <div class="status-item">
          <span class="status-label">GPU</span>
          <span :class="['status-dot', gpuStatus === 'working' ? 'working' : 'inactive']" />
          <span class="status-text">{{ gpuStatus === 'working' ? 'Working' : 'Idle' }}</span>
        </div>
        <div class="status-item model-status-item">
          <span class="status-label">Model</span>
          <div class="model-indicators">
            <div class="model-indicator">
              <span :class="[
                'status-dot', 
                'model-dot',
                state.modelLoading && state.modelLoadingName === 'preview' ? 'loading' : 
                state.previewLoaded ? 'loaded' : 'unloaded'
              ]" />
              <span class="model-indicator-label">Preview</span>
            </div>
            <div class="model-indicator">
              <span :class="[
                'status-dot',
                'model-dot', 
                state.modelLoading && state.modelLoadingName === 'self_forcing' ? 'loading' : 
                state.selfForcingLoaded ? 'loaded' : 'unloaded'
              ]" />
              <span class="model-indicator-label">Self-forcing</span>
            </div>
            <div class="model-indicator">
              <span :class="[
                'status-dot',
                'model-dot', 
                state.modelLoading && state.modelLoadingName === 'refine' ? 'loading' : 
                state.refineLoaded ? 'loaded' : 'unloaded'
              ]" />
              <span class="model-indicator-label">Refine</span>
            </div>
          </div>
        </div>
        <div v-if="gpuMemory?.available" class="status-item vram-item">
          <span class="status-label">VRAM</span>
          <div class="vram-indicator">
            <svg class="vram-circle" viewBox="0 0 36 36">
              <circle 
                class="vram-bg" 
                cx="18" cy="18" r="15.9" 
                fill="none" 
                stroke-width="3"
              />
              <circle 
                :class="['vram-progress', gpuMemoryColor]"
                cx="18" cy="18" r="15.9" 
                fill="none" 
                stroke-width="3"
                stroke-linecap="round"
                :stroke-dasharray="`${gpuMemoryPercent}, 100`"
              />
            </svg>
          </div>
          <span class="status-text vram-text">
            {{ gpuMemory.used_gb.toFixed(1) }} / {{ gpuMemory.total_gb.toFixed(1) }} GB
          </span>
        </div>
      </div>
      
      <!-- Model Load Failure Alert -->
      <div v-if="state.modelLoadFailed" class="model-error-alert">
        <div class="error-icon">⚠️</div>
        <div class="error-content">
          <div class="error-title">
            Model Load Failed: {{ state.failedModelName }}
          </div>
          <div class="error-message">
            {{ state.fallbackToPreview ? 'Fell back to Preview model' : 'System unusable' }}
          </div>
          <div v-if="state.modelErrorMessage" class="error-details">
            {{ state.modelErrorMessage }}
          </div>
        </div>
      </div>
      
      <button class="log-btn" @click="downloadLogs">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <polyline points="10 9 9 9 8 9"/>
        </svg>
        Log
      </button>
    </section>

    <!-- Action Buttons -->
    <div class="action-section">
      <button 
        :class="['download-btn', { disabled: !canDownload }]"
        :disabled="!canDownload"
        @click="downloadVideo"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download
      </button>
      
      <button 
        :class="['generate-btn', { generating: isWorking, saving: state.isSaving }]"
        :disabled="!canGenerate || isWorking"
        @click="handleGenerate"
      >
        <span class="gen-shimmer"></span>
        <span class="gen-glow"></span>
        <span class="gen-glow-pulse"></span>
        <span class="gen-rays"></span>
        <span class="gen-sparkle gen-sparkle-1"></span>
        <span class="gen-sparkle gen-sparkle-2"></span>
        <span class="gen-sparkle gen-sparkle-3"></span>
        <span class="gen-content">
          <svg v-if="isWorking" class="spinner" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="60" stroke-linecap="round"/>
          </svg>
          {{ generateBtnText }}
        </span>
      </button>
      
      <div v-if="generationError" class="generation-error">
        {{ generationError }}
      </div>
    </div>
  </aside>
</template>

<style scoped>
.settings {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  overflow-x: hidden;
}

/* Custom scrollbar styling */
.settings::-webkit-scrollbar {
  width: 6px;
}

.settings::-webkit-scrollbar-track {
  background: var(--color-bg);
  border-radius: 3px;
}

.settings::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
  transition: background 0.2s ease;
}

.settings::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-dim);
}

.settings-section {
  padding: var(--spacing-sm) var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.status-section {
  flex-shrink: 0;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--spacing-xs);
  font-family: var(--font-body);
}

/* Model Selection Grid */
.model-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--spacing-sm);
}

.model-card {
  position: relative;
  aspect-ratio: 1;
  background: linear-gradient(145deg, #1a1a1a 0%, #0d0d0d 100%);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
}

/* Refine card - full width rectangular */
.model-card.refine-card {
  aspect-ratio: auto;
  height: 56px;
  width: 100%;
}

.model-card:hover {
  border-color: var(--color-text-dim);
  transform: translateY(-2px);
}

.model-card.active {
  border-color: var(--color-accent);
}

/* Glow effects */
.model-glow {
  position: absolute;
  inset: 0;
  background: radial-gradient(
    ellipse at 50% 80%,
    rgba(88, 101, 242, 0.15) 0%,
    transparent 60%
  );
  opacity: 0;
  transition: opacity 0.3s ease;
}

.model-card:hover .model-glow,
.model-card.active .model-glow {
  opacity: 1;
}

.model-glow-pulse {
  position: absolute;
  inset: -20%;
  background: radial-gradient(
    circle at 50% 100%,
    rgba(88, 101, 242, 0.4) 0%,
    rgba(88, 101, 242, 0.1) 30%,
    transparent 50%
  );
  opacity: 0;
  animation: glow-pulse 3s ease-in-out infinite;
  pointer-events: none;
}

.model-card.active .model-glow-pulse {
  opacity: 1;
}

@keyframes glow-pulse {
  0%, 100% {
    transform: scale(0.8) translateY(10%);
    opacity: 0.3;
  }
  50% {
    transform: scale(1.2) translateY(-5%);
    opacity: 0.8;
  }
}

/* Model content */
.model-content {
  position: relative;
  z-index: 1;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  padding: var(--spacing-sm);
}

.model-content.horizontal {
  flex-direction: row;
  gap: var(--spacing-sm);
}

.model-brand {
  font-size: 10px;
  font-weight: 500;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: color 0.3s ease;
}

.model-card.active .model-brand {
  color: var(--color-accent);
}

.model-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text);
  text-align: center;
  line-height: 1.2;
}

.model-card.active .model-name {
  color: white;
}

/* Add inner border glow for active state */
.model-card.active::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: var(--radius-lg);
  padding: 1px;
  background: linear-gradient(
    135deg,
    rgba(88, 101, 242, 0.6) 0%,
    rgba(88, 101, 242, 0.1) 50%,
    rgba(88, 101, 242, 0.6) 100%
  );
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: border-glow 3s linear infinite;
}

@keyframes border-glow {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

/* Disabled states for model cards */
.model-card:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  pointer-events: none;
}

.model-card:disabled .model-glow,
.model-card:disabled .model-glow-pulse {
  display: none;
}

/* Loading state for model cards */
.model-card.loading {
  border-color: #f5a623;
}

.model-card.loading .model-content {
  color: #999;
}

.model-spinner {
  width: 18px;
  height: 18px;
  animation: spin 1s linear infinite;
  color: #f5a623;
}

.model-loading-text {
  font-size: 11px;
  color: #999;
  text-align: center;
  line-height: 1.2;
}

/* Disabled section styling */
.settings-section.disabled {
  opacity: 0.6;
  pointer-events: none;
}

.settings-section.disabled .section-title::after {
  content: ' (Locked)';
  font-size: 10px;
  color: var(--color-text-muted);
  font-weight: normal;
}

/* Scale control */
.scale-control {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.scale-control input {
  flex: 1;
}

.scale-value {
  min-width: 36px;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-accent);
  font-variant-numeric: tabular-nums;
}

.scale-note {
  font-size: 9px;
  font-weight: normal;
  color: var(--color-text-muted);
  text-transform: none;
  letter-spacing: 0;
  margin-left: 4px;
}

/* Segments title style for refine mode */
.segments-title {
  color: #5865f2 !important;
}

.segments-value {
  color: #5865f2 !important;
}

/* Status */
.status-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-sm);
}

.status-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.status-label {
  font-size: 12px;
  color: var(--color-text-muted);
  min-width: 50px;
}

.status-text {
  font-size: 12px;
  color: var(--color-text);
}

/* Status dot states */
.status-dot.working {
  background: #f5a623;
  box-shadow: 0 0 8px rgba(245, 166, 35, 0.6);
  animation: pulse-working 1.5s ease-in-out infinite;
}

@keyframes pulse-working {
  0%, 100% {
    box-shadow: 0 0 4px rgba(245, 166, 35, 0.4);
  }
  50% {
    box-shadow: 0 0 12px rgba(245, 166, 35, 0.8);
  }
}

/* Model status indicators */
.model-status-item {
  flex-direction: row;
  align-items: center;
  gap: var(--spacing-sm);
}

.model-indicators {
  display: flex;
  gap: var(--spacing-md);
}

.model-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);  /* 与 status-item 保持一致 */
}

.model-indicator-label {
  font-size: 12px;
  color: var(--color-text);  /* 与 status-text 保持一致 */
  text-transform: capitalize;
}

.model-dot {
  width: 8px;
  height: 8px;
}

/* Model dot states */
.model-dot.unloaded {
  background: #666666;
  box-shadow: 0 0 2px rgba(102, 102, 102, 0.4);
}

.model-dot.loaded {
  background: #43b581;
  box-shadow: 0 0 6px rgba(67, 181, 129, 0.6);
}

.model-dot.loading {
  background: #f5a623;
  box-shadow: 0 0 8px rgba(245, 166, 35, 0.6);
  animation: pulse-working 1.5s ease-in-out infinite;
}

/* Model error alert */
.model-error-alert {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  background: rgba(240, 71, 71, 0.1);
  border: 1px solid rgba(240, 71, 71, 0.3);
  border-radius: var(--radius-md);
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
}

.error-icon {
  font-size: 20px;
  flex-shrink: 0;
}

.error-content {
  flex: 1;
  min-width: 0;
}

.error-title {
  font-weight: 600;
  color: #f04747;
  font-size: 12px;
  margin-bottom: 4px;
}

.error-message {
  font-size: 11px;
  color: var(--color-text);
  margin-bottom: 4px;
}

.error-details {
  font-size: 10px;
  color: var(--color-text-muted);
  font-family: monospace;
  word-break: break-word;
}

/* VRAM indicator */
.vram-item {
  align-items: center;
}

.vram-indicator {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.vram-circle {
  transform: rotate(-90deg);
  width: 100%;
  height: 100%;
}

.vram-bg {
  stroke: var(--color-border);
}

.vram-progress {
  transition: stroke-dasharray 0.3s ease;
}

.vram-progress.green {
  stroke: #43b581;
}

.vram-progress.orange {
  stroke: #f5a623;
}

.vram-progress.red {
  stroke: #f04747;
}

.vram-text {
  font-variant-numeric: tabular-nums;
  font-size: 11px;
}

.log-btn {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--color-text-muted);
}

.log-btn:hover {
  background: var(--color-panel-hover);
  color: var(--color-text);
}

.log-btn svg {
  width: 14px;
  height: 14px;
}

/* Action section */
.action-section {
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  flex-shrink: 0;
}

.download-btn {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  transition: all 0.2s ease;
}

.download-btn:hover:not(:disabled) {
  background: var(--color-panel-hover);
  border-color: var(--color-text-dim);
}

.download-btn.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.download-btn svg {
  width: 18px;
  height: 18px;
}

/* ============================================
   PREMIUM GENERATE BUTTON - Luxurious Black + Purple Glow
   ============================================ */
.generate-btn {
  position: relative;
  width: 100%;
  padding: 16px 24px;
  background: linear-gradient(
    165deg,
    #1a1a1a 0%,
    #0a0a0a 40%,
    #151515 100%
  );
  border: 1px solid rgba(88, 101, 242, 0.15);
  border-radius: var(--radius-lg);
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.95);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
  box-shadow: 
    0 4px 15px rgba(0, 0, 0, 0.5),
    0 1px 3px rgba(0, 0, 0, 0.3),
    0 0 20px rgba(88, 101, 242, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.generate-btn:hover:not(:disabled) {
  transform: translateY(-3px);
  border-color: rgba(88, 101, 242, 0.4);
  box-shadow: 
    0 12px 35px rgba(0, 0, 0, 0.6),
    0 4px 25px rgba(88, 101, 242, 0.25),
    0 0 40px rgba(88, 101, 242, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.generate-btn:active:not(:disabled) {
  transform: translateY(-1px);
}

.generate-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.generate-btn.generating {
  background: linear-gradient(165deg, #0d0d0d 0%, #050505 100%);
  border-color: rgba(88, 101, 242, 0.5);
  color: rgba(88, 101, 242, 0.9);
}

.gen-content {
  position: relative;
  z-index: 5;
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

/* Shimmer sweep effect - purple tint */
.gen-shimmer {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    105deg,
    transparent 20%,
    rgba(255, 255, 255, 0.02) 30%,
    rgba(88, 101, 242, 0.12) 50%,
    rgba(255, 255, 255, 0.02) 70%,
    transparent 80%
  );
  background-size: 250% 100%;
  animation: shimmer-sweep 3s ease-in-out infinite;
}

@keyframes shimmer-sweep {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Static ambient glow - purple */
.gen-glow {
  position: absolute;
  inset: 0;
  background: 
    radial-gradient(ellipse at 30% 0%, rgba(88, 101, 242, 0.12) 0%, transparent 50%),
    radial-gradient(ellipse at 70% 100%, rgba(88, 101, 242, 0.08) 0%, transparent 50%);
  opacity: 1;
}

/* Pulsing glow - purple */
.gen-glow-pulse {
  position: absolute;
  inset: -30%;
  background: radial-gradient(
    ellipse at 50% 50%,
    rgba(88, 101, 242, 0.2) 0%,
    rgba(88, 101, 242, 0.08) 30%,
    transparent 60%
  );
  opacity: 0;
  animation: gen-pulse 2.5s ease-in-out infinite;
  pointer-events: none;
}

.generate-btn:not(:disabled) .gen-glow-pulse {
  opacity: 1;
}

@keyframes gen-pulse {
  0%, 100% {
    transform: scale(0.9);
    opacity: 0.4;
  }
  50% {
    transform: scale(1.3);
    opacity: 1;
  }
}

/* Rotating light rays - purple */
.gen-rays {
  position: absolute;
  inset: -100%;
  background: conic-gradient(
    from 0deg at 50% 50%,
    transparent 0deg,
    rgba(88, 101, 242, 0.06) 30deg,
    transparent 60deg,
    transparent 120deg,
    rgba(88, 101, 242, 0.06) 150deg,
    transparent 180deg,
    transparent 240deg,
    rgba(88, 101, 242, 0.06) 270deg,
    transparent 300deg,
    transparent 360deg
  );
  animation: rays-rotate 12s linear infinite;
  pointer-events: none;
}

@keyframes rays-rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Floating sparkle particles - purple */
.gen-sparkle {
  position: absolute;
  width: 3px;
  height: 3px;
  background: rgba(120, 130, 255, 0.9);
  border-radius: 50%;
  box-shadow: 
    0 0 6px rgba(88, 101, 242, 0.8),
    0 0 12px rgba(88, 101, 242, 0.4);
  pointer-events: none;
  opacity: 0;
}

.generate-btn:not(:disabled) .gen-sparkle {
  animation: sparkle-float 4s ease-in-out infinite;
}

.gen-sparkle-1 {
  left: 15%;
  animation-delay: 0s !important;
}

.gen-sparkle-2 {
  left: 50%;
  animation-delay: 1.3s !important;
}

.gen-sparkle-3 {
  left: 85%;
  animation-delay: 2.6s !important;
}

@keyframes sparkle-float {
  0%, 100% {
    opacity: 0;
    transform: translateY(20px) scale(0);
  }
  10% {
    opacity: 1;
    transform: translateY(15px) scale(1);
  }
  50% {
    opacity: 0.8;
    transform: translateY(-5px) scale(0.8);
  }
  90% {
    opacity: 0.3;
    transform: translateY(-15px) scale(0.5);
  }
}

/* Premium animated border - purple */
.generate-btn:not(:disabled)::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: var(--radius-lg);
  padding: 1px;
  background: linear-gradient(
    135deg,
    rgba(88, 101, 242, 0.6) 0%,
    rgba(120, 130, 255, 0.2) 25%,
    rgba(88, 101, 242, 0.4) 50%,
    rgba(120, 130, 255, 0.2) 75%,
    rgba(88, 101, 242, 0.6) 100%
  );
  background-size: 300% 300%;
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: gen-border-glow 3s ease-in-out infinite;
}

/* Secondary inner glow border */
.generate-btn:not(:disabled)::after {
  content: '';
  position: absolute;
  inset: 1px;
  border-radius: calc(var(--radius-lg) - 1px);
  background: linear-gradient(
    180deg,
    rgba(88, 101, 242, 0.08) 0%,
    transparent 50%
  );
  pointer-events: none;
}

@keyframes gen-border-glow {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.spinner {
  width: 18px;
  height: 18px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.generation-error {
  padding: var(--spacing-sm);
  background: rgba(240, 71, 71, 0.1);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--color-error);
}

/* Saving state - green glow */
.generate-btn.saving {
  background: linear-gradient(165deg, #0d1a0d 0%, #050a05 100%);
  border-color: rgba(67, 181, 129, 0.5);
  color: rgba(67, 181, 129, 0.9);
}

.generate-btn.saving .gen-glow {
  background: 
    radial-gradient(ellipse at 30% 0%, rgba(67, 181, 129, 0.12) 0%, transparent 50%),
    radial-gradient(ellipse at 70% 100%, rgba(67, 181, 129, 0.08) 0%, transparent 50%);
}

.generate-btn.saving .gen-glow-pulse {
  background: radial-gradient(
    ellipse at 50% 50%,
    rgba(67, 181, 129, 0.2) 0%,
    rgba(67, 181, 129, 0.08) 30%,
    transparent 60%
  );
}
</style>
