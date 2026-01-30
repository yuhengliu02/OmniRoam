<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppState } from '@/stores/appState'
import { useApi } from '@/composables/useApi'

const { state, setActiveVideo, clearVideos, clearInputImage, setInputImage } = useAppState()
const api = useApi()

const isClearing = ref(false)
const isSettingInput = ref<string | null>(null)

// Dynamic tooltip based on selected model
const useAsInputTitle = computed(() => {
  return state.selectedModel === 'refine' 
    ? 'Use full video as refine input' 
    : 'Use last frame as input'
})

async function handleClearAll() {
  if (isClearing.value) return
  
  if (!confirm('Clear all generated videos?')) return
  
  isClearing.value = true
  try {
    await api.clearAll()
    clearVideos()
    clearInputImage()
  } catch (err) {
    console.error('Failed to clear:', err)
    alert('Failed to clear videos')
  } finally {
    isClearing.value = false
  }
}

function selectVideo(filename: string) {
  setActiveVideo(filename)
}

function getThumbnailUrl(filename: string) {
  return api.getVideoUrl(filename)
}

async function useAsInput(filename: string, event: Event) {
  event.stopPropagation()
  
  if (isSettingInput.value) return
  
  isSettingInput.value = filename
  try {
    // Pass the current selected model to the backend
    const currentModel = state.selectedModel
    const response = await api.useGalleryVideo(filename, currentModel)
    if (response.status === 'success' && response.frame_path) {
      setInputImage(
        response.frame_path,
        api.getInputImageUrl(response.frame_path),
        false,
        response.is_video_input ?? false,
        response.original_video_filename ?? null
      )
    }
  } catch (err: any) {
    console.error('Failed to use video as input:', err)
    alert(err.message || 'Failed to use video as input')
  } finally {
    isSettingInput.value = null
  }
}
</script>

<template>
  <aside class="gallery panel">
    <div class="gallery-header">
      <h2 class="gallery-title">Gallery</h2>
    </div>

    <div class="gallery-content">
      <!-- Empty state -->
      <div v-if="state.videos.length === 0" class="empty-state">
        <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <path d="M3 9h18M9 21V9"/>
        </svg>
        <span>Empty</span>
      </div>

      <!-- Video list -->
      <div v-else class="video-list">
        <div 
          v-for="video in state.videos"
          :key="video"
          :class="['video-item', { active: state.activeVideo === video }]"
          @click="selectVideo(video)"
        >
          <div class="video-thumbnail">
            <video 
              :src="getThumbnailUrl(video)"
              muted
              preload="metadata"
            />
            <div class="video-overlay">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
            </div>
            <!-- Use as Input Button -->
            <button 
              :class="['use-input-btn', { loading: isSettingInput === video }]"
              :disabled="isSettingInput !== null || state.isGenerating || state.isSaving"
              @click="useAsInput(video, $event)"
              :title="useAsInputTitle"
            >
              <svg v-if="isSettingInput !== video" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M5 12h14"/>
                <path d="M12 5v14"/>
              </svg>
              <span v-else class="loading-spinner"></span>
            </button>
          </div>
          <div class="video-name">{{ video.replace('.mp4', '') }}</div>
        </div>
      </div>
    </div>

    <div class="gallery-footer">
      <button 
        class="btn btn-secondary clear-btn"
        :disabled="state.videos.length === 0 || isClearing"
        @click="handleClearAll"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
        </svg>
        Clear All
      </button>
    </div>
  </aside>
</template>

<style scoped>
.gallery {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.gallery-header {
  padding: var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.gallery-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
  font-family: var(--font-body);
}

.gallery-content {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-sm);
}

.video-list {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--spacing-sm);
}

.video-item {
  cursor: pointer;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 2px solid transparent;
  transition: all 0.2s ease;
}

.video-item:hover {
  border-color: var(--color-border);
}

.video-item.active {
  border-color: var(--color-accent);
}

.video-thumbnail {
  position: relative;
  aspect-ratio: 2 / 1;
  background: var(--color-bg);
  overflow: hidden;
}

.video-thumbnail video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.video-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.3);
  opacity: 0;
  transition: opacity 0.2s ease;
}

.video-item:hover .video-overlay {
  opacity: 1;
}

.video-overlay svg {
  width: 32px;
  height: 32px;
  color: white;
}

/* Use as Input Button */
.use-input-btn {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: rgba(88, 101, 242, 0.9);
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: all 0.2s ease;
  z-index: 10;
}

.video-item:hover .use-input-btn {
  opacity: 1;
}

.use-input-btn:hover:not(:disabled) {
  background: rgba(88, 101, 242, 1);
  transform: scale(1.1);
  box-shadow: 0 0 10px rgba(88, 101, 242, 0.5);
}

.use-input-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.use-input-btn svg {
  width: 14px;
  height: 14px;
  color: white;
}

.use-input-btn.loading {
  opacity: 1;
}

.use-input-btn .loading-spinner {
  width: 12px;
  height: 12px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.video-name {
  padding: var(--spacing-xs) var(--spacing-sm);
  font-size: 11px;
  color: var(--color-text-muted);
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.gallery-footer {
  padding: var(--spacing-md);
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}

.clear-btn {
  width: 100%;
  justify-content: center;
  gap: var(--spacing-sm);
}

.clear-btn svg {
  width: 16px;
  height: 16px;
}

.empty-state {
  padding: var(--spacing-xl);
}

.empty-state-icon {
  width: 40px;
  height: 40px;
}
</style>

