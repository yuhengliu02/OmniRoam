<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useApi } from '@/composables/useApi'
import { useAppState } from '@/stores/appState'
import TopBar from '@/components/TopBar.vue'
import Gallery from '@/components/Gallery.vue'
import Workspace from '@/components/Workspace.vue'
import Settings from '@/components/Settings.vue'
import Footer from '@/components/Footer.vue'

const api = useApi()
const { state, updateSystemStatus, setVideos, setOnGenerationComplete } = useAppState()

// Mobile detection
const isMobile = ref(false)

// Track backend connectivity
const wasBackendOnline = ref(false)
const videosLoaded = ref(false)

// VRAM update timers
let vramUpdateTimer20s: number | null = null
let vramUpdateTimer10s: number | null = null

function checkMobile() {
  // Check if portrait orientation or narrow screen
  isMobile.value = window.innerHeight > window.innerWidth || window.innerWidth < 1024
}

// Status polling
let statusPollInterval: number | null = null

async function pollStatus() {
  try {
    const status = await api.getStatus()
    updateSystemStatus(status)
    
    // If backend just came online (or videos not loaded yet), reload videos
    if (!wasBackendOnline.value || !videosLoaded.value) {
      console.log('[App] Backend is online, loading videos...')
      await loadVideos()
      wasBackendOnline.value = true
    }
    
    // If we're in saving state, keep polling for videos until the pending video appears
    if (state.isSaving && state.pendingVideoFilename) {
      await loadVideos()
    }
  } catch (err) {
    // Backend is offline - reset flag so we reload videos when it comes back
    wasBackendOnline.value = false
    console.error('Status poll failed:', err)
  }
}

async function loadVideos() {
  try {
    const response = await api.getVideos()
    setVideos(response.videos) // This will also check and complete saving if pending video found
    videosLoaded.value = true
    console.log('[App] Videos loaded:', response.videos.length, 'videos')
  } catch (err) {
    videosLoaded.value = false
    console.error('Failed to load videos:', err)
  }
}

// Schedule VRAM update after generation starts (20 seconds)
function scheduleVramUpdate20s() {
  // Clear any existing timer
  if (vramUpdateTimer20s !== null) {
    clearTimeout(vramUpdateTimer20s)
  }
  vramUpdateTimer20s = window.setTimeout(async () => {
    console.log('[App] Updating VRAM status (20s after generation start)')
    try {
      const status = await api.getStatus()
      updateSystemStatus(status)
    } catch (err) {
      console.error('Failed to update VRAM:', err)
    }
    vramUpdateTimer20s = null
  }, 20000)
}

// Schedule VRAM update after saving complete (10 seconds for GPU cache clear)
function scheduleVramUpdate10s() {
  // Clear any existing timer
  if (vramUpdateTimer10s !== null) {
    clearTimeout(vramUpdateTimer10s)
  }
  vramUpdateTimer10s = window.setTimeout(async () => {
    console.log('[App] Updating VRAM status (10s after saving complete)')
    try {
      const status = await api.getStatus()
      updateSystemStatus(status)
    } catch (err) {
      console.error('Failed to update VRAM:', err)
    }
    vramUpdateTimer10s = null
  }, 10000)
}

// Handle generation completion - auto-refresh gallery and auto-play
async function handleGenerationComplete(videoFilename: string) {
  console.log('[App] Generation completed, waiting for video in gallery:', videoFilename)
  
  // Acknowledge completion to reset the flag on server
  try {
    await api.acknowledgeCompletion()
  } catch (err) {
    console.error('Failed to acknowledge completion:', err)
  }
  
  // Immediately try to load videos
  await loadVideos()
  
  // If video not found yet, it will be picked up by polling in pollStatus
  // The saving state will be released when video appears in gallery (handled by checkAndCompleteSaving)
  
  console.log('[App] Waiting for video to appear in gallery:', videoFilename)
}

// Watch for generation start to schedule VRAM update
watch(
  () => state.isGenerating,
  (isGenerating, wasGenerating) => {
    if (isGenerating && !wasGenerating) {
      // Generation just started, schedule VRAM update in 20 seconds
      scheduleVramUpdate20s()
    }
  }
)

// Watch for saving complete to schedule VRAM update (10 seconds after GPU cache clears)
watch(
  () => state.isSaving,
  (isSaving, wasSaving) => {
    if (!isSaving && wasSaving) {
      // Saving just completed, schedule VRAM update in 10 seconds
      console.log('[App] Saving completed, scheduling VRAM update in 10s')
      scheduleVramUpdate10s()
    }
  }
)

onMounted(() => {
  checkMobile()
  window.addEventListener('resize', checkMobile)
  
  // Set up completion callback
  setOnGenerationComplete(handleGenerationComplete)
  
  // Initial load
  pollStatus()
  loadVideos()
  
  // Start polling (every 1 second for more responsive completion detection)
  statusPollInterval = window.setInterval(pollStatus, 1000)
})

onUnmounted(() => {
  window.removeEventListener('resize', checkMobile)
  
  // Clear completion callback
  setOnGenerationComplete(null)
  
  if (statusPollInterval !== null) {
    clearInterval(statusPollInterval)
  }
  
  // Clear VRAM update timers
  if (vramUpdateTimer20s !== null) {
    clearTimeout(vramUpdateTimer20s)
  }
  if (vramUpdateTimer10s !== null) {
    clearTimeout(vramUpdateTimer10s)
  }
})
</script>

<template>
  <!-- Mobile Warning Overlay -->
  <div v-if="isMobile" class="mobile-warning">
    <svg class="mobile-warning-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
    <p class="mobile-warning-text">
      Please use a desktop browser to access this system.
    </p>
  </div>

  <!-- Main App -->
  <div v-else class="app-container">
    <TopBar />
    
    <main class="main-content">
      <Gallery class="gallery-panel" />
      <Workspace class="workspace-panel" />
      <Settings class="settings-panel" />
    </main>
    
    <Footer />
    
    <div class="analytics-tracker">
      <a href="http://www.clustrmaps.com/map/Omniroam-personalized-studio.com" title="Visit tracker for Omniroam-personalized-studio.com">
        <img src="//www.clustrmaps.com/map_v2.png?d=4o1BpOii8ZfFXxzEHhPTcLthimIe43N4ctC7NFBrz2E" />
      </a>
    </div>
  </div>
</template>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.main-content {
  flex: 1;
  display: flex;
  gap: var(--spacing-2xl);
  padding: 0 var(--spacing-2xl);
  overflow: hidden;
  min-height: 0;
}

.gallery-panel {
  width: 420px;
  flex-shrink: 0;
}

.workspace-panel {
  flex: 1;
  min-width: 0;
}

.settings-panel {
  width: 420px;
  flex-shrink: 0;
}

.analytics-tracker {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 1px;
  height: 1px;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  overflow: hidden;
  z-index: -9999;
  transform: scale(0);
  clip: rect(0, 0, 0, 0);
}

.analytics-tracker a,
.analytics-tracker img {
  display: block !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  visibility: hidden !important;
  pointer-events: none !important;
  position: absolute !important;
  overflow: hidden !important;
}
</style>

