<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useAppState, type ViewMode } from '@/stores/appState'

const props = defineProps<{
  src: string
  mode: ViewMode
  isPreset?: boolean
}>()

const { setCameraOrientation } = useAppState()

// A-Frame scene state
const sceneReady = ref(false)
const sceneKey = ref(0) // Force re-render when src changes

// ERP yaw offset to center the view
const ERP_YAW_OFFSET_DEG = -90

// Aspect ratio based on mode
const aspectRatio = computed(() => {
  return props.mode === 'perspective' ? '16 / 9' : '2 / 1'
})

// A-Frame camera tracking
let cameraTrackingInterval: number | null = null

function startCameraTracking() {
  if (cameraTrackingInterval) return
  
  cameraTrackingInterval = window.setInterval(() => {
    const camera = document.querySelector('#image-perspective-camera')
    if (camera) {
      const rotation = camera.getAttribute('rotation') as any
      if (rotation) {
        setCameraOrientation(rotation.y || 0, rotation.x || 0)
      }
    }
  }, 100)
}

function stopCameraTracking() {
  if (cameraTrackingInterval) {
    clearInterval(cameraTrackingInterval)
    cameraTrackingInterval = null
  }
}

// Reset camera view to center
function resetCameraView() {
  nextTick(() => {
    const camera = document.querySelector('#image-perspective-camera')
    if (camera) {
      // Reset rotation
      camera.setAttribute('rotation', '0 0 0')
      
      // Reset look-controls internal state
      const lc = (camera as any).components?.['look-controls']
      if (lc && lc.pitchObject && lc.yawObject) {
        lc.pitchObject.rotation.x = 0
        lc.yawObject.rotation.y = 0
      }
    }
  })
}

// Handle mode changes
watch(() => props.mode, async (newMode) => {
  if (newMode === 'perspective') {
    await nextTick()
    sceneReady.value = true
    startCameraTracking()
    // Small delay to ensure A-Frame is initialized
    setTimeout(resetCameraView, 100)
  } else {
    stopCameraTracking()
    sceneReady.value = false
  }
}, { immediate: true })

// Handle src changes - force A-Frame to reload
watch(() => props.src, () => {
  if (props.mode === 'perspective') {
    sceneKey.value++
    nextTick(() => {
      setTimeout(resetCameraView, 100)
    })
  }
})

onMounted(() => {
  if (props.mode === 'perspective') {
    sceneReady.value = true
    startCameraTracking()
  }
})

onUnmounted(() => {
  stopCameraTracking()
})
</script>

<template>
  <div class="image-viewer">
    <!-- ERP Mode - Flat Image Display -->
    <div v-if="mode === 'erp'" class="image-container" :style="{ aspectRatio }">
      <img :src="src" alt="ERP Image" />
      <div class="content-tag">
        {{ isPreset ? 'preset image' : 'uploaded image' }}
      </div>
    </div>

    <!-- Perspective Mode - A-Frame 360° View -->
    <div v-else class="image-container perspective" :style="{ aspectRatio }">
      <a-scene 
        v-if="sceneReady"
        :key="sceneKey"
        embedded 
        vr-mode-ui="enabled: false"
        renderer="antialias: true"
        class="aframe-scene"
      >
        <a-sky 
          :src="src" 
          :rotation="`0 ${ERP_YAW_OFFSET_DEG} 0`"
        />
        
        <a-entity 
          id="image-perspective-camera"
          camera 
          look-controls="reverseMouseDrag: false; magicWindowTrackingEnabled: false"
          wasd-controls="enabled: false"
          position="0 0 0"
        />
      </a-scene>

      <!-- Instruction overlay -->
      <div v-if="sceneReady" class="interaction-hint">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"/>
        </svg>
        <span>Drag to look around</span>
      </div>

      <div class="content-tag perspective-tag">
        {{ isPreset ? 'preset image' : 'uploaded image' }} · 360° view
      </div>

      <!-- Loading state -->
      <div v-if="!sceneReady" class="perspective-loading">
        <div class="loading-spinner"></div>
        <span>Loading 360° view...</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-viewer {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.image-container {
  flex: 1;
  min-height: 0;
  background: black;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.image-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.image-container.perspective {
  background: #0a0a0a;
}

.aframe-scene {
  position: absolute;
  inset: 0;
}

.content-tag {
  position: absolute;
  top: var(--spacing-md);
  right: var(--spacing-md);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: rgba(0, 0, 0, 0.7);
  border-radius: var(--radius-sm);
  font-size: 11px;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  z-index: 10;
}

.perspective-tag {
  backdrop-filter: blur(8px);
  background: rgba(0, 0, 0, 0.6);
}

.interaction-hint {
  position: absolute;
  bottom: var(--spacing-lg);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  border-radius: var(--radius-lg);
  font-size: 12px;
  color: var(--color-text-muted);
  z-index: 10;
  pointer-events: none;
  animation: fade-hint 3s ease-in-out forwards;
}

.interaction-hint svg {
  width: 16px;
  height: 16px;
  opacity: 0.7;
}

@keyframes fade-hint {
  0%, 70% {
    opacity: 1;
  }
  100% {
    opacity: 0;
  }
}

.perspective-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  color: var(--color-text-muted);
  font-size: 14px;
  background: rgba(0, 0, 0, 0.8);
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--color-border);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>


