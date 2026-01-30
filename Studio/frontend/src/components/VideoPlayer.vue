<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick, inject, type Ref } from 'vue'
import { useAppState, type ViewMode } from '@/stores/appState'

const props = defineProps<{
  src: string
  mode: ViewMode
}>()

const { setCameraOrientation } = useAppState()

// Inject shared state from parent (Workspace)
const parentVideoRef = inject<Ref<HTMLVideoElement | null>>('videoRef')
const parentIsPlaying = inject<Ref<boolean>>('isPlaying')
const parentCurrentTime = inject<Ref<number>>('currentTime')
const parentDuration = inject<Ref<number>>('duration')
const parentCurrentFrame = inject<Ref<number>>('currentFrame')
const parentTotalFrames = inject<Ref<number>>('totalFrames')

// Local video element ref
const localVideoRef = ref<HTMLVideoElement | null>(null)
const aframeVideoRef = ref<HTMLVideoElement | null>(null)

// A-Frame scene ref
const sceneReady = ref(false)

const FPS = 30

// Aspect ratio based on mode
const aspectRatio = computed(() => {
  return props.mode === 'perspective' ? '16 / 9' : '2 / 1'
})

// Video event handlers - sync with parent
function onLoadedMetadata() {
  if (localVideoRef.value) {
    if (parentDuration) parentDuration.value = localVideoRef.value.duration
    if (parentTotalFrames) parentTotalFrames.value = Math.floor(localVideoRef.value.duration * FPS)
  }
}

function onTimeUpdate() {
  if (localVideoRef.value) {
    if (parentCurrentTime) parentCurrentTime.value = localVideoRef.value.currentTime
    if (parentCurrentFrame) parentCurrentFrame.value = Math.floor(localVideoRef.value.currentTime * FPS)
  }
}

function onPlay() {
  if (parentIsPlaying) parentIsPlaying.value = true
}

function onPause() {
  if (parentIsPlaying) parentIsPlaying.value = false
}

// Sync video ref with parent
watch(localVideoRef, (newRef) => {
  if (parentVideoRef && newRef) {
    parentVideoRef.value = newRef
  }
}, { immediate: true })

// A-Frame camera tracking
let cameraTrackingInterval: number | null = null

function startCameraTracking() {
  if (cameraTrackingInterval) return
  
  cameraTrackingInterval = window.setInterval(() => {
    const camera = document.querySelector('#perspective-camera')
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

// Sync A-Frame video with main video
watch(() => props.mode, async (newMode) => {
  if (newMode === 'perspective') {
    await nextTick()
    sceneReady.value = true
    startCameraTracking()
  } else {
    stopCameraTracking()
    sceneReady.value = false
  }
})

// Update A-Frame video when main video updates (use parent's currentTime)
watch(() => parentCurrentTime?.value, (time) => {
  if (aframeVideoRef.value && props.mode === 'perspective' && time !== undefined) {
    if (Math.abs(aframeVideoRef.value.currentTime - time) > 0.1) {
      aframeVideoRef.value.currentTime = time
    }
  }
})

watch(() => parentIsPlaying?.value, (playing) => {
  if (aframeVideoRef.value && props.mode === 'perspective') {
    if (playing) {
      aframeVideoRef.value.play()
    } else {
      aframeVideoRef.value.pause()
    }
  }
})

onMounted(() => {
  if (props.mode === 'perspective') {
    startCameraTracking()
  }
})

onUnmounted(() => {
  stopCameraTracking()
  // Clear parent ref when unmounting
  if (parentVideoRef) {
    parentVideoRef.value = null
  }
})
</script>

<template>
  <div class="video-player">
    <!-- ERP Mode -->
    <div v-if="mode === 'erp'" class="video-container" :style="{ aspectRatio }">
      <video
        ref="localVideoRef"
        :src="src"
        @loadedmetadata="onLoadedMetadata"
        @timeupdate="onTimeUpdate"
        @play="onPlay"
        @pause="onPause"
        loop
        playsinline
      />
    </div>

    <!-- Perspective Mode (A-Frame) -->
    <div v-else class="video-container perspective" :style="{ aspectRatio }">
      <video
        ref="localVideoRef"
        :src="src"
        @loadedmetadata="onLoadedMetadata"
        @timeupdate="onTimeUpdate"
        @play="onPlay"
        @pause="onPause"
        loop
        playsinline
        style="display: none;"
      />
      
      <a-scene 
        v-if="sceneReady"
        embedded 
        vr-mode-ui="enabled: false"
        class="aframe-scene"
      >
        <a-assets>
          <video 
            ref="aframeVideoRef"
            id="pano-video" 
            :src="src" 
            loop 
            crossorigin="anonymous"
            playsinline
          />
        </a-assets>
        
        <a-videosphere 
          src="#pano-video" 
          rotation="0 -90 0"
        />
        
        <a-entity 
          id="perspective-camera"
          camera 
          look-controls="reverseMouseDrag: false"
          position="0 0 0"
        />
      </a-scene>

      <!-- Fallback when scene not ready -->
      <div v-if="!sceneReady" class="perspective-loading">
        Loading perspective view...
      </div>
    </div>

    <!-- Content tag -->
    <div class="content-tag">generated video</div>
  </div>
</template>

<style scoped>
.video-player {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.video-container {
  flex: 1;
  min-height: 0;
  background: black;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.video-container video {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-container.perspective {
  background: #111;
}

.aframe-scene {
  position: absolute;
  inset: 0;
}

.perspective-loading {
  color: var(--color-text-muted);
  font-size: 14px;
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
</style>

