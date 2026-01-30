<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as THREE from 'three'
import type { TrajectoryPreset } from '@/stores/appState'

const props = defineProps<{
  selected: TrajectoryPreset | null
  disabled?: boolean
  disabledTrajectories?: TrajectoryPreset[]  // Trajectories that are not available for the current model
}>()

const emit = defineEmits<{
  select: [trajectory: TrajectoryPreset]
}>()

function isTrajectoryDisabled(id: TrajectoryPreset): boolean {
  return props.disabled || (props.disabledTrajectories?.includes(id) ?? false)
}

function handleSelect(id: TrajectoryPreset) {
  if (!isTrajectoryDisabled(id)) {
    emit('select', id)
  }
}

const trajectories: { id: TrajectoryPreset; label: string }[] = [
  { id: 'forward', label: 'Forward' },
  { id: 'right', label: 'Right' },
  { id: 'backward', label: 'Backward' },
  { id: 'left', label: 'Left' },
  { id: 's_curve', label: 'S-Curve' },
  { id: 'loop', label: 'Loop' },
]

// Refs for canvas elements
const canvasRefs = ref<{ [key: string]: HTMLCanvasElement | null }>({})
const sceneData = ref<{ [key: string]: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; trajectory: THREE.Line; animationId: number } }>({})

function setCanvasRef(id: string, el: HTMLCanvasElement | null) {
  canvasRefs.value[id] = el
}

function createTrajectoryPoints(id: TrajectoryPreset): THREE.Vector3[] {
  const points: THREE.Vector3[] = []
  const segments = 50
  
  switch (id) {
    case 'forward':
      for (let i = 0; i <= segments; i++) {
        const t = i / segments
        points.push(new THREE.Vector3(0, 0, -t * 2))
      }
      break
      
    case 'backward':
      for (let i = 0; i <= segments; i++) {
        const t = i / segments
        points.push(new THREE.Vector3(0, 0, t * 2))
      }
      break
      
    case 'right':
      for (let i = 0; i <= segments; i++) {
        const t = i / segments
        points.push(new THREE.Vector3(t * 2, 0, 0))
      }
      break
      
    case 'left':
      for (let i = 0; i <= segments; i++) {
        const t = i / segments
        points.push(new THREE.Vector3(-t * 2, 0, 0))
      }
      break
      
    case 's_curve':
      for (let i = 0; i <= segments; i++) {
        const t = i / segments
        const x = Math.sin(t * Math.PI * 2) * 0.8
        const z = -t * 2
        points.push(new THREE.Vector3(x, 0, z))
      }
      break
      
    case 'loop':
      for (let i = 0; i <= segments; i++) {
        const t = i / segments
        const angle = t * Math.PI * 2
        const x = Math.cos(angle) * 0.8
        const z = Math.sin(angle) * 0.8
        points.push(new THREE.Vector3(x, 0, z))
      }
      break
  }
  
  return points
}

function initScene(id: TrajectoryPreset, canvas: HTMLCanvasElement) {
  // Scene
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0a0a0a)
  
  // Camera
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
  camera.position.set(1.5, 2, 2.5)
  camera.lookAt(0, 0, 0)
  
  // Renderer
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
  renderer.setSize(80, 80)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  
  // Grid
  const gridHelper = new THREE.GridHelper(3, 6, 0x333333, 0x222222)
  scene.add(gridHelper)
  
  // Create trajectory
  const points = createTrajectoryPoints(id)
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({ 
    color: 0x5865f2,
    linewidth: 2,
  })
  const trajectory = new THREE.Line(geometry, material)
  scene.add(trajectory)
  
  // Add arrow at end
  if (id !== 'loop') {
    const lastPoint = points[points.length - 1]
    const secondLastPoint = points[points.length - 2]
    const direction = new THREE.Vector3().subVectors(lastPoint, secondLastPoint).normalize()
    
    const arrowHelper = new THREE.ArrowHelper(
      direction,
      lastPoint,
      0.3,
      0x5865f2,
      0.15,
      0.1
    )
    scene.add(arrowHelper)
  }
  
  // Add start point sphere
  const startGeometry = new THREE.SphereGeometry(0.08, 16, 16)
  const startMaterial = new THREE.MeshBasicMaterial({ color: 0x43b581 })
  const startSphere = new THREE.Mesh(startGeometry, startMaterial)
  startSphere.position.copy(points[0])
  scene.add(startSphere)
  
  // Add glow effect
  const glowGeometry = new THREE.SphereGeometry(0.12, 16, 16)
  const glowMaterial = new THREE.MeshBasicMaterial({ 
    color: 0x43b581, 
    transparent: true, 
    opacity: 0.3 
  })
  const glowSphere = new THREE.Mesh(glowGeometry, glowMaterial)
  glowSphere.position.copy(points[0])
  scene.add(glowSphere)
  
  // Animation
  let time = 0
  function animate() {
    const animationId = requestAnimationFrame(animate)
    time += 0.01
    
    // Subtle rotation
    scene.rotation.y = Math.sin(time * 0.5) * 0.1
    
    // Glow pulse
    glowSphere.scale.setScalar(1 + Math.sin(time * 2) * 0.2)
    
    renderer.render(scene, camera)
    
    if (sceneData.value[id]) {
      sceneData.value[id].animationId = animationId
    }
  }
  
  const animationId = requestAnimationFrame(animate)
  
  sceneData.value[id] = { scene, camera, renderer, trajectory, animationId }
}

function cleanupScene(id: string) {
  const data = sceneData.value[id]
  if (data) {
    cancelAnimationFrame(data.animationId)
    data.renderer.dispose()
    data.scene.clear()
    delete sceneData.value[id]
  }
}

onMounted(() => {
  // Initialize scenes after a short delay to ensure canvases are rendered
  setTimeout(() => {
    trajectories.forEach(traj => {
      const canvas = canvasRefs.value[traj.id]
      if (canvas) {
        initScene(traj.id, canvas)
      }
    })
  }, 100)
})

onUnmounted(() => {
  // Cleanup all scenes
  Object.keys(sceneData.value).forEach(cleanupScene)
})
</script>

<template>
  <div :class="['trajectory-presets', { disabled: disabled }]">
    <button
      v-for="traj in trajectories"
      :key="traj.id"
      :class="['preset-btn', { selected: selected === traj.id, 'trajectory-disabled': isTrajectoryDisabled(traj.id) && !disabled }]"
      :disabled="isTrajectoryDisabled(traj.id)"
      @click="handleSelect(traj.id)"
      :title="isTrajectoryDisabled(traj.id) && !disabled ? 'Not available for Self-forcing model' : ''"
    >
      <div class="preset-glow"></div>
      <div class="preset-glow-pulse"></div>
      <div class="preset-3d">
        <canvas 
          :ref="(el) => setCanvasRef(traj.id, el as HTMLCanvasElement)"
          class="trajectory-canvas"
        />
      </div>
      <span class="preset-label">{{ traj.label }}</span>
      <span v-if="isTrajectoryDisabled(traj.id) && !disabled" class="not-available-badge">N/A</span>
    </button>
  </div>
</template>

<style scoped>
.trajectory-presets {
  display: flex;
  gap: var(--spacing-sm);
  justify-content: center;
  padding: var(--spacing-sm) 0;
  flex-wrap: wrap;
}

.preset-btn {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm);
  border-radius: var(--radius-lg);
  transition: all 0.3s ease;
  background: linear-gradient(145deg, #1a1a1a 0%, #0d0d0d 100%);
  border: 1px solid var(--color-border);
  overflow: hidden;
}

.preset-btn:hover {
  border-color: var(--color-text-dim);
  transform: translateY(-2px);
}

.preset-btn.selected {
  border-color: var(--color-accent);
}

/* Glow effects */
.preset-glow {
  position: absolute;
  inset: 0;
  background: radial-gradient(
    ellipse at 50% 80%,
    rgba(88, 101, 242, 0.15) 0%,
    transparent 70%
  );
  opacity: 0;
  transition: opacity 0.3s ease;
}

.preset-btn:hover .preset-glow,
.preset-btn.selected .preset-glow {
  opacity: 1;
}

.preset-glow-pulse {
  position: absolute;
  inset: -30%;
  background: radial-gradient(
    circle at 50% 100%,
    rgba(88, 101, 242, 0.4) 0%,
    rgba(88, 101, 242, 0.1) 30%,
    transparent 50%
  );
  opacity: 0;
  animation: preset-glow-pulse 3s ease-in-out infinite;
  pointer-events: none;
}

.preset-btn.selected .preset-glow-pulse {
  opacity: 1;
}

@keyframes preset-glow-pulse {
  0%, 100% {
    transform: scale(0.8) translateY(10%);
    opacity: 0.3;
  }
  50% {
    transform: scale(1.2) translateY(-5%);
    opacity: 0.7;
  }
}

/* Active border glow */
.preset-btn.selected::before {
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
  background-size: 200% 200%;
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: preset-border-glow 3s linear infinite;
}

@keyframes preset-border-glow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.preset-3d {
  width: 80px;
  height: 80px;
  position: relative;
  z-index: 1;
  border-radius: var(--radius-md);
  overflow: hidden;
}

.trajectory-canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.preset-label {
  position: relative;
  z-index: 1;
  font-size: 11px;
  color: var(--color-text-muted);
  font-weight: 500;
  transition: color 0.3s ease;
}

.preset-btn.selected .preset-label {
  color: var(--color-accent);
}

/* Disabled state */
.trajectory-presets.disabled {
  opacity: 0.5;
  pointer-events: none;
}

.preset-btn:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.preset-btn:disabled .preset-glow,
.preset-btn:disabled .preset-glow-pulse {
  display: none;
}

/* Trajectory-specific disabled state (when model doesn't support it) */
.preset-btn.trajectory-disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.preset-btn.trajectory-disabled .preset-3d {
  filter: grayscale(0.8);
}

.preset-btn.trajectory-disabled .preset-glow,
.preset-btn.trajectory-disabled .preset-glow-pulse {
  display: none;
}

.preset-btn.trajectory-disabled:hover {
  transform: none;
  border-color: var(--color-border);
}

.not-available-badge {
  position: absolute;
  top: 4px;
  right: 4px;
  font-size: 8px;
  font-weight: 700;
  color: var(--color-text-muted);
  background: rgba(0, 0, 0, 0.6);
  padding: 2px 4px;
  border-radius: 3px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
</style>
