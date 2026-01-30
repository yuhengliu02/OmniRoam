<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as THREE from 'three'

const props = defineProps<{
  active: boolean
  yaw: number
  pitch: number
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let hemisphere: THREE.Mesh
let lightBeam: THREE.Mesh
let spotLight: THREE.Mesh
let targetDot: THREE.Mesh
let targetGlow: THREE.Mesh
let animationId: number

function init() {
  if (!canvasRef.value) return
  
  // Scene
  scene = new THREE.Scene()
  scene.background = null // Transparent background
  
  // Camera - positioned to see only upper hemisphere (dome view)
  camera = new THREE.PerspectiveCamera(40, 1.8, 0.1, 100)
  camera.position.set(0, 0.8, 2.8)
  camera.lookAt(0, 0.4, 0)
  
  // Renderer
  renderer = new THREE.WebGLRenderer({ 
    canvas: canvasRef.value, 
    antialias: true, 
    alpha: true 
  })
  renderer.setSize(180, 100)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(0x000000, 0)
  
  // Create hemisphere dome (upper half only) - wireframe
  const hemisphereGeometry = new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2)
  const hemisphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x5865f2,
    wireframe: true,
    transparent: true,
    opacity: 0.25,
  })
  hemisphere = new THREE.Mesh(hemisphereGeometry, hemisphereMaterial)
  scene.add(hemisphere)
  
  // Add semi-transparent inner surface for the dome
  const innerGeometry = new THREE.SphereGeometry(0.99, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2)
  const innerMaterial = new THREE.MeshBasicMaterial({
    color: 0x5865f2,
    transparent: true,
    opacity: 0.03,
    side: THREE.BackSide,
  })
  const innerHemisphere = new THREE.Mesh(innerGeometry, innerMaterial)
  scene.add(innerHemisphere)
  
  // Create base circle (ground plane)
  const baseGeometry = new THREE.RingGeometry(0, 1, 48)
  const baseMaterial = new THREE.MeshBasicMaterial({
    color: 0x5865f2,
    transparent: true,
    opacity: 0.08,
    side: THREE.DoubleSide,
  })
  const basePlane = new THREE.Mesh(baseGeometry, baseMaterial)
  basePlane.rotation.x = -Math.PI / 2
  basePlane.position.y = 0.001
  scene.add(basePlane)
  
  // Base ring outline
  const baseRingGeometry = new THREE.RingGeometry(0.97, 1, 48)
  const baseRingMaterial = new THREE.MeshBasicMaterial({
    color: 0x5865f2,
    transparent: true,
    opacity: 0.4,
    side: THREE.DoubleSide,
  })
  const baseRing = new THREE.Mesh(baseRingGeometry, baseRingMaterial)
  baseRing.rotation.x = -Math.PI / 2
  scene.add(baseRing)
  
  // Create latitude lines at 30° and 60°
  const latitudes = [30, 60]
  latitudes.forEach(lat => {
    const radius = Math.cos(lat * Math.PI / 180)
    const y = Math.sin(lat * Math.PI / 180)
    const latGeometry = new THREE.RingGeometry(radius - 0.008, radius + 0.008, 48)
    const latMaterial = new THREE.MeshBasicMaterial({
      color: 0x5865f2,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
    })
    const latRing = new THREE.Mesh(latGeometry, latMaterial)
    latRing.rotation.x = -Math.PI / 2
    latRing.position.y = y
    scene.add(latRing)
  })
  
  // Create meridian lines (arcs from center to top)
  const meridians = [0, 45, 90, 135, 180, 225, 270, 315]
  meridians.forEach(angle => {
    const points: THREE.Vector3[] = []
    for (let i = 0; i <= 16; i++) {
      const phi = (i / 16) * (Math.PI / 2)  // From 0 to 90 degrees
      const x = Math.cos(phi) * Math.sin(angle * Math.PI / 180)
      const y = Math.sin(phi)
      const z = Math.cos(phi) * Math.cos(angle * Math.PI / 180)
      points.push(new THREE.Vector3(x, y, z))
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    const material = new THREE.LineBasicMaterial({
      color: 0x5865f2,
      transparent: true,
      opacity: 0.15,
    })
    const meridian = new THREE.Line(geometry, material)
    scene.add(meridian)
  })
  
  // === LIGHTHOUSE / FLASHLIGHT CENTER ===
  // Center light source (the "lighthouse")
  const centerGeometry = new THREE.SphereGeometry(0.06, 16, 16)
  const centerMaterial = new THREE.MeshBasicMaterial({ 
    color: 0xfae372,
  })
  const centerPoint = new THREE.Mesh(centerGeometry, centerMaterial)
  centerPoint.position.set(0, 0.02, 0)
  scene.add(centerPoint)
  
  // Center glow
  const centerGlowGeometry = new THREE.SphereGeometry(0.1, 16, 16)
  const centerGlowMaterial = new THREE.MeshBasicMaterial({
    color: 0xfae372,
    transparent: true,
    opacity: 0.3,
  })
  const centerGlow = new THREE.Mesh(centerGlowGeometry, centerGlowMaterial)
  centerGlow.position.set(0, 0.02, 0)
  scene.add(centerGlow)
  
  // === LIGHT BEAM (cone from center to target) ===
  // Create a cone geometry for the light beam
  const beamGeometry = new THREE.ConeGeometry(0.15, 1, 16, 1, true)
  const beamMaterial = new THREE.MeshBasicMaterial({
    color: 0xfae372,
    transparent: true,
    opacity: 0.15,
    side: THREE.DoubleSide,
  })
  lightBeam = new THREE.Mesh(beamGeometry, beamMaterial)
  lightBeam.position.set(0, 0.5, 0)
  scene.add(lightBeam)
  
  // Spot light effect (brighter cone inside)
  const spotGeometry = new THREE.ConeGeometry(0.08, 1, 16, 1, true)
  const spotMaterial = new THREE.MeshBasicMaterial({
    color: 0xfae372,
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide,
  })
  spotLight = new THREE.Mesh(spotGeometry, spotMaterial)
  spotLight.position.set(0, 0.5, 0)
  scene.add(spotLight)
  
  // === TARGET DOT on sphere wall ===
  const dotGeometry = new THREE.CircleGeometry(0.06, 16)
  const dotMaterial = new THREE.MeshBasicMaterial({
    color: 0xfae372,
    transparent: true,
    opacity: 0.9,
    side: THREE.DoubleSide,
  })
  targetDot = new THREE.Mesh(dotGeometry, dotMaterial)
  scene.add(targetDot)
  
  // Target glow (larger, more transparent)
  const glowGeometry = new THREE.CircleGeometry(0.12, 16)
  const glowMaterial = new THREE.MeshBasicMaterial({
    color: 0xfae372,
    transparent: true,
    opacity: 0.4,
    side: THREE.DoubleSide,
  })
  targetGlow = new THREE.Mesh(glowGeometry, glowMaterial)
  scene.add(targetGlow)
  
  // Update beam and target position
  updateBeamPosition()
  
  // Animation
  animate()
}

function updateBeamPosition() {
  if (!lightBeam || !spotLight || !targetDot || !targetGlow) return
  
  const yawRad = props.yaw * Math.PI / 180
  const pitchRad = Math.max(0, props.pitch) * Math.PI / 180  // Clamp pitch to positive
  
  // Calculate target position on hemisphere
  const radius = 0.98
  const x = Math.sin(yawRad) * Math.cos(pitchRad) * radius
  const y = Math.sin(pitchRad) * radius
  const z = Math.cos(yawRad) * Math.cos(pitchRad) * radius
  
  // Calculate beam length and position
  const targetPos = new THREE.Vector3(x, Math.max(y, 0.05), z)
  const centerPos = new THREE.Vector3(0, 0.02, 0)
  const direction = targetPos.clone().sub(centerPos)
  const length = direction.length()
  
  // Position and orient the light beam
  const midPoint = centerPos.clone().add(direction.clone().multiplyScalar(0.5))
  lightBeam.position.copy(midPoint)
  spotLight.position.copy(midPoint)
  
  // Scale beam to match distance
  lightBeam.scale.set(1, length, 1)
  spotLight.scale.set(1, length, 1)
  
  // Orient beam to point from center to target
  lightBeam.lookAt(targetPos)
  lightBeam.rotateX(Math.PI / 2)
  spotLight.lookAt(targetPos)
  spotLight.rotateX(Math.PI / 2)
  
  // Position target dot on sphere wall
  targetDot.position.copy(targetPos)
  targetGlow.position.copy(targetPos)
  
  // Orient dot to face outward (normal to sphere surface)
  targetDot.lookAt(targetPos.clone().multiplyScalar(2))
  targetGlow.lookAt(targetPos.clone().multiplyScalar(2))
}

let time = 0
function animate() {
  animationId = requestAnimationFrame(animate)
  time += 0.016
  
  if (hemisphere) {
    // Update wireframe opacity based on active state
    const targetOpacity = props.active ? 0.35 : 0.15
    const material = hemisphere.material as THREE.MeshBasicMaterial
    material.opacity += (targetOpacity - material.opacity) * 0.1
  }
  
  if (lightBeam && spotLight) {
    // Animate beam opacity
    const beamMat = lightBeam.material as THREE.MeshBasicMaterial
    const spotMat = spotLight.material as THREE.MeshBasicMaterial
    
    if (props.active) {
      beamMat.opacity = 0.12 + Math.sin(time * 2) * 0.05
      spotMat.opacity = 0.2 + Math.sin(time * 2) * 0.08
    } else {
      beamMat.opacity = 0.05
      spotMat.opacity = 0.08
    }
  }
  
  if (targetGlow) {
    // Glow pulse animation
    const scale = 1 + Math.sin(time * 3) * 0.3
    targetGlow.scale.setScalar(scale)
    
    const material = targetGlow.material as THREE.MeshBasicMaterial
    material.opacity = props.active ? 0.4 + Math.sin(time * 2) * 0.2 : 0.15
  }
  
  if (targetDot) {
    const material = targetDot.material as THREE.MeshBasicMaterial
    material.color.setHex(props.active ? 0xfae372 : 0x666666)
    material.opacity = props.active ? 0.9 : 0.4
  }
  
  renderer?.render(scene, camera)
}

watch(() => [props.yaw, props.pitch], () => {
  updateBeamPosition()
})

watch(() => props.active, () => {
  // Update colors when active state changes
  if (hemisphere) {
    const material = hemisphere.material as THREE.MeshBasicMaterial
    material.color.setHex(props.active ? 0x5865f2 : 0x444444)
  }
})

onMounted(() => {
  init()
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  renderer?.dispose()
  scene?.clear()
})
</script>

<template>
  <div class="perspective-indicator">
    <div class="hemisphere-3d">
      <canvas ref="canvasRef" class="hemisphere-canvas" />
    </div>
    
    <!-- Labels -->
    <div class="indicator-labels">
      <span class="label">
        Yaw: {{ yaw.toFixed(0) }}°
      </span>
      <span class="label">
        Pitch: {{ pitch.toFixed(0) }}°
      </span>
    </div>
  </div>
</template>

<style scoped>
.perspective-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
}

.hemisphere-3d {
  width: 180px;
  height: 100px;
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: radial-gradient(
    ellipse at 50% 100%,
    rgba(88, 101, 242, 0.08) 0%,
    transparent 70%
  );
}

.hemisphere-canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.indicator-labels {
  display: flex;
  gap: var(--spacing-md);
}

.label {
  font-size: 11px;
  color: var(--color-text-muted);
  font-variant-numeric: tabular-nums;
}
</style>
