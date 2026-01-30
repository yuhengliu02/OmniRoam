<script setup lang="ts">
import { ref } from 'vue'
import { useAppState } from '@/stores/appState'
import { useApi } from '@/composables/useApi'

// Import local preset images
import preset1 from '@/assets/1.jpg'
import preset2 from '@/assets/2.jpg'
import preset3 from '@/assets/3.jpg'
import preset4 from '@/assets/4.jpg'
import preset5 from '@/assets/5.jpg'
import preset6 from '@/assets/6.jpg'

const { setInputImage, setActiveVideo } = useAppState()
const api = useApi()

interface LocalPreset {
  id: string
  name: string
  src: string
  path: string
}

const presets: LocalPreset[] = [
  { id: '1', name: 'Preset 1', src: preset1, path: '1.jpg' },
  { id: '2', name: 'Preset 2', src: preset2, path: '2.jpg' },
  { id: '3', name: 'Preset 3', src: preset3, path: '3.jpg' },
  { id: '4', name: 'Preset 4', src: preset4, path: '4.jpg' },
  { id: '5', name: 'Preset 5', src: preset5, path: '5.jpg' },
  { id: '6', name: 'Preset 6', src: preset6, path: '6.jpg' },
]

const selectedPreset = ref<string | null>(null)
const isLoading = ref(false)

async function selectPreset(preset: LocalPreset) {
  if (isLoading.value) return
  
  isLoading.value = true
  selectedPreset.value = preset.id
  
  try {
    // Try to select via API (backend will handle preset images)
    const response = await api.selectPreset(preset.path)
    if (response.status === 'success') {
      setInputImage(
        preset.path,
        api.getInputImageUrl(preset.path),
        true
      )
      setActiveVideo(null) // Clear active video to show image
    }
  } catch (err) {
    console.error('Failed to select preset:', err)
    // Fallback: use local image directly
    setInputImage(preset.path, preset.src, true)
    setActiveVideo(null)
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="preset-images">
    <button
      v-for="preset in presets"
      :key="preset.id"
      :class="['preset-item', { selected: selectedPreset === preset.id }]"
      @click="selectPreset(preset)"
      :disabled="isLoading"
    >
      <img 
        :src="preset.src" 
        :alt="preset.name"
      />
    </button>
  </div>
</template>

<style scoped>
.preset-images {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) 0;
  width: 100%;
}

.preset-item {
  width: 100%;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 2px solid transparent;
  transition: all 0.2s ease;
  background: var(--color-panel);
  cursor: pointer;
  padding: 0;
}

.preset-item:hover {
  border-color: var(--color-border);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.preset-item.selected {
  border-color: var(--color-accent);
  box-shadow: 0 0 16px rgba(88, 101, 242, 0.4);
}

.preset-item:disabled {
  opacity: 0.6;
  cursor: wait;
}

.preset-item img {
  width: 100%;
  aspect-ratio: 2 / 1;
  object-fit: cover;
  display: block;
}
</style>
