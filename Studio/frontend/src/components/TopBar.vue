<script setup lang="ts">
import { useAppState } from '@/stores/appState'
import logoImage from '@/assets/logo.png'

const { state, setViewMode } = useAppState()
</script>

<template>
  <header class="top-bar">
    <div class="left-section">
      <div class="logo-area">
        <div class="logo-container">
          <img :src="logoImage" alt="OmniRoam" class="logo-image" />
        </div>
        <span class="brand-name">OmniRoam</span>
      </div>
    </div>

    <div class="center-section">
      <div class="view-toggle">
        <button 
          :class="['toggle-btn', { active: state.viewMode === 'erp' }]"
          @click="setViewMode('erp')"
        >
          <span class="btn-glow"></span>
          <span class="btn-glow-pulse"></span>
          <span class="btn-glow-rays"></span>
          <span class="btn-text">ERP</span>
        </button>
        <button 
          :class="['toggle-btn', { active: state.viewMode === 'perspective' }]"
          @click="setViewMode('perspective')"
        >
          <span class="btn-glow"></span>
          <span class="btn-glow-pulse"></span>
          <span class="btn-glow-rays"></span>
          <span class="btn-text">Perspective</span>
        </button>
      </div>
    </div>

    <div class="right-section">
      <a 
        href="https://omni-roam.com" 
        class="icon-btn tooltip" 
        data-tooltip="Official Website"
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
        </svg>
      </a>
      
      <a 
        href="#" 
        class="icon-btn tooltip" 
        data-tooltip="GitHub"
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
      </a>
    </div>
  </header>
</template>

<style scoped>
.top-bar {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--spacing-xl);
  background: transparent;
  flex-shrink: 0;
}

.left-section {
  display: flex;
  align-items: center;
}

.logo-area {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.logo-container {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-image {
  width: 36px;
  height: 36px;
  object-fit: contain;
}

.brand-name {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.center-section {
  display: flex;
  justify-content: center;
}

.view-toggle {
  display: flex;
  background: linear-gradient(145deg, #1a1a1a 0%, #0d0d0d 100%);
  border: 1px solid var(--color-border);
  border-radius: 20px;
  padding: 4px;
  gap: 4px;
}

.toggle-btn {
  position: relative;
  padding: var(--spacing-sm) var(--spacing-lg);
  border-radius: 16px;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-muted);
  transition: all 0.3s ease;
  overflow: hidden;
  background: transparent;
}

.toggle-btn:hover {
  color: var(--color-text);
}

.toggle-btn.active {
  color: white;
  background: linear-gradient(145deg, rgba(88, 101, 242, 0.9) 0%, rgba(88, 101, 242, 0.7) 100%);
}

.btn-text {
  position: relative;
  z-index: 2;
}

/* Static glow background */
.btn-glow {
  position: absolute;
  inset: 0;
  background: radial-gradient(
    ellipse at 50% 100%,
    rgba(88, 101, 242, 0.4) 0%,
    transparent 70%
  );
  opacity: 0;
  transition: opacity 0.3s ease;
}

.toggle-btn:hover .btn-glow {
  opacity: 0.5;
}

.toggle-btn.active .btn-glow {
  opacity: 1;
}

/* Pulsing glow effect - active for both hover and selected */
.btn-glow-pulse {
  position: absolute;
  inset: -50%;
  background: radial-gradient(
    circle at 50% 80%,
    rgba(88, 101, 242, 0.6) 0%,
    rgba(88, 101, 242, 0.2) 25%,
    transparent 50%
  );
  opacity: 0;
  pointer-events: none;
}

.toggle-btn:hover .btn-glow-pulse,
.toggle-btn.active .btn-glow-pulse {
  opacity: 1;
  animation: btn-glow-pulse 2s ease-in-out infinite;
}

.toggle-btn.active .btn-glow-pulse {
  background: radial-gradient(
    circle at 50% 80%,
    rgba(120, 130, 255, 0.8) 0%,
    rgba(88, 101, 242, 0.3) 25%,
    transparent 50%
  );
}

@keyframes btn-glow-pulse {
  0%, 100% {
    transform: scale(0.8);
    opacity: 0.4;
  }
  50% {
    transform: scale(1.3);
    opacity: 1;
  }
}

/* Light rays effect for selected state */
.btn-glow-rays {
  position: absolute;
  inset: -100%;
  background: 
    conic-gradient(
      from 0deg at 50% 100%,
      transparent 0deg,
      rgba(88, 101, 242, 0.3) 10deg,
      transparent 20deg,
      transparent 70deg,
      rgba(88, 101, 242, 0.3) 80deg,
      transparent 90deg,
      transparent 140deg,
      rgba(88, 101, 242, 0.3) 150deg,
      transparent 160deg,
      transparent 200deg,
      rgba(88, 101, 242, 0.3) 210deg,
      transparent 220deg,
      transparent 270deg,
      rgba(88, 101, 242, 0.3) 280deg,
      transparent 290deg,
      transparent 340deg,
      rgba(88, 101, 242, 0.3) 350deg,
      transparent 360deg
    );
  opacity: 0;
  pointer-events: none;
}

.toggle-btn.active .btn-glow-rays {
  opacity: 1;
  animation: rays-rotate 8s linear infinite;
}

@keyframes rays-rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Active border glow */
.toggle-btn.active::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 16px;
  padding: 1px;
  background: linear-gradient(
    135deg,
    rgba(120, 130, 255, 1) 0%,
    rgba(88, 101, 242, 0.3) 50%,
    rgba(120, 130, 255, 1) 100%
  );
  background-size: 200% 200%;
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: border-rotate 2s linear infinite;
}

@keyframes border-rotate {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.right-section {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.icon-btn {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-md);
  color: var(--color-text-muted);
  transition: all 0.2s ease;
  text-decoration: none;
}

.icon-btn:hover {
  background: var(--color-panel-hover);
  color: var(--color-text);
}

.icon-btn svg {
  width: 20px;
  height: 20px;
}
</style>
