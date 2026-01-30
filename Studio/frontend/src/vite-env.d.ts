/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// A-Frame types
declare namespace JSX {
  interface IntrinsicElements {
    'a-scene': any
    'a-assets': any
    'a-asset-item': any
    'a-sky': any
    'a-videosphere': any
    'a-entity': any
  }
}

