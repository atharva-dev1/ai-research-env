import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/reset':  'http://localhost:7860',
      '/step':   'http://localhost:7860',
      '/state':  'http://localhost:7860',
      '/health': 'http://localhost:7860',
      '/tasks':  'http://localhost:7860',
    }
  },
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
  }
})
