import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { youtubeApiPlugin } from './server/youtubePlugin.ts'

export default defineConfig({
  plugins: [react(), youtubeApiPlugin()],
  server: {
    proxy: {
      // FastAPI analysis backend — keep /api/youtube on Vite plugin
      '/api/analyze': 'http://127.0.0.1:8000',
      '/api/jobs': 'http://127.0.0.1:8000',
      '/api/health': 'http://127.0.0.1:8000',
    },
  },
})
