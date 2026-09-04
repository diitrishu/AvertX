import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // VITE_API_URL is injected at build time by Render's environment variables.
  // In local dev the proxy below handles /api -> localhost:8000.
  define: {
    __API_URL__: JSON.stringify(process.env.VITE_API_URL ?? ''),
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
