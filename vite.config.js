import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    watch: {
      // Backend veya hafıza dosyaları değiştiğinde tarayıcının otomatik yenilenmesini önle
      ignored: [
        '**/agent/**', 
        '**/config/**', 
        '**/tools/**', 
        '**/model_py/**',
        '**/*.json',     
        '**/*.db',       
        '**/*.sqlite3',
        '**/*.log',
        '**/__pycache__/**'
      ]
    }
  }
})
