import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served at the domain root by the `is-the-mountain-out` Worker (wrangler.toml),
// which also answers /state.json from the R2 binding. In `vite dev` there is
// no Worker in front, so the same path is proxied to the bucket's r2.dev URL —
// the SPA code never needs to know which one it is talking to.
const STATE_ORIGIN = 'https://pub-66d3d1f139004e29b2afcb5fba49bdb3.r2.dev'

export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5188,
    strictPort: true,
    proxy: {
      '/state.json': { target: STATE_ORIGIN, changeOrigin: true },
      '/history.jsonl': { target: STATE_ORIGIN, changeOrigin: true },
    },
  },
})
