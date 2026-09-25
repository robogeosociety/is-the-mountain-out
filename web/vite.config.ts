import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base ('./'), because the same artifact is served from two places:
// the custom domain at the root, and robogeosociety.github.io/is-the-mountain-out/
// as the fallback. A hard-coded '/' breaks every asset on the second one.
//
// state.json and history.jsonl are NOT built artifacts — .github/workflows/
// publish.yml copies them into web/dist from /Volumes/dev/mountain/live, where
// the mini's 15-minute tick writes them. For `vite dev`, drop a copy into
// web/public/ (gitignored) and it is served at the same relative path:
//
//   scp tommydoerr@tommys-mac-mini.local:/Volumes/dev/mountain/live/state.json web/public/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5188,
    strictPort: true,
  },
})
