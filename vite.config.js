import { defineConfig } from ‘vite’
import react from ‘@vitejs/plugin-react’
import { resolve } from ‘path’
import fs from ‘fs’

function copyWasm() {
return {
name: ‘copy-wasm’,
buildStart() {
const src = resolve(‘node_modules/@mediapipe/tasks-vision/wasm’)
const dest = resolve(‘public/wasm’)
if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true })
for (const f of fs.readdirSync(src)) {
fs.copyFileSync(resolve(src, f), resolve(dest, f))
}
}
}
}

export default defineConfig({
plugins: [copyWasm(), react()],
optimizeDeps: {
exclude: [’@mediapipe/tasks-vision’]
},
server: {
headers: {
‘Cross-Origin-Opener-Policy’: ‘same-origin’,
‘Cross-Origin-Embedder-Policy’: ‘require-corp’
}
}
})