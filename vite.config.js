import { defineConfig } from “vite”;
import react from “@vitejs/plugin-react”;
import { VitePWA } from “vite-plugin-pwa”;
import { resolve } from “path”;
import fs from “fs”;

function copyWasm() {
return {
name: “copy-mediapipe-wasm”,
buildStart() {
const src  = resolve(“node_modules/@mediapipe/tasks-vision/wasm”);
const dest = resolve(“public/wasm”);
if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
for (const f of fs.readdirSync(src)) {
fs.copyFileSync(resolve(src, f), resolve(dest, f));
}
},
};
}

export default defineConfig({
plugins: [
copyWasm(),
react(),
VitePWA({
registerType: “autoUpdate”,
manifest: {
name: “Step-Up Counter”,
short_name: “StepUp”,
description: “AI-powered step-up rep counter using MediaPipe Pose”,
theme_color: “#08080F”,
background_color: “#08080F”,
display: “standalone”,
orientation: “portrait”,
start_url: “/”,
icons: [
{ src: “/icons/icon-192.png”, sizes: “192x192”, type: “image/png”, purpose: “any maskable” },
{ src: “/icons/icon-512.png”, sizes: “512x512”, type: “image/png”, purpose: “any maskable” },
],
},
workbox: {
globPatterns: [”**/*.{js,css,html,svg,png,wasm}”],
maximumFileSizeToCacheInBytes: 15 * 1024 * 1024,
runtimeCaching: [
{
urlPattern: /^https://storage.googleapis.com/mediapipe-models/.*/i,
handler: “CacheFirst”,
options: {
cacheName: “mediapipe-models”,
expiration: { maxEntries: 3, maxAgeSeconds: 60 * 60 * 24 * 30 },
cacheableResponse: { statuses: [0, 200] },
},
},
],
},
}),
],
optimizeDeps: {
exclude: [”@mediapipe/tasks-vision”],
},
server: {
headers: {
“Cross-Origin-Opener-Policy”: “same-origin”,
“Cross-Origin-Embedder-Policy”: “require-corp”,
},
},
});