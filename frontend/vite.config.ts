import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dev server proxies /api to the FastAPI backend so the browser only ever
// talks to one local origin. Override with VITE_API_TARGET if needed.
const target = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Never inline fonts as data: URIs – the backend's Content-Security-Policy only allows fonts from 'self'.
  build: { assetsInlineLimit: (file: string) => (/\.(woff2?|ttf|otf)$/.test(file) ? false : undefined) },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target, changeOrigin: false } },
  },
  preview: { host: "127.0.0.1", port: 4173, proxy: { "/api": { target } } },
});
