import { defineConfig, devices } from "@playwright/test";

// End-to-end tests against a running backend that serves the built UI, with the Ollama simulator:
//   cd backend && python -m devtools.fake_ollama --port 11500 &
//   OLLAMA_BASE_URL=http://127.0.0.1:11500 EMBEDDING_MODEL=sim-embed DEFAULT_MODEL=sim-small:1b \
//     DATA_DIR=$(mktemp -d) python -m uvicorn app.main:app --port 8000 &
//   cd frontend && npm run build && npx playwright test
// E2E_BASE_URL changes the address; PW_CHROMIUM_PATH uses an already installed Chromium.
export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000",
    trace: "retain-on-failure",
    launchOptions: process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {},
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] }, testMatch: /smoke/ },
  ],
});
