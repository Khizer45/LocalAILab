import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Unit and component tests (jsdom). End-to-end tests live in e2e/ and run with Playwright.
export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", include: ["src/**/*.test.{ts,tsx}"], setupFiles: ["src/test-setup.ts"], restoreMocks: true },
});
