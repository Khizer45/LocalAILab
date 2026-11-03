import { expect, test } from "@playwright/test";

// Every page renders without a JavaScript error and shows its heading.
const PAGES: [string, string][] = [
  ["/", "Overview"], ["/setup", "Setup"], ["/chat", "Chat"], ["/documents", "Documents"], ["/work-agent", "Work agent"],
  ["/benchmarks", "Benchmarks"], ["/benchmarks/suites", "Test suites"], ["/arena", "Model arena"], ["/agents", "Agent arena"],
  ["/ml", "ML lab"], ["/automation", "Automation"], ["/admin", "Administration"], ["/settings", "Settings"],
];

for (const [path, heading] of PAGES) {
  test(`page ${path} renders`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    await expect(page.getByText(/Data stays on this computer|Cloud models allowed/)).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test("the setup check lists the database schema and Ollama", async ({ page }) => {
  await page.goto("/setup");
  await expect(page.getByText("Database schema")).toBeVisible();
  await expect(page.getByText("Ollama runtime")).toBeVisible();
  await expect(page.getByText(/Qwen Research License/)).toBeVisible();
});
