import { expect, test } from "@playwright/test";

test("a simulator benchmark is labelled as simulated and shows statistics", async ({ page }) => {
  await page.goto("/benchmarks");
  await page.getByRole("radio", { name: /Quick/ }).check();
  const models = page.getByRole("group", { name: "Models to compare" });
  await models.getByLabel(/sim-small:1b/).check();
  await models.getByLabel(/sim-medium:3b/).check();
  const name = `e2e benchmark ${Date.now()}`;
  await page.getByLabel("Run name").fill(name);
  await page.getByRole("button", { name: "Start benchmark" }).click();
  await expect(page).toHaveURL(/\/benchmarks\/runs\/\d+/);
  await expect(page.getByRole("heading", { level: 1, name })).toBeVisible();
  await expect(page.getByText(/(\d+) of \1 generations/)).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("SIMULATED RUN")).toBeVisible();
  await expect(page.getByText("Effective quality (95% CI)")).toBeVisible();
  await expect(page.getByText("integrity: complete")).toBeVisible();
});

test("the work agent waits for approval before writing a file", async ({ page, request }) => {
  await page.goto("/work-agent");
  await page.getByLabel("Model").selectOption("sim-medium:3b");
  const name = `e2e-${Date.now()}.md`;
  await page.getByLabel("What should it do?").fill(`Write a note. CALL write_file {"path": "${name}", "content": "hello"}`);
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.getByText("Approval needed: write_file")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Reject" }).click();
  await expect(page.getByText("rejected", { exact: true })).toBeVisible();
  await expect(page.getByText(/did not approve/).first()).toBeVisible({ timeout: 30_000 });
  const file = await (await request.get(`/api/work-agent/files/content?path=${name}`)).json();
  expect(file.error).toMatch(/not found/i); // nothing was written
});

test("a suite can be copied, edited and saved as a new version", async ({ page }) => {
  await page.goto("/benchmarks/suites");
  await page.locator("li", { hasText: "quick · v" }).getByTitle("Copy into a new suite you can edit").click();
  const id = `e2e-${Date.now()}`;
  await page.getByLabel("Suite id").fill(id);
  await page.getByRole("button", { name: "Save version" }).click();
  await expect(page.getByText(/Saved as version 1\.0\.0/)).toBeVisible();
  await page.getByLabel("Name", { exact: true }).fill("Changed name");
  await page.getByRole("button", { name: "Save version" }).click();
  await expect(page.getByText(/Saved as version 1\.0\.1/)).toBeVisible();
});
