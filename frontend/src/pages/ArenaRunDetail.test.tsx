import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ArenaResult, ArenaSummary } from "../api/arena";
import { ProtocolV2, VerdictBadge } from "./ArenaRunDetail";

const result = (key: string, name: string, val: number, test: number, source: ArenaResult["source"] = "builtin"): ArenaResult => ({
  id: Math.random(), key, source, name, status: "ok", error: null, details: null,
  metrics: { n: 140, f1_macro: test, fit_ms: 1, latency_ms_per_example: 1, latency_ms_single_p50: 1, latency_ms_single_p95: 2, throughput_per_s: 1, model_size_bytes: 1 },
  val_metrics: { n: 140, f1_macro: val, fit_ms: 1, latency_ms_per_example: 1, latency_ms_single_p50: 1, latency_ms_single_p95: 2, throughput_per_s: 1, model_size_bytes: 1 },
});

const summary: ArenaSummary = {
  protocol: "v2", task: "classification", kind: "tabular", labels: ["no", "yes"], primary_metric: "f1_macro", primary_label: "Macro F1",
  ranking: ["builtin:lr", "builtin:svm", "builtin:mlp", "baseline:dummy"], test_size: 140, higher_is_better: true,
  split: { n_train: 420, n_val: 140, n_test: 140, sha256: "abc", selection_split: "val" },
  recommendation: { key: "builtin:svm", name: "SVM", reason: "equivalent and faster", status: "provisional" },
  final: { key: "builtin:svm", name: "SVM", split: "test", metric: "f1_macro", label: "Macro F1", value: 0.82, ci: [0.75, 0.88], n: 140, unbiased: true, note: "Chosen on the validation split." },
  selection: {
    split: "validation", primary_metric: "f1_macro", primary_label: "Macro F1", higher_is_better: true, alpha: 0.05, objective: "efficiency",
    constraints: { max_latency_ms: 15 }, resampling: { method: "stratified", n_boot: 2000, seed: 42 }, leader: "builtin:lr", margin: 0.02,
    margin_rule: "absolute (default)", family_size: 2, correction: "Holm", status: "provisional", shortlist: ["builtin:lr"], warnings: [], excluded: { "builtin:rf": ["latency 19.7 ms > 15 ms"] },
    ranking: ["builtin:lr", "builtin:svm", "builtin:mlp", "builtin:rf", "baseline:dummy"],
    comparisons: {
      "builtin:svm": { n: 140, a: "builtin:lr", delta: 0.005, ci90: [-0.012, 0.018], verdict: "equivalent" },
      "builtin:mlp": { n: 140, a: "builtin:lr", delta: 0.19, ci90: [0.12, 0.27], verdict: "worse" },
      "baseline:dummy": { n: 140, a: "builtin:lr", delta: 0.45, ci90: [0.4, 0.5], verdict: "worse" },
    },
  },
};

describe("Model Arena protocol v2 view", () => {
  const results = [result("builtin:lr", "Logistic Regression", 0.85, 0.84), result("builtin:svm", "SVM", 0.845, 0.82), result("builtin:mlp", "MLP", 0.66, 0.75), result("builtin:rf", "Random Forest", 0.81, 0.82), result("baseline:dummy", "Baseline", 0.4, 0.4, "baseline")];

  it("shows the unbiased final test estimate of the recommended model", () => {
    render(<ProtocolV2 summary={summary} results={results} cls />);
    expect(screen.getByText("82.0%")).toBeTruthy();
    expect(screen.getByText(/unbiased – test rows not used/)).toBeTruthy();
  });

  it("labels each comparison with its verdict, never as a tie", () => {
    render(<ProtocolV2 summary={summary} results={results} cls />);
    const table = screen.getByRole("table");
    const row = (name: string) => within(table).getByText(name).closest("tr")!;
    expect(within(row("SVM")).getByText("equivalent")).toBeTruthy();
    expect(within(row("MLP")).getByText("worse")).toBeTruthy();
    expect(within(row("Logistic Regression")).getByText("leader")).toBeTruthy();
    expect(within(row("Random Forest")).getByText(/excluded: latency/)).toBeTruthy();
    expect(screen.queryByText(/≈ tie/)).toBeNull(); // v1 wording that suggested equivalence
    // Δ is shown as model minus leader: -0.5 points for SVM
    expect(within(row("SVM")).getByText(/-0\.5 pt/)).toBeTruthy();
  });

  it("explains an inconclusive verdict as missing evidence", () => {
    render(<VerdictBadge comp={{ n: 50, a: "x", verdict: "inconclusive" }} text={{ inconclusive: "more data needed", equivalent: "", worse: "", better: "", not_tested: "" }} />);
    expect(screen.getByTitle("more data needed").textContent).toBe("inconclusive");
  });
});
