import { describe, expect, it } from "vitest";
import { toPayload, type SuiteDraft } from "./Suites";

const draft = (evaluator: string): SuiteDraft => ({
  id: " my-suite ", name: "My suite", version: "", description: "", system: "", default_max_tokens: 200,
  cases: [{ id: "c1", category: "", prompt: "2+2?", reference: "4", evaluator }],
});

describe("suite builder payload", () => {
  it("parses evaluators and fills defaults", () => {
    const p = toPayload(draft('{"type": "numeric", "value": 4}'));
    expect(p.id).toBe("my-suite");
    expect(p.version).toBe("1.0.0");
    expect(p.system).toBeNull();
    expect(p.cases[0]).toMatchObject({ id: "c1", category: "general", evaluator: { type: "numeric", value: 4 } });
  });
  it("names the case whose evaluator is not valid JSON", () => {
    expect(() => toPayload(draft("{type: numeric"))).toThrow(/Case 1 \(c1\)/);
  });
});
