import { describe, expect, it } from "vitest";
import { bytes, duration, ms, pct, when } from "./format";

describe("format helpers", () => {
  it("formats sizes", () => {
    expect(bytes(512)).toBe("512 B");
    expect(bytes(1536)).toBe("1.5 KB");
    expect(bytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(bytes(null)).toBe("–");
  });
  it("formats latencies with sensible precision", () => {
    expect(ms(0.25)).toBe("0.25 ms");
    expect(ms(5.44)).toBe("5.4 ms");
    expect(ms(250)).toBe("250 ms");
    expect(ms(2500)).toBe("2.50 s");
    expect(ms(undefined)).toBe("–");
  });
  it("formats rates and durations", () => {
    expect(pct(0.847)).toBe("85%");
    expect(pct(null)).toBe("–");
    expect(duration(null)).toBe("–");
  });
});

describe("when()", () => {
  it("treats zone-less API timestamps as UTC and keeps explicit offsets", () => {
    const utc = when("2026-10-13T12:55:00");
    expect(utc).toBe(when("2026-10-13T12:55:00Z"));
    expect(when("2026-10-13T17:55:00+05:00")).toBe(utc);
    expect(when("2026-10-13T08:55:00-04:00")).toBe(utc);
    expect(when(null)).toBe("–");
  });
});
