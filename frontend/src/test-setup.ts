import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library only cleans up automatically when test globals are enabled; do it explicitly.
afterEach(() => cleanup());
