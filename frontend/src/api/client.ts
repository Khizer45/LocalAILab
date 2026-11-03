// Minimal API client. All requests go to the same origin (/api), which the Vite
// dev server or nginx proxies to the local FastAPI backend.

export class ApiError extends Error {
  status: number;
  code: string;
  hint?: string;
  constructor(status: number, code: string, message: string, hint?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.hint = hint;
  }
}

const TOKEN_KEY = "local-ai-lab-token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable – session-only login */
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  const t = getToken();
  if (t) h.set("Authorization", `Bearer ${t}`);
  return h;
}

async function toError(res: Response): Promise<ApiError> {
  let body: { error?: { code?: string; message?: string; hint?: string; details?: { loc: string[]; msg: string }[] } } = {};
  try {
    body = await res.json();
  } catch {
    /* not JSON */
  }
  const e = body.error ?? {};
  let message = e.message ?? `Request failed (${res.status})`;
  if (e.details?.length) message += " " + e.details.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ");
  if (res.status === 401) window.dispatchEvent(new Event("auth-required"));
  return new ApiError(res.status, e.code ?? "error", message, e.hint);
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, ...rest } = init;
  const h = headers(rest.headers);
  if (json !== undefined) h.set("Content-Type", "application/json");
  let res: Response;
  try {
    res = await fetch(path, { ...rest, headers: h, body: json !== undefined ? JSON.stringify(json) : rest.body });
  } catch {
    throw new ApiError(0, "backend_unreachable", "Cannot reach the Local AI Lab backend.", "Start it with scripts\\start_backend.bat (or uvicorn app.main:app).");
  }
  if (!res.ok) throw await toError(res);
  if (res.status === 204) return undefined as T;
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

export type StreamEvent =
  | { type: "start"; conversation_id: number | null; user_message_id: number | null; model: string; provider: string }
  | { type: "sources"; sources: Source[]; warnings: string[]; retrieval?: { embedder: string; min_score: number; searched_documents: number } | null }
  | { type: "thinking"; content: string }
  | { type: "token"; content: string }
  | { type: "done"; message_id: number | null; status: string; metrics: Metrics }
  | { type: "error"; code: string; message: string; hint?: string };

export interface Source {
  n: number;
  document_id: number;
  filename: string;
  page: number | null;
  chunk_index: number;
  score: number;
  flagged: boolean;
  snippet: string;
}

export interface Metrics {
  ttft_ms: number | null;
  total_ms: number | null;
  load_ms: number | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  tokens_per_s: number | null;
  tokens_source: string;
  done_reason: string | null;
}

/** POST and consume an NDJSON stream, calling onEvent for every event. */
export async function streamNdjson<E = StreamEvent>(path: string, body: unknown, onEvent: (e: E) => void, signal?: AbortSignal) {
  const h = headers({ "Content-Type": "application/json" });
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: h, body: JSON.stringify(body), signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    throw new ApiError(0, "backend_unreachable", "Cannot reach the Local AI Lab backend.");
  }
  if (!res.ok) throw await toError(res);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) onEvent(JSON.parse(line) as E);
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") throw err;
  }
}

export function download(path: string) {
  // Authenticated download via fetch → blob (works with and without auth)
  return fetch(path, { headers: headers() }).then(async (res) => {
    if (!res.ok) throw await toError(res);
    const blob = await res.blob();
    const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") ?? "")?.[1] ?? "download";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  });
}
