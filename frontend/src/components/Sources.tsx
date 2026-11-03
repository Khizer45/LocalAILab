import { AlertTriangle, FileText } from "lucide-react";
import { useState } from "react";
import type { Metrics, Source } from "../api/client";
import { ms, num } from "../lib/format";
import { cx } from "./ui";

export function SourceList({ sources, warnings }: { sources: Source[]; warnings?: string[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!sources.length && !warnings?.length) return null;
  return (
    <div className="mt-3 space-y-2">
      {warnings?.map((w) => (
        <div key={w} className="flex gap-2 text-[12.5px] text-amber"><AlertTriangle className="size-3.5 mt-0.5 shrink-0" aria-hidden />{w}</div>
      ))}
      {sources.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {sources.map((s) => (
            <button
              key={s.n}
              onClick={() => setOpen(open === s.n ? null : s.n)}
              aria-expanded={open === s.n}
              className={cx("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[12.5px]",
                open === s.n ? "border-amber/60 bg-amber/10" : "border-line hover:border-line-strong", s.flagged && "border-coral/50")}
            >
              <span className="text-amber font-medium">S{s.n}</span>
              <FileText className="size-3.5 text-muted" aria-hidden />
              <span className="max-w-[220px] truncate">{s.filename}</span>
              {s.page != null && <span className="text-muted">p. {s.page}</span>}
              <span className="num text-faint">{s.score.toFixed(2)}</span>
            </button>
          ))}
        </div>
      )}
      {open != null && (() => {
        const s = sources.find((x) => x.n === open)!;
        return (
          <blockquote className="rounded-md border border-line bg-bg px-3 py-2 text-[12.5px] text-muted whitespace-pre-wrap">
            <div className="mb-1 text-ink">{s.filename}{s.page != null ? `, page ${s.page}` : ""}, chunk {s.chunk_index + 1}, similarity {s.score.toFixed(3)}</div>
            {s.snippet}{s.snippet.length >= 600 ? "…" : ""}
          </blockquote>
        );
      })()}
    </div>
  );
}

export function MetricsLine({ metrics, model }: { metrics: Metrics | null; model?: string | null }) {
  if (!metrics) return null;
  const est = metrics.tokens_source === "estimated";
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-0.5 text-[12px] text-faint num">
      {model && <span>{model}</span>}
      {metrics.ttft_ms != null && <span title="Time to first token, measured by Local AI Lab">first token {ms(metrics.ttft_ms)}</span>}
      {metrics.total_ms != null && <span title="Total response time, measured">total {ms(metrics.total_ms)}</span>}
      {metrics.tokens_per_s != null && <span title={est ? "Estimated from characters ÷ 4" : "Reported by the runtime"}>{num(metrics.tokens_per_s)} tok/s{est ? " (est.)" : ""}</span>}
      {metrics.output_tokens != null && <span>{metrics.output_tokens} tokens</span>}
      {metrics.load_ms != null && metrics.load_ms > 50 && <span title="Model load time reported by Ollama">load {ms(metrics.load_ms)}</span>}
      {metrics.done_reason === "length" && <span className="text-amber">stopped at max tokens</span>}
    </div>
  );
}
