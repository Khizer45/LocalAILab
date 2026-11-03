import { AlertTriangle, Bot, Flag, Info, ShieldAlert, Wrench } from "lucide-react";
import type { TraceEvent } from "../api/agents";
import { ms, num } from "../lib/format";
import { Badge, cx } from "./ui";

function Json({ value }: { value: unknown }) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return <pre className="mt-1 max-h-56 overflow-auto rounded border border-line bg-bg px-2.5 py-1.5 text-[12px] leading-snug whitespace-pre-wrap break-words">{text}</pre>;
}

/** Step-by-step view of one agent episode: model turns, tool calls and results, final answer. */
export function TraceView({ events, live }: { events: TraceEvent[]; live?: boolean }) {
  return (
    <ol className="space-y-2.5">
      {events.map((e, i) => {
        if (e.type === "start") {
          return (
            <li key={i} className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
              <Badge tone={e.mode === "native" ? "teal" : "violet"}>{e.mode} tool calling</Badge>
              <span>{e.tools.length ? `${e.tools.length} tools: ${e.tools.join(", ")}` : "no tools"}</span>
            </li>
          );
        }
        if (e.type === "note") {
          return <li key={i} className="flex gap-2 text-[12.5px] text-violet"><Info className="size-4 shrink-0" aria-hidden />{e.message}</li>;
        }
        if (e.type === "model") {
          return (
            <li key={i} className="rounded-md border border-line bg-panel px-3 py-2">
              <div className="flex flex-wrap items-center gap-2 text-[12px] text-muted">
                <Bot className="size-4 text-sky" aria-hidden /> <span className="text-ink">Model · step {e.step}</span>
                <span className="num">{ms(e.ms)}</span>
                {e.output_tokens != null && <span className="num">{e.output_tokens} tokens</span>}
                {e.tokens_per_s != null && <span className="num">{num(e.tokens_per_s)} t/s</span>}
                {e.parsed === "invalid" && <Badge tone="coral">not valid protocol JSON</Badge>}
              </div>
              {e.content && (e.parsed === undefined || e.parsed === "invalid") && <div className="mt-1 text-[13px] whitespace-pre-wrap break-words">{e.content}</div>}
              {e.tool_calls.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {e.tool_calls.map((c, j) => <code key={j} className="rounded bg-raised px-1.5 py-0.5 text-[12px] text-amber">{c.name}({typeof c.arguments === "string" ? c.arguments : JSON.stringify(c.arguments)})</code>)}
                </div>
              )}
            </li>
          );
        }
        if (e.type === "tool") {
          return (
            <li key={i} className={cx("ml-5 rounded-md border px-3 py-2", e.valid ? "border-line bg-bg/40" : "border-coral/40 bg-coral/5")}>
              <div className="flex flex-wrap items-center gap-2 text-[12px] text-muted">
                <Wrench className="size-4 text-amber" aria-hidden /> <span className="text-ink">Tool {e.name}</span>
                <span className="num">{ms(e.ms)}</span>
                {e.risky && <Badge tone="coral"><ShieldAlert className="size-3" />side effects (mocked)</Badge>}
                {e.unknown && <Badge tone="coral">unknown tool</Badge>}
                {!e.valid && !e.unknown && <Badge tone="coral">invalid arguments</Badge>}
              </div>
              {e.errors.length > 0 && <div className="mt-1 text-[12px] text-coral">{e.errors.join("; ")}</div>}
              <details className="mt-1 text-[12.5px]">
                <summary className="cursor-pointer text-muted">Arguments and result</summary>
                <div className="grid gap-2 md:grid-cols-2">
                  <div><div className="text-[11.5px] text-faint mt-1">Arguments</div><Json value={e.arguments} /></div>
                  <div><div className="text-[11.5px] text-faint mt-1">Result</div><Json value={e.result} /></div>
                </div>
              </details>
            </li>
          );
        }
        if (e.type === "final") {
          const stopped = e.stop_reason !== "final";
          return (
            <li key={i} className={cx("rounded-md border px-3 py-2", stopped ? "border-amber/40 bg-amber/5" : "border-teal/40 bg-teal/5")}>
              <div className="flex flex-wrap items-center gap-2 text-[12px] text-muted">
                <Flag className={cx("size-4", stopped ? "text-amber" : "text-teal")} aria-hidden />
                <span className="text-ink">{stopped ? `Stopped: ${e.stop_reason.replace("_", " ")}` : "Final answer"}</span>
                {e.totals.total_ms != null && <span className="num">{ms(e.totals.total_ms)} total</span>}
                {e.totals.steps != null && <span className="num">{e.totals.steps} steps</span>}
                {e.totals.tool_calls != null && <span className="num">{e.totals.tool_calls} tool calls</span>}
              </div>
              {e.answer && <div className="mt-1 text-[13.5px] whitespace-pre-wrap break-words">{e.answer}</div>}
              {e.error && <div className="mt-1 text-[12.5px] text-coral">{e.error}</div>}
            </li>
          );
        }
        return (
          <li key={i} className="flex gap-2 rounded-md border border-coral/40 bg-coral/5 px-3 py-2 text-[13px]">
            <AlertTriangle className="size-4 shrink-0 text-coral" aria-hidden />
            <span>{e.message}{e.hint && <span className="block text-muted text-[12.5px]">{e.hint}</span>}</span>
          </li>
        );
      })}
      {live && <li className="text-[12.5px] text-muted">Working…</li>}
    </ol>
  );
}
