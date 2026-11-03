import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Download, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, download } from "../api/client";
import { BundleExport } from "../components/Bundles";
import type { AgentResultRow, AgentRun } from "../api/agents";
import { TraceView } from "../components/TraceView";
import { Badge, Button, cx, ErrorNotice, Notice, PageHeader, Panel, Spinner, StatusBadge, Table } from "../components/ui";
import { duration, humanize, ms, num, pct, SERIES, when } from "../lib/format";
import { chartTheme } from "./Benchmarks";

const CHECK_ORDER = ["tool_selection", "arguments", "order", "no_tools", "safety", "max_tool_calls", "final_answer", "completed"];

function Rate({ v, good = 0.8 }: { v: number | null; good?: number }) {
  if (v == null) return <span className="text-faint">–</span>;
  return <span className={cx("num", v >= good ? "text-teal" : v < 0.5 ? "text-coral" : "text-ink")}>{pct(v)}</span>;
}

export default function AgentRunDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["agent-run", id],
    queryFn: () => api<AgentRun>(`/api/agents/runs/${id}`),
    refetchInterval: (query) => (["running", "queued"].includes(query.state.data?.status ?? "") ? 1500 : false),
  });
  const [sel, setSel] = useState<number | null>(null);
  const detail = useQuery({ queryKey: ["agent-result", id, sel], queryFn: () => api<AgentResultRow>(`/api/agents/runs/${id}/results/${sel}`), enabled: sel != null });
  const run = q.data;
  const colors = useMemo(() => Object.fromEntries((run?.models ?? []).map((m, i) => [m, SERIES[i % SERIES.length]])), [run?.models]);

  if (q.isLoading) return <Spinner />;
  if (q.isError || !run) return <ErrorNotice error={q.error} />;

  const live = run.status === "running" || run.status === "queued";
  const results = run.results ?? [];
  const labels = run.check_labels ?? {};
  const taskIds = run.config.task_ids;
  const cats = [...new Set(results.map((r) => r.category))];
  const catData = cats.map((c) => ({ category: humanize(c), ...Object.fromEntries(run.summary.map((s) => [s.model, s.by_category[c] ? Math.round(s.by_category[c].success_rate * 100) : null])) }));
  const cell = (task: string, model: string) => results.filter((r) => r.task_id === task && r.model === model);
  const selected = results.find((r) => r.id === sel);
  const task = selected ? run.tasks?.[selected.task_id] : undefined;

  const remove = async () => {
    if (!confirm("Delete this agent run?")) return;
    await api(`/api/agents/runs/${run.id}`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: ["agent-runs"] });
    nav("/agents?tab=runs");
  };

  return (
    <div className="max-w-[1320px]">
      <PageHeader
        title={run.name}
        description={<>Suite <span className="text-ink">{run.suite_id}</span> v{run.suite_version} ({taskIds.length} tasks), mode {run.config.mode}, max {run.config.max_steps} steps, temperature {run.config.temperature}. Started {when(run.started_at ?? run.created_at)}{run.duration_s != null && `, took ${duration(run.duration_s)}`}.</>}
        actions={live ? <Button onClick={() => api(`/api/agents/runs/${run.id}/cancel`, { method: "POST" }).then(() => q.refetch())}>Cancel</Button> : <>
          <Button onClick={() => download(`/api/agents/runs/${run.id}/export?format=md`)}><Download className="size-4" />Report (.md)</Button>
          <Button onClick={() => download(`/api/agents/runs/${run.id}/export?format=csv`)}><Download className="size-4" />CSV</Button>
          <Button onClick={() => download(`/api/agents/runs/${run.id}/export?format=json`)}><Download className="size-4" />JSON + traces</Button>
          <BundleExport kind="agent" runId={run.id} />
          <Button variant="danger" onClick={remove}><Trash2 className="size-4" />Delete</Button>
        </>}
      />

      <div className="mb-6 rounded-lg border border-line bg-panel px-5 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={run.status} />
          <span className="num text-[13px] text-muted">{run.progress_done} of {run.progress_total} episodes</span>
          {run.current_item && <span className="text-[13px] text-muted">Now: <span className="text-ink">{run.current_item}</span></span>}
          {run.mlflow_run_id && <Badge tone="violet">MLflow {run.mlflow_run_id.slice(0, 8)}</Badge>}
        </div>
        <div className="mt-3 h-1.5 rounded-full bg-raised overflow-hidden"><div className={cx("h-full", live ? "bg-amber" : "bg-teal")} style={{ width: `${(run.progress_done / Math.max(1, run.progress_total)) * 100}%` }} /></div>
        {run.error && <div className="mt-3"><ErrorNotice error={{ message: run.error }} /></div>}
        {Object.entries(run.model_info ?? {}).filter(([, i]) => i.fallback || i.error).map(([m, i]) => (
          <div key={m} className="mt-2 text-[12.5px] text-amber">{m}: {i.error ?? i.fallback}</div>
        ))}
      </div>

      {run.warnings?.map((w) => <div key={w} className="mb-3"><Notice tone="warn">{w}</Notice></div>)}

      {run.summary.some((s) => s.safety_tasks) && (
        <Panel title="Safety" bodyClass="p-0" className="mb-6">
          <Table>
            <thead><tr><th>Model</th><th title="Share of safety-relevant tasks with a forbidden or unrequested risky tool call (95% Wilson interval)">Violation rate (95% CI)</th><th>Safety tasks</th><th title="Tasks scored 0 because of a severe violation">Hard fails</th><th title="Replies that followed the tool-call protocol without repair">Protocol compliance</th></tr></thead>
            <tbody>{run.summary.map((s) => (
              <tr key={s.model}>
                <td><span className="inline-block size-2.5 rounded-full mr-2 align-middle" style={{ background: colors[s.model] }} />{s.model}</td>
                <td className={cx("num whitespace-nowrap", (s.safety_violation_rate ?? 0) > 0 ? "text-coral" : "text-teal")}>{pct(s.safety_violation_rate ?? null)}{s.safety_violation_rate_ci && <span className="ml-1 text-[11.5px] text-faint">{pct(s.safety_violation_rate_ci[0])}–{pct(s.safety_violation_rate_ci[1])}</span>}</td>
                <td className="num">{s.safety_tasks ?? 0}</td>
                <td className={cx("num", (s.hard_fails ?? 0) > 0 && "text-coral")}>{s.hard_fails ?? 0}</td>
                <td><Rate v={s.protocol_compliance_rate ?? null} good={0.95} /></td>
              </tr>
            ))}</tbody>
          </Table>
          <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">A severe violation (calling a forbidden tool, or a risky tool the user did not ask for) scores the task 0 regardless of the answer. The upper end of the interval is what you can rule out: zero violations in 10 tasks still allows a true rate of about 28%.</div>
        </Panel>
      )}

      <Panel title="Leaderboard" bodyClass="p-0" className="mb-6">
        <Table>
          <thead><tr><th>Model</th><th>Mode</th><th>Success</th><th>Score</th><th>Right tools</th><th>Right args</th><th>Correct answer</th><th>Safety</th><th>Restraint</th><th>Avg steps</th><th>Invalid calls</th><th>Latency p50 / p95</th><th>Tok/s</th></tr></thead>
          <tbody>
            {run.summary.map((s) => (
              <tr key={s.model}>
                <td><span className="inline-block size-2.5 rounded-full mr-2 align-middle" style={{ background: colors[s.model] }} />{s.model}</td>
                <td>{s.modes.map((m) => <Badge key={m} tone={m === "native" ? "teal" : "violet"}>{m}</Badge>)}</td>
                <td className="font-medium whitespace-nowrap"><Rate v={s.success_rate} />{s.success_rate_ci && <span className="block text-[11.5px] font-normal text-faint">{pct(s.success_rate_ci[0])}–{pct(s.success_rate_ci[1])}</span>}</td>
                <td className="num">{num(s.score, 3)}</td>
                <td><Rate v={s.tool_selection_rate} /></td>
                <td><Rate v={s.argument_accuracy} /></td>
                <td><Rate v={s.final_answer_rate} /></td>
                <td><Rate v={s.safety_rate} good={1} /></td>
                <td><Rate v={s.no_tool_rate} /></td>
                <td className="num">{num(s.avg_steps, 2)}</td>
                <td className="num">{pct(s.invalid_call_rate)}{s.format_errors ? <span className="block text-[11.5px] text-faint">{s.format_errors} format errors</span> : null}</td>
                <td className="num whitespace-nowrap">{ms(s.latency_ms_p50)} / {ms(s.latency_ms_p95)}</td>
                <td className="num">{num(s.tokens_per_s_mean)}</td>
              </tr>
            ))}
          </tbody>
        </Table>
        <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">Success = all checks of a task passed. Restraint = no tool call when none was needed. Safety = no forbidden tool called (also under prompt injection). Latency is per task episode (all steps).</div>
      </Panel>

      {catData.length > 0 && (
        <Panel title="Success rate by category" className="mb-6">
          <div className="h-[260px]">
            <ResponsiveContainer>
              <BarChart data={catData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={chartTheme.axis} />
                <YAxis domain={[0, 100]} unit="%" tick={chartTheme.axis} />
                <Tooltip {...chartTheme.tooltip} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {run.summary.map((s) => <Bar key={s.model} dataKey={s.model} fill={colors[s.model]} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <Panel title="Tasks" bodyClass="p-0">
          <Table>
            <thead><tr><th>Task</th>{run.models.map((m) => <th key={m} className="max-w-[120px] truncate" title={m}>{m}</th>)}</tr></thead>
            <tbody>
              {taskIds.map((t) => (
                <tr key={t}>
                  <td className="max-w-[300px]">
                    <div className="truncate" title={run.tasks?.[t]?.prompt}>{run.tasks?.[t]?.prompt ?? t}</div>
                    <div className="text-[11.5px] text-faint">{t} · {humanize(run.tasks?.[t]?.category ?? "")}</div>
                  </td>
                  {run.models.map((m) => (
                    <td key={m} className="whitespace-nowrap">
                      {cell(t, m).map((r) => (
                        <button key={r.id} onClick={() => setSel(r.id)} title={`score ${r.score}`} className={cx("mr-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[12px] border", sel === r.id ? "border-amber" : "border-transparent hover:border-line-strong", r.success ? "bg-teal/12 text-teal" : r.status !== "ok" ? "bg-coral/15 text-coral" : "bg-coral/8 text-coral")}>
                          {r.success ? <Check className="size-3.5" aria-label="passed" /> : <X className="size-3.5" aria-label="failed" />}
                          <span className="num">{Math.round(r.score * 100)}</span>
                        </button>
                      ))}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </Panel>

        <Panel title={selected ? `${selected.model} · ${selected.task_id}` : "Trace"}>
          {!selected ? <p className="text-[13px] text-faint">Click a result on the left to see every step the agent took and which checks passed.</p> : (
            <div className="space-y-4">
              {task && (
                <div className="text-[13px]">
                  <div className="text-muted">Task</div>
                  <div>{task.prompt}</div>
                  {task.reference && <div className="mt-1 text-[12.5px] text-faint">Reference answer: {task.reference}</div>}
                </div>
              )}
              <div className="flex flex-wrap gap-1.5">
                {CHECK_ORDER.filter((c) => selected.checks?.[c]).map((c) => {
                  const ch = selected.checks![c];
                  return <Badge key={c} tone={ch.passed ? "teal" : "coral"} title={JSON.stringify(ch)}>{ch.passed ? "✓" : "✗"} {labels[c] ?? humanize(c)}</Badge>;
                })}
              </div>
              {selected.hard_fail && <Notice tone="error" title="Hard fail – severe safety violation, task scored 0" />}
              {!!selected.safety_violations?.length && (
                <ul className="space-y-0.5 text-[12.5px] text-coral">{selected.safety_violations.map((v, i) => <li key={i}>• call {v.call + 1}: <span className="num">{v.tool}</span> – {v.rule} ({v.severity})</li>)}</ul>
              )}
              {selected.protocol_ok === false && <div className="text-[12.5px] text-amber">The reply did not follow the tool-call format exactly (repaired or invalid).</div>}
              {selected.error && <Notice tone="error" title={selected.error} />}
              {detail.isLoading ? <Spinner /> : detail.data?.trace ? <TraceView events={detail.data.trace} /> : null}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
