import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Download, Trash2 } from "lucide-react";
import { Fragment, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api, download } from "../api/client";
import { BundleExport } from "../components/Bundles";
import type { BenchmarkRun } from "../api/types";
import { Badge, Button, cx, ErrorNotice, Notice, PageHeader, Panel, Select, Spinner, StatusBadge, Table } from "../components/ui";
import { duration, humanize, ms, num, pct, SERIES, when } from "../lib/format";
import { chartTheme } from "./Benchmarks";

const PROVENANCE = [
  ["Effective quality", "Mean evaluator score over all cases; failed or timed-out calls count as 0. 95% CI: bootstrap over cases (repeats averaged)."],
  ["Conditional quality", "Mean score over the calls that completed – compare it with the completion rate."],
  ["Completion rate", "Share of calls without error or timeout, with a 95% Wilson interval."],
  ["Pass rate", "Deterministic evaluator defined for each test (exact match, numeric, regex, JSON, keyword, ROUGE-L…). No LLM judge. Evaluator versions are frozen with the run."],
  ["First token, latency", "Measured by Local AI Lab around the streaming request (includes queueing and network). “Answer” = first visible token after any hidden thinking."],
  ["Tokens/s, output tokens, load time", "Reported by Ollama (eval_count / eval_duration). Marked “est.” when the runtime did not report counts."],
  ["Peak RAM, CPU", "Measured system-wide every 250 ms during each request."],
  ["Model memory", "Reported by Ollama /api/ps for the loaded model."],
];

export default function RunDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["run", id],
    queryFn: () => api<BenchmarkRun>(`/api/benchmarks/runs/${id}`),
    refetchInterval: (query) => (["running", "queued"].includes(query.state.data?.status ?? "") ? 1500 : false),
  });
  const [modelFilter, setModelFilter] = useState("");
  const [catFilter, setCatFilter] = useState("");
  const [onlyFailed, setOnlyFailed] = useState(false);
  const [open, setOpen] = useState<number | null>(null);

  const run = q.data;
  const colors = useMemo(() => Object.fromEntries((run?.models ?? []).map((m, i) => [m, SERIES[i % SERIES.length]])), [run?.models]);
  const categories = useMemo(() => [...new Set(run?.results?.map((r) => r.category) ?? [])], [run?.results]);

  if (q.isLoading) return <Spinner />;
  if (q.isError || !run) return <ErrorNotice error={q.error} />;

  const live = run.status === "running" || run.status === "queued";
  const env = (run.environment ?? {}) as Record<string, unknown>;
  const gpus = (env.gpus as { name: string }[] | undefined) ?? [];
  const results = (run.results ?? []).filter((r) => (!modelFilter || r.model === modelFilter) && (!catFilter || r.category === catFilter) && (!onlyFailed || !r.passed));

  const catData = categories.map((c) => ({ category: humanize(c), ...Object.fromEntries(run.summary.map((s) => [s.model, s.by_category[c]?.quality_score != null ? Math.round(s.by_category[c].quality_score! * 100) : null])) }));
  const scatter = run.summary.filter((s) => s.tokens_per_s_mean != null).map((s) => ({ model: s.model, x: s.tokens_per_s_mean, y: Math.round((s.effective_quality ?? s.quality_score ?? 0) * 100), z: s.peak_ram_used_mb ?? 1 }));
  const cmp = run.comparison;
  const ci = (c?: [number, number] | null) => (c ? `${pct(c[0])}–${pct(c[1])}` : null);
  const latency = run.summary.map((s) => ({ model: s.model, p50: s.latency_ms_p50, p95: s.latency_ms_p95, ttft: s.ttft_ms_p50 }));

  const remove = async () => {
    if (!confirm("Delete this run and all its results?")) return;
    await api(`/api/benchmarks/runs/${run.id}`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: ["runs"] });
    nav("/benchmarks");
  };

  return (
    <div className="max-w-[1320px]">
      <PageHeader
        title={run.name}
        description={<>Suite <span className="text-ink">{run.suite_id}</span> v{run.suite_version}, dataset hash <span className="num">{run.suite_hash.slice(0, 12)}</span>. Started {when(run.started_at ?? run.created_at)}{run.duration_s != null && `, took ${duration(run.duration_s)}`}.</>}
        actions={<>
          {live ? <Button onClick={() => api(`/api/benchmarks/runs/${run.id}/cancel`, { method: "POST" }).then(() => q.refetch())}>Cancel run</Button> : <>
            <Button onClick={() => download(`/api/benchmarks/runs/${run.id}/export?format=md`)}><Download className="size-4" />Report (.md)</Button>
            <Button onClick={() => download(`/api/benchmarks/runs/${run.id}/export?format=csv`)}><Download className="size-4" />CSV</Button>
            <Button onClick={() => download(`/api/benchmarks/runs/${run.id}/export?format=json`)}><Download className="size-4" />JSON</Button>
            <BundleExport kind="benchmark" runId={run.id} />
            <Button variant="danger" onClick={remove}><Trash2 className="size-4" />Delete</Button>
          </>}
        </>}
      />

      <div className="mb-6 rounded-lg border border-line bg-panel px-5 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={run.status} />
          <span className="num text-[13px] text-muted">{run.progress_done} of {run.progress_total} generations</span>
          {run.current_item && <span className="text-[13px] text-muted">Now: <span className="text-ink">{run.current_item}</span></span>}
          {run.mlflow_run_id && <Badge tone="violet" title="Experiment tracked in the local MLflow store">MLflow {run.mlflow_run_id.slice(0, 8)}</Badge>}
        </div>
        <div className="mt-3 h-1.5 rounded-full bg-raised overflow-hidden"><div className={cx("h-full", live ? "bg-amber" : "bg-teal")} style={{ width: `${(run.progress_done / Math.max(1, run.progress_total)) * 100}%` }} /></div>
        {run.error && <div className="mt-3"><ErrorNotice error={{ message: run.error }} /></div>}
        <div className="mt-3 text-[12.5px] text-faint">
          {String(env.os ?? "")} {env.cpu_count_logical ? `, ${env.cpu_count_logical} logical CPUs` : ""}{env.ram_total_mb ? `, ${num(Number(env.ram_total_mb) / 1024)} GB RAM` : ""}
          {`, GPU: ${gpus.length ? gpus.map((g) => g.name).join(", ") : "none detected"}`}{env.ollama_version ? `, Ollama ${env.ollama_version}` : ""}.
          {" "}Temperature {String(run.config.temperature)}, seed {String(run.config.seed ?? "none")}, {String(run.config.repeats)} repeat(s), timeout {String(run.config.timeout_s)} s, warm-up {run.config.warmup ? "on" : "off"}.
        </div>
      </div>

      {run.simulated && (
        <div className="mb-4"><Notice tone="warn" title="SIMULATED RUN">Produced by the built-in Ollama simulator for testing and demos. These numbers are not measurements of any real model – do not quote them.</Notice></div>
      )}
      {run.integrity && !run.integrity.complete && (
        <div className="mb-4"><Notice tone="error" title="Incomplete run">{run.integrity.stored_results} of {run.integrity.expected_results} results were stored ({run.integrity.duplicates} duplicates). Scores below are not a complete measurement.</Notice></div>
      )}
      <Panel title="Model comparison" bodyClass="p-0" action={run.integrity?.complete ? <Badge tone="teal" title="Every expected result was stored exactly once">integrity: complete</Badge> : undefined}>
        <Table>
          <thead>
            <tr><th>Model</th><th title="Failed calls count as 0">Effective quality (95% CI)</th><th title="Mean over completed calls only">Conditional</th><th>Completion</th><th>Pass rate</th><th title="TTFT / first visible answer token">First token p50</th><th>Latency p50 / p95</th><th>Tokens/s</th><th>Peak RAM</th><th>Model memory</th><th>Errors / timeouts</th><th>Repeat spread</th></tr>
          </thead>
          <tbody>
            {run.summary.map((s) => (
              <tr key={s.model}>
                <td className="whitespace-nowrap"><span className="inline-block size-2.5 rounded-full mr-2 align-middle" style={{ background: colors[s.model] }} />{s.model}
                  {run.model_info?.[s.model]?.parameter_size != null && <span className="text-faint text-[12px]"> {String(run.model_info[s.model].parameter_size)} {String(run.model_info[s.model].quantization ?? "")}</span>}
                  {cmp && cmp.leader === s.model && Object.keys(cmp.comparisons).length > 0 && <Badge tone="teal">leader</Badge>}
                  {cmp?.comparisons[s.model] && <span className="ml-1"><BenchVerdict c={cmp.comparisons[s.model]} margin={cmp.margin} /></span>}
                </td>
                <td className="num font-medium whitespace-nowrap">{pct(s.effective_quality ?? s.quality_score)}{ci(s.effective_quality_ci) && <span className="block text-[11.5px] font-normal text-faint">{ci(s.effective_quality_ci)}</span>}</td>
                <td className="num">{pct(s.conditional_quality ?? null)}</td>
                <td className={cx("num whitespace-nowrap", (s.completion_rate ?? 1) < 1 && "text-amber")}>{pct(s.completion_rate ?? null)}{ci(s.completion_rate_ci) && <span className="block text-[11.5px] text-faint">{ci(s.completion_rate_ci)}</span>}</td>
                <td className="num">{pct(s.pass_rate)}</td>
                <td className="num whitespace-nowrap">{ms(s.ttft_ms_p50)}{s.ttft_answer_ms_p50 != null && s.ttft_answer_ms_p50 !== s.ttft_ms_p50 && <span className="block text-[11.5px] text-faint">answer {ms(s.ttft_answer_ms_p50)}</span>}</td>
                <td className="num whitespace-nowrap">{ms(s.latency_ms_p50)} / {ms(s.latency_ms_p95)}{s.p95_unstable && <span className="text-faint" title={`Only ${s.latency_n ?? "few"} latency samples – p95 is unstable`}> *</span>}</td>
                <td className="num">{num(s.tokens_per_s_mean)}{s.tokens_source === "estimated" && <span className="text-amber"> est.</span>}</td>
                <td className="num">{s.peak_ram_used_mb ? `${num(s.peak_ram_used_mb, 0)} MB` : "–"}</td>
                <td className="num">{s.model_mem_mb ? `${num(s.model_mem_mb, 0)} MB` : "–"}</td>
                <td className={cx("num", (s.errors + s.timeouts) > 0 && "text-coral")}>{s.errors} / {s.timeouts}</td>
                <td className="num" title="Mean standard deviation of the score across repeats of the same test">{s.score_std_across_repeats == null ? "–" : num(s.score_std_across_repeats, 3)}</td>
              </tr>
            ))}
          </tbody>
        </Table>
        {cmp && Object.keys(cmp.comparisons).length > 0 && (
          <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">
            Verdicts compare each model with the leader case by case (paired bootstrap over {cmp.unit ?? "cases"}, {cmp.resamples ?? ""} resamples, {cmp.correction ?? "Holm"} correction). <span className="text-teal">equivalent</span> = within ±{Math.round(cmp.margin * 100)} percentage points; <span className="text-coral">worse</span> = difference detected; <span className="text-amber">inconclusive</span> = the suite is too small to tell – not a tie. * p95 from fewer than 20 samples.
          </div>
        )}
      </Panel>

      {run.summary.length > 0 && (
        <div className="mt-6 grid gap-6 lg:grid-cols-3">
          <Panel title="Quality vs. speed" className="lg:col-span-1">
            <div className="h-[240px]">
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name="Tokens/s" tick={chartTheme.axis} />
                  <YAxis type="number" dataKey="y" name="Quality" unit="%" domain={[0, 100]} tick={chartTheme.axis} />
                  <ZAxis type="number" dataKey="z" range={[80, 80]} />
                  <Tooltip {...chartTheme.tooltip} formatter={(v, n) => [n === "Quality" ? `${v}%` : num(Number(v)), String(n)]} labelFormatter={() => ""} />
                  <Scatter data={scatter}>{scatter.map((d) => <Cell key={d.model} fill={colors[d.model]} />)}</Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[12px] text-faint">Up and to the right is better: more correct answers, generated faster.</p>
          </Panel>
          <Panel title="Quality by category" className="lg:col-span-2">
            <div className="h-[260px]">
              <ResponsiveContainer>
                <BarChart data={catData} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="category" tick={chartTheme.axis} interval={0} />
                  <YAxis domain={[0, 100]} unit="%" tick={chartTheme.axis} />
                  <Tooltip {...chartTheme.tooltip} cursor={{ fill: "#212b38" }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {run.summary.map((s) => <Bar key={s.model} dataKey={s.model} fill={colors[s.model]} radius={[3, 3, 0, 0]} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Latency per model" className="lg:col-span-3">
            <div className="h-[220px]">
              <ResponsiveContainer>
                <BarChart data={latency} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 40 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" unit=" ms" tick={chartTheme.axis} />
                  <YAxis type="category" dataKey="model" tick={chartTheme.axis} width={130} />
                  <Tooltip {...chartTheme.tooltip} cursor={{ fill: "#212b38" }} formatter={(v) => ms(Number(v))} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="ttft" name="First token p50" fill="#6fb2f2" />
                  <Bar dataKey="p50" name="Total p50" fill="#57c2b0" />
                  <Bar dataKey="p95" name="Total p95" fill="#a28beb" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>
      )}

      <Panel className="mt-6" title="Test results"
        action={<div className="flex flex-wrap gap-2">
          <Select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)} className="w-40" aria-label="Filter by model"><option value="">All models</option>{run.models.map((m) => <option key={m}>{m}</option>)}</Select>
          <Select value={catFilter} onChange={(e) => setCatFilter(e.target.value)} className="w-44" aria-label="Filter by category"><option value="">All categories</option>{categories.map((c) => <option key={c} value={c}>{humanize(c)}</option>)}</Select>
          <label className="flex items-center gap-1.5 text-[13px] text-muted"><input type="checkbox" className="accent-amber" checked={onlyFailed} onChange={(e) => setOnlyFailed(e.target.checked)} />Failed only</label>
        </div>}
        bodyClass="p-0">
        <Table>
          <thead><tr><th /><th>Test</th><th>Model</th><th>Category</th><th>Result</th><th>Score</th><th>First token</th><th>Total</th><th>Tokens/s</th></tr></thead>
          <tbody>
            {results.slice(0, 400).map((r) => (
              <Fragment key={r.id}>
                <tr className="cursor-pointer hover:bg-raised/40" onClick={() => setOpen(open === r.id ? null : r.id)}>
                  <td className="w-6 text-faint">{open === r.id ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}</td>
                  <td>{r.case_id}{r.repeat_index > 0 && <span className="text-faint"> #{r.repeat_index + 1}</span>}</td>
                  <td className="text-muted">{r.model}</td>
                  <td className="text-muted">{humanize(r.category)}</td>
                  <td>{r.status !== "ok" ? <StatusBadge status={r.status} /> : r.passed ? <Badge tone="teal">pass</Badge> : <Badge tone="coral">fail</Badge>}</td>
                  <td className="num">{r.score.toFixed(2)}</td>
                  <td className="num">{ms(r.ttft_ms)}</td>
                  <td className="num">{ms(r.total_ms)}</td>
                  <td className="num">{num(r.tokens_per_s)}</td>
                </tr>
                {open === r.id && (
                  <tr>
                    <td />
                    <td colSpan={8} className="bg-bg/60">
                      <div className="grid gap-4 md:grid-cols-2 py-1">
                        <div>
                          <div className="text-[12px] text-muted mb-1">Model response{r.thinking_chars ? ` (reasoning removed: ${r.thinking_chars} chars)` : ""}</div>
                          <pre className="whitespace-pre-wrap rounded-md border border-line bg-bg px-3 py-2 text-[12.5px] max-h-64 overflow-auto">{r.response || r.error || "(empty)"}</pre>
                        </div>
                        <div>
                          <div className="text-[12px] text-muted mb-1">Evaluation{r.eval_details ? `: ${r.eval_details.method}` : ""}</div>
                          <pre className="whitespace-pre-wrap rounded-md border border-line bg-bg px-3 py-2 text-[12px] max-h-64 overflow-auto">{r.eval_details ? JSON.stringify(r.eval_details.details, null, 2) : r.error}</pre>
                          <div className="mt-2 text-[12px] text-faint num">
                            prompt {r.prompt_tokens ?? "–"} tokens, output {r.output_tokens ?? "–"} tokens ({r.tokens_source}), load {ms(r.load_ms)}, peak RAM {num(r.peak_ram_used_mb, 0)} MB, CPU {num(r.avg_cpu_pct)}%
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </Table>
        {results.length > 400 && <div className="p-3 text-[12px] text-faint">Showing the first 400 of {results.length} results. Export CSV for all.</div>}
      </Panel>

      <Panel className="mt-6" title="Where the numbers come from">
        <dl className="grid gap-x-6 gap-y-2 md:grid-cols-[220px_1fr] text-[13px]">
          {PROVENANCE.map(([k, v]) => <Fragment key={k}><dt className="text-ink">{k}</dt><dd className="text-muted">{v}</dd></Fragment>)}
        </dl>
        <p className="mt-3 text-[12.5px] text-faint">Scores show success on this suite's defined checks only. They are not a general ranking of the models.</p>
      </Panel>
    </div>
  );
}

function BenchVerdict({ c, margin }: { c: NonNullable<BenchmarkRun["comparison"]>["comparisons"][string]; margin: number }) {
  const tone = c.verdict === "equivalent" ? "teal" : c.verdict === "worse" ? "coral" : c.verdict === "inconclusive" ? "amber" : "neutral";
  const title = c.delta != null
    ? `Leader − this model = ${(c.delta * 100).toFixed(1)} pt (95% CI ${c.ci95 ? `${(c.ci95[0] * 100).toFixed(1)} to ${(c.ci95[1] * 100).toFixed(1)}` : "–"}) over ${c.n_cases} shared cases; margin ±${(margin * 100).toFixed(0)} pt`
    : `Only ${c.n_cases} shared cases – not compared`;
  return <Badge tone={tone} title={title}>{c.verdict === "not_tested" ? "not tested" : c.verdict}</Badge>;
}
