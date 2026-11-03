import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ListChecks, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { BundleImport } from "../components/Bundles";
import type { BenchmarkRun, SuiteSummary } from "../api/types";
import { Badge, Button, cx, Empty, ErrorNotice, Field, Input, PageHeader, Panel, Readout, Select, StatusBadge, Table, Toggle } from "../components/ui";
import { duration, humanize, num, pct, SERIES, when } from "../lib/format";
import { useModels } from "../lib/queries";

interface Overview {
  total_runs: number;
  completed_runs: number;
  best: { model: string; quality_score: number; run_id: number } | null;
  fastest: { model: string; tokens_per_s: number; run_id: number } | null;
  total_bench_time_s: number;
  trend: { run_id: number; suite_id: string; model: string; finished_at: string; quality_score: number | null; tokens_per_s: number | null }[];
}

export const chartTheme = {
  grid: "#2b3644",
  axis: { fill: "#8c99aa", fontSize: 12 },
  tooltip: { contentStyle: { background: "#19212c", border: "1px solid #3a4757", borderRadius: 6, fontSize: 12 }, labelStyle: { color: "#e4e8ee" } },
};

function TrendChart({ trend }: { trend: Overview["trend"] }) {
  // Scores are only comparable within one suite, so the trend is shown per suite.
  const suites = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of trend) counts.set(p.suite_id, (counts.get(p.suite_id) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
  }, [trend]);
  const [suite, setSuite] = useState("");
  const active = suite || suites[0] || "";
  const { rows, models } = useMemo(() => {
    const byRun = new Map<number, Record<string, number | string>>();
    const models: string[] = [];
    for (const p of trend.filter((t) => t.suite_id === active)) {
      if (!models.includes(p.model)) models.push(p.model);
      const row = byRun.get(p.run_id) ?? { run: `Run ${p.run_id}` };
      if (p.quality_score != null) row[p.model] = Math.round(p.quality_score * 100);
      byRun.set(p.run_id, row);
    }
    return { rows: [...byRun.values()], models };
  }, [trend, active]);
  if (!trend.length) return <Empty title="No completed runs yet">The quality trend appears here after the first benchmark finishes.</Empty>;
  return (
    <div>
      <div className="mb-3 flex items-center gap-2 text-[13px] text-muted">
        <label htmlFor="trend-suite">Suite</label>
        <Select id="trend-suite" className="w-56" value={active} onChange={(e) => setSuite(e.target.value)}>
          {suites.map((s) => <option key={s} value={s}>{s}</option>)}
        </Select>
        <span className="text-faint">Each point is one completed run of this suite.</span>
      </div>
      <div className="h-[240px]">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
            <XAxis dataKey="run" tick={chartTheme.axis} />
            <YAxis domain={[0, 100]} tick={chartTheme.axis} unit="%" />
            <Tooltip {...chartTheme.tooltip} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {models.map((m, i) => (
              <Line key={m} dataKey={m} stroke={SERIES[i % SERIES.length]} strokeWidth={2} dot={{ r: 3 }} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function RunForm() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const models = useModels();
  const suites = useQuery({ queryKey: ["suites"], queryFn: () => api<{ suites: SuiteSummary[] }>("/api/benchmarks/suites") });
  const [suiteId, setSuiteId] = useState("quick");
  const [picked, setPicked] = useState<string[]>([]);
  const [cats, setCats] = useState<string[]>([]);
  const [repeats, setRepeats] = useState(1);
  const [warmup, setWarmup] = useState(true);
  const [temperature, setTemperature] = useState(0);
  const [seed, setSeed] = useState("42");
  const [numCtx, setNumCtx] = useState(2048);
  const [timeout, setTimeoutS] = useState("");
  const [name, setName] = useState("");
  const [shuffle, setShuffle] = useState(false);

  const chatModels = (models.data?.models ?? []).filter((m) => !m.is_embedding);
  const suite = suites.data?.suites.find((s) => s.id === suiteId);
  useEffect(() => setCats([]), [suiteId]);
  const caseCount = suite ? (cats.length ? cats.reduce((a, c) => a + (suite.categories[c] ?? 0), 0) : suite.case_count) : 0;
  const total = caseCount * picked.length * repeats;

  const start = useMutation({
    mutationFn: () =>
      api<BenchmarkRun>("/api/benchmarks/runs", {
        method: "POST",
        json: {
          suite_id: suiteId, models: picked, categories: cats.length ? cats : undefined, repeats, warmup, temperature,
          seed: seed === "" ? null : Number(seed), num_ctx: numCtx, timeout_s: timeout ? Number(timeout) : undefined, name: name || undefined,
          model_order: shuffle ? "shuffled" : "as_given", case_order: shuffle ? "shuffled" : "suite",
        },
      }),
    onSuccess: (run) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      nav(`/benchmarks/runs/${run.id}`);
    },
  });

  return (
    <Panel title="New benchmark run">
      <div className="space-y-5">
        <fieldset>
          <legend className="text-[12.5px] text-muted mb-2">Test suite</legend>
          <div className="space-y-1.5">
            {suites.data?.suites.map((s) => (
              <label key={s.id} className={cx("block cursor-pointer rounded-md border px-3 py-2", s.id === suiteId ? "border-amber/60 bg-amber/5" : "border-line hover:border-line-strong")}>
                <div className="flex items-center gap-2">
                  <input type="radio" name="suite" value={s.id} checked={s.id === suiteId} onChange={() => setSuiteId(s.id)} className="accent-amber" />
                  <span className="text-[13.5px]">{s.name}</span>
                  <span className="ml-auto num text-[12px] text-faint">{s.case_count} tests, v{s.version}</span>
                </div>
                {s.id === suiteId && <p className="mt-1 pl-6 text-[12.5px] text-muted">{s.description}</p>}
              </label>
            ))}
          </div>
        </fieldset>

        {suite && Object.keys(suite.categories).length > 1 && (
          <fieldset>
            <legend className="text-[12.5px] text-muted mb-2">Categories (none selected = all)</legend>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(suite.categories).map(([c, n]) => {
                const on = cats.includes(c);
                return (
                  <button key={c} type="button" aria-pressed={on} onClick={() => setCats((x) => (on ? x.filter((y) => y !== c) : [...x, c]))}
                    className={cx("rounded-full border px-2.5 py-0.5 text-[12.5px]", on ? "border-amber/60 bg-amber/10 text-ink" : "border-line text-muted hover:text-ink")}>
                    {humanize(c)} <span className="num text-faint">{n}</span>
                  </button>
                );
              })}
            </div>
          </fieldset>
        )}

        <fieldset>
          <legend className="text-[12.5px] text-muted mb-2">Models to compare</legend>
          {chatModels.length === 0 ? <p className="text-[13px] text-faint">No chat models found. Download one on the <Link to="/setup" className="text-amber hover:underline">Setup</Link> page.</p> : (
            <div className="space-y-1">
              {chatModels.map((m) => (
                <label key={m.id} className={cx("flex items-center gap-2 text-[13.5px]", !m.allowed && "opacity-50")} title={m.blocked_reason ?? undefined}>
                  <input type="checkbox" className="accent-amber" disabled={!m.allowed} checked={picked.includes(m.id)}
                    onChange={(e) => setPicked((p) => (e.target.checked ? [...p, m.id] : p.filter((x) => x !== m.id)))} />
                  <span className="truncate">{m.name}</span>
                  {m.parameter_size && <span className="text-faint text-[12px]">{m.parameter_size}</span>}
                  {m.provider === "cloud" && <Badge tone="amber">cloud</Badge>}
                </label>
              ))}
            </div>
          )}
        </fieldset>

        <div className="grid grid-cols-2 gap-3">
          <Field label="Repeats per test"><Input type="number" min={1} max={10} value={repeats} onChange={(e) => setRepeats(Math.max(1, Number(e.target.value)))} /></Field>
          <Field label="Temperature"><Input type="number" step="0.1" min={0} max={2} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} /></Field>
          <Field label="Seed"><Input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="none" /></Field>
          <Field label="Context tokens"><Input type="number" min={256} step={256} value={numCtx} onChange={(e) => setNumCtx(Number(e.target.value))} /></Field>
          <Field label="Timeout per test (s)"><Input type="number" min={5} value={timeout} onChange={(e) => setTimeoutS(e.target.value)} placeholder="from Settings" /></Field>
          <Field label="Run name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="automatic" /></Field>
        </div>
        <Toggle checked={warmup} onChange={setWarmup} label="Warm up each model first" description="Loads the model before timing, so cold-start load time does not distort latency." />
        <Toggle checked={shuffle} onChange={setShuffle} label="Shuffle model and case order" description="Spreads thermal throttling and background load evenly over the models (order is recorded with the run)." />
        {start.isError && <ErrorNotice error={start.error} />}
        <div className="flex items-center gap-3">
          <Button variant="primary" onClick={() => start.mutate()} busy={start.isPending} disabled={!picked.length || !caseCount}><Play className="size-4" />Start benchmark</Button>
          <span className="text-[12.5px] text-muted num">{total ? `${total} generations` : "Pick at least one model"}</span>
        </div>
      </div>
    </Panel>
  );
}

export default function Benchmarks() {
  const qc = useQueryClient();
  const overview = useQuery({ queryKey: ["bench-overview"], queryFn: () => api<Overview>("/api/benchmarks/overview"), refetchInterval: 10000 });
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<{ runs: BenchmarkRun[] }>("/api/benchmarks/runs"),
    refetchInterval: (q) => (q.state.data?.runs.some((r) => r.status === "running" || r.status === "queued") ? 1500 : 10000),
  });
  const o = overview.data;

  const cancel = async (id: number) => {
    await api(`/api/benchmarks/runs/${id}/cancel`, { method: "POST" });
    qc.invalidateQueries({ queryKey: ["runs"] });
  };

  return (
    <div className="max-w-[1320px]">
      <PageHeader title="Benchmarks" description="Run the same versioned test suite on several models and compare task success, latency, throughput and memory."
        actions={<Link to="/benchmarks/suites"><Button><ListChecks className="size-4" />Test suites</Button></Link>} />
      <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
        <RunForm />
        <div className="space-y-6 min-w-0">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-4 rounded-lg border border-line bg-panel px-5 py-4">
            <Readout label="Total runs" value={o?.total_runs ?? "–"} sub={`${o?.completed_runs ?? 0} completed`} />
            <Readout label="Best quality" value={o?.best ? pct(o.best.quality_score) : "–"} sub={o?.best?.model} tone="teal" />
            <Readout label="Fastest" value={o?.fastest ? `${num(o.fastest.tokens_per_s)} t/s` : "–"} sub={o?.fastest?.model} tone="amber" />
            <Readout label="Total bench time" value={duration(o?.total_bench_time_s)} sub="across completed runs" />
          </div>
          <Panel title="Quality trend">{o ? <TrendChart trend={o.trend} /> : null}</Panel>
          <Panel title="Runs" bodyClass="p-0">
            {runs.isError ? <div className="p-4"><ErrorNotice error={runs.error} /></div> : !runs.data?.runs.length ? (
              <div className="p-4"><Empty title="No runs yet">Choose a suite and at least one model, then start a benchmark.</Empty></div>
            ) : (
              <Table>
                <thead><tr><th>Run</th><th>Status</th><th>Best quality</th><th>Duration</th><th>Started</th><th /></tr></thead>
                <tbody>
                  {runs.data.runs.map((r) => (
                    <tr key={r.id}>
                      <td className="max-w-[280px]">
                        <Link to={`/benchmarks/runs/${r.id}`} className="block truncate hover:text-amber" title={r.name}>{r.name}</Link>
                        <div className="text-[12px] text-faint">{r.suite_id} v{r.suite_version}</div>
                      </td>
                      <td>
                        <StatusBadge status={r.status} />
                        {(r.status === "running" || r.status === "queued") && (
                          <div className="mt-1 w-28">
                            <div className="h-1 rounded-full bg-raised overflow-hidden"><div className="h-full bg-amber" style={{ width: `${(r.progress_done / Math.max(1, r.progress_total)) * 100}%` }} /></div>
                            <div className="num text-[11.5px] text-faint mt-0.5">{r.progress_done}/{r.progress_total}</div>
                          </div>
                        )}
                      </td>
                      <td className="whitespace-nowrap">{r.summary[0] ? <span className="num">{pct(r.summary[0].quality_score)} <span className="text-faint">{r.summary[0].model}</span></span> : "–"}</td>
                      <td className="num">{duration(r.duration_s)}</td>
                      <td className="text-muted whitespace-nowrap">{when(r.started_at ?? r.created_at)}</td>
                      <td className="text-right">{(r.status === "running" || r.status === "queued") && <Button variant="ghost" onClick={() => cancel(r.id)}>Cancel</Button>}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Panel>
          <Panel title="Experiment bundles">
            <p className="mb-3 text-[12.5px] text-muted">Open a bundle exported from another computer (Benchmarks, Model arena or Agent arena): verify its checksums, import the results, or re-run the same cases here.</p>
            <BundleImport />
          </Panel>
        </div>
      </div>
    </div>
  );
}
