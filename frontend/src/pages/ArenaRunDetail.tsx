import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Award, Download, FlaskConical, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api, download } from "../api/client";
import { BundleExport } from "../components/Bundles";
import type { ArenaResult, ArenaRun, ArenaSummary, PairedComparison, Verdict } from "../api/arena";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { Badge, Button, cx, ErrorNotice, Notice, PageHeader, Panel, Readout, Select, Spinner, StatusBadge, Table } from "../components/ui";
import { bytes, duration, ms, num, when } from "../lib/format";
import { chartTheme } from "./Benchmarks";

const SOURCE_TONE = { baseline: "neutral", builtin: "sky", uploaded: "amber", llm: "violet" } as const;
const SOURCE_LABEL = { baseline: "baseline", builtin: "built-in", uploaded: "your model", llm: "LLM" } as const;
const SOURCE_COLOR = { baseline: "#8c99aa", builtin: "#6fb2f2", uploaded: "#f2b84b", llm: "#a28beb" } as const;

const p1 = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const f3 = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(3));
const fp = (p: number) => (p < 0.001 ? "<0.001" : p.toFixed(3));

const VERDICT_TONE: Record<Verdict, "teal" | "coral" | "amber" | "sky" | "neutral"> = {
  equivalent: "teal", worse: "coral", inconclusive: "amber", better: "sky", not_tested: "neutral",
};
const VERDICT_LABEL: Record<Verdict, string> = {
  equivalent: "equivalent", worse: "worse", inconclusive: "inconclusive", better: "better?", not_tested: "not tested",
};
const sgn = (v: number | null | undefined, pct: boolean) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${pct ? (v * 100).toFixed(1) + " pt" : v.toFixed(3)}`);

export function VerdictBadge({ comp, text }: { comp?: PairedComparison; text?: Record<Verdict, string> }) {
  if (!comp) return <span className="text-faint">–</span>;
  return <Badge tone={VERDICT_TONE[comp.verdict]} title={text?.[comp.verdict] ?? comp.note}>{VERDICT_LABEL[comp.verdict]}</Badge>;
}

function latency(r: ArenaResult) {
  return r.metrics?.latency_ms_single_p50 ?? r.metrics?.latency_ms_per_example ?? null;
}

export default function ArenaRunDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["arena-run", id],
    queryFn: () => api<ArenaRun>(`/api/arena/runs/${id}`),
    refetchInterval: (query) => (["running", "queued"].includes(query.state.data?.status ?? "") ? 1500 : false),
  });
  const [selected, setSelected] = useState<string | null>(null);
  const run = q.data;
  const summary = run?.summary;
  const cls = run?.dataset.task === "classification";
  const v2 = summary?.protocol === "v2";
  const primary = summary?.primary_metric ?? (cls ? "f1_macro" : "r2");

  const ordered = useMemo(() => {
    const res = run?.results ?? [];
    const rank = new Map((summary?.ranking ?? []).map((k, i) => [k, i]));
    return [...res].sort((a, b) => (rank.get(a.key) ?? 1e9) - (rank.get(b.key) ?? 1e9) || a.id - b.id);
  }, [run?.results, summary?.ranking]);

  if (q.isLoading) return <Spinner />;
  if (q.isError || !run) return <ErrorNotice error={q.error} />;

  const live = run.status === "running" || run.status === "queued";
  const rec = summary?.recommendation;
  const ok = ordered.filter((r) => r.status === "ok" && r.metrics);
  const current = ordered.find((r) => r.key === (selected ?? rec?.key ?? ok[0]?.key)) ?? ok[0];
  const pareto = new Set(summary?.pareto ?? []);
  const best = ok[0];
  const real = ok.filter((r) => r.source !== "baseline" && latency(r) != null);
  const fastest = real.length ? real.reduce((a, b) => (latency(b)! < latency(a)! ? b : a)) : null;

  const metricOf = (r: ArenaResult) => (r.metrics?.[primary as "f1_macro" | "r2"] as number | undefined) ?? null;
  const ciOf = (r: ArenaResult) => (primary === "f1_macro" ? r.metrics?.f1_macro_ci : r.metrics?.r2_ci) ?? null;
  const barData = ok.map((r) => {
    const v = metricOf(r) ?? 0;
    const ci = ciOf(r);
    return { name: r.name, key: r.key, value: v, err: ci ? [Math.max(0, v - ci[0]), Math.max(0, ci[1] - v)] : [0, 0], source: r.source };
  });
  const scatter = ok.filter((r) => latency(r) != null).map((r) => ({ name: r.name, x: Math.max(0.001, latency(r)!), y: metricOf(r) ?? 0, z: pareto.has(r.key) ? 120 : 50, source: r.source }));

  const remove = async () => {
    if (!confirm("Delete this comparison and its results?")) return;
    await api(`/api/arena/runs/${run.id}`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: ["arena-runs"] });
    nav("/arena?tab=runs");
  };

  const env = (run.environment ?? {}) as Record<string, unknown>;
  const libs = (env.libraries ?? {}) as Record<string, string>;

  return (
    <div className="max-w-[1320px]">
      <PageHeader
        title={run.name}
        description={<>Dataset <span className="text-ink">{run.dataset.name}</span> ({run.dataset.kind} {run.dataset.task}) – train {summary?.split?.n_train ?? run.dataset.n_train}{v2 && summary?.split ? ` / validation ${summary.split.n_val}` : ""} / test {summary?.split?.n_test ?? run.dataset.n_test}, seed {run.dataset.seed}, split {run.dataset.split_source}. Started {when(run.started_at ?? run.created_at)}{run.duration_s != null && `, took ${duration(run.duration_s)}`}.</>}
        actions={live ? <Button onClick={() => api(`/api/arena/runs/${run.id}/cancel`, { method: "POST" }).then(() => q.refetch())}>Cancel</Button> : <>
          <Button onClick={() => download(`/api/arena/runs/${run.id}/export?format=md`)}><Download className="size-4" />Report (.md)</Button>
          <Button onClick={() => download(`/api/arena/runs/${run.id}/export?format=csv`)}><Download className="size-4" />CSV</Button>
          <Button onClick={() => download(`/api/arena/runs/${run.id}/export?format=json`)}><Download className="size-4" />JSON</Button>
          <BundleExport kind="arena" runId={run.id} />
          {run.has_model_file && <Button onClick={() => download(`/api/arena/runs/${run.id}/model`)} title="scikit-learn bundle of the recommended (or best) built-in model"><Download className="size-4" />Trained model</Button>}
          <Button variant="danger" onClick={remove}><Trash2 className="size-4" />Delete</Button>
        </>}
      />

      <div className="mb-6 rounded-lg border border-line bg-panel px-5 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={run.status} />
          <span className="num text-[13px] text-muted">{run.progress_done} of {run.progress_total} steps</span>
          {run.current_item && <span className="text-[13px] text-muted">Now: <span className="text-ink">{run.current_item}</span></span>}
          {run.mlflow_run_id && <Badge tone="violet">MLflow {run.mlflow_run_id.slice(0, 8)}</Badge>}
          {run.dataset_id && <Link to="/arena?tab=datasets" className="ml-auto text-[12.5px] text-muted hover:text-amber">Dataset details</Link>}
        </div>
        <div className="mt-3 h-1.5 rounded-full bg-raised overflow-hidden"><div className={cx("h-full", live ? "bg-amber" : "bg-teal")} style={{ width: `${(run.progress_done / Math.max(1, run.progress_total)) * 100}%` }} /></div>
        {run.error && <div className="mt-3"><ErrorNotice error={{ message: run.error }} /></div>}
      </div>

      {rec && (
        <div className="mb-6 flex gap-3 rounded-lg border border-teal/40 bg-teal/8 px-5 py-4">
          <Award className="mt-0.5 size-5 shrink-0 text-teal" aria-hidden />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-cond text-[17px] font-semibold">Recommended for this computer: {rec.name}</span>
              {rec.status && <Badge tone={rec.status === "clear" ? "teal" : "amber"} title={rec.status === "clear" ? "No other model could be as good on the validation data" : "Some models could not be separated from the leader – see the shortlist"}>{rec.status}</Badge>}
            </div>
            <div className="text-[13px] text-muted">{rec.reason}</div>
            {rec.caveats?.map((c) => <div key={c} className="mt-1 text-[12.5px] text-amber">• {c}</div>)}
          </div>
        </div>
      )}
      {summary?.warnings?.map((w) => <div key={w} className="mb-3"><Notice tone="warn">{w}</Notice></div>)}
      {summary && !v2 && run.status === "completed" && (
        <div className="mb-4"><Notice tone="warn" title="Older comparison (protocol v1)">
          This run chose the model on the same test rows it reports, and “≈ tie” only meant “no significant difference”, not equivalence. Its scores are optimistic. Run the comparison again to get validation-based selection, equivalence tests and an unbiased final estimate.
        </Notice></div>
      )}
      {v2 && summary && <ProtocolV2 summary={summary} results={ordered} cls={!!cls} />}

      {best && (
        <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-4 rounded-lg border border-line bg-panel px-5 py-4">
          {v2 && summary?.final ? (
            <Readout label={`Final test ${summary.final.label}`} value={cls ? p1(summary.final.value) : f3(summary.final.value)} sub={`${summary.final.name}${summary.final.ci ? ` · 95% CI ${cls ? `${p1(summary.final.ci[0])}–${p1(summary.final.ci[1])}` : `${f3(summary.final.ci[0])}–${f3(summary.final.ci[1])}`}` : ""}`} tone="teal" />
          ) : (
            <Readout label={`Best ${summary?.primary_label ?? ""}`} value={cls ? p1(metricOf(best)) : f3(metricOf(best))} sub={best.name} tone="teal" />
          )}
          <Readout label="Fastest prediction" value={fastest ? ms(latency(fastest)) : "–"} sub={fastest?.name} tone="amber" />
          <Readout label="Models compared" value={ok.length} sub={`${ordered.length - ok.length} failed`} />
          <Readout label={v2 ? "Train / validation / test" : "Test examples"} value={v2 && summary?.split ? `${summary.split.n_train} / ${summary.split.n_val} / ${summary.split.n_test}` : summary?.test_size ?? run.dataset.n_test} sub={v2 ? "same rows for every model" : "same for every model"} />
        </div>
      )}

      <Panel title="Leaderboard" bodyClass="p-0" className="mb-6">
        <Table>
          <thead>
            <tr>
              <th>#</th><th>Model</th>
              {v2 && <th title="Used to choose the model">{summary?.primary_label ?? "Score"} validation</th>}
              <th title={v2 ? "Reported for information; the choice was made on validation" : undefined}>{summary?.primary_label ?? "Score"} {v2 ? "test " : ""}(95% CI)</th>
              {cls ? <><th>Accuracy</th><th>Balanced acc.</th></> : <><th>MAE</th><th>RMSE</th></>}
              <th title="Cross-validation on the training split">CV (train)</th>
              <th title="Median time for one prediction (image built-ins include feature extraction)">Latency</th>
              <th>Train time</th><th>Size</th><th title={v2 ? "Paired bootstrap on the validation split, equivalence margin and Holm correction" : undefined}>{v2 ? "vs leader (validation)" : "vs best"}</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((r, i) => {
              const m = r.metrics;
              const t = summary?.tests?.[r.key];
              const comp = summary?.selection?.comparisons[r.key];
              const excluded = summary?.selection?.excluded[r.key];
              const valMetric = (r.val_metrics?.[primary as "f1_macro" | "r2"] as number | undefined) ?? null;
              const ci = ciOf(r);
              const cv = r.details?.cv as { mean?: number; std?: number; error?: string } | undefined;
              const isRec = rec?.key === r.key;
              return (
                <tr key={r.key} onClick={() => r.status === "ok" && setSelected(r.key)} className={cx("cursor-pointer hover:bg-raised/40", current?.key === r.key && "bg-raised/60", isRec && "shadow-[inset_3px_0_0_var(--color-teal)]")}>
                  <td className="num text-muted">{r.status === "ok" ? i + 1 : "–"}</td>
                  <td className="max-w-[280px]">
                    <div className="truncate" title={r.name}>{r.name}</div>
                    <div className="mt-0.5 flex flex-wrap gap-1">
                      <Badge tone={SOURCE_TONE[r.source]}>{SOURCE_LABEL[r.source]}</Badge>
                      {isRec && <Badge tone="teal">recommended</Badge>}
                      {pareto.has(r.key) && r.source !== "baseline" && <Badge tone="neutral" title="No other model is both more accurate and faster">efficient</Badge>}
                      {(r.details?.caveats as string[] | undefined)?.length ? <Badge tone="amber" title={(r.details!.caveats as string[]).join(" ")}>caveat</Badge> : null}
                    </div>
                  </td>
                  {r.status !== "ok" || !m ? (
                    <td colSpan={v2 ? 9 : 8} className="text-coral text-[12.5px]">{r.error ?? r.status}</td>
                  ) : (
                    <>
                      {v2 && <td className="num whitespace-nowrap">{valMetric == null ? <span className="text-faint" title="Not evaluated on validation (e.g. trained on train+validation)">–</span> : cls ? p1(valMetric) : f3(valMetric)}</td>}
                      <td className="num whitespace-nowrap">{cls ? p1(metricOf(r)) : f3(metricOf(r))}{ci && <span className="block text-[11.5px] text-faint">{cls ? `${p1(ci[0])}–${p1(ci[1])}` : `${f3(ci[0])}–${f3(ci[1])}`}</span>}</td>
                      {cls ? <><td className="num">{p1(m.accuracy)}</td><td className="num">{p1(m.balanced_accuracy)}</td></> : <><td className="num">{num(m.mae, 3)}</td><td className="num">{num(m.rmse, 3)}</td></>}
                      <td className="num whitespace-nowrap">{cv?.mean != null ? <>{cls ? p1(cv.mean) : f3(cv.mean)}<span className="text-faint"> ±{cls ? (cv.std! * 100).toFixed(1) : cv.std!.toFixed(3)}</span></> : "–"}</td>
                      <td className="num whitespace-nowrap">{ms(latency(r))}{m.latency_ms_single_p95 != null && <span className="block text-[11.5px] text-faint">p95 {ms(m.latency_ms_single_p95)}</span>}</td>
                      <td className="num whitespace-nowrap">{m.fit_ms != null ? ms(m.fit_ms) : "–"}</td>
                      <td className="num whitespace-nowrap">{m.model_size_bytes ? bytes(m.model_size_bytes) : "–"}</td>
                      <td className="whitespace-nowrap">
                        {v2 ? (summary?.selection?.leader === r.key ? <Badge tone="teal">leader</Badge>
                          : excluded ? <Badge tone="neutral" title={excluded.join("; ")}>excluded</Badge>
                          : comp ? <span className="inline-flex items-center gap-1.5"><VerdictBadge comp={comp} text={summary?.verdict_text} />{comp.delta != null && <span className="num text-[11.5px] text-faint">Δ {sgn(-comp.delta, !!cls)}</span>}</span>
                          : <span className="text-faint">–</span>)
                        : best?.key === r.key ? <Badge tone="teal">best</Badge> : t?.p_value == null ? <span className="text-faint">–</span> : t.significantly_worse
                          ? <Badge tone="coral" title={t.test ?? ""}>worse · p{fp(t.p_value).startsWith("<") ? "" : "="}{fp(t.p_value)}</Badge>
                          : <Badge tone="neutral" title={t.test ?? ""}>≈ tie · p={fp(t.p_value)}</Badge>}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </Table>
        <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">Click a row for details (throughput, memory, CV folds, errors). Latency = median time for one prediction.{v2 ? " Rows are ordered by the validation score that was used for the choice; Δ is the model minus the leader." : ""} {summary?.note}</div>
      </Panel>

      {ok.length > 0 && (
        <div className="mb-6 grid gap-6 lg:grid-cols-2">
          <Panel title={`${v2 ? "Test " : ""}${summary?.primary_label ?? "Score"} with 95% confidence interval`}>
            <div style={{ height: Math.max(180, barData.length * 30 + 40) }}>
              <ResponsiveContainer>
                <BarChart data={barData} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={cls ? [0, 1] : ["auto", 1]} tick={chartTheme.axis} tickFormatter={(v) => (cls ? `${Math.round(v * 100)}%` : Number(v).toFixed(2))} />
                  <YAxis type="category" dataKey="name" width={190} tick={{ ...chartTheme.axis, fontSize: 11 }} tickFormatter={(v: string) => (v.length > 30 ? v.slice(0, 29) + "…" : v)} />
                  <Tooltip {...chartTheme.tooltip} formatter={(v) => (cls ? p1(Number(v)) : f3(Number(v)))} />
                  <Bar dataKey="value" name={summary?.primary_label} radius={[0, 3, 3, 0]} isAnimationActive={false}>
                    {barData.map((d) => <Cell key={d.key} fill={SOURCE_COLOR[d.source]} fillOpacity={d.key === rec?.key ? 1 : 0.7} />)}
                    <ErrorBar dataKey="err" width={4} strokeWidth={1.5} stroke="#e4e8ee" direction="x" isAnimationActive={false} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Quality vs. speed">
            <div className="h-[300px]">
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 24, left: 0 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name="Latency" scale="log" domain={["auto", "auto"]} tick={chartTheme.axis} tickFormatter={(v) => ms(Number(v))} label={{ value: "latency per prediction (log scale) →", position: "insideBottom", offset: -14, fill: "#8c99aa", fontSize: 12 }} />
                  <YAxis type="number" dataKey="y" name={summary?.primary_label} tick={chartTheme.axis} tickFormatter={(v) => (cls ? `${Math.round(v * 100)}%` : Number(v).toFixed(2))} />
                  <ZAxis dataKey="z" range={[50, 160]} />
                  <Tooltip {...chartTheme.tooltip} formatter={(v, n) => (n === "Latency" ? ms(Number(v)) : cls ? p1(Number(v)) : f3(Number(v)))} labelFormatter={() => ""} />
                  <Scatter data={scatter} name="models" isAnimationActive={false}>
                    {scatter.map((d) => <Cell key={d.name} fill={SOURCE_COLOR[d.source]} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[12px] text-faint">Top-left is best (accurate and fast). Larger dots are “efficient” – no other model beats them on both.</p>
          </Panel>
        </div>
      )}

      {current && current.metrics && (
        <Panel title={<>Details: {current.name}</>} action={
          <Select className="w-64" value={current.key} onChange={(e) => setSelected(e.target.value)} aria-label="Choose model">
            {ok.map((r) => <option key={r.key} value={r.key}>{r.name}</option>)}
          </Select>
        } className="mb-6">
          <ResultDetail r={current} cls={!!cls} />
        </Panel>
      )}

      <details className="rounded-lg border border-line bg-panel px-5 py-3 text-[13px]">
        <summary className="cursor-pointer text-muted">Environment and settings</summary>
        <div className="mt-2 grid gap-1 text-muted md:grid-cols-2">
          <div>OS: <span className="text-ink">{String(env.os ?? "–")}</span></div>
          <div>CPUs: <span className="text-ink">{String(env.cpu_count_logical ?? "–")}</span>, RAM <span className="text-ink">{String(env.ram_total_mb ?? "–")} MB</span></div>
          <div>Libraries: <span className="text-ink">{Object.entries(libs).map(([k, v]) => `${k} ${v}`).join(", ") || "–"}</span></div>
          <div>Cross-validation: <span className="text-ink">{run.config.cv_folds || "off"}</span>, tuning <span className="text-ink">{run.config.tune ? "on" : "off"}</span></div>
          <div>Dataset SHA-256: <span className="num text-ink">{run.dataset.sha256.slice(0, 16)}…</span></div>
          <div>Split SHA-256: <span className="num text-ink">{run.dataset.split_sha256 ? `${run.dataset.split_sha256.slice(0, 16)}…` : "–"}</span> · protocol <span className="text-ink">{run.protocol ?? "v1"}</span></div>
          {v2 && summary?.selection && <div>Statistics: <span className="text-ink">{summary.selection.resampling.method} bootstrap × {summary.selection.resampling.n_boot}, α = {summary.selection.alpha}, correction {summary.selection.correction ?? "–"}</span></div>}
          <div>App version: <span className="text-ink">{String(env.app_version ?? "–")}</span></div>
        </div>
      </details>
    </div>
  );
}

function ResultDetail({ r, cls }: { r: ArenaResult; cls: boolean }) {
  const m = r.metrics!;
  const d = r.details ?? {};
  const errors = (d.errors_sample ?? []) as Record<string, unknown>[];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {cls ? <>
          <Readout label="Macro F1" value={p1(m.f1_macro)} tone="amber" />
          <Readout label="Accuracy" value={p1(m.accuracy)} tone="teal" />
          <Readout label="Macro precision" value={p1(m.precision_macro)} />
          <Readout label="Macro recall" value={p1(m.recall_macro)} />
          <Readout label="ROC AUC" value={m.roc_auc != null ? f3(m.roc_auc) : "–"} sub={m.unparseable ? `${m.unparseable} unparseable answers` : undefined} />
        </> : <>
          <Readout label="R²" value={f3(m.r2)} tone="amber" />
          <Readout label="MAE" value={num(m.mae, 3)} tone="teal" />
          <Readout label="RMSE" value={num(m.rmse, 3)} />
          <Readout label="Median abs. error" value={num(m.median_ae, 3)} />
          <Readout label="MAPE" value={m.mape != null ? p1(m.mape) : "–"} />
        </>}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 border-t border-line pt-4">
        <Readout label="Latency (1 prediction)" value={ms(m.latency_ms_single_p50 ?? m.latency_ms_per_example)} sub={m.latency_ms_single_p95 != null ? `p95 ${ms(m.latency_ms_single_p95)}` : undefined} />
        <Readout label="Throughput (batch)" value={m.throughput_per_s != null ? `${num(m.throughput_per_s, 0)}/s` : "–"} />
        <Readout label="Training time" value={m.fit_ms != null ? ms(m.fit_ms) : "–"} sub={m.fit_ms == null ? "not trained here" : undefined} />
        <Readout label="Model size" value={m.model_size_bytes ? bytes(m.model_size_bytes) : "–"} />
        <Readout label="Memory increase" value={m.memory_increase_mb != null ? `${num(m.memory_increase_mb, 0)} MB` : "–"} sub={m.peak_process_rss_mb != null ? `peak process ${num(m.peak_process_rss_mb, 0)} MB` : undefined} />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-[12.5px] text-muted">
        {d.feature_extractor && <span>Features: <span className="text-ink">{d.feature_extractor.label}</span> ({d.feature_extractor.dim} dims, {ms(d.feature_extractor.ms_per_image)} per image)</span>}
        {d.cv?.scores && <span>CV folds: <span className="num text-ink">{(d.cv.scores as number[]).map((v) => (cls ? (v * 100).toFixed(1) : v.toFixed(3))).join(" · ")}</span></span>}
        {d.cv?.error && <span className="text-amber">CV: {d.cv.error}</span>}
        {d.tuning && <span>Tuned ({d.tuning.iterations} tries): <span className="text-ink">{Object.entries(d.tuning.best_params as Record<string, unknown>).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")}</span></span>}
        {m.parameters ? <span>Parameters: <span className="num text-ink">{(m.parameters / 1e6).toFixed(2)} M</span></span> : null}
        {d.evaluated != null && <span>Evaluated on <span className="num text-ink">{d.evaluated}</span> of {d.of} test examples</span>}
        {m.tokens_per_s_mean != null && <span>Generation speed: <span className="num text-ink">{num(m.tokens_per_s_mean)} tokens/s</span></span>}
        {d.wall_s != null && <span>Wall time: <span className="num text-ink">{d.wall_s}s</span></span>}
      </div>
      {[...(d.caveats ?? []), ...(d.notes ?? []), ...(d.warnings ?? [])].map((n: string) => <div key={n} className="text-[12.5px] text-amber">• {n}</div>)}
      {cls && m.confusion_matrix && (
        <div className="grid gap-6 lg:grid-cols-2">
          <ConfusionMatrix cm={m.confusion_matrix} />
          {m.per_class && (
            <Table>
              <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(m.per_class).map(([c, v]) => (
                <tr key={c}><td className="max-w-[180px] truncate" title={c}>{c}</td><td className="num">{num(v.precision, 2)}</td><td className="num">{num(v.recall, 2)}</td><td className={cx("num", v.f1 < 0.5 && "text-coral")}>{num(v.f1, 2)}</td><td className="num">{v.support}</td></tr>
              ))}</tbody>
            </Table>
          )}
        </div>
      )}
      {errors.length > 0 && (
        <div>
          <div className="mb-1 text-[12.5px] text-muted">{cls ? "Some misclassified test examples" : "Largest errors on the test set"}</div>
          <Table>
            <thead><tr><th>Row</th>{cls ? <><th>Example</th><th>Expected</th><th>Predicted</th></> : <><th>Expected</th><th>Predicted</th><th>Abs. error</th></>}</tr></thead>
            <tbody>{errors.map((e, i) => (
              <tr key={i}>
                <td className="num text-muted">{String(e.row)}</td>
                {cls ? <>
                  <td className="max-w-[420px] truncate" title={String(e.text ?? e.image ?? JSON.stringify(e.values))}>{String(e.text ?? e.image ?? Object.entries((e.values ?? {}) as Record<string, unknown>).map(([k, v]) => `${k}=${v}`).join(", "))}</td>
                  <td>{String(e.expected)}</td>
                  <td className="text-coral">{e.predicted == null ? "(no valid label)" : String(e.predicted)}</td>
                </> : <>
                  <td className="num">{num(e.expected as number, 3)}</td>
                  <td className="num">{num(e.predicted as number, 3)}</td>
                  <td className="num text-coral">{num(e.abs_error as number, 3)}</td>
                </>}
              </tr>
            ))}</tbody>
          </Table>
        </div>
      )}
      {d.inspection && (
        <details className="text-[12.5px] text-muted">
          <summary className="cursor-pointer">Model file inspection</summary>
          <pre className="mt-1 overflow-x-auto rounded bg-bg p-2 text-[12px]">{JSON.stringify({ inspection: d.inspection, preprocess: d.preprocess }, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}

export function ProtocolV2({ summary, results, cls }: { summary: ArenaSummary; results: ArenaResult[]; cls: boolean }) {
  const sel = summary.selection;
  const fin = summary.final;
  const byKey = new Map(results.map((r) => [r.key, r]));
  const fmt = (v: number | null | undefined) => (cls ? p1(v) : f3(v));
  const fmtD = (v: number | null | undefined) => sgn(v, cls);
  if (!sel) return null;
  const rows = sel.ranking.filter((k) => !k.startsWith("baseline:"));
  const baseline = sel.ranking.find((k) => k.startsWith("baseline:"));
  const selOnTest = summary.split?.selection_split === "test";
  return (
    <div className="mb-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <Panel title={<span className="inline-flex items-center gap-2"><FlaskConical className="size-4 text-teal" aria-hidden />Final estimate</span>}>
        {fin ? (
          <div className="space-y-3">
            <Readout label={`${fin.label} of ${fin.name} on the test split`} value={fmt(fin.value)} sub={fin.ci ? `95% CI ${fmt(fin.ci[0])} – ${fmt(fin.ci[1])} · n = ${fin.n ?? "–"}` : `n = ${fin.n ?? "–"}`} tone={fin.unbiased ? "teal" : "amber"} />
            <Badge tone={fin.unbiased ? "teal" : "amber"}>{fin.unbiased ? "unbiased – test rows not used for the choice" : "exploratory – chosen on the test rows"}</Badge>
            <p className="text-[12.5px] text-muted">{fin.note}</p>
          </div>
        ) : <p className="text-[13px] text-muted">No model could be recommended (see warnings).</p>}
        {summary.limitations && summary.limitations.length > 0 && (
          <details className="mt-4 text-[12.5px] text-muted">
            <summary className="cursor-pointer text-ink">What this result does not show</summary>
            <ul className="mt-1.5 list-disc space-y-1 pl-4">{summary.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
          </details>
        )}
      </Panel>
      <Panel title={`How the model was chosen (${selOnTest ? "test" : "validation"} split)`} bodyClass="p-0">
        <div className="px-4 pt-3 text-[12.5px] text-muted">
          Leader: <span className="text-ink">{byKey.get(sel.leader ?? "")?.name ?? "–"}</span> · equivalence margin <span className="num text-ink">±{sel.margin != null ? (cls ? `${(sel.margin * 100).toFixed(1)} pt` : sel.margin.toPrecision(3)) : "–"}</span> <span className="text-faint">({sel.margin_rule})</span> · objective <span className="text-ink">{sel.objective === "efficiency" ? "fastest of the equivalent models" : "best quality"}</span>
          {Object.keys(sel.constraints).length > 0 && <> · constraints <span className="text-ink">{Object.entries(sel.constraints).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(", ")}</span></>}
        </div>
        <Table>
          <thead><tr><th>Model</th><th>Validation</th><th>Δ vs leader (90% CI)</th><th>Verdict</th></tr></thead>
          <tbody>
            {rows.map((k) => {
              const r = byKey.get(k);
              const c = sel.comparisons[k];
              const vm = (r?.val_metrics ?? (selOnTest ? r?.metrics : null))?.[sel.primary_metric as "f1_macro"] as number | undefined;
              const isLeader = sel.leader === k;
              return (
                <tr key={k} className={cx(summary.recommendation?.key === k && "shadow-[inset_3px_0_0_var(--color-teal)]")}>
                  <td className="max-w-[240px]"><div className="truncate" title={r?.name ?? k}>{r?.name ?? k}</div>
                    {sel.shortlist.includes(k) && !isLeader && <Badge tone="amber" title="Could be as good as the leader – a larger validation split would tell">shortlist</Badge>}
                  </td>
                  <td className="num">{fmt(vm)}</td>
                  <td className="num whitespace-nowrap">{isLeader ? "–" : c?.delta == null ? <span className="text-faint">{c?.note ?? "–"}</span> : <>{fmtD(-c.delta)}{c.ci90 && <span className="block text-[11.5px] text-faint">{fmtD(-c.ci90[1])} to {fmtD(-c.ci90[0])}</span>}</>}</td>
                  <td>{isLeader ? <Badge tone="teal">leader</Badge> : sel.excluded[k] ? <Badge tone="neutral" title={sel.excluded[k].join("; ")}>excluded: {sel.excluded[k][0]}</Badge> : <VerdictBadge comp={c} text={summary.verdict_text} />}</td>
                </tr>
              );
            })}
            {baseline && sel.comparisons[baseline] && (
              <tr className="text-muted">
                <td>{byKey.get(baseline)?.name ?? "baseline"} <Badge>baseline</Badge></td>
                <td className="num">{fmt((byKey.get(baseline)?.val_metrics ?? (selOnTest ? byKey.get(baseline)?.metrics : null))?.[sel.primary_metric as "f1_macro"] as number | undefined)}</td>
                <td className="num">{fmtD(sel.comparisons[baseline].delta != null ? -sel.comparisons[baseline].delta! : null)}</td>
                <td><span className="text-[12px]">reference only</span></td>
              </tr>
            )}
          </tbody>
        </Table>
        <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">
          <span className="text-teal">equivalent</span>: the 90% CI of Δ lies inside the margin (TOST). <span className="text-coral">worse</span>: a difference was detected. <span className="text-amber">inconclusive</span>: the data cannot tell – this is not a tie. p-values are {sel.correction ?? "Holm"}-corrected over {sel.family_size ?? 0} comparison(s).
        </div>
      </Panel>
    </div>
  );
}
