import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FileJson, Play, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, streamNdjson } from "../api/client";
import type { AgentRun, AgentSuiteSummary, TraceEvent } from "../api/agents";
import { Tabs, useTab } from "../components/Tabs";
import { TraceView } from "../components/TraceView";
import { Badge, Button, cx, ErrorNotice, Field, Input, Notice, PageHeader, Panel, Select, StatusBadge, Table, Textarea, Toggle } from "../components/ui";
import { duration, humanize, pct, when } from "../lib/format";
import { useModels } from "../lib/queries";

const TABS = ["run", "playground", "suites", "runs"] as const;
type Tab = (typeof TABS)[number];
const LABELS: Record<Tab, string> = { run: "Benchmark", playground: "Playground", suites: "Suites", runs: "History" };

function useSuites() {
  return useQuery({ queryKey: ["agent-suites"], queryFn: () => api<{ suites: AgentSuiteSummary[] }>("/api/agents/suites") });
}

function useChatModels() {
  const models = useModels();
  return (models.data?.models ?? []).filter((m) => !m.is_embedding);
}

function ToolSupport({ caps }: { caps: string[] }) {
  return caps.includes("tools") ? <Badge tone="teal" title="Ollama reports native tool calling">native tools</Badge> : <Badge tone="violet" title="Tested with tools described in the prompt (JSON protocol)">prompted</Badge>;
}

// ------------------------------------------------------------------ benchmark form
function RunForm() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const suites = useSuites();
  const chatModels = useChatModels();
  const [suiteId, setSuiteId] = useState("agent-quick");
  const [cats, setCats] = useState<string[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [mode, setMode] = useState("auto");
  const [repeats, setRepeats] = useState(1);
  const [maxSteps, setMaxSteps] = useState(6);
  const [temperature, setTemperature] = useState(0);
  const [seed, setSeed] = useState("42");
  const [numCtx, setNumCtx] = useState(4096);
  const [timeout, setTimeoutS] = useState("");
  const [name, setName] = useState("");
  const suite = suites.data?.suites.find((s) => s.id === suiteId);
  useEffect(() => setCats([]), [suiteId]);
  const tasks = suite ? (cats.length ? cats.reduce((a, c) => a + (suite.categories[c] ?? 0), 0) : suite.task_count) : 0;

  const start = useMutation({
    mutationFn: () => api<AgentRun>("/api/agents/runs", {
      method: "POST",
      json: { suite_id: suiteId, models: picked, mode, categories: cats.length ? cats : undefined, repeats, max_steps: maxSteps, temperature, seed: seed === "" ? null : Number(seed), num_ctx: numCtx, timeout_s: timeout ? Number(timeout) : undefined, name: name || undefined },
    }),
    onSuccess: (run) => {
      qc.invalidateQueries({ queryKey: ["agent-runs"] });
      nav(`/agents/runs/${run.id}`);
    },
  });

  return (
    <Panel title="New agent benchmark">
      <div className="space-y-5">
        <fieldset>
          <legend className="mb-2 text-[12.5px] text-muted">Task suite</legend>
          <div className="space-y-1.5">
            {suites.data?.suites.map((s) => (
              <label key={s.id} className={cx("block cursor-pointer rounded-md border px-3 py-2", s.id === suiteId ? "border-amber/60 bg-amber/5" : "border-line hover:border-line-strong")}>
                <div className="flex items-center gap-2">
                  <input type="radio" name="agent-suite" checked={s.id === suiteId} onChange={() => setSuiteId(s.id)} className="accent-amber" />
                  <span className="text-[13.5px]">{s.name}</span>
                  {s.source === "custom" && <Badge tone="violet">yours</Badge>}
                  <span className="ml-auto num text-[12px] text-faint">{s.task_count} tasks · {s.tool_count} tools</span>
                </div>
                {s.id === suiteId && <p className="mt-1 pl-6 text-[12.5px] text-muted">{s.description}</p>}
              </label>
            ))}
          </div>
        </fieldset>
        {suite && Object.keys(suite.categories).length > 1 && (
          <fieldset>
            <legend className="mb-2 text-[12.5px] text-muted">Categories (none = all)</legend>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(suite.categories).map(([c, n]) => {
                const on = cats.includes(c);
                return <button key={c} type="button" aria-pressed={on} onClick={() => setCats((x) => (on ? x.filter((y) => y !== c) : [...x, c]))} className={cx("rounded-full border px-2.5 py-0.5 text-[12.5px]", on ? "border-amber/60 bg-amber/10 text-ink" : "border-line text-muted hover:text-ink")}>{humanize(c)} <span className="num text-faint">{n}</span></button>;
              })}
            </div>
          </fieldset>
        )}
        <fieldset>
          <legend className="mb-2 text-[12.5px] text-muted">Models</legend>
          {!chatModels.length ? <p className="text-[13px] text-faint">No chat models found. Try <code>ollama pull qwen2.5:1.5b</code> (supports tool calling).</p> : (
            <div className="space-y-1">
              {chatModels.map((m) => (
                <label key={m.id} className={cx("flex items-center gap-2 text-[13.5px]", !m.allowed && "opacity-50")} title={m.blocked_reason ?? undefined}>
                  <input type="checkbox" className="accent-amber" disabled={!m.allowed} checked={picked.includes(m.id)} onChange={(e) => setPicked((p) => (e.target.checked ? [...p, m.id] : p.filter((x) => x !== m.id)))} />
                  <span className="truncate">{m.name}</span>
                  {m.parameter_size && <span className="text-[12px] text-faint">{m.parameter_size}</span>}
                  <ToolSupport caps={m.capabilities} />
                </label>
              ))}
            </div>
          )}
        </fieldset>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Tool-calling mode" hint={mode === "auto" ? "Native when the model supports it, else prompted JSON" : undefined}>
            <Select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="auto">Auto</option><option value="native">Native only</option><option value="prompted">Prompted JSON for all</option>
            </Select>
          </Field>
          <Field label="Max steps per task"><Input type="number" min={1} max={20} value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} /></Field>
          <Field label="Repeats"><Input type="number" min={1} max={5} value={repeats} onChange={(e) => setRepeats(Math.max(1, Number(e.target.value)))} /></Field>
          <Field label="Temperature"><Input type="number" step="0.1" min={0} max={2} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} /></Field>
          <Field label="Seed"><Input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="none" /></Field>
          <Field label="Context tokens"><Input type="number" min={512} step={512} value={numCtx} onChange={(e) => setNumCtx(Number(e.target.value))} /></Field>
          <Field label="Timeout per task (s)"><Input type="number" min={10} value={timeout} onChange={(e) => setTimeoutS(e.target.value)} placeholder="180" /></Field>
          <Field label="Run name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="automatic" /></Field>
        </div>
        {start.isError && <ErrorNotice error={start.error} />}
        <div className="flex items-center gap-3">
          <Button variant="primary" onClick={() => start.mutate()} busy={start.isPending} disabled={!picked.length || !tasks}><Play className="size-4" />Start agent benchmark</Button>
          <span className="num text-[12.5px] text-muted">{picked.length ? `${tasks * picked.length * repeats} agent episodes` : "Pick at least one model"}</span>
        </div>
      </div>
    </Panel>
  );
}

function RunsTable({ compact }: { compact?: boolean }) {
  const qc = useQueryClient();
  const runs = useQuery({
    queryKey: ["agent-runs"],
    queryFn: () => api<{ runs: AgentRun[] }>("/api/agents/runs"),
    refetchInterval: (q) => (q.state.data?.runs.some((r) => r.status === "running" || r.status === "queued") ? 1500 : 15000),
  });
  const list = (runs.data?.runs ?? []).slice(0, compact ? 6 : 200);
  return (
    <Panel title={compact ? "Recent agent runs" : "All agent runs"} bodyClass="p-0">
      {!list.length ? <div className="p-4 text-[13px] text-faint">No runs yet.</div> : (
        <Table>
          <thead><tr><th>Run</th>{!compact && <><th>Status</th><th>Best success</th><th>Duration</th><th>Started</th><th /></>}</tr></thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.id}>
                <td className={compact ? "max-w-[420px]" : "max-w-[260px]"}>
                  <Link to={`/agents/runs/${r.id}`} className="block truncate hover:text-amber" title={r.name}>{r.name}</Link>
                  <div className="flex flex-wrap items-center gap-1.5 text-[12px] text-faint">
                    {compact && <StatusBadge status={r.status} />}
                    <span>{r.suite_id} v{r.suite_version}</span>
                    {(r.status === "running" || r.status === "queued") && <span className="num">{r.progress_done}/{r.progress_total}</span>}
                    {compact && r.summary[0] && <span>best: <span className="num text-ink">{pct(r.summary[0].success_rate)}</span> {r.summary[0].model}</span>}
                  </div>
                </td>
                {!compact && <>
                  <td><StatusBadge status={r.status} /></td>
                  <td className="whitespace-nowrap">{r.summary[0] ? <span className="num">{pct(r.summary[0].success_rate)} <span className="text-faint">{r.summary[0].model}</span></span> : "–"}</td>
                  <td className="num">{duration(r.duration_s)}</td>
                  <td className="text-muted whitespace-nowrap">{when(r.started_at ?? r.created_at)}</td>
                  <td className="text-right">{(r.status === "running" || r.status === "queued") && <Button variant="ghost" onClick={() => api(`/api/agents/runs/${r.id}/cancel`, { method: "POST" }).then(() => qc.invalidateQueries({ queryKey: ["agent-runs"] }))}>Cancel</Button>}</td>
                </>}
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ playground
const EXAMPLES = [
  "What's the weather like in Lahore right now?",
  "What is the total of order A-1002 in Pakistani rupees?",
  "Draft an email to sara@example.com about tomorrow's 10am meeting, but do NOT send it.",
  "Use the calculator: what is 17.5% of 2,480?",
];

function Playground() {
  const suites = useSuites();
  const chatModels = useChatModels().filter((m) => m.allowed);
  const [model, setModel] = useState("");
  const [suiteId, setSuiteId] = useState("agent-core");
  const [tools, setTools] = useState<string[] | null>(null);
  const [docs, setDocs] = useState(false);
  const [mode, setMode] = useState("auto");
  const [maxSteps, setMaxSteps] = useState(6);
  const [system, setSystem] = useState("");
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const abort = useRef<AbortController | null>(null);
  const suite = suites.data?.suites.find((s) => s.id === suiteId);
  useEffect(() => { if (!model && chatModels[0]) setModel(chatModels[0].id); }, [chatModels, model]);
  useEffect(() => setTools(null), [suiteId]);
  const active = tools ?? suite?.tools ?? [];

  const run = async () => {
    setEvents([]);
    setError(null);
    setRunning(true);
    abort.current = new AbortController();
    try {
      await streamNdjson("/api/agents/playground", {
        model, prompt, suite_id: suiteId || undefined, tools: suiteId ? active : undefined,
        extra_builtins: docs ? ["search_documents"] : [], system: system || undefined, mode, max_steps: maxSteps,
      }, (e) => setEvents((prev) => [...prev, e as unknown as TraceEvent]), abort.current.signal);
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid items-start gap-6 xl:grid-cols-[380px_1fr]">
      <Panel title="Try an agent">
        <div className="space-y-4">
          <Field label="Model">
            <Select value={model} onChange={(e) => setModel(e.target.value)}>
              {chatModels.map((m) => <option key={m.id} value={m.id}>{m.name}{m.capabilities.includes("tools") ? " (native tools)" : ""}</option>)}
            </Select>
          </Field>
          <Field label="Tools from suite">
            <Select value={suiteId} onChange={(e) => setSuiteId(e.target.value)}>
              <option value="">No suite tools</option>
              {suites.data?.suites.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
            </Select>
          </Field>
          {suite && (
            <div className="flex flex-wrap gap-1.5">
              {suite.tools.map((t) => {
                const on = active.includes(t);
                return <button key={t} type="button" aria-pressed={on} onClick={() => setTools(on ? active.filter((x) => x !== t) : [...active, t])} className={cx("rounded-full border px-2.5 py-0.5 text-[12.5px] font-mono", on ? "border-amber/60 bg-amber/10 text-ink" : "border-line text-faint")}>{t}</button>;
              })}
            </div>
          )}
          <Toggle checked={docs} onChange={setDocs} label="Search my documents" description="Adds a real search_documents tool over the files you indexed on the Documents page." />
          <div className="grid grid-cols-2 gap-3">
            <Field label="Mode"><Select value={mode} onChange={(e) => setMode(e.target.value)}><option value="auto">Auto</option><option value="native">Native</option><option value="prompted">Prompted JSON</option></Select></Field>
            <Field label="Max steps"><Input type="number" min={1} max={20} value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} /></Field>
          </div>
          <Field label="System prompt (optional)" hint="Empty = the suite's system prompt"><Textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} /></Field>
        </div>
      </Panel>
      <div className="space-y-4 min-w-0">
        <Panel title="Prompt">
          <form onSubmit={(e) => { e.preventDefault(); if (!running) run(); }} className="space-y-3">
            <Textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} aria-label="Prompt" />
            <div className="flex flex-wrap gap-1.5">{EXAMPLES.map((x) => <button key={x} type="button" onClick={() => setPrompt(x)} className="rounded-full border border-line px-2.5 py-0.5 text-[12px] text-muted hover:text-ink">{x.length > 48 ? x.slice(0, 47) + "…" : x}</button>)}</div>
            <div className="flex gap-2">
              <Button type="submit" variant="primary" busy={running} disabled={!model || !prompt.trim()}><Play className="size-4" />Run agent</Button>
              {running && <Button type="button" onClick={() => abort.current?.abort()}><Square className="size-4" />Stop</Button>}
            </div>
          </form>
        </Panel>
        {error ? <ErrorNotice error={error} /> : null}
        <Panel title="Trace">
          {!events.length && !running ? <p className="text-[13px] text-faint">Each model turn, tool call and tool result appears here as it happens. Tools from suites are mocked – nothing real is sent or deleted.</p> : <TraceView events={events} live={running} />}
        </Panel>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ suites
function Suites() {
  const qc = useQueryClient();
  const suites = useSuites();
  const [text, setText] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const detail = useQuery({ queryKey: ["agent-suite", open], queryFn: () => api<AgentSuiteSummary & { definition: Record<string, unknown> }>(`/api/agents/suites/${open}`), enabled: !!open });
  const parse = () => {
    try {
      return JSON.parse(text);
    } catch (e) {
      throw new Error(`Not valid JSON: ${(e as Error).message}`);
    }
  };
  const validate = useMutation({ mutationFn: () => api<{ valid: boolean; summary: AgentSuiteSummary }>("/api/agents/suites/validate", { method: "POST", json: parse() }) });
  const save = useMutation({
    mutationFn: () => api<AgentSuiteSummary>("/api/agents/suites", { method: "POST", json: parse() }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agent-suites"] }),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api(`/api/agents/suites/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agent-suites"] }),
  });
  const loadTemplate = async () => setText(JSON.stringify(await api("/api/agents/template"), null, 2));
  const defs = detail.data?.definition as { tools?: { name: string; description: string; builtin?: string; risky?: boolean }[]; tasks?: { id: string; category: string; prompt: string }[] } | undefined;

  return (
    <div className="grid items-start gap-6 xl:grid-cols-2">
      <div className="space-y-3 min-w-0">
        {suites.data?.suites.map((s) => (
          <div key={s.id} className="rounded-lg border border-line bg-panel p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{s.name}</span>
              <Badge tone={s.source === "custom" ? "violet" : "neutral"}>{s.source === "custom" ? "yours" : "built-in"}</Badge>
              <span className="num text-[12px] text-faint">{s.id} v{s.version} · {s.task_count} tasks · {s.tool_count} tools</span>
              <span className="ml-auto flex gap-1">
                <Button variant="ghost" onClick={() => setOpen(open === s.id ? null : s.id)}>{open === s.id ? "Hide" : "View"}</Button>
                {s.source === "custom" && <Button variant="ghost" onClick={() => confirm(`Delete suite “${s.name}”?`) && remove.mutate(s.id)} aria-label="Delete suite"><Trash2 className="size-4" /></Button>}
              </span>
            </div>
            <p className="mt-1 text-[12.5px] text-muted">{s.description}</p>
            <div className="mt-2 flex flex-wrap gap-1.5">{Object.entries(s.categories).map(([c, n]) => <Badge key={c}>{humanize(c)} <span className="num text-faint">{n}</span></Badge>)}</div>
            {open === s.id && defs && (
              <div className="mt-3 space-y-3 text-[12.5px]">
                <div>
                  <div className="mb-1 text-muted">Tools</div>
                  <ul className="space-y-0.5">{defs.tools?.map((t) => <li key={t.name}><code className="text-amber">{t.name}</code> {t.builtin && <Badge tone="teal">built-in</Badge>} {t.risky && <Badge tone="coral">side effects (mocked)</Badge>} <span className="text-muted">{t.description}</span></li>)}</ul>
                </div>
                <div>
                  <div className="mb-1 text-muted">Tasks</div>
                  <ul className="max-h-64 space-y-0.5 overflow-auto">{defs.tasks?.map((t) => <li key={t.id}><span className="num text-faint">{t.id}</span> <Badge>{humanize(t.category)}</Badge> {t.prompt}</li>)}</ul>
                </div>
                <Button onClick={() => setText(JSON.stringify(detail.data!.definition, null, 2))}><FileJson className="size-4" />Copy into editor</Button>
              </div>
            )}
          </div>
        ))}
        {remove.isError && <ErrorNotice error={remove.error} />}
      </div>
      <Panel title="Create or edit a custom suite">
        <div className="space-y-3">
          <p className="text-[13px] text-muted">Describe your own tools (name, description, JSON-schema parameters and mock responses) and tasks with the checks that define success. Test models on <em>your</em> workflow before connecting real APIs. Format: <code>backend/datasets/agent/README.md</code>.</p>
          <div className="flex flex-wrap gap-2">
            <Button onClick={loadTemplate}><FileJson className="size-4" />Load template</Button>
            <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-raised px-3 text-[13px] hover:border-line-strong">
              Open .json file
              <input type="file" accept=".json,application/json" className="sr-only" onChange={async (e) => { const f = e.target.files?.[0]; if (f) setText(await f.text()); }} />
            </label>
          </div>
          <Textarea rows={20} value={text} onChange={(e) => setText(e.target.value)} className="font-mono text-[12px]" placeholder='{"id": "my-suite", "name": "…", "tools": [...], "tasks": [...]}' spellCheck={false} aria-label="Suite JSON" />
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => validate.mutate()} busy={validate.isPending} disabled={!text.trim()}><CheckCircle2 className="size-4" />Validate</Button>
            <Button variant="primary" onClick={() => save.mutate()} busy={save.isPending} disabled={!text.trim()}>Save suite</Button>
          </div>
          {validate.isError && <ErrorNotice error={validate.error} />}
          {validate.data && !validate.isError && <Notice title="Valid suite">{validate.data.summary.task_count} tasks, {validate.data.summary.tool_count} tools.</Notice>}
          {save.isError && <ErrorNotice error={save.error} />}
          {save.data && <Notice title={`Saved “${save.data.name}”`}>Pick it on the Benchmark tab.</Notice>}
        </div>
      </Panel>
    </div>
  );
}

export default function AgentArena() {
  const [tab, setTab] = useTab(TABS, "run");
  return (
    <div className="max-w-[1320px]">
      <PageHeader title="Agent arena" description="Which local model makes the best agent on this computer? Models get tools and multi-step tasks; every tool call, argument and answer is checked automatically – including safety (no forbidden actions, no obeying injected instructions)." />
      <Tabs tabs={TABS} active={tab} onChange={setTab} labels={LABELS} />
      {tab === "run" && (
        <div className="grid items-start gap-6 xl:grid-cols-[420px_1fr]">
          <RunForm />
          <div className="space-y-4 min-w-0">
            <Notice title="What is measured">
              <ul className="mt-1 list-disc space-y-0.5 pl-4">
                <li><span className="text-ink">Right tools / arguments / order</span> – did the model call what the task needs, with correct values?</li>
                <li><span className="text-ink">Correct answer</span> – checked with the same deterministic evaluators as the benchmarks.</li>
                <li><span className="text-ink">Restraint</span> – no tool calls when a question needs none.</li>
                <li><span className="text-ink">Safety</span> – no forbidden actions, even when a document or tool result contains injected instructions.</li>
                <li><span className="text-ink">Recovery, steps, invalid calls, latency</span> – how efficiently it gets there.</li>
              </ul>
            </Notice>
            <RunsTable compact />
          </div>
        </div>
      )}
      {tab === "playground" && <Playground />}
      {tab === "suites" && <Suites />}
      {tab === "runs" && <RunsTable />}
    </div>
  );
}
