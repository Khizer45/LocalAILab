import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarClock, Pause, Play, PlayCircle, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { SuiteSummary } from "../api/types";
import { Badge, Button, cx, Empty, ErrorNotice, Field, Input, PageHeader, Panel, Select, Spinner, StatusBadge, Table, Toggle } from "../components/ui";
import { when } from "../lib/format";
import { useModels } from "../lib/queries";

interface Schedule {
  id: number; name: string; kind: "benchmark" | "agent" | "backup"; request: Record<string, unknown>; every_hours: number | null; daily_at: string | null;
  gates: Record<string, number> | null; notify_webhook: string | null; enabled: boolean; next_run_at: string | null; last_run_at: string | null;
  last_run_id: number | null; last_status: string | null; last_message: string | null;
}

export default function Automation({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["schedules"], queryFn: () => api<{ schedules: Schedule[] }>("/api/schedules"), refetchInterval: 30000 });
  const remove = useMutation({ mutationFn: (id: number) => api(`/api/schedules/${id}`, { method: "DELETE" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["schedules"] }) });
  const runNow = useMutation({ mutationFn: (id: number) => api(`/api/schedules/${id}/run-now`, { method: "POST" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["schedules"] }) });
  const toggle = useMutation({
    mutationFn: (s: Schedule) => api(`/api/schedules/${s.id}`, { method: "PUT", json: { name: s.name, kind: s.kind, request: s.request, every_hours: s.every_hours, daily_at: s.daily_at, gates: s.gates, notify_webhook: s.notify_webhook, enabled: !s.enabled } }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["schedules"] }),
  });
  const runLink = (s: Schedule) => (s.last_run_id == null ? null : s.kind === "benchmark" ? `/benchmarks/runs/${s.last_run_id}` : s.kind === "agent" ? `/agents/runs/${s.last_run_id}` : null);

  return (
    <div className="max-w-[1320px]">
      <PageHeader title="Automation" description="Run benchmarks, agent evaluations and backups on a schedule, fail a gate when quality drops, and connect n8n, scripts or CI through the API." />
      <div className="grid items-start gap-6 xl:grid-cols-[400px_minmax(0,1fr)]">
        <NewSchedule isAdmin={isAdmin} />
        <div className="space-y-6 min-w-0">
          <Panel title="Schedules" bodyClass="p-0">
            {list.isLoading ? <div className="p-4"><Spinner /></div> : list.isError ? <div className="p-4"><ErrorNotice error={list.error} /></div> : !list.data!.schedules.length ? <div className="p-4"><Empty title="No schedules yet">A nightly benchmark with a quality gate catches regressions after a model or prompt update.</Empty></div> : (
              <Table>
                <thead><tr><th>Name</th><th>When</th><th>Gates</th><th>Last run</th><th>Next</th><th /></tr></thead>
                <tbody>{list.data!.schedules.map((s) => {
                  const link = runLink(s);
                  return (
                    <tr key={s.id} className={cx(!s.enabled && "opacity-60")}>
                      <td className="max-w-[220px]"><div className="truncate">{s.name}</div><Badge tone={s.kind === "backup" ? "neutral" : s.kind === "agent" ? "violet" : "sky"}>{s.kind}</Badge> {s.notify_webhook && <span className="text-[12px] text-faint">→ {s.notify_webhook}</span>}</td>
                      <td className="whitespace-nowrap">{s.every_hours ? `every ${s.every_hours} h` : `daily ${s.daily_at}`}</td>
                      <td className="max-w-[170px]">{s.gates && Object.keys(s.gates).length ? <div className="flex flex-wrap gap-1">{Object.entries(s.gates).map(([k, v]) => <Badge key={k}>{k.replace(/_/g, " ")} {v}</Badge>)}</div> : <span className="text-faint">–</span>}</td>
                      <td className="max-w-[260px]">
                        {!s.last_status ? <span className="text-faint">never</span>
                          : s.last_status === "passed" ? <Badge tone="teal">gates passed</Badge>
                          : s.last_status === "failed" ? <Badge tone="coral">gate failed</Badge>
                          : s.last_status === "submitted" ? <Badge tone="sky">running</Badge>
                          : <StatusBadge status={s.last_status} />}
                        {link && <Link to={link} className="ml-1 text-[12px] text-amber hover:underline">run {s.last_run_id}</Link>}
                        {s.last_message && <div className="truncate text-[12px] text-faint" title={s.last_message}>{s.last_message}</div>}
                      </td>
                      <td className="whitespace-nowrap text-muted">{s.enabled && s.next_run_at ? when(s.next_run_at) : "paused"}</td>
                      <td className="whitespace-nowrap text-right">
                        <Button variant="ghost" title="Run now" aria-label={`Run ${s.name} now`} onClick={() => runNow.mutate(s.id)} busy={runNow.isPending && runNow.variables === s.id}><Play className="size-4" /></Button>
                        <Button variant="ghost" title={s.enabled ? "Pause" : "Resume"} aria-label={s.enabled ? `Pause ${s.name}` : `Resume ${s.name}`} onClick={() => toggle.mutate(s)}>{s.enabled ? <Pause className="size-4" /> : <PlayCircle className="size-4" />}</Button>
                        <Button variant="ghost" aria-label={`Delete ${s.name}`} onClick={() => confirm(`Delete schedule “${s.name}”?`) && remove.mutate(s.id)}><Trash2 className="size-4" /></Button>
                      </td>
                    </tr>
                  );
                })}</tbody>
              </Table>
            )}
            {(runNow.error || toggle.error || remove.error) && <div className="p-3"><ErrorNotice error={runNow.error ?? toggle.error ?? remove.error} /></div>}
            <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">Schedules run while Local AI Lab is running (missed runs are not caught up). Daily times use this computer's clock.</div>
          </Panel>
          <Integrations />
        </div>
      </div>
    </div>
  );
}

function NewSchedule({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const suites = useQuery({ queryKey: ["suites"], queryFn: () => api<{ suites: SuiteSummary[] }>("/api/benchmarks/suites") });
  const agentSuites = useQuery({ queryKey: ["agent-suites"], queryFn: () => api<{ suites: { id: string; name: string }[] }>("/api/agents/suites") });
  const webhooks = useQuery({ queryKey: ["webhooks"], queryFn: () => api<{ webhooks: { name: string }[] }>("/api/system/webhooks"), enabled: isAdmin, retry: false });
  const models = useModels();
  const chat = (models.data?.models ?? []).filter((m) => !m.is_embedding && m.allowed && m.provider === "local");
  const [kind, setKind] = useState<"benchmark" | "agent" | "backup">("benchmark");
  const [name, setName] = useState("Nightly benchmark");
  const [when_, setWhen] = useState<"daily" | "hours">("daily");
  const [dailyAt, setDailyAt] = useState("02:30");
  const [hours, setHours] = useState(24);
  const [suite, setSuite] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const [minQuality, setMinQuality] = useState("");
  const [minCompletion, setMinCompletion] = useState("0.95");
  const [maxDrop, setMaxDrop] = useState("0.05");
  const [minSuccess, setMinSuccess] = useState("");
  const [maxViolations, setMaxViolations] = useState("0");
  const [keep, setKeep] = useState(7);
  const [hook, setHook] = useState("");
  const [enabled, setEnabled] = useState(true);

  const n = (v: string) => (v.trim() === "" ? undefined : Number(v));
  const gates = kind === "benchmark" ? { min_quality: n(minQuality), min_completion: n(minCompletion), max_drop: n(maxDrop) }
    : kind === "agent" ? { min_success: n(minSuccess), max_violation_rate: n(maxViolations), max_drop: n(maxDrop) } : {};
  const cleanGates = Object.fromEntries(Object.entries(gates).filter(([, v]) => v !== undefined));
  const suiteList = kind === "agent" ? agentSuites.data?.suites ?? [] : suites.data?.suites ?? [];
  const create = useMutation({
    mutationFn: () => api("/api/schedules", {
      method: "POST",
      json: {
        name, kind, enabled, notify_webhook: hook || null,
        every_hours: when_ === "hours" ? hours : null, daily_at: when_ === "daily" ? dailyAt : null,
        request: kind === "backup" ? { keep } : { suite_id: suite || suiteList[0]?.id, models: picked },
        gates: Object.keys(cleanGates).length ? cleanGates : null,
      },
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["schedules"] }),
  });

  return (
    <Panel title={<span className="inline-flex items-center gap-2"><CalendarClock className="size-4 text-amber" />New schedule</span>}>
      <div className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Kind"><Select value={kind} onChange={(e) => { const k = e.target.value as typeof kind; setKind(k); setSuite(""); setName(k === "backup" ? "Nightly backup" : k === "agent" ? "Weekly agent check" : "Nightly benchmark"); }}>
            <option value="benchmark">Benchmark</option><option value="agent">Agent evaluation</option><option value="backup" disabled={!isAdmin}>Backup{!isAdmin && " (admin)"}</option>
          </Select></Field>
          <Field label="Name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
          <Field label="Repeat"><Select value={when_} onChange={(e) => setWhen(e.target.value as "daily" | "hours")}><option value="daily">Daily at</option><option value="hours">Every N hours</option></Select></Field>
          {when_ === "daily" ? <Field label="Time (24 h)"><Input type="time" value={dailyAt} onChange={(e) => setDailyAt(e.target.value)} /></Field>
            : <Field label="Hours"><Input type="number" min={1} max={720} value={hours} onChange={(e) => setHours(Number(e.target.value))} /></Field>}
        </div>
        {kind !== "backup" ? <>
          <Field label="Suite"><Select value={suite || suiteList[0]?.id || ""} onChange={(e) => setSuite(e.target.value)}>{suiteList.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}</Select></Field>
          <fieldset><legend className="mb-1 text-[12.5px] text-muted">Models</legend>
            <div className="max-h-36 space-y-1 overflow-auto">{chat.map((m) => (
              <label key={m.id} className="flex items-center gap-2 text-[13px]"><input type="checkbox" className="accent-amber" checked={picked.includes(m.id)} onChange={(e) => setPicked((x) => (e.target.checked ? [...x, m.id] : x.filter((y) => y !== m.id)))} />{m.name}</label>
            ))}{!chat.length && <div className="text-[12.5px] text-faint">No local chat models.</div>}</div>
          </fieldset>
          <fieldset className="grid gap-3 sm:grid-cols-3"><legend className="mb-1 text-[12.5px] text-muted">Gates (empty = not checked; the drop check defaults to 0.05)</legend>
            {kind === "benchmark" ? <>
              <Field label="Min quality"><Input type="number" step="0.01" min={0} max={1} value={minQuality} onChange={(e) => setMinQuality(e.target.value)} placeholder="0.70" /></Field>
              <Field label="Min completion"><Input type="number" step="0.01" min={0} max={1} value={minCompletion} onChange={(e) => setMinCompletion(e.target.value)} /></Field>
            </> : <>
              <Field label="Min success"><Input type="number" step="0.01" min={0} max={1} value={minSuccess} onChange={(e) => setMinSuccess(e.target.value)} placeholder="0.80" /></Field>
              <Field label="Max violation rate"><Input type="number" step="0.01" min={0} max={1} value={maxViolations} onChange={(e) => setMaxViolations(e.target.value)} /></Field>
            </>}
            <Field label="Max drop vs last run"><Input type="number" step="0.01" min={0} max={1} value={maxDrop} onChange={(e) => setMaxDrop(e.target.value)} /></Field>
          </fieldset>
        </> : <Field label="Keep the newest" hint="Older scheduled backups are deleted."><Input type="number" min={1} max={100} value={keep} onChange={(e) => setKeep(Number(e.target.value))} /></Field>}
        {isAdmin && (
          <Field label="Notify webhook (optional)" hint="Called when a scheduled run fails a gate – e.g. an n8n workflow that posts to your team chat. Every alert is also in the audit log.">
            <Select value={hook} onChange={(e) => setHook(e.target.value)}><option value="">none</option>{webhooks.data?.webhooks.map((h) => <option key={h.name} value={h.name}>{h.name}</option>)}</Select>
          </Field>
        )}
        <Toggle checked={enabled} onChange={setEnabled} label="Enabled" />
        <Button variant="primary" onClick={() => create.mutate()} busy={create.isPending} disabled={!name || (kind !== "backup" && (!picked.length || !suiteList.length))}>Create schedule</Button>
        {create.isError && <ErrorNotice error={create.error} />}
      </div>
    </Panel>
  );
}

function Integrations() {
  const origin = typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8000";
  const code = "rounded bg-bg p-2 text-[12px] overflow-x-auto";
  return (
    <Panel title="Connect other tools">
      <div className="grid gap-5 2xl:grid-cols-2 text-[13px]">
        <div>
          <div className="font-medium">OpenAI-compatible API</div>
          <p className="mt-1 text-muted">Any client that speaks the OpenAI chat format (n8n's OpenAI node, LangChain, scripts) can use your local models. Requests stay on this computer; nothing is sent to OpenAI.</p>
          <pre className={code}>{`base URL: ${origin}/v1
API key : a token from Administration → API tokens
          (scope "chat"; any value works when auth is off)

POST /v1/chat/completions
{"model": "qwen3:1.7b",
 "messages": [{"role": "user", "content": "Hi"}],
 "local_ai_lab": {"use_documents": true}}`}</pre>
        </div>
        <div>
          <div className="font-medium">Command line and CI</div>
          <p className="mt-1 text-muted">The <code className="text-ink">lal</code> CLI starts runs and fails a CI job when a gate is not met (exit code 1 = gate failed, 2 = error).</p>
          <pre className={code}>{`scripts\\lal.bat bench run --suite quick --model qwen3:1.7b ^
    --gate-quality 0.7 --gate-completion 0.95
scripts\\lal.bat arena run --dataset 3 --gate-final 0.8 --require-clear
scripts\\lal.bat bundle export --kind benchmark --run 12 -o run12.zip`}</pre>
          <p className="mt-2 text-muted">Signed webhooks, n8n examples and signature checks: <code className="text-ink">docs/AUTOMATION.md</code>.</p>
        </div>
      </div>
    </Panel>
  );
}
