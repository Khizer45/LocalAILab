import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, File, Folder, Play, Square, X } from "lucide-react";
import { useRef, useState } from "react";
import { api, streamNdjson } from "../api/client";
import { ModelSelect } from "../components/ModelSelect";
import { Badge, Button, cx, ErrorNotice, Field, Input, Notice, PageHeader, Panel, Select, Spinner, Table, Textarea } from "../components/ui";
import { bytes, ms } from "../lib/format";

type Rule = "allow" | "ask" | "deny";
interface ToolRow { name: string; description: string; policy: Rule; writes: boolean }
interface Config { workspace: string; webhooks: string[]; approval_timeout_s: number; tools: ToolRow[] }
type Ev =
  | { type: "start"; mode: string; model: string; tools: string[] }
  | { type: "model"; step: number; ms: number; content: string; tool_calls?: { name: string; arguments: unknown }[]; repaired?: boolean }
  | { type: "tool"; step: number; name: string; arguments: unknown; result: unknown; ms: number; decision?: string; valid?: boolean }
  | { type: "approval_required"; approval_id: string; tool: string; arguments: unknown; preview: unknown }
  | { type: "approval_decision"; approval_id: string; decision: string; note?: string | null }
  | { type: "final"; answer: string; stop_reason: string; totals?: Record<string, number> }
  | { type: "note"; message: string }
  | { type: "error"; code: string; message: string; hint?: string };

const RULE_TONE = { allow: "teal", ask: "amber", deny: "coral" } as const;

export default function WorkAgent({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const cfg = useQuery({ queryKey: ["work-agent-config"], queryFn: () => api<Config>("/api/work-agent/config"), retry: false });
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("Read notes/meeting.txt and write a three-line summary to notes/summary.md.");
  const [maxSteps, setMaxSteps] = useState(6);
  const [events, setEvents] = useState<Ev[]>([]);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<unknown>(null);
  const [decided, setDecided] = useState<Record<string, string>>({});
  const abort = useRef<AbortController | null>(null);

  const policy = useMutation({
    mutationFn: (body: Record<string, Rule>) => api("/api/work-agent/policy", { method: "PUT", json: body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["work-agent-config"] }),
  });
  const decide = useMutation({
    mutationFn: ({ id, approve }: { id: string; approve: boolean }) => api(`/api/work-agent/approvals/${id}`, { method: "POST", json: { approve } }),
    onSuccess: (_d, v) => setDecided((x) => ({ ...x, [v.id]: v.approve ? "approved" : "rejected" })),
  });

  const start = async () => {
    setEvents([]); setRunError(null); setDecided({}); setRunning(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    try {
      await streamNdjson<Ev>("/api/work-agent/run", { model, prompt, max_steps: maxSteps, interactive: true }, (e) => setEvents((x) => [...x, e]), ctrl.signal);
    } catch (err) { setRunError(err); }
    setRunning(false);
    qc.invalidateQueries({ queryKey: ["work-files"] });
  };
  const stop = () => { abort.current?.abort(); setRunning(false); };

  if (cfg.isError) return <div className="max-w-3xl"><PageHeader title="Work agent" /><ErrorNotice error={cfg.error} /></div>;
  const final = events.find((e): e is Extract<Ev, { type: "final" }> => e.type === "final");
  const decisions = Object.fromEntries(events.filter((e): e is Extract<Ev, { type: "approval_decision" }> => e.type === "approval_decision").map((e) => [e.approval_id, e.decision]));

  return (
    <div className="max-w-[1320px]">
      <PageHeader title="Work agent" description={<>A local model that can use a small set of tools on files in one workspace folder. Reading is allowed; writing files and calling webhooks <strong className="text-ink">wait for your approval</strong> unless an administrator changes the policy. Every tool call is in the audit log.</>} />
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="space-y-6 min-w-0">
          <Panel title="Task">
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-[1fr_140px]">
                <Field label="Model" hint="Small models (1–4B) often call tools wrongly – check each approval preview."><ModelSelect value={model} onChange={setModel} /></Field>
                <Field label="Max steps"><Input type="number" min={1} max={12} value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} /></Field>
              </div>
              <Field label="What should it do?"><Textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} /></Field>
              <div className="flex gap-2">
                {running ? <Button onClick={stop}><Square className="size-4" />Stop</Button> : <Button variant="primary" onClick={start} disabled={!model || !prompt.trim()}><Play className="size-4" />Run</Button>}
                {running && <Spinner label="Working…" />}
              </div>
              {runError != null && <ErrorNotice error={runError} />}
            </div>
          </Panel>

          {events.length > 0 && (
            <Panel title="Steps">
              <ol className="space-y-3">
                {events.map((e, i) => <li key={i}><EventView e={e} decision={e.type === "approval_required" ? (decisions[e.approval_id] ?? decided[e.approval_id]) : undefined}
                  onDecide={(approve) => e.type === "approval_required" && decide.mutate({ id: e.approval_id, approve })} busy={decide.isPending} /></li>)}
              </ol>
              {decide.isError && <div className="mt-3"><ErrorNotice error={decide.error} /></div>}
            </Panel>
          )}
          {final && (
            <Panel title="Answer">
              <div className="whitespace-pre-wrap text-[13.5px]">{final.answer || <span className="text-faint">(no answer – stopped: {final.stop_reason})</span>}</div>
              {final.totals && <div className="mt-2 text-[12px] text-faint">{final.totals.tool_calls ?? 0} tool calls · stop reason {final.stop_reason}</div>}
            </Panel>
          )}
        </div>

        <div className="space-y-6 min-w-0">
          <Panel title="Tool policy" bodyClass="p-0">
            {cfg.isLoading ? <div className="p-4"><Spinner /></div> : (
              <Table>
                <thead><tr><th>Tool</th><th>Rule</th></tr></thead>
                <tbody>{cfg.data!.tools.map((t) => (
                  <tr key={t.name}>
                    <td><div className="num">{t.name}{t.writes && <Badge tone="amber">changes things</Badge>}</div><div className="text-[12px] text-faint">{t.description}</div></td>
                    <td>{isAdmin ? (
                      <Select className="w-24" value={t.policy} onChange={(e) => policy.mutate({ [t.name]: e.target.value as Rule })} aria-label={`Rule for ${t.name}`}>
                        <option value="allow">allow</option><option value="ask">ask</option><option value="deny">deny</option>
                      </Select>
                    ) : <Badge tone={RULE_TONE[t.policy]}>{t.policy}</Badge>}</td>
                  </tr>
                ))}</tbody>
              </Table>
            )}
            {policy.isError && <div className="p-3"><ErrorNotice error={policy.error} /></div>}
            {cfg.data && <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">Approvals expire after {cfg.data.approval_timeout_s} s. Runs started by scripts or schedules (non-interactive) refuse every “ask” tool. Webhooks the agent may call: {cfg.data.webhooks.join(", ") || "none"}.</div>}
          </Panel>
          {cfg.data && <Workspace root={cfg.data.workspace} />}
          <Notice tone="warn" title="Treat the agent like a new intern">Text inside files and documents can try to instruct the model (prompt injection). The approval step shows exactly what will be written or sent – read it before you approve.</Notice>
        </div>
      </div>
    </div>
  );
}

function EventView({ e, decision, onDecide, busy }: { e: Ev; decision?: string; onDecide: (approve: boolean) => void; busy: boolean }) {
  const box = "rounded-md border border-line px-3 py-2 text-[13px]";
  switch (e.type) {
    case "start":
      return <div className="text-[12.5px] text-muted">Started with <span className="text-ink">{e.model}</span> ({e.mode} tool calling) · tools: {e.tools.join(", ")}</div>;
    case "model":
      return (
        <div className={box}>
          <div className="text-[12px] text-faint">step {e.step} · model · {ms(e.ms)}{e.repaired && " · format repaired"}</div>
          {e.content && <div className="mt-1 whitespace-pre-wrap">{e.content}</div>}
          {e.tool_calls?.map((c, i) => <div key={i} className="mt-1 num text-[12px] text-sky">→ {c.name}({JSON.stringify(c.arguments)})</div>)}
        </div>
      );
    case "tool":
      return (
        <div className={cx(box, e.valid === false && "border-coral/40")}>
          <div className="text-[12px] text-faint">step {e.step} · tool <span className="num text-ink">{e.name}</span> · {ms(e.ms)}{e.decision && ` · ${e.decision}`}</div>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap text-[12px] text-muted">{JSON.stringify(e.result, null, 2)}</pre>
        </div>
      );
    case "approval_required":
      return (
        <div className="rounded-md border border-amber/50 bg-amber/8 px-3 py-2.5 text-[13px]">
          <div className="font-medium">Approval needed: <span className="num">{e.tool}</span></div>
          <pre className="mt-1.5 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-bg p-2 text-[12px]">{typeof e.preview === "string" ? e.preview : JSON.stringify(e.preview ?? e.arguments, null, 2)}</pre>
          {decision ? <div className="mt-2"><Badge tone={decision === "approved" ? "teal" : "coral"}>{decision}</Badge></div> : (
            <div className="mt-2 flex gap-2">
              <Button variant="primary" onClick={() => onDecide(true)} busy={busy}><Check className="size-4" />Approve</Button>
              <Button variant="danger" onClick={() => onDecide(false)} busy={busy}><X className="size-4" />Reject</Button>
            </div>
          )}
        </div>
      );
    case "approval_decision":
      return null;
    case "final":
      return <div className="text-[12.5px] text-muted">Finished ({e.stop_reason}).</div>;
    case "note":
      return <div className="text-[12.5px] text-amber">{e.message}</div>;
    case "error":
      return <ErrorNotice error={{ message: e.message, hint: e.hint }} />;
  }
}

function Workspace({ root }: { root: string }) {
  const [folder, setFolder] = useState(".");
  const [open, setOpen] = useState<string | null>(null);
  const list = useQuery({ queryKey: ["work-files", folder], queryFn: () => api<{ folder: string; items: { name: string; type: "file" | "folder"; size_bytes: number | null }[]; error?: string }>(`/api/work-agent/files?folder=${encodeURIComponent(folder)}`) });
  const content = useQuery({ queryKey: ["work-file", open], queryFn: () => api<{ content?: string; error?: string; truncated?: boolean }>(`/api/work-agent/files/content?path=${encodeURIComponent(open!)}`), enabled: !!open });
  const join = (name: string) => (folder === "." ? name : `${folder}/${name}`);
  return (
    <Panel title="Workspace" bodyClass="p-0">
      <div className="border-b border-line px-4 py-2 text-[12px] text-faint break-all">{root}{folder !== "." && <> / <span className="text-ink">{folder}</span> <button className="ml-2 text-amber hover:underline" onClick={() => setFolder(folder.split("/").slice(0, -1).join("/") || ".")}>up</button></>}</div>
      {list.data?.error ? <div className="p-4 text-[13px] text-faint">{list.data.error}</div> : !list.data?.items.length ? <div className="p-4 text-[13px] text-faint">Empty. Put files here for the agent to work on – it cannot see anything outside this folder.</div> : (
        <ul className="max-h-72 overflow-auto">{list.data.items.map((it) => (
          <li key={it.name}>
            <button className="flex w-full items-center gap-2 px-4 py-1.5 text-left text-[13px] hover:bg-raised/60" onClick={() => (it.type === "folder" ? setFolder(join(it.name)) : setOpen(join(it.name)))}>
              {it.type === "folder" ? <Folder className="size-4 text-amber" /> : <File className="size-4 text-muted" />}
              <span className="min-w-0 flex-1 truncate">{it.name}</span>
              {it.size_bytes != null && <span className="num text-[12px] text-faint">{bytes(it.size_bytes)}</span>}
            </button>
          </li>
        ))}</ul>
      )}
      {open && (
        <div className="border-t border-line p-3">
          <div className="mb-1 flex items-center gap-2 text-[12px] text-muted"><span className="num min-w-0 flex-1 truncate">{open}</span><button onClick={() => setOpen(null)} className="hover:text-ink" aria-label="Close file"><X className="size-4" /></button></div>
          {content.isLoading ? <Spinner /> : <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded bg-bg p-2 text-[12px]">{content.data?.content ?? content.data?.error}</pre>}
        </div>
      )}
    </Panel>
  );
}
