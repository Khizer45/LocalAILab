import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, Copy, Download, KeyRound, RotateCcw, ShieldCheck, Trash2, Upload, Webhook as WebhookIcon } from "lucide-react";
import { useState } from "react";
import { api, download } from "../api/client";
import { Tabs, useTab } from "../components/Tabs";
import { Badge, Button, cx, Empty, ErrorNotice, Field, Input, Notice, PageHeader, Panel, Select, Spinner, Table, Toggle } from "../components/ui";
import { bytes, when } from "../lib/format";

const TABS = ["backups", "tokens", "webhooks", "audit"] as const;
const LABELS = { backups: "Backup & restore", tokens: "API tokens", webhooks: "Webhooks", audit: "Audit log" } as const;

export default function Admin({ isAdmin }: { isAdmin: boolean }) {
  const [tab, setTab] = useTab(TABS, isAdmin ? "backups" : "tokens");
  return (
    <div className="max-w-[1200px]">
      <PageHeader title="Administration" description="Backups, access tokens for scripts and n8n, outgoing webhooks and the audit log." />
      <Tabs tabs={TABS} active={tab} onChange={setTab} labels={LABELS} />
      {tab !== "tokens" && !isAdmin ? <Notice tone="warn" title="Administrators only">Ask an administrator for backups, webhooks or the audit log.</Notice> : <>
        {tab === "backups" && <Backups />}
        {tab === "tokens" && <Tokens isAdmin={isAdmin} />}
        {tab === "webhooks" && <Webhooks />}
        {tab === "audit" && <Audit />}
      </>}
    </div>
  );
}

// ------------------------------------------------------------------ backups
interface BackupRow { name: string; size_bytes: number; created_at: string; kind?: string }
interface BackupList { backups: BackupRow[]; pending_restore: Record<string, unknown> | null; backups_dir: string }

function Backups() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["backups"], queryFn: () => api<BackupList>("/api/system/backups") });
  const [label, setLabel] = useState("");
  const [mlflow, setMlflow] = useState(true);
  const [backbones, setBackbones] = useState(false);
  const [message, setMessage] = useState<{ tone: "info" | "warn" | "error"; title: string; body?: string } | null>(null);
  const refresh = () => qc.invalidateQueries({ queryKey: ["backups"] });
  const create = useMutation({
    mutationFn: () => api<{ name: string; files: number }>("/api/system/backups", { method: "POST", json: { label: label || undefined, include_mlflow: mlflow, include_backbones: backbones } }),
    onSuccess: (r) => { refresh(); setLabel(""); setMessage({ tone: "info", title: `Backup ${r.name} created and verified.` }); },
  });
  const verify = useMutation({
    mutationFn: (name: string) => api<{ files: number; manifest?: { app_version?: string } }>(`/api/system/backups/${encodeURIComponent(name)}/verify`, { method: "POST" }),
    onSuccess: (r, name) => setMessage({ tone: "info", title: `${name}: every file matches its SHA-256 (${r.files ?? "?"} files).` }),
    onError: (e, name) => setMessage({ tone: "error", title: `${name} failed verification`, body: (e as Error).message }),
  });
  const restore = useMutation({
    mutationFn: (name: string) => api<{ next_step: string }>(`/api/system/backups/${encodeURIComponent(name)}/restore`, { method: "POST" }),
    onSuccess: (r) => { refresh(); setMessage({ tone: "warn", title: "Restore staged", body: r.next_step }); },
  });
  const upload = useMutation({
    mutationFn: (f: File) => { const fd = new FormData(); fd.append("file", f); return api<{ next_step: string }>("/api/system/backups/restore-upload", { method: "POST", body: fd }); },
    onSuccess: (r) => { refresh(); setMessage({ tone: "warn", title: "Restore staged", body: r.next_step }); },
  });
  const cancel = useMutation({ mutationFn: () => api("/api/system/backups/pending-restore/cancel", { method: "DELETE" }), onSuccess: refresh });
  const remove = useMutation({ mutationFn: (name: string) => api(`/api/system/backups/${encodeURIComponent(name)}`, { method: "DELETE" }), onSuccess: refresh });
  const err = create.error ?? restore.error ?? upload.error ?? cancel.error ?? remove.error;

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      <div className="space-y-4">
        <Panel title="Create a backup">
          <div className="space-y-3">
            <p className="text-[12.5px] text-muted">A ZIP of the database (consistent snapshot), uploaded documents, Arena datasets and models, and settings, with a SHA-256 for every file. Ollama models are not included – download them again from Setup.</p>
            <Field label="Label (optional)"><Input value={label} maxLength={40} onChange={(e) => setLabel(e.target.value)} placeholder="before-upgrade" /></Field>
            <Toggle checked={mlflow} onChange={setMlflow} label="Include MLflow experiments" />
            <Toggle checked={backbones} onChange={setBackbones} label="Include image backbones" description="Large; they can be downloaded again." />
            <Button variant="primary" onClick={() => create.mutate()} busy={create.isPending}><Archive className="size-4" />Create backup</Button>
          </div>
        </Panel>
        <Panel title="Restore from a file">
          <p className="mb-3 text-[12.5px] text-muted">The backup is verified and staged. It replaces the current data at the next start; the current data is first moved to <code className="text-ink">backups/replaced-&lt;time&gt;</code>.</p>
          <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-raised px-3 text-[13px] hover:border-line-strong">
            <Upload className="size-4" />{upload.isPending ? "Uploading…" : "Choose backup ZIP"}
            <input type="file" accept=".zip" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f && confirm(`Stage ${f.name} for restore? All current data will be replaced at the next start.`)) upload.mutate(f); }} />
          </label>
        </Panel>
      </div>
      <div className="space-y-4 min-w-0">
        {message && <Notice tone={message.tone} title={message.title}>{message.body}</Notice>}
        {err && <ErrorNotice error={err} />}
        {list.data?.pending_restore && (
          <Notice tone="warn" title="A restore is waiting for the next start">
            <div className="flex flex-wrap items-center gap-3"><span>Restart Local AI Lab to apply it.</span><Button onClick={() => cancel.mutate()} busy={cancel.isPending}>Cancel restore</Button></div>
          </Notice>
        )}
        <Panel title="Backups" bodyClass="p-0">
          {list.isLoading ? <div className="p-4"><Spinner /></div> : list.isError ? <div className="p-4"><ErrorNotice error={list.error} /></div> : !list.data!.backups.length ? <div className="p-4"><Empty title="No backups yet">Create one before upgrading or moving to another computer.</Empty></div> : (
            <Table>
              <thead><tr><th>File</th><th>Size</th><th>Created</th><th /></tr></thead>
              <tbody>{list.data!.backups.map((b) => {
                const zip = b.name.endsWith(".zip");
                return (
                  <tr key={b.name}>
                    <td className="max-w-[340px]"><div className="truncate num" title={b.name}>{b.name}</div>{b.kind && <div className="text-[12px] text-faint">{b.kind}</div>}</td>
                    <td className="num whitespace-nowrap">{bytes(b.size_bytes)}</td>
                    <td className="whitespace-nowrap text-muted">{when(b.created_at)}</td>
                    <td className="whitespace-nowrap text-right">
                      {zip && <>
                        <Button variant="ghost" title="Check every file's SHA-256" onClick={() => verify.mutate(b.name)} busy={verify.isPending && verify.variables === b.name}><ShieldCheck className="size-4" /></Button>
                        <Button variant="ghost" title="Download" onClick={() => download(`/api/system/backups/${encodeURIComponent(b.name)}/download`)}><Download className="size-4" /></Button>
                        <Button variant="ghost" title="Restore at next start" onClick={() => confirm(`Restore ${b.name}? All current data will be replaced at the next start (it is kept in backups/replaced-<time>).`) && restore.mutate(b.name)}><RotateCcw className="size-4" /></Button>
                      </>}
                      <Button variant="ghost" title="Delete" onClick={() => confirm(`Delete ${b.name}?`) && remove.mutate(b.name)}><Trash2 className="size-4" /></Button>
                    </td>
                  </tr>
                );
              })}</tbody>
            </Table>
          )}
          {list.data && <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">Stored in <code className="text-ink">{list.data.backups_dir}</code>. Copy backups to another drive – a backup on the same disk does not protect against disk failure. Backups contain your documents: keep them as safe as the originals.</div>}
        </Panel>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ tokens
interface TokenRow { id: number; name: string; prefix: string; scopes: string[]; created_at: string; expires_at: string; last_used_at: string | null; revoked_at: string | null; active: boolean }

function Tokens({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["tokens"], queryFn: () => api<{ tokens: TokenRow[]; scopes: Record<string, string> }>("/api/auth/tokens") });
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<string[]>(["read"]);
  const [days, setDays] = useState(90);
  const [created, setCreated] = useState<string | null>(null);
  const create = useMutation({
    mutationFn: () => api<{ token: string }>("/api/auth/tokens", { method: "POST", json: { name, scopes, expires_days: days } }),
    onSuccess: (r) => { setCreated(r.token); setName(""); qc.invalidateQueries({ queryKey: ["tokens"] }); },
  });
  const revoke = useMutation({ mutationFn: (id: number) => api(`/api/auth/tokens/${id}`, { method: "DELETE" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["tokens"] }) });
  const revokeAll = useMutation({ mutationFn: () => api<{ revoked: number }>("/api/auth/sessions/revoke-all", { method: "POST" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["tokens"] }) });
  const all = list.data?.scopes ?? {};

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
      <Panel title="New token">
        <div className="space-y-3">
          <p className="text-[12.5px] text-muted">For scripts, the CLI, CI jobs and n8n. Send it as <code className="text-ink">Authorization: Bearer lal_…</code>. Give each tool its own token with the fewest scopes it needs.</p>
          <Field label="Name"><Input value={name} maxLength={100} onChange={(e) => setName(e.target.value)} placeholder="n8n on the office server" /></Field>
          <fieldset className="space-y-1.5">
            <legend className="mb-1 text-[12.5px] text-muted">Scopes</legend>
            {Object.entries(all).map(([s, desc]) => (
              <label key={s} className={cx("flex items-start gap-2 text-[13px]", s === "admin" && !isAdmin && "opacity-50")}>
                <input type="checkbox" className="mt-0.5 accent-amber" disabled={s === "admin" && !isAdmin} checked={scopes.includes(s)} onChange={(e) => setScopes((x) => (e.target.checked ? [...x, s] : x.filter((y) => y !== s)))} />
                <span><span className="num text-ink">{s}</span> <span className="block text-[12px] text-faint">{desc}</span></span>
              </label>
            ))}
          </fieldset>
          <Field label="Expires after"><Select value={days} onChange={(e) => setDays(Number(e.target.value))}>{[7, 30, 90, 180, 366].map((d) => <option key={d} value={d}>{d} days</option>)}</Select></Field>
          <Button variant="primary" onClick={() => create.mutate()} busy={create.isPending} disabled={!name.trim() || !scopes.length}><KeyRound className="size-4" />Create token</Button>
          {create.isError && <ErrorNotice error={create.error} />}
          {created && (
            <Notice tone="warn" title="Copy the token now – it is shown only once">
              <div className="mt-1 flex items-center gap-2"><code className="min-w-0 flex-1 break-all rounded bg-bg px-2 py-1 text-[12px] text-ink">{created}</code>
                <Button variant="ghost" aria-label="Copy token" onClick={() => navigator.clipboard?.writeText(created)}><Copy className="size-4" /></Button></div>
            </Notice>
          )}
        </div>
      </Panel>
      <div className="space-y-4 min-w-0">
        <Panel title="Your tokens" bodyClass="p-0" action={<Button variant="danger" onClick={() => confirm("Revoke every token and sign out every other session of this account?") && revokeAll.mutate()} busy={revokeAll.isPending}>Revoke all</Button>}>
          {list.isLoading ? <div className="p-4"><Spinner /></div> : !list.data?.tokens.length ? <div className="p-4 text-[13px] text-faint">No API tokens.</div> : (
            <Table>
              <thead><tr><th>Name</th><th>Scopes</th><th>Last used</th><th>Expires</th><th /></tr></thead>
              <tbody>{list.data.tokens.map((t) => (
                <tr key={t.id} className={cx(!t.active && "opacity-60")}>
                  <td><div>{t.name}</div><div className="num text-[12px] text-faint">{t.prefix}…</div></td>
                  <td><div className="flex flex-wrap gap-1">{t.scopes.map((s) => <Badge key={s} tone={s === "admin" ? "coral" : "neutral"}>{s}</Badge>)}</div></td>
                  <td className="text-muted whitespace-nowrap">{t.last_used_at ? when(t.last_used_at) : "never"}</td>
                  <td className="text-muted whitespace-nowrap">{t.revoked_at ? <Badge tone="coral">revoked</Badge> : !t.active ? <Badge>expired</Badge> : when(t.expires_at)}</td>
                  <td className="text-right">{t.active && <Button variant="ghost" onClick={() => revoke.mutate(t.id)} busy={revoke.isPending && revoke.variables === t.id}>Revoke</Button>}</td>
                </tr>
              ))}</tbody>
            </Table>
          )}
        </Panel>
        {revokeAll.data && <Notice title={`${revokeAll.data.revoked} token(s) and session(s) revoked.`} />}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ webhooks
interface HookRow { id: number; name: string; url: string; description: string; allow_agent: boolean; secret?: string }

function Webhooks() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["webhooks"], queryFn: () => api<{ webhooks: HookRow[] }>("/api/system/webhooks") });
  const [form, setForm] = useState({ name: "", url: "", description: "", allow_agent: false });
  const [secret, setSecret] = useState<string | null>(null);
  const create = useMutation({
    mutationFn: () => api<HookRow>("/api/system/webhooks", { method: "POST", json: form }),
    onSuccess: (h) => { setSecret(h.secret ?? null); setForm({ name: "", url: "", description: "", allow_agent: false }); qc.invalidateQueries({ queryKey: ["webhooks"] }); },
  });
  const test = useMutation({ mutationFn: (name: string) => api<{ status?: number; ok?: boolean }>(`/api/system/webhooks/${encodeURIComponent(name)}/test`, { method: "POST" }) });
  const remove = useMutation({ mutationFn: (name: string) => api(`/api/system/webhooks/${encodeURIComponent(name)}`, { method: "DELETE" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["webhooks"] }) });
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
      <Panel title="Add a webhook">
        <div className="space-y-3">
          <p className="text-[12.5px] text-muted">Local AI Lab POSTs JSON to this URL when a scheduled run fails a gate, when you send a test, and – if allowed – when the Work Agent calls it. Each request is signed with HMAC-SHA256 in <code className="text-ink">X-LocalAILab-Signature</code> – verify it in n8n. In Offline-Only mode only this computer (e.g. n8n on localhost) and hosts in <code className="text-ink">OFFLINE_ALLOWED_HOSTS</code> can be called.</p>
          <Field label="Name"><Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="n8n-alerts" /></Field>
          <Field label="URL"><Input value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} placeholder="http://localhost:5678/webhook/…" /></Field>
          <Field label="Description"><Input value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></Field>
          <Toggle checked={form.allow_agent} onChange={(v) => setForm({ ...form, allow_agent: v })} label="The Work Agent may call it" description="Still subject to the agent's allow / ask / deny policy." />
          <Button variant="primary" onClick={() => create.mutate()} busy={create.isPending} disabled={!form.name || !form.url}><WebhookIcon className="size-4" />Add webhook</Button>
          {create.isError && <ErrorNotice error={create.error} />}
          {secret && <Notice tone="warn" title="Signing secret – copy it now"><code className="break-all text-[12px] text-ink">{secret}</code></Notice>}
        </div>
      </Panel>
      <Panel title="Webhooks" bodyClass="p-0">
        {!list.data?.webhooks.length ? <div className="p-4 text-[13px] text-faint">No webhooks.</div> : (
          <Table>
            <thead><tr><th>Name</th><th>URL</th><th>Agent</th><th /></tr></thead>
            <tbody>{list.data.webhooks.map((h) => (
              <tr key={h.id}>
                <td><div>{h.name}</div><div className="text-[12px] text-faint">{h.description}</div></td>
                <td className="max-w-[300px] truncate num text-[12px]" title={h.url}>{h.url}</td>
                <td>{h.allow_agent ? <Badge tone="amber">allowed</Badge> : <span className="text-faint">–</span>}</td>
                <td className="whitespace-nowrap text-right">
                  <Button variant="ghost" onClick={() => test.mutate(h.name)} busy={test.isPending && test.variables === h.name}>Send test</Button>
                  <Button variant="ghost" aria-label={`Delete ${h.name}`} onClick={() => confirm(`Delete webhook ${h.name}?`) && remove.mutate(h.name)}><Trash2 className="size-4" /></Button>
                </td>
              </tr>
            ))}</tbody>
          </Table>
        )}
        {test.data && <div className="border-t border-line p-3"><Notice tone={test.data.ok === false ? "warn" : "info"} title={`Test sent${test.data.status ? ` – HTTP ${test.data.status}` : ""}`} /></div>}
        {test.isError && <div className="border-t border-line p-3"><ErrorNotice error={test.error} /></div>}
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------------ audit
interface AuditEvent { id: number; at: string; user: string | null; action: string; target: string | null; outcome: string; ip: string | null; details: Record<string, unknown> | null }

function Audit() {
  const [action, setAction] = useState("");
  const q = useQuery({ queryKey: ["audit", action], queryFn: () => api<{ events: AuditEvent[] }>(`/api/system/audit?limit=500${action ? `&action=${encodeURIComponent(action)}` : ""}`) });
  const prefixes = ["", "auth.", "token.", "settings.", "backup.", "model.", "webhook.", "schedule.", "work_agent."];
  return (
    <Panel title="Audit log" bodyClass="p-0" action={
      <Select className="w-44" value={action} onChange={(e) => setAction(e.target.value)} aria-label="Filter by action">
        {prefixes.map((p) => <option key={p} value={p}>{p ? p.replace(/\.$/, "") : "All actions"}</option>)}
      </Select>
    }>
      {q.isLoading ? <div className="p-4"><Spinner /></div> : q.isError ? <div className="p-4"><ErrorNotice error={q.error} /></div> : !q.data!.events.length ? <div className="p-4 text-[13px] text-faint">No events.</div> : (
        <Table>
          <thead><tr><th>Time</th><th>User</th><th>Action</th><th>Target</th><th>Outcome</th><th>IP</th></tr></thead>
          <tbody>{q.data!.events.map((e) => (
            <tr key={e.id}>
              <td className="whitespace-nowrap text-muted">{when(e.at)}</td>
              <td>{e.user ?? "–"}</td>
              <td className="num text-[12.5px]" title={e.details ? JSON.stringify(e.details) : undefined}>{e.action}</td>
              <td className="max-w-[260px] truncate" title={e.target ?? ""}>{e.target ?? "–"}</td>
              <td><Badge tone={e.outcome === "ok" ? "teal" : e.outcome === "denied" || e.outcome === "failed" ? "coral" : "neutral"}>{e.outcome}</Badge></td>
              <td className="num text-[12px] text-muted">{e.ip ?? "–"}</td>
            </tr>
          ))}</tbody>
        </Table>
      )}
      <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">Security-relevant actions only – never document contents or prompts. The newest 50 000 events are kept.</div>
    </Panel>
  );
}
