import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, CircleDashed, Download, ExternalLink, RefreshCw, Trash2, XCircle, AlertTriangle } from "lucide-react";
import { useState } from "react";
import { api, streamNdjson } from "../api/client";
import { Badge, Button, cx, ErrorNotice, Notice, PageHeader, Panel, Select, Spinner, Table } from "../components/ui";

interface Check { id: string; label: string; status: "ok" | "warn" | "fail" | "skip"; detail: string; fix: string | null }
interface Fit { verdict: "fits" | "tight" | "too_big" | "unknown"; where?: string; estimated_gb?: number; kv_cache_gb?: number; kv_source?: string }
interface CatalogRow { name: string; family: string; params_b?: number; download_gb: number; use: string[]; license: string; commercial: string; license_url?: string; installed: boolean; fit: Fit; note?: string }
interface Catalog { note: string; catalog: CatalogRow[]; other_installed: { name: string; size_gb: number; license: string; fit: Fit }[]; num_ctx: number; system: { ram_total_gb: number; ram_available_gb: number; gpus: { name: string; memory_total_mb: number }[]; disk_free_gb: number } }
type PullEvent = { type: "progress"; status?: string; percent?: number | null } | { type: "error"; message: string; hint?: string } | { type: "done"; model: string };

const ICON = { ok: CheckCircle2, warn: AlertTriangle, fail: XCircle, skip: CircleDashed } as const;
const COLOR = { ok: "text-teal", warn: "text-amber", fail: "text-coral", skip: "text-faint" } as const;
const FIT_TONE = { fits: "teal", tight: "amber", too_big: "coral", unknown: "neutral" } as const;
const FIT_LABEL = { fits: "fits", tight: "tight", too_big: "too big", unknown: "unknown" } as const;

export default function Setup({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const check = useQuery({ queryKey: ["setup-check"], queryFn: () => api<{ status: string; checks: Check[] }>("/api/setup/check") });
  const [numCtx, setNumCtx] = useState(4096);
  const cat = useQuery({ queryKey: ["setup-models", numCtx], queryFn: () => api<Catalog>(`/api/setup/models?num_ctx=${numCtx}`) });
  const [pulling, setPulling] = useState<Record<string, { percent: number | null; status: string; error?: string; hint?: string }>>({});

  const pull = async (name: string) => {
    setPulling((p) => ({ ...p, [name]: { percent: null, status: "starting" } }));
    try {
      await streamNdjson<PullEvent>("/api/setup/models/pull", { name }, (e) => {
        if (e.type === "progress") setPulling((p) => ({ ...p, [name]: { percent: e.percent ?? p[name]?.percent ?? null, status: e.status ?? "" } }));
        if (e.type === "error") setPulling((p) => ({ ...p, [name]: { percent: null, status: "failed", error: e.message, hint: e.hint } }));
        if (e.type === "done") setPulling((p) => ({ ...p, [name]: { percent: 100, status: "installed" } }));
      });
    } catch (err) {
      const e = err as { message?: string; hint?: string };
      setPulling((p) => ({ ...p, [name]: { percent: null, status: "failed", error: e.message ?? String(err), hint: e.hint } }));
    }
    qc.invalidateQueries({ queryKey: ["setup-models"] });
    qc.invalidateQueries({ queryKey: ["setup-check"] });
    qc.invalidateQueries({ queryKey: ["models"] });
  };
  const remove = useMutation({
    mutationFn: (name: string) => api(`/api/setup/models/${encodeURIComponent(name)}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["setup-models"] }); qc.invalidateQueries({ queryKey: ["models"] }); },
  });

  const sys = cat.data?.system;
  return (
    <div className="max-w-[1320px]">
      <PageHeader title="Setup" description="Check that everything this app needs is in place, and install local models that fit this computer – with their licences shown before you download." />
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.6fr)]">
        <Panel title="System check" action={<Button variant="ghost" onClick={() => check.refetch()} busy={check.isFetching} aria-label="Run the checks again"><RefreshCw className="size-4" /></Button>}>
          {check.isLoading ? <Spinner /> : check.isError ? <ErrorNotice error={check.error} /> : (
            <ul className="space-y-2.5">
              {check.data!.checks.map((c) => {
                const Icon = ICON[c.status];
                return (
                  <li key={c.id} className="flex gap-2.5">
                    <Icon className={cx("mt-0.5 size-4 shrink-0", COLOR[c.status])} aria-label={c.status} />
                    <div className="min-w-0 text-[13px]">
                      <div className="text-ink">{c.label}</div>
                      <div className="text-muted break-words">{c.detail}</div>
                      {c.fix && <div className="mt-0.5 text-[12.5px] text-amber">→ {c.fix}</div>}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>

        <Panel title="Models" bodyClass="p-0" action={
          <label className="flex items-center gap-2 text-[12.5px] text-muted">Context
            <Select className="w-28" value={numCtx} onChange={(e) => setNumCtx(Number(e.target.value))} aria-label="Context length used for the memory estimate">
              {[2048, 4096, 8192, 16384, 32768].map((n) => <option key={n} value={n}>{n.toLocaleString()}</option>)}
            </Select>
          </label>
        }>
          {cat.isLoading ? <div className="p-4"><Spinner /></div> : cat.isError ? <div className="p-4"><ErrorNotice error={cat.error} /></div> : (
            <>
              {sys && <div className="border-b border-line px-4 py-2.5 text-[12.5px] text-muted">This computer: <span className="text-ink">{sys.ram_total_gb} GB RAM</span> ({sys.ram_available_gb} GB free), {sys.gpus.length ? sys.gpus.map((g) => `${g.name} ${Math.round(g.memory_total_mb / 1024)} GB`).join(", ") : "no NVIDIA GPU"}, <span className="text-ink">{sys.disk_free_gb} GB</span> disk free.</div>}
              <Table>
                <thead><tr><th>Model</th><th>Download</th><th title="Estimated memory need: weights + KV cache for the chosen context + runtime overhead">Fit</th><th>Licence</th><th /></tr></thead>
                <tbody>
                  {cat.data!.catalog.map((m) => {
                    const p = pulling[m.name];
                    const restricted = m.commercial.startsWith("no");
                    return (
                      <tr key={m.name}>
                        <td className="max-w-[220px]"><div className="num">{m.name}</div><div className="text-[12px] text-faint">{m.family}{m.params_b ? ` · ${m.params_b}B` : ""} · {m.use.join(", ")}</div></td>
                        <td className="num whitespace-nowrap">{m.download_gb} GB</td>
                        <td><Badge tone={FIT_TONE[m.fit.verdict]} title={m.fit.estimated_gb ? `≈ ${m.fit.estimated_gb} GB needed (KV cache ${m.fit.kv_cache_gb} GB, ${m.fit.kv_source}); ${m.fit.where}` : undefined}>{FIT_LABEL[m.fit.verdict]}{m.fit.estimated_gb ? ` · ${m.fit.estimated_gb} GB` : ""}</Badge></td>
                        <td className="max-w-[240px]">
                          <div className="flex items-center gap-1"><span className="truncate" title={m.license}>{m.license}</span>{m.license_url && <a href={m.license_url} target="_blank" rel="noreferrer noopener" className="text-muted hover:text-amber" aria-label={`Open the ${m.name} model card`}><ExternalLink className="size-3.5" /></a>}</div>
                          <div className={cx("text-[12px]", restricted ? "text-coral" : "text-faint")} title={m.commercial}>commercial use: {m.commercial.length > 40 ? m.commercial.slice(0, 40) + "…" : m.commercial}</div>
                        </td>
                        <td className="whitespace-nowrap text-right">
                          {m.installed ? (
                            <span className="inline-flex items-center gap-2"><Badge tone="teal">installed</Badge>{isAdmin && <Button variant="ghost" aria-label={`Remove ${m.name}`} onClick={() => confirm(`Remove ${m.name} from Ollama? You can download it again later.`) && remove.mutate(m.name)}><Trash2 className="size-4" /></Button>}</span>
                          ) : p && p.status !== "failed" && p.status !== "installed" ? (
                            <div className="w-36"><div className="text-[12px] text-muted truncate">{p.status}{p.percent != null && ` ${p.percent}%`}</div><div className="mt-1 h-1.5 rounded-full bg-raised overflow-hidden"><div className="h-full bg-amber" style={{ width: `${p.percent ?? 2}%` }} /></div></div>
                          ) : isAdmin ? (
                            <Button onClick={() => (!restricted || confirm(`${m.name} is licensed for research / non-commercial use only (${m.license}). Download anyway?`)) && pull(m.name)}><Download className="size-4" />Download</Button>
                          ) : <span className="text-[12px] text-faint">admin only</span>}
                          {p?.error && <div className="mt-1 max-w-[240px] whitespace-normal text-left text-[12px] text-coral">{p.error}{p.hint && <span className="block text-muted">{p.hint}</span>}</div>}
                        </td>
                      </tr>
                    );
                  })}
                  {cat.data!.other_installed.map((m) => (
                    <tr key={m.name}>
                      <td><div className="num">{m.name}</div><div className="text-[12px] text-faint">installed, not in the catalogue</div></td>
                      <td className="num">{m.size_gb} GB</td>
                      <td><Badge tone={FIT_TONE[m.fit.verdict]}>{FIT_LABEL[m.fit.verdict]}{m.fit.estimated_gb ? ` · ${m.fit.estimated_gb} GB` : ""}</Badge></td>
                      <td className="max-w-[240px] truncate text-muted" title={m.license}>{m.license}</td>
                      <td className="text-right">{isAdmin && <Button variant="ghost" aria-label={`Remove ${m.name}`} onClick={() => confirm(`Remove ${m.name} from Ollama?`) && remove.mutate(m.name)}><Trash2 className="size-4" /></Button>}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
              <div className="border-t border-line px-4 py-2.5 text-[12px] text-faint">{cat.data!.note} Downloads are done by Ollama and need the internet: Offline-Only mode blocks them. An interrupted download resumes when you start it again.</div>
            </>
          )}
          {remove.isError && <div className="p-4"><ErrorNotice error={remove.error} /></div>}
        </Panel>
      </div>
      <div className="mt-6"><Notice title="Models are licensed separately">Local AI Lab does not include model weights. Each model has its own licence (Apache-2.0, MIT, Llama Community License, Gemma Terms, research-only…). Read it before using a model commercially or shipping it to others – see docs/LEGAL_AND_LICENSING.md.</Notice></div>
    </div>
  );
}
