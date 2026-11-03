import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { BenchmarkRun, DocumentInfo } from "../api/types";
import { Badge, Empty, Meter, Notice, PageHeader, Panel, Readout, StatusBadge, Table } from "../components/ui";
import { bytes, duration, num, pct, when } from "../lib/format";
import { useModels } from "../lib/queries";

interface Resources {
  os: string;
  cpu_count_logical: number;
  cpu_percent: number;
  ram_total_mb: number;
  ram_used_mb: number;
  disk_total_gb: number;
  disk_free_gb: number;
  gpus: { name: string; memory_total_mb: number; memory_used_mb: number }[];
  runtime_rss_mb: number | null;
  loaded_models: { name: string; size_mb: number; size_vram_mb: number; expires_at: string }[];
}

interface Ready {
  status: string;
  checks: Record<string, { ok: boolean; [k: string]: unknown }>;
}

const CHECK_LABELS: Record<string, string> = {
  database: "Database", ollama: "Model runtime (Ollama)", embeddings: "Embeddings", storage: "Storage",
  mlflow: "Experiment tracking (MLflow)", privacy: "Privacy guard", benchmark_worker: "Benchmark worker",
};

function checkDetail(id: string, c: Ready["checks"][string]): string {
  switch (id) {
    case "database": return c.ok ? `${c.latency_ms} ms` : String(c.error);
    case "ollama": return c.ok ? `v${c.version}, ${c.latency_ms} ms` : String(c.hint ?? c.error);
    case "embeddings": return `${c.backend} (${c.model})${c.note ? ` – ${c.note}` : ""}`;
    case "storage": return `${c.free_gb} GB free`;
    case "mlflow": return c.enabled ? (c.installed ? "Local SQLite store" : "Enabled but not installed") : "Disabled";
    case "privacy": return `${c.offline_only ? "Offline-Only on" : "Offline-Only off"}, socket guard ${c.socket_guard_installed ? "active" : "off"}`;
    case "benchmark_worker": return c.busy ? "Running a benchmark" : "Idle";
    default: return "";
  }
}

export default function Overview() {
  const models = useModels();
  const res = useQuery({ queryKey: ["resources"], queryFn: () => api<Resources>("/api/system/resources"), refetchInterval: 5000 });
  const ready = useQuery({ queryKey: ["ready"], queryFn: () => api<Ready>("/api/health/ready"), refetchInterval: 15000 });
  const rec = useQuery({ queryKey: ["recommendations"], queryFn: () => api<{ chat_models: string[]; note: string; commands: string[]; memory_budget_gb: number }>("/api/system/recommendations") });
  const docs = useQuery({ queryKey: ["documents"], queryFn: () => api<{ documents: DocumentInfo[] }>("/api/documents") });
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => api<{ runs: BenchmarkRun[] }>("/api/benchmarks/runs?limit=6"), refetchInterval: 10000 });

  const chatModels = models.data?.models.filter((m) => !m.is_embedding) ?? [];
  const local = models.data?.status.local;
  const r = res.data;

  return (
    <div className="max-w-[1200px]">
      <PageHeader title="Overview" description="Runtime status, hardware headroom and recent activity of your local AI workbench." />

      {local && !local.reachable && (
        <div className="mb-5">
          <Notice tone="warn" title={`Ollama is not reachable at ${local.base_url}`}>
            {local.hint} Then download a model, for example <code>ollama pull llama3.2:1b</code>.
          </Notice>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-5 rounded-lg border border-line bg-panel px-5 py-4 mb-6">
        <Readout label="Chat models installed" value={chatModels.filter((m) => m.provider === "local").length}
          sub={chatModels.some((m) => m.provider === "cloud") ? `+ ${chatModels.filter((m) => m.provider === "cloud").length} cloud` : "all local"} />
        <Readout label="Documents indexed" value={docs.data?.documents.filter((d) => d.status === "ready").length ?? "–"}
          sub={`${num(docs.data?.documents.reduce((a, d) => a + d.chunk_count, 0) ?? 0, 0)} chunks`} />
        <Readout label="Benchmark runs" value={runs.data?.runs.length ?? "–"} sub="most recent shown below" />
        <Readout label="Network policy" value={models.data?.offline_only === false ? "Cloud allowed" : "Offline only"}
          tone={models.data?.offline_only === false ? "amber" : "teal"} sub="change in Settings" />
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[1.1fr_1fr]">
        <Panel title="This computer" action={r && <span className="text-[12px] text-faint">{r.os}, {r.cpu_count_logical} logical CPUs</span>}>
          {!r ? <div className="text-muted">Reading hardware…</div> : (
            <div className="space-y-4">
              <Meter label="CPU load" value={Math.round(r.cpu_percent)} max={100} unit="%" />
              <Meter label="Memory in use" value={r.ram_used_mb} max={r.ram_total_mb} unit=" MB"
                detail={r.runtime_rss_mb ? `Ollama processes use ${num(r.runtime_rss_mb, 0)} MB` : undefined} />
              <Meter label="Disk used" value={Math.round((r.disk_total_gb - r.disk_free_gb) * 10) / 10} max={r.disk_total_gb} unit=" GB"
                detail={`${r.disk_free_gb} GB free for models and data`} />
              <div className="text-[13px]">
                <div className="text-muted mb-1">GPU</div>
                {r.gpus.length ? r.gpus.map((g) => <div key={g.name}>{g.name} <span className="text-faint">({num(g.memory_total_mb / 1024)} GB)</span></div>)
                  : <div className="text-faint">No NVIDIA GPU detected – models run on the CPU.</div>}
              </div>
              <div className="text-[13px]">
                <div className="text-muted mb-1">Loaded in memory now</div>
                {r.loaded_models.length ? (
                  <ul className="space-y-1">
                    {r.loaded_models.map((m) => (
                      <li key={m.name} className="flex justify-between gap-3"><span>{m.name}</span><span className="num text-muted">{num(m.size_mb, 0)} MB{m.size_vram_mb ? `, ${num(m.size_vram_mb, 0)} MB on GPU` : ""}</span></li>
                    ))}
                  </ul>
                ) : <div className="text-faint">No model loaded. Ollama loads a model on first use and unloads it after 5 idle minutes.</div>}
              </div>
            </div>
          )}
        </Panel>

        <div className="space-y-6">
          <Panel title="Health checks" action={ready.data && <StatusBadge status={ready.data.status === "ok" ? "ok" : "partial"} />}>
            <ul className="space-y-2.5">
              {ready.data && Object.entries(ready.data.checks).map(([id, c]) => (
                <li key={id} className="flex gap-2.5 text-[13px]">
                  {c.ok ? <CheckCircle2 className="size-4 mt-0.5 text-teal shrink-0" aria-label="OK" /> : <CircleAlert className="size-4 mt-0.5 text-coral shrink-0" aria-label="Problem" />}
                  <div className="min-w-0"><div>{CHECK_LABELS[id] ?? id}</div><div className="text-faint break-words">{checkDetail(id, c)}</div></div>
                </li>
              ))}
            </ul>
          </Panel>
          {rec.data && (
            <Panel title="Models that fit this hardware">
              <p className="text-[13px] text-muted">{rec.data.note} Memory budget for a model: about {rec.data.memory_budget_gb} GB.</p>
              <div className="mt-2 flex flex-wrap gap-1.5">{rec.data.chat_models.map((m) => <Badge key={m}>{m}</Badge>)}</div>
              <pre className="mt-3 rounded-md bg-bg border border-line px-3 py-2 text-[12.5px] overflow-x-auto">{rec.data.commands.join("\n")}</pre>
            </Panel>
          )}
        </div>
      </div>

      <Panel className="mt-6" title="Recent benchmark runs" action={<Link to="/benchmarks" className="text-[13px] text-amber hover:underline">Open benchmarks</Link>} bodyClass="p-0">
        {runs.data?.runs.length ? (
          <Table>
            <thead><tr><th>Run</th><th>Status</th><th>Best model</th><th>Quality</th><th>Tokens/s</th><th>Duration</th><th>Started</th></tr></thead>
            <tbody>
              {runs.data.runs.map((run) => {
                const best = run.summary[0];
                return (
                  <tr key={run.id}>
                    <td><Link className="hover:text-amber" to={`/benchmarks/runs/${run.id}`}>{run.name}</Link></td>
                    <td><StatusBadge status={run.status} /></td>
                    <td>{best?.model ?? "–"}</td>
                    <td className="num">{pct(best?.quality_score)}</td>
                    <td className="num">{num(best?.tokens_per_s_mean)}</td>
                    <td className="num">{duration(run.duration_s)}</td>
                    <td className="text-muted">{when(run.started_at ?? run.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        ) : (
          <div className="p-4"><Empty title="No benchmark runs yet" action={<Link to="/benchmarks" className="text-amber text-[13px] hover:underline">Run your first benchmark</Link>}>Compare installed models on the same test suite to see quality, speed and memory side by side.</Empty></div>
        )}
      </Panel>
      <p className="mt-3 text-[12px] text-faint">Model sizes on disk: {chatModels.filter(m => m.provider === "local").map((m) => `${m.name} ${bytes(m.size_bytes)}`).join(", ") || "none"}.</p>
    </div>
  );
}
