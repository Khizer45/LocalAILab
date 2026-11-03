import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Search, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { api, ApiError, streamNdjson, type Metrics, type Source } from "../api/client";
import type { DocumentInfo } from "../api/types";
import { Markdown } from "../components/Markdown";
import { ModelSelect } from "../components/ModelSelect";
import { RetrievalEval } from "../components/RetrievalEval";
import { MetricsLine, SourceList } from "../components/Sources";
import { Badge, Button, cx, Empty, ErrorNotice, Input, Notice, PageHeader, Panel, StatusBadge, Table } from "../components/ui";
import { bytes, when } from "../lib/format";
import { useSettings } from "../lib/queries";

interface DocsResponse {
  documents: DocumentInfo[];
  embedder: { backend: string; model: string; key: string; note: string | null };
  max_upload_mb: number;
}

export default function Documents() {
  const qc = useQueryClient();
  const settings = useSettings();
  const docs = useQuery({
    queryKey: ["documents"],
    queryFn: () => api<DocsResponse>("/api/documents"),
    refetchInterval: (q) => (q.state.data?.documents.some((d) => d.status === "pending" || d.status === "processing") ? 1500 : false),
  });
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState<string[]>([]);
  const [uploadErrors, setUploadErrors] = useState<{ name: string; error: ApiError }[]>([]);
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<number[]>([]);

  const [model, setModel] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<{ text: string; sources: Source[]; warnings: string[]; metrics: Metrics | null; model: string } | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<unknown>(null);

  const upload = async (files: FileList | File[]) => {
    const list = Array.from(files);
    setUploadErrors([]);
    setUploading(list.map((f) => f.name));
    for (const f of list) {
      const form = new FormData();
      form.append("file", f);
      try {
        await api("/api/documents", { method: "POST", body: form });
      } catch (e) {
        setUploadErrors((errs) => [...errs, { name: f.name, error: e as ApiError }]);
      }
      setUploading((u) => u.filter((n) => n !== f.name));
      qc.invalidateQueries({ queryKey: ["documents"] });
    }
  };

  const remove = async (d: DocumentInfo) => {
    if (!confirm(`Delete “${d.filename}” and its index? This cannot be undone.`)) return;
    await api(`/api/documents/${d.id}`, { method: "DELETE" });
    setSelected((s) => s.filter((x) => x !== d.id));
    qc.invalidateQueries({ queryKey: ["documents"] });
  };

  const reindex = async (d: DocumentInfo) => {
    await api(`/api/documents/${d.id}/reindex`, { method: "POST" });
    qc.invalidateQueries({ queryKey: ["documents"] });
  };

  const ask = async () => {
    if (!question.trim()) return;
    setAsking(true);
    setAskError(null);
    const m = model || settings.data?.runtime.default_model || "";
    const state = { text: "", sources: [] as Source[], warnings: [] as string[], metrics: null as Metrics | null, model: m };
    setAnswer({ ...state });
    try {
      await streamNdjson("/api/documents/ask", { question, model: m, document_ids: selected.length ? selected : undefined }, (ev) => {
        if (ev.type === "token") state.text += ev.content;
        else if (ev.type === "sources") { state.sources = ev.sources; state.warnings = ev.warnings; }
        else if (ev.type === "done") state.metrics = ev.metrics;
        else if (ev.type === "error") setAskError(new ApiError(500, ev.code, ev.message, ev.hint));
        setAnswer({ ...state });
      });
    } catch (e) {
      setAskError(e);
      setAnswer(null);
    } finally {
      setAsking(false);
    }
  };

  const d = docs.data;
  const ready = d?.documents.filter((x) => x.status === "ready") ?? [];

  return (
    <div className="max-w-[1200px]">
      <PageHeader
        title="Documents"
        description="Index PDF, TXT and Markdown files on this computer, then ask questions and get answers with page-level citations."
        actions={<Button variant="primary" onClick={() => fileRef.current?.click()}><Upload className="size-4" />Add files</Button>}
      />
      <input ref={fileRef} type="file" multiple accept=".pdf,.txt,.md,.markdown" className="hidden" onChange={(e) => { if (e.target.files) upload(e.target.files); e.target.value = ""; }} />

      {d?.embedder.note && <div className="mb-4"><Notice tone="warn" title="Using the offline fallback embedder">{d.embedder.note}</Notice></div>}

      <div className="grid gap-6 xl:grid-cols-[1.25fr_1fr]">
        <div className="space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files); }}
            className={cx("rounded-lg border border-dashed px-4 py-5 text-center text-[13px] transition-colors",
              dragging ? "border-amber bg-amber/5" : "border-line text-muted")}
          >
            Drop files here to index them locally. Up to {d?.max_upload_mb ?? "–"} MB each. Text is extracted, split into chunks and embedded with{" "}
            <span className="text-ink">{d ? `${d.embedder.backend} (${d.embedder.model})` : "…"}</span>.
            {uploading.length > 0 && <div className="mt-2 text-amber">Uploading {uploading.join(", ")}…</div>}
          </div>
          {uploadErrors.map((u) => <ErrorNotice key={u.name} error={new ApiError(u.error.status, u.error.code, `${u.name}: ${u.error.message}`, u.error.hint)} />)}

          <Panel title="Library" bodyClass="p-0" action={selected.length > 0 && <button className="text-[12.5px] text-muted hover:text-ink" onClick={() => setSelected([])}>Search all documents</button>}>
            {docs.isError ? <div className="p-4"><ErrorNotice error={docs.error} /></div> : !d?.documents.length ? (
              <div className="p-4"><Empty title="No documents yet">Add a policy, report or manual. Files are copied into the local data folder and never uploaded anywhere else.</Empty></div>
            ) : (
              <Table>
                <thead><tr><th className="w-8"><span className="sr-only">Search in</span></th><th>Name</th><th>Status</th><th>Pages</th><th>Chunks</th><th>Size</th><th>Added</th><th /></tr></thead>
                <tbody>
                  {d.documents.map((doc) => (
                    <tr key={doc.id}>
                      <td>
                        <input type="checkbox" aria-label={`Search only selected documents: ${doc.filename}`} disabled={doc.status !== "ready"}
                          checked={selected.includes(doc.id)} onChange={(e) => setSelected((s) => e.target.checked ? [...s, doc.id] : s.filter((x) => x !== doc.id))}
                          className="accent-amber" />
                      </td>
                      <td className="max-w-[260px]">
                        <div className="truncate" title={doc.filename}>{doc.filename}</div>
                        {doc.error && <div className="text-[12px] text-coral">{doc.error}</div>}
                        <div className="flex flex-wrap gap-1 mt-0.5">
                          {doc.needs_reindex && <Badge tone="amber" title="Indexed with a different embedding model than the current one">re-index needed</Badge>}
                          {doc.flagged_chunks > 0 && <Badge tone="coral" title="Some passages contain instruction-like text (possible prompt injection). They are treated as data only.">{doc.flagged_chunks} flagged</Badge>}
                        </div>
                      </td>
                      <td><StatusBadge status={doc.status} /></td>
                      <td className="num">{doc.file_type === "pdf" ? doc.page_count : "–"}</td>
                      <td className="num">{doc.chunk_count}</td>
                      <td className="num text-muted whitespace-nowrap">{bytes(doc.size_bytes)}</td>
                      <td className="text-muted whitespace-nowrap">{when(doc.created_at)}</td>
                      <td className="whitespace-nowrap text-right">
                        <button className="rounded p-1.5 text-faint hover:text-ink" title="Re-index" aria-label={`Re-index ${doc.filename}`} onClick={() => reindex(doc)}><RefreshCw className="size-3.5" /></button>
                        <button className="rounded p-1.5 text-faint hover:text-coral" title="Delete" aria-label={`Delete ${doc.filename}`} onClick={() => remove(doc)}><Trash2 className="size-3.5" /></button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Panel>
          <RetrievalEval documentIds={selected} />
        </div>

        <Panel title="Ask your documents" className="h-fit xl:sticky xl:top-6">
          <div className="space-y-3">
            <ModelSelect value={model || settings.data?.runtime.default_model || ""} onChange={setModel} />
            <form onSubmit={(e) => { e.preventDefault(); ask(); }} className="flex gap-2">
              <Input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. How many days of annual leave do employees get?" aria-label="Question" />
              <Button type="submit" variant="primary" busy={asking} disabled={!question.trim() || ready.length === 0}><Search className="size-4" />Ask</Button>
            </form>
            <div className="text-[12px] text-faint">
              Searching {selected.length ? `${selected.length} selected document${selected.length > 1 ? "s" : ""}` : `all ${ready.length} ready document${ready.length === 1 ? "" : "s"}`}.
              Answers use only the retrieved passages; if nothing relevant is found you get “not found” instead of a guess.
            </div>
            {askError != null && <ErrorNotice error={askError} />}
            {answer && (
              <div className="border-t border-line pt-3">
                {answer.text ? <Markdown text={answer.text} /> : <div className="text-muted text-[13px]">Searching and generating…</div>}
                <SourceList sources={answer.sources} warnings={answer.warnings} />
                <MetricsLine metrics={answer.metrics} model={answer.model} />
              </div>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
