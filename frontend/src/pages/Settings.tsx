import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert, Save } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RuntimeSettings, SettingsResponse } from "../api/types";
import { ModelSelect } from "../components/ModelSelect";
import { Button, ErrorNotice, Field, Input, Notice, PageHeader, Panel, Select, Textarea, Toggle } from "../components/ui";
import { useModels, useSettings } from "../lib/queries";

interface PrivacyStatus {
  checks: { id: string; label: string; ok: boolean; detail: string }[];
  limitations: string[];
  guard: { recent_blocked: { host: string; layer: string; at: number }[]; allowed_hosts: string[] };
}

export default function SettingsPage({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const settings = useSettings();
  const models = useModels();
  const privacy = useQuery({ queryKey: ["privacy"], queryFn: () => api<PrivacyStatus>("/api/privacy/status") });
  const [form, setForm] = useState<RuntimeSettings | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (settings.data) setForm(settings.data.runtime);
  }, [settings.data]);

  const save = useMutation({
    mutationFn: (body: Partial<RuntimeSettings> & { clear_rag_min_score?: boolean }) => api<SettingsResponse>("/api/settings", { method: "PATCH", json: body }),
    onSuccess: (data) => {
      qc.setQueryData(["settings"], data);
      qc.invalidateQueries({ queryKey: ["models"] });
      qc.invalidateQueries({ queryKey: ["privacy"] });
      qc.invalidateQueries({ queryKey: ["documents"] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    },
  });

  if (settings.isError) return <ErrorNotice error={settings.error} />;
  if (!form || !settings.data) return null;
  const st = settings.data.static;
  const set = <K extends keyof RuntimeSettings>(k: K, v: RuntimeSettings[K]) => setForm({ ...form, [k]: v });
  const embedders = (models.data?.models ?? []).filter((m) => m.is_embedding && m.provider === "local");
  const chat = (models.data?.models ?? []).filter((m) => !m.is_embedding);
  const submit = () => {
    const { rag_min_score, ...rest } = form;
    save.mutate(rag_min_score == null ? { ...rest, clear_rag_min_score: true } : form);
  };

  return (
    <div className="max-w-[1100px]">
      <PageHeader
        title="Settings"
        description="Privacy policy, allowed models and resource limits. Changes apply immediately to new requests."
        actions={<Button variant="primary" onClick={submit} busy={save.isPending} disabled={!isAdmin}><Save className="size-4" />{saved ? "Saved" : "Save changes"}</Button>}
      />
      {!isAdmin && <div className="mb-4"><Notice tone="warn" title="Read-only">Only an administrator can change settings.</Notice></div>}
      {save.isError && <div className="mb-4"><ErrorNotice error={save.error} /></div>}

      <div className="grid items-start gap-6 lg:grid-cols-2">
        <Panel title="Privacy">
          <div className="space-y-4">
            <Toggle checked={form.offline_only} onChange={(v) => set("offline_only", v)} disabled={!isAdmin}
              label="Offline-Only mode"
              description="Blocks cloud models and every network request from the backend to hosts outside this computer (and the allow-list below)." />
            <Toggle checked={form.allow_documents_with_cloud} onChange={(v) => set("allow_documents_with_cloud", v)} disabled={!isAdmin || form.offline_only}
              label="Allow documents with cloud models"
              description="When off, document Q&A refuses to send excerpts of your files to a cloud model." />
            {!form.offline_only && (
              <Notice tone="warn" title="Offline-Only is off">
                Prompts sent to cloud models leave this computer.{st.ollama_cloud_api_key_set ? "" : " Ollama Cloud is not configured (OLLAMA_CLOUD_ENABLED and OLLAMA_API_KEY in the backend .env)."}
              </Notice>
            )}
            <ul className="space-y-2 border-t border-line pt-4">
              {privacy.data?.checks.map((c) => (
                <li key={c.id} className="flex gap-2.5 text-[13px]">
                  {c.ok ? <CheckCircle2 className="size-4 mt-0.5 text-teal shrink-0" aria-label="OK" /> : <CircleAlert className="size-4 mt-0.5 text-amber shrink-0" aria-label="Attention" />}
                  <div className="min-w-0"><div>{c.label}</div><div className="text-faint break-words">{c.detail}</div></div>
                </li>
              ))}
            </ul>
            {privacy.data && privacy.data.guard.recent_blocked.length > 0 && (
              <div className="text-[12.5px]">
                <div className="text-muted">Recently blocked connections</div>
                <ul className="mt-1 text-faint">{privacy.data.guard.recent_blocked.map((b) => <li key={`${b.host}-${b.at}`}>{b.host} ({b.layer} layer), {new Date(b.at * 1000).toLocaleTimeString()}</li>)}</ul>
              </div>
            )}
            <details className="text-[12.5px] text-muted">
              <summary className="cursor-pointer">What these protections do not cover</summary>
              <ul className="mt-2 list-disc pl-5 space-y-1">{privacy.data?.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
            </details>
          </div>
        </Panel>

        <Panel title="Models">
          <div className="space-y-4">
            <Field label="Default chat model" hint="Used for new chats, document Q&A and as the benchmark default.">
              <ModelSelect value={form.default_model} onChange={(v) => set("default_model", v)} />
            </Field>
            <Field label="Embedding model for documents" hint="Changing it requires re-indexing existing documents.">
              <Select value={form.embedding_model} onChange={(e) => set("embedding_model", e.target.value)} disabled={!isAdmin}>
                {!embedders.some((m) => m.name.replace(/:latest$/, "") === form.embedding_model.replace(/:latest$/, "")) && form.embedding_model !== "hashing-v1" &&
                  <option value={form.embedding_model}>{form.embedding_model} (not installed – fallback used)</option>}
                {embedders.map((m) => <option key={m.id} value={m.name.replace(/:latest$/, "")}>{m.name}</option>)}
                <option value="hashing-v1">Lexical hashing (offline fallback, no download)</option>
              </Select>
            </Field>
            <fieldset>
              <legend className="text-[12.5px] text-muted mb-1.5">Allowed models (none checked = all installed models allowed)</legend>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {chat.map((m) => (
                  <label key={m.id} className="flex items-center gap-2 text-[13.5px]">
                    <input type="checkbox" className="accent-amber" disabled={!isAdmin} checked={form.allowed_models.includes(m.id)}
                      onChange={(e) => set("allowed_models", e.target.checked ? [...form.allowed_models, m.id] : form.allowed_models.filter((x) => x !== m.id))} />
                    {m.name} {m.provider === "cloud" && <span className="text-amber text-[12px]">cloud</span>}
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
        </Panel>

        <Panel title="Resource limits">
          <div className="grid grid-cols-2 gap-4">
            <Field label="Max upload size (MB)"><Input type="number" min={1} max={500} value={form.max_upload_mb} onChange={(e) => set("max_upload_mb", Number(e.target.value))} disabled={!isAdmin} /></Field>
            <Field label="Max context tokens" hint="Caps num_ctx. Larger contexts need more RAM."><Input type="number" min={512} step={512} value={form.max_context_tokens} onChange={(e) => set("max_context_tokens", Number(e.target.value))} disabled={!isAdmin} /></Field>
            <Field label="Max output tokens"><Input type="number" min={16} value={form.max_output_tokens} onChange={(e) => set("max_output_tokens", Number(e.target.value))} disabled={!isAdmin} /></Field>
            <Field label="Chat timeout (s)"><Input type="number" min={5} value={form.request_timeout_s} onChange={(e) => set("request_timeout_s", Number(e.target.value))} disabled={!isAdmin} /></Field>
            <Field label="Benchmark timeout per test (s)"><Input type="number" min={5} value={form.benchmark_timeout_s} onChange={(e) => set("benchmark_timeout_s", Number(e.target.value))} disabled={!isAdmin} /></Field>
            <Field label="Passages retrieved per question"><Input type="number" min={1} max={20} value={form.retrieval_top_k} onChange={(e) => set("retrieval_top_k", Number(e.target.value))} disabled={!isAdmin} /></Field>
            <Field label="Minimum similarity" hint="Empty = embedder default (0.30 neural, 0.08 lexical).">
              <Input type="number" min={0} max={1} step={0.01} value={form.rag_min_score ?? ""} placeholder="default" disabled={!isAdmin}
                onChange={(e) => set("rag_min_score", e.target.value === "" ? null : Number(e.target.value))} />
            </Field>
          </div>
        </Panel>

        <Panel title="Assistant">
          <Field label="Default system prompt" hint="Used when a conversation has no system prompt of its own.">
            <Textarea rows={6} value={form.default_system_prompt} onChange={(e) => set("default_system_prompt", e.target.value)} disabled={!isAdmin} />
          </Field>
        </Panel>

        <Panel title="Server configuration" className="lg:col-span-2">
          <p className="text-[13px] text-muted mb-3">Read from environment variables or the backend <code>.env</code> file. Restart the backend after changing them.</p>
          <dl className="grid gap-x-6 gap-y-1.5 sm:grid-cols-[220px_1fr] text-[13px]">
            {Object.entries(st).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k.toUpperCase()}</dt>
                <dd className="break-all">{Array.isArray(v) ? v.join(", ") : String(v)}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      </div>
    </div>
  );
}
