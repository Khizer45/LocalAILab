import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, FileDown, FileUp, Plus, Save, Trash2, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { SuiteSummary } from "../api/types";
import { Badge, Button, cx, Empty, ErrorNotice, Field, Input, Notice, PageHeader, Panel, Spinner, Table, Textarea } from "../components/ui";
import { when } from "../lib/format";

export interface CaseDraft { id: string; category: string; prompt: string; reference: string; max_tokens?: number | null; evaluator: string }
export interface SuiteDraft { id: string; name: string; version: string; description: string; system: string; default_max_tokens: number; cases: CaseDraft[] }
interface SuiteFull extends SuiteSummary { system: string | null; default_max_tokens: number; cases: { id: string; category: string; prompt: string; reference: string; max_tokens?: number | null; evaluator: Record<string, unknown> }[] }
interface VersionRow { version: string; sha256: string; name: string; cases: number; created_at: string; note: string | null; deleted: boolean }

const EVALUATOR_EXAMPLES: Record<string, Record<string, unknown>> = {
  exact_match: { type: "exact_match", answers: ["Paris"] },
  contains_all: { type: "contains_all", keywords: ["refund", "14 days"], forbidden: ["guarantee"] },
  contains_any: { type: "contains_any", keywords: ["yes", "correct"] },
  numeric: { type: "numeric", value: 42, tolerance: 0.01 },
  regex: { type: "regex", pattern: "^\\d{4}-\\d{2}-\\d{2}$" },
  label: { type: "label", labels: ["billing", "technical", "account"], answer: "billing" },
  json_object: { type: "json_object", required_keys: ["name", "age"], types: { age: "number" } },
  abstain: { type: "abstain", max_words: 40 },
  rouge_l: { type: "rouge_l", reference: "The meeting moved to Friday at 10.", threshold: 0.5 },
  word_limit: { type: "word_limit", max_words: 50 },
  all_of: { type: "all_of", checks: [{ type: "contains_all", keywords: ["Friday"] }, { type: "word_limit", max_words: 30 }] },
};

const blankCase = (n: number): CaseDraft => ({ id: `case-${n}`, category: "general", prompt: "", reference: "", evaluator: JSON.stringify(EVALUATOR_EXAMPLES.contains_all, null, 2) });
const blank = (): SuiteDraft => ({ id: "my-suite", name: "My suite", version: "1.0.0", description: "", system: "", default_max_tokens: 200, cases: [blankCase(1)] });

function toDraft(s: SuiteFull, copy: boolean): SuiteDraft {
  return {
    id: copy ? `${s.id}-copy`.slice(0, 61) : s.id, name: copy ? `${s.name} (copy)` : s.name, version: copy ? "1.0.0" : s.version,
    description: s.description ?? "", system: s.system ?? "", default_max_tokens: s.default_max_tokens ?? 200,
    cases: s.cases.map((c) => ({ id: c.id, category: c.category, prompt: c.prompt, reference: c.reference ?? "", max_tokens: c.max_tokens ?? null, evaluator: JSON.stringify(c.evaluator, null, 2) })),
  };
}

export function toPayload(d: SuiteDraft) {
  const cases = d.cases.map((c, i) => {
    let evaluator: unknown;
    try { evaluator = JSON.parse(c.evaluator); } catch { throw new Error(`Case ${i + 1} (${c.id}): the evaluator is not valid JSON.`); }
    return { id: c.id.trim(), category: c.category.trim() || "general", prompt: c.prompt, reference: c.reference, ...(c.max_tokens ? { max_tokens: c.max_tokens } : {}), evaluator };
  });
  return { id: d.id.trim(), name: d.name.trim(), version: d.version.trim() || "1.0.0", description: d.description, system: d.system || null, default_max_tokens: d.default_max_tokens, cases };
}

export default function Suites() {
  const qc = useQueryClient();
  const suites = useQuery({ queryKey: ["suites"], queryFn: () => api<{ suites: SuiteSummary[] }>("/api/benchmarks/suites") });
  const [draft, setDraft] = useState<SuiteDraft | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  const load = async (id: string, copy: boolean) => {
    const s = await api<SuiteFull>(`/api/benchmarks/suites/${encodeURIComponent(id)}`);
    setDraft(toDraft(s, copy));
    setEditing(copy ? null : id);
    setLocalError(null);
  };
  const versions = useQuery({
    queryKey: ["suite-versions", editing],
    queryFn: () => api<{ versions: VersionRow[] }>(`/api/benchmarks/suites/${encodeURIComponent(editing!)}/versions`),
    enabled: !!editing,
  });
  const save = useMutation({
    mutationFn: () => api<SuiteSummary>("/api/benchmarks/suites", { method: "POST", json: { suite: toPayload(draft!), note: note || undefined } }),
    onSuccess: (s) => {
      qc.invalidateQueries({ queryKey: ["suites"] });
      qc.invalidateQueries({ queryKey: ["suite-versions"] });
      setEditing(s.id);
      setDraft((d) => (d ? { ...d, version: s.version } : d));
      setNote("");
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api(`/api/benchmarks/suites/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["suites"] }); setDraft(null); setEditing(null); },
  });

  const onSave = () => {
    setLocalError(null);
    try { toPayload(draft!); } catch (e) { setLocalError((e as Error).message); return; }
    save.mutate();
  };
  const exportJson = () => {
    try {
      const blob = new Blob([JSON.stringify(toPayload(draft!), null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${draft!.id}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) { setLocalError((e as Error).message); }
  };
  const importJson = async (f: File) => {
    try {
      const data = JSON.parse(await f.text());
      setDraft(toDraft({ ...data, case_count: 0, categories: {}, sha256: "", source: "" } as SuiteFull, false));
      setEditing(null);
      setLocalError(null);
    } catch { setLocalError("That file is not a valid suite JSON."); }
  };

  const list = suites.data?.suites ?? [];
  const builtin = list.filter((s) => !s.source.startsWith("custom"));
  const custom = list.filter((s) => s.source.startsWith("custom"));

  return (
    <div className="max-w-[1320px]">
      <PageHeader title="Test suites" description={<>Write your own benchmark tests with deterministic evaluators. Every save creates a new version; runs freeze the exact cases and evaluator versions they used. <Link to="/benchmarks" className="text-amber hover:underline">Back to benchmarks</Link></>}
        actions={<>
          <Button onClick={() => { setDraft(blank()); setEditing(null); }}><Plus className="size-4" />New suite</Button>
          <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-raised px-3 text-[13px] hover:border-line-strong"><FileUp className="size-4" />Import JSON
            <input type="file" accept=".json" className="sr-only" onChange={(e) => e.target.files?.[0] && importJson(e.target.files[0])} /></label>
        </>} />
      <div className="grid items-start gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="Your suites" bodyClass="p-0">
            {suites.isLoading ? <div className="p-4"><Spinner /></div> : !custom.length ? <div className="p-4 text-[13px] text-faint">None yet – start a new suite or copy a built-in one.</div> : (
              <ul>{custom.map((s) => (
                <li key={s.id} className={cx("flex items-center gap-2 border-b border-line/60 px-4 py-2 last:border-0", editing === s.id && "bg-raised/60")}>
                  <button className="min-w-0 flex-1 text-left" onClick={() => load(s.id, false)}>
                    <div className="truncate text-[13.5px]">{s.name}</div>
                    <div className="text-[12px] text-faint">{s.id} · v{s.version} · {s.case_count} cases</div>
                  </button>
                  <Button variant="ghost" aria-label={`Delete ${s.name}`} onClick={() => confirm(`Hide suite “${s.name}”? Its version history and past runs are kept.`) && remove.mutate(s.id)}><Trash2 className="size-4" /></Button>
                </li>
              ))}</ul>
            )}
          </Panel>
          <Panel title="Built-in suites" bodyClass="p-0">
            <ul>{builtin.map((s) => (
              <li key={s.id} className="flex items-center gap-2 border-b border-line/60 px-4 py-2 last:border-0">
                <div className="min-w-0 flex-1"><div className="truncate text-[13.5px]">{s.name}</div><div className="text-[12px] text-faint">{s.id} · v{s.version} · {s.case_count} cases</div></div>
                <Button variant="ghost" title="Copy into a new suite you can edit" onClick={() => load(s.id, true)}><Copy className="size-4" /></Button>
              </li>
            ))}</ul>
          </Panel>
          {editing && versions.data && (
            <Panel title="Version history" bodyClass="p-0">
              <Table>
                <thead><tr><th>Version</th><th>Cases</th><th>Saved</th></tr></thead>
                <tbody>{versions.data.versions.slice().reverse().map((v) => (
                  <tr key={v.version + v.created_at}><td><span className="num">{v.version}</span>{v.deleted && <Badge tone="coral">deleted</Badge>}<div className="text-[11.5px] text-faint num" title={v.sha256}>{v.sha256.slice(0, 10)}</div></td><td className="num">{v.cases}</td><td className="text-[12px] text-muted">{when(v.created_at)}{v.note && <div className="text-faint">{v.note}</div>}</td></tr>
                ))}</tbody>
              </Table>
            </Panel>
          )}
        </div>

        <div className="space-y-6 min-w-0">
          {!draft ? <Empty title="Choose or create a suite">Copy a built-in suite to start from working examples, or start empty. Evaluators are deterministic (no LLM judge) so results are repeatable.</Empty> : (
            <Panel title={editing ? `Editing ${draft.id}` : "New suite"} action={<div className="flex gap-2">
              <Button onClick={exportJson}><FileDown className="size-4" />Export</Button>
              <Button variant="primary" onClick={onSave} busy={save.isPending}><Save className="size-4" />Save version</Button>
            </div>}>
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-4">
                  <Field label="Suite id" hint="lowercase, digits, - and _"><Input value={draft.id} disabled={!!editing} onChange={(e) => setDraft({ ...draft, id: e.target.value })} /></Field>
                  <Field label="Name"><Input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} /></Field>
                  <Field label="Version" hint="Bumped automatically if taken"><Input value={draft.version} onChange={(e) => setDraft({ ...draft, version: e.target.value })} /></Field>
                  <Field label="Default max tokens"><Input type="number" min={1} max={8192} value={draft.default_max_tokens} onChange={(e) => setDraft({ ...draft, default_max_tokens: Number(e.target.value) })} /></Field>
                </div>
                <Field label="Description"><Input value={draft.description} onChange={(e) => setDraft({ ...draft, description: e.target.value })} /></Field>
                <Field label="System prompt (optional, for all cases)"><Textarea rows={2} value={draft.system} onChange={(e) => setDraft({ ...draft, system: e.target.value })} /></Field>
                <div className="space-y-3">
                  {draft.cases.map((c, i) => (
                    <CaseEditor key={i} c={c} index={i}
                      onChange={(nc) => setDraft({ ...draft, cases: draft.cases.map((x, j) => (j === i ? nc : x)) })}
                      onRemove={() => setDraft({ ...draft, cases: draft.cases.filter((_, j) => j !== i) })} />
                  ))}
                  <Button onClick={() => setDraft({ ...draft, cases: [...draft.cases, blankCase(draft.cases.length + 1)] })}><Plus className="size-4" />Add case</Button>
                </div>
                <Field label="Version note (optional)"><Input value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} placeholder="What changed in this version?" /></Field>
                {localError && <Notice tone="error" title={localError} />}
                {save.isError && <ErrorNotice error={save.error} />}
                {save.isSuccess && <Notice title={`Saved as version ${save.data.version}`}>Pick it on the Benchmarks page. Saving identical content does not create a new version.</Notice>}
              </div>
            </Panel>
          )}
          <EvaluatorTry />
        </div>
      </div>
    </div>
  );
}

function CaseEditor({ c, index, onChange, onRemove }: { c: CaseDraft; index: number; onChange: (c: CaseDraft) => void; onRemove: () => void }) {
  return (
    <div className="rounded-md border border-line p-3">
      <div className="mb-2 flex items-center gap-2 text-[12.5px] text-muted"><span className="num">#{index + 1}</span><span className="flex-1" /><Button variant="ghost" onClick={onRemove} aria-label="Remove case"><Trash2 className="size-4" /></Button></div>
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Case id"><Input value={c.id} onChange={(e) => onChange({ ...c, id: e.target.value })} /></Field>
        <Field label="Category"><Input value={c.category} onChange={(e) => onChange({ ...c, category: e.target.value })} /></Field>
        <Field label="Max tokens (optional)"><Input type="number" min={1} max={8192} value={c.max_tokens ?? ""} onChange={(e) => onChange({ ...c, max_tokens: e.target.value ? Number(e.target.value) : null })} /></Field>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <Field label="Prompt"><Textarea rows={4} value={c.prompt} onChange={(e) => onChange({ ...c, prompt: e.target.value })} /></Field>
        <Field label="Evaluator (JSON)" hint={<>Types: {Object.keys(EVALUATOR_EXAMPLES).map((t, i) => <span key={t}>{i > 0 && ", "}<button type="button" className="hover:text-amber" onClick={() => onChange({ ...c, evaluator: JSON.stringify(EVALUATOR_EXAMPLES[t], null, 2) })}>{t}</button></span>)}</>}>
          <Textarea rows={4} className="font-mono text-[12px]" value={c.evaluator} onChange={(e) => onChange({ ...c, evaluator: e.target.value })} spellCheck={false} />
        </Field>
      </div>
      <div className="mt-3"><Field label="Reference answer (shown in reports; optional)"><Input value={c.reference} onChange={(e) => onChange({ ...c, reference: e.target.value })} /></Field></div>
    </div>
  );
}

function EvaluatorTry() {
  const [response, setResponse] = useState("The refund is possible within 14 days of purchase.");
  const [spec, setSpec] = useState(JSON.stringify(EVALUATOR_EXAMPLES.contains_all, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);
  const run = useMutation({
    mutationFn: (evaluator: unknown) => api<{ score: number; passed: boolean; method: string; details: Record<string, unknown>; evaluator_version: string }>("/api/benchmarks/evaluators/try", { method: "POST", json: { response, evaluator } }),
  });
  useEffect(() => setParseError(null), [spec]);
  const go = () => {
    try { run.mutate(JSON.parse(spec)); } catch { setParseError("The evaluator is not valid JSON."); }
  };
  return (
    <Panel title="Try an evaluator" action={<Button onClick={go} busy={run.isPending}><Wand2 className="size-4" />Check</Button>}>
      <p className="mb-3 text-[12.5px] text-muted">Paste a model answer and an evaluator to see the score before you save it in a suite. Evaluators catch negation (“not 14 days”), forbidden terms, conflicting numbers and ambiguous labels – but they are rules, not understanding: check a few real answers by hand.</p>
      <div className="grid gap-3 lg:grid-cols-2">
        <Field label="Model answer"><Textarea rows={6} value={response} onChange={(e) => setResponse(e.target.value)} /></Field>
        <Field label="Evaluator (JSON)"><Textarea rows={6} className="font-mono text-[12px]" value={spec} onChange={(e) => setSpec(e.target.value)} spellCheck={false} /></Field>
      </div>
      {parseError && <div className="mt-3"><Notice tone="error" title={parseError} /></div>}
      {run.isError && <div className="mt-3"><ErrorNotice error={run.error} /></div>}
      {run.data && (
        <div className="mt-3 rounded-md border border-line p-3 text-[13px]">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={run.data.passed ? "teal" : "coral"}>{run.data.passed ? "pass" : "fail"}</Badge>
            <span>score <span className="num text-ink">{run.data.score.toFixed(3)}</span></span>
            <span className="text-muted">{run.data.method} · evaluator {run.data.evaluator_version}</span>
          </div>
          <pre className="mt-2 overflow-x-auto rounded bg-bg p-2 text-[12px] text-muted">{JSON.stringify(run.data.details, null, 2)}</pre>
        </div>
      )}
    </Panel>
  );
}
