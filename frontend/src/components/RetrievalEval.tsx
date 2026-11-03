import { useMutation } from "@tanstack/react-query";
import { FileUp, Gauge } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import { Badge, Button, cx, ErrorNotice, Notice, Panel, Readout, Table } from "./ui";

interface QRow { id: string; question: string; answerable: boolean; first_relevant_rank?: number | null; false_retrieval?: boolean; relevant_found_in_production?: boolean; retrieved: { rank: number; filename: string; page: number | null; score: number; relevant: boolean; above_threshold: boolean }[] }
interface Result { summary: Record<string, number | number[] | string | null> & { questions: number; answerable: number; unanswerable: number; k_values: number[]; embedder: string; min_score: number }; questions: QRow[]; notes: string[] }

const p = (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(0)}%` : "–");

/** Measure document search quality with a question file: Recall@k, MRR, nDCG and false retrievals. */
export function RetrievalEval({ documentIds }: { documentIds: number[] }) {
  const [file, setFile] = useState<{ name: string; data: { name?: string; questions: unknown[] } } | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const run = useMutation({
    mutationFn: () => api<Result>("/api/documents/retrieval-eval", { method: "POST", json: { name: file!.data.name, questions: file!.data.questions, document_ids: documentIds.length ? documentIds : undefined } }),
  });
  const load = async (f: File) => {
    setParseError(null);
    try {
      const data = JSON.parse(await f.text());
      if (!Array.isArray(data.questions)) throw new Error("no questions");
      setFile({ name: f.name, data });
    } catch { setParseError("Expected a JSON file with a \"questions\" list (see samples/retrieval/sample_qa.json)."); }
  };
  const s = run.data?.summary;
  const k = s?.k_values?.includes(5) ? 5 : s?.k_values?.[s.k_values.length - 1];
  const ci = (key: string) => { const v = s?.[key]; return Array.isArray(v) ? `${p(v[0])}–${p(v[1])}` : undefined; };
  return (
    <Panel title="Retrieval evaluation" action={<span className="text-[12px] text-faint">is the right passage found?</span>}>
      <div className="space-y-3">
        <p className="text-[12.5px] text-muted">Upload questions with the passage that answers each one (file + text it contains). Questions without a relevant passage test whether search correctly finds nothing. Try <code className="text-ink">samples/retrieval/sample_qa.json</code> with the sample documents.</p>
        <div className="flex flex-wrap items-center gap-2">
          <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-raised px-3 text-[13px] hover:border-line-strong">
            <FileUp className="size-4" />{file ? file.name : "Choose question file (.json)"}
            <input type="file" accept=".json" className="sr-only" onChange={(e) => e.target.files?.[0] && load(e.target.files[0])} />
          </label>
          <Button variant="primary" onClick={() => run.mutate()} busy={run.isPending} disabled={!file}><Gauge className="size-4" />Evaluate</Button>
          {file && <span className="text-[12px] text-faint">{file.data.questions.length} questions{documentIds.length ? ` · ${documentIds.length} selected documents` : " · all documents"}</span>}
        </div>
        {parseError && <Notice tone="error" title={parseError} />}
        {run.isError && <ErrorNotice error={run.error} />}
        {s && (
          <>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Readout label={`Recall@${k}`} value={p(s[`recall@${k}`])} sub={ci(`recall@${k}_ci`)} tone="teal" />
              <Readout label="MRR" value={typeof s.mrr === "number" ? s.mrr.toFixed(2) : "–"} sub={Array.isArray(s.mrr_ci) ? `${(s.mrr_ci[0] as number).toFixed(2)}–${(s.mrr_ci[1] as number).toFixed(2)}` : undefined} />
              <Readout label="Answer in context" value={p(s.answer_context_rate)} sub="above the cut-off" />
              <Readout label="False retrieval" value={p(s.false_retrieval_rate)} sub={`${s.unanswerable} unanswerable questions`} tone={(s.false_retrieval_rate as number) > 0.3 ? "coral" : undefined} />
            </div>
            <Table>
              <thead><tr><th>k</th>{s.k_values.map((kk) => <th key={kk} className="num">{kk}</th>)}</tr></thead>
              <tbody>
                {(["recall", "hit", "ndcg"] as const).map((m) => <tr key={m}><td className="text-muted">{m === "ndcg" ? "nDCG" : m === "hit" ? "Hit rate" : "Recall"}</td>{s.k_values.map((kk) => <td key={kk} className="num">{p(s[`${m}@${kk}`])}</td>)}</tr>)}
              </tbody>
            </Table>
            <details className="text-[12.5px]">
              <summary className="cursor-pointer text-muted">Per question ({run.data!.questions.length})</summary>
              <ul className="mt-2 space-y-1.5">{run.data!.questions.map((q) => {
                const bad = q.answerable ? !q.first_relevant_rank : q.false_retrieval;
                return (
                  <li key={q.id} className={cx("rounded border px-2 py-1.5", bad ? "border-coral/40" : "border-line")}>
                    <div className="flex items-start gap-2"><span className="num text-faint">{q.id}</span><span className="min-w-0 flex-1">{q.question}</span>
                      {q.answerable ? <Badge tone={q.first_relevant_rank ? (q.first_relevant_rank <= 3 ? "teal" : "amber") : "coral"}>{q.first_relevant_rank ? `rank ${q.first_relevant_rank}` : "not found"}</Badge>
                        : <Badge tone={q.false_retrieval ? "coral" : "teal"}>{q.false_retrieval ? "retrieved something" : "nothing above cut-off"}</Badge>}</div>
                    <div className="mt-1 text-[11.5px] text-faint">{q.retrieved.slice(0, 3).map((r) => `${r.rank}. ${r.filename}${r.page ? ` p${r.page}` : ""} (${r.score.toFixed(2)}${r.relevant ? ", relevant" : ""})`).join(" · ")}</div>
                  </li>
                );
              })}</ul>
            </details>
            <div className="text-[12px] text-faint">Embedder {s.embedder}, production cut-off {typeof s.min_score === "number" ? s.min_score.toFixed(2) : "–"}. {run.data!.notes.join(" ")}</div>
          </>
        )}
      </div>
    </Panel>
  );
}
