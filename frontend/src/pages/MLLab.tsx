import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { Badge, Button, Empty, ErrorNotice, Input, Notice, PageHeader, Panel, Readout, Table } from "../components/ui";
import { humanize, ms, num, pct, when } from "../lib/format";

interface Dataset {
  dataset_id: string;
  version: string;
  sha256: string;
  seed: number;
  labels: string[];
  sizes: { train: number; val: number; test: number };
  class_counts: Record<string, Record<string, number>>;
  examples: { text: string; label: string }[];
}

interface TestMetrics {
  n: number;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  latency_ms_per_example: number;
  per_class: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
}

interface TrainingRun {
  id: number;
  selected_model: string;
  created_at: string;
  mlflow_run_id: string | null;
  artifact_path: string;
  metrics: {
    model_selection: { candidates: { model: string; val_f1_macro: number; val_accuracy: number; fit_ms: number }[]; selected: string };
    test: TestMetrics;
    test_errors: { text: string; expected: string; predicted: string }[];
    notes: string;
  };
}

interface ComparisonRow {
  approach: string;
  model: string;
  source: string;
  n: number;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  latency_ms_per_example: number | null;
  unparseable_predictions: number;
}

export default function MLLab() {
  const qc = useQueryClient();
  const dataset = useQuery({ queryKey: ["ml-dataset"], queryFn: () => api<Dataset>("/api/ml/dataset") });
  const baseline = useQuery({ queryKey: ["ml-baseline"], queryFn: () => api<{ latest: TrainingRun | null; history: TrainingRun[] }>("/api/ml/baseline") });
  const comparison = useQuery({ queryKey: ["ml-comparison"], queryFn: () => api<{ rows: ComparisonRow[]; note: string; test_size: number }>("/api/ml/comparison") });
  const train = useMutation({
    mutationFn: () => api<TrainingRun>("/api/ml/baseline/train", { method: "POST" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["ml-baseline"] }); qc.invalidateQueries({ queryKey: ["ml-comparison"] }); },
  });
  const [text, setText] = useState("");
  const predict = useMutation({ mutationFn: () => api<{ label: string; model: string; latency_ms: number; scores: Record<string, number> | null }>("/api/ml/baseline/predict", { method: "POST", json: { text } }) });

  const latest = baseline.data?.latest;
  const t = latest?.metrics.test;
  const ds = dataset.data;

  if (dataset.isError) return <ErrorNotice error={dataset.error} />;

  return (
    <div className="max-w-[1200px]">
      <PageHeader
        title="ML lab"
        description="A traditional machine-learning baseline (TF-IDF + linear classifiers) for customer-intent classification, compared with zero-shot LLMs on the same held-out test set."
        actions={<Button variant="primary" onClick={() => train.mutate()} busy={train.isPending}><Play className="size-4" />{latest ? "Retrain baseline" : "Train baseline"}</Button>}
      />
      {train.isError && <div className="mb-4"><ErrorNotice error={train.error} /></div>}

      <div className="grid items-start gap-6 lg:grid-cols-[1fr_1.4fr]">
        <Panel title="Dataset">
          {ds && (
            <div className="space-y-3 text-[13px]">
              <p className="text-muted">Intent dataset v{ds.version}, {ds.labels.length} classes, split with seed {ds.seed} (stratified). SHA-256 <span className="num text-faint">{ds.sha256.slice(0, 16)}…</span></p>
              <div className="grid grid-cols-3 gap-4">
                <Readout label="Train" value={ds.sizes.train} />
                <Readout label="Validation" value={ds.sizes.val} />
                <Readout label="Test (held out)" value={ds.sizes.test} />
              </div>
              <div className="flex flex-wrap gap-1.5">{ds.labels.map((l) => <Badge key={l}>{humanize(l)} <span className="num text-faint">{ds.class_counts.test[l]}</span></Badge>)}</div>
              <details>
                <summary className="cursor-pointer text-muted">Example training messages</summary>
                <ul className="mt-2 space-y-1">{ds.examples.map((e) => <li key={e.text}><span className="text-faint">{humanize(e.label)}:</span> {e.text}</li>)}</ul>
              </details>
            </div>
          )}
        </Panel>

        <Panel title="Test-set results" action={latest && <span className="text-[12px] text-faint">trained {when(latest.created_at)}{latest.mlflow_run_id ? `, MLflow ${latest.mlflow_run_id.slice(0, 8)}` : ""}</span>}>
          {!t ? <Empty title="Not trained yet">Training takes a few seconds on a CPU. Three candidate models are compared on the validation split; the best one is evaluated once on the test split.</Empty> : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <Readout label="Accuracy" value={pct(t.accuracy)} tone="teal" />
                <Readout label="Macro precision" value={pct(t.precision_macro)} />
                <Readout label="Macro recall" value={pct(t.recall_macro)} />
                <Readout label="Macro F1" value={pct(t.f1_macro)} tone="amber" />
                <Readout label="Latency" value={ms(t.latency_ms_per_example)} sub="per message, CPU" />
              </div>
              <div className="text-[13px] text-muted">Selected model: <span className="text-ink">{humanize(latest!.selected_model)}</span>. {latest!.metrics.notes}</div>
              <Table>
                <thead><tr><th>Candidate</th><th>Validation macro F1</th><th>Validation accuracy</th><th>Fit time</th></tr></thead>
                <tbody>{latest!.metrics.model_selection.candidates.map((c) => (
                  <tr key={c.model}><td>{humanize(c.model)} {c.model === latest!.selected_model && <Badge tone="teal">selected</Badge>}</td><td className="num">{num(c.val_f1_macro, 3)}</td><td className="num">{pct(c.val_accuracy)}</td><td className="num">{ms(c.fit_ms)}</td></tr>
                ))}</tbody>
              </Table>
            </div>
          )}
        </Panel>
      </div>

      {t && (
        <div className="mt-6 grid items-start gap-6 lg:grid-cols-2">
          <Panel title="Confusion matrix"><ConfusionMatrix cm={t.confusion_matrix} /></Panel>
          <Panel title="Per-class metrics" bodyClass="p-0">
            <Table>
              <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(t.per_class).map(([c, m]) => (
                <tr key={c}><td>{humanize(c)}</td><td className="num">{num(m.precision, 2)}</td><td className="num">{num(m.recall, 2)}</td><td className="num">{num(m["f1-score"], 2)}</td><td className="num">{m.support}</td></tr>
              ))}</tbody>
            </Table>
            {latest!.metrics.test_errors.length > 0 && (
              <div className="border-t border-line p-4 text-[12.5px]">
                <div className="text-muted mb-1">Misclassified test messages</div>
                <ul className="space-y-1">{latest!.metrics.test_errors.map((e) => <li key={e.text}>“{e.text}” <span className="text-faint">expected {humanize(e.expected)}, got</span> <span className="text-coral">{humanize(e.predicted)}</span></li>)}</ul>
              </div>
            )}
          </Panel>
        </div>
      )}

      <div className="mt-6 grid items-start gap-6 lg:grid-cols-[1fr_1.6fr]">
        <Panel title="Try the classifier">
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); predict.mutate(); }}>
            <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. My invoice shows the wrong amount" aria-label="Message to classify" />
            <Button type="submit" busy={predict.isPending} disabled={!text.trim() || !latest}>Classify</Button>
          </form>
          {predict.isError && <div className="mt-3"><ErrorNotice error={predict.error} /></div>}
          {predict.data && (
            <div className="mt-3 text-[13px]">
              <div>Predicted intent: <span className="font-medium text-amber">{humanize(predict.data.label)}</span> <span className="text-faint">in {ms(predict.data.latency_ms)}</span></div>
              {predict.data.scores && (
                <ul className="mt-2 space-y-1">
                  {Object.entries(predict.data.scores).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                    <li key={k} className="flex justify-between gap-3"><span className="text-muted">{humanize(k)}</span><span className="num">{num(v, 3)}</span></li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </Panel>

        <Panel title="Traditional ML vs. LLM zero-shot" bodyClass="p-0">
          {!comparison.data?.rows.length ? <div className="p-4"><Empty title="Nothing to compare yet">Train the baseline, then run the “Intent Classification” suite on the <Link to="/benchmarks" className="text-amber hover:underline">Benchmarks</Link> page.</Empty></div> : (
            <>
              <Table>
                <thead><tr><th>Approach</th><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Latency / message</th><th>Unparseable</th></tr></thead>
                <tbody>{comparison.data.rows.map((r) => (
                  <tr key={r.source + r.model}><td className="text-muted">{r.approach}</td><td>{r.approach.startsWith("Traditional") ? humanize(r.model) : r.model}<div className="text-[12px] text-faint">{r.source}</div></td><td className="num">{pct(r.accuracy)}</td><td className="num">{num(r.f1_macro, 3)}</td><td className="num">{ms(r.latency_ms_per_example)}</td><td className="num">{r.unparseable_predictions}</td></tr>
                ))}</tbody>
              </Table>
              <div className="p-4"><Notice>{comparison.data.note}</Notice></div>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
