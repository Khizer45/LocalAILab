import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, Download, FileUp, Play, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, download } from "../api/client";
import type { ArenaCatalog, ArenaDataset, ArenaDatasetDetail, ArenaModel, ArenaRun, DatasetPreview, TrainingData } from "../api/arena";
import { Tabs, useTab } from "../components/Tabs";
import { Badge, Button, cx, Empty, ErrorNotice, Field, Input, Notice, PageHeader, Panel, Select, StatusBadge, Table, Textarea, Toggle } from "../components/ui";
import { bytes, duration, num, when } from "../lib/format";
import { useModels } from "../lib/queries";

const TABS = ["compare", "datasets", "models", "runs"] as const;
type Tab = (typeof TABS)[number];
const TAB_LABELS: Record<Tab, string> = { compare: "Compare", datasets: "Datasets", models: "Models & extractors", runs: "History" };

const kindTone = { tabular: "sky", text: "violet", image: "amber" } as const;

export function useArenaCatalog() {
  return useQuery({ queryKey: ["arena-catalog"], queryFn: () => api<ArenaCatalog>("/api/arena/catalog") });
}

function useDatasets() {
  return useQuery({ queryKey: ["arena-datasets"], queryFn: () => api<{ datasets: ArenaDataset[] }>("/api/arena/datasets") });
}

function useArenaModels() {
  return useQuery({ queryKey: ["arena-models"], queryFn: () => api<{ models: ArenaModel[] }>("/api/arena/models") });
}

function form(data: Record<string, string | number | boolean | File | null | undefined>) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(data)) {
    if (v === undefined || v === null || v === "") continue;
    fd.append(k, v instanceof File ? v : String(v));
  }
  return fd;
}

// ------------------------------------------------------------------ datasets
function DatasetUpload({ onCreated }: { onCreated: (d: ArenaDataset) => void }) {
  const qc = useQueryClient();
  const catalog = useArenaCatalog();
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [task, setTask] = useState("auto");
  const [textColumn, setTextColumn] = useState("");
  const [ignored, setIgnored] = useState<string[]>([]);
  const [testPct, setTestPct] = useState(20);
  const [valPct, setValPct] = useState(20);
  const [strategy, setStrategy] = useState("random");
  const [groupCol, setGroupCol] = useState("");
  const [timeCol, setTimeCol] = useState("");
  const [seed, setSeed] = useState(42);
  const [name, setName] = useState("");

  const preview = useMutation({
    mutationFn: (f: File) => api<DatasetPreview>("/api/arena/datasets/preview", { method: "POST", body: form({ file: f }) }),
    onSuccess: (p) => {
      setTarget(p.suggested_target ?? "");
      setTask("auto");
      setTextColumn("");
      setIgnored([]);
      setStrategy("random");
      setGroupCol("");
      setTimeCol("");
    },
  });
  const create = useMutation({
    mutationFn: () =>
      api<ArenaDataset>("/api/arena/datasets", {
        method: "POST",
        body: form({ file, name: name || undefined, target: target || undefined, task, text_column: textColumn || undefined, ignore: ignored.join(","), test_size: testPct / 100, val_size: valPct / 100, seed,
          split_strategy: strategy, group_column: strategy === "group" ? groupCol : undefined, time_column: strategy === "time" ? timeCol : undefined }),
      }),
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["arena-datasets"] });
      setFile(null);
      preview.reset();
      onCreated(d);
    },
  });

  const p = preview.data;
  const cols = p?.columns ?? [];
  const textCols = cols.filter((c) => c.type === "text" && c.name !== target);
  const limits = catalog.data?.limits;

  return (
    <Panel title="Add a dataset" action={limits && <span className="text-[12px] text-faint">up to {limits.max_upload_mb} MB, {limits.max_rows.toLocaleString()} rows, {limits.max_images.toLocaleString()} images</span>}>
      <div className="space-y-4">
        <div className="text-[13px] text-muted">
          <p><span className="text-ink">CSV</span> for tabular data (numbers and categories) or text (one text column + a label). <span className="text-ink">ZIP of images</span> with one folder per class (<code className="text-ink">cats/1.jpg</code>, <code className="text-ink">dogs/2.jpg</code>), optionally already split into <code className="text-ink">train/</code> and <code className="text-ink">test/</code>.</p>
          <p className="mt-1">No data at hand? Try the samples in <code className="text-ink">samples/arena/</code> (churn, house prices, support tickets, shape images).</p>
        </div>
        <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-line-strong px-4 py-5 text-[13px] text-muted hover:border-amber/60 hover:text-ink">
          <FileUp className="size-4" aria-hidden />
          {file ? <span className="text-ink">{file.name} <span className="text-faint">({bytes(file.size)})</span></span> : "Choose a .csv or .zip file"}
          <input type="file" accept=".csv,.tsv,.txt,.zip" className="sr-only" onChange={(e) => {
            const f = e.target.files?.[0] ?? null;
            setFile(f);
            setName(f ? f.name.replace(/\.[^.]+$/, "") : "");
            if (f) preview.mutate(f);
          }} />
        </label>
        {preview.isPending && <div className="text-[13px] text-muted">Reading the file…</div>}
        {preview.isError && <ErrorNotice error={preview.error} />}

        {p?.format === "csv" && (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Target column (what to predict)">
                <Select value={target} onChange={(e) => { setTarget(e.target.value); setIgnored((x) => x.filter((c) => c !== e.target.value)); }}>
                  {cols.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                </Select>
              </Field>
              <Field label="Task" hint={task === "auto" ? `Detected from the target column` : undefined}>
                <Select value={task} onChange={(e) => setTask(e.target.value)}>
                  <option value="auto">Auto-detect</option>
                  <option value="classification">Classification (categories)</option>
                  <option value="regression">Regression (a number)</option>
                </Select>
              </Field>
              {textCols.length > 0 && (
                <Field label="Text column" hint="Pick one to treat the dataset as text classification (TF-IDF models and LLMs).">
                  <Select value={textColumn} onChange={(e) => setTextColumn(e.target.value)}>
                    <option value="">Auto</option>
                    {textCols.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </Select>
                </Field>
              )}
              <Field label="Dataset name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
            </div>
            <div className="rounded-md border border-line">
              <Table>
                <thead><tr><th>Use</th><th>Column</th><th>Type</th><th>Missing</th><th>Distinct</th><th>Examples</th></tr></thead>
                <tbody>
                  {cols.map((c) => {
                    const isTarget = c.name === target;
                    const auto = c.type === "id" || c.type === "empty";
                    return (
                      <tr key={c.name} className={cx(isTarget && "bg-amber/5")}>
                        <td>{isTarget ? <Badge tone="amber">target</Badge> : (
                          <input type="checkbox" className="accent-amber" disabled={auto} checked={!auto && !ignored.includes(c.name)} title={auto ? "Ignored automatically (ID-like or empty)" : undefined}
                            onChange={(e) => setIgnored((x) => (e.target.checked ? x.filter((y) => y !== c.name) : [...x, c.name]))} />
                        )}</td>
                        <td className="max-w-[180px] truncate" title={c.name}>{c.name}</td>
                        <td><Badge tone={c.type === "numeric" ? "sky" : c.type === "text" ? "violet" : c.type === "id" ? "neutral" : "teal"}>{c.type}</Badge></td>
                        <td className="num">{c.missing}</td>
                        <td className="num">{c.unique}</td>
                        <td className="max-w-[260px] truncate text-muted" title={c.examples.join(", ")}>{c.examples.join(", ")}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </Table>
            </div>
            <div className="text-[12px] text-faint">{p.rows?.toLocaleString()} rows. ID-like, empty and constant columns are ignored automatically.</div>
          </div>
        )}

        {p?.format === "zip" && (
          <div className="space-y-2 text-[13px]">
            <div><span className="num text-ink">{p.images}</span> images in <span className="num text-ink">{Object.keys(p.classes ?? {}).length}</span> classes · split: <span className="text-ink">{p.split_source === "provided" ? "train/ and test/ folders provided" : "random (stratified)"}</span></div>
            <div className="flex flex-wrap gap-1.5">{Object.entries(p.classes ?? {}).map(([c, n]) => <Badge key={c}>{c} <span className="num text-faint">{n}</span></Badge>)}</div>
            {p.warnings?.map((w) => <div key={w} className="text-[12.5px] text-amber">{w}</div>)}
            <Field label="Dataset name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
          </div>
        )}

        {p && (
          <div className="space-y-3">
            {p.format === "csv" && (
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="How to split the rows" hint={strategy === "group" ? "All rows of one group (customer, patient, document…) stay in the same split – use it when rows of one entity are similar." : strategy === "time" ? "Oldest rows train, newer rows validate, newest rows test – use it when the model will predict the future." : "Stratified random split – fine when rows are independent."}>
                  <Select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                    <option value="random">Random (stratified)</option>
                    <option value="group">By group (no group in two splits)</option>
                    <option value="time">By time (train on the past)</option>
                  </Select>
                </Field>
                {strategy === "group" && (
                  <Field label="Group column"><Select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                    <option value="">Choose…</option>{cols.filter((c) => c.name !== target).map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </Select></Field>
                )}
                {strategy === "time" && (
                  <Field label="Time column"><Select value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>
                    <option value="">Choose…</option>{cols.filter((c) => c.name !== target).map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </Select></Field>
                )}
              </div>
            )}
            <div className="grid gap-3 sm:grid-cols-4 items-end">
              <Field label="Test split (%)" hint={p.split_source === "provided" ? "Your test/ folder is used as is." : "Used once, for the final estimate."}>
                <Input type="number" min={5} max={50} value={testPct} disabled={p.split_source === "provided"} onChange={(e) => setTestPct(Math.min(50, Math.max(5, Number(e.target.value))))} />
              </Field>
              <Field label="Validation (% of the rest)" hint="Used to choose the model. 0 = exploratory only.">
                <Input type="number" min={0} max={50} value={valPct} onChange={(e) => setValPct(Math.min(50, Math.max(0, Number(e.target.value))))} />
              </Field>
              <Field label="Random seed"><Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
              <Button variant="primary" onClick={() => create.mutate()} busy={create.isPending} disabled={!file || (strategy === "group" && !groupCol) || (strategy === "time" && !timeCol)}><Upload className="size-4" />Create dataset</Button>
            </div>
            {valPct === 0 && <div className="text-[12.5px] text-amber">Without a validation split the model is chosen on the test rows, so the reported score is optimistic (exploratory).</div>}
          </div>
        )}
        {create.isError && <ErrorNotice error={create.error} />}
      </div>
    </Panel>
  );
}

function DatasetCard({ d, onCompare }: { d: ArenaDataset; onCompare: () => void }) {
  const qc = useQueryClient();
  const remove = useMutation({
    mutationFn: () => api(`/api/arena/datasets/${d.id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["arena-datasets"] }),
  });
  const counts = (d.profile.class_counts ?? {}) as Record<string, number>;
  const stats = d.profile.target_stats as { min: number; max: number; mean: number } | undefined;
  return (
    <div className="rounded-lg border border-line bg-panel p-4">
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium truncate">{d.name}</span>
            <Badge tone={kindTone[d.kind]}>{d.kind}</Badge>
            <Badge>{d.task}</Badge>
          </div>
          <div className="mt-0.5 text-[12.5px] text-muted">
            <span className="num">{d.n_rows.toLocaleString()}</span> {d.kind === "image" ? "images" : "rows"} · train <span className="num">{d.n_train}</span> / validation <span className="num">{d.n_val ?? 0}</span> / test <span className="num">{d.n_test}</span>
            {" "}· split <span className="text-ink">{d.split_source}{d.group_column ? ` by ${d.group_column}` : d.time_column ? ` by ${d.time_column}` : ""}</span>
            {d.target && <> · target <span className="text-ink">{d.target}</span></>}
            {d.text_column && <> · text <span className="text-ink">{d.text_column}</span></>}
            {d.kind === "tabular" && <> · {d.columns.features?.length} features</>}
            <> · seed {d.seed} · {when(d.created_at)}</>
          </div>
        </div>
        <Button variant="primary" onClick={onCompare}><Play className="size-4" />Compare models</Button>
        <Button onClick={() => download(`/api/arena/datasets/${d.id}/split.csv`)} title="Which row is in train / validation / test – train your own model (e.g. in the Colab notebooks) on exactly the training rows"><Download className="size-4" />Split</Button>
        <Button variant="danger" onClick={() => confirm(`Delete dataset “${d.name}”? Past comparison results are kept.`) && remove.mutate()} busy={remove.isPending} aria-label="Delete dataset"><Trash2 className="size-4" /></Button>
      </div>
      {d.task === "classification" && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {Object.entries(counts).slice(0, 30).map(([c, n]) => <Badge key={c} title={c}>{c.length > 24 ? c.slice(0, 24) + "…" : c} <span className="num text-faint">{n}</span></Badge>)}
          {Object.keys(counts).length > 30 && <Badge>+{Object.keys(counts).length - 30} more</Badge>}
        </div>
      )}
      {d.split_sha256 && <div className="mt-1 text-[12px] text-faint">Split SHA-256 <span className="num">{d.split_sha256.slice(0, 16)}…</span> – models trained in the notebooks carry this in their metadata.</div>}
      {stats && <div className="mt-2 text-[12.5px] text-muted">Target range <span className="num text-ink">{num(stats.min, 2)} – {num(stats.max, 2)}</span>, mean <span className="num text-ink">{num(stats.mean, 2)}</span></div>}
      {d.warnings.length > 0 && (
        <ul className="mt-2 space-y-0.5 text-[12.5px] text-amber">
          {d.warnings.map((w) => <li key={w}>⚠ {w}</li>)}
        </ul>
      )}
      {remove.isError && <div className="mt-2"><ErrorNotice error={remove.error} /></div>}
    </div>
  );
}

// ------------------------------------------------------------------ models
function ModelUpload() {
  const qc = useQueryClient();
  const catalog = useArenaCatalog();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [kind, setKind] = useState("image");
  const [role, setRole] = useState("predictor");
  const [task, setTask] = useState("classification");
  const [labels, setLabels] = useState("");
  const [normalization, setNormalization] = useState("imagenet");
  const [size, setSize] = useState("");
  const [layout, setLayout] = useState("auto");
  const [color, setColor] = useState("rgb");
  const [resizeMode, setResizeMode] = useState("resize");
  const [featureCols, setFeatureCols] = useState("");
  const [trusted, setTrusted] = useState(false);
  const [trainingData, setTrainingData] = useState<TrainingData>("unknown");
  const [provNote, setProvNote] = useState("");
  const isPickle = !!file && /\.(joblib|pkl|pickle)$/i.test(file.name);

  const upload = useMutation({
    mutationFn: () =>
      api<ArenaModel>("/api/arena/models", {
        method: "POST",
        body: form({
          file, name: name || undefined, kind, role, task, labels: labels || undefined,
          ...(kind === "image" ? { normalization, size: size || undefined, layout, color, resize_mode: resizeMode } : {}),
          feature_columns: featureCols || undefined, trusted: isPickle ? trusted : undefined,
          training_data: trainingData, provenance_note: provNote || undefined,
        }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["arena-models"] });
      setFile(null);
      setName("");
    },
  });

  return (
    <Panel title="Upload your model">
      <div className="space-y-4">
        <p className="text-[13px] text-muted">
          <span className="text-ink">ONNX</span> works for any framework (PyTorch, Keras/TensorFlow, scikit-learn); uploaded models run in a separate, memory- and time-limited worker process. scikit-learn <span className="text-ink">.joblib/.pkl</span> files are accepted from administrators who trust the file.
        </p>
        <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-line-strong px-4 py-4 text-[13px] text-muted hover:border-amber/60 hover:text-ink">
          <FileUp className="size-4" aria-hidden />
          {file ? <span className="text-ink">{file.name} <span className="text-faint">({bytes(file.size)})</span></span> : "Choose a .onnx, .joblib or .pkl file"}
          <input type="file" accept=".onnx,.joblib,.pkl,.pickle" className="sr-only" onChange={(e) => {
            const f = e.target.files?.[0] ?? null;
            setFile(f);
            setName(f ? f.name.replace(/\.[^.]+$/, "") : "");
            if (f && /\.(joblib|pkl|pickle)$/i.test(f.name) && kind === "image") setKind("tabular");
          }} />
        </label>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. MobileNetV2 fine-tuned" /></Field>
          <Field label="Data type">
            <Select value={kind} onChange={(e) => { setKind(e.target.value); if (e.target.value !== "image") setRole("predictor"); }}>
              <option value="image" disabled={isPickle}>Images</option>
              <option value="tabular">Tabular (CSV columns)</option>
              <option value="text">Text</option>
            </Select>
          </Field>
          <Field label="Use as">
            <Select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="predictor">Classifier / regressor (compare it)</option>
              <option value="feature_extractor" disabled={kind !== "image" || isPickle}>Feature extractor (transfer learning)</option>
            </Select>
          </Field>
          {role === "predictor" && (
            <Field label="Task">
              <Select value={task} onChange={(e) => setTask(e.target.value)}>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </Select>
            </Field>
          )}
        </div>
        {role === "predictor" && task === "classification" && (
          <Field label="Class labels in output order (optional)" hint="One per line or comma-separated. Leave empty if the model was trained with the class folders/labels in alphabetical order (Keras image_dataset_from_directory and torchvision ImageFolder do this).">
            <Textarea rows={2} value={labels} onChange={(e) => setLabels(e.target.value)} placeholder="cat, dog, horse" />
          </Field>
        )}
        {kind === "image" && (
          <fieldset className="grid gap-3 sm:grid-cols-2">
            <legend className="mb-2 text-[12.5px] text-muted">Image preprocessing (must match how the model was trained)</legend>
            <Field label="Normalisation" hint={catalog.data?.normalizations[normalization]}>
              <Select value={normalization} onChange={(e) => setNormalization(e.target.value)}>
                {Object.keys(catalog.data?.normalizations ?? { imagenet: "" }).map((n) => <option key={n} value={n}>{n}</option>)}
              </Select>
            </Field>
            <Field label="Input size (px)" hint="Empty = read from the model"><Input type="number" min={8} max={1024} value={size} onChange={(e) => setSize(e.target.value)} placeholder="auto" /></Field>
            <Field label="Layout"><Select value={layout} onChange={(e) => setLayout(e.target.value)}><option value="auto">Auto (from input shape)</option><option value="nchw">NCHW (PyTorch)</option><option value="nhwc">NHWC (TensorFlow/Keras)</option></Select></Field>
            <Field label="Colour order"><Select value={color} onChange={(e) => setColor(e.target.value)}><option value="rgb">RGB</option><option value="bgr">BGR (OpenCV / Caffe)</option></Select></Field>
            <Field label="Resize"><Select value={resizeMode} onChange={(e) => setResizeMode(e.target.value)}><option value="resize">Stretch to size</option><option value="center_crop">Resize + centre crop (ImageNet style)</option></Select></Field>
          </fieldset>
        )}
        {kind === "tabular" && !isPickle && (
          <Field label="Feature columns (optional)" hint="For ONNX models with one numeric input: the CSV columns in training order, comma-separated.">
            <Input value={featureCols} onChange={(e) => setFeatureCols(e.target.value)} placeholder="age, income, tenure_months" />
          </Field>
        )}
        {role === "predictor" && (
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="What was it trained on?" hint="A model that saw the validation or test rows gets optimistic scores. Models exported by the notebooks record this automatically.">
              <Select value={trainingData} onChange={(e) => setTrainingData(e.target.value as TrainingData)}>
                <option value="unknown">I don't know</option>
                <option value="train_split">Only the training split exported from this app</option>
                <option value="external">Separate data that does not contain these rows</option>
              </Select>
            </Field>
            <Field label="Note (optional)"><Input value={provNote} onChange={(e) => setProvNote(e.target.value)} maxLength={500} placeholder="e.g. Colab notebook, 2026-10-12" /></Field>
          </div>
        )}
        {isPickle && (
          <Notice tone="warn" title="Pickle files can run code when loaded">
            <Toggle checked={trusted} onChange={setTrusted} label="I trust this file (I created it or got it from a source I trust)" description="Requires administrator rights. Prefer ONNX when you can." />
          </Notice>
        )}
        <Button variant="primary" onClick={() => upload.mutate()} busy={upload.isPending} disabled={!file || (isPickle && !trusted)}><Upload className="size-4" />Upload and check model</Button>
        {upload.isError && <ErrorNotice error={upload.error} />}
        <details className="text-[12.5px] text-muted">
          <summary className="cursor-pointer text-ink">How do I convert my model to ONNX?</summary>
          <div className="mt-2 space-y-2">
            <div><span className="text-ink">PyTorch</span> (e.g. a fine-tuned MobileNetV2 / EfficientNet):<pre className="mt-1 overflow-x-auto rounded bg-bg p-2 text-[12px]">{`model.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, x, "model.onnx", input_names=["input"],
                  output_names=["logits"], dynamic_axes={"input": {0: "batch"}})`}</pre></div>
            <div><span className="text-ink">Keras / TensorFlow</span>:<pre className="mt-1 overflow-x-auto rounded bg-bg p-2 text-[12px]">{`pip install tf2onnx
model.export("saved_model")            # Keras 3
python -m tf2onnx.convert --saved-model saved_model --output model.onnx`}</pre></div>
            <div><span className="text-ink">scikit-learn</span>:<pre className="mt-1 overflow-x-auto rounded bg-bg p-2 text-[12px]">{`pip install skl2onnx
from skl2onnx import to_onnx
onx = to_onnx(pipeline, X_train[:1])
open("model.onnx", "wb").write(onx.SerializeToString())`}</pre></div>
          </div>
        </details>
      </div>
    </Panel>
  );
}

function Backbones() {
  const qc = useQueryClient();
  const catalog = useArenaCatalog();
  const dl = useMutation({
    mutationFn: (id: string) => api(`/api/arena/backbones/${id}/download`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["arena-catalog"] }),
  });
  return (
    <Panel title="Pretrained feature extractors (images)">
      <p className="mb-3 text-[13px] text-muted">A pretrained network turns each image into a feature vector; the built-in algorithms then learn <em>your</em> classes on top (transfer learning). Downloaded once, then used fully offline.</p>
      <div className="space-y-2">
        {catalog.data?.backbones.map((b) => (
          <div key={b.id} className="flex flex-wrap items-center gap-3 rounded-md border border-line px-3 py-2">
            <div className="min-w-0 flex-1">
              <div className="text-[13.5px]">{b.name} <span className="text-faint text-[12px]">{b.size_mb} MB · {b.feature_dim} features</span></div>
              <div className="text-[12px] text-faint">{b.notes} {b.source}.</div>
            </div>
            {b.installed ? <Badge tone="teal">installed</Badge> : (
              <Button onClick={() => dl.mutate(b.id)} busy={dl.isPending && dl.variables === b.id}><Download className="size-4" />Download</Button>
            )}
          </div>
        ))}
      </div>
      {dl.isError && <div className="mt-3"><ErrorNotice error={dl.error} /></div>}
      <p className="mt-3 text-[12px] text-faint">Offline-Only mode blocks downloads. Run <code className="text-ink">scripts\download_backbones.bat</code> once instead (it verifies the SHA-256), or upload any ONNX image classifier as a “feature extractor”.</p>
    </Panel>
  );
}

function ModelList() {
  const qc = useQueryClient();
  const models = useArenaModels();
  const remove = useMutation({
    mutationFn: (id: number) => api(`/api/arena/models/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["arena-models"] }),
  });
  if (!models.data?.models.length) return <div className="p-4"><Empty title="No models uploaded yet">Upload an ONNX or scikit-learn model to compare it against the built-in algorithms on your data.</Empty></div>;
  return (
    <Table>
      <thead><tr><th>Model</th><th>Type</th><th>Size</th><th /></tr></thead>
      <tbody>
        {models.data.models.map((m) => (
          <tr key={m.id}>
            <td className="max-w-[220px]">
              <div className="truncate" title={m.name}>{m.name}</div>
              <div className="truncate text-[12px] text-faint" title={m.inspection.inputs?.map((i) => `${i.name} [${i.shape.join("×")}]`).join(", ")}>
                {m.filename}{m.inspection.inputs?.[0] ? ` · input [${m.inspection.inputs[0].shape.join("×")}]` : m.inspection.class ? ` · ${m.inspection.class}` : ""}
                {m.inspection.parameters ? ` · ${(m.inspection.parameters / 1e6).toFixed(2)} M params` : ""}
              </div>
            </td>
            <td>
              <div className="flex flex-wrap gap-1"><Badge tone={m.format === "onnx" ? "teal" : "amber"}>{m.format}</Badge><Badge tone={kindTone[m.kind]}>{m.kind}</Badge>
                {m.role === "predictor" && <Badge tone={m.provenance?.training_data === "unknown" || !m.provenance ? "amber" : "neutral"} title={m.provenance?.note ?? undefined}>{m.provenance?.training_data === "train_split" ? (m.provenance.source === "metadata" ? "train split (metadata)" : "train split (declared)") : m.provenance?.training_data === "external" ? "external data" : "training data unknown"}</Badge>}
              </div>
              <div className="text-muted text-[12px]">{m.role === "feature_extractor" ? `feature extractor (${m.inspection.feature_dim})` : m.task}</div>
            </td>
            <td className="num whitespace-nowrap">{bytes(m.size_bytes)}</td>
            <td className="text-right"><Button variant="ghost" onClick={() => confirm(`Delete “${m.name}”?`) && remove.mutate(m.id)} aria-label="Delete model"><Trash2 className="size-4" /></Button></td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

// ------------------------------------------------------------------ compare
function CompareForm({ datasetId, setDatasetId, goTo }: { datasetId: number | null; setDatasetId: (id: number) => void; goTo: (t: Tab) => void }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const datasets = useDatasets();
  const catalog = useArenaCatalog();
  const arenaModels = useArenaModels();
  const llms = useModels();
  const detail = useQuery({
    queryKey: ["arena-dataset", datasetId],
    queryFn: () => api<ArenaDatasetDetail>(`/api/arena/datasets/${datasetId}`),
    enabled: datasetId != null,
  });
  const ds = detail.data;
  const [algos, setAlgos] = useState<string[]>([]);
  const [uploaded, setUploaded] = useState<number[]>([]);
  const [llmPicked, setLlmPicked] = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>([]);
  const [cvFolds, setCvFolds] = useState(5);
  const [tune, setTune] = useState(false);
  const [llmMax, setLlmMax] = useState(100);
  const [name, setName] = useState("");
  const [objective, setObjective] = useState<"efficiency" | "quality">("efficiency");
  const [margin, setMargin] = useState("");
  const [minQuality, setMinQuality] = useState("");
  const [maxLatency, setMaxLatency] = useState("");
  const [maxSize, setMaxSize] = useState("");
  const [maxMemory, setMaxMemory] = useState("");
  const numOrNull = (v: string) => (v.trim() === "" ? null : Number(v));

  useEffect(() => {
    if (!datasets.data?.datasets.length || datasetId != null) return;
    setDatasetId(datasets.data.datasets[0].id);
  }, [datasets.data, datasetId, setDatasetId]);

  useEffect(() => {
    if (!ds) return;
    setAlgos(ds.algorithms.filter((a) => a.default && a.available && a.id !== "dummy").map((a) => a.id));
    setUploaded([]);
    setLlmPicked([]);
    const mobilenet = catalog.data?.backbones.find((b) => b.id === "mobilenetv2")?.installed;
    setFeatures(ds.kind === "image" ? (mobilenet ? ["raw_pixels", "backbone:mobilenetv2"] : ["raw_pixels"]) : []);
  }, [ds?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const extractors = useMemo(() => {
    const list: { key: string; label: string; ok: boolean; note?: string }[] = [{ key: "raw_pixels", label: "Raw pixels (32×32, no neural network)", ok: true }];
    for (const b of catalog.data?.backbones ?? []) list.push({ key: `backbone:${b.id}`, label: b.name, ok: b.installed, note: b.installed ? undefined : "not downloaded" });
    for (const m of arenaModels.data?.models ?? []) if (m.role === "feature_extractor") list.push({ key: `model:${m.id}`, label: `${m.name} (yours)`, ok: true });
    return list;
  }, [catalog.data, arenaModels.data]);

  const start = useMutation({
    mutationFn: () =>
      api<ArenaRun>("/api/arena/runs", {
        method: "POST",
        json: { dataset_id: datasetId, algorithms: algos, uploaded_models: uploaded, llm_models: llmPicked, image_features: features, cv_folds: cvFolds, tune, llm_max_examples: llmMax, name: name || undefined,
          objective, equivalence_margin: numOrNull(margin), min_quality: numOrNull(minQuality), max_latency_ms: numOrNull(maxLatency),
          max_size_mb: numOrNull(maxSize), max_memory_mb: numOrNull(maxMemory) },
      }),
    onSuccess: (run) => {
      qc.invalidateQueries({ queryKey: ["arena-runs"] });
      nav(`/arena/runs/${run.id}`);
    },
  });

  if (datasets.isError) return <ErrorNotice error={datasets.error} />;
  if (datasets.data && !datasets.data.datasets.length)
    return <Empty title="Add a dataset first" action={<Button variant="primary" onClick={() => goTo("datasets")}><Database className="size-4" />Add a dataset</Button>}>Upload a CSV or a ZIP of images – or use one of the samples in <code>samples/arena/</code>.</Empty>;

  const toggle = <T,>(list: T[], set: (v: T[]) => void, v: T) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const chatModels = (llms.data?.models ?? []).filter((m) => !m.is_embedding && m.allowed);
  const imageMultiplier = ds?.kind === "image" ? Math.max(1, features.length) : 1;
  const count = 1 + algos.length * imageMultiplier + uploaded.length + llmPicked.length;

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
      <Panel title="New comparison">
        <div className="space-y-5">
          <Field label="Dataset">
            <Select value={datasetId ?? ""} onChange={(e) => setDatasetId(Number(e.target.value))}>
              {datasets.data?.datasets.map((d) => <option key={d.id} value={d.id}>{d.name} – {d.kind} {d.task}, {d.n_rows} {d.kind === "image" ? "images" : "rows"}</option>)}
            </Select>
          </Field>
          {ds && (
            <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
              <Badge tone={kindTone[ds.kind]}>{ds.kind}</Badge><Badge>{ds.task}</Badge>
              <span>train <span className="num text-ink">{ds.n_train}</span> / validation <span className="num text-ink">{ds.n_val ?? 0}</span> / test <span className="num text-ink">{ds.n_test}</span></span>
              {!ds.n_val && <Badge tone="amber" title="A validation split is carved from the training rows when the run starts (older datasets), or the run is exploratory if that is impossible.">no validation split yet</Badge>}
              {ds.task === "classification" && <span>{ds.labels.length} classes</span>}
              {ds.warnings.length > 0 && <Badge tone="amber" title={ds.warnings.join("\n")}>{ds.warnings.length} data warning{ds.warnings.length > 1 ? "s" : ""}</Badge>}
            </div>
          )}

          {ds?.kind === "image" && (
            <fieldset>
              <legend className="mb-2 text-[12.5px] text-muted">Image features for the built-in algorithms</legend>
              <div className="space-y-1">
                {extractors.map((x) => (
                  <label key={x.key} className={cx("flex items-center gap-2 text-[13.5px]", !x.ok && "opacity-50")}>
                    <input type="checkbox" className="accent-amber" disabled={!x.ok} checked={features.includes(x.key)} onChange={() => toggle(features, setFeatures, x.key)} />
                    {x.label} {x.note && <span className="text-[12px] text-faint">({x.note} – see Models & extractors)</span>}
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {ds && (
            <fieldset>
              <legend className="mb-2 text-[12.5px] text-muted">Built-in algorithms (trained here on the training split)</legend>
              <div className="grid gap-1.5 sm:grid-cols-2">
                {ds.algorithms.filter((a) => a.id !== "dummy").map((a) => (
                  <label key={a.id} title={a.description + (a.available ? "" : ` – install with: pip install ${a.requires}`)} className={cx("flex items-start gap-2 rounded-md border px-2.5 py-1.5 text-[13px]", algos.includes(a.id) ? "border-amber/50 bg-amber/5" : "border-line", !a.available && "opacity-50")}>
                    <input type="checkbox" className="mt-0.5 accent-amber" disabled={!a.available} checked={algos.includes(a.id)} onChange={() => toggle(algos, setAlgos, a.id)} />
                    <span className="min-w-0">
                      <span className="block truncate">{a.name}</span>
                      <span className="block text-[11.5px] text-faint">{a.family} · {a.speed}{!a.available && ` · needs ${a.requires}`}</span>
                    </span>
                  </label>
                ))}
              </div>
              <p className="mt-1.5 text-[12px] text-faint">A majority-class / mean baseline is always included – every real model should beat it.</p>
            </fieldset>
          )}

          {ds && (
            <fieldset>
              <legend className="mb-2 text-[12.5px] text-muted">Your uploaded models (evaluated, not retrained)</legend>
              {!ds.models.filter((m) => m.role === "predictor").length ? <p className="text-[12.5px] text-faint">None yet. <button className="text-amber hover:underline" onClick={() => goTo("models")}>Upload a model</button> to compare it here.</p> : (
                <div className="space-y-1">
                  {ds.models.filter((m) => m.role === "predictor").map((m) => (
                    <label key={m.id} className={cx("flex items-center gap-2 text-[13.5px]", m.reason && "opacity-50")} title={m.reason ?? undefined}>
                      <input type="checkbox" className="accent-amber" disabled={!!m.reason} checked={uploaded.includes(m.id)} onChange={() => toggle(uploaded, setUploaded, m.id)} />
                      {m.name} {m.reason && <span className="text-[12px] text-faint">({m.reason})</span>}
                    </label>
                  ))}
                </div>
              )}
            </fieldset>
          )}

          {ds?.kind === "text" && ds.task === "classification" && (
            <fieldset>
              <legend className="mb-2 text-[12.5px] text-muted">Local LLMs (zero-shot, no training)</legend>
              <div className="space-y-1">
                {chatModels.map((m) => (
                  <label key={m.id} className="flex items-center gap-2 text-[13.5px]">
                    <input type="checkbox" className="accent-amber" checked={llmPicked.includes(m.id)} onChange={() => toggle(llmPicked, setLlmPicked, m.id)} />
                    {m.name} {m.parameter_size && <span className="text-[12px] text-faint">{m.parameter_size}</span>}
                  </label>
                ))}
                {!chatModels.length && <p className="text-[12.5px] text-faint">No chat models available in Ollama.</p>}
              </div>
              {llmPicked.length > 0 && (
                <div className="mt-2 grid gap-3 sm:grid-cols-2">
                  <Field label="Max test examples per LLM" hint="LLMs are slow; a stratified sample keeps runs short."><Input type="number" min={5} max={5000} value={llmMax} onChange={(e) => setLlmMax(Number(e.target.value))} /></Field>
                </div>
              )}
              <p className="mt-1.5 text-[12px] text-faint">Zero-shot works best when class names are words (“billing”), not codes (“0”, “1”).</p>
            </fieldset>
          )}

          {ds && (
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Cross-validation folds" hint="On the training split, for stability">
                <Select value={cvFolds} onChange={(e) => setCvFolds(Number(e.target.value))}>
                  <option value={0}>Off</option><option value={3}>3</option><option value={5}>5</option><option value={10}>10</option>
                </Select>
              </Field>
              <Field label="Run name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="automatic" /></Field>
              <div className="flex items-end pb-1"><Toggle checked={tune} onChange={setTune} label="Tune hyper-parameters" description="Random search, 3-fold CV (slower)" /></div>
            </div>
          )}
          {ds && (
            <details className="rounded-md border border-line px-3 py-2">
              <summary className="cursor-pointer text-[13px] text-ink">Recommendation rules (optional)</summary>
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <Field label="Prefer" hint={objective === "efficiency" ? "The fastest model that is practically equivalent to the best one" : "The model with the best validation score"}>
                  <Select value={objective} onChange={(e) => setObjective(e.target.value as "efficiency" | "quality")}>
                    <option value="efficiency">Efficiency</option><option value="quality">Quality</option>
                  </Select>
                </Field>
                <Field label="Equivalence margin" hint={ds.task === "classification" ? "Largest difference that does not matter to you, e.g. 0.02 = 2 points of F1. Empty = 0.02." : "Empty = 5% of the leader's score."}>
                  <Input type="number" step="0.005" min={0} max={0.5} value={margin} onChange={(e) => setMargin(e.target.value)} placeholder="default" />
                </Field>
                <Field label={`Minimum ${ds.task === "classification" ? "macro F1" : "R²"}`}><Input type="number" step="0.01" value={minQuality} onChange={(e) => setMinQuality(e.target.value)} placeholder="none" /></Field>
                <Field label="Max latency (ms / prediction)"><Input type="number" min={0} value={maxLatency} onChange={(e) => setMaxLatency(e.target.value)} placeholder="none" /></Field>
                <Field label="Max model size (MB)"><Input type="number" min={0} value={maxSize} onChange={(e) => setMaxSize(e.target.value)} placeholder="none" /></Field>
                <Field label="Max memory (MB)"><Input type="number" min={0} value={maxMemory} onChange={(e) => setMaxMemory(e.target.value)} placeholder="none" /></Field>
              </div>
              <p className="mt-2 text-[12px] text-faint">Models that break a limit are excluded from the recommendation but still shown.</p>
            </details>
          )}
          {start.isError && <ErrorNotice error={start.error} />}
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" onClick={() => start.mutate()} busy={start.isPending} disabled={!ds || count < 2 || (ds.kind === "image" && algos.length > 0 && !features.length)}><Play className="size-4" />Start comparison</Button>
            <span className="text-[12.5px] text-muted num">{count} contenders (incl. baseline)</span>
          </div>
        </div>
      </Panel>
      <div className="space-y-4">
        <Notice title="How the comparison works">
          <ol className="mt-1 list-decimal space-y-1 pl-4">
            <li>Your data is split once into train / validation / test (fixed seed; random, by group or by time). Exact duplicates and near-duplicates across splits are reported.</li>
            <li>Built-in algorithms train on the training part; uploaded models and LLMs are only evaluated.</li>
            <li>The model is <em>chosen on validation</em>: paired bootstrap against the leader with an equivalence margin and Holm correction. Verdicts are equivalent, worse or inconclusive – inconclusive is not a tie.</li>
            <li>The test split is used once, to report the chosen model's score with a 95% CI – an unbiased estimate for similar data.</li>
          </ol>
        </Notice>
        <RecentRuns compact />
      </div>
    </div>
  );
}

function RecentRuns({ compact }: { compact?: boolean }) {
  const qc = useQueryClient();
  const runs = useQuery({
    queryKey: ["arena-runs"],
    queryFn: () => api<{ runs: ArenaRun[] }>("/api/arena/runs"),
    refetchInterval: (q) => (q.state.data?.runs.some((r) => r.status === "running" || r.status === "queued") ? 1500 : 15000),
  });
  const list = (runs.data?.runs ?? []).slice(0, compact ? 6 : 200);
  const cancel = async (id: number) => {
    await api(`/api/arena/runs/${id}/cancel`, { method: "POST" });
    qc.invalidateQueries({ queryKey: ["arena-runs"] });
  };
  return (
    <Panel title={compact ? "Recent comparisons" : "All comparisons"} bodyClass="p-0">
      {!list.length ? <div className="p-4 text-[13px] text-faint">No comparisons yet.</div> : (
        <Table>
          <thead><tr><th>Run</th>{!compact && <><th>Status</th><th>Recommended</th><th>Duration</th><th>Started</th><th /></>}</tr></thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.id}>
                <td className={compact ? "max-w-[300px]" : "max-w-[260px]"}>
                  <Link to={`/arena/runs/${r.id}`} className="block truncate hover:text-amber" title={r.name}>{r.name}</Link>
                  <div className="flex flex-wrap items-center gap-1.5 text-[12px] text-faint">
                    {compact && <StatusBadge status={r.status} />}
                    <span>{r.dataset.kind} {r.dataset.task} · {r.contenders.length} models</span>
                    {(r.status === "running" || r.status === "queued") && <span className="num">{r.progress_done}/{r.progress_total}</span>}
                  </div>
                </td>
                {!compact && <>
                  <td><StatusBadge status={r.status} /></td>
                  <td className="max-w-[220px] truncate">{r.summary?.recommendation?.name ?? "–"}</td>
                  <td className="num">{duration(r.duration_s)}</td>
                  <td className="text-muted whitespace-nowrap">{when(r.started_at ?? r.created_at)}</td>
                  <td className="text-right">{(r.status === "running" || r.status === "queued") && <Button variant="ghost" onClick={() => cancel(r.id)}>Cancel</Button>}</td>
                </>}
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}

export default function ModelArena() {
  const [tab, setTab] = useTab(TABS, "compare");
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const datasets = useDatasets();
  return (
    <div className="max-w-[1320px]">
      <PageHeader
        title="Model arena"
        description="Bring your own dataset and models – CSV tables, text or images; ONNX, scikit-learn or local LLMs – and compare them with built-in algorithms: chosen on a validation split, reported once on a held-out test split, on this computer."
      />
      <Tabs tabs={TABS} active={tab} onChange={setTab} labels={TAB_LABELS} />
      {tab === "compare" && <CompareForm datasetId={datasetId} setDatasetId={setDatasetId} goTo={setTab} />}
      {tab === "datasets" && (
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <DatasetUpload onCreated={(d) => { setDatasetId(d.id); }} />
          <div className="space-y-3 min-w-0">
            {datasets.isError && <ErrorNotice error={datasets.error} />}
            {datasets.data && !datasets.data.datasets.length && <Empty title="No datasets yet">Your datasets stay in <code>backend/data/arena</code> on this computer.</Empty>}
            {datasets.data?.datasets.map((d) => <DatasetCard key={d.id} d={d} onCompare={() => { setDatasetId(d.id); setTab("compare"); }} />)}
          </div>
        </div>
      )}
      {tab === "models" && (
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <ModelUpload />
          <div className="space-y-6 min-w-0">
            <Backbones />
            <Panel title="Uploaded models" bodyClass="p-0"><ModelList /></Panel>
          </div>
        </div>
      )}
      {tab === "runs" && <RecentRuns />}
    </div>
  );
}

