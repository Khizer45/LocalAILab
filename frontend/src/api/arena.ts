// Types for the Model Arena API (/api/arena).

export type DataKind = "tabular" | "text" | "image";
export type Task = "classification" | "regression";

export interface AlgorithmInfo {
  id: string;
  name: string;
  family: string;
  tasks: Task[];
  kinds: DataKind[];
  speed: "fast" | "medium" | "slow";
  description: string;
  tunable: boolean;
  available: boolean;
  requires: string | null;
  default: boolean;
}

export interface BackboneInfo {
  id: string;
  name: string;
  size_mb: number;
  feature_dim: number;
  source: string;
  notes: string;
  installed: boolean;
  url: string;
  file: string;
}

export interface ArenaCatalog {
  algorithms: AlgorithmInfo[];
  backbones: BackboneInfo[];
  normalizations: Record<string, string>;
  limits: { max_upload_mb: number; max_rows: number; max_columns: number; max_images: number };
}

export interface ColumnPreview {
  name: string;
  type: string;
  missing: number;
  unique: number;
  examples: string[];
}

export interface DatasetPreview {
  format: "csv" | "zip";
  rows?: number;
  columns?: ColumnPreview[];
  suggested_target?: string;
  suggested_task?: Task;
  suggested_kind?: DataKind;
  head?: Record<string, string | null>[];
  images?: number;
  classes?: Record<string, number>;
  split_source?: string;
  warnings?: string[];
}

export interface ArenaDataset {
  id: number;
  name: string;
  kind: DataKind;
  task: Task;
  target: string | null;
  text_column: string | null;
  columns: { features?: string[]; ignored?: string[]; types?: Record<string, string> };
  labels: string[];
  n_rows: number;
  n_train: number;
  n_val: number;
  n_test: number;
  test_size: number;
  val_size: number;
  seed: number;
  split_source: string;
  group_column: string | null;
  time_column: string | null;
  split_sha256: string | null;
  filename: string;
  size_bytes: number;
  sha256: string;
  profile: Record<string, any>;
  warnings: string[];
  created_at: string;
}

export interface ArenaDatasetDetail extends ArenaDataset {
  algorithms: AlgorithmInfo[];
  models: { id: number; name: string; role: string; reason: string | null }[];
}

export interface TensorInfo {
  name: string;
  shape: (number | string)[];
  type: string;
}

export interface ArenaModel {
  id: number;
  name: string;
  filename: string;
  format: "onnx" | "joblib";
  kind: DataKind;
  task: string;
  role: "predictor" | "feature_extractor";
  size_bytes: number;
  sha256: string;
  config: { labels?: string[] | null; preprocess?: Record<string, string | number>; feature_columns?: string[] };
  inspection: {
    inputs?: TensorInfo[];
    outputs?: TensorInfo[];
    parameters?: number | null;
    classes?: string[] | null;
    feature_dim?: number;
    class?: string;
  };
  trusted: boolean;
  provenance: Provenance | null;
  created_at: string;
}

export type TrainingData = "unknown" | "train_split" | "external";

export interface Provenance {
  training_data: TrainingData;
  note?: string | null;
  dataset_sha256?: string | null;
  split_sha256?: string | null;
  trained_on?: string | null;
  source?: "metadata" | "user";
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
  unparseable?: number[] | null;
}

export interface ArenaMetrics {
  n: number;
  accuracy?: number;
  balanced_accuracy?: number;
  f1_macro?: number;
  f1_weighted?: number;
  precision_macro?: number;
  recall_macro?: number;
  roc_auc?: number | null;
  unparseable?: number;
  per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix?: ConfusionMatrixData;
  accuracy_ci?: [number, number];
  f1_macro_ci?: [number, number];
  r2?: number;
  mae?: number;
  rmse?: number;
  median_ae?: number;
  mape?: number | null;
  r2_ci?: [number, number];
  mae_ci?: [number, number];
  fit_ms: number | null;
  latency_ms_per_example: number | null;
  latency_ms_single_p50: number | null;
  latency_ms_single_p95: number | null;
  throughput_per_s: number | null;
  model_size_bytes: number | null;
  parameters?: number | null;
  peak_process_rss_mb?: number;
  memory_increase_mb?: number;
  feature_ms_per_example?: number;
  tokens_per_s_mean?: number | null;
}

export interface ArenaResult {
  id: number;
  key: string;
  source: "baseline" | "builtin" | "uploaded" | "llm";
  name: string;
  status: string;
  metrics: ArenaMetrics | null;
  val_metrics?: ArenaMetrics | null;
  details: Record<string, any> | null;
  error: string | null;
}

export interface PairTest {
  test: string | null;
  p_value: number | null;
  n: number;
  significantly_worse?: boolean;
  note?: string;
}

export type Verdict = "equivalent" | "worse" | "better" | "inconclusive" | "not_tested";

export interface PairedComparison {
  n: number;
  a: string;
  delta?: number;
  ci95?: [number, number] | null;
  ci90?: [number, number] | null;
  p_diff?: number | null;
  p_equivalence?: number | null;
  p_diff_holm?: number | null;
  p_equivalence_holm?: number | null;
  margin?: number;
  verdict: Verdict;
  note?: string;
  supplementary?: { test?: string; p_value?: number | null };
}

export interface Selection {
  split: "validation" | "test";
  primary_metric: string;
  primary_label: string;
  higher_is_better: boolean;
  alpha: number;
  objective: "efficiency" | "quality";
  constraints: Record<string, number>;
  resampling: { method: string; n_boot: number; seed: number; [k: string]: unknown };
  comparisons: Record<string, PairedComparison>;
  excluded: Record<string, string[]>;
  ranking: string[];
  leader?: string;
  margin?: number;
  margin_rule?: string;
  family_size?: number;
  correction?: string;
  status: "clear" | "provisional" | "none";
  shortlist: string[];
  warnings: string[];
}

export interface FinalEstimate {
  key: string;
  name: string;
  split: "test";
  metric: string;
  label: string;
  value: number | null;
  ci: [number, number] | null;
  n: number | null;
  unbiased: boolean;
  note: string;
}

export interface ArenaSummary {
  protocol?: "v2";
  primary_metric: "f1_macro" | "r2" | string;
  primary_label: string;
  ranking: string[];
  best?: string;
  tests?: Record<string, PairTest>; // protocol v1 only
  pareto?: string[];
  recommendation: {
    key: string; name: string; reason: string;
    tied_with_best?: string[]; // v1
    status?: "clear" | "provisional"; equivalent_to_leader?: string[]; caveats?: string[]; // v2
  } | null;
  warnings?: string[];
  note?: string;
  task: Task;
  kind: DataKind;
  labels: string[];
  test_size: number;
  // protocol v2
  higher_is_better?: boolean;
  split?: { n_train: number; n_val: number; n_test: number; sha256: string | null; selection_split: "val" | "test" };
  selection?: Selection;
  final?: FinalEstimate | null;
  test_view?: { ranking: string[]; leader: string | null; comparisons: Record<string, PairedComparison> };
  limitations?: string[];
  verdict_text?: Record<Verdict, string>;
}

export interface ArenaRun {
  id: number;
  name: string;
  dataset_id: number | null;
  dataset: { id: number; name: string; kind: DataKind; task: Task; n_rows: number; n_train: number; n_val?: number; n_test: number; seed: number; split_source: string; split_sha256?: string | null; labels: string[]; sha256: string; target: string | null; filename: string };
  config: { cv_folds: number; tune: boolean; llm_max_examples: number; algorithms: string[]; uploaded_models: number[]; llm_models: string[]; image_features: string[]; objective?: string; equivalence_margin?: number | null; [k: string]: unknown };
  protocol?: "v1" | "v2";
  contenders: { key: string; source: string; name: string }[];
  status: string;
  progress_done: number;
  progress_total: number;
  current_item: string | null;
  error: string | null;
  environment: Record<string, unknown> | null;
  summary: ArenaSummary | null;
  has_model_file: boolean;
  mlflow_run_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  results?: ArenaResult[];
}
