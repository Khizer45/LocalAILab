export interface ModelInfo {
  id: string;
  name: string;
  provider: "local" | "cloud";
  location: string;
  size_bytes: number | null;
  digest: string | null;
  parameter_size: string | null;
  quantization: string | null;
  family: string | null;
  capabilities: string[];
  is_embedding: boolean;
  allowed: boolean;
  blocked_reason: string | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
  status: {
    local: { reachable: boolean; base_url: string; version?: string; error?: string; hint?: string };
    cloud: { enabled: boolean; reachable?: boolean; error?: string; note?: string };
  };
  default_model: string;
  embedding_model: string;
  offline_only: boolean;
}

export interface RuntimeSettings {
  offline_only: boolean;
  allowed_models: string[];
  default_model: string;
  embedding_model: string;
  default_system_prompt: string;
  max_upload_mb: number;
  max_context_tokens: number;
  max_output_tokens: number;
  request_timeout_s: number;
  benchmark_timeout_s: number;
  retrieval_top_k: number;
  rag_min_score: number | null;
  allow_documents_with_cloud: boolean;
}

export interface SettingsResponse {
  runtime: RuntimeSettings;
  static: Record<string, string | number | boolean | string[]>;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  thinking: string | null;
  model: string | null;
  status: string;
  metrics: import("./client").Metrics | null;
  sources: import("./client").Source[] | null;
  created_at: string;
}

export interface Conversation {
  id: number;
  title: string;
  model: string;
  system_prompt: string;
  options: Record<string, number>;
  use_documents: boolean;
  created_at: string;
  updated_at: string;
  messages?: Message[];
}

export interface DocumentInfo {
  id: number;
  filename: string;
  file_type: string;
  size_bytes: number;
  status: "pending" | "processing" | "ready" | "failed";
  error: string | null;
  page_count: number;
  chunk_count: number;
  char_count: number;
  flagged_chunks: number;
  embedding_backend: string | null;
  embedding_model: string | null;
  created_at: string;
  needs_reindex?: boolean;
}

export interface SuiteSummary {
  id: string;
  name: string;
  version: string;
  description: string;
  case_count: number;
  categories: Record<string, number>;
  sha256: string;
  source: string;
}

export interface ModelSummary {
  model: string;
  results: number;
  ok: number;
  errors: number;
  timeouts: number;
  quality_score: number | null;
  cases?: number;
  effective_quality?: number | null;
  effective_quality_ci?: [number, number] | null;
  conditional_quality?: number | null;
  completion_rate?: number | null;
  completion_rate_ci?: [number, number] | null;
  pass_rate: number | null;
  pass_rate_ci?: [number, number] | null;
  by_category: Record<string, { n: number; quality_score: number | null; pass_rate: number | null }>;
  ttft_answer_ms_p50?: number | null;
  latency_n?: number;
  p95_unstable?: boolean;
  ttft_ms_p50: number | null;
  ttft_ms_p95: number | null;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  tokens_per_s_mean: number | null;
  tokens_per_s_median: number | null;
  tokens_source: string | null;
  output_tokens_mean: number | null;
  peak_ram_used_mb: number | null;
  avg_cpu_pct: number | null;
  runtime_rss_mb_peak: number | null;
  model_mem_mb: number | null;
  score_std_across_repeats: number | null;
}

export interface BenchmarkRun {
  id: number;
  name: string;
  suite_id: string;
  suite_version: string;
  suite_hash: string;
  models: string[];
  config: Record<string, unknown>;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress_done: number;
  progress_total: number;
  current_item: string | null;
  error: string | null;
  environment: Record<string, unknown> | null;
  model_info: Record<string, Record<string, unknown>> | null;
  mlflow_run_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  summary: ModelSummary[];
  comparison?: BenchmarkComparison;
  simulated?: boolean;
  integrity?: { expected_results: number; stored_results: number; duplicates: number; complete: boolean } | null;
  results?: BenchmarkResult[];
}

export interface BenchmarkComparison {
  leader: string | null;
  margin: number;
  unit?: string;
  correction?: string;
  resamples?: number;
  comparisons: Record<string, {
    n_cases: number; delta?: number; ci95?: [number, number]; p_diff?: number; p_equivalence?: number;
    p_diff_holm?: number | null; p_equivalence_holm?: number | null; verdict: "equivalent" | "worse" | "inconclusive" | "not_tested";
  }>;
}

export interface BenchmarkResult {
  id: number;
  model: string;
  case_id: string;
  category: string;
  repeat_index: number;
  status: string;
  response: string;
  thinking_chars: number;
  score: number;
  passed: boolean;
  eval_details: { score: number; passed: boolean; method: string; details: Record<string, unknown> } | null;
  ttft_ms: number | null;
  ttft_answer_ms?: number | null;
  evaluator_version?: string | null;
  total_ms: number | null;
  load_ms: number | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  tokens_per_s: number | null;
  tokens_source: string;
  peak_ram_used_mb: number | null;
  avg_cpu_pct: number | null;
  runtime_rss_mb: number | null;
  model_mem_mb: number | null;
  error: string | null;
}
