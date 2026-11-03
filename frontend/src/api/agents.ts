// Types for the Agent Arena API (/api/agents).

export interface AgentSuiteSummary {
  id: string;
  name: string;
  version: string;
  description: string;
  task_count: number;
  tool_count: number;
  categories: Record<string, number>;
  tools: string[];
  sha256: string;
  source: string;
}

export interface AgentSummary {
  model: string;
  tasks: number;
  success_rate: number | null;
  success_rate_ci?: [number, number] | null;
  hard_fails?: number;
  safety_tasks?: number;
  safety_violation_rate?: number | null;
  safety_violation_rate_ci?: [number, number] | null;
  protocol_compliance_rate?: number | null;
  score: number | null;
  tool_selection_rate: number | null;
  argument_accuracy: number | null;
  order_rate: number | null;
  no_tool_rate: number | null;
  safety_rate: number | null;
  final_answer_rate: number | null;
  completion_rate: number | null;
  avg_steps: number | null;
  avg_tool_calls: number | null;
  invalid_call_rate: number | null;
  format_errors: number;
  errors: number;
  timeouts: number;
  modes: string[];
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  tokens_per_s_mean: number | null;
  peak_ram_used_mb: number | null;
  by_category: Record<string, { n: number; success_rate: number; score: number }>;
}

export interface ToolCall {
  name: string;
  arguments: unknown;
}

export type TraceEvent =
  | { type: "start"; mode: string; model: string; tools: string[] }
  | { type: "model"; step: number; ms: number; content: string; tool_calls: ToolCall[]; prompt_tokens?: number; output_tokens?: number; tokens_per_s?: number; parsed?: string; thinking_chars?: number }
  | { type: "tool"; step: number; name: string; arguments: unknown; result: unknown; ms: number; valid: boolean; errors: string[]; unknown: boolean; risky: boolean }
  | { type: "final"; answer: string; stop_reason: string; totals: Record<string, number | null>; error?: string }
  | { type: "error"; code?: string; message: string; hint?: string }
  | { type: "note"; message: string };

export interface CheckResult {
  score: number;
  passed: boolean;
  [key: string]: unknown;
}

export interface AgentResultRow {
  id: number;
  run_id: number;
  model: string;
  task_id: string;
  category: string;
  repeat_index: number;
  status: string;
  mode: string;
  success: boolean;
  score: number;
  hard_fail?: boolean;
  safety_violations?: { call: number; tool: string; severity: string; rule: string }[] | null;
  protocol_ok?: boolean | null;
  checks: Record<string, CheckResult> | null;
  steps: number;
  tool_calls: number;
  invalid_calls: number;
  format_errors: number;
  stop_reason: string | null;
  final_answer: string;
  total_ms: number | null;
  tokens_per_s: number | null;
  error: string | null;
  trace?: TraceEvent[];
}

export interface AgentRun {
  id: number;
  name: string;
  suite_id: string;
  suite_version: string;
  suite_hash: string;
  models: string[];
  config: { mode: string; max_steps: number; temperature: number; seed: number | null; repeats: number; timeout_s: number; task_ids: string[] };
  status: string;
  progress_done: number;
  progress_total: number;
  current_item: string | null;
  error: string | null;
  environment: Record<string, unknown> | null;
  model_info: Record<string, { supports_native_tools?: boolean | null; mode?: string; fallback?: string; parameter_size?: string | null; quantization?: string | null; error?: string }> | null;
  mlflow_run_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  summary: AgentSummary[];
  warnings?: string[];
  simulated?: boolean;
  results?: AgentResultRow[];
  tasks?: Record<string, { prompt: string; category: string; reference: string; expect: unknown }>;
  check_labels?: Record<string, string>;
}
