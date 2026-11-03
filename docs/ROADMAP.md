# Product Roadmap

What it takes to turn Local AI Lab from a portfolio project into a product that people install and use. Grouped by the role that would own the work; ✅ = done (v1.1 or v1.2).

**Positioning:** *An offline-first workbench to compare models on your own data and hardware with an honest protocol – choose on validation, report once on test – and to run local models and agents with permissions you control.*

## Phase 1 – Adoption (anyone can install and use it)

| Item | Role | Status |
|---|---|---|
| One-click Windows installer (.exe/MSI with embedded Python and the built UI) or a Tauri desktop app with tray icon; detect/install Ollama | Software engineer | ⬜ |
| Onboarding: system check → models that fit, with licences → download from the UI with a progress bar | Software engineer | ✅ Setup page (v1.2); a guided first-run wizard ⬜ |
| Update script that keeps data, `.env` and the virtual environment | Software engineer | ✅ `scripts\update_windows.bat` |
| Alembic migrations, backup/restore of `backend\data` | Software engineer | ✅ v1.2 (pre-migration copy, verified backups, staged restore) |
| Custom benchmark-suite builder in the UI (prompts + expected answers + evaluator) | AI/ML engineer | ✅ v1.2 with version history and evaluator try-it |
| Smarter evaluators: number words, answer markers, negation, conflicts, types; calibration set | AI/ML engineer | ✅ (independent human labels for the calibration set ⬜) |
| Signed releases, semantic versioning, changelog, GitHub Releases | Software engineer | 🟡 changelog + semver ✅; signing and releases ⬜ |

## Phase 2 – Differentiator (compare anything, honestly)

| Item | Role | Status |
|---|---|---|
| Bring your own dataset: CSV (tabular/text), image ZIP; profile and leakage/duplicate/imbalance checks | AI/ML engineer | ✅ Model Arena |
| Bring your own model: ONNX (any framework), scikit-learn joblib; label mapping and preprocessing | AI/ML engineer | ✅ |
| Transfer learning with pretrained CNNs (MobileNetV2, EfficientNet-Lite4, uploaded extractors) | AI/ML engineer | ✅ |
| Statistics: train/validation/test protocol, paired bootstrap, TOST equivalence, Holm, cluster bootstrap, Pareto front, recommendation with constraints | AI/ML engineer | ✅ v1.2 (validated by simulation) |
| Hyper-parameter tuning | AI/ML engineer | ✅ randomised search (Optuna: ⬜) |
| Export the trained winner | MLOps | ✅ joblib bundle (ONNX export via skl2onnx: ⬜) |
| Fine-tune small transformers / CNNs in the app (GPU optional) | AI/ML engineer | ⬜ |
| RAG evaluation: retrieval recall@k, MRR, nDCG, false retrieval | AI/ML engineer | ✅ v1.2 (citation correctness and faithfulness ⬜) |
| Quantization study (Q4 vs Q8), context-length sweep, GPU/VRAM metrics | AI/ML engineer | ⬜ |
| Dataset versioning with diffs, multiple splits / k-fold test evaluation | MLOps | 🟡 split hash, group/time splits, lineage of runs per dataset ✅; repeated splits ⬜ |

## Phase 3 – Agents and automation

| Item | Role | Status |
|---|---|---|
| Agent benchmark: tool choice, arguments, order, restraint, safety, recovery, efficiency | Agentic AI engineer | ✅ Agent Arena |
| Native tool calling + prompted fallback, traces, playground | Agentic AI engineer | ✅ |
| User-defined tools and tasks (mocked) | Agent developer | ✅ custom suites |
| Real tools behind human approval (file workspace, webhooks), allow/ask/deny policy, audit | Agentic AI engineer | ✅ v1.2 Work Agent (OpenAPI tools ⬜) |
| MCP server (expose “search my documents”, “run benchmark”) and MCP client (use external tool servers) | Agent developer | ⬜ (next: the Work Agent's policy and audit layer would apply to MCP tools too) |
| OpenAI-compatible local API (`/v1/chat/completions`) with API keys, so other apps use Local AI Lab as a private gateway | AI automation | ✅ v1.2 (scoped tokens) |
| Scheduled jobs with quality gates and alerts; signed webhooks | AI automation | ✅ v1.2 (watch folder, benchmark-on-pull ⬜) |
| CLI (`lal bench …`, `lal arena …`) for CI pipelines with gate exit codes | AI automation / MLOps | ✅ v1.2 |
| Multi-turn agent tasks with a simulated user; LLM-judge for open-ended tasks (clearly separated) | Agentic AI engineer | ⬜ |

## Platform quality (continuous)

| Item | Role | Status |
|---|---|---|
| Automated tests (249 backend), Python 3.11–3.13, CI incl. Windows, smoke test of every feature | Software engineer | ✅ (Windows CI job not yet run) |
| Frontend tests (Vitest + Playwright E2E) | Software engineer | ✅ v1.2 (9 unit, 31 browser) |
| RBAC (admin/user/viewer), login rate limiting, password reset, audit log, HTTPS guide | Software engineer | 🟡 admin/user, scoped tokens, rate limiting, audit log ✅; viewer role, password reset, HTTPS guide ⬜ |
| Optional encryption at rest (SQLCipher / encrypted data folder) | Software engineer | ⬜ |
| OpenTelemetry traces and Prometheus metrics (opt-in, local) | MLOps | ⬜ |
| Plugin system (evaluators, model runners, agent tools as Python entry points) | Software engineer | ⬜ |
| Dark/light theme, Urdu UI, accessibility audit, documentation website, demo video | Software engineer | ⬜ |
