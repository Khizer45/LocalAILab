# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org/): a new minor version may add
database tables and columns (migrated automatically), a major version may change how results are computed in a way
that makes old and new results incomparable.

## 1.2.0 – 2026-10-13

Theme: **honest results, recoverable installations, safe automation.**

### Changed – results are computed differently

- **Model Arena protocol v2.** Models are chosen on a **validation** split and the chosen model is scored **once** on
  the test split (v1.1 chose and scored on the same test rows). Paired bootstrap with an equivalence margin (TOST)
  and Holm correction replaces “≈ tie”: verdicts are *equivalent*, *worse* or *inconclusive*. Recommendations are
  *clear* or *provisional* (with a shortlist) and can respect latency, size, memory and quality limits. Old runs are
  labelled *protocol v1*; old datasets get a validation split from their training rows (test rows unchanged).
- **Benchmark quality** is split into *effective quality* (failures count as 0, with a 95 % CI), *conditional
  quality* and *completion rate*; models get paired verdicts against the leader.
- **Evaluators v2** (versioned): negation-aware keywords, forbidden terms, tiered numeric answers that reject
  conflicting answers, ambiguous labels fail, JSON value types, abstention limits, spec validation.
- **Agent scoring v2**: a severe safety violation is a hard fail (score 0); unrequested risky actions count as
  violations; alternative valid tool paths are accepted; repaired protocol replies are counted.
- PDF text extraction uses **pypdf** (BSD-3-Clause) instead of PyMuPDF (AGPL-3.0); PyMuPDF remains optional.
- `qwen2.5:3b` (Qwen Research License, non-commercial) removed from hardware recommendations.

### Added

- Model Arena: group and time splits, split fingerprint and `split.csv` export, near-duplicate and target-leakage
  checks, uploaded-model provenance (declared or from ONNX metadata), dataset lineage, sandboxed worker for uploaded
  models, constraints and objective, final unbiased estimate, winner's-bias warning.
- Notebooks (`notebooks/`) to train scikit-learn and PyTorch models on exactly the exported training rows and export
  ONNX with provenance metadata (Jupyter or Google Colab).
- Benchmarks: suite builder with version history, evaluator try-it, frozen suite snapshot per run, integrity check,
  simulator flag, first-visible-answer TTFT, shuffled order, evaluator calibration set and agreement command.
- Documents: retrieval evaluation (Recall@k, hit@k, MRR, nDCG, answer-in-context, false retrieval).
- Agent Arena: `agent-safety` suite (16 tasks), safety violation rate with Wilson interval, protocol compliance,
  mode-mismatch warning.
- **Work Agent** with workspace-confined tools, allow / ask / deny policy and interactive approvals.
- **Setup page**: system check, model catalogue with licence, commercial-use flag and memory fit, download with
  progress, delete.
- **Alembic migrations** with a pre-migration database copy; **backups** with checksums, verify, staged restore.
- **Administration**: scoped, expiring API tokens, revoke-all, webhooks, audit log.
- **Automation**: OpenAI-compatible `/v1` API, `lal` CLI with CI gate exit codes, schedules with gates and alerts,
  HMAC-signed webhooks, experiment bundles (full / metadata scope) with verify, import and re-run.
- Security: sign-in rate limiting, remote-access guard, DNS-rebinding/CSRF guard, password change revokes other
  sessions.
- Frontend tests (Vitest, Playwright), CI on Windows and Python 3.11–3.13, pip-audit, npm audit, licence report,
  Dependabot; `scripts/validate_statistics.py`; smoke test extended to 12 checks.
- Docs: `AUTOMATION.md`, `LEGAL_AND_LICENSING.md`, updated manuals and methodology.

### Fixed

- **ONNX Runtime sent usage telemetry** (`mobile.events.data.microsoft.com`) from native code that the socket guard
  cannot see; `ORT_DISABLE_TELEMETRY=1` is now forced at start-up.
- Backup times were shown five hours off in time zones east of UTC.
- `scripts\lal.bat` changed the caller's working folder and lost the exit code.
- Form labels were not connected to their inputs (accessibility).

### Upgrade notes

Make a backup, unzip over your folder, run `scripts\update_windows.bat`. Re-run Model Arena comparisons you want to
quote. Re-index PDFs if you need identical text extraction across documents.

## 1.1.0 – 2026-10-10

- Model Arena: bring your own dataset (CSV, image ZIP) and models (ONNX, joblib, local LLMs), 12 built-in
  algorithms, transfer learning with MobileNetV2 / EfficientNet-Lite4, statistics and a recommendation.
- Agent Arena: tool-use benchmark (native and prompted), custom suites, playground.
- Number-word and answer-marker support in the numeric evaluator; shared job lock; update script.

## 1.0.0

- Offline assistant, private document Q&A with citations, benchmarking with deterministic evaluators, ML baseline,
  MLflow, Offline-Only mode with HTTP and socket guards, optional multi-user login, Docker, CI.
