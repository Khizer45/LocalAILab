# Datasets

All datasets in this folder were written for Local AI Lab. They contain no personal or real customer/patient data.

## Benchmark suites (`benchmark/*.json`)

| File | id | Version | Cases | Evaluation criteria |
|---|---|---|---|---|
| `quick_v1.json` | `quick` | 1.0.0 | 8 | one case per category, see `core` |
| `core_v1.json` | `core` | 1.0.0 | 32 | QA: exact/keyword match · reasoning: final number (± tolerance) · summarization: required keywords (≥ 2 of 3) + ≤ 35 words + ROUGE-L ≥ 0.15 · instruction following: regex / JSON keys+values / bullet count / word limits · extraction: regex or JSON · classification: chosen label |
| `rag_grounding_v1.json` | `rag-grounding` | 1.1.0 | 6 | *Context grounding & abstention (no retrieval)*. Answerable: keyword/number from the given context; unanswerable: a short response that says the answer is not in the context and names no amount |

Schema: `id`, `name`, `version`, `description`, optional `system`, `default_max_tokens`, `cases[]` (`id`, `category`, `prompt`, `reference`, optional `system`, `max_tokens`, `evaluator`). See `docs/BENCHMARK_METHODOLOGY.md` for every evaluator type.

Rules for editing:
1. Never change a case without bumping `version` – results are tied to the file hash.
2. `reference` must pass the case's own evaluator (`pytest tests/test_benchmark.py::test_all_suites_are_valid_and_versioned`).
3. Prompts state the expected format explicitly (“Reply with the number only”), because the evaluators check format.

## Intent dataset (`intent/intent_v1.csv`)

- 150 short customer-service messages, 25 per class: `billing`, `technical_support`, `account_access`, `appointment`, `feedback`, `general_info`.
- Columns: `text`, `label`. Version 1.0.0.
- Split (in code, `app/ml/intent_data.py`): stratified, seed 42 → train 104 / validation 23 / test 23.
- The test split doubles as the `intent-classification` benchmark suite, so the scikit-learn baseline and LLMs are scored on identical messages.
- Known ambiguity: some messages could reasonably belong to two classes (e.g. a complaint about billing). Labels follow the main intent; expect a ceiling below 100% for any classifier.

## Agent suites (`agent/*.json`)

| File | id | Version | Tasks | Purpose |
|---|---|---|---|---|
| `agent_core_v1.json` | `agent-core` | 1.0.0 | 23 | tool use in six categories: single tool, argument extraction, multi-step, no tool needed, safety (forbidden actions, injected instructions in text and tool results), error recovery |
| `agent_quick_v1.json` | `agent-quick` | 1.0.0 | 6 | one task per category |

Tools are mocks with invented data (orders, weather, prices) or side-effect-free built-ins. Format, matchers and checks: [`agent/README.md`](agent/README.md).

## Model Arena

User datasets are not stored here – they are uploaded in the app and kept in `data/arena`. Synthetic sample files are in `samples/arena` at the repository root.
