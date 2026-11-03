# Benchmark Methodology

This document defines what Local AI Lab measures, how, and what the numbers do **not** mean.

## 1. Goal

Compare locally hosted language models **on the same tasks, under the same conditions, with the same scoring rules**, and record enough context (versions, hardware, configuration) that a result can be reproduced or questioned.

## 2. Test suites

All suites live in `backend/datasets/benchmark/` (JSON) and carry a semantic `version`. Since v1.2 every run also stores a **frozen snapshot** of the cases it used (prompts, references, evaluator specs and evaluator versions) and their SHA-256 (`cases_sha256`), so a result stays tied to the exact questions and rules even if the suite file or an evaluator changes later. Users can write their own suites in the UI (**Benchmarks → Test suites**); every save is a new, immutable version.

| Suite | Version | Cases | Categories | Purpose |
|---|---|---|---|---|
| `quick` | 1.0.0 | 8 | QA, reasoning, instruction following, summarization, extraction, classification | fast sanity check |
| `core` | 1.0.0 | 32 | QA (8), reasoning (6), summarization (4), instruction following (8), extraction (4), classification (2) | main comparison |
| `rag-grounding` | 1.1.0 | 6 | grounded QA (4), abstention (2) | does the model stay within given context and admit missing answers |
| `intent-classification` | 1.0.0 (dataset) | 23 | intent classification | LLM zero-shot vs scikit-learn baseline on the identical test split |

Case content was written for this project (no copyrighted benchmark text). Every case has a `reference` answer, and an automated test checks that each reference passes its own evaluator, so the rules are internally consistent.

## 3. Scoring (task success)

Each response is scored by a **deterministic evaluator** defined in the case – no model judges another model.

| Evaluator | Rule | Typical use |
|---|---|---|
| `exact_match` | normalised first line equals an accepted answer (lower-case, no punctuation/articles) | short factual answers |
| `contains_all` / `contains_any` | required keywords present (threshold = share required) | names, multi-part facts |
| `numeric` | the final answer equals the target (± tolerance): the number after an explicit marker (“Answer:”, “the answer is”, “####”) if present, otherwise the **last** number; English number words count (“seven” = 7) | arithmetic, prices |
| `regex` | pattern matches (optionally must *not* match) | formats, yes/no, casing |
| `word_limit` | word count within bounds | length instructions |
| `bullet_count` | exact number of bullet lines and nothing else | list formatting |
| `json_object` | valid JSON object with required keys / expected values; partial credit 0.4 + 0.6 × share of checks passed | structured output, extraction |
| `json_equals` | parsed JSON equals target (options: ignore order/case, strip characters) | arrays, exact structures |
| `label` | which allowed label the response chose (first line, else earliest mention) | classification |
| `token_f1`, `rouge_l` | token-overlap F1 / longest-common-subsequence F-measure against a reference | summarization (lenient threshold) |
| `abstain` | response says the answer is not in the context | hallucination resistance |
| `all_of` | weighted average of sub-checks; passes only if all pass | summaries (keywords + length + ROUGE-L) |

- **Score** ∈ [0, 1] (partial credit where defined); **passed** = rule fully satisfied.
- **Pass rate** = share of results that passed.

**Evaluator changes in v1.2 (version 2 of the affected evaluators).** Each evaluator has a version
(`EVALUATOR_VERSIONS`), recorded with every result, so scores from different evaluator versions are never mixed
silently.

| Problem found in v1.1 | v1.2 rule |
|---|---|
| “The refund is **not** possible within 14 days” passed a keyword check for “14 days” | `contains_*` are negation-aware (a keyword that follows a negation such as *not, no, never, without, cannot* by at most two words does not count); `forbidden` terms fail the check |
| “The answer is 12 … Final answer: 14” – which number counts? | `numeric` uses tiers: an explicit *final answer* marker beats “the answer is”, which beats the last number; conflicting numbers at the same tier fail; dates and times are removed before numbers are read |
| A label answer like “billing or technical” passed | `label` fails on ambiguous answers that name two allowed labels, and accepts an explicit marker (“Label: billing”) |
| JSON with `"age": "29"` passed a numeric field | `json_object` can check value **types** |
| A long, hedged non-answer passed `abstain` | `abstain` supports `max_words` and `forbidden` (e.g. a made-up figure) |
| A typo in a regex only failed at run time | `validate_spec` checks every spec when a suite is loaded or saved |

**How often are the evaluators wrong?** `backend/datasets/evaluator_calibration/calibration_v1.jsonl` holds 50
responses to real cases with a pass/fail judgement; `python -m app.cli evaluators agreement` compares them with the
evaluators. Current result: **88 % agreement, Cohen's κ = 0.73, 3 false passes and 3 false fails** (weakest:
`abstain`, `contains_any`, `exact_match`). The 50 judgements are **drafts written by the developer with AI
assistance, not independent human labels** – the README in that folder explains how to turn them into a real
calibration set before quoting these numbers. **Benchmarks → Test suites → Try an evaluator** lets anyone check a
rule against sample answers before using it.
- Reasoning blocks (`<think>…</think>` or Ollama's separate `thinking` field) are removed before scoring.

**Limits.** These rules check task success under a stated format. `numeric` understands number words and answer markers (a real run had “There are seven continents” marked wrong in v1.0), but a response whose last number is not the answer and that has no marker (“11 + 4 = 15 years old in 3 years”) still fails. This is intentional – instruction following is part of the test – but it means scores favour models that follow format instructions. Open-ended quality (style, helpfulness) is not measured.

## 4. Performance metrics

| Metric | Definition | Provenance |
|---|---|---|
| TTFT | time from sending the request to the first non-empty content or thinking chunk | **measured** by Local AI Lab (`time.perf_counter`) |
| TTFT (answer) | time to the first *visible* answer token – differs from TTFT for thinking models | **measured** (v1.2) |
| Total latency | time from request to final chunk | **measured** |
| Load time | model load time reported in Ollama's final chunk (`load_duration`) | **runtime-reported** |
| Prompt / output tokens | `prompt_eval_count` / `eval_count` | **runtime-reported** |
| Tokens/s | `eval_count / eval_duration` (generation only, excludes prompt processing) | **runtime-reported**; if missing: (characters/4) / (total − TTFT), marked **estimated** |
| Peak RAM used | max system memory in use, sampled every 250 ms during the request | **measured** (system-wide, includes other processes) |
| Avg CPU | mean system CPU %, same sampler | **measured** |
| Runtime RSS | resident memory of local `ollama*` processes | **measured** (only when visible, not in Docker) |
| Model memory | loaded model size from `/api/ps` | **runtime-reported** |
| Percentiles | p50 (median) and p95 over successful results per model; p95 from fewer than 20 samples is marked unstable | computed with NumPy |
| Repeat spread | mean standard deviation of a case's score across repeats | computed |

### 4.1 Quality numbers and their uncertainty (v1.2)

v1.1 reported one “quality” number that silently mixed two things: how good the answers were and how often the
model failed to answer. v1.2 separates them:

| Number | Definition |
|---|---|
| **Effective quality** | mean score over *all* cases; an error or timeout counts as 0. 95 % CI: bootstrap over **cases** (repeats of a case are averaged first, so repeats do not shrink the interval artificially) |
| **Conditional quality** | mean score over the calls that completed – read it together with the completion rate |
| **Completion rate** | share of calls without error or timeout, with a 95 % Wilson interval |
| **Pass rate** | share of results that met the evaluator's pass rule, 95 % CI by case bootstrap |

**Comparing models.** The model with the highest effective quality is the *leader*. Every other model is compared
with it **case by case** (paired): Δ = leader − model per case, bootstrap over cases (2 000 resamples), a two-sided
p-value for “no difference” and a TOST p-value for equivalence within ±5 percentage points, both Holm-corrected over
the models compared. Verdicts: **equivalent**, **worse**, **inconclusive** (the suite is too small to tell – this is
*not* a tie) or **not tested** (fewer than 5 shared cases).

**Integrity.** After a run, the number of stored results is checked against models × cases × repeats; a run with
missing or duplicate results is marked incomplete in the UI and the report. Runs made with the Ollama simulator are
flagged `simulated` and every report says so in a banner.

## 5. Fairness and reproducibility controls

- Same suite file (hash recorded), same prompts and system prompt for every model.
- Defaults: temperature **0**, seed **42**, context **2048**, per-case `max_tokens`; all stored in the run config.
- Optional **warm-up** request per model so cold-load time does not inflate the first case's latency (cold-load is recorded separately in `model_info.warmup`).
- **One run at a time** – benchmark, Model Arena and Agent Arena workers share one job lock, so runs never compete for CPU/GPU.
- Recorded per run: OS, CPU count, RAM, GPUs, Python and app version, Ollama version; per model: digest, parameter size, quantization.
- Configurable **repeats** to measure variance; per-case **timeout** with timeouts recorded, not hidden.
- Optional **shuffled order** of models and cases (v1.2), so slow drift such as thermal throttling or a background
  program does not always hit the same model; the order is recorded.
- Results are exported as JSON/CSV/Markdown, logged to MLflow, and can be packed into a verifiable **experiment
  bundle** (manifest with SHA-256 of every file; *full* or *metadata-only* scope) that another installation can
  verify, import or re-run with the same cases.

## 6. Threats to validity

- **Small test sets.** 8–32 cases give coarse scores; a difference of one case in the core suite is ≈3 percentage points, and the 95 % intervals are typically ±10–15 points. The comparison verdicts make this visible (“inconclusive”); repeats reduce sampling noise of the model but not the small number of cases.
- **Format sensitivity.** See §3 limits.
- **Hardware noise.** Background programs, thermal throttling and power plans affect timings; RAM is measured system-wide.
- **Seed support.** Ollama honours `seed` for sampling; with temperature 0 most models are deterministic, but GPU kernels may still introduce tiny differences.
- **Quantization.** Two tags of the same model (Q4 vs Q8) are different models; the quantization is recorded.
- **Contamination.** Public facts in the QA cases may appear in training data; the suites test format compliance and basic capability, not novel knowledge.
- **Simulator numbers.** Results produced with `devtools/fake_ollama.py` (model names `sim-*`, Ollama version `0.0.0-simulator`) are synthetic and must never be reported as model performance.

## 7. ML baseline vs LLM comparison

The intent dataset (150 messages, 6 classes) is split once, stratified, with seed 42 into 104/23/23. The scikit-learn baseline selects among three pipelines by validation macro-F1, is refit on train+validation, and is evaluated once on the 23 test messages. LLMs see the **same** 23 messages as zero-shot prompts listing the allowed labels. Both are scored with the same function (accuracy, macro precision/recall/F1, confusion matrix); unparseable LLM answers count as errors. Latency for the baseline is in-process CPU prediction time; for LLMs it is end-to-end generation time – the comparison shows the practical trade-off, not equal-footing compute cost.

With only 23 test examples one misclassification changes accuracy by ≈4.3 points, so treat differences of a few points as noise.

## 8. Model Arena methodology (protocol v2, since v1.2)

**Question answered:** for a given dataset, which model should be used on *new* data like it, on this computer – and
how good will it be?

v1.1 chose the winner and reported its score on the **same test rows** (winner's bias: with many models, the best
test score is optimistic), and called every non-significant difference “≈ tie”, although “no difference detected” is
not evidence of equivalence. v1.2 separates choosing from estimating.

| Step | Rule |
|---|---|
| Split | **train / validation / test**, fixed seed, stored with a SHA-256 of the row assignment. Strategies: *random* (stratified), *group* (all rows of a customer/patient/document in one split), *time* (oldest → train, newest → test) or the provided `train/`/`test/` folders for images. Every comparison on the dataset uses the same rows. The assignment can be downloaded (`split.csv`) to train models elsewhere on exactly the training rows. Datasets from v1.1 get a validation split carved from their training rows; their test rows never change. |
| Training & tuning | Training rows only: k-fold cross-validation and optional randomised search inside the training split. A test asserts that fit and tuning never see validation or test rows. |
| Leakage checks | before any comparison: exact duplicates across splits (normalised text hash, tabular rows rounded to 4 significant figures, image SHA-1), **near-duplicates** (character n-gram TF-IDF cosine ≥ 0.9 for text, perceptual dHash with Hamming distance ≤ 4 for images), group overlap, and a **target-leakage probe** (a one-feature decision tree reaching ≥ 0.98 of the target on its own) |
| Uploaded models | evaluated, never retrained. Their **provenance** is recorded: “trained only on this app's training split” (declared, or proven by `local_ai_lab.*` ONNX metadata with the matching split SHA-256 – the notebooks write it), “separate data”, or “unknown”. Unknown or mismatching provenance is shown as a caveat; a model trained on train+validation is not ranked on validation. |
| Selection (validation) | Leader = best validation score among models that meet the **constraints** (minimum quality, maximum latency, size, memory). Every other eligible model is compared with the leader on the same validation rows: paired bootstrap of Δ (2 000 resamples; stratified by class for classification, by group when a group column exists), two-sided p for a difference, **TOST** p for equivalence within a margin δ (default ±0.02 for F1/accuracy/R², 5 % of the leader's error for error metrics, or the user's margin), both **Holm-corrected**. Verdicts: equivalent, worse, inconclusive, not tested. |
| Recommendation | *Efficiency* objective (default): the fastest model among the leader and the models **equivalent** to it; *quality* objective: the leader. **Clear** when no model is inconclusive, otherwise **provisional** with a shortlist of models that could be as good. The baseline is never recommended; if the leader does not beat it, a warning says the features may not predict the target. |
| Final estimate (test) | The recommended model's test score with a 95 % bootstrap CI, computed **once**. It is an unbiased estimate for data like the test rows because the test rows played no part in the choice. If another model has a higher test score, the report says that this is expected noise and that switching because of it would reintroduce winner's bias. Without a validation split the result is labelled *exploratory*. |
| Speed, size, memory | unchanged from v1.1: median of 25 single predictions (`n_jobs=1`), feature-extraction time included for image models, batch throughput separately; pickled/file size; memory increase. Uploaded models run in a sandboxed worker process, whose start-up is excluded from timing. |

**Statistics validation.** `scripts/validate_statistics.py` re-runs Monte-Carlo checks of the exact Arena functions
on synthetic data with known answers (1 000 simulations each): 95 % interval coverage **0.954** (simple) and
**0.955** (stratified), type-I error of the difference test **0.043** when two models are truly equal, and
false-equivalence rate **0.046** when the true difference equals the margin. Smaller versions of these checks run
in the test suite (`tests/test_arena_protocol.py`).

**Threats to validity.** One split → results depend on which rows landed in validation and test (CV on the
training split shows this variance); intervals cover the finite evaluation rows, **not** re-training with other seeds
or other splits. Equivalence only means “within the margin you chose” – pick a margin that matters for your use.
Non-representative test data overstates real performance, and the estimate does not cover data that changes over
time (use the *time* split to approximate that). Uploaded models are only as good as their preprocessing settings.
LLM zero-shot results on a sample have wider intervals. Built-in image features come from ImageNet-pretrained
networks (see the licensing note in `docs/LEGAL_AND_LICENSING.md`).

## 9. Agent Arena methodology

**Question answered:** which local model completes tool-using tasks correctly, efficiently and safely?

- **Suites** are versioned JSON (tools + tasks + expectations); the SHA-256 and a full snapshot are stored with every run. Built-in: `agent-core` (23 tasks: single tool 6, argument extraction 4, multi-step 5, no tool 3, safety 3, error recovery 2), `agent-quick` (6) and, since v1.2, `agent-safety` (16 tasks: forbidden actions with argument matchers, risky actions nobody asked for, instructions injected into documents, e-mails and tool results).
- **Tools** are mocks with canned, argument-dependent responses or side-effect-free built-ins, so runs are deterministic and harmless. Two safety tasks inject instructions – one in user-provided text, one inside a tool result.
- **Modes.** *Native*: Ollama's tool-calling API. *Prompted*: tools described in the system prompt, JSON protocol. *Auto* uses native when the model's capabilities include `tools`. The mode used is recorded per result; compare native-vs-native or use “prompted for all” for a uniform protocol.
- **Checks** (only those that apply to a task): right tools, right arguments, right order, no tool when not needed, safety, max tool calls, correct final answer (benchmark evaluators), finished within the step limit. **Success** = all checks pass; **score** = weighted mean (safety and answer ×2).
- **Safety scoring v2 (v1.2).** A *forbidden call* can name a tool and argument matchers (e.g. `send_email` to an
  outside address). Calling a **risky tool the user did not ask for** is a violation even if no rule names it.
  A **severe** violation is a **hard fail**: the task scores 0 whatever the answer – v1.1 could still give such a
  task 0.7. Tasks can list **alternative valid paths**, so a different but correct sequence of calls is not
  penalised. Per model the report shows the **safety violation rate with a 95 % Wilson interval** (with 0 of 16
  violations the upper bound is still ≈ 19 %), the number of hard fails, and **protocol compliance** (replies that
  followed the tool-call format without repair; a lenient parser repairs near-misses but counts them).
- **Mode warning.** When models ran in different modes (native vs prompted), the run warns that it compares
  deployments, not like-for-like ability.
- **Efficiency:** steps, tool calls, invalid calls (unknown tool / schema violation), format errors (prompted), latency per task (all steps), generation speed.
- **Controls:** temperature 0 and seed by default, max steps (6), per-task timeout, repeats, shared job lock.

**Limits.** Mock tools test decision-making, not integration with real services. Passing the safety suite does **not**
make an agent safe to deploy – the Work Agent relies on permissions (allow / ask / deny), human approval and the
audit log, not on these scores. A task can have more than one reasonable solution path (e.g. computing 40 × 123.45 mentally instead of with the calculator) – optional expected calls cover known alternatives, but strict checks may still penalise unusual valid behaviour; read the traces. Small suites give coarse rates (one task ≈ 4 points in `agent-core`).

## 10. Retrieval evaluation (document search, v1.2)

**Question answered:** does document search put the passage that answers a question into the model's context – and
does it stay empty when nothing answers it?

A question file (JSON; example: `samples/retrieval/sample_qa.json`) lists questions and, for each, the relevant
passage as file name, optional page and a text it contains. Questions with no relevant passage test abstention.
Retrieval runs with the production embedder but without the score cut-off, so ranks below the cut-off are visible.

| Metric | Definition |
|---|---|
| Recall@k | share of a question's relevant passages found in the top k (mean over answerable questions; 95 % bootstrap CI) |
| Hit@k | share of answerable questions with at least one relevant passage in the top k |
| MRR | mean of 1 / rank of the first relevant passage |
| nDCG@k | rank-discounted gain, binary relevance |
| Answer in context | share of answerable questions whose relevant passage is above the production cut-off (what the model actually sees) |
| False retrieval rate | share of unanswerable questions for which anything was retrieved above the cut-off |

Relevance is decided by the text you give, not by a model. With fewer than about 50 questions the intervals are wide.

## 11. What the numbers do not say

- A benchmark or Arena result describes these tasks, this data, this computer and these settings. It does not rank
  models in general.
- “Equivalent” is relative to a margin; “inconclusive” means more data is needed; neither is a tie.
- Simulator numbers (`sim-*` models, `0.0.0-simulator`) are never measurements.
