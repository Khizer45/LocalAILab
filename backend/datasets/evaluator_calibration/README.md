# Evaluator calibration set

`calibration_v1.jsonl` pairs realistic model responses to real benchmark cases with a pass/fail judgement, so the
**error rate of the deterministic evaluators themselves** can be measured (false passes and false fails), as
recommended in the v1.1 methodology review. No LLM judge is involved.

Each line: `id`, `suite`, `case_id` (the evaluator spec is taken from that case, so the check always uses the
current suite file), `response`, `human_pass`, `note`, `adjudicated_by`.

## Grading rule

* `question_answering`, `reasoning`, `classification`, `summarization`, `grounded_qa`, `abstention`: graded on
  **content** – is the answer right and unambiguous? Extra wording is allowed.
* `instruction_following`, `extraction`: graded on content **and** the explicit output format ("only", "exactly",
  "nothing else").

## Status – please read

The 50 judgements in v1 are a **draft written by the developer (AI-assisted)**: `adjudicated_by` is `"draft"`.
They are useful as a regression check and to show known failure modes (e.g. the "last number" rule, exact-match
strictness, lexical summaries that contradict the text), but they are **not** independent human labels. Before
quoting evaluator error rates in a report or thesis:

1. have at least one person (better two) grade every response without looking at the evaluator result;
2. set `adjudicated_by` to their initials, resolve disagreements, and record the agreement between graders;
3. add new responses collected from real model runs (export a run as JSON – every response is stored).

Items `cal-047`–`cal-050` were also used when the `abstain` phrase list was extended in v1.2, so agreement on them
is optimistic.

Run: `python -m app.cli evaluators agreement` (from `backend/`).
