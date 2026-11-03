# Automation and integrations

Local AI Lab can be driven by scripts, CI pipelines and workflow tools such as **n8n**, and can run work on a
schedule. Everything below talks to the backend you already run on your computer; nothing is sent to a cloud
service unless you configure one yourself.

| You want to… | Use |
|---|---|
| Call your local models from n8n, LangChain or any OpenAI-format client | the OpenAI-compatible API, section 2 |
| Fail a CI job when a model gets worse | the `lal` command line, section 3 |
| Re-run a benchmark every night and get a message when quality drops | schedules + a webhook, sections 4–5 |
| Let a local model read and write files for you, with your approval | the Work Agent, section 6 |
| Move results between computers or archive them | experiment bundles, section 7 |

## 1. Access tokens

When authentication is off (the default for a single user on `127.0.0.1`), local tools need no token. Other
computers are refused unless you turn authentication on (see `docs/PRIVACY_SECURITY.md`).

With `AUTH_ENABLED=true`, create a token in **Administration → API tokens** and send it as
`Authorization: Bearer lal_…`. Give every tool its own token with only the scopes it needs and an expiry:

| Scope | Allows |
|---|---|
| `read` | GET requests: results, models, suites, settings |
| `chat` | chat, document Q&A, retrieval evaluation, and the `/v1` API |
| `documents` | uploading, re-indexing and deleting documents |
| `evaluate` | starting/cancelling benchmark, Model Arena and Agent Arena runs; schedules; bundles |
| `agent` | running the Work Agent (approval-gated tools stay gated) |
| `admin` | everything – administrators only |

Tokens can be revoked at any time; *Revoke all* signs out every session and token of the account. Token use
appears in the audit log.

## 2. OpenAI-compatible API (`/v1`)

The backend speaks the OpenAI request format, so tools that support "OpenAI-compatible" endpoints can use your
local models. **No request goes to OpenAI** – the name only describes the format.

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | models you are allowed to use (cloud models only when Offline-Only mode is off) |
| `POST /v1/chat/completions` | `stream: true` (server-sent events) and `tools` (function calling) supported |
| `POST /v1/embeddings` | uses the configured local embedding model |

Base URL: `http://127.0.0.1:8000/v1`. API key: a token with the `chat` scope (any value when auth is off).

Ground the answer in your indexed documents with the `local_ai_lab` extension field. The response then carries the
sources it used:

```json
POST /v1/chat/completions
{
  "model": "qwen3:1.7b",
  "messages": [{"role": "user", "content": "How many days of annual leave do we get?"}],
  "local_ai_lab": {"use_documents": true, "document_ids": [3, 5]}
}
```

The same rules apply as in the web interface: Offline-Only mode blocks cloud models, and documents are never sent
to a cloud model unless that is explicitly allowed in Settings.

**n8n:** add an *OpenAI* credential with *Base URL* `http://127.0.0.1:8000/v1` (or the address of the computer
running Local AI Lab, if authentication is on) and the token as API key, then use n8n's OpenAI chat-model nodes
with a local model name such as `qwen3:1.7b`.

## 3. Command line and CI gates

`scripts\lal.bat` (Windows) or `python -m app.cli` (from `backend/`) runs runs and checks gates. Exit code **0** =
all gates passed, **1** = a gate failed, **2** = error – so it can fail a CI job.

```bat
REM benchmark: every model must reach 70 % effective quality and 95 % completion
scripts\lal.bat bench run --suite core --model qwen3:1.7b --model llama3.2:1b --gate-quality 0.7 --gate-completion 0.95

REM agent safety: no safety violations allowed
scripts\lal.bat agents run --suite agent-safety --model qwen3:4b --gate-max-violation-rate 0

REM Model Arena: final test estimate at least 0.80 and a clear (not provisional) recommendation
scripts\lal.bat arena run --dataset 3 --gate-final 0.8 --require-clear

REM first-run checks of a running server
scripts\lal.bat setup check

REM local commands (work on the data folder, no server needed)
scripts\lal.bat backup create --label before-upgrade --keep 10
scripts\lal.bat db current
```

Options common to server commands: `--server` (default `http://127.0.0.1:8000`, or `LAL_SERVER`), `--token`
(or `LAL_TOKEN`), `--json` for machine-readable output, `--timeout`.

A CI pipeline needs a machine with Ollama and the models – usually a self-hosted runner. The repository's own CI
uses the built-in Ollama simulator, which tests the software, not models.

## 4. Schedules

**Automation → New schedule** (or `POST /api/schedules`) runs work every N hours or daily at a time (this computer's
clock), while Local AI Lab is running. Missed runs are not caught up.

| Kind | Request | Gates |
|---|---|---|
| `benchmark` | suite and models (any benchmark option) | `min_quality` (effective quality), `min_completion`, `max_drop` |
| `agent` | agent suite and models | `min_success`, `max_violation_rate`, `max_drop` |
| `backup` (admins) | `keep`: number of scheduled backups to keep | – |

`max_drop` (default 0.05) compares each model with the previous completed run of the same schedule. A failed
gate marks the schedule *gate failed*, writes a `schedule.alert` event to the audit log and, if a webhook is set,
calls it. *Run now* starts a schedule immediately.

## 5. Webhooks (outgoing)

Administrators add webhooks in **Administration → Webhooks**. Local AI Lab sends `POST` requests with this body:

```json
{"event": "schedule.alert", "payload": {"schedule": "Nightly core benchmark", "message": "qwen3:1.7b: effective_quality fell from 0.81 to 0.72 (run 41 → 44)"}}
```

Events: `schedule.alert`, `test`, and `work_agent.call` (only for webhooks marked *the Work Agent may call it*).

Headers: `X-LocalAILab-Event: <event>` and `X-LocalAILab-Signature: sha256=<hex>`, where hex is
HMAC-SHA256 of the **raw request body** with the webhook's secret (shown once when the webhook is created).

In Offline-Only mode the request goes through the network guard: only this computer (for example n8n on
`localhost:5678`) and hosts in `OFFLINE_ALLOWED_HOSTS` can be called.

### Verifying the signature in n8n

1. *Webhook* node: method POST, *Options → Raw Body* on.
2. *Code* node (JavaScript). n8n must allow the built-in module (`NODE_FUNCTION_ALLOW_BUILTIN=crypto`) and, for
   `$env`, environment access in Code nodes (`N8N_BLOCK_ENV_ACCESS_IN_NODE=false`). Keep the secret out of the
   workflow itself. The name of the binary property holding the raw body can differ between n8n versions – check
   the Webhook node's output.

```js
const crypto = require('crypto');
const secret = $env.LOCAL_AI_LAB_WEBHOOK_SECRET;
const item = $input.first();
const raw = Buffer.from(item.binary.data.data, 'base64');           // raw body from the Webhook node
const expected = 'sha256=' + crypto.createHmac('sha256', secret).update(raw).digest('hex');
const got = item.json.headers['x-localailab-signature'] || '';
if (got.length !== expected.length || !crypto.timingSafeEqual(Buffer.from(got), Buffer.from(expected))) {
  throw new Error('Invalid Local AI Lab signature');
}
return [{ json: JSON.parse(raw.toString('utf8')) }];
```

The signature has no timestamp; if replayed messages matter, keep a short list of recent messages and ignore
repeats.

### Licence note

n8n is under the Sustainable Use License. Running it for your own internal workflows is allowed; offering n8n
to customers (hosting it for them, letting them build workflows in it, white-labelling) needs a commercial
agreement with n8n. Local AI Lab does not include or redistribute n8n. See `docs/LEGAL_AND_LICENSING.md`.

## 6. Work Agent

The **Work agent** page (or `POST /api/work-agent/run`) lets a local model use these tools inside one workspace
folder (`data/workspace`, or `WORK_AGENT_WORKSPACE`):

| Tool | Default rule | Effect |
|---|---|---|
| `search_documents`, `calculator`, `current_datetime`, `list_files`, `read_file` | allow | read only |
| `draft_email` | allow | saves an `.eml` draft in `workspace/outbox`; nothing is sent |
| `write_file` | **ask** | creates or overwrites a text file in the workspace |
| `call_webhook` | **ask** | calls a webhook marked for agent use |

An administrator can set each tool to *allow*, *ask* or *deny*. For *ask*, the run pauses and shows exactly what
will be written or sent; nothing happens until a person approves (approvals expire after 5 minutes). Runs without a
person – `"interactive": false`, schedules, scripts – **refuse** every *ask* tool and tell the model so. Paths cannot
leave the workspace. Every tool call and decision is in the audit log.

Interactive runs stream NDJSON events (`start`, `model`, `tool`, `approval_required`, `approval_decision`,
`final`, `error`); approve with `POST /api/work-agent/approvals/{id}` and `{"approve": true}`.

Small local models make mistakes with tools. Read every approval preview; text in files can try to instruct the
model (prompt injection), which is why writing and sending need approval.

## 7. Experiment bundles

Every benchmark, Model Arena and Agent Arena run can be exported as a ZIP (**Bundle** button, or
`lal bundle export`):

* **full** – configuration, results, responses and traces, the frozen suite snapshot; can be re-run elsewhere;
* **metadata** – scores and settings without prompts, answers, labels or examples; safer to share.

`manifest.json` lists every file with its SHA-256 and says which files contain user data. **Benchmarks →
Experiment bundles** verifies a bundle, imports it as a read-only copy, or re-runs the same cases on this computer
(benchmark and agent bundles).
