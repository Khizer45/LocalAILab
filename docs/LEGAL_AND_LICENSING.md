# Legal, licensing and responsible-use notes

> This page collects the facts a person who **distributes, sells or deploys** Local AI Lab needs to check. It is a
> practical checklist written by the developers, **not legal advice**. Licences and terms change – verify them
> before you ship, and ask a lawyer for anything that matters commercially. Facts below were checked on
> 2026-10-13 against the package metadata and the model cards named.

## 1. Local AI Lab itself

* The repository does not yet contain a `LICENSE` file. The owner decides how the product is licensed (open source,
  e.g. MIT/Apache-2.0, or a proprietary licence). Every runtime dependency is permissively licensed (section 2), so
  both options are possible. **Add the licence file before distributing.**
* "Local AI Lab" names this project only. It is not affiliated with Ollama, OpenAI, Google, Meta, Microsoft, IBM,
  Alibaba, n8n or any model provider. Their names are used only to say what the software works with (e.g.
  "OpenAI-compatible API" means the request format; no request is sent to OpenAI). Do not use their logos.

## 2. Third-party software

`scripts/license_report.py` writes `THIRD_PARTY_LICENSES.md` from the installed packages and the frontend lock file
and exits with an error if a strong-copyleft (GPL/AGPL) package is installed. Run it for every release.

| Component | Licence | Notes |
|---|---|---|
| FastAPI, Pydantic, SQLAlchemy, Alembic, httpx, Uvicorn, pandas, NumPy, SciPy, scikit-learn, joblib, psutil | MIT / BSD-3-Clause | permissive |
| ONNX Runtime | MIT | its built-in usage telemetry is switched off (`ORT_DISABLE_TELEMETRY=1`, see section 6) |
| ONNX, python-multipart, MLflow | Apache-2.0 | permissive; keep the NOTICE files when redistributing |
| Pillow | MIT-CMU (HPND) | permissive |
| pypdf | BSD-3-Clause | default PDF text extraction since v1.2 |
| certifi (dependency) | MPL-2.0 | weak copyleft – fine to use unmodified; modified files must stay MPL |
| React, Recharts, TanStack Query, react-router, react-markdown, lucide-react | MIT / ISC | bundled into the web UI |
| IBM Plex Sans fonts | SIL Open Font License 1.1 | licence texts ship in `frontend/public/licenses/` (copied into the built UI) |
| **PyMuPDF** | **AGPL-3.0 or Artifex commercial licence** | **removed from the default install in v1.2.** Optional (`PDF_PARSER=pymupdf` after `pip install pymupdf`) – only if your distribution complies with the AGPL or you buy a commercial licence |
| XGBoost / LightGBM (optional) | Apache-2.0 / MIT | permissive |
| Ollama (separate install) | MIT | installed by the user; Local AI Lab talks to it over HTTP |

Development-only tools (pytest, ruff, hypothesis [MPL-2.0], openai, skl2onnx, pip-audit, Vite, TypeScript,
lightningcss [MPL-2.0]) are not shipped in the built product.

## 3. Models are licensed separately – read each one

Local AI Lab does not include LLM weights. Users download them through Ollama, and each model has its own licence.
**Setup → Models** shows the licence and whether commercial use is allowed for the models in the built-in catalogue
(`backend/datasets/model_catalog.json`):

| Model family | Licence (checked on the model card) | Commercial use |
|---|---|---|
| Qwen3 (0.6B–8B), Qwen2.5 0.5B/1.5B/7B | Apache-2.0 | yes |
| **Qwen2.5 3B** (and 72B) | **Qwen Research License** | **no – research / non-commercial only**; removed from the hardware recommendations in v1.2 |
| Llama 3.2 1B/3B | Llama 3.2 Community License | yes, with conditions: "Built with Llama" attribution when you distribute, the 700 million monthly-active-user threshold, Meta's acceptable-use policy, and naming rules for derived models |
| Gemma 3 | Gemma Terms of Use | yes, subject to Google's prohibited-use policy, which must be passed on to your users |
| Phi-4-mini | MIT | yes |
| Granite 3.3 | Apache-2.0 | yes |
| nomic-embed-text | Apache-2.0 | yes |

* **Image backbones** (MobileNetV2, EfficientNet-Lite4) come from the ONNX Model Zoo (repository licence
  Apache-2.0) and were trained on ImageNet, whose terms of access limit the *dataset* to non-commercial research and
  education. Whether that restriction carries over to trained weights is not settled. For a commercial product, get
  advice, or train your own extractor (the PyTorch notebook can train one from scratch) and upload it.
* **YOLO / Ultralytics** is *not* included. Ultralytics YOLO is AGPL-3.0 (or an Ultralytics Enterprise licence). If
  you add object detection with it, its licence applies to your product.
* **Models users upload** are their responsibility; Local AI Lab records what the uploader declares about the
  training data (provenance) but cannot verify it.

## 4. Data

* Users' documents, datasets and prompts stay on their computer (section 6). The operator of an installation is
  responsible for having the right to process that data (e.g. GDPR or the data-protection law that applies to them)
  and for access control (`AUTH_ENABLED=true` for anything shared).
* The **audit log** stores user names, IP addresses, actions and short argument summaries (never documents or prompt
  text), keeps the newest 50 000 events, and is visible to administrators only. Tell your users that it exists.
* **Google Colab** (optional notebooks in `notebooks/`) is Google's cloud service under Google's terms. Do **not**
  upload confidential, personal or customer data there unless your organisation has approved it; the notebooks run
  locally with Jupyter as well.
* Sample data in `samples/` is synthetic or fictional (generated by `scripts/make_arena_samples.py` and written for
  the project). Built-in benchmark and agent suites were written for this project.
* Experiment bundles in *metadata* scope leave out prompts, responses, traces, answers, error examples and labels;
  *full* bundles contain them. The manifest says which files contain user data.

## 5. Integrations

* **n8n** is under the Sustainable Use License (fair-code, not OSI open source). Using it for your own internal
  business purposes is allowed; according to n8n's licence FAQ this includes using it behind your own product *as
  long as only your backend talks to n8n and your end users do not use the n8n editor or configure n8n workflows*.
  Offering n8n itself – white-labelling it, letting customers build workflows in it, or hosting n8n for paying
  customers – needs a commercial agreement with n8n. Local AI Lab does **not** embed or redistribute n8n: it talks
  to an n8n instance the user runs (outgoing signed webhooks, and the OpenAI-compatible `/v1` API that n8n can
  call). Customers who self-host n8n do so under n8n's licence themselves.
  An MIT-licensed alternative for embedding is the Activepieces Community Edition (its enterprise features are
  licensed separately).
* **Ollama Cloud** (optional, off by default and blocked in Offline-Only mode) is a third-party service with its own
  terms and data handling.

## 6. Privacy claims – what is and is not promised

Write "offline-first" or "runs locally", not "100 % offline" or "cannot leak data":

* Offline-Only mode is an **application-level** restriction. It blocks cloud models and outbound requests made by
  the Python backend (HTTP layer and a socket-level audit hook). It is not an operating-system firewall.
* Native libraries, the Ollama server, the browser and the operating system can open their own connections. In
  v1.2 we found that **ONNX Runtime 1.31 sent usage telemetry to `mobile.events.data.microsoft.com` from
  long-running processes**; the Python socket guard cannot see native code, and
  `onnxruntime.disable_telemetry_events()` did not stop it. `ORT_DISABLE_TELEMETRY=1` did, and it is now forced at
  start-up (verified by watching an egress proxy during the full test suite). Other known telemetry switches
  (Hugging Face, MLflow, OpenTelemetry, `DO_NOT_TRACK`) are set too. Ollama's own cloud features are disabled with
  `OLLAMA_NO_CLOUD=1`.
* The first install, model downloads, backbone downloads and optional cloud features need the internet.
* For a hard guarantee, block outbound traffic with a firewall rule or use a machine without a network.

## 7. Claims about evaluation results

* The recommended model's test score is an unbiased estimate **for data like the test rows**; it does not promise
  the same performance on other data. Equivalence means "within the margin you chose". Inconclusive means "the data
  cannot tell", not "tied". Reports say so; do not shorten this in marketing.
* Agent safety scores come from small, mock-tool suites. Passing them does not make an agent safe to deploy; the
  Work Agent's real controls are permissions, approvals and the audit log.
* Simulator runs (the built-in Ollama simulator) are labelled *SIMULATED* and must not be presented as model
  measurements.

## 8. Before you ship – checklist

- [ ] Add a `LICENSE` for Local AI Lab and a NOTICE that keeps the Apache-2.0 notices of dependencies.
- [ ] Run `scripts/license_report.py` on the release environment; ship `THIRD_PARTY_LICENSES.md` and the font licences.
- [ ] Do not bundle model weights unless their licence allows redistribution; if you bundle Llama, follow its
      attribution and notice rules; never bundle research-only models (e.g. Qwen2.5 3B) in a commercial product.
- [ ] Decide on the image backbones for commercial use (section 3).
- [ ] Keep PyMuPDF out unless you comply with the AGPL or have a commercial licence.
- [ ] If you host n8n for customers, get the n8n commercial licence.
- [ ] Publish a privacy notice that matches section 6 and mentions the audit log.
- [ ] Turn on authentication and HTTPS for any shared/network installation.
