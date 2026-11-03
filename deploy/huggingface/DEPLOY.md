# Online demo on Hugging Face Spaces

The online demo runs Local AI Lab with the built-in model simulator in a single Docker container. It is meant for
showing the interface; real models run on your own computer.

## What makes the public demo safe

| Risk | Protection |
|---|---|
| Visitors changing settings, turning off Offline-Only mode, downloading models | The shared user is not an administrator, so the app refuses every admin action; these routes are also on a deny-list |
| Running code through uploaded pickle/joblib models | Only administrators may load pickles – nobody is an administrator in the demo; ONNX models run in the sandboxed worker (1.5 GB, 5 min, no network) |
| Reading other visitors' data through backups or the audit log | Backup and audit routes are disabled |
| Using the demo as a free AI API | The OpenAI-compatible `/v1` API is disabled |
| Spam and resource abuse | 30 state-changing requests per minute per visitor, small upload limits (5 MB documents, 10 MB datasets), no schedules |
| Data piling up | All data is deleted and the demo is re-seeded every 6 hours (`DEMO_RESET_HOURS`) |
| Changing the code at runtime | Application files belong to root and are read-only; the server runs as an unprivileged user |
| Secrets leaking | There are none: no API keys, no real models, no cloud access (Offline-Only mode and the network guard stay on) |
| Click-jacking | The page may only be framed by huggingface.co |

## Deploy

1. Create a Space at <https://huggingface.co/new-space>: SDK **Docker**, template **Blank**, hardware **CPU basic**
   (free), visibility **Public**.
2. Build the Space folder: `python deploy/huggingface/build_space.py` (creates `../local-ai-lab-space`).
3. Upload it:
   ```bash
   git clone https://huggingface.co/spaces/<your-user>/<space-name> hf-space
   # copy everything from local-ai-lab-space into hf-space, then:
   cd hf-space
   git lfs install
   git add . && git commit -m "Local AI Lab demo" && git push
   ```
   Git asks for your Hugging Face user name and an **access token** with *write* permission as the password
   (Settings → Access Tokens). Never put the token in a file or in the repository.
4. The Space builds (about 5–10 minutes) and starts. The first visit after a reset shows seeded example results
   after about a minute.

Optional settings (Space → Settings → Variables): `DEMO_RESET_HOURS`, `DEMO_WRITES_PER_MINUTE`.
