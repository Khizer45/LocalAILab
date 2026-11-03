# Project Falcon: engineering notes (fictional sample)

## Architecture
Project Falcon uses a FastAPI backend, a React frontend and an SQLite database. Background jobs run in a single worker thread. Deployment is done with Docker Compose on a single on-premise server.

## Decisions
- We chose SQLite over PostgreSQL because the system has fewer than 20 concurrent users.
- Embeddings are computed locally with nomic-embed-text so that patient documents never leave the hospital network.
- Model responses are streamed to the browser as newline-delimited JSON.

## Open risks
1. The server has 16 GB of RAM, so models larger than 8B parameters cannot be loaded together with the database cache.
2. Backups run nightly at 02:00 and are stored on an encrypted external disk for 30 days.
