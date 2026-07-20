# Study Agent Platform Progress

## 2026-05-10

### Completed

- Added `learn-platform/` as an independent Next.js + TypeScript app so the existing Jekyll homepage stays stable.
- Added server-side API routes:
  - `POST /api/assistant/chat`
  - `GET /api/cards/today`
  - `POST /api/reviews`
  - `POST /api/import/cards`
  - `GET /api/health`
  - `POST /api/session`
- Added Hermes forwarding logic with Cloudflare Access headers and `x-hermes-shared-secret`.
- Added Supabase REST helpers and a Free-plan-friendly schema in `learn-platform/db/schema.sql`.
- Added a responsive study UI with dashboard stats, review card flow, source citations, and chat panel.
- Added local book import helper script at `learn-platform/scripts/prepare_import.py`.
- Added setup docs for Cloudflare Tunnel, Hermes agent prompt, Vercel, and Supabase.
- Added full implementation plan at `learn-platform/docs/implementation-plan.md`.
- Added Hermes + Feishu setup guide at `learn-platform/docs/hermes-feishu-setup.md`.
- Added Jekyll homepage entry page `learn.md` and navigation item `Learn`.
- Installed npm dependencies successfully and generated `learn-platform/package-lock.json`.
- Confirmed WSL `Ubuntu-22.04` is running.
- Confirmed Hermes is installed at `/home/volodymyr/.local/bin/hermes`.
- Confirmed Hermes gateway is already running as a systemd service.
- Confirmed Feishu is connected in websocket mode.
- Confirmed Hermes API server health endpoint responds at `http://127.0.0.1:8642/health`.
- Confirmed Feishu DM home channel exists: `oc_e62ff5e9873efc83f9b1bb541cfbceac`.
- Confirmed one Feishu user is already approved through Hermes pairing.
- Added study-agent SOUL prompt at `learn-platform/docs/hermes-study-agent-soul.md`.
- Installed the study-agent prompt into WSL Hermes default profile at `~/.hermes/SOUL.md`.
- Backed up the previous Hermes prompt at `~/.hermes/SOUL.md.before-study-agent-20260510`.
- Verified Hermes answers as Wenzhe's learning assistant with a one-shot prompt.
- Updated the web assistant proxy to call Hermes' OpenAI-compatible `POST /v1/chat/completions` endpoint.
- Added `GET /api/library`.
- Updated review scheduling so `cards.next_review_at` changes after each review.
- Reworked the web UI into Dashboard, Review, Chat, Library, and Import workflows.
- Added Import UI for structured local parser JSON.
- TypeScript diagnostics pass after the workflow update.

### Current Blocker

- No current source-code blocker.
- Local Codex sandbox had trouble running `next build`, but the user reported local `npm build` succeeded outside the sandbox.

### Next Steps

- TypeScript diagnostics passed with `node .\node_modules\typescript\bin\tsc --noEmit`.
- Updated `learn-platform/next.config.ts` to disable Next.js build workers and parallel server compile traces for better Windows local-build compatibility.
- Hermes gateway reports an outdated service definition; later run a gateway restart/update command when safe.
- `API_SERVER_KEY` is currently missing, so the local API server should be protected before Cloudflare Tunnel exposure.
- Hermes cron is running but has no scheduled jobs yet.
- Before exposing Hermes with Cloudflare Tunnel, set `API_SERVER_KEY` in Hermes and Vercel.
- Configure Supabase and Vercel environment variables.
- Configure Cloudflare Tunnel to point at Hermes `127.0.0.1:8642`.
- Run browser checks on desktop and mobile viewports after deployment or local dev server startup.
