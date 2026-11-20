# Deployment checklist: week of 16–20 November 2026

For: whoever deploys the agent service. Follow it top to bottom; every step has a check. Background and the full runbook:
`docs/ROLLOUT_RUNBOOK.md`.

## Why this deploy is different

On 20 November every Claude request on the advanced tier failed in production with
`Messages.stream() got an unexpected keyword argument 'output_config'`. The code had been tested with newer AI SDKs than
the server had installed: `requirements.txt` did not pin them, so the server kept older ones. From this deploy on:

* **Dependencies come from `requirements.lock`**, which pins every package to the exact version that passed the tests.
* **The app checks its SDKs at startup** and logs a CRITICAL alert if they do not match the code.
* **Feature settings live in `config.py`.** No new `.env` variables are needed for this deploy.

## 0. Before you start

- [ ] Python **3.12** on the server (the lockfile was built for 3.12): `python3.12 --version`.
- [ ] The **frontend has shipped first**, or ships together with this deploy. Design schema 1.1 is now on by default.
      An editor that cannot render 1.1 must send `schema_version: "1.0"` on design requests (`docs/FRONTEND_HANDOFF.md` §4.8).
      The email changes are additive and need no ordering.
- [ ] If you deploy from a zip or a copy rather than `git pull`, it must include the new and changed files, especially:
      `requirements.lock`, `src/sdk_contracts.py`, `src/email_template/rewrite.py`, `src/email_template/layout.py`, and
      the bundled fonts `src/design_canvas/fonts/` (52 files, 6.5 MB).

## 1. Install dependencies into a NEW virtualenv

Build the new environment next to the running one, so the running service is untouched until the switch and the old
environment stays available for rollback.

```bash
cd /home/ubuntu/agent                      # the service's WorkingDirectory
git pull                                   # or unpack the release
python3.12 -m venv /home/ubuntu/.venvs/aienv-2026-11-20
/home/ubuntu/.venvs/aienv-2026-11-20/bin/pip install --upgrade pip
/home/ubuntu/.venvs/aienv-2026-11-20/bin/pip install -r requirements.lock
```

Check that the installed versions match the lock exactly:

```bash
/home/ubuntu/.venvs/aienv-2026-11-20/bin/pip freeze | grep -E "^(anthropic|openai|google-genai|agno)=="
```

You must see exactly `agno==2.8.5`, `anthropic==0.125.0`, `google-genai==2.29.0`, `openai==3.26.1`.

**Do not** install from `requirements.txt` in production, and do not reuse the old virtualenv or a cached Docker
layer without checking these versions afterwards. Either one can leave the old SDKs in place, which caused the outage.

## 2. Pre-deploy check (with the new virtualenv)

```bash
cd /home/ubuntu/agent
/home/ubuntu/.venvs/aienv-2026-11-20/bin/python scripts/preflight_models.py
```

- [ ] It must end with **`PREFLIGHT OK`** (exit code 0).
- The first part runs offline: `SDK contracts OK (48 request paths)`. A `SDK CONTRACT MISMATCH` line means the wrong
  package versions are installed. Go back to step 1; do not deploy.
- The second part looks up every configured model with the platform keys from `.env` (free lookups, nothing generated).
  `MISSING` or `key rejected` means a key or model problem. Fix it before deploying.

## 3. Switch the service and restart

Point the service at the new virtualenv. In `/etc/systemd/system/flaskagent.service` (or `agent.service`), change the
`ExecStart` path:

```ini
ExecStart=/home/ubuntu/.venvs/aienv-2026-11-20/bin/gunicorn api:app \
  --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:5000 \
  --access-logfile - --error-logfile - --log-level info
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart flaskagent          # or: agent
```

## 4. Verify after the restart (first 5 minutes)

```bash
sudo journalctl -u flaskagent --since "5 min ago" | grep -E "SDK contract check|Model preflight|ROLLOUT_ALERT|sdk_contract_mismatch"
```

- [ ] `SDK contract check OK (48 request paths; anthropic=0.125.0, openai=3.26.1, google-genai=2.29.0, agno=2.8.5)`, once per worker.
- [ ] `Model preflight OK: claude=accepted, openai=accepted, gemini=accepted`.
- [ ] **No** `ROLLOUT_ALERT` lines. In particular, no `event=sdk_contract_mismatch`.

Smoke tests from the editor (or ask the frontend to run them):
- [ ] **Claude:** select a paragraph, choose Anthropic Claude, prompt "Add 50 words about Antarctic weather before
      this paragraph". This is the request that failed. It must answer with no error toast. (The startup line
      `SDK contract check OK` above is the definitive check for the bug; this confirms it end to end.)
- [ ] **Email:** generate one template from the email feature (or `POST /agent/email/generate`).
- [ ] **Design:** one design request. Check that it renders in the editor (schema 1.1, or 1.0 if the frontend pins it).

## 5. Monitoring to keep

- **Alert on these log patterns** (journald / your log shipper):
  - `ROLLOUT_ALERT`: any provider or model problem (dead key, missing model, rejected requests).
  - `event=sdk_contract_mismatch` (level CRITICAL): the installed SDKs do not match the code. Treat it as a failed
    deploy, because requests are being served by a fallback provider.
- `GET /agent/health/providers` shows the same alerts, failures and fallbacks as JSON. It answers only with the header
  `X-Agent-Backend-Secret: <AGENT_BACKEND_SECRET>`; when that secret is not configured, the endpoint returns 404 and the
  logs are the source.

## 6. Settings: what not to change

The production values are the defaults in `config.py`. For this deploy:
- [ ] Do **not** add new variables to `.env`.
- [ ] Check that `.env` does not *override* a feature flag by accident. A value in `.env` wins over `config.py`:
      `grep -E "_ENABLED|_TWO_PASS|SHADOW|SDK_CONTRACT" .env`. Expected: no `=false` for `DESIGN_SCHEMA_V11_ENABLED`,
      `DESIGN_VISION_TWO_PASS`, `DESIGN_IMAGE_FILL_ENABLED`, `EMAIL_AI_ENABLED`, `PROVIDER_FALLBACK_ENABLED`, `TASK_ROUTING_ENABLED`.
- [ ] Leave these **off**:
  - `BILLING_WEBHOOK_IDEMPOTENT`, until the billing receiver deduplicates `Idempotency-Key` (`docs/BILLING_DELIVERY.md`);
  - `VOICE_FLOW_ENABLED`, unless `VOICE_FLOW_TOKEN` is set (the voice routes are otherwise open);
  - `LOG_REQUEST_BODIES`, because it would log users' text.

## 7. Rollback

Code and environment roll back together: point `ExecStart` back to the previous virtualenv, check out the previous
release, `daemon-reload` and restart. Keep the previous virtualenv until this release has run cleanly for a few days.
A rollback brings back the old SDKs, and with them the Claude error, so treat it as a last resort.

## 8. Keeping things in sync afterwards

- **One source of versions.** Every server, CI job and image installs from `requirements.lock`. When someone changes
  `requirements.txt`, they regenerate the lock (the command is in its header), run the tests, and ship both files together.
- **Every deploy runs `scripts/preflight_models.py`** after installing and before restarting. In CI, run
  `python scripts/preflight_models.py --sdk-only` (offline, no keys needed).
- **All instances get the same lockfile.** A worker or instance on different versions shows up as an
  `sdk_contract_mismatch` alert from that host.
- **Feature flags change only in `config.py`**, through a commit, so code and settings ship together.
