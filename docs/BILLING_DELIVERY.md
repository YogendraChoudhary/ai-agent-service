# Python billing delivery and cancellation

The Python service persists credit-deduction messages in Redis before handing them to the HTTP delivery worker. Both billing entry points use this path, including streaming, autocomplete and voice. `/agent/process` still leaves text billing to the upstream application backend. BYOK plans retain their existing no-credit-charge behavior.

## Failed delivery keeps the credit hold

Enqueuing a charge atomically moves its active reservation from `credits_reserved:<user>` to `credits_pending_billing:<user>` and records the billing message. The pending hold and queued message have no expiry. New reservations and available-credit checks subtract both active and pending amounts. A billing-backend HTTP 500 or connection timeout therefore cannot return those credits to the spendable balance.

Each delivery carries a random, stable `Idempotency-Key` header. The existing JSON payload and `X-Webhook-Secret` contract are unchanged. A successful HTTP 2xx response marks the message acknowledged. After the existing `CREDIT_SETTLE_DELAY_SECONDS`, the pending hold is removed exactly once. The receiving backend must acknowledge successful usage recording and apply its credit/cache update under that contract; this Python change cannot verify an external backend's transaction semantics.

Each application worker drains the same queue. Redis leases prevent concurrent delivery of a message, and lease tokens prevent an old worker from overwriting a newer worker's result. A restarted worker can recover queued messages and finish releasing acknowledged messages without posting them again. Completed-event fingerprints are retained for seven days to reject immediate duplicate enqueue attempts and conflicting payloads.

## Automatic payment retries are opt-in

`BILLING_WEBHOOK_IDEMPOTENT` defaults to **false**. Leave it false until the billing receiver is confirmed to deduplicate `Idempotency-Key` atomically with the credit deduction.

- With the default, each message gets one transport attempt. A failed, timed-out or uncertain attempt stays persisted with its credits held for reconciliation. Restarting a worker does not blindly repost that message.
- With `BILLING_WEBHOOK_IDEMPOTENT=true`, unacknowledged messages retry with the same key and unchanged payload, using a 2-second backoff that increases to at most 300 seconds. Retries are not discarded after a fixed attempt count.
- A timeout can happen after the receiver has already charged the account. HTTP 500 can also follow a committed operation. Retrying either without receiver-side deduplication can charge twice; this is why Python does not enable retries by default.
- Missing webhook configuration leaves messages pending rather than discarding their billing records or releasing their holds.

No external backend was modified for this change. Until duplicate protection is verified there, failed messages require operator reconciliation. Retaining holds is deliberate: it can temporarily reduce a user's available balance during a billing outage, but it prevents repeated unbilled provider usage. Do not delete pending messages or counters merely to clear a held balance; first establish whether the receiver recorded the charge.

## Receiver contract for the application backend (agreed November 2026)

The backend team confirmed their receiver currently ignores `design`, `images[]` and `Idempotency-Key`, so a design
is charged as ordinary text by characters. They will change the receiver as follows; nothing changes in this
service's payload, which already carries everything needed.

**Design charges.** Designs keep `operation_type: "text"` (so the current receiver still accepts them). When the body
has a `design` object, charge it flat and ignore `input_chars` / `output_chars`:

```json
{"operation_type": "text", "input_chars": 142, "output_chars": 0, "design": {"sub_mode": "create_from_prompt", "credits_scaled": 30}}
```

| `design.sub_mode` | Charge | `credits_scaled` |
|---|---|---|
| `create_from_prompt`, `create_from_vision` | 3 credits | 30 |
| `edit_existing_ast` | 0.5 credits | 5 |

`credits_scaled` is the agent's price for that `sub_mode` (1 credit = 10 scaled); either field gives the same result.
Check for `design` before applying the character price. `/agent/process` does not send this webhook: for a design made
there, the backend bills from the request `type` (`generate_design` = 3, `edit_design` = 0.5).

**Images.** `images: [{"provider", "model", "credits_scaled"}]` lists each picture with its model's price. Until the
receiver reads it, `image_count` × 5 credits applies, as today.

**Duplicates.** Each message carries a stable `Idempotency-Key` header (also the event id), unchanged across retries.
The receiver must:

1. Record the key in the same database transaction as the credit deduction, so a key is never stored without its
   charge or charged without its key.
2. Answer a repeated key with **HTTP 2xx** and no second charge. Only 2xx counts as delivered here; a 409 or other
   error makes the agent keep retrying that message.
3. Keep keys for at least 7 days. Retries back off up to every 300 seconds with no attempt limit.

**Switching on retries.** Only after (1)–(3) are deployed, set `BILLING_WEBHOOK_IDEMPOTENT=true` on every agent
worker. Until then it stays `false`: each message is posted once, and a failed or timed-out delivery keeps its credit
hold for manual reconciliation instead of risking a double charge.

## Deployment and operations

- Run the normal FastAPI lifespan. It starts and stops the delivery worker around the Redis connection lifecycle. Upgrade every application worker so all reservation checks include pending billing.
- Keep `CREDIT_PRECHECK_ENABLED=true` for managed accounts. Disabling it intentionally disables admission checks against available credits.
- Use Redis persistence appropriate for financial delivery records, with eviction disabled for the billing data. Surviving an application restart relies on the queued data surviving in Redis; this is not a backup against Redis data loss or a crash before usage can be enqueued.
- Configure `BACKEND_WEBHOOK_URL`, `WEBHOOK_SECRET` and the existing settlement delay. HTTP redirects are not followed, so webhook credentials are not forwarded to a redirected host.
- Inspect pending messages with `HGETALL agent:billing:jobs` and due times with `ZRANGE agent:billing:due 0 -1 WITHSCORES`. Messages include counts, billing metadata, attempt count, state and the latest error class/status; no provider keys or document text are stored.
- Logs identify held/unacknowledged event IDs. Investigate configuration/authentication errors and unresolved messages. If retries were disabled, verify receiver-side deduplication or reconcile the specific charge before enabling automatic replay.

The settlement delay retains the existing assumption that the backend's balance update reaches Redis before the hold is released. Actual receiver idempotency, Redis durability and balance propagation must be checked in deployment; they were not changed or verified against production here.

## Canceled document edits

Explicit `document_context` Edit and legacy `document_text` edits now record model work only immediately before the provider invocation. Cancellation is checked again after constructing the agent. If canceled before a provider call, the stream reports `done.aborted=true`, releases its active reservation and sends no successful billing deduction. No-model clarification results also incur no text charge. Successful edits preserve their patch/apply contract and remain billable. Cancellation after provider work has started retains the existing billing policy for that work.

## Verification

Install test dependencies with `.venv/bin/python -m pip install -r requirements-dev.txt`.

Run `.venv/bin/python -m pytest tests/test_billing_delivery.py tests/test_document_edit_billing.py tests/voice_flow/test_access.py tests/document_edit` for billing and edit boundaries. Tests execute the real Redis Lua scripts against an isolated Redis emulator, including atomic hold transfer, failed delivery, stable-key retries, post-commit timeouts, duplicate events, worker recovery and concurrent claims. API integration verifies that a failed billing handoff consumes the available hold and blocks additional paid provider work until settlement; cancellation tests cover both edit contracts.

Verified on 3 November 2026: **2,180 tests and 1,662 subtests passed**. The provider-free document context and legacy document edit demos, dependency consistency, Python compilation and whitespace checks also passed. No production deployment or external billing receiver was contacted.
