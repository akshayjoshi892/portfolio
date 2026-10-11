# Part XXXIII — Tool Engineering

## Before you start

**What you need already:** [Part VI](part06-structured-outputs-tool-calling.md) (tool calling, circuit breakers), [Part IX](part09-agent-state.md) (the lost-update/race-condition concept), [Part XXIV](part24-mcp-server-development.md) (real tools with real side effects).

**What this part gives you:** A formal answer to a question that's been lurking since Part VI: which tools are safe to retry automatically, and which ones can cause real damage — like a duplicate charge or a duplicate email — if you get that wrong?

**Time:** 2 days.

**When you finish, you'll be able to:** Classify any tool by its risk and retry-safety properties, and enforce that classification in code so a dangerous tool can never be silently retried in a way that duplicates a real-world effect.

---

## Stage 1 — The scenario that makes this matter

Here's a situation that will eventually happen to any agent system with side-effecting tools, if it hasn't already: an agent calls a `send_email` tool. The email genuinely gets sent. But the network response confirming success is slow, or gets lost — from the agent's point of view, indistinguishable from the call having failed. Following [Part VI](part06-structured-outputs-tool-calling.md)'s retry discipline, a timeout looks retriable. So the agent retries.

The email gets sent a second time.

This isn't a hypothetical edge case — it's a direct, predictable consequence of applying the retry logic you built for read-only tools (like `calculator` or `search`, where retrying a "maybe it failed, maybe it succeeded and we just didn't hear back" case is completely harmless) to a tool where retrying on that same ambiguity duplicates a real-world effect.

**Check yourself before Stage 2:** Why is retrying an ambiguous-outcome timeout safe for `calculator` but potentially harmful for `send_email`?

---

## Stage 2 — The property that determines the answer: idempotency

The concept that resolves this is **idempotency**: an operation is idempotent if doing it twice has the same effect as doing it once.

`set_status(order_id, "shipped")` is idempotent — calling it twice leaves the order in exactly the same state as calling it once. `send_email(...)` is not — calling it twice sends two emails, a genuinely different outcome than calling it once. `increment_counter()` is not idempotent either, for the same reason — twice adds 2, not 1.

```python
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"        # read-only, or trivially reversible
    MEDIUM = "medium"   # write, but low-consequence or easily corrected
    HIGH = "high"        # write, high-consequence, hard or impossible to undo


@dataclass
class ToolContract:
    name: str
    idempotent: bool
    side_effecting: bool
    risk: RiskLevel
```

Notice this makes the property **explicit and inspectable in code** — not just something a developer is supposed to remember about a given tool. That's the whole point: a fact that only lives in someone's head gets forgotten. A fact encoded as a field on a `ToolContract` can be checked by code, every single time, automatically.

**Check yourself before Stage 3:** Classify these as idempotent or not: `delete_file(path)` (deleting an already-deleted file), `charge_card(amount)`, `mark_as_read(message_id)`.

---

## Stage 3 — Enforce the property, don't just document it

Here's where declaring the contract pays off: you can now write execution logic that treats idempotent and non-idempotent tools *differently*, automatically, rather than trusting every developer to remember which is which for every tool they write.

```python
import time


class ToolExecutionError(Exception):
    pass


def execute_with_contract(contract: ToolContract, fn, args: dict,
                           idempotency_key: str | None = None, max_retries: int = 2) -> dict:
    if contract.side_effecting and not contract.idempotent and idempotency_key is None:
        raise ValueError(
            f"{contract.name}: non-idempotent, side-effecting tools require an "
            f"idempotency_key before they can even be called — this is not optional"
        )

    # Idempotent tools (or read-only, non-side-effecting ones) can be retried freely.
    # Non-idempotent side-effecting tools get exactly ONE attempt — because
    # retrying on an ambiguous failure risks duplicating a real-world effect.
    attempts_allowed = max_retries + 1 if contract.idempotent else 1
    last_error: Exception | None = None

    for attempt in range(1, attempts_allowed + 1):
        try:
            return fn(args)
        except Exception as e:
            last_error = e
            if not contract.idempotent:
                break   # never blindly retry a non-idempotent side-effecting call
            time.sleep(0.2 * attempt)

    raise ToolExecutionError(f"{contract.name} failed after {attempts_allowed} attempt(s): {last_error}")
```

Look at the first check: a non-idempotent, side-effecting tool **cannot even be called** without an `idempotency_key` — the function refuses outright, rather than allowing an unsafe call and hoping the caller remembered to be careful. And look at `attempts_allowed`: an idempotent tool gets real retries, exactly like Part VI's tools; a non-idempotent one gets exactly one attempt, no retry at all, because retrying it is precisely the dangerous behavior from Stage 1.

**Check yourself before Stage 4:** Why does `execute_with_contract` refuse to even attempt a call for a non-idempotent, side-effecting tool with no `idempotency_key`, rather than just warning and proceeding?

---

## Stage 4 — The real fix: make the operation itself idempotent

Refusing to retry an unsafe tool prevents duplication, but it also means a genuinely transient failure (a real timeout, no actual email sent) now requires a human to sort out, instead of being safely automatic. There's a better fix: make the operation *itself* idempotent, using the `idempotency_key`, so it becomes safe to retry after all.

```python
import uuid

SENT_EMAIL_KEYS: set[str] = set()   # in production: a durable store, not memory


def send_email_impl(args: dict) -> dict:
    key = args["idempotency_key"]

    if key in SENT_EMAIL_KEYS:
        # We've already processed this exact request. Don't send again —
        # just report success, as if this were the first successful attempt.
        return {"status": "already_sent", "to": args["to"]}

    # ... actually send the email here ...
    SENT_EMAIL_KEYS.add(key)
    return {"status": "sent", "to": args["to"]}
```

Here's the trick: `send_email` is inherently non-idempotent — calling it twice normally means two emails. But if the *implementation itself* checks "have I already processed this exact idempotency key" before doing the real work, calling it twice with the *same key* now produces the same effect as calling it once (one email sent, second call is a safe no-op that just confirms the first one's result). You've converted a non-idempotent operation into an idempotent one, at the cost of the caller needing to supply a stable, unique key per logical action — and now retrying really is safe.

This is the mechanism real payment systems, real email services, and real production agent tools actually use — it's not a curiosity, it's the standard fix for exactly this problem.

**Check yourself before the project:** How does adding an idempotency-key check inside `send_email_impl` change whether it's safe to retry, compared to the plain version from Stage 1?

---

## Stage 5 — Risk classification drives more than just retries

`RiskLevel` from Stage 2 has another job beyond the retry decision — it's the input to whether a tool needs a human-approval gate before it runs at all, which you'll build fully in [Part XXXIV](part34-human-in-the-loop.md).

```python
def requires_human_approval(contract: ToolContract) -> bool:
    return contract.risk == RiskLevel.HIGH and contract.side_effecting
```

The design principle worth internalizing: **risk classification is decided once, when the tool is defined, and every downstream system (retries, approval gating, audit logging) reads from that one classification** rather than each system re-deciding independently whether a given tool is "dangerous." This consistency matters — if retry logic and approval logic disagreed about which tools were risky, you'd have a real gap between what gets extra scrutiny and what gets extra safety.

**Check yourself before the project:** Why is it better to have one central `ToolContract.risk` field that multiple systems (retries, approval) read from, rather than each system independently deciding what counts as risky?

---

# The Project: A Tool Registry That Enforces Contracts

## What you're building

Extend your Part VI `ToolRegistry` so every tool must declare a `ToolContract` before registration — and the registry rejects any tool that's `side_effecting` and non-idempotent without a declared idempotency-key parameter in its schema.

```bash
$ python contract_registry.py
```

```
registering: calculator (idempotent=True, side_effecting=False, risk=LOW)
registering: search (idempotent=True, side_effecting=False, risk=LOW)
registering: send_email (idempotent=False, side_effecting=True, risk=HIGH)
  -> requires idempotency_key parameter: CONFIRMED PRESENT
registering: delete_all_records (idempotent=False, side_effecting=True, risk=HIGH)
  -> REJECTED: no idempotency_key parameter declared in schema

calling send_email twice with the SAME idempotency_key...
  1st call: status=sent
  2nd call: status=already_sent (no duplicate email!)
```

## Why this project

This is the discipline that prevents the exact real-world incident from Stage 1 — a duplicated side effect from an innocent, well-intentioned retry — and it's a pattern you'll reuse directly in [Part XXXVI](part36-reliability.md) and [Part XLIX](part49-distributed-agents.md).

## Build it in stages

---

### Stage A — `ToolContract` and registration-time enforcement

**Goal:** A registry that refuses to register a non-idempotent, side-effecting tool unless its schema includes an `idempotency_key` field.

**Do:**
1. Implement `ToolContract` and extend your Part VI `ToolSpec` to include it
2. Add a registration-time check: if `side_effecting and not idempotent`, verify the tool's Pydantic args model has an `idempotency_key` field, or reject registration with a clear error
3. Register 3–4 tools spanning different risk/idempotency combinations, and confirm the check fires correctly for each

**You're done when:** You can register a correctly-declared non-idempotent tool successfully, and watch a badly-declared one get rejected at registration time — before it's ever callable, not after.

---

### Stage B — `execute_with_contract`, wired into real execution

**Goal:** The retry-vs-single-attempt logic from Stage 3, applied to real tool execution in your registry.

**Do:**
1. Implement `execute_with_contract`
2. Test it against an idempotent tool that fails intermittently — confirm it retries and eventually succeeds
3. Test it against a non-idempotent tool that fails — confirm it does *not* retry, and fails cleanly after exactly one attempt

**You're done when:** You've watched both behaviors happen for real, on real (if simulated) failures.

---

### Stage C — Make a non-idempotent tool genuinely idempotent

**Goal:** `send_email_impl` from Stage 4, using an idempotency-key store to make repeated calls with the same key safe.

**Do:**
1. Implement it with a real dedup store (a `set`, or a small SQLite table for persistence)
2. Call it twice with the *same* idempotency key — confirm only one "real" send happens, and the second call reports `already_sent` rather than sending again
3. Call it twice with *different* keys — confirm both genuinely send (this is correct — different keys mean different logical actions)

**You're done when:** You've confirmed both the dedup behavior (same key) and the correct non-dedup behavior (different keys) work as intended.

---

### Stage D — Wire risk into approval gating

**Goal:** `requires_human_approval`, correctly flagging only high-risk, side-effecting tools.

**Do:**
1. Implement it
2. Confirm it returns `True` for your high-risk email/delete-style tools and `False` for your low-risk calculator/search tools
3. (You don't need the full approval flow yet — that's Part XXXIV — just confirm the flag fires correctly)

**You're done when:** You can point to code that correctly distinguishes tools needing human review from ones that don't, based purely on their declared contract.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| A non-idempotent tool gets registered without an idempotency-key requirement | Registration-time check only looking at `idempotent`/`side_effecting` flags, not actually inspecting the schema for the required field | Actually introspect the Pydantic model's fields, don't just trust the developer declared it correctly elsewhere |
| Idempotency-key dedup doesn't work across process restarts | Using an in-memory `set` instead of a persistent store | Move the dedup store to SQLite or Postgres, reusing Part IX's persistence patterns |
| A genuinely idempotent tool (like `set_status`) gets denied retries | Incorrectly marked `idempotent=False` in its contract | Double-check the classification — `set_status` really is idempotent; the contract should reflect that |

## Going further (optional)

1. **A compensating action.** For a tool like `charge_card`, define a corresponding `refund_card` and think through when an agent should automatically trigger the compensation versus when it should require human approval to do so.
2. **Expiring idempotency keys.** Real systems don't keep dedup keys forever — add a TTL to your dedup store and think through what should happen if a retry arrives *after* the key has expired.
3. **A contract linter.** Write a small script that scans all registered tools and flags any whose declared contract looks suspicious (e.g., a name containing "delete" or "send" marked `idempotent=True`) — a sanity check against misclassification.

---

## Checkpoint

**Knowledge:**
1. Define idempotency precisely, and explain why it's the deciding factor for whether a tool call can be safely retried.
2. Why does making an operation idempotent (via a dedup key) solve the retry-safety problem better than simply refusing to retry?
3. Why should risk classification live in one place that multiple systems (retries, approval) read from, rather than being independently decided by each?

**Implementation:** Build `execute_with_contract` and an idempotency-key-based dedup wrapper from memory.

**Debugging:** A production incident report shows a customer was charged twice for one action. Trace through exactly which tool-engineering rule from this part was most likely violated, and where in the code you'd look first.

**Architecture:** Design a compensating-transaction system for a multi-step agent workflow involving payment, inventory reservation, and shipping — where a failure partway through needs to safely undo whatever already succeeded.

---

**Next:** [Part XXXIV — Human-in-the-Loop](part34-human-in-the-loop.md), where the `requires_human_approval` flag from this part becomes a real, durable pause-and-resume mechanism.
