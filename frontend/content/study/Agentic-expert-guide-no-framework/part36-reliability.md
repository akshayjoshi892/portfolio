# Part XXXVI — Reliability

## Before you start

**What you need already:** [Part I](part01-prerequisites.md) (retry/backoff), [Part VI](part06-structured-outputs-tool-calling.md) (timeouts, circuit breakers), [Part IX](part09-agent-state.md) (checkpointing), [Part XXXIII](part33-tool-engineering.md) (idempotency).

**What this part gives you:** These mechanisms have all been built in isolation, each solving one specific failure. This part is about composing them into a coherent strategy for a whole system, plus one new piece: what to do with a task that has genuinely, permanently failed, rather than pretending it might still succeed.

**Time:** 2 days.

**When you finish, you'll be able to:** Build a system where a permanently-failing task doesn't get silently lost, and degrade gracefully — visibly, not silently — when a dependency is unavailable.

---

## Stage 1 — What happens to a task that never succeeds

Every retry mechanism you've built has a ceiling — a `max_attempts`, after which it gives up. Fair enough. But what happens to the task *after* it gives up?

```python
async def process_with_retry(task, process_fn, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return await process_fn(task)
        except Exception:
            if attempt < max_attempts:
                await asyncio.sleep(2 ** attempt)
    return None   # <-- the task just... vanishes
```

`return None` is the problem worth staring at. The task failed permanently. Nothing recorded that it failed, why, or that it needs someone's attention. If this were processing a batch of 10,000 documents overnight, and 40 of them permanently failed, you'd have no way to know which 40, or that anything went wrong at all — the job "completed," it just silently completed less work than it should have.

**Check yourself before Stage 2:** Why is silently discarding a permanently-failed task worse than a loud crash, even though the crash would be more disruptive in the moment?

---

## Stage 2 — The dead-letter queue: failed work goes somewhere, not nowhere

```python
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FailedTask:
    task_id: str
    payload: dict
    error: str
    attempts: int
    last_attempt_at: float = field(default_factory=time.time)


class DeadLetterQueue:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def push(self, task: FailedTask) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(task.__dict__) + "\n")


async def process_with_dlq(task_id: str, payload: dict, process_fn, dlq: DeadLetterQueue, max_attempts=3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await process_fn(payload)
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                await asyncio.sleep(2 ** attempt)

    # Exhausted retries — don't discard, PARK it for later inspection
    dlq.push(FailedTask(task_id=task_id, payload=payload, error=str(last_error), attempts=max_attempts))
    return None
```

The difference from Stage 1 is small in code, large in consequence: instead of silently vanishing, a permanently-failed task gets written to a durable, inspectable log — its ID, its original payload (so it can be retried later once the underlying issue is fixed), the error, and how many attempts it took. This is called a **dead-letter queue**, named after the postal system's dead-letter office for mail that can't be delivered — it doesn't get thrown away, it gets set aside for someone to look at.

**Check yourself before Stage 3:** What information does `FailedTask` preserve that would let you actually fix and retry the task later, rather than just knowing "something failed"?

---

## Stage 3 — Replaying: turning "parked" back into "processed"

A dead-letter queue is only useful if you can act on it — once whatever caused the failures is fixed (a bug patched, a downstream service recovered), you want to retry everything that got parked, not manually re-trigger each one.

```python
    def replay_all(self, process_fn) -> list[str]:
        if not self.path.exists():
            return []

        succeeded, remaining_lines = [], []
        for line in self.path.read_text().splitlines():
            task_data = json.loads(line)
            try:
                process_fn(task_data["payload"])
                succeeded.append(task_data["task_id"])
            except Exception:
                remaining_lines.append(line)   # still fails — stays parked

        self.path.write_text("\n".join(remaining_lines) + ("\n" if remaining_lines else ""))
        return succeeded
```

Notice this reuses the *original payload*, unchanged, and any task that still fails on replay stays in the queue rather than being lost — you can call `replay_all` repeatedly, safely, and it only ever removes tasks that genuinely succeed this time. This connects directly to [Part XXXIII](part33-tool-engineering.md): replaying a task that has real side effects is only safe if the underlying operation is idempotent, or if the task carries its own idempotency key — otherwise a task that actually *did* partially succeed before failing could get duplicated on replay.

**Check yourself before Stage 4:** Why is it important that `replay_all` uses the exact original payload, and why does idempotency matter specifically for the tasks you're replaying?

---

## Stage 4 — Degrading visibly, not silently

One more reliability concern worth its own treatment: what should happen when a whole *component* your system depends on is unavailable — not one task failing, but (say) your RAG retrieval system being down entirely?

```python
async def answer_with_degradation(query: str, rag_fn, llm_only_fn) -> dict:
    try:
        return {"mode": "rag", "answer": await rag_fn(query)}
    except Exception as e:
        print(f"RAG unavailable ({e}); degrading to ungrounded answer")
        answer = await llm_only_fn(query)
        return {
            "mode": "degraded_no_retrieval",
            "answer": answer,
            "warning": "Our knowledge base is temporarily unavailable — this answer "
                       "is not grounded in your documents and may be less accurate.",
        }
```

The critical detail: the returned dict includes `mode` and, when degraded, an explicit `warning`. This means whatever calls `answer_with_degradation` — a UI, an API consumer, a human — knows *for certain* whether it's getting a fully-grounded answer or a lower-confidence fallback. Compare this to a version that just silently returns whatever answer it managed to produce, with no indication of which path was taken: a user reading that answer has no way to know it's less trustworthy than usual, and might act on it with the same confidence they would a normal, grounded response.

**Check yourself before the project:** Why is a degraded response that's clearly labeled as degraded strictly better than one that silently looks identical to a normal response?

---

# The Project: A Reliable Batch Pipeline With Dead-Letter Recovery

## What you're building

A batch document re-processing pipeline (reusing Parts XI–XIV's ingestion/embedding work) with retry, a dead-letter queue, and a working replay mechanism — tested with real, injected failures.

```bash
$ python reliable_pipeline.py --process ./docs
```

```
processing 50 documents...
  47 succeeded
  3 failed after 3 attempts each, parked in dead-letter queue
     (doc_23: connection timeout, doc_31: malformed content, doc_44: embedding API error)

$ python reliable_pipeline.py --replay
replaying 3 parked tasks...
  doc_23: succeeded (transient issue resolved)
  doc_31: still failing (malformed content) — remains parked
  doc_44: succeeded

2 tasks recovered, 1 remains parked for manual review
```

## Why this project

This closes the loop on every reliability mechanism you've built — retries have a real place to send permanent failures, and permanent failures have a real path back to success once conditions change, rather than being lost or requiring painful manual re-triggering.

## Build it in stages

---

### Stage A — Real processing with injected failures

**Goal:** A batch job over real documents (reuse Parts XI–XIV) that occasionally, deliberately fails, so you have something real to recover from.

**Do:**
1. Build a processing function (chunk + embed + index, reusing earlier parts) for a folder of test documents
2. Inject deliberate, controllable failures — e.g., make processing fail for documents matching a specific pattern, or randomly fail a percentage of the time
3. Run the batch, observe which succeed and which fail

**You're done when:** You have a real batch run with a mix of successes and genuine failures.

---

### Stage B — Retry, then dead-letter on exhaustion

**Goal:** `process_with_dlq`, wired into your Stage A pipeline.

**Do:**
1. Implement it as shown
2. Re-run your batch, confirm transient-seeming failures get retried (and some may now succeed on a later attempt), while consistently-failing ones end up parked in the DLQ

**You're done when:** Your DLQ file contains real, informative records for every task that ultimately failed after exhausting retries.

---

### Stage C — Replay, and confirm it's genuinely safe

**Goal:** `replay_all`, correctly reprocessing parked tasks and correctly leaving still-failing ones parked.

**Do:**
1. "Fix" the underlying cause for some (but not all) of your injected failures — e.g., remove the failure-triggering condition for some document IDs but not others
2. Run `replay_all`
3. Confirm the now-fixable tasks succeed and are removed from the queue, while the still-broken ones remain parked

**You're done when:** You've watched a partial recovery happen correctly — some tasks recovered, others correctly remaining parked — matching the example output's shape.

---

### Stage D — Graceful degradation for a real dependency

**Goal:** `answer_with_degradation`, tested against a genuinely unavailable RAG backend.

**Do:**
1. Wire it into a real RAG query function (reuse Part XI/XVI)
2. Deliberately make the RAG backend unavailable (stop your Qdrant container, or point at a wrong URL)
3. Confirm queries still get answered, but clearly labeled as degraded, with the `warning` field present

**You're done when:** You've watched the degraded path fire for real, and confirmed the response is honestly labeled rather than indistinguishable from a normal, grounded answer.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Replaying a task duplicates a side effect it had already partially completed | Task processing isn't actually idempotent | Apply Part XXXIII's idempotency-key discipline to any task with real side effects before relying on replay safety |
| Dead-letter queue grows without ever being cleaned up | No process for reviewing/discarding tasks that will never succeed (e.g., permanently malformed input) | Add a manual review/discard step for tasks that fail replay repeatedly, distinct from tasks worth retrying |
| Degraded mode never actually triggers even when the dependency is down | Exception from the dependency being swallowed somewhere upstream instead of propagating to the `try/except` in `answer_with_degradation` | Trace the exception path — confirm the RAG function's real failure actually surfaces as an exception, not a silently-empty result |

## Going further (optional)

1. **A DLQ inspection CLI** — list parked tasks with their error and attempt count, sorted by how recently they failed, so a human reviewing the queue can prioritize.
2. **Categorize failures.** Distinguish "transient, worth auto-retrying on a schedule" from "permanent, needs a human" based on the error type, and only run `replay_all` automatically for the transient category.
3. **Combine with Part IX.** Checkpoint your batch job's overall progress (how many of 10,000 documents processed so far), so a crash partway through a very large batch doesn't require restarting the entire run from the beginning, even before any individual task's retry/DLQ logic kicks in.

---

## Checkpoint

**Knowledge:**
1. Why is silently discarding a permanently-failed task worse than logging it, even though both mean the task didn't succeed?
2. Why must a task be idempotent (or carry an idempotency key) to be safely replayable from a dead-letter queue?
3. Why is a clearly-labeled degraded response better than one that looks identical to normal output?

**Implementation:** Build `DeadLetterQueue` with push/replay, and `answer_with_degradation`, from memory.

**Debugging:** A nightly batch job's success count seems to mysteriously shrink over time, with no corresponding error reports. What reliability mechanism (DLQ, checkpointing, idempotency) is most likely missing, and why would that specific gap produce this specific symptom?

**Architecture:** Design a fault-tolerant pipeline for a 24-hour long-running multi-step agent task that must survive at least one process restart, composing checkpointing (Part IX), retries with a DLQ (this part), and idempotent step design (Part XXXIII).

---

**Next:** [Part XXXVII — Agent Testing](part37-agent-testing.md), where you'll build a real, layered test suite for everything you've built — because an agent system this complex needs more than manual spot-checking to trust.
