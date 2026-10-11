# Part XLIX — Distributed Agents

## Before you start

**What you need already:** [Part IX Stage 4](part09-agent-state.md) (distributed locks), [Part XXXIII](part33-tool-engineering.md) (idempotency), [Part XLVII Stage 3](part47-production-architecture.md) (why you enqueue rather than run synchronously).

**What this part gives you:** The real worker pool your Part XLVII gateway was enqueuing work *to* — with the specific delivery guarantee that makes a distributed system trustworthy: work is never silently lost, and — combined with idempotency — never silently duplicated either.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Build a worker pool where a crashed worker's in-flight task gets picked up by another worker, with a real test proving no task is ever lost or duplicated.

---

## Stage 1 — A guarantee you should stop hoping for

You'll sometimes hear "exactly-once delivery" as a goal. It's worth being precise about why that's not actually achievable across a real network, before building around a false promise.

Imagine a worker processes a task successfully, then crashes *before* it can tell the queue "I'm done, don't give this to anyone else." From the queue's point of view, it genuinely cannot distinguish "the worker died before finishing" from "the worker finished but died before reporting it." Any system has to pick a side of this ambiguity: either risk losing the task (if it assumes success), or risk it being redelivered to another worker (if it assumes failure). There is no way to eliminate the ambiguity itself — it's fundamental to any system where confirmation has to travel across a network that can drop messages or where a process can die mid-operation.

**Check yourself before Stage 2:** Why can't a queue system reliably distinguish "the worker crashed before finishing" from "the worker finished but crashed before confirming"?

---

## Stage 2 — The achievable guarantee: at-least-once delivery, plus idempotency

Given that ambiguity is unavoidable, real systems pick the *safer* side: **at-least-once delivery** — a task is guaranteed to be delivered at least once, possibly more than once, but never silently dropped. Then — and this is the part that makes it actually usable — you combine it with exactly the idempotency discipline you already built in [Part XXXIII](part33-tool-engineering.md): if a task might be delivered twice, make processing it twice produce the same effect as processing it once.

```
at-least-once delivery  +  idempotent processing  =  the practical
                                                       effect of exactly-once,
                                                       without needing the
                                                       impossible guarantee
```

**Check yourself before Stage 3:** Why does combining at-least-once delivery with idempotent processing achieve the *practical effect* of exactly-once, even though true exactly-once delivery isn't achievable?

---

## Stage 3 — Building it: consumer groups, and acking only after real success

```python
from __future__ import annotations
import asyncio
import json
import uuid
import redis.asyncio as aioredis

STREAM_NAME = "agent_tasks"
GROUP_NAME = "agent_workers"


async def ensure_group(r: aioredis.Redis) -> None:
    try:
        await r.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def enqueue_task(r: aioredis.Redis, payload: dict) -> str:
    task_id = payload.get("idempotency_key") or str(uuid.uuid4())
    payload["idempotency_key"] = task_id
    await r.xadd(STREAM_NAME, {"data": json.dumps(payload)})
    return task_id
```

Notice `idempotency_key` gets attached at enqueue time, generated once and carried through — this is exactly [Part XXXIII](part33-tool-engineering.md)'s idempotency-key requirement for non-idempotent side-effecting work, now applied at the task-queue level: every task carries the key that will let processing detect and safely ignore a redelivered duplicate.

```python
PROCESSED_IDS: set[str] = set()   # in production: durable (Postgres), not in-memory


async def process_task(payload: dict) -> None:
    key = payload["idempotency_key"]
    if key in PROCESSED_IDS:
        print(f"skip duplicate delivery: {key}")
        return
    print(f"processing task {key}")
    # ... real work here ...
    PROCESSED_IDS.add(key)


async def worker_loop(worker_name: str, r: aioredis.Redis) -> None:
    await ensure_group(r)
    while True:
        entries = await r.xreadgroup(GROUP_NAME, worker_name, {STREAM_NAME: ">"}, count=1, block=2000)
        if not entries:
            continue
        for _, messages in entries:
            for msg_id, fields in messages:
                payload = json.loads(fields["data"])
                try:
                    await process_task(payload)
                    await r.xack(STREAM_NAME, GROUP_NAME, msg_id)   # ONLY after real success
                except Exception as e:
                    print(f"{worker_name}: task failed, will be redelivered: {e}")
                    # deliberately NOT acked — Redis keeps it pending, claimable
                    # by another worker after a visibility timeout
```

The line that makes this all correct is `await r.xack(...)` placed **strictly after** `process_task` has genuinely completed — never before, never unconditionally. If `process_task` raises an exception, the message is *not* acknowledged, meaning Redis considers it still pending — it will eventually be redelivered (to this worker or another) rather than silently lost. This is Stage 1's guarantee made concrete: the system errs toward redelivery, never toward silent loss, and `PROCESSED_IDS` (Stage 2's idempotency check) is what makes a redelivery safe rather than a duplicate side effect.

**Check yourself before Stage 4:** What would go wrong if `xack` were called *before* `process_task`, rather than after it succeeds?

---

## Stage 4 — The subtlety worth naming: a durable dedup store, not memory

`PROCESSED_IDS` in Stage 3 is deliberately flagged as "in production: durable, not in-memory" — worth understanding exactly why an in-memory `set` isn't good enough. If the worker process itself crashes and restarts, an in-memory set is wiped clean — the worker would have no memory of which tasks it already processed, and a redelivered task (which is expected and normal under at-least-once delivery) would be treated as brand new, defeating the entire point of the idempotency check. The dedup record needs to survive exactly the same kind of crash your checkpoint store (Part IX) was built to survive — because it's protecting against exactly the same class of failure.

**Check yourself before the project:** Why does an in-memory `PROCESSED_IDS` set fail to protect against duplicate processing across a worker restart, even though it works fine for duplicates arriving while the worker stays running?

---

# The Project: A Distributed Worker Pool, Chaos-Tested

## What you're building

Extend the Redis Streams worker pool with a durable, Postgres-backed dedup store, and a chaos test that kills a worker mid-task and proves the task completes exactly once, end to end.

```bash
$ python chaos_test.py
```

```
enqueued task: idempotency_key=a1b2c3

worker-1 picked up task, began processing...
[SIMULATED CRASH: worker-1 killed after starting work, before acking]

worker-2 claimed the pending, unacked message after visibility timeout
worker-2 checked durable dedup store: key a1b2c3 NOT YET processed
worker-2 processed task, acked

VERIFICATION: real side effect (e.g., a row in a results table) exists
exactly once. No duplication despite the mid-task crash.
```

## Why this project

This is the delivery guarantee your Part XLVII gateway's enqueued work actually depends on — build it correctly and every agent task your platform processes inherits this same "never lost, never duplicated" property.

## Build it in stages

---

### Stage A — Durable dedup store

**Goal:** Replace `PROCESSED_IDS` with a real Postgres table, surviving worker restarts.

**Do:**
1. Create a small `processed_tasks` table (idempotency key, processed timestamp)
2. Rewrite `process_task` to check and record against this table instead of an in-memory set
3. Restart your worker process mid-testing and confirm dedup still works correctly across the restart

**You're done when:** Dedup state genuinely survives a worker process restart, unlike the in-memory version.

---

### Stage B — Real consumer-group worker pool

**Goal:** Multiple real worker processes, correctly sharing work from one Redis stream via a consumer group.

**Do:**
1. Run 2–3 worker instances simultaneously, all in the same consumer group
2. Enqueue several tasks, confirm they get distributed across workers (not all processed by one worker)
3. Confirm each task is acked exactly once, by exactly one worker

**You're done when:** You've watched real work distribute correctly across multiple concurrent workers.

---

### Stage C — Kill a worker mid-task, deliberately

**Goal:** The chaos test from the example output, for real.

**Do:**
1. Add a deliberate hook to a test task that lets you kill the worker process at a controlled point — after it's started "processing" (simulate real work with a side effect, like writing a row to a results table) but before it acks
2. Trigger the kill
3. Confirm, after a visibility timeout, another worker claims the pending message and completes it

**You're done when:** You've watched a real crash-and-recovery cycle happen, with the task's actual result (the results-table row) present exactly once afterward.

---

### Stage D — Verify no duplication, with a hard assertion

**Goal:** Not just "it seems fine" — a real, automated check.

**Do:**
1. Add a genuine side effect to your test task's processing (e.g., appending to a results table, or incrementing a counter)
2. After the chaos test completes, query that side effect's actual state
3. Assert it reflects exactly one execution, not two — even though the task was, in Redis's view, delivered at least twice (once to the crashed worker, once to the recovering one)

**You're done when:** You have a hard, automated assertion — not a visual inspection — confirming exactly-once *effect*, built on at-least-once delivery plus idempotency.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| A task gets processed twice despite the dedup check | Dedup check-and-record isn't atomic — two workers can both check "not yet processed" before either records it | Use a database-level uniqueness constraint (`INSERT ... ON CONFLICT DO NOTHING`) rather than a check-then-insert pattern, so the atomicity is guaranteed by Postgres, not your application code |
| A crashed worker's task is never picked up by another worker | No visibility-timeout/claim mechanism implemented — an unacked message just sits pending forever with nothing reclaiming it | Implement `XCLAIM` (or your queue system's equivalent) to reclaim messages pending longer than a threshold |
| Worker pool doesn't actually distribute load — one worker does everything | All workers accidentally using different consumer group names, so each sees the full stream independently rather than sharing it | Confirm all workers use the exact same `GROUP_NAME` |

## Going further (optional)

1. **A visibility-timeout reclaim loop** running separately from the main worker loop, periodically checking for and reclaiming stale pending messages.
2. **Dynamic worker scaling** — a simple script that spins up additional workers when the queue backlog grows beyond a threshold, and scales back down when it's empty.
3. **Per-`run_id` ordering**, using Part IX's distributed lock — if a single logical agent session's steps must execute in strict order even across multiple workers, combine the queue with a lock so no two workers process the same `run_id`'s steps concurrently.

---

## Checkpoint

**Knowledge:**
1. Why is true exactly-once delivery not achievable across a network boundary, and what do you build instead to get the same practical effect?
2. What would go wrong if a message were acked *before* its task processing completed, rather than after?
3. Why must the dedup store be durable rather than in-memory, specifically to protect against worker restarts?

**Implementation:** Build the consumer-group worker loop with ack-after-success and a durable, atomic dedup check, from memory.

**Debugging:** A distributed task was processed twice despite having an idempotency key. What's the most likely root cause — dedup store durability, key generation, or ack timing — and how would you distinguish between them?

**Architecture:** Design task scheduling and dynamic worker scaling for a workload with highly bursty demand (10x traffic spikes for short periods), building on this part's delivery guarantees.

---

**Next:** [Part L — Advanced Orchestration](part50-advanced-orchestration.md), where the fixed DAGs from Part XXVI grow dynamic structure at runtime, and durable execution gets applied at the graph-node level instead of the whole-agent level.
