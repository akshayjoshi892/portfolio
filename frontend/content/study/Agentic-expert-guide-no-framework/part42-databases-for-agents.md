# Part XLII — Databases for Agents

## Before you start

**What you need already:** [Part IX](part09-agent-state.md) (Postgres for checkpointing), [Part I Stage 9](part01-prerequisites.md) (Redis for caching/locks), [Part XVI](part16-qdrant-deep-dive.md) (Qdrant for vectors). You've used each in isolation — this part is about deliberately choosing which one fits which job, in one system.

**What this part gives you:** A decision framework for "which store handles this data," grounded in the specific properties of each store you've already directly experienced, plus assembling all three into one working, tested data layer.

**Time:** 2 days.

**When you finish, you'll be able to:** Justify, from firsthand experience rather than a rule of thumb, exactly why a given piece of data belongs in Postgres, Redis, or Qdrant — and combine all three correctly in one system.

---

## Stage 1 — You already know the answer, from having used each one for its actual strength

Rather than presenting an abstract comparison, recall what you personally used each store *for*, and why the alternative would have been worse for that specific job.

**Postgres**, in Part IX: you needed atomic, versioned writes with a real transactional guarantee — `WHERE version = $3` had to be checked and updated as one indivisible operation, or two concurrent workers could corrupt each other's checkpoints. Redis or Qdrant don't give you that same transactional guarantee out of the box; Postgres does, because relational databases are built specifically around exactly this kind of correctness guarantee.

**Redis**, in Part I and Part IX: you needed a distributed lock — `SET NX EX` — fast, ephemeral, shared across processes. Postgres *can* do locking, but Redis is purpose-built for exactly this kind of high-frequency, low-latency, short-lived coordination, and it's dramatically faster for that specific job.

**Qdrant**, in Parts XV–XVI: you needed similarity search over millions of vectors in milliseconds. Neither Postgres nor Redis can do this efficiently at scale — it requires the specialized indexing structure (HNSW) you built by hand and then used properly in a real database.

**Check yourself before Stage 2:** For each of the three stores, state the one specific property that made it the right choice for the job you used it for, based on your own experience, not a memorized rule.

---

## Stage 2 — The decision, generalized

| Need | Store | Why, from what you've directly seen |
|---|---|---|
| Strong consistency, relational queries, an audit trail | Postgres | Transactional guarantees (Part IX), structured queries |
| Sub-millisecond ephemeral state, locks, rate limits | Redis | Speed for short-lived, high-frequency operations (Part I, IX) |
| Semantic similarity search | Qdrant | Purpose-built ANN indexing (Part XV–XVI) |
| Large binary blobs (PDFs, images) | Object storage (not covered hands-on, but the principle: never inline large blobs in a relational row) | Keeps relational tables small and fast; blobs have completely different access patterns |

The generalized rule underneath this table: **each store makes a specific trade-off, and you choose based on which trade-off matches the data's actual access pattern** — not based on "which database do we already have running," which is how real systems accidentally end up storing vectors in Postgres rows or trying to use Redis as a system of record.

**Check yourself before Stage 3:** Why would storing a large PDF file directly as a column value in a Postgres row be a bad idea, even though Postgres technically supports large binary columns?

---

## Stage 3 — Assemble the three into one coherent data layer

```python
from __future__ import annotations
from dataclasses import dataclass
import asyncpg
import redis.asyncio as aioredis


@dataclass
class DataLayer:
    pg_pool: asyncpg.Pool
    redis: aioredis.Redis

    @classmethod
    async def connect(cls, pg_dsn: str, redis_url: str) -> "DataLayer":
        pg_pool = await asyncpg.create_pool(pg_dsn, min_size=2, max_size=20)
        redis_client = aioredis.from_url(redis_url)
        return cls(pg_pool=pg_pool, redis=redis_client)

    async def save_checkpoint(self, run_id: str, state_json: str, version: int) -> bool:
        # exactly Part IX's optimistic-concurrency pattern
        async with self.pg_pool.acquire() as conn:
            result = await conn.execute(
                """
                INSERT INTO checkpoints (run_id, state, version, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (run_id) DO UPDATE
                    SET state = $2, version = $3, updated_at = now()
                    WHERE checkpoints.version = $3 - 1
                """,
                run_id, state_json, version,
            )
            return result.endswith("1")

    async def acquire_session_lock(self, run_id: str, ttl_s: int = 30) -> bool:
        # exactly Part IX Stage 6's distributed lock pattern
        return bool(await self.redis.set(f"lock:{run_id}", "1", nx=True, ex=ttl_s))

    async def release_session_lock(self, run_id: str) -> None:
        await self.redis.delete(f"lock:{run_id}")
```

Notice `save_checkpoint` and `acquire_session_lock` are exactly the code you already wrote in Part IX — nothing new mechanically. What's new here is the *composition*: one class, `DataLayer`, giving your application a single, coherent entry point to both stores, rather than scattering raw `asyncpg`/`redis` calls throughout your codebase. This is the same "compose, don't scatter" principle from [Part XXVIII](part28-agent-harness.md)'s harness, applied to your data access layer instead of your agent loop.

**Check yourself before Stage 4:** Why is composing checkpoint saves and lock acquisition into one `DataLayer` class better than calling `asyncpg` and `redis` directly, scattered throughout your application code?

---

## Stage 4 — A real audit-log schema, and the indexes it actually needs

One more concrete piece worth building deliberately: an audit trail for tool calls (connecting to [Part XXIV Stage 5](part24-mcp-server-development.md)'s audit logging), with the specific indexes that make it usable during an actual incident, not just a write-only log nobody can query efficiently.

```sql
CREATE TABLE tool_audit_log (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args JSONB NOT NULL,
    result JSONB,
    status TEXT NOT NULL,
    actor TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_run_id ON tool_audit_log (run_id);
CREATE INDEX idx_audit_tool_name ON tool_audit_log (tool_name);
```

Why these two specific indexes, not just "index everything"? Because they match the two real questions you'll actually ask during an incident: "what happened during this specific run" (`WHERE run_id = ?`) and "every time this specific tool was called, across all runs, show me" (`WHERE tool_name = ?`). Indexes you'll never query by are wasted write overhead; indexes matching your actual incident-response questions are what make the audit log genuinely useful under pressure, rather than a slow table nobody can practically search.

**Check yourself before the project:** Why index specifically on `run_id` and `tool_name`, rather than, say, every column in the table?

---

# The Project: A Full Data Architecture, Crash-Tested

## What you're building

Postgres for state/audit, Redis for locks, Qdrant for retrieval — deployed together, with a test proving a simulated crash-and-resume never duplicates a tool-audit entry for the same logical action.

```bash
$ docker compose up -d
```

```
postgres  Up
redis     Up
qdrant    Up
```

```bash
$ python data_layer_test.py --simulate-crash
```

```
run started, checkpoint saved after step 1
[simulated crash before step 2's checkpoint]
resuming run...
step 2 re-executed (expected, since it wasn't checkpointed)
audit log check: exactly 1 entry for step 2's tool call (no duplication)

ALL CHECKS PASSED
```

## Why this project

This is the data foundation every later production-focused part assumes — [Part XLVII](part47-production-architecture.md)'s reference architecture and [Part XLIX](part49-distributed-agents.md)'s distributed workers both build directly on this pattern.

## Build it in stages

---

### Stage A — Docker Compose for all three stores

**Goal:** Postgres, Redis, and Qdrant running together, with persistent volumes.

**Do:**
1. Write a `docker-compose.yml` bringing up all three
2. Confirm each is reachable and data survives a `docker compose restart`

**You're done when:** All three services are up, and you've confirmed (by writing and re-reading test data) that each one's persistence actually works.

---

### Stage B — `DataLayer`, with checkpoint and lock methods

**Goal:** The class from Stage 3, fully implemented and tested.

**Do:**
1. Implement `connect`, `save_checkpoint`, `acquire_session_lock`, `release_session_lock`
2. Write a quick test confirming a version conflict is correctly detected (reuse Part IX's test)
3. Write a quick test confirming a lock can't be acquired twice concurrently

**You're done when:** Both tests pass against real Postgres and Redis instances.

---

### Stage C — The audit log, wired into a real agent

**Goal:** Every tool call in a real agent run gets logged, queryable by `run_id` and `tool_name`.

**Do:**
1. Create the schema with its indexes
2. Wire audit logging into your Part XXVIII harness's tool execution path
3. Run a real multi-step agent task, then query the audit log by `run_id` and confirm it shows a complete, accurate record

**You're done when:** You can reconstruct exactly what a run did, purely from querying the audit log.

---

### Stage D — The crash-and-resume test, with Qdrant retrieval in the mix

**Goal:** The full example output — a simulated crash mid-run, resumed correctly, with no duplicated audit entries, and a RAG step (querying Qdrant) working correctly throughout.

**Do:**
1. Build an agent run that includes at least one retrieval step (Qdrant) and one side-effecting, idempotency-key-protected tool call (reusing Part XXXIII)
2. Simulate a crash after step 1's checkpoint but before step 2's
3. Resume, confirm step 2 correctly re-executes, and confirm the audit log shows exactly one entry for it (not duplicated), thanks to the idempotency-key discipline

**You're done when:** You've watched the full cycle work correctly, end to end, across all three stores, with real evidence (the audit log query) that nothing was duplicated.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Connection pool exhaustion under concurrent testing | `asyncpg.create_pool`'s `max_size` too small, or connections not being released properly | Increase pool size for testing, and confirm you're using `async with self.pg_pool.acquire()` correctly everywhere |
| Data disappears after `docker compose restart` | Missing or misconfigured volume mounts for one of the services | Check each service's volume configuration in your compose file |
| Audit log query by `run_id` is slow at real scale | Index not actually being used, or missing entirely | Confirm the index exists and check your query plan (`EXPLAIN`) to verify it's being used |

## Going further (optional)

1. **A read replica for Postgres**, and think through which of your queries (checkpoint writes, audit log reads) should go to the primary versus could be served from a replica.
2. **Redis persistence configuration** (`appendonly yes`) — confirm locks and any other Redis-stored state you rely on actually survive a Redis restart, not just a graceful shutdown.
3. **A schema migration test** — add a new column to your checkpoint schema and confirm old, already-saved checkpoints still load correctly (connecting back to Part IX's versioning discussion).

---

## Checkpoint

**Knowledge:**
1. Justify, from your own direct experience in earlier parts, why Postgres was the right choice for checkpointing and Redis was the right choice for locking.
2. Why shouldn't a large binary blob (a PDF, an image) be stored directly as a Postgres column value?
3. Why index an audit log specifically on the columns you'll actually query by, rather than indexing everything?

**Implementation:** Build `DataLayer` with checkpoint and lock methods, plus the audit log schema with correct indexes, from memory.

**Debugging:** Two concurrent processes both believe they hold the session lock for the same `run_id`. What Redis configuration or code issue would you check first?

**Architecture:** Design the full data architecture (which store, for which concern) for an enterprise-scale multi-agent platform, justifying every choice from your own hands-on experience with each store's actual properties.

---

**Next:** [Part XLIII — Knowledge Graphs](part43-knowledge-graphs.md), where you'll meet a kind of retrieval that vector search structurally cannot do — precise, multi-hop relationship queries.
