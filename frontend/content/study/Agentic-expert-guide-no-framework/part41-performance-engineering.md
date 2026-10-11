# Part XLI — Performance Engineering

## Before you start

**What you need already:** [Part III Stage 7](part03-llm-fundamentals.md) (prefill vs. decode), [Part I Stage 3](part01-prerequisites.md) (concurrency), [Part VI Stage 7](part06-structured-outputs-tool-calling.md) (parallel tool calls).

**What this part gives you:** A systematic way to decompose "why is my agent slow" into its actual component causes — because the fix for a slow prefill is completely different from the fix for slow decode, and applying the wrong one wastes effort without moving the number you care about.

**Time:** 2 days.

**When you finish, you'll be able to:** Break down a run's total latency into its real components, and apply the specific fix that matches the actual bottleneck instead of guessing.

---

## Stage 1 — Decompose the total, using what you already know

A single agent turn's latency has real, distinct components:

```
total latency
  = network round-trip
  + model prefill time  (scales with INPUT length — Part III)
  + model decode time   (scales with OUTPUT length — Part III)
  + tool call time (sequential, unless parallelized — Part VI)
  + retrieval time (Part XV)
  + your own orchestration overhead
```

Every component here is something you've already measured in isolation somewhere earlier in this curriculum. What's new in this part is treating them as a *decomposition* — measuring where a specific slow run's time actually went, rather than optimizing blindly.

**Check yourself before Stage 2:** If a run's decode time dominates its total latency, would shortening the input prompt meaningfully help? Why or why not, given what you know about prefill vs. decode from Part III?

---

## Stage 2 — The highest-leverage fix you already know how to apply: parallelize independent tool calls

You measured this exact speedup in [Part I Stage 3](part01-prerequisites.md). Applied specifically to agent tool calls:

```python
import asyncio
import time


async def call_tool(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} result"


async def sequential_tools(calls: list[tuple[str, float]]) -> list[str]:
    return [await call_tool(name, delay) for name, delay in calls]


async def parallel_tools(calls: list[tuple[str, float]]) -> list[str]:
    return await asyncio.gather(*(call_tool(name, delay) for name, delay in calls))


calls = [("search", 0.3), ("db_lookup", 0.2), ("weather", 0.25)]

start = time.monotonic()
asyncio.run(sequential_tools(calls))
print(f"sequential: {time.monotonic() - start:.2f}s")   # ~0.75s

start = time.monotonic()
asyncio.run(parallel_tools(calls))
print(f"parallel: {time.monotonic() - start:.2f}s")     # ~0.3s
```

The rule this generalizes to: **whenever a model requests multiple tool calls in one response with no data dependency between them, run them concurrently.** Most tool-calling APIs support returning multiple tool calls in one response for exactly this reason — the model has already told you these are independent by requesting them together.

**Check yourself before Stage 3:** Why is it safe to parallelize `search` and `weather` calls but not safe to parallelize a call that depends on another call's result?

---

## Stage 3 — A mistake that costs latency invisibly: rebuilding your HTTP client

```python
import httpx

# BAD: a new client, and a new TCP/TLS handshake, on every single call
async def call_api_slow(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        return response.json()


# GOOD: one client, its connection pool reused across every call
_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))

async def call_api_fast(url: str, payload: dict) -> dict:
    response = await _client.post(url, json=payload, timeout=30)
    return response.json()
```

This is easy to miss because it doesn't show up as a bug — it works correctly, it's just slower than it needs to be, silently, on every single call. Establishing a new connection (the TCP handshake, then the TLS handshake) has real, fixed overhead — often tens of milliseconds — that a reused, persistent connection pool avoids entirely for calls after the first. At real request volume, this adds up to a meaningful, invisible latency tax you're paying for no reason.

**Check yourself before Stage 4:** Why doesn't `call_api_slow` produce incorrect results, only slower ones — and why does that make this specific mistake easy to miss during development?

---

## Stage 4 — Caching: only safe for genuinely deterministic requests

You've built caching once already, in [Part XLI's cousin concept](part03-llm-fundamentals.md)'s prompt caching. Here's a response-level cache, and the one condition that must hold for it to be *correct*, not just fast:

```python
import hashlib
import json
import time


class ResponseCache:
    def __init__(self, ttl_s: float = 300):
        self.ttl_s = ttl_s
        self._store: dict[str, tuple[float, str]] = {}

    def _key(self, prompt: str, params: dict) -> str:
        blob = json.dumps({"prompt": prompt, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, prompt: str, params: dict) -> str | None:
        key = self._key(prompt, params)
        entry = self._store.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if time.time() - cached_at > self.ttl_s:
            del self._store[key]
            return None
        return value

    def set(self, prompt: str, params: dict, value: str) -> None:
        self._store[self._key(prompt, params)] = (time.time(), value)
```

The critical condition, worth stating explicitly because getting it wrong is a correctness bug disguised as a performance win: **only cache requests that are genuinely deterministic and not user-specific.** A `temperature=0` request with no user-specific content in the prompt is a reasonable candidate. A request containing "the user's name is Jane, respond personally" is not — caching it means the *next* user with a superficially similar-but-different request could receive Jane's cached, personalized response. That's not a performance bug. It's a data leak, and it connects directly back to [Part XXXV](part35-security.md)'s discipline about not letting one user's data leak into another's context.

**Check yourself before the project:** Give an example of a request that's safe to cache, and one that isn't, and explain exactly why the unsafe one is dangerous — not just "wrong," but specifically why.

---

# The Project: Performance Benchmarks, Before and After

## What you're building

A benchmark harness measuring the real effect of parallelization, connection reuse, and caching on your Part XXVIII harness-based agent — quantified, not assumed.

```bash
$ python perf_benchmark.py
```

```
Baseline (sequential tools, new HTTP client per call, no cache):
  p50: 2.1s   p95: 3.4s

+ Parallel tool calls:
  p50: 1.3s   p95: 2.0s   (-38% p50)

+ Reused HTTP client:
  p50: 1.1s   p95: 1.7s   (-15% further)

+ Response caching (on cacheable requests only):
  p50: 0.4s   p95: 1.6s   (dramatic improvement for repeated queries;
                            p95 barely changes since most requests are unique)
```

## Why this project

This is the measured, evidence-based version of "make it faster" — you'll walk away knowing exactly which lever moved which number, and by how much, for your own real system.

## Build it in stages

---

### Stage A — Baseline measurement

**Goal:** Real p50/p95 numbers for your current agent, unoptimized.

**Do:**
1. Run your Part XXVIII harness-based agent against 30+ varied requests
2. Record latency for each, compute p50/p95 (reuse Part XXXIX's percentile code)

**You're done when:** You have real baseline numbers to compare every subsequent change against.

---

### Stage B — Parallelize tool calls, re-measure

**Goal:** Confirm the specific improvement from concurrent independent tool calls.

**Do:**
1. Identify or construct test cases where your agent legitimately requests 2+ independent tool calls in one turn
2. Apply `asyncio.gather` to them
3. Re-run your 30+ request benchmark, compare p50/p95 against baseline

**You're done when:** You have a real before/after number specifically attributable to this change.

---

### Stage C — Reuse the HTTP client, re-measure

**Goal:** Confirm the connection-pooling improvement.

**Do:**
1. Fix any per-call client instantiation in your `LLMClient` or tool implementations
2. Re-run the benchmark
3. Compare against Stage B's numbers

**You're done when:** You have a real number for this specific change, isolated from the others.

---

### Stage D — Response caching, applied only where safe

**Goal:** Caching, correctly scoped to deterministic, non-user-specific requests only.

**Do:**
1. Identify which of your test requests are genuinely cache-safe (deterministic, not user-specific) versus not
2. Apply `ResponseCache` only to the safe category
3. Re-run the benchmark with some repeated cache-safe queries included, measure the improvement
4. Explicitly verify: run a cache-unsafe (user-specific) request twice with different user contexts, confirm it does NOT return a cached, wrong-user response

**You're done when:** You have real numbers showing caching's benefit on repeated safe queries, and — just as importantly — a passing test confirming unsafe queries are never incorrectly cached.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Parallelizing tool calls shows no improvement | The "independent" tool calls actually have a hidden data dependency, so they can't safely run concurrently, or weren't genuinely parallelized (still awaited one at a time in a loop) | Confirm you're using `gather(*(...))`, and confirm there's truly no dependency between the calls |
| Connection reuse shows no improvement | Client was already being reused correctly, or the bottleneck genuinely lies elsewhere (verify with Stage A's decomposition) | Use Part XXXIX's tracing to see which specific span dominates before assuming this fix applies |
| Caching accidentally serves one user's cached response to another | Cache key doesn't include user/session-scoping for requests that should have it | Include user/session identifiers in the cache key for anything remotely personalized, or exclude such requests from caching entirely |

## Going further (optional)

1. **Trace-driven root cause.** For your specific p95 outlier requests, pull their full Part XXXIX trace and identify exactly which span dominates — confirm your intuition about the bottleneck with real data, not assumption.
2. **Tune retrieval performance**, reusing Part XV's `ef` trade-off, and measure its contribution to overall agent latency specifically.
3. **A latency budget per component**, allocating a target (e.g., "retrieval must stay under 100ms, tool calls under 500ms combined") and alerting when any component exceeds its budget in production.

---

## Checkpoint

**Knowledge:**
1. Why would shortening a prompt not help a run whose latency is dominated by decode time, not prefill?
2. Why does reusing an HTTP client improve latency without changing correctness at all?
3. What's the specific condition a request must meet to be safely cacheable, and what goes wrong (not just "slow," but genuinely wrong) if you cache a request that doesn't meet it?

**Implementation:** Build `parallel_tools`, a reused-connection HTTP wrapper, and a correctly-scoped `ResponseCache` from memory.

**Debugging:** p95 latency for an agent endpoint spikes intermittently, but p50 looks completely normal. Using Part XXXIX's tracing, what would you look for to explain an intermittent, not-constant, tail-latency problem?

**Architecture:** Design a caching strategy that's safe for a multi-tenant agent product with per-user personalization, without either leaking data across users or giving up all caching benefit.

---

**Next:** [Part XLII — Databases for Agents](part42-databases-for-agents.md), where the Postgres, Redis, and Qdrant pieces you've used throughout this curriculum get assembled into one coherent, intentional data architecture.
