# Part XXXIX — Observability

## Before you start

**What you need already:** [Part XXVIII Stage 3](part28-agent-harness.md) — you built a minimal `Tracer` with a flat list of timestamped events. This part grows that into a real system that can answer "what exactly happened during this specific run, and why."

**What this part gives you:** The infrastructure that makes every other reliability and evaluation mechanism in this curriculum actually debuggable in production — because a system you can't observe is a system you can only guess about when something goes wrong.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Build a real tracing system with nested, causally-related spans, and know the difference between a latency problem you can see in an average and one that's hidden by it.

---

## Stage 1 — What your Part XXVIII tracer can't tell you

Go back to Part XXVIII's `Tracer`. It's a flat list of events — useful, but it can't answer a question you'll genuinely need answered in production: "this run took 4 seconds — where did those 4 seconds actually go?" A flat list of `{"type": "tool_call", ...}` entries doesn't preserve *which step contained which sub-steps*, or how long each one individually took relative to the whole run.

**Check yourself before Stage 2:** What specific question can a flat event list not answer that you'd need answered to debug "why was this run slow"?

---

## Stage 2 — Spans: a nested, timed record of what happened, and in what order

A **span** represents one unit of work — a model call, a tool call, an entire agent run — with a start time, an end time, and (crucially) a reference to whatever span it happened *inside* of. This nesting is what a flat event list was missing.

```python
from __future__ import annotations
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    attributes: dict = field(default_factory=dict)
    status: str = "in_progress"


class Tracer:
    def __init__(self):
        self.spans: list[Span] = []
        self._current_span_id: str | None = None

    @contextmanager
    def start_span(self, name: str, trace_id: str, **attributes):
        span = Span(
            span_id=str(uuid.uuid4())[:8], trace_id=trace_id,
            parent_span_id=self._current_span_id, name=name,
            start_time=time.monotonic(), attributes=attributes,
        )
        prev_parent = self._current_span_id
        self._current_span_id = span.span_id
        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            span.end_time = time.monotonic()
            self.spans.append(span)
            self._current_span_id = prev_parent
```

Walk through the `@contextmanager` carefully — this is the mechanism that makes nesting automatic. Before yielding, it records `self._current_span_id` (whatever span was active *before* this one started) as `parent_span_id`, then sets *itself* as the currently active span. Anything started inside this `with` block automatically becomes its child. On exit — success or exception, guaranteed by the `finally` — it restores the previous parent, so a sibling span started after this one correctly nests under the *outer* parent, not incorrectly under this now-finished span.

```python
tracer = Tracer()


def call_model(trace_id: str, prompt: str) -> str:
    with tracer.start_span("model_call", trace_id, model="gpt-4o-mini"):
        time.sleep(0.05)
        return "model response"


def call_tool(trace_id: str, tool_name: str) -> str:
    with tracer.start_span("tool_call", trace_id, tool=tool_name):
        time.sleep(0.02)
        return "tool result"


def agent_run(goal: str) -> None:
    trace_id = str(uuid.uuid4())[:8]
    with tracer.start_span("agent_run", trace_id, goal=goal):
        call_model(trace_id, goal)
        call_tool(trace_id, "search")
        call_model(trace_id, "synthesize")
```

**Check yourself before Stage 3:** Why does `_current_span_id` need to be restored to its previous value in the `finally` block, rather than just left as whatever it was set to?

---

## Stage 3 — Rendering the trace tree: the actual payoff

```python
    def print_trace_tree(self, trace_id: str) -> None:
        relevant = [s for s in self.spans if s.trace_id == trace_id]
        by_parent: dict[str | None, list[Span]] = {}
        for s in relevant:
            by_parent.setdefault(s.parent_span_id, []).append(s)

        def render(parent_id: str | None, depth: int) -> None:
            for span in by_parent.get(parent_id, []):
                duration = (span.end_time or time.monotonic()) - span.start_time
                print(f"{'  ' * depth}[{span.name}] {duration*1000:.1f}ms status={span.status} {span.attributes}")
                render(span.span_id, depth + 1)

        render(None, 0)


agent_run("research renewable energy")
tracer.print_trace_tree(tracer.spans[-1].trace_id)
```

```
[agent_run] 71.4ms status=ok {'goal': 'research renewable energy'}
  [model_call] 50.2ms status=ok {'model': 'gpt-4o-mini'}
  [tool_call] 20.1ms status=ok {'tool': 'search'}
  [model_call] 0.3ms status=ok {'model': 'gpt-4o-mini'}
```

This is the answer Stage 1's flat list couldn't give you: "where did the time go" is now directly visible — most of this run's 71ms was the first model call, not the tool call. In a real run with dozens of steps, this tree is what you'd actually read to diagnose "why was this specific run slow" or "why did this specific run fail" — you can see exactly which nested operation took long or errored, in the context of everything around it.

**Check yourself before Stage 4:** Given the printed tree, if this run had taken 500ms total instead of 71ms, how would you determine which specific sub-operation was responsible, just from reading the tree?

---

## Stage 4 — Percentiles: why an average hides the problem that actually matters

Here's a genuinely important distinction for production systems, connecting to [Part XXXVII Stage 5](part37-agent-testing.md)'s load test. Suppose you have 100 requests, 99 of which take 200ms and 1 of which takes 8 seconds. The average is about 280ms — looks fine. But that one request represents a real user who waited 8 seconds, and if your traffic is large enough, "1 in 100" isn't rare — it's happening constantly, to a steady stream of different users, every one of whom has a genuinely bad experience the average completely hides.

```python
from collections import defaultdict


def aggregate_metrics(spans: list[Span]) -> dict:
    by_name = defaultdict(list)
    for s in spans:
        if s.end_time:
            by_name[s.name].append((s.end_time - s.start_time) * 1000)

    report = {}
    for name, durations in by_name.items():
        durations.sort()
        n = len(durations)
        report[name] = {
            "count": n,
            "p50_ms": durations[int(n * 0.5)],
            "p95_ms": durations[min(int(n * 0.95), n - 1)],
            "p99_ms": durations[min(int(n * 0.99), n - 1)],
            "error_rate": sum(1 for s in spans if s.name == name and s.status == "error") / n,
        }
    return report
```

**p50** ("median") is the typical experience — half of requests are faster, half slower. **p95** and **p99** are the tail — the experience of your unluckiest 5% and 1% of requests. A healthy system has p99 not too much larger than p50; a large gap between them (p50 = 200ms, p99 = 8000ms) is a specific, real signal — something is intermittently going wrong (a retry firing, a cold cache, a struggling downstream dependency) that an average would never surface.

**Check yourself before the project:** In the 99-requests-at-200ms-and-1-at-8-seconds example, why does reporting only the average actively hide the problem, rather than just describing it imprecisely?

---

## Stage 5 — What must never end up in a trace

One more thing worth being explicit about before you wire this into a real system: traces get stored, often for a long time, often somewhere less tightly access-controlled than your primary application. Logging full prompts and responses verbatim, with no redaction, risks putting PII or secrets into that storage — directly connecting to [Part XXXV](part35-security.md)'s security discipline, now applied to the observability layer itself.

```python
import re


def redact(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL_REDACTED]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", text)
    return text
```

This is a starting point, not a complete solution — real redaction is genuinely hard to get exhaustively right. The important habit is treating trace content with the same caution as any other place sensitive data could leak, rather than assuming "it's just internal logs" makes it safe by default.

**Check yourself before the project:** Why is trace/log storage a real risk for PII leakage, even though it's "just internal debugging data"?

---

# The Project: A Tracing Dashboard for Your Agent Harness

## What you're building

Wire the full `Tracer` into your Part XXVIII harness, persist traces, and build a simple report/dashboard that surfaces per-span-type latency percentiles and error rates — plus redaction on anything logged.

```bash
$ python trace_dashboard.py --trace-id a1b2c3d4
```

```
[agent_run] 2340ms status=ok
  [model_call] 890ms status=ok {model: gpt-4o-mini}
  [tool_call] 1200ms status=error {tool: search, error: "timed out after 5s"}
  [tool_call] 45ms status=ok {tool: calculator}
  [model_call] 205ms status=ok {model: gpt-4o-mini}
```

```bash
$ python trace_dashboard.py --report
```

```
model_call:  count=340  p50=210ms  p95=680ms  p99=2100ms  error_rate=0.6%
tool_call:   count=520  p50=45ms   p95=1400ms  p99=5000ms  error_rate=3.8%

ALERT: tool_call p99 (5000ms) is 111x its p50 (45ms) — likely an
intermittent failure mode worth investigating.
```

## Why this project

This is the debugging infrastructure every later agent's problems will get diagnosed through — including the online-evaluation sampling from [Part XXXVIII](part38-agent-evaluation.md) and the incident response you'll formalize in [Part LXVIII](part68-production-readiness-checklist.md).

## Build it in stages

---

### Stage A — Nested spans, wired into your real harness

**Goal:** Every meaningful operation in your Part XXVIII harness — the run itself, each model call, each tool call — wrapped in `start_span`.

**Do:**
1. Add tracing calls throughout your harness's `run` method
2. Run a real multi-step agent task
3. Print the trace tree and confirm it accurately reflects the real nested structure of what happened

**You're done when:** A real run's trace tree correctly shows nesting and accurate durations for every sub-operation.

---

### Stage B — Persist traces, don't just keep them in memory

**Goal:** Traces survive past the current process — reusable across runs, queryable later.

**Do:**
1. Write spans to a simple persistent store (SQLite is fine) instead of just an in-memory list
2. Confirm you can look up and render a trace tree for a run from a *previous* process invocation

**You're done when:** You can trace a run, restart your script entirely, and still retrieve and render that same trace.

---

### Stage C — Aggregate metrics with real percentiles

**Goal:** `aggregate_metrics`, run against a real batch of traced runs.

**Do:**
1. Run at least 30–50 real agent requests (varying inputs) to generate a meaningful sample
2. Compute p50/p95/p99 per span type
3. Identify, from your own real numbers, whether any span type shows a concerning gap between p50 and p99

**You're done when:** You have real percentile numbers and, ideally, a genuine finding — some operation with a suspiciously large tail — that you can investigate.

---

### Stage D — Redaction, applied before storage

**Goal:** No raw email addresses, SSNs, or (if you can construct a realistic test case) API keys end up in stored trace attributes.

**Do:**
1. Apply `redact()` to any span attribute that might contain user-supplied or model-generated free text before it's persisted
2. Test it against a deliberately constructed input containing a fake email address and SSN-shaped string, confirm both get redacted in the stored trace
3. Confirm span attributes that are *not* free text (tool names, status codes) are left untouched

**You're done when:** You've confirmed redaction fires on realistic sensitive-looking content while leaving structural metadata intact.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Trace tree shows incorrect nesting (a span appears under the wrong parent) | `_current_span_id` restoration logic broken, likely from a span started outside a proper `with` block | Confirm every span is started via `with tracer.start_span(...)`, never called without the context manager |
| Percentile computation gives nonsensical results | Sorting bug, or computing on too small a sample to be meaningful | Confirm the list is sorted before indexing, and use a large enough sample (30+ is a reasonable minimum) |
| Redaction misses obvious PII | Regex too narrow (e.g., only matches one email format) | Test against several realistic input variations, not just one canonical example |

## Going further (optional)

1. **Alerting.** Extend `aggregate_metrics` to automatically flag any span type whose p99/p50 ratio exceeds a threshold, as in the example output's alert line.
2. **Correlate traces with evaluation scores.** For runs your Part XXXVIII evaluator scored poorly, pull up their full trace and see if a pattern emerges (e.g., low-scoring runs disproportionately involve a specific tool's errors).
3. **A minimal web view** rendering a trace tree in a browser instead of terminal text — useful once you're debugging traces regularly rather than one-off.

---

## Checkpoint

**Knowledge:**
1. What can a nested span tree tell you that a flat event list can't?
2. Why does a p99 latency spike matter even if it only affects 1% of requests?
3. Why is trace/log storage a real PII risk, even for "internal debugging" data?

**Implementation:** Build the nested `Tracer` with correct parent restoration, and `aggregate_metrics` with real percentiles, from memory.

**Debugging:** A trace shows a `model_call` span taking 4 seconds, but your provider's own dashboard shows sub-second latency for that exact request around the same time. What would you investigate to explain the discrepancy?

**Architecture:** Design a redaction policy that balances debuggability (you need enough detail to diagnose real issues) against PII risk (you can't store everything raw) for a production trace store.

---

**Next:** [Part XL — Cost Optimization](part40-cost-optimization.md), where the token/latency numbers you're now tracing become inputs to real, deliberate cost-control decisions.
