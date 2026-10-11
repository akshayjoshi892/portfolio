# Part XLVIII — Deployment

## Before you start

**What you need already:** [Part XLVII](part47-production-architecture.md) (the gateway/queue/worker architecture), [Part I Stage 9](part01-prerequisites.md) (your first Docker containers), [Part XVIII Stage 6](part18-rag-evaluation.md) (the regression gate concept).

**What this part gives you:** Taking the architecture from Part XLVII and actually deploying it — with health checks that mean something specific for an agent worker, and a CI pipeline that uses your Part XVIII/XXXVII evaluation and test suites as real deploy gates, not just documentation.

**Time:** 2 days.

**When you finish, you'll be able to:** Write health checks that correctly distinguish "can't reach my dependencies" from "I'm alive but my agent loop is stuck," and design a CI pipeline that actually blocks a bad deploy using tools you already built.

---

## Stage 1 — Two different questions, two different checks

Container orchestrators (Kubernetes, and simpler setups too) ask a running service two genuinely different questions, and conflating them causes real production problems.

**Readiness**: "Can you currently handle new work?" — if no, stop routing new requests here, but don't kill the process; it might recover.

**Liveness**: "Are you still alive and functioning at all?" — if no, something is badly wrong and the process should be restarted.

For an ordinary web service, these are similar. For an agent worker, they're meaningfully different, and it's worth thinking through why before writing the checks.

**Check yourself before Stage 2:** What's the practical difference in what an orchestrator *does* in response to a failed readiness check versus a failed liveness check?

---

## Stage 2 — What readiness actually means for an agent worker

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/ready")
async def readiness_check():
    checks = {
        "database": await check_postgres_connection(),
        "queue": await check_redis_connection(),
        "vector_db": await check_qdrant_connection(),
    }
    all_healthy = all(checks.values())
    return {"ready": all_healthy, "checks": checks}, (200 if all_healthy else 503)
```

A worker is "ready" specifically when it can reach every dependency it needs to actually do its job — Postgres (checkpoints), Redis (the queue), Qdrant (retrieval), from [Part XLII](part42-databases-for-agents.md). If any of these is unreachable, routing new work to this worker would just mean that work fails immediately — better to mark it not-ready and let the orchestrator route new work elsewhere until connectivity recovers, exactly the graceful-degradation instinct from [Part XXXVI](part36-reliability.md), applied at the infrastructure level instead of inside a single request.

**Check yourself before Stage 3:** Why is "can I reach my dependencies" the right readiness check for a worker, rather than something simpler like "is the process running"?

---

## Stage 3 — What liveness means, and why it can catch a real, specific bug class

```python
import time

_last_heartbeat = time.monotonic()


def record_heartbeat() -> None:
    """Called at the top of every agent loop iteration (Part VII's loop),
    proving the loop is actually progressing, not stuck."""
    global _last_heartbeat
    _last_heartbeat = time.monotonic()


@app.get("/health")
async def liveness_check():
    stale_for = time.monotonic() - _last_heartbeat
    if stale_for > 300:   # no loop progress in 5 minutes — something is badly wrong
        return {"alive": False, "stale_for_s": stale_for}, 503
    return {"alive": True}, 200
```

Here's the genuinely interesting connection worth making explicit: what would cause `_last_heartbeat` to go stale? One real possibility is exactly [Part VII](part07-what-is-an-agent.md)'s worst-case scenario — a bug in an agent's termination logic causing it to loop without ever hitting its intended stopping condition (imagine the `max_steps` check itself having a bug, or a tool call hanging in a way your Part VI timeout somehow didn't catch). A liveness check failing specifically because heartbeats have gone stale is a direct, production-level signal that one of your earlier bounding mechanisms may have failed — worth treating with real urgency, not as routine infrastructure noise, because it's telling you a safety mechanism you built specifically to prevent this may itself have a bug.

**Check yourself before Stage 4:** Why should a liveness-check failure for an agent worker be treated as a potential loop-bounding bug, not just ordinary infrastructure flakiness?

---

## Stage 4 — Rolling deployment, and the setting that prevents a capacity dip

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-worker
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
        - name: worker
          image: registry.example.com/agent-worker:1.4.2
          readinessProbe:
            httpGet: { path: /ready, port: 8080 }
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8080 }
            periodSeconds: 20
```

`maxUnavailable: 0` is the setting worth understanding precisely: it tells the orchestrator "never drop below 3 running, ready replicas during a rollout." Combined with `maxSurge: 1` (allow one *extra* replica temporarily during the transition), the rollout proceeds by bringing up a new version alongside the old ones, waiting for it to pass its readiness probe, *then* retiring an old one — never dipping capacity, because the readiness probe is the gate that decides when it's actually safe to route traffic to the new version.

**Check yourself before Stage 5:** Why does `maxUnavailable: 0` combined with a real readiness probe (not just "process started") prevent a capacity dip during deployment, where a naive "kill old, start new" approach wouldn't?

---

## Stage 5 — The CI pipeline: your own evaluation and test suites, made into a gate

This is where [Part XVIII](part18-rag-evaluation.md)'s RAG regression gate and [Part XXXVII](part37-agent-testing.md)'s layered test suite stop being tools you run manually and become an automated, non-negotiable step before any deploy.

```text
Push to branch
   -> Lint + deterministic unit tests (Part XXXVII, fast — seconds)
   -> Build container image
   -> Integration tests against real dependencies (Part XXXVII — Postgres/Redis/Qdrant in CI)
   -> RAG/prompt regression eval gate (Part XVIII — FAILS THE BUILD on regression)
   -> Deploy to staging
   -> Adversarial/security smoke test (Part XXXV's exploit test, re-run against staging)
   -> Manual or automated promotion to production (rolling deployment, Stage 4)
```

The critical realization: **you already built every one of these gates in an earlier part.** This stage isn't introducing new testing concepts — it's about wiring existing checks (that you might currently run manually, when you remember to) into something that runs automatically, on every change, and — this is the part that actually matters — genuinely blocks a bad deploy rather than just producing a report someone might read.

**Check yourself before the project:** For each stage in this pipeline, name the earlier Part where you built the actual mechanism being run.

---

# The Project: A Full CI/CD Pipeline for Your Agent System

## What you're building

Docker Compose for local development, Kubernetes manifests with real health checks, and a CI pipeline (using GitHub Actions or your CI tool of choice) that runs your Part XVIII and Part XXXVII suites as genuine, build-blocking gates.

```bash
$ git push origin feature/new-prompt
```

```
✓ lint + unit tests (12s)
✓ build image (34s)
✓ integration tests (28s)
✗ RAG regression eval: recall@5 regressed 0.85 -> 0.60 (exceeds tolerance)

BUILD FAILED — deployment blocked.
```

## Why this project

This is where everything you've built across 47 parts stops being individually-tested code and becomes a real, continuously-verified system — the difference between "I have good tests" and "bad changes structurally cannot reach production."

## Build it in stages

---

### Stage A — Readiness and liveness endpoints, tested against real failures

**Goal:** `/ready` and `/health`, correctly distinguishing dependency-unavailable from loop-stuck.

**Do:**
1. Implement both against your Part XLVII worker
2. Stop your Postgres container — confirm `/ready` correctly returns unhealthy while `/health` still reports alive (the process itself is fine, just can't reach a dependency)
3. Simulate a stuck loop (temporarily disable heartbeat recording) — confirm `/health` eventually reports not-alive

**You're done when:** You've caused both failure types deliberately and confirmed each check responds correctly and distinctly.

---

### Stage B — Docker Compose and Kubernetes manifests

**Goal:** The full stack (gateway, worker, Postgres, Redis, Qdrant) running via Compose locally, and equivalent Kubernetes manifests for the worker with correct health check wiring.

**Do:**
1. Extend your Part XLII Compose file with the gateway and worker services
2. Write Kubernetes manifests for the worker deployment, with readiness/liveness probes wired to Stage A's endpoints
3. Confirm the readiness probe actually gates traffic — a worker that fails `/ready` shouldn't receive new work

**You're done when:** The full stack runs locally via Compose, and you have Kubernetes manifests ready for a real cluster (test against a local cluster like `kind` or `minikube` if available).

---

### Stage C — Rolling deployment, verified with zero capacity dip

**Goal:** A real image version bump, deployed as a rolling update, with zero request failures during the transition.

**Do:**
1. Deploy version 1 of your worker
2. Run a simple load-testing script against it continuously (reuse Part XXXVII's load test)
3. While the load test runs, deploy version 2 as a rolling update
4. Confirm zero failed requests during the transition, and that traffic correctly shifts to the new version only after it passes readiness

**You're done when:** You've watched a real rolling update complete with continuous, uninterrupted traffic throughout.

---

### Stage D — The CI pipeline, genuinely blocking a bad deploy

**Goal:** The exact pipeline from Stage 5, with a real regression that gets caught.

**Do:**
1. Set up CI running your Part XXXVII test suite and Part XVIII evaluation suite
2. Deliberately introduce a regression (worsen your chunk size, weaken a prompt) in a branch
3. Push it, confirm the CI pipeline fails specifically at the evaluation gate, with a clear message about what regressed
4. Fix the regression, confirm the pipeline passes and would proceed to deployment

**You're done when:** You've watched your own CI pipeline genuinely block a deliberately-introduced bad change, with a clear, specific reason reported.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Readiness probe never passes even when dependencies are healthy | Probe timeout too short for the actual connection-check latency | Increase `periodSeconds`/timeout, or diagnose why the connection check itself is slow |
| Rolling update causes a brief spike in failed requests | `maxUnavailable` not actually set to 0, or readiness probe too permissive (passes before the worker is genuinely ready) | Confirm both the manifest setting and that `/ready` genuinely checks all real dependencies |
| CI pipeline passes despite an obvious regression | Evaluation gate step not actually configured to fail the build on a failing check (e.g., exit code ignored) | Confirm the CI step's exit code is checked and treated as a real build failure, not just logged |

## Going further (optional)

1. **Canary deployment** — route a small percentage of real traffic to the new version before a full rollout, using your evaluation metrics on that canary traffic as an additional automated gate.
2. **Automated rollback** — if error rates or evaluation scores spike after a deploy, automatically revert to the previous version rather than requiring manual intervention.
3. **Secrets management**, connecting to Part XXXV — confirm no API keys or credentials are baked into your container images or committed to your manifests, loaded instead from your orchestrator's secret-mounting mechanism.

---

## Checkpoint

**Knowledge:**
1. What's the practical difference in orchestrator behavior between a failed readiness check and a failed liveness check?
2. Why should a liveness failure for an agent worker be treated as a potential loop-bounding bug specifically?
3. Why does `maxUnavailable: 0` combined with a real readiness probe prevent a capacity dip during rollout?

**Implementation:** Build `/ready` and `/health` endpoints with correct, distinct failure semantics, from memory.

**Debugging:** A rolling deployment causes a brief spike in failed requests. What specific health-check or manifest misconfiguration would you check first?

**Architecture:** Design a full CI/CD pipeline for your agent system, specifying exactly which earlier Part's test/evaluation mechanism gates each stage, and what happens automatically when a gate fails.

---

**Next:** [Part XLIX — Distributed Agents](part49-distributed-agents.md), where the single worker from this part becomes a fleet — and delivery guarantees (never losing a task, never duplicating one) become the central concern.
