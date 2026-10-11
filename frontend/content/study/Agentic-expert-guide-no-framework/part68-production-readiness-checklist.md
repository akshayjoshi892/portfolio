# Part LXVIII — Production Readiness Checklist

## What this page is for, in plain terms

This is the pre-flight checklist you run through before actually shipping an agent system to real users — the moment where "it works on my machine, on my test cases" needs to become "I've deliberately checked every category of thing that goes wrong in production and either handled it or consciously accepted the risk." Every single checkbox links to the exact stage of the exact part where you built the real mechanism, so an unchecked item is never a mystery — it's a known, specific gap with a known, specific fix already written down elsewhere in this curriculum.

**How to use this honestly, with a worked example:** you're about to launch, and you get to the "Reliability" section's item "non-idempotent side-effecting tools require an idempotency key." If your `send_email` tool doesn't have one, don't just check the box anyway because you're in a hurry — that's exactly the [Part XXXIII](part33-tool-engineering.md) gap that causes a real, specific, predictable incident (a customer getting the same email twice after a retry). The honest move, if you truly can't fix it before launch, is to leave it unchecked and write down, explicitly, that you're accepting that specific risk for now — not to silently check a box that isn't actually true.

Before calling an agent system production-ready, verify every item below. Each maps to the Part where it was built.

## Correctness
- [ ] Task success rate measured against a representative golden set ([Part XXXVIII](part38-agent-evaluation.md)), not just spot-checked manually.
- [ ] Tool-argument accuracy verified, not just tool-selection accuracy.
- [ ] Structured outputs schema-validated at the application boundary, even when using constrained decoding ([Part VI §4](part06-structured-outputs-tool-calling.md)).

## Reliability
- [ ] Every external call (model, tool, retrieval) has a timeout ([Part VI §7](part06-structured-outputs-tool-calling.md)).
- [ ] Retries distinguish retriable vs. non-retriable errors, with backoff+jitter ([Part IV §6](part04-llm-application-engineering.md)).
- [ ] Non-idempotent side-effecting tools require an idempotency key and are never blindly retried ([Part XXXIII §3](part33-tool-engineering.md)).
- [ ] A dead-letter queue captures tasks that exhaust retries ([Part XXXVI §3](part36-reliability.md)).
- [ ] Fallback model/provider configured for outage scenarios ([Part IV §3](part04-llm-application-engineering.md)).
- [ ] Circuit breakers protect against repeatedly-failing dependencies ([Part VI §10](part06-structured-outputs-tool-calling.md)).

## Security
- [ ] All retrieved/tool-result content is delimited as untrusted data before re-entering model context ([Part XXXV §3](part35-security.md)).
- [ ] Tool permissions are scoped per task/session, not globally granted ([Part XXXV §4](part35-security.md)).
- [ ] High-risk, side-effecting tools require human approval by default ([Part XXXIV](part34-human-in-the-loop.md)).
- [ ] Path/table/host allowlists enforced on any filesystem/database/network-capable tool ([Part XXIV §2–3](part24-mcp-server-development.md)).
- [ ] Adversarial (prompt-injection) tests exist in the test suite and pass ([Part XXXVII §3](part37-agent-testing.md)).
- [ ] Secrets are never inline in code, config files, or logs ([Part XLVIII §4](part48-deployment.md)).

## Privacy / Data Governance
- [ ] Memory/state stores are scoped per user/tenant with enforced isolation ([Part X §6](part10-memory.md)).
- [ ] A data-deletion path exists and actually removes data from all stores, including vector indexes ([Part X §6](part10-memory.md)).
- [ ] PII/secrets are redacted from trace/log storage ([Part XXXIX §6](part39-observability.md)).

## Observability
- [ ] Every run has a correlation/trace ID propagated across all async boundaries ([Part XXXIX §3](part39-observability.md)).
- [ ] Latency is reported as percentiles (p50/p95/p99), not just averages ([Part XLI §1](part41-performance-engineering.md)).
- [ ] Error rates are tracked per tool/model/component, not just globally.
- [ ] Alerting exists for latency, error rate, and cost anomalies.

## Cost
- [ ] Per-run cost is tracked and enforced against a budget in code, not just monitored after the fact ([Part XL §5](part40-cost-optimization.md)).
- [ ] Model routing sends easy subtasks to cheaper models ([Part XL §3](part40-cost-optimization.md)).
- [ ] Prompt caching is used for stable prefixes where supported ([Part III §16](part03-llm-fundamentals.md)).

## Latency / Performance
- [ ] Independent tool calls are parallelized ([Part XLI §2](part41-performance-engineering.md)).
- [ ] Connection pooling is used for all external HTTP calls ([Part XLI §4](part41-performance-engineering.md)).
- [ ] Vector search is tuned (ef/payload indexes) and benchmarked at production scale ([Part XV §5, §7](part15-vector-databases.md)).

## Scalability
- [ ] Task processing is decoupled from the request path via a queue for anything beyond trivial latency ([Part XLVII §3](part47-production-architecture.md)).
- [ ] Workers scale horizontally with health checks gating traffic ([Part XLVIII §3](part48-deployment.md)).
- [ ] Database connection pools and vector DB sharding are sized for expected peak load ([Part XLII](part42-databases-for-agents.md)/[Part XV §8](part15-vector-databases.md)).

## Testing
- [ ] Unit, tool, integration, property-based e2e, adversarial, and chaos tests all exist ([Part XXXVII](part37-agent-testing.md)).
- [ ] A chaos test verifies crash-and-resume never duplicates a non-idempotent side effect ([Part XXXVII §3](part37-agent-testing.md)).
- [ ] A load test reports p50/p95/p99 under realistic concurrency ([Part XXXVII §5](part37-agent-testing.md)).

## Evaluation
- [ ] Offline evaluation gate blocks deploys on regression ([Part XVIII §4](part18-rag-evaluation.md)/[Part XXXVIII §9](part38-agent-evaluation.md)).
- [ ] Online evaluation samples production traffic for ongoing quality monitoring ([Part XXXVIII §6](part38-agent-evaluation.md)).
- [ ] Any LLM-as-judge is calibrated against human-labeled examples ([Part XXXVIII §5](part38-agent-evaluation.md)).

## Human Oversight
- [ ] Risk-based approval policy exists and is enforced in code, not just documented ([Part XXXIV §4](part34-human-in-the-loop.md)).
- [ ] Pending approvals are durably persisted, surviving a process restart ([Part XXXIV §3](part34-human-in-the-loop.md)).
- [ ] Escalation/timeout exists for approvals that sit unresolved too long.

## Failure Recovery
- [ ] All state is checkpointed atomically after every meaningful step ([Part IX §4](part09-agent-state.md)).
- [ ] Resumed runs are verified not to duplicate non-idempotent actions ([Part XXXVI](part36-reliability.md)).
- [ ] Graceful degradation is visible to the caller/user, never silent ([Part XXXVI §4](part36-reliability.md)).

## Tool Permissions
- [ ] Every tool has a declared `ToolContract` (idempotency, risk, side-effect classification) before registration ([Part XXXIII §3](part33-tool-engineering.md)).
- [ ] MCP server trust tiers are explicit, and untrusted-tier results/descriptions are treated with elevated scrutiny ([Part XXV §4](part25-agent-plus-mcp.md)).

## Prompt Security
- [ ] Prompt templates never interpolate untrusted text into instruction-bearing regions without delimiting ([Part V §8](part05-prompt-engineering.md)).
- [ ] Prompts are versioned in source control with regression tests gating changes ([Part V §10](part05-prompt-engineering.md)).

## Dependency Management
- [ ] Framework/SDK versions are pinned, with a documented upgrade process (Part LXIII's version-sensitivity discipline).
- [ ] A model-provider or framework version bump is tested against the evaluation suite before rollout.

## Model Fallback
- [ ] At least one fallback model/provider is configured and tested under simulated primary-provider failure ([Part IV §3](part04-llm-application-engineering.md)).

## Monitoring / Incident Response
- [ ] Dashboards exist for the metrics in the Observability section above.
- [ ] An incident-response runbook references the [Failure Mode Catalog](part57-failure-driven-learning.md) for common symptom → diagnosis paths.
- [ ] A "kill switch" exists to halt agent activity for a specific tenant/session/tool immediately if needed.

## Using This Checklist
Run through every item honestly before a production launch — an unchecked item is not automatically a blocker, but it must be a **conscious, documented decision** to accept that risk, not an oversight. Revisit this checklist after any significant incident, adding new items as your own system reveals gaps this generic list doesn't cover.
