# Part LVII — Failure-Driven Learning: The Failure Mode Catalog

## What this page is for, in plain terms

This is the page you open during an actual incident — production is misbehaving, you have a vague symptom, and you need to narrow it down fast. It's organized backwards from how you normally read a curriculum: instead of "here's a concept, here's how it works," it's "here's a symptom you're staring at right now, here's where to look."

**How to actually use it, with a worked example:** suppose your monitoring shows agent runs occasionally producing duplicate emails to customers. Scan the tables below for something matching "duplicate" — you'll land on "Duplicate execution" in the Agent/Orchestration table. Its diagnosis column tells you to check the audit log for an idempotency-key on that tool call. If there isn't one, you've found your root cause in under a minute, and the fix column points you straight back to [Part XXXIII §3](part33-tool-engineering.md) rather than making you re-derive the solution from scratch under incident pressure.

Every entry follows: **symptom → root cause → diagnosis → fix → prevention**. Entries cross-reference the Part where the underlying mechanism (and its proper handling) is covered.

## 1. Model/LLM-Layer Failures

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Malformed tool call | Model emits invalid JSON/args | Log raw model output before parsing | Constrained decoding + schema validation ([Part VI](part06-structured-outputs-tool-calling.md)) | Regression tests on tool-call format ([Part XXXVII](part37-agent-testing.md)) |
| Hallucinated tool | Model calls a tool name that doesn't exist | Registry lookup miss logged distinctly from execution errors | Return structured "unknown tool" error, don't crash ([Part VI §8](part06-structured-outputs-tool-calling.md)) | Keep tool list in context accurate/minimal ([Part XXXII](part32-context-engineering.md)) |
| Incorrect arguments | Ambiguous tool description or under-specified schema | Compare model's args against schema field descriptions | Improve schema `description` fields; add few-shot examples ([Part V §6](part05-prompt-engineering.md)) | Tool-argument-accuracy eval metric ([Part XXXVIII §4](part38-agent-evaluation.md)) |
| Unavailable tool | Tool server down or circuit-broken | Check circuit breaker state ([Part VI §10](part06-structured-outputs-tool-calling.md)) | Remove from offered tool list until recovered | Health checks + fallback tools |
| Timeout | Tool or model call exceeds bound | Check per-call latency traces ([Part XXXIX](part39-observability.md)) | `asyncio.wait_for` with sane timeout ([Part VI §7](part06-structured-outputs-tool-calling.md)) | Timeout budgets set per tool risk/latency profile |
| Rate limit | Too many concurrent/sequential provider calls | Check 429 response codes in logs | Exponential backoff + fallback provider ([Part IV §6](part04-llm-application-engineering.md)) | Concurrency semaphore ([Part I §6](part01-prerequisites.md)) |
| Model failure (5xx) | Provider outage | Check provider status + error rate metric ([Part XXXIX §4](part39-observability.md)) | Fallback provider ([Part IV §3](part04-llm-application-engineering.md)) | Multi-provider architecture from day one |

## 2. Retrieval/RAG-Layer Failures

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Vector database failure | Node down, connection exhausted | Check connection pool metrics | Retry + replica failover ([Part XV §8](part15-vector-databases.md)) | Replication configured ([Part XVI §5](part16-qdrant-deep-dive.md)) |
| Stale embeddings | Corpus re-embedded with a different model without full migration | Compare stored model-version metadata across vectors | Full re-index migration ([Part XIV §6](part14-embeddings.md)) | Model-version tagging enforced at write time |
| Bad chunking | Chunk boundary splits a fact from its context | Recall@K eval on affected queries drops | Switch strategy (recursive/parent-child, [Part XIII](part13-chunking.md)) | Chunking-strategy eval harness ([Part XIII §6](part13-chunking.md)) |
| Irrelevant retrieval | Embedding/domain mismatch, or missing metadata filter | Inspect top-K similarity scores for the query | Domain-tuned embeddings or add filtering ([Part XIV §5](part14-embeddings.md)) | Retrieval eval gate in CI ([Part XVIII §4](part18-rag-evaluation.md)) |
| Hallucinated answer despite RAG | Generation not actually grounded in retrieved context | Run faithfulness/groundedness check ([Part XVIII §3](part18-rag-evaluation.md)) | Add explicit groundedness verification step | Faithfulness metric tracked continuously |

## 3. Agent/Orchestration-Layer Failures

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Infinite agent loop | No termination bound, or termination logic buggy | Check step count vs. `max_steps` in trace | Enforce hard bound ([Part VII §5](part07-what-is-an-agent.md)) | Cost/step ceilings in harness ([Part XXVIII](part28-agent-harness.md)) |
| State corruption | Non-atomic writes, concurrent writers | Check for torn/partial JSON in checkpoint store | Atomic writes + optimistic concurrency ([Part IX §4](part09-agent-state.md)) | Version field enforced on every write |
| Memory pollution | Irrelevant/conflicting memories injected into context | Inspect retrieved memory relevance scores | Conflict-aware writes, decay ([Part X §5–6](part10-memory.md)) | Memory consolidation job |
| Prompt injection succeeded | Untrusted content not delimited, or over-broad tool permissions | Check whether the executed tool was outside task-scoped permissions | Least-privilege scoping ([Part XXXV §3](part35-security.md)) | Adversarial tests in CI ([Part XXXVII](part37-agent-testing.md)) |
| Tool poisoning | Malicious/compromised MCP server description manipulated agent behavior | Review server trust tier ([Part XXV §4](part25-agent-plus-mcp.md)) | Vet/allowlist MCP servers | Trust-boundary policy enforced at connection time |
| MCP server failure | Server crash or protocol mismatch | Check session health / capability negotiation logs | Reconnect with backoff, structured error to model | Health checks per server ([Part XXIII §6](part23-mcp.md)) |
| Partial multi-agent failure | One agent in a team fails, others proceed on incomplete input | Check task `context` dependency completion before downstream execution | Fail the dependent step explicitly rather than silently continuing ([Part XXVI §6](part26-agent-orchestration.md)) | DAG dependency validation |
| Race condition | Unlocked shared state (blackboard) | Reproduce under concurrency load test | Locking around shared writes ([Part XXVII §5](part27-multi-agent-systems.md)) | Concurrency-specific test coverage |
| Duplicate execution | Non-idempotent tool retried after ambiguous-outcome timeout | Check idempotency-key presence in audit log | Idempotency keys enforced at tool-contract level ([Part XXXIII §3](part33-tool-engineering.md)) | Reject non-idempotent tools without a key at registration |
| Lost checkpoint | Crash during a non-atomic write, or missing persistence | Check for a checkpoint file gap around the crash timestamp | Atomic write pattern ([Part IX §4](part09-agent-state.md)) | Checkpoint after every step, not batched |

## 4. Using This Catalog
When a production incident occurs, locate the closest matching symptom row, follow its diagnosis step to confirm root cause (don't assume — verify), apply the fix, and then implement the prevention measure as a permanent code/process change, not a one-off patch. Add new rows to this catalog as your own systems reveal failure modes not yet listed here — the catalog is a living document, not a finished checklist.

## Checkpoint
- **Knowledge**: For any 5 rows above, explain in your own words why the "prevention" column addresses the root cause and not just the symptom.
- **Implementation**: Reproduce 3 of these failure modes deliberately in a test system and verify your fix resolves them.
- **Debugging**: Given a production incident report with only a vague symptom description, practice narrowing it to a specific row in this catalog using only the diagnosis steps.
- **Architecture**: Design an incident-response runbook that references this catalog for the top 10 most likely failure modes in your specific system.
