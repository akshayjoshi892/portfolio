# Part LIV — Capstone Projects

## What this page is for, in plain terms

A capstone is bigger than anything in the 50-project ladder — it's a real system combining 5–10 parts of this curriculum at once, the kind of thing that would take a week or more, not an afternoon. Ten are listed below, each with the specific parts it draws on named explicitly, so you know exactly what to review before starting.

**You don't need to build all ten.** Most learners pick 2–4 that match what they actually want to get good at (e.g., if you care about RAG products, Capstones 1 and 9; if you care about coding agents, Capstone 3). Building even one capstone properly — with real evaluation, real security testing, real deployment — teaches more than reading all ten descriptions.

**A concrete example of what "capstone-sized" means:** Capstone 1 (Production RAG Platform) isn't "add Qdrant to your RAG pipeline" (that's Part XVI, a few hours). It's: real multi-format ingestion (Part XII) feeding a hybrid-search Qdrant index (Part XVI) with reranking (Part XVII), gated by a continuous evaluation suite (Part XVIII) that blocks bad deploys, serving multiple tenants with enforced permission boundaries (Part XLVII) and full request tracing (Part XXXIX). That's a multi-day build, not a single afternoon — plan accordingly.

Each capstone integrates most of the curriculum's parts into one system. Use the [Project Requirement Format](part55-project-requirement-format.md) for each before starting implementation.

## Capstone 1 — Production RAG Platform
**Integrates:** Parts XI–XVIII, XLII, XLVII–XLVIII.
Support multiple document types (Part XII), Qdrant-backed hybrid retrieval (Part XVI), reranking (Part XVII), a continuous evaluation gate (Part XVIII), citations (Part XI), per-tenant permissions (Part XLII/XLVII), and full observability (Part XXXIX).

## Capstone 2 — Autonomous Research Agent
**Integrates:** Parts XXX, XXXI, XLV, XLVI, XXXIV, XXXV.
Planning-driven multi-hop research (Part XLVI), web search via a browser agent (Part XLV), evidence validation and fact-checking (Part XXXI), citation generation, reflection-based quality gating (Part VIII), and human approval before any external action beyond reading (Part XXXIV).

## Capstone 3 — Software Engineering Agent
**Integrates:** Parts XLIV, XXXIII, XXXVI, XXXVII, XXXV.
Repository understanding (AST-based symbol extraction), planning a change, sandboxed code modification, test execution and debug loop with a bounded attempt count, Git integration producing a reviewable patch branch, and a human-approval merge gate.

## Capstone 4 — Multi-Agent Enterprise Assistant
**Integrates:** Parts XXVII, X, XI, XXV, XXXIV, XLVII.
Specialized agents per domain, a Supervisor for delegation (Part VIII), shared long-term memory scoped per user/tenant (Part X), RAG over enterprise documents, MCP-based tool access to internal systems (Part XXV), and human approval for any write-capable action.

## Capstone 5 — Agent Harness (Full Build, No Framework)
**Integrates:** Part XXVIII entirely, plus IX, X, XXXIII, XXXV, XXXIX, XL.
Build the complete harness from scratch: model interface with multi-provider fallback, context manager, state manager with checkpointing, memory, tool registry/executor with contracts and circuit breakers, policy/permission engine, retry manager, evaluator hooks, tracing, cost controller, and sandboxing — this is the project every other capstone could be rebuilt on top of.

## Capstone 6 — Agent Evaluation Platform
**Integrates:** Part XXXVIII entirely, plus XVIII, XXXVII.
Measure task success, latency, cost, tool-selection/argument accuracy, retrieval quality, and trajectory quality across a growing golden dataset, with a calibrated LLM-as-judge and a CI-integrated regression gate.

## Capstone 7 — MCP Ecosystem
**Integrates:** Parts XXIII–XXV.
Build at least 4 MCP servers (filesystem, database, a third-party API wrapper, and a Qdrant-backed retrieval server) plus one intelligent MCP client/agent that dynamically discovers and routes across all of them with trust-boundary-aware permission scoping.

## Capstone 8 — Distributed Agent Runtime
**Integrates:** Part XLIX entirely, plus IX, XXXVI, XLII, XLVIII.
Workers, a durable task queue with at-least-once delivery and idempotent processing, distributed locks, checkpointed/resumable agent runs, deployed via the Kubernetes manifests pattern from Part XLVIII, load-tested (Part XLI) under simulated worker crashes (Part XXXVII's chaos testing).

## Capstone 9 — Agentic RAG Platform
**Integrates:** Parts XVII, XXX, XXXI, XVIII.
Combine planning-driven query decomposition, query rewriting, corrective retrieval with a web-search fallback, reranking, reflection-based answer quality gating, tool-assisted claim verification, and full citation generation — the most retrieval-sophisticated system in the curriculum.

## Capstone 10 — Autonomous Software Factory
**Integrates:** Nearly everything — Parts VIII, XXVII, XXX, XXXIII, XXXIV, XXXV, XXXVI, XLIV.

```text
Requirement
    v
Product Analyst  (Part XXX-style planning: decompose requirement into specs)
    v
Architect        (design decisions, documented, human-reviewable)
    v
Planner           (Part XXX dependency-graph plan)
    v
Coder             (Part XLIV, sandboxed)
    v
Tester            (Part XLIV/XXXVII)
    v
Reviewer          (Part VIII critic architecture)
    v
Debugger          (Part XXXI self-correction with verification)
    v
Security Agent    (Part XXXV — scans the diff for injected vulnerabilities before merge)
    v
Human Approval    (Part XXXIV — mandatory, not optional, gate)
    v
Deployment        (Part XLVIII, only after human sign-off)
```

Every stage must be independently traceable (Part XXXIX), cost-bounded (Part XL), and recoverable from a crash at any stage (Part IX/XXXVI) — this capstone is intended as the curriculum's terminal integration exercise before the [Final Expert Project](part70-final-expert-project.md).

## How to Approach a Capstone
1. Write the full [Project Requirement Format](part55-project-requirement-format.md) document first — architecture, failure scenarios, security considerations — before any code.
2. Build the Beginner version first (mocked externals, single-tenant, no distribution) to validate the architecture.
3. Layer in Intermediate (real externals, basic reliability) then Advanced (production reliability/security/observability) versions incrementally — never attempt the Advanced version directly.
4. Run the Expert Checkpoints from every integrated Part against your final system before considering it complete.
