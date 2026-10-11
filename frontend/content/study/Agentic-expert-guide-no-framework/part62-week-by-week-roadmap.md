# Part LXII — Week-by-Week Roadmap

## What this page is for, in plain terms

Part LXI told you *what stage* you're at. This page turns that into an actual weekly schedule you could put on a calendar — useful if "read this curriculum eventually" isn't concrete enough to actually stick to, and you want deadlines.

**How to adapt it honestly:** the 24-week pace assumes roughly 8–10 focused hours a week. If you genuinely have less time, don't compress the *content* — stretch the *calendar*. A part like Part XXXV (Security) with its build-exploit-then-fix project genuinely needs the time it needs; rushing it defeats the point, since the whole value is in personally watching an exploit succeed and then fail. If a given week's checkpoint questions (linked from each part) aren't passing cleanly, that week isn't done — carry it into the next week rather than moving on with a gap.

A 24-week roadmap prioritizing implementation over passive learning. Each week: theory, implementation, project, debugging exercise, reading, architecture exercise, evaluation exercise, checkpoint. Adjust pace to your own availability — the point is the *shape* (always implementation-anchored), not the exact week count.

| Week | Theory (Parts) | Project | Debugging exercise | Architecture exercise | Checkpoint gate |
|---|---|---|---|---|---|
| 1 | I (Prerequisites) | Async LLM client skeleton | Diagnose a deliberately blocked event loop | Design a 3-service local dev stack | Part I |
| 2 | II–III (ML + LLM Fundamentals) | Attention from scratch + streaming reconstruction | Diagnose truncated JSON output | Design a prompt-cache-friendly request structure | Parts II–III |
| 3 | IV (LLM App Engineering) | Multi-provider client with fallback | Diagnose a cost spike from a retry storm | Design a 3-provider fallback chain | Part IV |
| 4 | V (Prompt Engineering) | Prompt regression harness + self-consistency experiment | Diagnose a prompt regression after a wording tweak | Design a CI-gated prompt versioning workflow | Part V |
| 5 | VI (Tool Calling) | Multi-tool agent (calculator/weather/filesystem) | Diagnose a hallucinated tool call | Design a tool registry with circuit breakers | Part VI |
| 6 | VII–VIII (Agents + Architectures) | ReAct + Plan-and-Execute + Supervisor comparison | Diagnose an agent stuck at max_steps | Design an architecture choice for a given task | Parts VII–VIII |
| 7 | IX–X (State + Memory) | Checkpoint store + long-term memory system | Diagnose a lost checkpoint after crash | Design multi-tenant memory isolation | Parts IX–X |
| 8 | XI–XIII (RAG + Ingestion + Chunking) | End-to-end RAG pipeline with 3 chunking strategies compared | Diagnose garbled table retrieval | Design an ingestion pipeline for 10k docs/day | Parts XI–XIII |
| 9 | XIV–XVI (Embeddings + Vector DBs + Qdrant) | Qdrant-backed hybrid search RAG | Diagnose a retrieval quality regression | Design sharding for 500M points | Parts XIV–XVI |
| 10 | XVII–XVIII (Advanced RAG + Evaluation) | Corrective RAG + full eval harness | Diagnose a false-groundedness case | Design a CI regression gate for RAG | Parts XVII–XVIII |
| 11 | XIX–XXII (Frameworks) | Same task in LangChain, LangGraph, CrewAI | Diagnose a dropped tool call in a chain | Justify a framework choice in writing | Parts XIX–XXII |
| 12 | XXIII–XXV (MCP) | 2 MCP servers + multi-server agent client | Diagnose an unresponsive MCP server | Design MCP trust-boundary policy | Parts XXIII–XXV |
| 13 | XXVI–XXVII (Orchestration + Multi-Agent) | Research team + SWE team with bounded review/fix cycles | Diagnose a race condition on a blackboard | Design agent-spawning limits | Parts XXVI–XXVII |
| 14 | XXVIII–XXIX (Harness + From-Scratch) | Full harness from scratch (no framework) | Diagnose a budget-exceeded incident post-mortem | Design a multi-tenant harness | Parts XXVIII–XXIX |
| 15 | XXX–XXXI (Planning + Reasoning) | Dependency-graph planner with replanning | Diagnose a stuck plan (no ready steps) | Design a reasoning-strategy selection policy | Parts XXX–XXXI |
| 16 | XXXII–XXXIII (Context + Tool Engineering) | Context assembly pipeline + tool contract registry | Diagnose an answer based on stale context | Design a context token budget allocation | Parts XXXII–XXXIII |
| 17 | XXXIV–XXXV (HITL + Security) | Human-approval agent + vulnerable-then-secured exercise | Diagnose a leaked-data incident | Design a security architecture for 100 tools | Parts XXXIV–XXXV |
| 18 | XXXVI–XXXVII (Reliability + Testing) | Dead-letter queue pipeline + layered test suite | Diagnose a duplicated side effect after resume | Design a fault-tolerant 24-hour pipeline | Parts XXXVI–XXXVII |
| 19 | XXXVIII–XXXIX (Evaluation + Observability) | Agent evaluation platform + tracing dashboard | Diagnose an evaluation/production gap | Design a redaction-safe tracing policy | Parts XXXVIII–XXXIX |
| 20 | XL–XLI (Cost + Performance) | Cost-aware router + parallel-tool-call benchmark | Diagnose a prompt-cache miss | Design a per-tenant cost budget hierarchy | Parts XL–XLI |
| 21 | XLII–XLIII (Databases + Knowledge Graphs) | Full data architecture + a Graph RAG system | Diagnose a lock contention issue | Design shard/store choice matrix | Parts XLII–XLIII |
| 22 | XLIV–XLVI (Code/Browser/Research Agents) | One specialized agent of your choice, fully built | Diagnose a false "tests pass" report | Design sandboxing for a chosen specialization | Parts XLIV–XLVI |
| 23 | XLVII–L (Production + Deployment + Distributed + Advanced Orchestration) | Deploy the full stack via Compose then Kubernetes | Diagnose a rolling-deployment error spike | Design a migration to enterprise-scale architecture | Parts XLVII–L |
| 24 | LI–LII + Capstone kickoff | Interoperability demo + pick first Capstone | Critically evaluate one "autonomous agent" claim | Full Project Requirement Format for chosen Capstone | Parts LI–LII, begin Part LIV |

## Beyond Week 24
Continue with remaining Capstones ([Part LIV](part54-capstone-projects.md)), the Expert Final Examination ([Part LXIX](part69-expert-final-examination.md)), and the Final Expert Project ([Part LXX](part70-final-expert-project.md)) at a pace matched to your own depth goals — these are not week-boxed, since capstone-level integration work varies enormously in scope.
