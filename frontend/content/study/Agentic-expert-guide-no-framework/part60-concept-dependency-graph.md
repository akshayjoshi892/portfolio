# Part LX — Concept Dependency Graph

## What this page is for, in plain terms

If you ever find yourself confused reading a later part — the code assumes something you don't quite remember, or a term gets used without re-explanation — this page tells you exactly which earlier part to go back to. It's the map behind the linear reading order: the curriculum reads top-to-bottom, but the actual *dependency* structure (what genuinely requires what) is a graph, not a straight line, and this page draws that graph explicitly.

**A concrete example of how to use it:** suppose you're reading Part XX (LangGraph) and the reducer/checkpointer code feels unfamiliar rather than like a recognizable framework wrapper around something you already built. Check §2's table below — it tells you LangGraph explicitly assumes Part IX's checkpointing is "second nature." That's your signal: stop, go re-read Part IX until its atomic-write pattern is automatic, then come back to Part XX, where it should now click as "oh, this is just my Part IX pattern with a different API."

## 1. High-Level Dependency Chain

```text
Python + Software Engineering (Part I)
   v
ML Foundations (Part II)
   v
LLM Fundamentals (Part III)
   v
LLM Application Engineering (Part IV)
   v
Prompt Engineering (Part V)
   v
Structured Outputs + Tool Calling (Part VI)
   v
What Is an Agent? (Part VII)
   v
Agent Architectures (Part VIII)  <----------------+
   v                                                |
Agent State (Part IX)                               |
   v                                                 |
Memory (Part X)                                      |
   v                                                  |
RAG Fundamentals (Part XI) -> Ingestion (XII) -> Chunking (XIII) -> Embeddings (XIV) -> Vector DBs (XV) -> Qdrant (XVI)
   v                                                                                                      |
Advanced RAG (XVII) -> RAG Evaluation (XVIII)                                                              |
   v                                                                                                        |
LangChain (XIX) -> LangGraph (XX) -> CrewAI (XXI) -> Framework Comparison (XXII)                             |
   v                                                                                                          |
MCP (XXIII) -> MCP Servers (XXIV) -> Agent + MCP (XXV)                                                        |
   v                                                                                                           |
Orchestration (XXVI) -> Multi-Agent Systems (XXVII) --------------------------------------------------------->+
   v
Agent Harness (XXVIII) -> Build From Scratch (XXIX)
   v
Planning (XXX) -> Reasoning (XXXI) -> Context Engineering (XXXII) -> Tool Engineering (XXXIII)
   v
Human-in-the-Loop (XXXIV) -> Security (XXXV) -> Reliability (XXXVI)
   v
Testing (XXXVII) -> Evaluation (XXXVIII) -> Observability (XXXIX) -> Cost (XL) -> Performance (XLI)
   v
Databases (XLII) -> Knowledge Graphs (XLIII)
   v
Code Agents (XLIV) / Browser Agents (XLV) / Research Agents (XLVI)   [parallel specializations]
   v
Production Architecture (XLVII) -> Deployment (XLVIII) -> Distributed Agents (XLIX) -> Advanced Orchestration (L)
   v
Agent Protocols (LI) -> Advanced Research (LII)
   v
Project Ladder (LIII) -> Capstones (LIV)
```

## 2. Why This Order (Key Dependency Justifications)

| Dependency | Why it must come first |
|---|---|
| Python async (I) before any agent loop (VII) | Every agent loop example uses `asyncio`; without it, the code is unreadable |
| LLM fundamentals (III) before prompt engineering (V) | Understanding sampling/temperature is required to know *why* CoT/self-consistency work or fail |
| Tool calling (VI) before "what is an agent" (VII) | The agent loop's Action step *is* a tool call — VII's examples assume VI's mechanics |
| Agent state (IX) before memory (X) | Memory is explicitly defined as a persistent subset of state |
| Chunking/embeddings/vector DBs (XIII–XV) before Qdrant (XVI) | Qdrant is a concrete implementation of concepts that must be understood abstractly first (Rule 9) |
| Agent architectures (VIII) before frameworks (XIX–XXI) | Frameworks implement these architectures; without the concept, framework APIs look arbitrary |
| MCP (XXIII) before Agent+MCP (XXV) | XXV assumes fluency with MCP's client/server/tool primitives |
| Agent harness (XXVIII) before Planning/Reasoning/Context/Tool Engineering (XXX–XXXIII) | These are all harness *components* — understanding the whole before its parts in depth avoids fragmented mental models |
| Security (XXXV) before Production Architecture (XLVII) | A production architecture designed without security-first thinking requires costly retrofitting |
| Testing/Evaluation/Observability (XXXVII–XXXIX) before Deployment (XLVIII) | You cannot safely deploy what you cannot test or observe |
| Everything before Capstones (LIV) | Capstones integrate nearly the entire dependency graph — see each capstone's "Integrates" list in [Part LIV](part54-capstone-projects.md) |

## 3. Cross-Cutting Concepts (Not Strictly Linear)
Some concepts recur and deepen across many Parts rather than being learned once: **bounded loops/termination** (first in VII, reinforced in every architecture through L), **idempotency** (first in XXXIII, reused in XXXVI/XLIX), **atomic checkpointing** (first in IX, reused in XX/XXXIV/L), and **untrusted-content delimiting** (first in V §9, reused in XXV/XXXII/XXXV/XLV). Treat these as threads to track across the whole curriculum, not isolated topics.

## 4. Practical Use
Before starting any Part, verify you can answer the Knowledge checkpoint of every Part upstream of it in this graph (per [Part LVI](part56-expert-checkpoints.md)'s gating rule). Before starting a Capstone, verify against its specific "Integrates" list rather than the whole curriculum, to keep review scoped and practical.
