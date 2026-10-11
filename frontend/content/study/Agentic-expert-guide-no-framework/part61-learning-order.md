# Part LXI — Learning Order

## What this page is for, in plain terms

This is the "which stage am I actually at" self-assessment. The curriculum's 70 parts are numbered in the order you should read them, but that's a different question from "what should I be able to *do* right now, and what's next given my actual skill level" — especially if you're coming back after a break, or joining partway through with some prior experience. This page groups the whole curriculum into six honest stages (Foundation through Research) with concrete exit criteria for each, so you can self-place rather than guessing.

**How to use it if you're joining mid-curriculum with prior experience:** don't assume you can skip straight to "Advanced" because you've used LangChain before. Read the exit criteria for "Beginner" and "Intermediate" first — if you can't confidently build Part IX's atomic checkpoint pattern from memory (an Intermediate exit criterion), you have a real gap to fill before the Advanced-stage material about multi-agent security will make full sense, no matter how much framework experience you already have.

## Foundation
**Concepts**: Python async, software engineering basics, ML/probability foundations, Transformer/attention mechanics.
**Technologies**: Python 3.11+, Git, Docker, Postgres, Redis.
**Projects**: Async LLM client skeleton (Part I), cosine similarity/k-NN from scratch (Part II).
**Expected skills**: Comfortable reading and writing async Python; can explain attention mechanics without notes.
**Exit criteria**: Pass Knowledge + Implementation checkpoints for Parts I–III unaided.

## Beginner
**Concepts**: LLM API mechanics, prompt engineering, structured output, tool calling, what an agent is.
**Technologies**: A provider SDK/HTTP API, Pydantic, JSON Schema.
**Projects**: Provider-agnostic LLM client (Part IV), prompt regression harness (Part V), tool-calling loop (Part VI), minimal ReAct agent (Part VII/VIII).
**Expected skills**: Can build a reliable tool-using single agent from scratch, with retries/validation/bounded termination.
**Exit criteria**: Complete Level 1–3 of the [Project Ladder](part53-project-ladder.md) (projects 1–26).

## Intermediate
**Concepts**: Agent state/memory, RAG (fundamentals through evaluation), vector databases, MCP.
**Technologies**: Qdrant, an embedding model, an MCP SDK.
**Projects**: Full RAG pipeline with evaluation (Parts XI–XVIII), MCP server + client (Parts XXIII–XXV), agent harness draft (Part XXVIII).
**Expected skills**: Can design and evaluate a grounded RAG system; can build and secure an MCP tool server.
**Exit criteria**: Complete Level 4–5 of the Project Ladder (projects 27–39).

## Advanced
**Concepts**: Orchestration patterns, multi-agent systems, planning/reasoning, context/tool engineering, human-in-the-loop, security, reliability.
**Technologies**: LangGraph (or an equivalent graph runtime), a task queue (Redis Streams), an observability stack.
**Projects**: Multi-agent research/coding teams (Part XXVII), full agent harness from scratch (Part XXVIII), a secured agent (Part XXXV's vulnerable-then-secured exercise).
**Expected skills**: Can design multi-agent coordination, defend against prompt injection, and reason rigorously about reliability trade-offs.
**Exit criteria**: Complete Level 6 of the Project Ladder (projects 40–44); pass the Security part's practical exercise.

## Expert
**Concepts**: Testing/evaluation/observability/cost/performance engineering, databases-for-agents, knowledge graphs, specialized agents (code/browser/research), production architecture, deployment, distributed systems.
**Technologies**: Kubernetes, a durable task queue at scale, a knowledge graph store, CI/CD pipelines with regression gates.
**Projects**: Level 7 of the Project Ladder (projects 45–50); at least 3 Capstones from [Part LIV](part54-capstone-projects.md).
**Expected skills**: Can design, deploy, and operate a production multi-tenant agent platform with full observability, security, and cost controls.
**Exit criteria**: Complete at least 3 capstones and pass their integrated checkpoints.

## Research
**Concepts**: Agent protocols/interoperability, advanced/experimental research topics, current literature on agentic RL, world models, benchmarking.
**Technologies**: Whatever is current — this stage requires reading primary sources directly rather than relying on a fixed curriculum, since this frontier moves faster than any static document (Rule 10).
**Projects**: The [Final Expert Project](part70-final-expert-project.md); original exploration of one advanced-research topic with a written critique of its production-readiness.
**Expected skills**: Can distinguish mature engineering practice from active research, and can evaluate new techniques/papers critically rather than adopting them uncritically.
**Exit criteria**: Independent judgment — no further external exit criteria apply at this stage.

## Non-Negotiable Rule
The learner must build systems at every stage — passive reading/tutorial-watching does not advance you through these stages regardless of time spent. Use the [Project Requirement Format](part55-project-requirement-format.md) for every project and the [Expert Checkpoints](part56-expert-checkpoints.md) method to gate real progression.
