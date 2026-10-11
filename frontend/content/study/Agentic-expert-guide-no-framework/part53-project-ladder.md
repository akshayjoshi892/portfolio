# Part LIII — Project-Based Learning: The 50-Project Ladder

## What this page is for, in plain terms

Parts I–LII taught you each mechanism once, with one project per part. This page is different: it's a checklist of 50 smaller, focused projects, grouped into 7 levels, so you can *practice* — build the same kinds of things again, in new combinations, until they're fast and automatic rather than something you have to think hard about each time.

**How to actually use it:** pick the next unchecked project in order (don't skip levels — level 4 assumes level 3's muscle memory). Set a rough time box (most of these should take 1–4 hours once you're fluent with the underlying part). Build it without opening the reference part's code — if you get stuck, that's useful information: go re-read that part's relevant stage, then come back and finish the project yourself rather than copying.

**Example of how one project maps to a part:** Project 15, "Reranking," doesn't mean "implement a brand-new algorithm." It means: take your Part XI RAG pipeline, add the cross-encoder reranking step from [Part XVII §3](part17-advanced-rag.md), and confirm on your own test questions that reordering the top candidates before generation actually improves answer quality. The project is the *practice*, not new material.

Every project below builds on mechanisms already implemented earlier in this curriculum — cross-references point to the relevant part. Build each one **without copying code from that part**; the reference is for the underlying mechanism, not a template to paste.

## Level 1 — Fundamentals
1. LLM CLI — [Part IV](part04-llm-application-engineering.md)
2. Prompt template system — [Part V §4](part05-prompt-engineering.md)
3. Structured output parser — [Part VI](part06-structured-outputs-tool-calling.md)
4. Calculator tool — [Part VI §5](part06-structured-outputs-tool-calling.md)
5. Weather tool (mocked HTTP) — [Part VI §9](part06-structured-outputs-tool-calling.md)
6. Multi-tool application — [Part VI §9](part06-structured-outputs-tool-calling.md)
7. Streaming application — [Part III §13](part03-llm-fundamentals.md)
8. Conversation memory — [Part X §4](part10-memory.md)

## Level 2 — RAG
9. Basic document RAG — [Part XI](part11-rag-fundamentals.md)
10. PDF RAG — [Part XII §8](part12-document-ingestion.md)
11. Markdown RAG — [Part XII](part12-document-ingestion.md)
12. Qdrant RAG — [Part XVI §7](part16-qdrant-deep-dive.md)
13. Metadata filtering — [Part XV §7](part15-vector-databases.md)
14. Hybrid retrieval — [Part XVI §4](part16-qdrant-deep-dive.md)
15. Reranking — [Part XVII §3](part17-advanced-rag.md)
16. Query rewriting — [Part XVII §2](part17-advanced-rag.md)
17. Multi-query RAG — [Part XVII §2](part17-advanced-rag.md)
18. Evaluation framework — [Part XVIII §9](part18-rag-evaluation.md)

## Level 3 — Agents
19. ReAct agent — [Part VIII §2](part08-agent-architectures.md)
20. Tool-calling agent — [Part VI](part06-structured-outputs-tool-calling.md)
21. Planner — [Part XXX §7](part30-planning.md)
22. Executor — [Part VII §5](part07-what-is-an-agent.md)
23. Reflection agent — [Part VIII §1](part08-agent-architectures.md) architecture #7
24. Router — [Part VIII §1](part08-agent-architectures.md) architecture #4
25. Supervisor — [Part VIII §4](part08-agent-architectures.md)
26. Human-in-loop agent — [Part XXXIV §6](part34-human-in-the-loop.md)

## Level 4 — LangGraph
27. State graph — [Part XX §3](part20-langgraph.md)
28. Persistent graph — [Part XX §3](part20-langgraph.md)
29. Checkpointed agent — [Part XX §6](part20-langgraph.md)
30. Multi-agent graph — [Part XX §8](part20-langgraph.md)
31. Parallel graph — [Part XX §5](part20-langgraph.md)
32. Fault-tolerant graph — [Part XX §8](part20-langgraph.md) (crash/resume test)
33. Long-running agent — [Part L §2](part50-advanced-orchestration.md)

## Level 5 — MCP
34. MCP calculator — [Part XXIII §4](part23-mcp.md)
35. MCP filesystem — [Part XXIV §2](part24-mcp-server-development.md)
36. MCP database — [Part XXIV §3](part24-mcp-server-development.md)
37. MCP Qdrant — [Part XVI](part16-qdrant-deep-dive.md) + [Part XXIII](part23-mcp.md)
38. Multi-MCP agent — [Part XXV §3](part25-agent-plus-mcp.md)
39. Secure MCP environment — [Part XXIV §4](part24-mcp-server-development.md) + [Part XXXV](part35-security.md)

## Level 6 — Multi-Agent
40. Research team — [Part XXVII §3](part27-multi-agent-systems.md)
41. Coding team — [Part XXVII §4](part27-multi-agent-systems.md)
42. Planning team — [Part XXX](part30-planning.md) + [Part XXVII](part27-multi-agent-systems.md)
43. Review team — [Part VIII](part08-agent-architectures.md) architecture #7/#8 at team scale
44. Autonomous development system — [Part XXVII §7](part27-multi-agent-systems.md)

## Level 7 — Production
45. Agent observability platform — [Part XXXIX §7](part39-observability.md)
46. Agent evaluation platform — [Part XXXVIII §9](part38-agent-evaluation.md)
47. Cost-aware router — [Part XL §7](part40-cost-optimization.md)
48. Distributed agent runtime — [Part XLIX §7](part49-distributed-agents.md)
49. Production RAG platform — [Part XI](part11-rag-fundamentals.md)–[XVIII] combined, deployed per [Part XLVIII](part48-deployment.md)
50. Full autonomous agent harness — [Part XXVIII §6](part28-agent-harness.md)

## How to Use This Ladder
- Do not skip levels — each level's projects assume the mechanisms from prior levels are already solid.
- For each project, apply the Project Requirement Format from [Part LV](part55-project-requirement-format.md) before starting: write the objective, architecture, and failure scenarios *before* writing code.
- After each project, run its Part's checkpoint questions against your implementation, not just the reference material.

## Beyond 50
Additional projects worth adding as you progress: a hybrid workflow-agent system combining a fixed compliance pipeline with an agentic sub-step ([Part VIII](part08-agent-architectures.md) architecture #16); a Graph RAG system ([Part XLIII §8](part43-knowledge-graphs.md)); a browser research agent ([Part XLV §8](part45-browser-agents.md)); an agent-to-agent interoperability demo across two frameworks ([Part LI §6](part51-agent-protocols-interoperability.md)).
