# Part LXV — Expert System-Design Questions

## What this page is for, in plain terms

These 100 questions are the interview-room test of whether Parts I–LII actually stuck, or whether you can recognize code you've seen but couldn't design something new from scratch under pressure. Nobody expects you to answer all 100 in one sitting — the intended use is picking a handful relevant to whatever you're about to build, or working through them gradually as a standing practice habit.

**How to actually answer one, using question 16 as a worked example:** "Design an agent with 100 tools." Don't jump to a diagram. Work through it the way the "Consider" hint suggests: tool discovery (would you offer all 100 schemas on every call, burning context budget per [Part XXXII](part32-context-engineering.md), or filter dynamically?), context budget (100 tool schemas is a lot of tokens before the actual conversation even starts), permission scoping ([Part XXXV](part35-security.md) — no single task needs all 100), and circuit breakers ([Part VI §10](part06-structured-outputs-tool-calling.md) — with 100 tools, *something* is failing at any given moment at real scale). Write two or three paragraphs addressing each consideration explicitly before you'd consider the question "answered" — a one-line response ("use a registry") isn't a design, it's a slogan.

For each question, consider the listed concepts before attempting an answer — do not jump to a solution without addressing each. Answers are intentionally not provided; use the [Project Requirement Format](part55-project-requirement-format.md) to structure your response.

## RAG and Retrieval (1–15)
1. Design a production RAG system for 100 million documents. *Consider: chunking strategy at scale, embedding cost, sharding (Part XV/XVI), incremental ingestion (Part XII), evaluation.*
2. Design a RAG system that must support real-time document updates with sub-minute staleness.
3. Design a multi-tenant RAG platform with strict per-tenant data isolation.
4. Design a RAG system for a corpus mixing prose, tables, and code.
5. Design a Graph RAG system for a domain with dense entity relationships (e.g., org charts).
6. Design a RAG evaluation pipeline that catches regressions before they reach production.
7. Design a hybrid search system balancing keyword precision and semantic recall.
8. Design an embedding-model migration strategy with zero query-time downtime.
9. Design a RAG system resilient to a vector database regional outage.
10. Design a citation system that can be audited for correctness at scale.
11. Design a corrective RAG pipeline with a web-search fallback and cost controls.
12. Design a RAG system for highly regulated data (e.g., healthcare) with access controls per document.
13. Design a chunking strategy for a corpus with wildly varying document lengths (tweets to 500-page manuals).
14. Design a reranking pipeline that stays within a strict latency SLA.
15. Design a system to detect and prevent retrieval-based prompt injection at scale.

## Agents and Tool Use (16–35)
16. Design an agent with 100 tools. *Consider: tool discovery, context budget, permission scoping, circuit breakers.*
17. Design an agent architecture for a task requiring both deterministic compliance steps and adaptive research.
18. Design a ReAct agent that never exceeds a strict per-session cost budget.
19. Design a multi-agent coding system with human review gates at each stage.
20. Design a research agent that must cite verifiable sources for every claim.
21. Design a browser agent that can safely operate on authenticated sites without ever handling credentials directly.
22. Design a tool permission system for an agent serving both internal employees and external customers with different trust levels.
23. Design a code agent that can safely execute untrusted, LLM-generated code.
24. Design an agent harness supporting hot-swappable model providers mid-session.
25. Design a supervisor architecture that gracefully handles a worker agent's total failure.
26. Design a planning system that can replan mid-execution without losing completed work.
27. Design a reflection loop that provably terminates.
28. Design an agent evaluation trajectory format that supports both automated and human review.
29. Design a tool-calling system that prevents duplicate side effects under network retries.
30. Design an agent that must operate correctly with an unreliable, high-latency third-party API as its primary tool.
31. Design a router that dynamically improves its routing accuracy from production feedback.
32. Design a context management strategy for an agent handling both short Q&A and week-long research tasks.
33. Design an agent capable of gracefully degrading from "agentic" to "simple pipeline" mode under cost pressure.
34. Design a multi-agent negotiation system for resource allocation among competing sub-agents.
35. Design a self-improving prompt system with a hard bound on autonomous modification scope.

## MCP and Interoperability (36–50)
36. Design an MCP ecosystem for an enterprise. *Consider: server discovery, trust tiers, versioning, auditing.*
37. Design an MCP server exposing a legacy SOAP API safely.
38. Design a permission model for an MCP server shared across multiple internal teams.
39. Design a strategy for detecting a compromised/malicious MCP server at runtime.
40. Design an MCP client that gracefully handles a server's mid-session capability downgrade.
41. Design a versioning strategy for an MCP server used by both old and new clients simultaneously.
42. Design an agent-to-agent interoperability layer across 3 different frameworks.
43. Design an identity/auth system for cross-organizational agent-to-agent delegation.
44. Design a rate-limiting strategy shared fairly across multiple agents calling the same MCP server.
45. Design a fallback strategy when a critical MCP server becomes unavailable mid-task.
46. Design an audit logging system spanning tool calls across multiple MCP servers.
47. Design a sandboxing strategy for an MCP server exposing filesystem write access.
48. Design a schema evolution strategy for a widely-used MCP tool without breaking existing consumers.
49. Design a discovery mechanism for an agent to find and safely vet new MCP servers at runtime.
50. Design an MCP-based system replacing a legacy point-to-point integration architecture.

## Reliability, Security, Production (51–75)
51. Design fault-tolerant agent execution. *Consider: checkpointing, idempotency, dead-letter queues.*
52. Design an agent security architecture resistant to indirect prompt injection.
53. Design a distributed agent runtime handling 10,000 concurrent sessions.
54. Design a chaos-testing strategy for a multi-agent production system.
55. Design a secrets management strategy for agents calling dozens of external APIs.
56. Design a system that detects and contains a runaway/looping agent within seconds.
57. Design a graceful-degradation strategy when the primary LLM provider is down.
58. Design an incident response runbook for an agent that took an unauthorized action.
59. Design a cost-budget enforcement system spanning per-request, per-session, and per-tenant-per-day limits.
60. Design a zero-downtime deployment strategy for an agent runtime with long-running sessions.
61. Design an access-control system distinguishing read-only, write, and high-risk agent actions.
62. Design a system for safely rolling back a bad prompt/model deployment mid-incident.
63. Design a multi-region agent deployment with data residency requirements.
64. Design a rate-limiting strategy protecting against a single tenant's runaway agent affecting others.
65. Design an audit system that can reconstruct exactly what an agent did and why, for any historical run.
66. Design a human-approval system resistant to approval fatigue at high volume.
67. Design a testing strategy that catches prompt-injection regressions before deploy.
68. Design a performance budget and enforcement system for a latency-sensitive customer-facing agent.
69. Design a disaster-recovery plan for a vector database holding a company's entire knowledge base.
70. Design a system for safely testing a new agent architecture against production traffic without risking real side effects.

## Multi-Agent and Advanced Architecture (71–100)
71. Design agent memory. *Consider: short vs. long-term, consolidation, decay, privacy, multi-tenant isolation.*
72. Design agent evaluation. *Consider: trajectory, groundedness, calibration, online vs. offline.*
73. Design a hierarchical multi-agent system for a 5-domain enterprise support system.
74. Design a blackboard-based multi-agent research system with conflict resolution.
75. Design an autonomous software factory pipeline with mandatory human sign-off before deployment.
76. Design a system where agents can spawn sub-agents, with strict depth/cost bounds.
77. Design a knowledge graph ingestion pipeline that stays consistent as source documents change.
78. Design a system combining a traditional workflow engine with an LLM-driven agent framework.
79. Design an event-driven agent architecture reacting to real-time external triggers at scale.
80. Design a dynamic DAG orchestration system for a task whose structure is only known at runtime.
81. Design an agent platform supporting both LangGraph-based and CrewAI-based agents under one harness.
82. Design a system for benchmarking multiple agent architectures against the same task fairly.
83. Design a capability-negotiation protocol for agents built by different vendors.
84. Design a system that automatically routes tasks to the cheapest sufficiently-capable agent architecture.
85. Design a review/critic architecture for code-generation agents that catches security vulnerabilities.
86. Design a research-agent system with configurable rigor levels (quick answer vs. deeply verified report).
87. Design a debate architecture resistant to confident-wrong consensus.
88. Design an agent platform's onboarding flow for a new internal team building their first agent.
89. Design a system for safely deprecating an old tool/MCP server without breaking dependent agents.
90. Design a load-balancing strategy across multiple LLM providers with different rate limits and pricing.
91. Design a testing strategy for verifying an agent's tool-selection accuracy improves over successive prompt iterations.
92. Design a system for detecting when an agent's underlying model has silently degraded after a provider-side update.
93. Design a "kill switch" mechanism for immediately halting all agent activity for a compromised tenant.
94. Design a system reconciling contradictory outputs from parallel independent agent branches.
95. Design an agent platform's pricing/cost-attribution model for internal chargeback across teams.
96. Design a system for progressively increasing an agent's autonomy as its track record improves.
97. Design a compliance-audit-ready logging architecture for a regulated industry's agent deployment.
98. Design a system for safely A/B testing two different agent architectures on live traffic.
99. Design a full agent operating system integrating RAG, MCP, multi-agent orchestration, and human oversight ([Part LIV](part54-capstone-projects.md) Capstone 4/10 territory).
100. Design the entire production Agent Platform from [Part LXX](part70-final-expert-project.md) at enterprise scale, including every cross-cutting concern from this curriculum.
