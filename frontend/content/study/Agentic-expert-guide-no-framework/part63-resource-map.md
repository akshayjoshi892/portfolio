# Part LXIII — Resource Map

## What this page is for, in plain terms

Everything in Parts I–LII was written to teach durable concepts using specific tools as examples (a specific provider's API shape, a specific version of LangChain). Those specific details *will* drift out of date — that's not a flaw in the curriculum, it's the nature of fast-moving software. This page is where you go to find the *current* version of whatever a part showed you, and it's explicit about which category ("this idea will still be true in 5 years" vs. "this exact method name might not exist next release") anything falls into.

**A concrete example of the distinction this page draws:** Part XX taught you that LangGraph's core idea is "an explicit state graph with automatic, node-granular checkpointing." That idea is durable — it doesn't depend on any particular release. The exact code (`SqliteSaver.from_conn_string(...)`, the precise `interrupt_before` parameter name) is version-sensitive and could legitimately change. If code from this curriculum doesn't run against a current install, this page's discipline tells you what to do: check the *current* official docs for the durable idea's current API surface, rather than assuming your own code has a bug.

## Guiding Principle
Prioritize **primary sources** — official documentation and specifications — over secondary tutorials/blog posts, which go stale quickly for fast-moving tools. This part names categories and where to look, without fabricating specific URLs, package names, or claiming specific current features that may have changed since this curriculum was written — always verify against the current official source before relying on a specific API detail.

## Official Documentation and Specifications

| Area | What to consult |
|---|---|
| LLM provider APIs (OpenAI, Anthropic, others) | Each provider's official API reference — check the current docs for the latest model IDs, parameters, and structured-output/tool-calling mechanisms, since these evolve frequently |
| MCP | The official Model Context Protocol specification and SDK documentation |
| LangChain / LangGraph | Official docs at the LangChain project's documentation site; API references change across major versions — check the version pinned in your project |
| CrewAI | Official CrewAI documentation |
| Qdrant | Official Qdrant documentation and API reference |
| Python async | The official CPython `asyncio` documentation |
| Pydantic | Official Pydantic documentation (note: v1 vs v2 APIs differ substantially) |

## Academic Papers (Foundational — see also [Part LXIV](part64-research-paper-curriculum.md) for the full reading roadmap)
Look up directly rather than relying on secondhand summaries: the original Transformer paper, the ReAct paper, RAG's original paper, RLHF/InstructGPT-lineage papers, and RoPE's paper. Search academic sources (arXiv, ACL Anthology, official conference proceedings) for exact citations.

## Books
General categories worth seeking current editions of: a systems-design reference for distributed systems fundamentals (state, consistency, failure modes — directly relevant to Parts IX/XXXVI/XLIX); a deep learning fundamentals textbook for Part II/III's mathematical grounding; a software engineering practices reference for Part I's foundations.

## GitHub Repositories
- Official SDK repositories for whichever LLM providers, LangChain/LangGraph/CrewAI, and MCP servers you use — read the source, not just the docs, when debugging non-obvious behavior (this curriculum's "build from scratch first" philosophy extends to reading framework internals once you hit their limits).
- Reference MCP server implementations published by the protocol maintainers, useful as a second data point beyond this curriculum's from-scratch examples.

## Technical Blogs and Engineering Write-Ups
Prefer posts from teams operating agent systems at production scale (their own engineering blogs) over general "how to prompt engineer" content — production postmortems and architecture write-ups tend to age better and teach failure modes this curriculum's Part LVII catalog can't fully anticipate.

## Talks and Conferences
Conference talks from teams building production RAG/agent systems (search recent years of relevant AI/ML engineering conferences) — useful for production war-stories that complement this curriculum's theory.

## Courses
Useful as supplementary reinforcement, not a replacement for the build-first discipline this curriculum insists on (Rule 5) — treat any course as a source of *additional* practice projects, not passive content to consume.

## Benchmarks
Agent and RAG benchmarks are an active, fast-moving area (Part LII) — check current benchmark leaderboards/papers for what's actively measured, but apply Part XXXVIII's calibration discipline: a benchmark score is not a guarantee of your specific production reliability.

## Version-Sensitivity Discipline (Critical)

For LangChain, LangGraph, CrewAI, Qdrant, MCP, and every model provider API:
- **Durable concepts** (this curriculum's primary content): the *Runnable* composition idea, the state-graph-with-checkpointing idea, the role-based multi-agent idea, ANN indexing, the tool-discovery protocol idea — these survive version changes.
- **Version-sensitive APIs**: exact method names, exact parameter names, exact model identifiers, exact schema formats — these change across releases. Whenever this curriculum shows exact code, **verify it against current official documentation** before relying on it in production; treat every code sample here as illustrating the *pattern*, not as guaranteed-current API surface.
- When in doubt about whether something has changed, check the official changelog/release notes for the specific library before debugging your own code — a surprising number of "my code is broken" issues are actually "the API changed since this was written."
