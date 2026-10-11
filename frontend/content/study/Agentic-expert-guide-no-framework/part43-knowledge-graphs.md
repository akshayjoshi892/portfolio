# Part XLIII — Knowledge Graphs

## Before you start

**What you need already:** [Part XVII Stage 4](part17-advanced-rag.md) — you already hit the wall this part fixes: a genuinely multi-hop question that vector retrieval structurally cannot answer, no matter how good your embeddings or chunking are.

**What this part gives you:** A different retrieval mechanism — one built for precise relationship traversal instead of semantic similarity — and a real understanding of exactly which class of question needs it.

**Time:** 2 days.

**When you finish, you'll be able to:** Build and query a knowledge graph, extract structured relationships from unstructured text, and combine graph traversal with vector search where each one's strength covers the other's gap.

---

## Stage 1 — Recall exactly why vector search failed you

Go back to [Part XVII Stage 4](part17-advanced-rag.md): "Who is the CEO of the company that acquired our main competitor?" No single chunk contains the full chain — competitor identity, the acquisition, the CEO — spread across three separate documents. You fixed this with decomposition: break into sub-questions, retrieve for each separately.

That fix works, but it has a real limitation worth naming honestly: it requires the LLM to correctly *guess* how to decompose the question, and it runs a full similarity search per sub-question, hoping each one lands on the right document by semantic proximity. For a question like "who reports to the manager of the team that shipped feature X" — a precise, structural relationship query — semantic similarity is fundamentally the wrong tool. There's no meaningful sense in which "who reports to whom" is a *similarity* question at all.

**Check yourself before Stage 2:** Why is "who reports to the manager of the team that shipped feature X" not really a similarity question, even though it can be phrased as natural language?

---

## Stage 2 — A different data structure for a different kind of question

A **knowledge graph** represents facts as **triples** — subject, predicate, object — like `(Jane Doe, manages, Bob Smith)`. Instead of searching for text that's *similar* to a query, you **traverse** explicit, named relationships, precisely, the way you'd follow links in a diagram.

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str


class KnowledgeGraph:
    def __init__(self):
        self.triples: list[Triple] = []
        self._out_edges: dict[str, list[Triple]] = {}

    def add(self, subject: str, predicate: str, obj: str) -> None:
        t = Triple(subject, predicate, obj)
        self.triples.append(t)
        self._out_edges.setdefault(subject, []).append(t)

    def neighbors(self, entity: str, predicate: str | None = None) -> list[Triple]:
        edges = self._out_edges.get(entity, [])
        return [e for e in edges if predicate is None or e.predicate == predicate]
```

```python
kg = KnowledgeGraph()
kg.add("Jane Doe", "manages", "Bob Smith")
kg.add("Bob Smith", "manages", "Alice Wu")
kg.add("Jane Doe", "works_at", "Acme Corp")

print(kg.neighbors("Jane Doe"))
```

```
[Triple(subject='Jane Doe', predicate='manages', obj='Bob Smith'), Triple(subject='Jane Doe', predicate='works_at', obj='Acme Corp')]
```

There's no similarity score anywhere here. `neighbors("Jane Doe")` returns *exactly* the facts stored about Jane Doe — precisely, not approximately. This is the fundamental trade-off against vector search: a knowledge graph gives you precision and exactness for relationships that were actually captured in it, at the cost of not being able to handle fuzzy, semantic, "find me something related to this idea" queries at all — that's what vector search is for, and neither tool substitutes for the other.

**Check yourself before Stage 3:** Contrast `kg.neighbors("Jane Doe")` with `retrieve(index, "Jane Doe", ...)` from your RAG project. What does each one guarantee that the other doesn't?

---

## Stage 3 — Multi-hop traversal: the exact fix for the exact problem

```python
    def multi_hop(self, start: str, path: list[str]) -> list[str]:
        """Follow a fixed sequence of predicates, e.g. ['manages', 'manages']
        to find everyone TWO management levels below `start`."""
        frontier = [start]
        for predicate in path:
            next_frontier = []
            for entity in frontier:
                next_frontier.extend(e.obj for e in self.neighbors(entity, predicate))
            frontier = next_frontier
        return frontier


print(kg.multi_hop("Jane Doe", ["manages", "manages"]))
```

```
['Alice Wu']
```

This directly answers exactly the kind of question that broke vector retrieval in Part XVII — a precise chain of relationships, followed step by step, with no ambiguity or approximation at any hop. Compare this to Part XVII's decomposition fix: that approach used an LLM to *guess* at sub-questions and hoped semantic search would land on the right facts each time. This approach, when the graph actually contains the relevant relationships, gets the exact right answer deterministically, every time.

**Check yourself before Stage 4:** Why is `multi_hop`'s result deterministic and exact, while Part XVII's decomposition-based multi-hop RAG was only probabilistically likely to find the right chain?

---

## Stage 4 — Where the graph's facts come from: extraction, and its real risk

You don't get a knowledge graph for free — you have to build it, usually by extracting triples from unstructured text, exactly the kind of documents you've been chunking and embedding throughout Parts XI–XVIII.

```python
async def extract_triples_from_text(text: str, client) -> list[Triple]:
    prompt = f"""Extract factual (subject, predicate, object) triples from this
text. Return a JSON list of [subject, predicate, object] arrays. Only extract
facts EXPLICITLY stated in the text — do not infer or guess anything not
directly present.

TEXT: {text}
"""
    response, _ = await client.ask(prompt, temperature=0)
    import json
    raw_triples = json.loads(response)
    return [Triple(s, p, o) for s, p, o in raw_triples]
```

Notice the explicit instruction: "only extract facts EXPLICITLY stated... do not infer or guess." This matters more here than it might in ordinary generation, for a specific, serious reason worth understanding: **a hallucinated triple doesn't look like a hallucination once it's in the graph.** In a normal generated answer, a wrong claim is one sentence among many, and a careful reader might catch it. But a fabricated triple sits in your graph *structurally identical* to every real one — `neighbors()` and `multi_hop()` will confidently return it, with the same apparent authority as a fact that was actually extracted correctly, because nothing in the graph's structure distinguishes "verified fact" from "model's confident guess." A graph built carelessly is arguably *worse* than no graph, because it looks authoritative while potentially being wrong in ways ordinary RAG's citation trail wouldn't hide as effectively.

**Check yourself before Stage 5:** Why is a hallucinated triple in a knowledge graph more dangerous, in a specific way, than a hallucinated claim in an ordinary generated RAG answer?

---

## Stage 5 — Entity resolution: the problem that quietly fragments your graph

One more real problem, easy to miss until it bites you. If your source documents refer to the same entity inconsistently — "Acme Corp," "Acme Corporation," "Acme" — naive extraction creates three *separate* nodes in your graph, none of them connected to each other. Any traversal starting from "Acme Corp" silently misses everything that was extracted under "Acme Corporation," with no error, no warning — the query just returns an incomplete answer.

```python
def resolve_entity(name: str, known_entities: list[str], similarity_fn) -> str:
    """Merge near-duplicate mentions into one canonical entity, using
    similarity (reusing Part II's cosine similarity) as the matching signal."""
    best_match, best_score = None, 0.0
    for known in known_entities:
        score = similarity_fn(name, known)
        if score > best_score:
            best_match, best_score = known, score
    return best_match if best_score > 0.85 else name   # new entity if no good match
```

Without this step, your graph doesn't fail loudly — it just silently fragments, and every query that should have traversed through a merged entity instead hits a dead end at one of its several disconnected aliases, with the query returning "no results" rather than any indication that the real answer exists under a slightly different name.

**Check yourself before the project:** Why does entity fragmentation fail silently rather than producing an obvious error?

---

# The Project: A Graph RAG System, Compared Against Pure Vector RAG

## What you're building

Extract triples from a small corpus, build a graph with entity resolution, and answer genuinely multi-hop questions — then compare against your Part XVII vector-based multi-hop approach on the exact same questions.

```bash
$ python graph_rag.py "Who is two management levels below Jane Doe?"
```

```
graph traversal: Jane Doe -manages-> Bob Smith -manages-> Alice Wu
answer: Alice Wu

--- comparison ---
vector-based multi-hop (Part XVII): "Alice Wu" (correct, but required 2 LLM
  decomposition calls and 2 retrieval passes, ~1.8s)
graph traversal (this part): "Alice Wu" (deterministic, 1 extraction pass
  done once at index time, then instant traversal, ~0.05s per query)
```

## Why this project

This gives you direct, measured evidence for exactly when graph-based retrieval is worth the extra ingestion complexity — precise relationship questions over a stable, entity-rich domain — versus when plain or vector-augmented RAG is simpler and sufficient.

## Build it in stages

---

### Stage A — Extract triples from a real small corpus

**Goal:** A `KnowledgeGraph` built from 10–20 short documents containing real relationships (org charts, company/acquisition data, or any entity-relationship-rich domain you have real content for).

**Do:**
1. Implement `extract_triples_from_text`
2. Run it across your corpus
3. Manually spot-check 10 extracted triples against the source text — confirm each is genuinely, explicitly stated, not inferred

**You're done when:** You've verified, by hand, that extraction is faithful — not hallucinating relationships that aren't actually in the source text.

---

### Stage B — Entity resolution, tested against real fragmentation

**Goal:** `resolve_entity`, correctly merging near-duplicate mentions in your real extracted data.

**Do:**
1. Deliberately check your corpus for entities referred to inconsistently (or add such variation if your source text doesn't naturally have it)
2. Apply resolution during graph construction
3. Confirm a query against the canonical name now correctly finds facts that were extracted under a variant name

**You're done when:** You've watched entity resolution actually fix a fragmentation case you introduced or found.

---

### Stage C — Multi-hop queries, correctness-checked

**Goal:** Answer 5 genuinely multi-hop questions using graph traversal.

**Do:**
1. Write 5 questions requiring 2+ hops through your graph
2. Implement traversal for each (extending `multi_hop` as needed for non-uniform hop patterns)
3. Manually verify each answer against the source documents

**You're done when:** All 5 multi-hop questions are answered correctly via traversal.

---

### Stage D — Head-to-head comparison against vector-based multi-hop

**Goal:** The exact same 5 questions, run through your Part XVII decomposition-based approach, compared on correctness, latency, and cost.

**Do:**
1. Run the same 5 questions through your Part XVII multi-hop RAG pipeline
2. Compare answers, latency, and number of LLM calls for both approaches
3. Write a short, honest conclusion: for this specific kind of question, which approach actually won, and by how much

**You're done when:** You have real, side-by-side numbers backing a genuine conclusion — not an assumption that graphs are "obviously better" for this use case.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Extracted triples include inferred relationships not stated in the text | Extraction prompt too permissive, allowing the model to "fill in" plausible-seeming connections | Tighten the prompt's explicit-only instruction; spot-check more aggressively |
| Multi-hop query returns nothing despite the facts existing in the source | Entity fragmentation (Stage 5), or the hop predicates don't exactly match what was extracted | Check exact predicate/entity name matching; apply entity resolution |
| Graph traversal is much faster but the vector-based version is nearly as accurate | This is expected on some corpora — record it honestly as a finding, not a failure of the project | Report which specific factor (precision, speed, or ingestion cost) actually differentiated the two approaches for your data |

## Going further (optional)

1. **Combine both approaches**, as in this part's Stage 6 pattern — use graph traversal for the structural part of a question and vector search for surrounding unstructured context, and test on a question that genuinely benefits from both.
2. **Graph updates over time.** Simulate a source document changing (a person's role changing) and think through how you'd update the graph without leaving stale triples behind — a real, unsolved-by-this-project problem worth understanding the difficulty of.
3. **A visual export** of your graph (even simple text-based, or using a graph visualization library) to actually see its structure, which is often more revealing than reading raw triples.

---

## Checkpoint

**Knowledge:**
1. Why is "who reports to the manager of team X" not really a semantic-similarity question, even phrased as natural language?
2. Why is a hallucinated triple more dangerous than a hallucinated sentence in an ordinary RAG answer?
3. Why does entity fragmentation fail silently, and what specifically does entity resolution fix?

**Implementation:** Build `KnowledgeGraph` with multi-hop traversal and entity resolution from memory.

**Debugging:** A multi-hop graph query returns no results despite the underlying facts clearly existing in your source documents. Walk through how you'd determine whether the cause is extraction, entity resolution, or the traversal query itself.

**Architecture:** Design a Graph RAG ingestion pipeline that stays consistent as source documents are updated or deleted over time — what has to happen to the graph when a source document that contributed several triples is removed?

---

**Next:** [Part XLIV — Code Agents](part44-code-agents.md), where you'll build an agent specialized for a domain with its own precise structure — source code — combining planning, tool use, and real verification (running actual tests) into one coherent system.
