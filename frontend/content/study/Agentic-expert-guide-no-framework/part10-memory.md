# Part X — Memory

## Before you start

**What you need already:** [Part IX](part09-agent-state.md) (checkpointing — this part builds on the same atomic-write, versioned-database discipline), [Part II](part02-ml-foundations.md) (cosine similarity — you'll use it again here).

**What this part gives you:** State (Part IX) survives a crash *within one run*. Memory is different: it's what an agent remembers *across* separate runs, days apart, for a specific user. Without it, every conversation starts from zero — you'd have to re-explain your preferences, your role, and everything you told it last week, every single time.

**Time:** 3–4 days.

**When you finish, you'll be able to:** Build a memory system that recalls the right facts about a user when relevant, forgets or supersedes facts that have gone stale, and never leaks one user's information into another user's conversation.

---

## Stage 1 — State vs. memory: a distinction that actually matters

You might reasonably ask: isn't memory just... more state? Why does it get its own part?

Here's the distinction, made precise:

| | Agent state (Part IX) | Memory (this part) |
|---|---|---|
| **Lifetime** | One run. Deleted or archived once that run finishes. | Indefinite. Persists across many separate runs, potentially years. |
| **Scope** | Specific to one task ("what step am I on in this research task") | Specific to a user or entity ("what does this user prefer, what do I know about them") |
| **What breaks without it** | A crash loses in-progress work | Every conversation starts from zero, no continuity |

Concretely: "the agent is currently on step 3 of researching Acme Corp" is state — it means nothing once that research task is done. "This user always wants concise answers" is memory — it should still be true and still useful three months and two hundred conversations from now.

**Check yourself before Stage 2:** Give an example of something that's clearly state, and something that's clearly memory, for a coding assistant.

---

## Stage 2 — Not all memory is the same shape

Before building anything, notice that "remember things about the user" actually splits into a few genuinely different kinds of remembering, and conflating them causes real bugs.

| Kind | What it holds | Example |
|---|---|---|
| **Semantic** | Facts | "The user's company is Acme Corp" |
| **Episodic** | Specific past events | "On March 3rd, the user asked to cancel a deployment" |
| **Procedural** | Learned preferences about *how* to behave | "This user prefers terse, no-preamble answers" |

Why the split matters: these have different lifespans and different rules for going stale. A semantic fact ("works at Acme Corp") stays true until it explicitly changes. An episodic memory ("asked about X on March 3rd") is permanently true as a historical record but becomes less *relevant* to retrieve as time passes. A procedural preference is really a pattern inferred from repeated behavior, and should update gradually rather than flip on one contradicting data point.

Treating all three the same — dumping everything into one undifferentiated pile and retrieving by similarity alone — is where a lot of memory systems go wrong in practice. You'll see exactly how in Stage 5.

**Check yourself before Stage 3:** Classify each: "the user is a backend engineer," "the user asked about rate limiting yesterday," "the user tends to want code examples, not prose explanations." Which kind of memory is each?

---

## Stage 3 — Build a working store: the retrieval half

You already have everything you need for this — it's [Part II](part02-ml-foundations.md)'s embedding search, applied to a new kind of content.

```python
from __future__ import annotations
import sqlite3
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryItem:
    id: int
    user_id: str
    kind: str          # "semantic" | "episodic" | "procedural"
    content: str
    created_at: float
    relevance_score: float = 0.0   # filled in at query time, not storage time


class MemoryStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self.conn.commit()

    def write(self, user_id: str, kind: str, content: str, embedding: np.ndarray) -> int:
        cursor = self.conn.execute(
            "INSERT INTO memories (user_id, kind, content, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, kind, content, embedding.astype(np.float32).tobytes(), time.time()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def retrieve(self, user_id: str, query_embedding: np.ndarray, k: int = 3) -> list[MemoryItem]:
        # NOTE: filtering by user_id happens in the SQL query itself, not
        # afterward in Python. This isn't a style choice — Stage 7 explains
        # exactly why that distinction matters.
        rows = self.conn.execute(
            "SELECT id, user_id, kind, content, embedding, created_at "
            "FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()

        items = []
        for row in rows:
            emb = np.frombuffer(row[4], dtype=np.float32)
            score = float(query_embedding @ emb / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-12
            ))
            items.append(MemoryItem(id=row[0], user_id=row[1], kind=row[2],
                                     content=row[3], created_at=row[5], relevance_score=score))

        items.sort(key=lambda m: m.relevance_score, reverse=True)
        return items[:k]
```

```python
def fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=(16,)).astype(np.float32)


store = MemoryStore()
store.write("user_1", "semantic", "User's company is Acme Corp", fake_embed("Acme Corp company"))
store.write("user_1", "procedural", "User prefers terse commit messages", fake_embed("terse commit messages"))
store.write("user_1", "episodic", "Cancelled a deploy on 2026-08-01", fake_embed("cancelled deploy"))

results = store.retrieve("user_1", fake_embed("What does the user's company do?"), k=2)
for r in results:
    print(f"[{r.kind}] score={r.relevance_score:.3f} {r.content}")
```

Nothing here is new mechanics — it's [Part II](part02-ml-foundations.md)'s cosine similarity, [Part IX](part09-agent-state.md)'s persistence discipline. What's new is what you're storing and *why*, which is what the rest of this part is about.

**Check yourself before Stage 4:** In `retrieve`, why does `WHERE user_id = ?` appear in the SQL query itself rather than as a filter applied to the results afterward in Python?

---

## Stage 4 — The write half: when does something become a memory?

Storage and retrieval are the easy parts. The genuinely hard question is: **when should the agent decide something is worth remembering at all?**

Get this wrong in one direction and you remember nothing useful — every conversation really does start from zero. Get it wrong in the other direction and you remember *everything*, including noise, and Stage 6 shows exactly why that's its own serious problem.

A reasonable middle ground: after a conversation, extract only things that are (a) durably useful beyond this one exchange, and (b) not already known.

```python
def extract_candidate_memories(conversation: list[dict], llm_call) -> list[dict]:
    """A real version sends the conversation to the LLM with instructions like:
    'Extract any facts, preferences, or notable events worth remembering long-term.
    Skip anything trivial or already obvious. Return a JSON list of {kind, content}.'
    Simulated here to show the shape."""
    return [
        {"kind": "semantic", "content": "User's company is Acme Corp"},
        {"kind": "procedural", "content": "User prefers terse commit messages"},
    ]


def write_new_memories(store: MemoryStore, user_id: str, conversation: list[dict],
                        llm_call, embed_fn) -> None:
    candidates = extract_candidate_memories(conversation, llm_call)
    for c in candidates:
        embedding = embed_fn(c["content"])
        store.write(user_id, c["kind"], c["content"], embedding)
        print(f"remembered [{c['kind']}]: {c['content']}")
```

Notice this doesn't run on every single message — it runs at natural boundaries (end of a conversation, or periodically) and asks specifically "what's worth keeping," rather than storing every sentence exchanged. That distinction alone prevents most of the noise problem before it starts.

**Check yourself before Stage 5:** Why not just store every message in the conversation as a memory, rather than extracting specific candidates?

---

## Stage 5 — The failure mode that makes memory dangerous: contradictions

Here's a scenario that will happen in any real system, and it's the reason "just retrieve by similarity" isn't sufficient on its own.

```python
# Three months ago:
store.write("user_1", "semantic", "User's preferred contact email is jane@old-company.com", fake_embed("email"))

# Today, the user says something different:
# "My email is now jane@new-company.com"
```

If you just *add* the new fact without doing anything about the old one, retrieval now has two memories that directly contradict each other, both scoring similarly on any query about "the user's email." Which one does the agent trust? Depending on retrieval's exact ranking that day, it might tell someone the *old* email — confidently, because from the agent's perspective it's just reading a stored fact, with nothing flagging that the fact is stale.

This is not a hypothetical edge case. Preferences change. Job titles change. Contact info changes. Any real deployment hits this within its first few weeks.

### The fix: check for conflicts before writing, don't just append

```python
def contradicts(existing_content: str, new_content: str, llm_call) -> bool:
    """A real version asks the LLM: 'Does statement B contradict statement A,
    assuming both can't simultaneously be true?' Simulated for clarity."""
    return "email" in existing_content and "email" in new_content


def write_with_conflict_check(store: MemoryStore, user_id: str, kind: str,
                               new_content: str, embed_fn, llm_call) -> None:
    new_embedding = embed_fn(new_content)

    # Look for a highly similar EXISTING memory of the same kind first
    existing = store.retrieve(user_id, new_embedding, k=1)
    existing = [m for m in existing if m.kind == kind and m.relevance_score > 0.85]

    if existing and contradicts(existing[0].content, new_content, llm_call):
        print(f"superseding: {existing[0].content!r} -> {new_content!r}")
        store.conn.execute("DELETE FROM memories WHERE id = ?", (existing[0].id,))
        store.conn.commit()

    store.write(user_id, kind, new_content, new_embedding)
```

The check isn't "is this similar to something we already know" — high similarity is expected and fine for reinforcing facts. It's specifically "does this *contradict* something we already know," which is a different, narrower question that needs its own check, because a naive similarity-only system can't tell "this confirms what we knew" apart from "this replaces what we knew."

**Check yourself before Stage 6:** Why isn't high similarity alone enough to detect that a new memory should replace an old one? What could go wrong if you deleted the old memory purely because a new one was similar, without checking for an actual contradiction?

---

## Stage 6 — The failure mode that makes memory harmful: pollution

Contradictions are about *correctness*. This next problem is about *relevance*, and it's subtler because nothing is technically "wrong" — the memories are all true, they're just crowding out what actually matters.

Imagine a user who's had 200 conversations with your agent over a year. Every one of those generated a few memories. Now the agent is answering a new question, and it retrieves the "most similar" memories — but "most similar" doesn't mean "most useful right now." A memory from 11 months ago about a project that's long finished might still score reasonably well on similarity, and get pulled into context alongside — or instead of — something genuinely relevant from last week.

This is **memory pollution**: irrelevant or stale memories drowning out what actually matters, degrading the agent's answers even though every individual memory is technically true.

### The fix: relevance decays with age

```python
def apply_decay(items: list[MemoryItem], half_life_days: float = 60.0) -> list[MemoryItem]:
    """Older episodic memories become progressively less likely to surface,
    without being deleted outright — they're still there if genuinely relevant,
    just less likely to win against something more recent and equally similar."""
    now = time.time()
    for item in items:
        age_days = (now - item.created_at) / 86400
        decay_factor = 0.5 ** (age_days / half_life_days)
        item.relevance_score *= decay_factor
    return items
```

`0.5 ** (age_days / half_life_days)` is the same shape as radioactive half-life: after one `half_life_days` period, the score is multiplied by 0.5. After two periods, by 0.25. It never quite reaches zero, but it approaches it — old memories fade rather than vanish or stay permanently as strong as something from yesterday.

Apply this *after* retrieval, as a re-ranking step:

```python
def retrieve_with_decay(store: MemoryStore, user_id: str, query_embedding: np.ndarray, k: int = 3) -> list[MemoryItem]:
    candidates = store.retrieve(user_id, query_embedding, k=k * 3)  # over-fetch, then re-rank
    decayed = apply_decay(candidates)
    decayed.sort(key=lambda m: m.relevance_score, reverse=True)
    return decayed[:k]
```

Notice: **decay applies naturally to episodic memories** ("cancelled a deploy on March 3rd" genuinely does matter less as time passes) but should probably *not* apply, or apply much more slowly, to semantic facts ("works at Acme Corp") that don't have a natural expiration just from time passing — they go stale from *contradiction* (Stage 5), not from age. This is exactly why Stage 2's distinction between memory kinds matters in practice, not just as a taxonomy exercise: you'd tune decay differently per kind.

**Check yourself before Stage 7:** Why does memory pollution happen even when every stored memory is factually true? Why should decay apply differently to episodic memories versus semantic facts?

---

## Stage 7 — The failure mode that makes memory a liability: leakage across users

Go back and look again at `retrieve()` from Stage 3:

```python
rows = self.conn.execute(
    "SELECT ... FROM memories WHERE user_id = ?",
    (user_id,),
).fetchall()
```

That `WHERE user_id = ?` is not a minor implementation detail. It is the single most important line in this entire part. If it were missing — if `retrieve()` searched across *all* memories regardless of whose they are, filtering by user only after the fact, or not at all — you'd have a real risk of one user's private information surfacing in another user's conversation. Not a quality bug. A serious privacy and security failure, and in a real product, potentially a legal one.

```python
# DO NOT do this:
def retrieve_unsafe(self, query_embedding, k=3):
    rows = self.conn.execute("SELECT * FROM memories").fetchall()   # no user_id filter!
    # ... scores everyone's memories together, returns whoever's is most similar
```

Run that against a real multi-user system for even a short time and you will, eventually, leak. Not because of a clever attack — because two users happen to ask similar-enough questions and the embeddings happen to rank across the user boundary.

The fix isn't subtle: **every single query against the memory store must be scoped to the requesting user, at the database level, not as an afterthought filter in application code.** Application-level filtering is a second line of defense at best — a bug in the filtering logic that only gets caught after the database has already handed back cross-user data is a bug that's already leaked something by the time you catch it.

**Try it:** Deliberately remove the `WHERE user_id = ?` clause from your own `retrieve()`, insert memories for two different fake users with similar content, and query as one of them. Watch the other user's memory come back. Put the clause back immediately. Seeing this once, deliberately, in a safe local test, is worth more than any warning in prose — it's the difference between "I know I should filter by user" and "I have personally watched what happens when I don't."

**Check yourself before the project:** Why is filtering by `user_id` inside the SQL query itself meaningfully safer than fetching everything and filtering in Python afterward?

---

# The Project: A Memory System With Conflict Resolution and Decay

## What you're building

A memory store, wired into a simple conversational agent, that remembers facts about a user across separate sessions — correctly superseding contradictions, decaying stale relevance, and provably never leaking across users.

```bash
$ python memory_agent.py --user jane --message "My company is Acme Corp"
```

```
remembered [semantic]: User's company is Acme Corp
```

```bash
$ python memory_agent.py --user jane --message "What company do I work at?"
```

```
recalled: [semantic] User's company is Acme Corp (score: 0.91)
answer: You work at Acme Corp.
```

```bash
$ python memory_agent.py --user jane --message "Actually, I moved to Globex Inc"
```

```
superseding: 'User's company is Acme Corp' -> 'User's company is Globex Inc'
remembered [semantic]: User's company is Globex Inc
```

## Why this project

This is the memory layer under every multi-session agent in the rest of this curriculum, including the multi-agent enterprise assistant capstone. Getting conflict resolution and user isolation right here means you won't be debugging a privacy leak or a confidently-wrong stale fact in production later.

## Build it in stages

---

### Stage A — Real embeddings, persistent storage

**Goal:** `MemoryStore` from Stage 3, backed by real embeddings instead of `fake_embed`, persisted to an actual file (not `:memory:`).

**Do:**
1. Use a real embedding model (reuse Part II's project setup)
2. Point `sqlite3.connect` at a real file path
3. Write 5 memories for one fake user, restart the script, confirm they're still there

**You're done when:** Memories survive a process restart and retrieval returns sensible results for a query phrased differently than the stored content (the same "different words, same meaning" test from Part II).

---

### Stage B — Wire in real extraction

**Goal:** After a simulated "conversation," the agent decides what's actually worth remembering, using a real LLM call.

**Do:**
1. Write a prompt for `extract_candidate_memories` that asks the model to output a JSON list of `{kind, content}` worth keeping — skip small talk, skip anything already obvious
2. Feed it a few different fake conversations (one with a clear fact worth remembering, one with nothing memorable) and confirm it extracts the right amount — including sometimes correctly extracting *nothing*

**You're done when:** A boring conversation with no durable facts produces an empty extraction list, and a conversation containing a real fact produces exactly that fact, not a flood of trivial restatements.

---

### Stage C — Conflict detection and supersession

**Goal:** A contradicting new fact replaces the old one instead of sitting alongside it.

**Do:**
1. Implement `contradicts()` with a real LLM call — ask specifically "do these two statements conflict, assuming only one can be true?"
2. Wire it into `write_with_conflict_check`
3. Test it: store "user's email is X," then store "user's email is Y," confirm the old one is deleted and only the new one remains
4. Test the negative case too: store "user's company is Acme Corp," then store "user likes concise answers" — confirm these do *not* get treated as conflicting (they're unrelated facts, not contradictions)

**You're done when:** Both the positive case (real contradiction, old fact removed) and the negative case (unrelated facts, both kept) behave correctly.

---

### Stage D — Decay, measured

**Goal:** Confirm decay actually changes ranking outcomes, not just that the formula runs without error.

**Do:**
1. Insert two memories with similar content but very different `created_at` timestamps — you can set this directly for testing (simulate one being 6 months old, one being 1 day old)
2. Query with something that matches both about equally on raw similarity
3. Confirm the recent one now ranks higher after `apply_decay`, even if the old one had a marginally higher raw similarity score before decay was applied

**You're done when:** You've watched decay actually flip a ranking, not just trust that the math should do that.

---

### Stage E — Prove user isolation, adversarially

**Goal:** Don't just write the `WHERE user_id = ?` clause and assume it works — actively try to break it.

**Do:**
1. Create memories for two fake users, `jane` and `bob`, with deliberately similar content (e.g., both mention "my manager is difficult")
2. Query as `jane` and confirm only `jane`'s memories are ever returned, never `bob`'s
3. Write an automated test that would fail loudly if someone later "refactored" the filtering out of the SQL query and into application code incorrectly

**You're done when:** You have a test that specifically exists to catch a cross-user leak, and you've confirmed it actually fails if you deliberately break the query (remove the `WHERE` clause temporarily, watch the test fail, put it back).

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Old and new contradicting facts both come back on retrieval | `contradicts()` too narrow, or the similarity threshold before even checking for contradiction is set too high | Lower the similarity threshold that triggers a conflict check, or test `contradicts()` in isolation against your specific pair of facts |
| Extraction produces a memory for every single message, including trivial ones | Extraction prompt too permissive | Tighten the prompt: explicitly instruct it to return an empty list when nothing is durably worth keeping |
| Decay makes very old but still-relevant facts (e.g., "works at Acme Corp," set 6 months ago) disappear from results | Applying decay uniformly to semantic facts, which shouldn't decay just from age | Only decay episodic memories by default; keep semantic facts governed by conflict-resolution (Stage C), not time |
| Cross-user test doesn't actually catch a leak | Filtering by user happens in Python after fetching everything, so a naive test with small data might not surface it | Explicitly write a test with deliberately similar cross-user content, as in Stage E, not just distinct random data |

## Going further (optional)

1. **Consolidation.** Write a batch job that finds near-duplicate episodic memories (very high similarity, same kind) and merges them into a single, more general memory — e.g., five separate "asked about rate limiting" episodes over a month become one "frequently asks about rate limiting" procedural memory. This is the mechanism that keeps a long-lived user's memory store from growing without bound.
2. **A "forget me" endpoint.** Implement real deletion — given a `user_id`, remove every one of their memories, completely, from the store. Confirm nothing lingers. This is a direct, hands-on preview of the data-governance concerns in [Part XXXV](part35-security.md).
3. **Compare decay half-lives.** Try `half_life_days` at 7, 30, and 180 on the same test data. Which produces the most sensible ranking for your specific use case? There's no universally correct answer — it depends on how often your users' context actually changes.

---

## Checkpoint

**Knowledge:**
1. State the precise distinction between agent state (Part IX) and memory (this part).
2. Name the three kinds of memory from Stage 2 and give a fresh example of each.
3. Why isn't high similarity alone sufficient to detect that a new memory should replace an old one?
4. Why does memory pollution happen even when every individual memory is true?
5. Why must user-scoping happen inside the database query, not as an application-level filter applied afterward?

**Implementation:** Build `write_with_conflict_check` and `retrieve_with_decay` from memory, including the user-scoping.

**Debugging:** A user reports the agent confidently told them an old, outdated fact about themselves that they'd explicitly corrected weeks earlier. Walk through which stage's mechanism most likely failed, and how you'd confirm it.

**Architecture:** You're designing memory for a multi-tenant product where each tenant is a company with many individual users, and memories should be scoped per-user *within* a tenant, with tenant-level isolation as an additional hard boundary on top. Sketch the schema and query changes needed beyond the single-user version built in this part.

---

**Next:** [Part XI — RAG Fundamentals](part11-rag-fundamentals.md), where the same retrieval mechanism you've now used twice — for search in Part II, for memory here — gets applied to grounding an agent's answers in real documents.
