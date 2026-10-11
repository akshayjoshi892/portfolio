# Part XIV — Embeddings

## Before you start

**What you need already:** [Part II](part02-ml-foundations.md) (vectors, cosine similarity, normalization — this part goes deeper on the same foundation, specifically for text). [Part XI](part11-rag-fundamentals.md)–[XIII] (you've been using embeddings as a black box; this part opens it).

**What this part gives you:** A real understanding of what an embedding model is actually doing, when a general-purpose one isn't good enough for your data, and — critically — what happens when you change embedding models on a system that's already live. That last one is a real production trap that's caught real teams off guard, and this part exists partly to make sure it doesn't catch you.

**Time:** 2 days.

**When you finish, you'll be able to:** Choose between embedding models with evidence instead of guessing, and safely migrate a live system from one embedding model to another without silently corrupting search results.

---

## Stage 1 — What an embedding model actually is

You've been calling `embed_fn(text)` since Part II and trusting it to turn text into a vector where similar meanings land close together. Time to open that box.

An embedding model is a neural network — the same building blocks from [Part II](part02-ml-foundations.md) and [Part III](part03-llm-fundamentals.md), attention layers and all — trained with a specific goal: given two pieces of text, if they're similar in meaning, their output vectors should have high cosine similarity; if they're unrelated, low similarity. It's not trained to *generate* text like an LLM is. Its entire job is to produce that one vector, well.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

pairs = [
    ("The cat sat on the mat.", "A feline rested on the rug."),      # paraphrase
    ("The cat sat on the mat.", "Stock prices fell sharply today."),  # unrelated
]

for a, b in pairs:
    embeddings = model.encode([a, b], normalize_embeddings=True)
    similarity = float(embeddings[0] @ embeddings[1])
    print(f"{similarity:.3f}   {a!r} vs {b!r}")
```

```
0.612   'The cat sat on the mat.' vs 'A feline rested on the rug.'
0.089   'The cat sat on the mat.' vs 'Stock prices fell sharply today.'
```

Notice: "cat" and "feline" never appear together, "sat on the mat" and "rested on the rug" share almost no words — and yet the model correctly scores them as meaningfully similar, while the genuinely unrelated pair scores near zero. This is exactly what training bought you: the model learned that these are *paraphrases*, not by matching words, but by learning what similar meaning looks like across millions of training examples. This is the entire reason embedding-based search beats plain keyword matching for the "different words, same meaning" case you saw work in Part II's project.

**Check yourself before Stage 2:** Why can an embedding model correctly score "cat" and "feline" content as similar, when a keyword-matching search never would?

---

## Stage 2 — Dense vs. sparse: two different ideas of "similar"

Everything so far has been **dense** embeddings — every one of the vector's dimensions has some non-zero value, and no single dimension corresponds to a specific word. There's a genuinely different approach worth knowing about: **sparse** embeddings, where most dimensions are exactly zero, and the non-zero ones roughly correspond to specific terms.

```python
# Illustrative, not a real sparse embedding implementation:
# dense:  [0.23, -0.71, 0.44, 0.12, -0.09, ...]   — every position has meaning, none tied to a word
# sparse: [0, 0, 0, 3.1, 0, 0, 0, 1.8, 0, ...]     — mostly zeros; non-zero positions ≈ specific terms
```

Dense embeddings are excellent at *semantic* similarity — the cat/feline example above. They're comparatively weaker at *exact term matching* — if a user searches for a specific product code, an exact error message, or a rare proper noun, a dense embedding model (trained on general meaning, not exact lexical matching) can sometimes rank a semantically-similar-but-wrong-term result above the document that contains the literal exact string the user typed.

Sparse methods (classic examples: TF-IDF, BM25, and learned variants like SPLADE) are the opposite: excellent at exact term matching, weaker at catching paraphrases and synonyms.

This isn't a "which is better" question — they fail on different things. That's exactly why **hybrid search**, which runs both and combines the results, exists — you'll build it properly in [Part XVII](part17-advanced-rag.md). For now, the important thing is recognizing that "embedding-based search" (dense only) has a real, specific blind spot: exact strings, codes, and rare terms.

**Check yourself before Stage 3:** Give an example query where a dense embedding model might underperform a simple exact-text search, and explain why.

---

## Stage 3 — Normalization, one more time, because it matters here specifically

You built `normalize()` in [Part II Stage 3](part02-ml-foundations.md) and saw why it lets you use the cheaper dot product instead of full cosine similarity. Real embedding libraries make this a parameter, not something you do by hand:

```python
embeddings = model.encode(["some text"], normalize_embeddings=True)
```

With `normalize_embeddings=True`, every vector that comes back already has length 1 — so `embeddings[0] @ embeddings[1]` (plain dot product) gives you exactly the same answer as full cosine similarity, at a fraction of the computational cost, because there's no square root or division left to do at query time. Every real vector database you'll use in [Part XV](part15-vector-databases.md) assumes this and defaults to a dot-product-optimized index — this is *why* that default makes sense, not an arbitrary implementation choice.

**Check yourself before Stage 4:** If you forget `normalize_embeddings=True`, what specifically goes wrong when you later compare vectors with plain dot product instead of full cosine similarity?

---

## Stage 4 — General-purpose models can fail on your specific domain

Here's a genuine risk worth testing for, not assuming away: an embedding model trained mostly on general web text, news, and books can underperform badly on text full of domain-specific jargon it rarely or never saw during training — legal contract language, medical terminology, your company's internal product names and acronyms.

```python
domain_pairs = [
    ("The plaintiff filed a motion for summary judgment.",
     "The claimant submitted a request to resolve the case without a full trial."),
    ("Patient presents with acute myocardial infarction.",
     "The patient is having a heart attack."),
]

for a, b in domain_pairs:
    embeddings = model.encode([a, b], normalize_embeddings=True)
    similarity = float(embeddings[0] @ embeddings[1])
    print(f"{similarity:.3f}   {a[:50]!r}... vs {b[:50]!r}...")
```

Run this with a general-purpose model on your own domain's real jargon and see how it does. A model that scored 0.6+ on the earlier cat/feline paraphrase might score noticeably lower here, even though "motion for summary judgment" and "request to resolve without a full trial" are just as much a paraphrase pair as "cat" and "feline" were — the model just has less training exposure to legal-specific phrasing to have learned that connection from.

The fix, when this is a real problem for your data: a domain-specific or fine-tuned embedding model, trained (or further trained) on text more like yours. But — and this is the important discipline — **don't assume you need one. Test first.** Many domains work fine with general-purpose models; jumping straight to a specialized model adds cost and complexity you might not need. This part's project makes you measure this on your own data rather than guess.

**Check yourself before Stage 5:** Why might a general-purpose embedding model underperform specifically on legal or medical text, given what you know about how it was trained?

---

## Stage 5 — The trap: embeddings from different models are not comparable

Here's the stage that exists specifically to prevent a real production incident. Read it carefully.

Two different embedding models can both produce, say, 384-number vectors. Structurally, they look identical — same shape, same data type. But they are **not comparable to each other**, even though nothing in the code would stop you from accidentally comparing them.

```python
model_a = SentenceTransformer("all-MiniLM-L6-v2")
model_b = SentenceTransformer("paraphrase-MiniLM-L3-v2")

text = "The user's subscription renews monthly."

vec_a = model_a.encode(text, normalize_embeddings=True)
vec_b = model_b.encode(text, normalize_embeddings=True)

# Both are valid, normalized vectors of similar dimensionality. But:
similarity = float(vec_a @ vec_b)
print(f"same text, different models: {similarity:.3f}")
```

You'll get some number — maybe 0.4, maybe 0.7 — for the *exact same sentence* compared against itself, just embedded by two different models. That number is meaningless. It's not measuring "how similar is this sentence to itself" (it should be 1.0). It's measuring an accident of how two unrelated models' internal, arbitrary coordinate systems happen to overlap, which is not a meaningful quantity at all.

### Why this causes a real incident

Imagine you built a vector index using `model_a`. Six months later, someone upgrades the embedding library, or switches providers, and the *query-time* embedding code starts using `model_b` — while the *already-stored* vectors in your index are still from `model_a`. Nothing crashes. No error is raised anywhere. Every query still returns results. They're just **silently, meaninglessly wrong** — ranked by a comparison that was never valid in the first place, and nothing in the system is positioned to notice.

### The fix: tag every stored vector with the model that produced it

```python
from dataclasses import dataclass


@dataclass
class TaggedEmbedding:
    vector: "np.ndarray"
    model_id: str   # e.g. "all-MiniLM-L6-v2"


def assert_compatible(query_model_id: str, index_model_id: str) -> None:
    if query_model_id != index_model_id:
        raise ValueError(
            f"embedding model mismatch: index was built with {index_model_id!r}, "
            f"but querying with {query_model_id!r}. These vectors are not comparable. "
            f"The index must be fully rebuilt with the new model before querying."
        )
```

This turns a silent, undetectable failure into a loud, immediate error — exactly the same philosophy as every validation check you've built since Part IV. **A model upgrade is not a small config change. It requires re-embedding your entire corpus and rebuilding the index from scratch**, because every previously-stored vector is now meaningless in comparison to new queries.

**Try it yourself:** Run the code above with two real, different embedding models and confirm you get a non-1.0 similarity for identical text. Seeing a "wrong" number for what should obviously be a perfect match is the moment this stops being an abstract warning and becomes something you've personally verified.

**Check yourself before the project:** Why does comparing vectors from two different embedding models produce a *number* rather than an error — and why does that make this specific bug so dangerous compared to one that crashes loudly?

---

# The Project: Choosing an Embedding Model With Evidence, and Surviving a Migration

## What you're building

An evaluation harness that compares two embedding models on your own data (deciding, with evidence, whether a bigger/different model is actually worth it), plus a safe migration procedure that never serves a query against mismatched vectors.

```bash
$ python embedding_eval.py --model-a all-MiniLM-L6-v2 --model-b bge-large-en
```

```
Running 15 test queries against both models...

all-MiniLM-L6-v2:  MRR: 0.71   avg query time: 8ms   dimensions: 384
bge-large-en:       MRR: 0.84   avg query time: 31ms  dimensions: 1024

bge-large-en wins on accuracy (+0.13 MRR) but costs ~4x the query latency
and ~2.7x the storage per vector. Decide based on your latency budget.
```

## Why this project

You now know, from Stage 5, exactly how badly a careless embedding-model change can go wrong in production. This project makes you practice both halves properly: choosing a model with real evidence, and migrating safely when you do need to change.

## Before you start

- Your search project from Part II, extended with real documents
- Two different embedding models to compare (any two real, different sentence-embedding models)

## Build it in stages

---

### Stage A — Reuse your evaluation harness from Part II

**Goal:** The MRR/recall@K evaluation you built in [Part II](part02-ml-foundations.md)'s project, ready to run against any embedding model as a parameter.

**Do:**
1. Refactor it so the embedding model is an argument, not hardcoded
2. Confirm it still produces the same numbers as before on your original model

**You're done when:** You can call your evaluation function with any embedding model and get back a score.

---

### Stage B — Compare two real models on your real data

**Goal:** Real MRR and recall@K numbers for two different models, plus their cost in latency and storage.

**Do:**
1. Run your evaluation harness with both models on the same document set and the same test questions
2. Measure query latency for each (time a single `model.encode()` call)
3. Note each model's output dimensionality (more dimensions = more storage per vector)
4. Report all of it side by side, as in the example output

**You're done when:** You have real numbers you can use to make an actual decision, not just an assumption that "bigger is better."

---

### Stage C — Demonstrate the mismatch bug, deliberately

**Goal:** Cause the exact silent-failure scenario from Stage 5, on purpose, in a safe test.

**Do:**
1. Build a small index using model A
2. Query it using model B's embeddings for the query (mixing models)
3. Observe: does it crash? Or does it return results that *look* normal but are actually meaningless?
4. Compare the ranking you get this way against the ranking you get when query and index use the same model consistently

**You're done when:** You've seen with your own eyes that the mismatched version doesn't crash — it just quietly returns different, wrong-feeling results, which is precisely why this bug is dangerous.

---

### Stage D — Build the safety check

**Goal:** `assert_compatible` wired into your actual retrieval code, so this specific bug becomes structurally impossible rather than just something you remember to avoid.

**Do:**
1. Tag every stored vector (or your whole index) with the `model_id` that produced it
2. Before every query, check the query embedding's model against the index's tagged model
3. Confirm that deliberately querying with a mismatched model now raises a clear error immediately, rather than silently returning bad results

**You're done when:** Stage C's exact scenario now fails loudly and immediately instead of silently.

---

### Stage E — Perform a real migration

**Goal:** Move a live index from model A to model B without ever serving a single query against mismatched vectors.

**Do:**
1. Build a *second*, parallel index using model B, alongside your existing model-A index — don't touch the original yet
2. Re-embed your full document set into the new index
3. Run your evaluation harness against the new index to confirm it performs as expected before cutting over
4. Only once you're confident, switch query traffic to the new index, and only then remove the old one

**You're done when:** You can describe, from having actually done it, why building the new index *alongside* the old one (rather than replacing in place) is what prevents a period of downtime or mismatched-vector queries during the transition.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Two models score identically on your eval | Test questions might be too easy for either model to distinguish quality — or the domain genuinely doesn't stress the difference | Add harder, more ambiguous test questions where model quality would actually matter |
| Mismatched-model comparison in Stage C doesn't look obviously "wrong" | The two models happen to produce roughly similar rankings on your small test set by coincidence | Try more test queries, or a query where the two models' understanding would clearly diverge |
| Migration Stage E briefly serves 0 results during cutover | Switching query traffic before the new index is fully populated and verified | Always verify the new index is complete and evaluated *before* redirecting any real traffic to it |

## Going further (optional)

1. **Multilingual test.** If you have access to a multilingual embedding model, test whether a French query can retrieve an English document discussing the same topic. This is the practical payoff of multilingual embeddings — cross-lingual retrieval — made concrete.
2. **Cost projection.** Using your Stage B latency and dimensionality numbers, project the storage cost difference between the two models at 1 million documents. Is the accuracy improvement worth it at that scale?
3. **A fine-tuning experiment**, if you have the resources: take a small set of domain-specific paraphrase pairs from your own data and see whether even light fine-tuning of a smaller model can close the gap with a larger general-purpose model, at a fraction of its latency cost.

---

## Checkpoint

**Knowledge:**
1. What is an embedding model actually trained to do, in one sentence?
2. Why do dense and sparse embeddings each have blind spots, and what's an example query that exposes each one?
3. Why does comparing embeddings from two different models produce a plausible-looking but meaningless number, rather than an error?
4. Why does a model upgrade require rebuilding the entire index, rather than just changing a config value?

**Implementation:** Build the model-mismatch safety check (`assert_compatible`, tagged embeddings) from memory.

**Debugging:** Retrieval quality across your whole system degraded overnight, for no apparent reason, right after a routine dependency update. What's the first thing you'd check, given what you now know about this part's central trap?

**Architecture:** Design a zero-downtime migration plan for a production vector index serving live queries, moving from one embedding model to another, using the parallel-index approach from Stage E.

---

**Next:** [Part XV — Vector Databases](part15-vector-databases.md), where you'll finally solve the scaling wall you measured yourself in Part II — searching millions of vectors in milliseconds instead of seconds.
