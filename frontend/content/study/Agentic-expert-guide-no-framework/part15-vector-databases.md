# Part XV — Vector Databases

## Before you start

**What you need already:** [Part II Stage 5](part02-ml-foundations.md) — you personally measured brute-force search slow down from milliseconds to over a second as the corpus grew. This part solves exactly that problem, which you already know is real because you watched it happen.

**What this part gives you:** An understanding of *how* vector databases achieve millisecond search over millions of vectors — not just that they do — plus the mechanics of filtering, persistence, and the trade-off every fast search index makes.

**Time:** 3 days.

**When you finish, you'll be able to:** Explain why approximate search is fast, tune the accuracy/speed trade-off deliberately, and combine similarity search with filters (like "only documents this user has access to") correctly.

---

## Stage 1 — Recall your own measurement

Go back to what you found in Part II:

```
    1,000 vectors:      1.8 ms
   10,000 vectors:     14.2 ms
  100,000 vectors:    138.6 ms
1,000,000 vectors:   1420.3 ms
```

Every search compares the query against *every single stored vector*. Double the vectors, double the time — a straight line. At a million vectors, you're at 1.4 seconds per search, which is unusable for anything interactive. At real production scale — tens of millions of documents — this approach simply cannot work, no matter how fast your CPU is, because the *number of comparisons* is the bottleneck, not the speed of any single comparison.

The fix has to change the *number of comparisons*, not just make each one faster. That's what this whole part is about.

**Check yourself before Stage 2:** Why doesn't a faster CPU fix the scaling problem you measured in Part II?

---

## Stage 2 — The idea: don't compare against everything

Here's the trick, in plain language before any algorithm: **organize the vectors in advance so that, at search time, you only need to look at a small, well-chosen subset — not all of them — and that subset is very likely to contain the true best matches.**

This is called **approximate nearest neighbor** search (ANN). "Approximate" is the honest, important word in that name: you're trading a small chance of missing the absolute best match for an enormous speedup. In practice, a well-tuned ANN index finds the true best match, or something extremely close to it, the vast majority of the time — and does so in milliseconds instead of seconds.

There are a few different ways to organize vectors for this. The one you'll actually use, because it's what most modern vector databases (including Qdrant, which you'll meet in [Part XVI](part16-qdrant-deep-dive.md)) default to, is called **HNSW**.

**Check yourself before Stage 3:** What does "approximate" mean in "approximate nearest neighbor," and what are you trading for the speedup?

---

## Stage 3 — HNSW, explained the way you'd actually navigate a map

HNSW stands for Hierarchical Navigable Small World. The name is dense; the underlying idea is intuitive if you think about it as a map with multiple zoom levels.

Imagine trying to find a specific address in a city you don't know. You wouldn't walk down every single street checking every address one by one — that's brute force. Instead, you'd probably start with a highway map (very few roads, covers huge distances fast), get roughly to the right neighborhood, then switch to a city map (more detail, smaller area), then finally a street-level map to find the exact address.

HNSW builds exactly this kind of multi-layer structure over your vectors:

```
Layer 2 (top, sparse):     A -------- E -------- H
                            |          |          |
Layer 1 (medium):          A -- C -- E -- F -- H -- J
                            |    |    |    |    |    |
Layer 0 (bottom, dense):   A-B-C-D-E-F-G-H-I-J-K-L-M-N
```

A search starts at the top layer, where there are very few points and connections span large distances — like the highway map, it gets you roughly in the right area in very few steps. Then it drops down a layer, where there are more points and finer connections, and refines its position. It keeps dropping and refining until it reaches the bottom layer, which has *every* vector, but by this point the search is already very close to the right answer, so it only needs to check a small number of nearby candidates rather than the whole dataset.

This is why HNSW search is fast: it never compares your query against every vector. It compares against a small number of "signposts" at each layer, using them to navigate toward the right neighborhood, and only does a full, careful comparison once it's already close.

**Check yourself before Stage 4:** Using the highway/city/street map analogy, explain why HNSW search doesn't need to examine every stored vector.

---

## Stage 4 — Build and query a real HNSW index

Time to stop reasoning about it abstractly and use a real implementation.

```python
import hnswlib
import numpy as np

dim = 128
index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(max_elements=100_000, ef_construction=200, M=16)

rng = np.random.default_rng(0)
vectors = rng.normal(size=(50_000, dim)).astype(np.float32)
ids = np.arange(50_000)

index.add_items(vectors, ids)

index.set_ef(50)   # query-time search thoroughness — more on this in Stage 5

query = vectors[42]   # search for something we know is in the index
labels, distances = index.knn_query(query, k=5)
print("top match:", labels[0][0], "distance:", distances[0][0])
```

```
top match: 42 distance: 0.0
```

The top result is exactly the vector we searched for, at distance 0 — as expected, since we queried with a vector already in the index. Now time it against a much larger set and compare to your Part II brute-force numbers:

```python
import time

for n in [10_000, 100_000, 1_000_000]:
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    idx.add_items(vecs, np.arange(n))
    idx.set_ef(50)

    query = rng.normal(size=(dim,)).astype(np.float32)
    start = time.monotonic()
    idx.knn_query(query, k=5)
    elapsed = time.monotonic() - start

    print(f"{n:>9,} vectors: {elapsed*1000:>7.2f} ms")
```

```
   10,000 vectors:    0.31 ms
  100,000 vectors:    0.48 ms
1,000,000 vectors:    0.79 ms
```

Compare directly against Part II's brute-force numbers at the same sizes: 14.2ms, 138.6ms, 1420.3ms. At a million vectors, HNSW is roughly **1,800 times faster** than brute force. And notice the shape of the HNSW numbers: they barely grow at all as the corpus grows — nowhere near the straight line you saw with brute force. That's the entire structural payoff of not comparing against everything.

**Check yourself before Stage 5:** Compare the growth pattern of the HNSW timings to the brute-force timings from Part II. What does the difference in *shape*, not just raw speed, tell you about how each one scales?

---

## Stage 5 — The knob: `ef`, and the trade-off it controls

That `set_ef(50)` call in the code above wasn't arbitrary. `ef` controls how many candidates the search examines at the bottom layer before deciding on its final answer — a direct, tunable knob on the speed/accuracy trade-off HNSW is built on.

```python
def measure_ef_tradeoff(index, queries, ground_truth_ids, ef_values):
    for ef in ef_values:
        index.set_ef(ef)

        correct = 0
        start = time.monotonic()
        for q, true_id in zip(queries, ground_truth_ids):
            labels, _ = index.knn_query(q, k=1)
            if labels[0][0] == true_id:
                correct += 1
        elapsed = time.monotonic() - start

        recall = correct / len(queries)
        avg_latency_ms = (elapsed / len(queries)) * 1000
        print(f"ef={ef:>4}:  recall={recall:.2%}  avg latency={avg_latency_ms:.3f}ms")
```

Running this against a real test set typically shows a pattern like:

```
ef=  10:  recall=87.00%  avg latency=0.21ms
ef=  50:  recall=97.00%  avg latency=0.42ms
ef= 200:  recall=99.60%  avg latency=1.30ms
ef=1000:  recall=99.98%  avg latency=5.80ms
```

Low `ef` means the search examines fewer candidates — very fast, but it occasionally misses the true best match, because it stopped looking too early. High `ef` means it examines more candidates — slower, but much more likely to find the actual best match.

This is a genuine, deliberate design decision you get to make, not a bug to fix: **a chatbot answering in real time might reasonably accept 97% recall for sub-millisecond latency. A nightly batch analytics job with no real-time constraint might reasonably want 99.98% recall and not care that it takes 6ms instead of 0.4ms.** There's no universally correct value — it depends entirely on what your system actually needs, and you set it deliberately based on that, not by copying a default from documentation.

**Check yourself before Stage 6:** If a production chatbot's search results occasionally seem to "miss" a document that should obviously have matched, what configuration value would you check first, given what this stage just showed you?

---

## Stage 6 — Filtering: "search, but only within X"

Real queries are rarely pure similarity search. You usually want: "find the most similar documents, but only ones this user has permission to see," or "only from the last 30 days," or "only in this specific project." This is **filtered search**, and it's where a naive approach breaks down in a way worth understanding.

### The naive approach, and why it fails at scale

```python
def naive_filtered_search(index, query, allowed_ids: set, k=5, over_fetch=5):
    """Fetch more than k, then filter down to allowed ones."""
    labels, distances = index.knn_query(query, k=k * over_fetch)
    filtered = [(l, d) for l, d in zip(labels[0], distances[0]) if l in allowed_ids]
    return filtered[:k]
```

This works when `allowed_ids` covers a large fraction of the index — fetch a bit extra, filter, done. But imagine `allowed_ids` covers only 1% of the corpus (a specific user's private documents, in a corpus shared across thousands of users). Fetching `k * 5` extra candidates and filtering might still return *zero* results, because the top matches by pure similarity are overwhelmingly likely to belong to someone else. You'd need to fetch an enormous, unpredictable number of extra candidates to reliably find even a handful that pass the filter — which defeats the entire speed advantage HNSW gave you in the first place.

### The real fix: filtering integrated into the search itself

Production vector databases solve this properly by making the filter part of the search's graph traversal itself — as it navigates through the layers described in Stage 3, it only considers candidates that already satisfy the filter, rather than searching blindly and filtering afterward. This requires the database to index the filterable fields (a **payload index**, covered properly with Qdrant in [Part XVI](part16-qdrant-deep-dive.md)) so it can efficiently answer "does this candidate satisfy the filter" without checking every single one by hand.

The practical lesson for now: **know whether your vector database applies filters before or after the similarity search.** "Before" (integrated into traversal) scales correctly even for highly selective filters. "After" (fetch-then-filter, like the naive version above) can silently return too few or zero results once your filter is selective enough — and this bug tends to appear only after you're already in production with real, skewed data, not during early testing with a small, evenly-distributed test set.

**Check yourself before the project:** Why does fetch-then-filter break down specifically when the filter is *highly selective* (matches only a small fraction of the corpus), even though it works fine for a loose filter?

---

# The Project: Benchmark the Wall You Measured, Then Break Through It

## What you're building

A direct, side-by-side comparison of your Part II brute-force search against a real HNSW index, on the same data, at the same scales — plus a working filtered search that correctly handles a highly selective filter.

```bash
$ python vector_db_benchmark.py
```

```
Comparing brute-force (Part II) vs. HNSW at matching corpus sizes:

  n=1,000     brute-force:    1.9ms   HNSW:   0.18ms   (10.6x faster)
  n=10,000    brute-force:   14.5ms   HNSW:   0.22ms   (65.9x faster)
  n=100,000   brute-force:  139.1ms   HNSW:   0.41ms   (339.3x faster)
  n=1,000,000 brute-force: 1418.7ms   HNSW:   0.83ms   (1709.3x faster)

ef sweep at n=100,000:
  ef=10:   recall=86.4%  latency=0.19ms
  ef=50:   recall=97.1%  latency=0.41ms
  ef=200:  recall=99.5%  latency=1.24ms

Filtered search (1% of corpus matches filter):
  naive fetch-then-filter (k=5, over_fetch=5x):  1/5 results found (FAILED)
  naive fetch-then-filter (k=5, over_fetch=50x): 5/5 results found, 2.1ms
```

## Why this project

This is the direct payoff of the scaling wall you measured in Part II — you now have both sides of the comparison, from your own benchmarks, and understand exactly why one approach scales and the other doesn't. This benchmark harness is also what you'll reuse to evaluate Qdrant itself in [Part XVI](part16-qdrant-deep-dive.md).

## Before you start

- `pip install hnswlib`
- Your brute-force `find_top_k_fast` function from Part II

## Build it in stages

---

### Stage A — Side-by-side benchmark at matching sizes

**Goal:** Reproduce the exact comparison in the example output, using your own two implementations.

**Do:**
1. At each corpus size (1k, 10k, 100k, 1M), build both a brute-force array and an HNSW index over the *same* random vectors
2. Time a search against each
3. Print the speedup ratio at each size

**You're done when:** You have real numbers confirming HNSW's near-flat scaling against brute-force's straight-line growth, directly comparable to your Part II measurements.

---

### Stage B — The `ef` sweep, with real recall measurement

**Goal:** Real recall numbers at different `ef` values, not just latency.

**Do:**
1. Build an index of 100,000 vectors
2. For a set of 200 test queries, compute the *true* nearest neighbor using brute force (this is your ground truth)
3. At each `ef` value (10, 50, 200, 1000), run the same queries through HNSW and check what fraction match the ground truth
4. Report recall and latency together at each `ef`

**You're done when:** You can point to a real number and say "at `ef=50` on my data, I get X% recall at Y ms" — and explain what changing `ef` would trade away.

---

### Stage C — Break the naive filtered search, on purpose

**Goal:** Watch fetch-then-filter fail on a highly selective filter, with your own eyes, before fixing it.

**Do:**
1. Build an index where each vector has an associated "owner" — 100 owners, one of them (`owner_0`) has only 1% of the vectors
2. Implement `naive_filtered_search`, searching for `k=5` results belonging to `owner_0`
3. Try it with `over_fetch=5` — count how many valid results actually came back
4. Try again with a much larger `over_fetch` and see how large it needs to be before you reliably get 5 results

**You're done when:** You've watched the naive approach genuinely fail (return fewer than `k` results) at a realistic over-fetch multiplier, and measured how much over-fetching it takes to fix — quantifying exactly how wasteful the naive approach is for selective filters.

---

### Stage D — A correct filtered search

**Goal:** A filtering approach that doesn't require guessing an over-fetch multiplier.

**Do:**
1. Since `hnswlib` doesn't support integrated filtering natively, implement a simple two-tier approach: maintain a separate index *per owner* for small/selective groups, falling back to full search + filter for large/unselective ones
2. Compare its reliability against the naive approach from Stage C on the same `owner_0` query

**You're done when:** Your filtered search reliably returns `k` results for the selective owner, every time, without needing an unpredictable over-fetch guess — and you understand this as a hand-built stand-in for what Qdrant's payload indexes will do properly in [Part XVI](part16-qdrant-deep-dive.md).

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| HNSW build takes a very long time at 1M vectors | `ef_construction` set very high, or `add_items` called one at a time instead of in a single batch | Batch `add_items` calls; keep `ef_construction` reasonable (100–300 is typical) for testing |
| Recall numbers look suspiciously perfect (100%) at every `ef` | Ground truth computed using the same approximate index instead of true brute force | Recompute ground truth with actual brute-force search, not HNSW at a high `ef` |
| Naive filtered search "works" even at low over-fetch | Test filter isn't actually selective enough — owner has more than 1% of vectors | Verify the exact fraction with a direct count before concluding the naive approach is fine |

## Going further (optional)

1. **Persistence.** Save your HNSW index to disk (`index.save_index(path)`), restart your program, load it back (`index.load_index(path, max_elements=...)`), and confirm search still works without rebuilding from scratch. This is the mechanism behind not having to re-embed your whole corpus every time your service restarts.
2. **Soft delete.** `hnswlib` supports `mark_deleted(id)`. Delete a vector, confirm it no longer appears in search results, and think through why "mark as deleted" rather than physically removing it might be the safer default for a production system under concurrent access.
3. **Compare `M`.** `M` controls how many connections each point maintains in the graph (Stage 3's layered map). Try `M=8` vs. `M=48` at a fixed `ef`, and observe the trade-off in both memory usage and recall.

---

## Checkpoint

**Knowledge:**
1. In your own words, using the map analogy, explain why HNSW search doesn't need to compare against every stored vector.
2. What does `ef` control, and what do you trade for a higher value?
3. Why does fetch-then-filter fail specifically for highly selective filters, and what does "filtering integrated into the search" do differently?

**Implementation:** Build an HNSW index, tune `ef`, and measure real recall against brute-force ground truth, from memory.

**Debugging:** A production search system's users start complaining that obviously-relevant documents are occasionally missing from results, though the system was working fine at launch with a smaller corpus. What configuration value would you check first, and why would this problem specifically correlate with corpus growth?

**Architecture:** You need sub-100ms search over 500 million vectors, split across a highly imbalanced set of tenants (one huge tenant, thousands of tiny ones). Sketch, at a high level, how you'd combine everything from this part — HNSW, the `ef` trade-off, and filtering strategy — into a real design.

---

**Next:** [Part XVI — Qdrant Deep Dive](part16-qdrant-deep-dive.md), where everything you just built by hand becomes a production-grade managed service, with proper filtering, persistence, and hybrid search built in.
