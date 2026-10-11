# Part XVI — Qdrant Deep Dive

## Before you start

**What you need already:** [Part XV](part15-vector-databases.md) — you understand HNSW, `ef`, and why fetch-then-filter breaks down for selective filters. This part is about using a real, production-grade tool that implements all of that correctly, so you don't have to hand-roll it.

**What this part gives you:** Everything you built by hand in Part XV — indexing, filtering, persistence — as a real service you talk to over the network, plus the pieces you couldn't easily build yourself: proper integrated filtering, hybrid search, and safe backups.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Run Qdrant, index real documents into it with metadata, run correctly-filtered searches at scale, and combine keyword and semantic search in one query.

---

## Stage 1 — Why a real vector database, not just `hnswlib`

You already have a working HNSW index from Part XV. So why not just keep using it?

Because `hnswlib` is a library — code that runs *inside* your Python process, holding everything in memory. It has no network interface, no built-in persistence beyond manual save/load, no filtering beyond what you build yourself, and no way for two separate services to share the same index without you building all of that infrastructure by hand.

A vector *database* like Qdrant is a separate, standalone service — you run it once, and any number of your applications talk to it over the network, the same way they'd talk to Postgres. It handles persistence, concurrent access from multiple clients, proper filtering, backups, and scaling, all as built-in features rather than things you'd have to engineer yourself.

The concepts transfer directly — you already know what HNSW is doing and why `ef` matters. This part is about the operational layer on top.

**Check yourself before Stage 2:** What's the practical difference between an in-process library like `hnswlib` and a standalone database service like Qdrant?

---

## Stage 2 — Get it running

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

The `-v` flag mounts a local folder for storage — without it, your data disappears when the container stops, which defeats the entire point of using a real database over an in-memory library.

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())
```

```
collections=[]
```

Connected, empty, ready.

**Check yourself before Stage 3:** Why does the `-v` flag matter, given what problem you're trying to solve by moving from `hnswlib` to a real database?

---

## Stage 3 — Collections: the container for your vectors

A **collection** is Qdrant's equivalent of a table — a named group of vectors that all share the same dimensionality and distance metric.

```python
from qdrant_client.models import VectorParams, Distance

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
```

`size=384` must match your embedding model's output dimensionality exactly — if you're using the same `all-MiniLM-L6-v2` model from earlier parts, that's 384. `Distance.COSINE` tells Qdrant which similarity metric to use — the same cosine similarity from [Part II](part02-ml-foundations.md), now computed by the database instead of by your own code.

**Check yourself before Stage 4:** What happens if you try to insert a 768-dimension vector into a collection configured with `size=384`?

---

## Stage 4 — Points: vectors with real metadata attached

A **point** is one stored item — a vector, an ID, and a **payload** (arbitrary metadata, like `NormalizedDoc.metadata` from [Part XII](part12-document-ingestion.md)).

```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=real_embed_fn("Employees get 15 days of paid leave per year.").tolist(),
            payload={"source": "leave_policy.md", "tenant_id": "acme", "section": "leave"},
        ),
        PointStruct(
            id=2,
            vector=real_embed_fn("The company was founded in 2015.").tolist(),
            payload={"source": "handbook.md", "tenant_id": "acme", "section": "history"},
        ),
    ],
)
```

"Upsert" means insert-or-update: if a point with that ID already exists, it's replaced; if not, it's created. This is exactly the pattern you want for the incremental re-indexing you built in [Part XII](part12-document-ingestion.md) — you can safely re-run ingestion on a changed document without worrying about duplicates.

**Check yourself before Stage 5:** Why does "upsert" semantics matter specifically for the incremental re-indexing workflow you built in Part XII?

---

## Stage 5 — Search, with a filter that actually scales

Here's where Part XV's hard-won lesson pays off directly. Recall the naive-filtering trap: fetch extra results, filter afterward, and watch it fail when the filter is selective. Qdrant filters *during* the search itself — the same "integrated into the graph traversal" idea Part XV described, implemented properly.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="documents",
    query_vector=real_embed_fn("How much leave do I get?").tolist(),
    query_filter=Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value="acme"))]
    ),
    limit=3,
)

for r in results:
    print(f"score={r.score:.3f}  {r.payload['source']}: {r.payload}")
```

```
score=0.847  leave_policy.md: {'source': 'leave_policy.md', 'tenant_id': 'acme', 'section': 'leave'}
```

Even if `tenant_id="acme"` matched only 0.1% of a million-point collection, this query returns correctly and quickly, because the filter is applied as part of the search's graph traversal, not as an afterthought on a fixed-size batch of candidates. This is the exact problem you deliberately broke in [Part XV Stage 6](part15-vector-databases.md) — now watch it be solved correctly.

### One thing that makes this actually fast: a payload index

For a filter Qdrant will check often, tell it to build an index on that field specifically — otherwise, on a very large collection, checking the filter condition itself can become the bottleneck:

```python
client.create_payload_index(
    collection_name="documents",
    field_name="tenant_id",
    field_schema="keyword",
)
```

**Check yourself before Stage 6:** Why does a highly selective filter (like `tenant_id="acme"` in a large multi-tenant collection) work correctly here, when the naive fetch-then-filter approach from Part XV would fail on the same filter?

---

## Stage 6 — Hybrid search: combining semantic and exact-match

Recall from [Part XIV Stage 2](part14-embeddings.md): dense embeddings catch meaning but can miss exact terms — a product code, an error string, a rare proper noun. Qdrant supports storing a sparse vector alongside the dense one, and combining the two rankings into a single result.

```python
from qdrant_client.models import SparseVectorParams, SparseVector, Prefetch, FusionQuery, Fusion

client.create_collection(
    collection_name="hybrid_docs",
    vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)

# In a real pipeline, the sparse vector comes from a keyword-scoring method
# (e.g. BM25 or SPLADE) — the indices/values below are illustrative.
client.upsert(
    collection_name="hybrid_docs",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense": real_embed_fn("Error code E4021 indicates a timeout.").tolist(),
                "sparse": SparseVector(indices=[204, 891], values=[1.0, 0.8]),
            },
            payload={"source": "error_codes.md"},
        )
    ],
)

results = client.query_points(
    collection_name="hybrid_docs",
    prefetch=[
        Prefetch(query=real_embed_fn("what does E4021 mean?").tolist(), using="dense", limit=20),
        Prefetch(query=SparseVector(indices=[204], values=[1.0]), using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
)
```

`Fusion.RRF` stands for Reciprocal Rank Fusion — instead of trying to directly compare a dense similarity score and a sparse keyword score (which aren't on the same scale and can't be meaningfully averaged), it combines the two *rankings*: a document that ranks well in both searches scores higher overall than one that only ranks well in one. This sidesteps the apples-to-oranges problem of comparing two different scoring systems directly. Full treatment of why and when hybrid search actually helps comes in [Part XVII](part17-advanced-rag.md) — for now, know the mechanism exists and roughly how it works.

**Check yourself before the project:** Why does Reciprocal Rank Fusion combine *rankings* rather than trying to average the raw dense and sparse scores directly?

---

# The Project: Migrate Your RAG Pipeline to Real Qdrant

## What you're building

Take the RAG system from [Part XI](part11-rag-fundamentals.md)'s project — which stored chunks in an in-memory Python list — and move it to real Qdrant, with metadata filtering and a payload index, benchmarked against your original in-memory version.

```bash
$ python qdrant_rag.py "What's the refund policy?" --tenant acme
```

```
searched 1,240 points (tenant-filtered), top score 0.84, 3.2ms
answer: Annual plans can be refunded within 30 days... [Source: pricing_policy.md]
```

## Why this project

This is the direct successor to Part XI's project and Part XV's benchmark — you're moving from "I understand how this works" to "I have it running as a real, persistent, filterable service." It's also the substrate every later multi-agent and production part assumes you have working.

## Build it in stages

---

### Stage A — Get Qdrant running and collection created

**Goal:** A real, empty `documents` collection, matching your embedding model's dimensionality.

**Do:**
1. Start Qdrant via Docker with a mounted volume
2. Create the collection with the correct `size` and `Distance.COSINE`
3. Confirm with `client.get_collections()` that it exists

**You're done when:** You can restart the Docker container and the (empty, for now) collection is still there — proof the volume mount is working.

---

### Stage B — Migrate your indexing pipeline

**Goal:** Your Part XII ingestion pipeline writes to Qdrant instead of an in-memory list.

**Do:**
1. Reuse your chunking and embedding code from Parts XI–XIV
2. For each chunk, `upsert` a point with its embedding and payload (source document, tenant if applicable, any other metadata from Part XII)
3. Index your real test document set

**You're done when:** `client.count(collection_name="documents")` reports the expected number of points, matching your chunk count.

---

### Stage C — Filtered search, with a payload index

**Goal:** Correctly filtered retrieval, fast, at scale.

**Do:**
1. Add a `tenant_id` (or similar) field to your payload if you don't already have one — even a fake multi-tenant split for testing purposes
2. Create a payload index on that field
3. Run a search filtered to one tenant, confirm you only get results from that tenant

**You're done when:** A filtered search returns correct, tenant-scoped results, and you can explain why this is reliable at scale in a way the naive Python-side filtering from Part XV wasn't.

---

### Stage D — Wire it into your RAG answer pipeline

**Goal:** Your full retrieve-then-generate flow from Part XI, now backed by Qdrant.

**Do:**
1. Replace your in-memory `retrieve()` call with a Qdrant `client.search()` call
2. Keep the relevance threshold from Part XI Stage 5 — check `r.score` against your threshold before generating
3. Confirm the full pipeline still produces correct, cited answers, and still correctly declines to answer when nothing relevant is found

**You're done when:** Your RAG pipeline behaves identically to Part XI's version in terms of answer quality, but is now backed by a real, persistent, filterable database.

---

### Stage E — Benchmark against your in-memory version

**Goal:** A real comparison, not an assumption that "Qdrant is obviously better."

**Do:**
1. At your current (probably small) document count, time a search against both your original Part XI in-memory version and the new Qdrant version
2. Note which is faster at this scale, and why (hint: think about network round-trip overhead vs. raw computation)
3. If you can, artificially inflate your corpus (duplicate documents with slight variations) to 50,000+ chunks and re-run the comparison

**You're done when:** You understand, from your own measurements, that Qdrant's advantage isn't necessarily "faster at tiny scale" — it's "correctly filterable, persistent, and scalable," which matters more as your data grows.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| `upsert` fails with a dimension mismatch error | Collection created with the wrong `size`, or mixing embedding models | Double-check your embedding model's actual output dimensionality against the collection config |
| Filtered search returns nothing, even though matching data exists | Payload field name mismatch between what was stored and what's filtered on | Print a raw payload from `client.scroll()` and confirm the exact field names |
| Data disappears after restarting Docker | No volume mounted, or mounted to the wrong path | Confirm the `-v` flag and that `qdrant_storage` actually appears on your local disk with files in it |
| Search is slower than expected on a filtered query | No payload index created on the filtered field | Add `create_payload_index` for any field you filter on regularly |

## Going further (optional)

1. **Snapshots.** Use `client.create_snapshot()` to back up your collection, then simulate a disaster by deleting the collection entirely and restoring from the snapshot. Confirm your data survives.
2. **Named vectors.** Store both a "title" embedding and a "body" embedding per document, and experiment with weighting or combining searches across both.
3. **Batch upsert timing.** Compare upserting 1,000 points one at a time versus in a single batched call. The difference is usually dramatic — this matters a great deal once you're indexing real-scale document sets.

---

## Checkpoint

**Knowledge:**
1. What's the practical difference between `hnswlib` (a library) and Qdrant (a database service)?
2. Why does Qdrant's filtered search avoid the naive fetch-then-filter problem from Part XV?
3. Why does hybrid search fuse rankings (RRF) rather than combining raw scores directly?

**Implementation:** Build a filtered, payload-indexed Qdrant collection with upsert and search, from memory.

**Debugging:** A filtered Qdrant query that used to be fast has become noticeably slow as your collection grew. What's the first thing you'd check?

**Architecture:** Design a Qdrant collection layout (single collection with a tenant filter, vs. one collection per tenant) for a SaaS product with a mix of a few huge customers and thousands of tiny ones. What are the trade-offs of each approach?

---

**Next:** [Part XVII — Advanced RAG](part17-advanced-rag.md), where you'll go beyond plain retrieve-then-generate — query rewriting, reranking, and retrieval that adapts to how hard the question actually is.
