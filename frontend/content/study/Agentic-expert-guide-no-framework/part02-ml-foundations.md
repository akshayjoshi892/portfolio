# Part II — The Math You Actually Need

## Before you start

**What you need already:** [Part I](part01-prerequisites.md). Basic Python. High school algebra. You do **not** need calculus, linear algebra coursework, or any prior ML.

**What this part gives you:** Four ideas that everything later is built on — vectors, similarity, nearest-neighbor search, and attention. This is not a machine learning course. It's the specific 5% of ML you need to understand why LLMs and search systems behave the way they do.

**Time:** 3–4 days.

**When you finish, you'll be able to:** Explain why a search system returns a wrong result, implement the search math yourself, and understand the mechanism inside an LLM well enough to reason about its failures.

**Install first:** `pip install numpy matplotlib`

---

## Stage 1 — A vector is just a list of numbers that describes something

Forget the word "vector" for a moment. Start with a spreadsheet.

Imagine describing fruit with three numbers: sweetness, size, redness — each from 0 to 10.

| Fruit | Sweetness | Size | Redness |
|---|---|---|---|
| Apple | 7 | 5 | 9 |
| Cherry | 8 | 1 | 10 |
| Lemon | 2 | 4 | 0 |

Each row is a **vector**. That's it. A vector is an ordered list of numbers describing one thing.

```python
apple  = [7, 5, 9]
cherry = [8, 1, 10]
lemon  = [2, 4, 0]
```

Each position in the list is a **dimension**. Our fruit have 3 dimensions. The one property that matters: **the same position always means the same thing.** Position 0 is always sweetness. You can't compare `[7, 5, 9]` to a list where position 0 means weight in grams — the numbers would be meaningless together.

### Why this matters for agents

Later, an **embedding model** will turn a sentence into a list of ~768 numbers. Nobody knows what each of those 768 positions means — the model learned them from data, and they don't map to words we have. But the principle holds exactly:

- Similar sentences get similar numbers
- The same position always means the same (unnamed) thing
- Two embeddings from *different models* are not comparable, for the same reason apple-vectors and weight-in-grams vectors aren't

That last point causes real production outages. You'll see it again in [Part XIV](part14-embeddings.md).

### Look at your vectors

You can genuinely see this with 2 dimensions. Drop redness:

```python
import matplotlib.pyplot as plt

fruits = {"apple": [7, 5], "cherry": [8, 1], "lemon": [2, 4], "grapefruit": [3, 6]}

for name, (sweet, size) in fruits.items():
    plt.scatter(sweet, size)
    plt.annotate(name, (sweet, size))

plt.xlabel("sweetness")
plt.ylabel("size")
plt.grid(True)
plt.show()
```

Apple and cherry sit near each other — both sweet. Lemon and grapefruit cluster elsewhere — both sour. **Similar things end up near each other.** That single sentence is the foundation of every search system in this curriculum.

With 768 dimensions you can't draw it, but the idea doesn't change.

**Check yourself before Stage 2:** In your own words, what is a vector? Why can't you compare vectors that came from different models?

---

## Stage 2 — Measuring how similar two things are

We can see that apple and cherry are close. Now we need a *number* for it, because a computer can't eyeball a graph.

### Attempt 1: straight-line distance

The distance between two points, from the Pythagorean theorem you learned in school:

```python
import math

def euclidean_distance(a, b):
    total = 0
    for x, y in zip(a, b):
        total += (x - y) ** 2
    return math.sqrt(total)

apple  = [7, 5, 9]
cherry = [8, 1, 10]
lemon  = [2, 4, 0]

print(euclidean_distance(apple, cherry))   # 4.24
print(euclidean_distance(apple, lemon))    # 10.34
```

Apple is closer to cherry (4.24) than to lemon (10.34). That matches intuition. Note this is a **distance** — smaller means more similar. That's backwards from a similarity score, which is worth keeping straight.

### Attempt 2: dot product

Another way — multiply matching positions and add up the results:

```python
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

print(dot_product(apple, cherry))   # 7*8 + 5*1 + 9*10 = 151
print(dot_product(apple, lemon))    # 7*2 + 5*4 + 9*0  = 34
```

Higher means more similar here. Apple-cherry (151) beats apple-lemon (34). Also correct.

But the dot product has a problem, and it's the important lesson of this stage.

**Check yourself before Stage 3:** For distance, does bigger mean more or less similar? For dot product?

---

## Stage 3 — The magnitude problem, and why we use cosine

Here's the trap. Add a giant fruit:

```python
watermelon = [7, 100, 3]     # medium sweet, enormous, barely red
apple      = [7, 5, 9]
cherry     = [8, 1, 10]

print(dot_product(apple, cherry))       # 151
print(dot_product(apple, watermelon))   # 7*7 + 5*100 + 9*3 = 576
```

The dot product says apple is **almost 4× more similar to watermelon than to cherry.** That's nonsense. Apples and cherries are both small sweet red fruits. Apples and watermelons have almost nothing in common.

### What went wrong

The watermelon's size number (100) is enormous. The dot product multiplies raw numbers, so any vector with big numbers scores high against *everything*. Size overwhelmed the actual comparison.

This has a name: the dot product is sensitive to **magnitude** (how big the numbers are), not just **direction** (the pattern or shape of the numbers).

For "how similar are these things," we almost always want direction. A tiny sweet red fruit and a medium sweet red fruit have the same *pattern* — sweet, red, not very big relative to each other. That's what we care about.

### The fix: cosine similarity

Divide out the size, leaving only the pattern. You do that by dividing each vector by its own length:

```python
def magnitude(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(a, b):
    return dot_product(a, b) / (magnitude(a) * magnitude(b) + 1e-12)
```

Now:

```python
print(cosine_similarity(apple, cherry))       # 0.90
print(cosine_similarity(apple, watermelon))   # 0.47
```

Apple–cherry 0.90, apple–watermelon 0.47. Correct at last.

### Reading a cosine score

Cosine similarity always lands between -1 and 1:

| Score | Meaning |
|---|---|
| 1.0 | Identical direction — as similar as possible |
| ~0.8–0.9 | Very similar |
| 0.0 | Unrelated |
| -1.0 | Exact opposites |

For text embeddings you'll mostly see 0.0 to 1.0, because embedding models rarely produce true opposites.

### What's that `1e-12` for?

If a vector is all zeros, its magnitude is 0, and you'd divide by zero and crash. Adding a tiny number (0.000000000001) prevents the crash without meaningfully changing any real result. You'll see this trick throughout the curriculum — it's standard practice.

### The shortcut that appears everywhere

If you divide a vector by its own length ahead of time, its length becomes exactly 1. That's called **normalizing**:

```python
def normalize(v):
    m = magnitude(v)
    return [x / m for x in v]
```

Here's why it matters: once both vectors are normalized, **cosine similarity and dot product give identical answers.**

```python
a_norm = normalize(apple)
b_norm = normalize(cherry)

print(cosine_similarity(apple, cherry))   # 0.90
print(dot_product(a_norm, b_norm))        # 0.90 — same!
```

Dot product is cheaper to compute (no square roots). So vector databases normalize everything once at storage time, then use the fast dot product forever after. When you read "this index uses dot product" in [Part XV](part15-vector-databases.md), this is why — and it's *only* correct because the vectors were normalized first.

> **The rule:** Use cosine similarity for text. Use dot product **only** on normalized vectors. Using dot product on unnormalized vectors silently biases results toward whatever has the biggest numbers — usually long documents.

**Try breaking it:** Build a vector `[50, 50, 50]` and compare it to everything with dot product. Watch it win every comparison despite describing nothing in particular. Then compare with cosine — sanity returns.

**Check yourself before Stage 4:** Why did the dot product prefer watermelon? What exactly does normalizing do? When are cosine and dot product the same?

---

## Stage 4 — Finding the closest matches

You now have a similarity score for two vectors. Search needs one more step: given a query and 10,000 stored vectors, find the 5 closest.

The simple approach: compare against every single one, sort, take the top 5.

```python
def find_top_k(query, corpus, k=3):
    scored = []
    for i, vector in enumerate(corpus):
        score = cosine_similarity(query, vector)
        scored.append((i, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


fruits = {
    "apple": [7, 5, 9], "cherry": [8, 1, 10], "lemon": [2, 4, 0],
    "grapefruit": [3, 6, 4], "strawberry": [8, 2, 10], "banana": [6, 6, 1],
}

names = list(fruits.keys())
corpus = list(fruits.values())

query = [8, 2, 9]        # "small, sweet, red"
for index, score in find_top_k(query, corpus, k=3):
    print(f"{names[index]:12s} {score:.3f}")
```

Output:

```
strawberry   0.998
cherry       0.995
apple        0.965
```

That's a working search engine. Swap fruit-vectors for sentence-embeddings and you have the retrieval half of every RAG system in this curriculum.

This is called **k-nearest neighbors** (k-NN) — find the k closest items. "Brute-force k-NN" means doing it this way: compare against literally everything.

**Check yourself before Stage 5:** Describe the three steps of brute-force k-NN without looking.

---

## Stage 5 — Watching brute force break

Brute force works fine for 6 fruits. Let's find out where it stops working. Don't take my word for it — measure it.

First, switch to NumPy. Pure Python loops are too slow to even test this at scale. NumPy does the same math on whole arrays at once, in optimized C code underneath:

```python
import numpy as np

def cosine_similarity_batch(query, corpus):
    """Compare one query against every row of corpus at once."""
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    return corpus_norm @ query_norm      # @ is matrix multiply


def find_top_k_fast(query, corpus, k=5):
    scores = cosine_similarity_batch(query, corpus)
    top_indices = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top_indices]
```

Three things to know about that code:

- `np.linalg.norm(v)` is magnitude — the same square-root-of-sum-of-squares from Stage 3
- `axis=1` means "compute it for each row separately"
- `@` is matrix multiplication, which here does all the dot products in one operation

Now measure how it scales:

```python
import time

rng = np.random.default_rng(0)
query = rng.normal(size=(768,)).astype(np.float32)

for n in [1_000, 10_000, 100_000, 1_000_000]:
    corpus = rng.normal(size=(n, 768)).astype(np.float32)

    start = time.monotonic()
    find_top_k_fast(query, corpus, k=5)
    elapsed = time.monotonic() - start

    print(f"{n:>9,} vectors: {elapsed*1000:>8.1f} ms")
```

Typical output (your numbers will differ, the *shape* won't):

```
    1,000 vectors:      1.8 ms
   10,000 vectors:     14.2 ms
  100,000 vectors:    138.6 ms
1,000,000 vectors:   1420.3 ms
```

### Read that table carefully

10× more vectors = 10× more time. Every single time. It's a straight line.

That's **linear scaling**, written O(n). It's predictable, which is nice, but it means:

- 1 million vectors → 1.4 seconds per search. Too slow for a chat interface.
- 10 million → 14 seconds. Unusable.
- 100 million → 140 seconds. Absurd.

And that's *one* search. Ten users at once makes it ten times worse.

### The way out

You cannot make comparing-a-million-things fast. So you stop comparing everything.

**Approximate nearest neighbor** (ANN) search organizes vectors in advance so a query only examines a few thousand candidates instead of all million. The trade: you might occasionally miss the true best match. In exchange, searches take milliseconds instead of seconds.

That's the entire reason vector databases exist. [Part XV](part15-vector-databases.md) covers how they do it. But now you know exactly what problem they're solving, because you measured it yourself.

**Check yourself before Stage 6:** What does O(n) mean here? At roughly what corpus size does brute force become unworkable for a chat app? What does ANN give up, and what does it gain?

---

## Stage 6 — What a neural network actually does

Switching topics. We've covered search. Now: what's inside an LLM?

Strip away the mystique and a neural network does one thing — **it turns a list of numbers into a different list of numbers**, repeatedly.

```
[list of numbers] → layer → [different list] → layer → [different list] → answer
```

Each layer does two steps.

### Step 1: mix the numbers together

Every output number is a weighted sum of every input number:

```python
import numpy as np

def linear_layer(inputs, weights, bias):
    return inputs @ weights + bias

inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([[0.5, -0.2],
                    [0.1,  0.8],
                    [-0.3, 0.4]])
bias = np.array([0.1, -0.1])

print(linear_layer(inputs, weights, bias))   # [-0.1  2.5]
```

The **weights** are the numbers that get learned during training. At the start they're random and the network outputs garbage. Training nudges them, over billions of examples, until the outputs are useful. An LLM with "70 billion parameters" has 70 billion of these weight numbers.

### Step 2: bend the result

```python
def relu(x):
    return np.maximum(0, x)      # negatives become 0, positives pass through
```

This is an **activation function**. It looks trivially simple, and it's essential.

Here's why: if you stack two mixing steps with nothing between them, the math collapses — two weighted sums in a row is just one weighted sum with different weights. A hundred layers would be no more capable than one. Adding a bend between layers breaks that collapse, so stacking actually buys you something.

### Training, briefly

1. Feed in an example. Get an output.
2. Compare to the right answer. The gap is the **loss** (bigger = worse).
3. Work out which weights contributed to the error and in which direction. This is **backpropagation** — the chain rule from calculus, applied backwards through the layers.
4. Nudge every weight slightly in the improving direction. This is **gradient descent**.
5. Repeat billions of times.

You will not train an LLM in this curriculum. You need this only to understand two things: the model's "knowledge" is frozen weights from training (which is why it doesn't know today's news — and why [Part XI](part11-rag-fundamentals.md)'s RAG exists), and "learning" means adjusting numbers to reduce error, nothing more mystical.

### One idea to carry with you

**Overfitting** is when a model memorizes its training examples instead of learning the general pattern. It scores perfectly on what it's seen and fails on anything new.

This isn't just a training concept — it happens to *you*. You'll tweak a prompt until it works on your 3 test questions, ship it, and watch it fail on real users. You overfit to your test cases. The defense is the same as in ML: keep a set of examples you *don't* tune against. [Part V](part05-prompt-engineering.md) builds exactly that.

**Check yourself before Stage 7:** What does one layer do? Why do you need activation functions between layers? What is overfitting, and how does it bite prompt engineers?

---

## Stage 7 — What attention adds

One more piece, and it's the one that made LLMs possible.

### The problem attention solves

Take this sentence:

> "The trophy didn't fit in the suitcase because **it** was too big."

What does "it" mean? The trophy. Now change one word:

> "The trophy didn't fit in the suitcase because **it** was too small."

Now "it" is the suitcase. Same sentence structure, opposite answer — you have to understand the whole sentence to resolve one word.

Older models (RNNs) read left to right, carrying a running summary. By the time they reached "it," early words had faded. Long-range connections were weak, and reading strictly in order meant no parallelism, so training was slow.

### The idea

Let every word look directly at every other word, and learn how much attention to pay to each.

When processing "it," the model can look at all of: trophy, suitcase, fit, big — and weight them. If "big" is present, weight "trophy" heavily. If "small," weight "suitcase."

No fading over distance. And every word can do this at the same time, which is what makes training on internet-scale data possible.

### How it works mechanically

Each word produces three vectors:

| Name | Role | Plain-language version |
|---|---|---|
| **Query** | What I'm looking for | "I'm a pronoun, I need a noun I refer to" |
| **Key** | What I offer | "I'm a noun, a physical object" |
| **Value** | What I actually contribute | The meaning to pass along if selected |

The procedure:

1. Take one word's **query**
2. Compare it against every word's **key** — with a dot product, the same one from Stage 2
3. Convert those scores to percentages that sum to 100% (a **softmax**)
4. Take a weighted blend of everyone's **values**, using those percentages

Step 2 is the connection back to Stage 2. Attention is *built on* the dot-product similarity you already know. "How relevant is this word to what I need" is the same operation as "how similar are these two fruits."

### Multi-head attention

Doing this once captures one kind of relationship. Real language has several at once — grammar, meaning, references. So models run several attention computations side by side with different learned weights, then combine the results. Each one is a **head**. "12-head attention" means 12 parallel versions, each free to specialize.

**Check yourself before Stage 8:** What problem does attention solve that reading left-to-right couldn't? What are query, key, and value? What operation compares a query to a key?

---

## Stage 8 — Build attention yourself

Reading about attention isn't the same as building it. It's about 10 lines. Do it.

```python
import numpy as np


def softmax(x, axis=-1):
    """Turn any list of numbers into percentages that sum to 1."""
    x = x - np.max(x, axis=axis, keepdims=True)   # for numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def self_attention(x, Wq, Wk, Wv):
    """
    x:  (seq_len, d_model) — one row per word
    Wq, Wk, Wv: learned weight matrices
    """
    Q = x @ Wq          # step 1: what each word is looking for
    K = x @ Wk          # step 1: what each word offers
    V = x @ Wv          # step 1: what each word contributes

    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)    # step 2: compare every query to every key

    weights = softmax(scores, axis=-1)   # step 3: convert to percentages

    return weights @ V, weights          # step 4: weighted blend of values


rng = np.random.default_rng(42)
seq_len, d_model = 3, 8                  # 3 words, 8 numbers each

x = rng.normal(size=(seq_len, d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d_model)) for _ in range(3))

output, weights = self_attention(x, Wq, Wk, Wv)

print("attention weights (row = word, column = how much it looks at that word):")
print(np.round(weights, 3))
print("\neach row sums to:", np.round(weights.sum(axis=1), 6))
print("\noutput shape:", output.shape)
```

Output:

```
attention weights (row = word, column = how much it looks at that word):
[[0.012 0.979 0.009]
 [0.    1.    0.   ]
 [0.897 0.023 0.08 ]]

each row sums to: [1. 1. 1.]

output shape: (3, 8)
```

### Verify it yourself

Look at row 0: `[0.012, 0.979, 0.009]`. Word 0 is paying 97.9% of its attention to word 1. Every row sums to exactly 1.0 — it's a proper set of percentages.

Prove to yourself that the output really is a weighted blend:

```python
manual = weights[0, 0] * V[0] + weights[0, 1] * V[1] + weights[0, 2] * V[2]
print(np.allclose(manual, output[0]))   # True
```

(You'll need to return `V` from the function, or recompute `V = x @ Wv`, to run that.)

That's it. That's attention. Real LLMs stack ~100 of these, with multiple heads each and some extra pieces — but the core is what you just wrote.

### Why divide by √d_k?

Without it, the dot products get large when vectors have many dimensions. Large numbers going into softmax produce percentages like `[0.9999, 0.0001]` — the model attends to exactly one word and ignores everything else, and training breaks. Dividing keeps the scores in a workable range.

**Check yourself:** Explain each of the four lines in `self_attention`. Why must each row of weights sum to 1?

---

# The Project: Build a Working Search Engine

## What you're building

A search tool over real text. Type a question, get back the most relevant passages, ranked.

```bash
$ python search.py "how do I reset my password"
```

```
1. [0.847] account_faq.txt:12
   To reset your password, click "Forgot password" on the login screen...

2. [0.792] troubleshooting.txt:45
   Password resets are sent to your registered email address...

3. [0.681] account_faq.txt:30
   Your username cannot be changed after account creation...

Searched 1,240 passages in 8ms
```

## Why this project

This is [Part XI](part11-rag-fundamentals.md)'s retrieval system in miniature. You build it now, with the math visible, so that when a framework does it for you later, you know exactly what's happening underneath — and can debug it when results are wrong.

## Before you start

- `pip install numpy sentence-transformers`
- A folder with 5–10 plain text files. Your own notes, documentation, book chapters — anything with real content. Don't use random generated text; you need to be able to tell whether the results are actually good.

## Build it in stages

---

### Stage A — Search with fake numbers

**Goal:** The search loop works end to end, before touching any ML model.

**Do:**
1. Read all your `.txt` files. Split each into paragraphs (split on blank lines).
2. Write `fake_embed(text)` that turns text into a vector *deterministically* — same text in, same vector out:
   ```python
   def fake_embed(text, dim=32):
       rng = np.random.default_rng(abs(hash(text)) % (2**32))
       v = rng.normal(size=(dim,)).astype(np.float32)
       return v / (np.linalg.norm(v) + 1e-12)
   ```
3. Embed every paragraph. Store `(filename, line_number, text, vector)`.
4. Embed the query the same way, run `find_top_k_fast`, print the top 3.

**You're done when:** It runs and prints 3 results with scores.

**Important:** The results will be *garbage*. That's correct and expected. Fake embeddings are random — they have no idea what words mean. You're testing the plumbing, not the quality. Searching for the exact text of a paragraph should return that paragraph first (same input → same vector → similarity 1.0). Verify that. If it doesn't, your plumbing is broken.

---

### Stage B — Real embeddings

**Goal:** Results that are actually good.

**Do:**
1. Replace `fake_embed` with a real model:
   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("all-MiniLM-L6-v2")

   def embed(texts: list[str]) -> np.ndarray:
       return model.encode(texts, normalize_embeddings=True)
   ```
2. Embed all paragraphs **in one batch call**, not one at a time. Batching is dramatically faster.
3. Search again.

**You're done when:** You search for a concept using *different words than the document uses* — and still get the right paragraph. That's the whole point. If your doc says "authentication failure" and you search "can't log in," a good embedding model finds it. A keyword search never would.

**Watch out for:** First run downloads ~90MB. `normalize_embeddings=True` means everything comes back pre-normalized, so you can use the fast dot product (Stage 3).

---

### Stage C — Make it usable

**Goal:** A real command-line tool.

**Do:**
1. Take the query from `sys.argv` instead of hardcoding it
2. Show filename and line number with each result
3. Truncate long paragraphs to ~200 characters in the display
4. Time the search and print it
5. Cache embeddings to disk (`np.save` / `np.load`) so restarts are instant

**You're done when:** You can run it repeatedly from the terminal, and it starts in under a second on the second run.

**Watch out for:** Your cache must be invalidated when the source files change. Simplest approach: store a hash of the file contents alongside the vectors, re-embed if it differs. This is the same problem [Part XII](part12-document-ingestion.md) solves properly.

---

### Stage D — Measure how good it is

**Goal:** Stop guessing whether it works. Measure it.

This stage is the one people skip. Don't — it's the most valuable part.

**Do:**
1. Write 10 test questions and, for each, note which paragraph *should* be the top result. Store as `(question, correct_paragraph_id)`.
2. Run all 10 questions. For each, record what rank the correct answer came back at (1st? 3rd? not in the top 10 at all?).
3. Compute two numbers:
   - **Recall@3** — fraction of questions where the right answer was in the top 3
   - **MRR** (Mean Reciprocal Rank) — average of `1/rank`. Correct answer at rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25, not found scores 0.

```python
def evaluate(test_cases, search_fn, k=3):
    recall_hits, reciprocal_ranks = 0, []

    for question, correct_id in test_cases:
        results = search_fn(question, k=10)
        result_ids = [r.paragraph_id for r in results]

        if correct_id in result_ids[:k]:
            recall_hits += 1

        if correct_id in result_ids:
            reciprocal_ranks.append(1 / (result_ids.index(correct_id) + 1))
        else:
            reciprocal_ranks.append(0.0)

    n = len(test_cases)
    return {
        f"recall@{k}": recall_hits / n,
        "mrr": sum(reciprocal_ranks) / n,
    }
```

**You're done when:** You have real numbers. Something like `recall@3: 0.80, mrr: 0.71`.

**Now use them.** Change one thing — split into sentences instead of paragraphs — and re-measure. Did it improve? Try a different embedding model. Re-measure. You now have a way to make decisions from evidence instead of vibes. That habit is what separates people who build working retrieval systems from people who guess.

---

### Stage E — Find where it breaks

**Goal:** Feel the scaling wall yourself.

**Do:**
1. Duplicate your paragraphs until you have 1,000, then 10,000, then 100,000 (append a counter so texts differ)
2. Time the search at each size
3. Plot or tabulate it

**You're done when:** You can state, from your own measurements, the corpus size at which your search exceeds 100ms.

**Then think about this:** That number is where you'd need a vector database. Not because someone said so — because you measured it. Keep the number. You'll compare against it in [Part XV](part15-vector-databases.md).

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Every result has a very similar score | Using fake embeddings; or comparing unnormalized vectors | Use real embeddings; normalize |
| Long paragraphs always win | Dot product on unnormalized vectors (Stage 3's trap) | Normalize, or use cosine |
| Exact-text search doesn't return that text first | Query and documents embedded differently | Same model, same settings, both sides |
| Very slow to start every time | Re-embedding on each run | Cache to disk |
| Good on some queries, terrible on others | Paragraphs too long or too short | Try different splitting; measure with Stage D |

## Going further (optional)

1. **Compare two embedding models.** Run Stage D's evaluation with `all-MiniLM-L6-v2` and a larger model. Is the bigger one better *on your data*? Often it is. Sometimes it isn't. Now you can tell.
2. **Compare chunk sizes.** Sentences vs. paragraphs vs. fixed 500-character windows. Measure each. This previews [Part XIII](part13-chunking.md), where chunking gets serious treatment — you'll arrive already knowing it matters.
3. **Add keyword search and combine.** Implement simple word-overlap scoring, then merge the two rankings. You'll find each catches things the other misses. That observation is the entire basis of hybrid search in [Part XVII](part17-advanced-rag.md).

---

## Checkpoint — do not skip

**Knowledge:**
1. What is a vector? Why can't vectors from two different models be compared?
2. Why does dot product favor big vectors, and how does cosine fix it?
3. When do cosine and dot product give identical results?
4. What does O(n) mean for brute-force search, and why does it eventually break?
5. What does an activation function do, and why does a network need one between layers?
6. What problem does attention solve? What are query, key, and value?

**Implementation:** On a blank page, write (a) cosine similarity, (b) top-k search, (c) single-head self-attention.

**Debugging:** Your search returns long documents for every query, regardless of the question. Which of these three is most likely: the embedding model, the similarity metric, or the chunking? How would you test your hypothesis in under five minutes?

**Architecture:** You have 50 million documents and need sub-100ms search. Brute force gives you 70 seconds. List three things you could change, and what each one costs you.

---

**Next:** [Part III — LLM Fundamentals](part03-llm-fundamentals.md), where the attention you just built gets stacked into an actual language model.
