# Part III — What's Actually Inside an LLM

## Before you start

**What you need already:** [Part I](part01-prerequisites.md) (async Python) and [Part II](part02-ml-foundations.md) (vectors, cosine similarity, attention). Stage 8 of Part II especially — this part stacks that attention into a real model.

**What this part gives you:** A mechanical understanding of what happens between you pressing "send" and text appearing. Not a research-level understanding — a *debugging-level* one. When an LLM does something strange, you'll know which part of the pipeline to suspect.

**Time:** 4–5 days.

**When you finish, you'll be able to:** Explain why an answer got cut off mid-sentence, why the first word takes 2 seconds and the rest stream fast, why the same prompt gives different answers, and why the model is confident about things that aren't true.

**Install first:** `pip install tiktoken httpx`

---

## Stage 1 — Text becomes numbers

A neural network can only do arithmetic. It can't process the letter "a." So the first step of everything is turning text into numbers.

The process is called **tokenization**, and it does not split on words or letters. It splits on chunks that show up frequently in text — a chunk might be a whole word, part of a word, a space, or a punctuation mark. Each chunk is a **token**, and each token has an ID number.

Run this:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "The agent retrieved three documents."
token_ids = enc.encode(text)

print(f"text:      {text}")
print(f"characters: {len(text)}")
print(f"tokens:     {len(token_ids)}")
print(f"token ids:  {token_ids}")
print()

for tid in token_ids:
    print(f"  {tid:>6}  ->  {enc.decode([tid])!r}")
```

Output:

```
text:      The agent retrieved three documents.
characters: 36
tokens:     6

  791  ->  'The'
 8479  ->  ' agent'
27231  ->  ' retrieved'
 2380  ->  ' three'
9477  ->  ' documents'
   13  ->  '.'
```

Look closely at that output. Two things worth noticing:

**The space belongs to the token.** `' agent'` includes the leading space. That's why the model handles spacing naturally — it's baked in, not a separate thing to track.

**One token ≈ one common word, here.** But that's only true for ordinary English.

### Where it gets uneven

```python
samples = {
    "english":  "The cat sat on the mat.",
    "code":     "def calculate(x: int) -> int:\n    return x * 2",
    "numbers":  "3.14159265358979",
    "rare word": "antidisestablishmentarianism",
}

for name, s in samples.items():
    n_tokens = len(enc.encode(s))
    print(f"{name:>10}: {len(s):>3} chars -> {n_tokens:>3} tokens "
          f"({len(s)/n_tokens:.1f} chars per token)")
```

```
   english:  23 chars ->   7 tokens (3.3 chars per token)
      code:  45 chars ->  18 tokens (2.5 chars per token)
   numbers:  16 chars ->  11 tokens (1.5 chars per token)
 rare word:  28 chars ->   7 tokens (4.0 chars per token)
```

Code costs more tokens per character than prose. Numbers cost a lot — they get chopped into small pieces.

### Why you should care about any of this

**You pay per token.** Not per word, not per character. A 1,000-word English document and a 1,000-word code file cost noticeably different amounts.

**Context limits are in tokens.** "128,000 token context" — you need to count tokens, not characters, to know if your input fits.

**It explains some famous LLM failures.** Ask a model how many r's are in "strawberry." It often gets it wrong. Look at why:

```python
for tid in enc.encode("strawberry"):
    print(repr(enc.decode([tid])))
```

```
'str'
'aw'
'berry'
```

The model never sees individual letters. It sees three chunks. Counting letters inside a chunk is genuinely hard for it — like being asked how many times the letter "e" appears in a word you only ever heard spoken. This isn't stupidity; it's a direct consequence of how text enters the model.

### The practical habit

Count tokens before you send:

```python
def count_tokens(text: str, model_encoding: str = "cl100k_base") -> int:
    return len(tiktoken.get_encoding(model_encoding).encode(text))
```

You'll use this for cost estimates ([Part XL](part40-cost-optimization.md)) and context budgeting ([Part XXXII](part32-context-engineering.md)).

> **Note:** Different model families use different tokenizers. `tiktoken` covers OpenAI models. Other providers tokenize differently, so token counts differ between them. Use the counter that matches your model, or treat counts as estimates.

**Check yourself before Stage 2:** What is a token? Why does code cost more tokens per character than English? Why do models struggle to count letters in a word?

---

## Stage 2 — Numbers become meaning, and gain a position

Token IDs are arbitrary labels. Token 8479 isn't "bigger" or "more" than token 2380 — the numbers carry no meaning. So the model immediately converts each ID into a vector, using a giant lookup table learned during training.

```
token id 8479  →  [0.23, -0.71, 0.44, ... ]   (e.g. 4096 numbers)
```

This is the **embedding** — the same concept as Part II's fruit vectors, except with thousands of dimensions whose meanings nobody labelled. Similar words end up with similar vectors, because training put them there.

So after this step, "The agent retrieved" is a stack of three vectors.

### The missing piece: order

Here's a problem worth pausing on. Look back at the attention code you wrote in [Part II Stage 8](part02-ml-foundations.md). Ask yourself: if you shuffled the rows of `x`, would the math care?

It wouldn't. Attention compares every word to every other word. It has no notion of "before" or "after." To attention, these are identical:

> "The dog bit the man"
> "The man bit the dog"

That's obviously wrong. So position information has to be *added in explicitly*.

Modern models mostly use **RoPE** (Rotary Position Embedding). The mechanism: rotate each word's query and key vectors by an angle proportional to its position. The elegant consequence is that when you then dot-product a query against a key, the result naturally depends on *how far apart* the two words are — relative position falls out of the math for free.

You don't need to implement RoPE. You need one consequence of it:

> **Models work best at sequence lengths near what they were trained on.** Push far past that and quality degrades, even when the model technically accepts the input. This is one reason "just paste everything into the huge context window" is a worse strategy than it sounds — and part of why [Part XI](part11-rag-fundamentals.md)'s retrieval exists.

**Check yourself before Stage 3:** Why does attention need position information added separately? What happens if you don't add it?

---

## Stage 3 — Stacking attention into a model

You built one attention layer in Part II. A real model stacks that same piece, over and over.

Here's one **transformer block**:

```
input vectors
     │
  normalize          ← keep numbers in a stable range
     │
  multi-head attention   ← the piece you built in Part II
     │
  add the input back     ← the "residual connection"
     │
  normalize
     │
  feed-forward network   ← the mixing layer from Part II Stage 6
     │
  add the input back
     │
output vectors  (same shape as input — so you can stack another one)
```

The input and output have the same shape. That's deliberate: it means you can stack these blocks indefinitely. Large models stack 32, 80, or more.

### The one part that deserves explanation

"Add the input back" — the **residual connection** — looks pointless. It's not. It's essential.

Without it, information has to survive passing through 80 layers of transformation. It gets washed out, and during training the correction signal fades to nothing before reaching the early layers (the "vanishing gradient" problem). Adding the input back gives information a clear path straight through the whole stack. Each layer then only has to learn a small *adjustment*, rather than reconstructing everything.

Stacking this deep network without residual connections simply does not train. It's one of the few tricks that separates "works" from "doesn't."

### Causal masking: no peeking ahead

One more thing. When generating text, a word must not see words that come after it — otherwise the model could cheat during training by looking at the answer.

The fix is mechanically simple: before the softmax, set all "future" scores to negative infinity. Softmax turns negative infinity into zero. Those positions get exactly 0% attention.

```python
import numpy as np

def causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores_adjustment = np.zeros((seq_len, seq_len))
    scores_adjustment[mask] = -np.inf
    return scores_adjustment

print(causal_mask(4))
```

```
[[  0. -inf -inf -inf]
 [  0.   0. -inf -inf]
 [  0.   0.   0. -inf]
 [  0.   0.   0.   0.]]
```

Add this to your attention scores before softmax (in the code from [Part II Stage 8](part02-ml-foundations.md)) and word 0 can only see itself, word 1 sees words 0–1, and so on. Try it — the weight matrix becomes lower-triangular, each row still summing to 1.

This is why text generates left to right. It's not a design choice layered on top; it's baked into the mask.

**Check yourself before Stage 4:** What does a residual connection do and why is it necessary? What does causal masking prevent?

---

## Stage 4 — It only predicts the next token

Here's the part that surprises people.

After all those layers, the model produces one thing: **a score for every token in its vocabulary**, representing how likely each is to come next. Those raw scores are called **logits**. Softmax turns them into probabilities.

```
"The capital of France is"
        │
    [the model]
        │
    " Paris"   62%
    " located"  8%
    " a"        5%
    " the"      3%
    ... (~100,000 more, each tiny)
```

Pick one. Append it. Feed the whole thing back in. Predict again. Repeat until you hit a stop token or a length limit.

**That is the entire generation process.** Writing an essay, debugging code, doing multi-step reasoning — all of it is this loop, running one token at a time.

### Why this matters more than it seems

Nearly every LLM quirk follows from this one fact.

**Hallucination isn't a bug.** The model picks likely-looking next tokens. "Likely-looking" and "true" are different things. There's no fact-checking step anywhere in the loop — nothing in the architecture even represents the idea of truth. A confident false answer is the system working exactly as built. This is precisely why [Part XI](part11-rag-fundamentals.md) exists: put the facts in the input, so likely-looking and true line up.

**The model can't plan ahead.** It commits to each token before considering the next. It can't write the end of a sentence, realize it's wrong, and rewrite — not within one pass. That limitation is why chain-of-thought prompting helps ([Part V](part05-prompt-engineering.md)): giving it space to write intermediate steps lets later tokens condition on earlier reasoning, which is the closest thing to "thinking before answering" the architecture allows.

**Output gets slower the longer it is.** Each token requires a full pass through the model. 500 tokens = 500 sequential passes. You can't parallelize this — token 12 needs token 11 to exist first. Stage 7 measures this.

**Check yourself before Stage 5:** What does the model actually output at each step? Why is hallucination an expected consequence of this design rather than a defect?

---

## Stage 5 — The knobs: temperature and top-p

The model gives you probabilities. How you pick from them is up to you, and it's controllable.

### Temperature

Temperature divides the logits before softmax. That sounds obscure, so watch what it does:

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

logits = np.array([5.0, 3.0, 1.0, 0.5])    # 4 candidate tokens
labels = ["Paris", "located", "a", "the"]

for temp in [0.1, 0.7, 1.0, 2.0]:
    probs = softmax(logits / temp)
    shown = "  ".join(f"{l}:{p:5.1%}" for l, p in zip(labels, probs))
    print(f"temp={temp:>4}:  {shown}")
```

```
temp= 0.1:  Paris:100.0%  located: 0.0%  a: 0.0%  the: 0.0%
temp= 0.7:  Paris:91.0%   located: 5.5%  a: 0.3%  the: 0.2%
temp= 1.0:  Paris:83.6%   located:11.3%  a: 1.5%  the: 0.9%
temp= 2.0:  Paris:57.0%   located:21.0%  a: 7.7%  the: 6.0%
```

- **Low temperature** sharpens the distribution. The top choice dominates. Output becomes predictable and repetitive.
- **High temperature** flattens it. Unlikely tokens get a real chance. Output becomes varied — and, past a point, incoherent.

### Top-p (nucleus sampling)

Temperature reshapes probabilities but keeps every token possible — including absurd ones. Top-p cuts the tail off entirely.

Top-p of 0.9 means: sort tokens by probability, keep adding until you've accumulated 90%, discard the rest, sample from what's left. Weird tokens get excluded outright.

### What to actually use

| Situation | Setting | Why |
|---|---|---|
| Tool calling, structured output, classification | `temperature=0` | You want one right answer, consistently. Creativity is a defect here. |
| Factual Q&A, summarization | `0` to `0.3` | Mild variation, still grounded |
| Brainstorming, creative writing | `0.7` to `1.0` | Variety is the point |
| Above 1.2 | Rarely | Usually incoherent |

**Default to 0 for anything in an agent.** Agent behavior needs to be reproducible so you can debug it. You'll spend a lot of this curriculum making agents predictable; don't add randomness where it buys you nothing.

### An honest footnote

Even at `temperature=0`, providers don't always give byte-identical output. Their servers batch requests together, and floating-point arithmetic results can shift slightly depending on how requests get grouped. Don't build anything that *requires* exact reproducibility from a hosted API — verify it rather than assuming it.

**Try it:** Call a real API with the same prompt 5 times at `temperature=0`, then 5 times at `1.0`. Compare. This takes two minutes and makes the concept permanent.

**Check yourself before Stage 6:** What does temperature do to the probability distribution? Why use 0 for tool calling?

---

## Stage 6 — The context window, and what happens when you exceed it

The **context window** is the maximum number of tokens the model can handle in one request — your input *and* its output combined.

```
[system prompt] + [conversation] + [documents] + [model's reply] ≤ context window
```

### Two things go wrong, and they're different

**1. You exceed it.** The provider truncates or errors. If it truncates silently, you get half a JSON object or an answer that stops mid-sentence. Always check `finish_reason` in the response — if it says `"length"`, your output got cut off and you should treat that as a failure, not a result.

**2. You fill it and quality drops anyway.** This is the subtler problem. Models retrieve information from the beginning and end of a long context more reliably than from the middle. It's known as **"lost in the middle."** A fact buried in the middle of 100,000 tokens can be effectively invisible, even though it's technically "in context."

### The consequence for how you build things

There's a tempting idea: context windows are huge now, so just paste in everything and let the model figure it out.

It doesn't work well, for three reasons:

| Problem | Effect |
|---|---|
| Cost | You pay for every input token, every request |
| Latency | More input means slower processing (Stage 7 shows why) |
| Quality | Irrelevant content dilutes attention; middle content gets missed |

So instead you select what goes in. That's retrieval ([Part XI](part11-rag-fundamentals.md)) and context engineering ([Part XXXII](part32-context-engineering.md)) — two large topics that both exist because of this stage.

### A budget you can reason about

```python
def check_context_budget(system_prompt, history, documents, max_output, limit):
    used = (count_tokens(system_prompt)
            + sum(count_tokens(m) for m in history)
            + sum(count_tokens(d) for d in documents))
    available = limit - used - max_output

    print(f"used: {used:,}  reserved for output: {max_output:,}  "
          f"headroom: {available:,}")
    if available < 0:
        raise ValueError(f"over budget by {-available:,} tokens")
    return available
```

Notice it subtracts `max_output` too. Forgetting that the reply shares the window is a very common bug — your input fits, generation starts, and then gets truncated.

**Check yourself before Stage 7:** What two different problems does a full context window cause? What does `finish_reason == "length"` mean and why should it be treated as an error?

---

## Stage 7 — Why the first token is slow and the rest are fast

Time to measure something real.

When you stream an LLM response, you'll notice a pause, then rapid text. There's a specific reason, and it's worth understanding because it determines how you optimize.

Generation happens in two distinct phases.

### Phase 1: prefill

The model processes your entire input at once. All tokens go through all layers together, in parallel, because they're all already known.

Cost: grows with **input** length. Parallelizable, so it's efficient per token.

### Phase 2: decode

Now it generates. One token. Then the next. Then the next. Each one needs the previous to exist, so they *cannot* be parallelized.

Cost: grows with **output** length. Strictly sequential.

```
[========== prefill: all input at once ==========] → first token appears
                                                      ↓
                                              [tok][tok][tok][tok]... one at a time
```

### The KV cache

There's a problem with decode. To generate token 50, attention needs keys and values for tokens 1–49. Recomputing all of them for every new token would be brutally slow.

So they're saved. After processing each token, its keys and values are cached. Token 50 computes its own K and V, then reads the other 49 from cache.

This is the **KV cache**, and it's why streaming feels steady — roughly constant time per token, rather than getting slower as the text grows.

### Measure it yourself

```python
import asyncio
import json
import time
import httpx
import os


async def measure_phases(prompt: str, max_tokens: int = 200):
    """Measure time-to-first-token vs. the rest."""
    start = time.monotonic()
    ttft = None
    token_count = 0

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", "https://api.openai.com/v1/chat/completions",
                                  json=payload, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    token_count += 1

    total = time.monotonic() - start
    decode_time = total - ttft
    print(f"  input tokens:      ~{count_tokens(prompt)}")
    print(f"  output tokens:     ~{token_count}")
    print(f"  time to 1st token:  {ttft:.2f}s   <- prefill")
    print(f"  remaining time:     {decode_time:.2f}s  <- decode")
    print(f"  decode speed:       {token_count/decode_time:.1f} tokens/sec")


async def main():
    print("SHORT input, LONG output:")
    await measure_phases("Write a 200-word story about a robot.", max_tokens=300)

    print("\nLONG input, SHORT output:")
    long_text = "The quick brown fox jumps over the lazy dog. " * 300
    await measure_phases(f"{long_text}\n\nReply with one word: what animal is mentioned?",
                          max_tokens=10)

asyncio.run(main())
```

Typical shape of the result:

```
SHORT input, LONG output:
  time to 1st token:  0.31s   <- prefill (fast, short input)
  remaining time:     4.82s   <- decode (slow, many tokens)

LONG input, SHORT output:
  time to 1st token:  1.24s   <- prefill (slower, long input)
  remaining time:     0.09s   <- decode (fast, few tokens)
```

### What this tells you about optimization

| Symptom | Cause | What actually helps |
|---|---|---|
| Slow to start responding | Long input → prefill | Shorter prompts, prompt caching |
| Starts fast, takes forever to finish | Long output → decode | Ask for shorter output, cap `max_tokens` |

These are different problems with different fixes. Shortening your prompt won't speed up a long answer. Knowing which phase you're in tells you where to look. [Part XLI](part41-performance-engineering.md) builds on this.

### Prompt caching

Since prefill processes your input, and much of your input (system prompt, tool definitions) is *identical* on every request — providers can cache that work. Send the same prefix, skip recomputing it, pay less and wait less.

The catch: it must be **byte-identical** and at the **start**. Put a timestamp or a session ID near the top of your system prompt and you break the cache on every single call. So:

```
[stable system prompt]     ← identical every time, cacheable
[stable tool definitions]  ← identical every time, cacheable
[conversation history]     ← changes
[current question]         ← changes
```

Stable content first, variable content last. Simple rule, meaningful savings. Revisited in [Part XL](part40-cost-optimization.md).

**Check yourself before Stage 8:** What's the difference between prefill and decode? Why can't decode be parallelized? What breaks prompt caching?

---

## Stage 8 — How a text predictor became an assistant

Everything so far describes a model that predicts likely next tokens. But that's not what you interact with. If you gave a raw next-token predictor the input "How do I reset my password?", a very plausible continuation is:

> "How do I change my email? How do I delete my account? — Frequently Asked Questions, page 2"

Which is a perfectly reasonable continuation of text found on the internet. It's just not an answer.

Turning a text predictor into something that answers takes additional training stages.

```
1. PRETRAINING
   Predict the next token across an enormous amount of text.
   Result: broad knowledge, fluent language, no helpful behavior.
        ↓
2. INSTRUCTION TUNING (supervised fine-tuning, "SFT")
   Train on curated (instruction, good response) pairs.
   Result: it now answers questions instead of continuing them.
        ↓
3. PREFERENCE OPTIMIZATION (RLHF, DPO, and relatives)
   Humans (or models) rank multiple responses. Train toward preferred ones.
   Result: more helpful, more honest, appropriately refuses harmful requests,
           follows system prompts more reliably.
        ↓
4. (SOME MODELS) REASONING TRAINING
   Reinforcement learning on problems with verifiable answers — math, code.
   Result: produces longer internal reasoning before answering.
           Better at multi-step problems, slower and more expensive.
```

### Three things this explains

**Why system prompts work at all.** Nothing in the architecture privileges a "system" message. Step 3 taught the model to weight it heavily. It's learned behavior, which is also why it's not absolute — a strong enough instruction elsewhere can sometimes override it. That's the root of prompt injection ([Part XXXV](part35-security.md)).

**Why "reasoning models" differ.** They're not architecturally different in kind. They had extra training that elicits longer deliberation. This is genuinely useful for hard multi-step problems and genuinely wasteful for simple ones — you're paying for tokens of deliberation on "what's the capital of France."

**Why models are sometimes confidently wrong.** Step 3 optimizes for responses humans *rate highly*. Humans tend to rate confident, well-structured answers highly. Confidence and correctness are being optimized together, imperfectly. The result is a system that can be fluently, structurally, convincingly wrong. Treat confidence as a writing style, not evidence.

**Check yourself before Stage 9:** Why does a pretrained-only model not answer questions properly? Why do system prompts work? Why is model confidence not a signal of correctness?

---

## Stage 9 — Fine-tuning vs. RAG vs. prompting

You now know enough to make a decision you'll face constantly: the model doesn't behave the way you need. What do you change?

Three options, in increasing cost and decreasing flexibility.

### Prompting

Change the instructions you send. Zero training cost, instant iteration, no infrastructure.

### RAG (retrieval)

Find relevant information and put it in the prompt. The model's frozen training-time knowledge gets supplemented with fresh facts at request time.

### Fine-tuning

Actually adjust the model's weights on your own examples.

- **Full fine-tuning** updates every weight. Expensive, needs real GPU infrastructure, and risks degrading the model's general abilities.
- **LoRA** freezes the original weights and trains small add-on matrices instead. Dramatically cheaper, and you can swap different LoRA add-ons onto one base model.
- **QLoRA** is LoRA plus compressing the frozen base model to lower precision, so it fits on a single GPU.

### How to choose

| What you need | Use | Why not the others |
|---|---|---|
| New/private/changing facts | **RAG** | Fine-tuning bakes in facts that then go stale; retraining for every document update is absurd |
| Consistent format, tone, or style | **Fine-tuning** | Prompting can get you most of the way, but drifts; fine-tuning is more reliable at volume |
| Task-specific instructions | **Prompting** | Anything else is overkill |
| High-volume narrow task, needs to be cheap and reliable | **Fine-tuning** a small model | A smaller fine-tuned model can beat a large prompted one, at a fraction of the cost |
| Model doesn't know your company's data | **RAG** | This is the textbook RAG case |

> **The mistake people make:** reaching for fine-tuning to teach the model facts. Fine-tuning is good at teaching *behavior* — how to respond. It's poor and expensive for teaching *knowledge* — what's true. Knowledge belongs in the prompt, retrieved at request time. Remember this one; it saves months of wasted effort.

**Check yourself:** For each of these, pick an approach and justify it: (a) the model should always reply in your company's JSON schema, (b) the model should know your product's pricing, which changes monthly, (c) the model should be more concise.

---

# The Project: Build an LLM Inspector

## What you're building

A command-line tool that shows you what's really happening inside an LLM request — tokens, timing, costs, and how settings change output.

```bash
$ python inspect_llm.py "Explain recursion in one sentence."
```

```
TOKENIZATION
  input:  7 tokens, 34 characters (4.9 chars/token)
  tokens: ['Ex','plain',' recursion',' in',' one',' sentence','.']

TIMING
  time to first token:   0.28s   (prefill)
  decode time:           0.91s   (14 tokens @ 15.4 tok/s)
  total:                 1.19s

COST
  input:      7 tokens × $0.15/1M = $0.0000011
  output:    14 tokens × $0.60/1M = $0.0000084
  total:                            $0.0000095

CONTEXT BUDGET
  used 21 / 128,000 tokens (0.02%)

RESPONSE
  Recursion is when a function calls itself to solve smaller
  instances of the same problem.
```

## Why this project

Everything in Parts I–III stops being abstract. You will look at real token counts, real prefill/decode splits, and real costs for your own prompts — and those numbers inform every decision you make in [Part IV](part04-llm-application-engineering.md) (client design), [Part XXXII](part32-context-engineering.md) (context budgets), and [Part XL](part40-cost-optimization.md) (cost control).

## Before you start

- API key in an environment variable
- `pip install tiktoken httpx`
- Your working `run_batch` from [Part I](part01-prerequisites.md) — you'll reuse its async structure

## Build it in stages

---

### Stage A — Tokenization report

**Goal:** Show what your prompt looks like as tokens. No API call yet.

**Do:**
1. Take a prompt from `sys.argv`
2. Encode it with `tiktoken`
3. Print the token count, character count, chars-per-token ratio
4. Print the first 20 tokens decoded individually

**You're done when:** You can run it on an English sentence, a code snippet, and a number-heavy string, and see the ratio change.

**Notice:** Which of your own real prompts is most token-expensive per character? That's where compression pays off most.

---

### Stage B — Streaming with timing

**Goal:** Call the API with streaming and measure prefill vs. decode.

**Do:**
1. Send a streaming request (build on Stage 7's code)
2. Record the moment the first content token arrives → that's TTFT
3. Count tokens as they stream
4. At the end, print TTFT, decode time, and tokens/second

**You're done when:** You can run it with a short prompt + long output, then a long prompt + short output, and clearly see which phase dominates each.

**Watch out for:** The first streamed chunk is sometimes empty or role-only. Only start the TTFT clock on the first chunk with actual content.

---

### Stage C — Cost calculation

**Goal:** Report what each call actually cost.

**Do:**
1. Make a price table: `{"gpt-4o-mini": {"input": 0.15, "output": 0.60}}` — dollars per million tokens. Get current numbers from your provider's pricing page.
2. Count input tokens with `tiktoken`; count output tokens from the stream
3. Compute and display input cost, output cost, total

**You're done when:** The numbers look sane — fractions of a cent for small calls.

**Notice:** Output tokens usually cost several times more than input tokens. Look at your price table and confirm the ratio for your model. That ratio is why "limit the output length" is often the single most effective cost lever ([Part XL](part40-cost-optimization.md)).

---

### Stage D — Context budget

**Goal:** Track how much of the window you're using, and refuse to exceed it.

**Do:**
1. Add a `--context-limit` argument (default to your model's real limit)
2. Compute: input tokens + `max_tokens` reserved for output
3. Print the percentage used
4. If the total would exceed the limit, error out *before* calling the API — don't pay for a request that'll be truncated
5. Check `finish_reason` in the response; if it's `"length"`, print a clear warning that the output was cut off

**You're done when:** You feed it a deliberately enormous prompt and get a clean, informative error instead of a confusing API failure.

---

### Stage E — Compare settings side by side

**Goal:** See temperature's effect with your own eyes, on your own prompts.

**Do:**
1. Add `--compare-temps` which runs the same prompt at temperature 0, 0.7, and 1.5
2. Run each **three times**
3. Print all nine outputs grouped by temperature

**You're done when:** You can see that temperature 0 gives near-identical output across runs, while 1.5 gives noticeably different ones.

**Then try this:** Use a prompt with a factual answer ("What's the capital of Australia?") and a creative one ("Write an opening line for a novel"). Notice that high temperature is harmless-to-useful on the creative prompt and actively risky on the factual one. That's the reasoning behind the settings table in Stage 5 — now you've verified it rather than taken my word for it.

---

### Stage F — Prompt cache experiment

**Goal:** Measure the effect of a stable prompt prefix.

**Do:**
1. Build a long system prompt (~1,000 tokens — paste in documentation or tool descriptions)
2. **Version A:** send it byte-identical on every request
3. **Version B:** insert `f"Request at {time.time()}"` at the very top
4. Run each 5 times. Compare TTFT and, if your provider reports cached token counts, compare those too

**You're done when:** You have measured numbers showing version A is faster and/or cheaper.

**Watch out for:** Not all providers/models cache automatically, and some require a minimum prefix length or an explicit flag. Check your provider's docs. If you see no difference, verify caching is actually available for your model before concluding the idea doesn't work.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Token counts don't match the API's reported usage | Wrong tokenizer for your model | Use the encoding matching your model; treat `tiktoken` counts for non-OpenAI models as estimates |
| TTFT is ~0 | Timing the request start instead of first content chunk | Start the clock at the first chunk containing text |
| Cost is wildly off | Confusing per-1K and per-1M pricing | Check your provider's units carefully |
| Response ends mid-sentence, no error | Ignoring `finish_reason` | Check it; treat `"length"` as a failure |
| Temperature 0 gives different outputs across runs | Provider-side batching (Stage 5's footnote) | Expected. Don't rely on exact reproducibility from a hosted API |

## Going further (optional)

1. **Cost projection.** Add a `--project N` flag estimating monthly cost at N requests/day. Do this for a real use case you care about. The number is often surprising and changes design decisions.
2. **Compare models.** Run the same prompt across three model tiers. Compare TTFT, tokens/sec, cost, and quality. Build a small table. This is the raw data behind model routing ([Part XL](part40-cost-optimization.md)).
3. **Batch inspection.** Combine with Part I's `run_batch` to inspect 50 prompts concurrently and report aggregate stats — p50/p95 TTFT, total cost, token distribution.

---

## Checkpoint — do not skip

**Knowledge:**
1. What is a token? Why does code cost more tokens per character than English?
2. Why does attention need position information added explicitly?
3. What does a residual connection do, and what fails without it?
4. What does the model actually output at each generation step?
5. Why is hallucination an expected consequence of the architecture?
6. What's the difference between prefill and decode? Why can't decode be parallelized?
7. What are the *two* separate problems with a very full context window?
8. Why do system prompts work, given that nothing in the architecture privileges them?
9. When should you fine-tune instead of using RAG, and when is that a mistake?

**Implementation:** Write, from memory: a token counter, a TTFT-measuring streaming call, and a context-budget check that accounts for reserved output tokens.

**Debugging:** An agent's JSON output is occasionally cut off mid-object. Walk through how you'd determine whether the cause is (a) hitting the context limit, (b) `max_tokens` set too low, or (c) the model just stopping early. What would you check first, and what field in the API response settles it?

**Architecture:** You're designing an agent that sends a 4,000-token system prompt plus tool definitions on every single call, thousands of times a day. Describe how you'd structure the request to minimize cost and latency, and what you'd measure to confirm it worked.

---

**Next:** [Part IV — LLM Application Engineering](part04-llm-application-engineering.md), where you wrap all of this in a client that survives production.
