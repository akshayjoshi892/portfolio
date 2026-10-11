# Part XIX — LangChain

## Before you start

**What you need already:** Everything through [Part XVIII](part18-rag-evaluation.md) — you've built an LLM client, tool calling, an agent loop, and a full RAG pipeline entirely by hand. That's the point of this part existing when it does, not earlier: you're about to see a framework automate things you already understand, which is a completely different experience than learning them from the framework's documentation as your first exposure.

**What this part gives you:** LangChain's actual abstraction, understood well enough to know what it's doing for you and what it's hiding — not a tour of its API surface, which changes between versions anyway.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Read and write LangChain code by understanding *why* it's shaped the way it is, because you recognize every piece as something you already built by hand.

---

## Stage 1 — The one idea LangChain is actually built on

Look back at every pipeline you've built since Part IV. Each one is a sequence: take input, transform it, pass the result to the next step, transform again.

```
prompt template  →  model call  →  parse the response
```

```
embed the query  →  search the index  →  build a grounded prompt  →  model call
```

Every single pipeline in this curriculum has this same shape: **a series of steps, each taking the previous step's output as its input.** LangChain's core insight is to give every step — a prompt template, a model call, a parser, a retriever — the *exact same interface*, so any of them can be connected to any other with the same simple syntax, regardless of what each one actually does internally.

That shared interface is called a **Runnable**. Once every piece implements it, you can chain them with a single operator:

```python
chain = prompt | model | output_parser
```

That `|` (the same pipe operator you might know from a Unix shell) means "feed the output of the left side into the input of the right side." It works because `prompt`, `model`, and `output_parser` are all Runnables — they all promise to accept an input and produce an output in a compatible, predictable shape.

**Check yourself before Stage 2:** In your own pipelines from earlier parts, what plays the role of "prompt," "model," and "output_parser" in, say, your Part XI RAG pipeline?

---

## Stage 2 — Your first chain, recognized as something you already built

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("user", "{question}"),
])

chain = prompt | model | StrOutputParser()

result = chain.invoke({"question": "What is RAG?"})
print(result)
```

Walk through exactly what this replaces, from your own code:

- `ChatPromptTemplate.from_messages(...)` is the `Template.substitute(...)` pattern you hand-built in [Part V Stage 4](part05-prompt-engineering.md) — filling variables into a system/user message structure.
- `ChatOpenAI(...)` wraps the HTTP call you hand-built in [Part IV Stage 2](part04-llm-application-engineering.md).
- `StrOutputParser()` extracts plain text from the response, roughly what your `client.ask()` already returned directly.
- `chain.invoke({"question": ...})` runs the whole thing — prompt fills in, request goes out, response comes back, text gets extracted.

You're not learning a new *concept* here. You're learning that this specific sequence of operations — which you've written by hand several times by now — has a standard, reusable shape a library can provide for you.

**Check yourself before Stage 3:** In your own words, what does `prompt | model | StrOutputParser()` actually do when `.invoke()` is called on it?

---

## Stage 3 — Structured output and tool calling, in LangChain's terms

You built structured output validation with Pydantic in [Part IV Stage 3](part04-llm-application-engineering.md), and tool calling with schema validation in [Part VI](part06-structured-outputs-tool-calling.md). LangChain wraps both in the same Runnable interface.

```python
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="the direct answer")
    confidence: float = Field(description="0-1 confidence score")


structured_model = model.with_structured_output(Answer)
structured_chain = prompt | structured_model

result: Answer = structured_chain.invoke({"question": "What is RAG?"})
print(result.answer, result.confidence)
```

`with_structured_output(Answer)` is doing exactly what your `ask_structured` method did in Part IV: send the schema, get the response, validate it against the Pydantic model, hand back a typed object instead of a raw string.

```python
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("disallowed characters")
    return eval(expression, {"__builtins__": {}}, {})


model_with_tools = model.bind_tools([calculator])
response = model_with_tools.invoke("What is 6 * 7?")
print(response.tool_calls)
```

```
[{'name': 'calculator', 'args': {'expression': '6 * 7'}, 'id': '...'}]
```

The `@tool` decorator is generating a JSON Schema from the function's type hints and docstring — the exact same schema-generation job Pydantic's `model_json_schema()` did for you by hand in [Part VI Stage 4](part06-structured-outputs-tool-calling.md). `bind_tools` attaches that schema to every request, and `response.tool_calls` gives you back the parsed tool call — the same structure your `execute_tool_call` function consumed in Part VI, just generated for you instead of hand-written.

**Check yourself before Stage 4:** What does `@tool` generate automatically that you generated by hand with Pydantic in Part VI?

---

## Stage 4 — A RAG chain, recognized piece by piece

Here's your entire Part XI–XVI RAG pipeline, expressed with LangChain's Runnables:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

vectorstore = Qdrant.from_texts(
    texts=["Employees get 15 days paid leave per year."],
    embedding=OpenAIEmbeddings(),
    url="http://localhost:6333",
    collection_name="docs",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {question}"
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    | rag_prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("How much leave do employees get?")
```

Map every piece to something you already built:

| LangChain piece | What you built by hand |
|---|---|
| `Qdrant.from_texts(...)` | [Part XVI](part16-qdrant-deep-dive.md)'s `client.upsert(...)` with embeddings |
| `retriever` | [Part XI](part11-rag-fundamentals.md)'s `retrieve()` function |
| `format_docs` | [Part XI](part11-rag-fundamentals.md)'s chunk-joining logic in `build_grounded_prompt` |
| `RunnableParallel(context=..., question=...)` | Computing two independent values you'll both need in the final prompt — new syntax, not a new idea |
| `rag_prompt` | [Part XI](part11-rag-fundamentals.md)'s prompt template with the "answer using only context" instruction |

The one genuinely new piece of syntax is `RunnableParallel`. It runs `retriever | format_docs` and `RunnablePassthrough()` (which just passes the original input through unchanged) *at the same time*, producing a dictionary with both results — `{"context": ..., "question": ...}` — which then feeds into `rag_prompt`'s two template variables. This is the framework's way of expressing "I need two different things computed from the same input, then combined" without you writing that fan-out logic by hand.

**Check yourself before Stage 5:** Why does the RAG chain need `RunnableParallel` here, when the earlier `prompt | model | parser` chain didn't need anything like it?

---

## Stage 5 — What you're actually trading

Here's the honest accounting, now that you've seen the mapping directly:

**What you gain:** less code to write and maintain for standard patterns, a large ecosystem of pre-built integrations (vector stores, document loaders, tool wrappers) so you don't reimplement common connectors, and `LangSmith` — a tracing product that hooks into the same callback system every chain already uses, giving you observability roughly for free once you're using the framework.

**What you give up:** when something goes wrong inside a chain, you're debugging through a layer of abstraction instead of your own code — and if you didn't already understand what a `Runnable`, a retriever, or `bind_tools` was actually doing underneath, that abstraction is opaque instead of transparent. This is exactly why this curriculum built everything by hand first (Rule 4 from the plan: build before abstracting) — you're now debugging a LangChain chain the same way you'd debug your own code, because you know what it's standing in for.

**Check yourself before the project:** If a LangChain-based tool-calling chain silently fails to call a tool it should have, what would you check first, given what you know from having built `ToolRegistry` by hand in Part VI?

---

# The Project: Rebuild Your RAG Pipeline in LangChain, Then Compare

## What you're building

Your Part XVI RAG pipeline, reimplemented using LangChain's Runnables — then a direct, honest comparison against your hand-built version.

```bash
$ python langchain_rag.py "How much leave do employees get?"
```

```
[LangChain version] Employees get 15 days of paid leave per year.
[Hand-built version] Employees get 15 days of paid leave per year.

Lines of code: LangChain=24, hand-built=61
Latency: LangChain=1.2s, hand-built=1.1s
```

## Why this project

The goal isn't "LangChain is better" or "hand-built is better" — it's confirming, with your own eyes, that they're doing the *same thing*, so you carry forward the understanding rather than a new, disconnected set of framework-specific knowledge.

## Build it in stages

---

### Stage A — Basic chain

**Goal:** `prompt | model | StrOutputParser()`, working, compared against your Part IV `LLMClient.ask()`.

**Do:**
1. Set up the basic chain from Stage 2
2. Run the same question through both LangChain and your own `LLMClient`
3. Confirm both give equivalent answers

**You're done when:** You can point at each LangChain piece and name the Part IV/V code it replaces.

---

### Stage B — Structured output and tools

**Goal:** `with_structured_output` and `bind_tools`, compared against your Part IV/VI code.

**Do:**
1. Reuse a Pydantic schema from your Part IV project
2. Run it through both `ask_structured` and LangChain's `with_structured_output`
3. Reuse a tool from your Part VI project, wrap it with `@tool`, and confirm `bind_tools` correctly triggers it on the same test questions your hand-built `ToolRegistry` handled

**You're done when:** Both structured output and tool calling produce equivalent results through both paths.

---

### Stage C — The full RAG chain

**Goal:** Your Qdrant-backed RAG pipeline, reimplemented with `RunnableParallel` and a retriever, against the *same* Qdrant collection you built in Part XVI.

**Do:**
1. Connect LangChain's `Qdrant` vector store wrapper to your existing collection (don't rebuild the index — point at what's already there)
2. Build the RAG chain as in Stage 4
3. Run your full Part XVIII test question set through it

**You're done when:** The LangChain version's answers are equivalent to your hand-built version's answers on the same test set, using the same underlying data.

---

### Stage D — Honest comparison

**Goal:** Real numbers, not a vibe-based preference.

**Do:**
1. Count lines of code for each version (excluding imports/setup)
2. Time both versions on the same 10 questions
3. Deliberately introduce a bug in each (e.g., a missing tool schema field) and time how long it takes you to diagnose it, given what you know about each implementation

**You're done when:** You have a genuine, evidence-based sense of the trade-off, specific to your own experience building both — not a recycled opinion from a blog post.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| `bind_tools` response has no `tool_calls` even for a clearly tool-requiring question | Model doesn't support native tool calling, or tools weren't actually bound to the instance being invoked | Confirm your model supports tool calling; check you're invoking `model_with_tools`, not the original `model` |
| `RunnableParallel` output doesn't have the keys your prompt template expects | Mismatch between the parallel dict's keys and the prompt's `{variable}` names | Print the `RunnableParallel` step's output in isolation (invoke it directly, without piping further) to see its actual shape |
| LangChain's retriever returns different results than your hand-built one on the same Qdrant collection | Different default `k`, or LangChain applying a different similarity metric than expected | Explicitly set `search_kwargs={"k": ...}` and confirm the collection's configured distance metric matches |

## Going further (optional)

1. **Add tracing.** Wire in a `BaseCallbackHandler` (as sketched in this part) and watch it fire for every step of your RAG chain — this is your first hands-on look at the observability hook every framework integration reuses, which becomes central in [Part XXXIX](part39-observability.md).
2. **Model fallback.** Use `model.with_fallbacks([...])` and compare it against your own hand-built `call_with_fallback` from Part IV — same idea, framework-provided.
3. **A streaming version** of the RAG chain using `.stream()` instead of `.invoke()`, compared against your Part IV streaming implementation.

---

## Checkpoint

**Knowledge:**
1. What single interface does every LangChain component (prompt, model, parser, retriever) share, and why does that make `|` composition possible?
2. What does `RunnableParallel` solve that a simple linear chain (`a | b | c`) can't?
3. What do you gain and what do you give up by using LangChain instead of your hand-built code?

**Implementation:** Build a RAG chain using `RunnableParallel`, a retriever, and a prompt from memory, without referring to this file.

**Debugging:** A LangChain agent silently doesn't call a tool it should have. Using what you know from Part VI about how tool calling actually works underneath, what would you check first?

**Architecture:** You're building a support-ticket triage pipeline needing model fallback and per-step latency tracing. Decide, with justification, whether you'd use LangChain's built-in mechanisms or your own hand-built equivalents from Parts IV and XXXIX (once you reach it) — and why.

---

**Next:** [Part XX — LangGraph](part20-langgraph.md), where the checkpointing you hand-built in Part IX becomes a first-class, built-in feature of a graph-based agent framework.
