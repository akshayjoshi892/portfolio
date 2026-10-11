# Part IV — Making LLM Calls Survive Production

## Before you start

**What you need already:** [Part I](part01-prerequisites.md) (async, retry, backoff, concurrency) and [Part III](part03-llm-fundamentals.md) (tokens, streaming, prefill/decode, context limits).

**What this part gives you:** A single reusable client class that every later part calls into. Right now, calling an LLM means "send a request, hope it works." This part turns that into "send a request, and if it fails, or a provider goes down, or the response is malformed, handle it correctly — automatically."

**Time:** 3–4 days.

**When you finish, you'll be able to:** Build a client that survives a provider outage by switching providers, validates every response before your code touches it, and never silently loses a request.

---

## Stage 1 — See the request pipeline as six failure points

Here's what actually happens between your code calling `client.ask("...")` and getting an answer back:

```
Your code
   │
   ▼
Build the request      ← could be malformed (bad JSON, wrong types)
   │
   ▼
Send over the network   ← could time out, drop, or the provider could be down
   │
   ▼
Provider processes it   ← could rate-limit you, error, or be temporarily overloaded
   │
   ▼
Response comes back     ← could be incomplete, truncated, wrong shape
   │
   ▼
Parse the response      ← could fail if the shape wasn't what you expected
   │
   ▼
Validate it              ← could be well-formed JSON that's still wrong (bad values)
   │
   ▼
Your code uses it
```

Six places between "your code" at the top and "your code" at the bottom where things can go wrong. A beginner client handles zero of them — it does the happy path and crashes on everything else. This part builds something that handles all six.

**Check yourself before Stage 2:** Without looking, list the six points where a request can fail.

---

## Stage 2 — One provider, handled honestly

Start small. Wrap a single provider's API in a function that does the minimum honest thing: tell you clearly what happened.

```python
import httpx


class LLMError(Exception):
    """Base class for anything that goes wrong calling an LLM."""


class RetriableError(LLMError):
    """Might succeed if you just try again: timeouts, 429, 5xx."""


class NonRetriableError(LLMError):
    """Will fail identically every time: bad request, bad auth."""


async def call_openai(client: httpx.AsyncClient, api_key: str, prompt: str) -> str:
    response = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )

    if response.status_code == 429 or response.status_code >= 500:
        raise RetriableError(f"provider returned {response.status_code}")
    if response.status_code >= 400:
        raise NonRetriableError(f"bad request: {response.status_code} {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]
```

Notice this is nothing new — it's Part I's `RetriableError` split applied to real HTTP status codes. A 429 (rate limited) or 5xx (server error) might succeed on retry. A 400 (you sent something wrong) will fail the exact same way every time; retrying it just burns time and money.

**Check yourself:** Why does a 429 get retried but a 400 doesn't? What would happen if you retried a 400 four times?

---

## Stage 3 — Structured output: making the model return a shape you can trust

So far, `call_openai` returns a raw string. Most of the time you want *structured* data — a specific set of fields with specific types, not text you have to parse by hand.

### The naive approach, and why it's fragile

```python
prompt = "Answer this and return JSON like {\"answer\": ..., \"confidence\": ...}"
```

This works most of the time. "Most of the time" is not good enough for code. The model might:
- Add a sentence before the JSON ("Sure, here's the answer: {...}")
- Use a slightly different field name
- Format a number as a string
- Just... not follow the instruction

### Three levels of guarantee

| Approach | What it guarantees |
|---|---|
| Ask nicely in the prompt | Nothing — you're hoping |
| "JSON mode" (a provider flag) | Output is *valid JSON* — not that it has the fields you want |
| Constrained decoding / schema mode | Output matches your exact schema, field names and types |

Use the strongest one your provider offers. But — and this is the point of this stage — **even with the strongest guarantee, validate anyway.** A schema can guarantee `confidence` is a number. It can't guarantee `confidence` is between 0 and 1, or that `answer` isn't an empty string.

### Building the validation boundary

This is where Pydantic earns its place — it turns "JSON that showed up" into "typed data your code can trust."

```python
from pydantic import BaseModel, Field, ValidationError


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


def parse_and_validate(raw_json: str) -> Answer:
    try:
        return Answer.model_validate_json(raw_json)
    except ValidationError as e:
        raise NonRetriableError(f"model output failed validation: {e}") from e
```

Try it against both good and bad input:

```python
print(parse_and_validate('{"answer": "Paris", "confidence": 0.95}'))
# answer='Paris' confidence=0.95

parse_and_validate('{"answer": "", "confidence": 1.5}')
# raises: NonRetriableError — empty answer AND confidence out of range
```

That second call is schema-shaped JSON — a string and a number, right field names — and it's still wrong. `min_length=1` and `ge=0.0, le=1.0` are the business-rule checks a schema alone can't give you. This is why "the API supports structured output" doesn't mean you can skip validation.

**Check yourself before Stage 4:** Give an example of JSON output that would pass a schema check but still be wrong. Why validate even when using the strongest guarantee level?

---

## Stage 4 — Surviving a provider outage: fallback

Providers go down. Not often, but for a business depending on one, "not often" eventually means "during your busiest hour."

The fix: have a second provider ready, and switch to it automatically.

```python
from dataclasses import dataclass
from typing import Protocol


class Provider(Protocol):
    name: str
    async def call(self, prompt: str) -> str: ...


@dataclass
class OpenAIProvider:
    api_key: str
    client: httpx.AsyncClient
    name: str = "openai"

    async def call(self, prompt: str) -> str:
        return await call_openai(self.client, self.api_key, prompt)


@dataclass
class AnthropicProvider:
    api_key: str
    client: httpx.AsyncClient
    name: str = "anthropic"

    async def call(self, prompt: str) -> str:
        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-haiku-4-5", "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise RetriableError(f"provider returned {response.status_code}")
        if response.status_code >= 400:
            raise NonRetriableError(f"bad request: {response.text}")
        return response.json()["content"][0]["text"]
```

Now the part that matters — trying each provider in order, with the retry logic from Part I applied to each one before moving to the next:

```python
import asyncio


async def call_with_fallback(providers: list[Provider], prompt: str,
                              attempts_per_provider: int = 2) -> tuple[str, str]:
    """Returns (result_text, name_of_provider_that_succeeded)."""
    last_error = None

    for provider in providers:
        for attempt in range(1, attempts_per_provider + 1):
            try:
                result = await provider.call(prompt)
                return result, provider.name
            except RetriableError as e:
                last_error = e
                if attempt < attempts_per_provider:
                    await asyncio.sleep(2 ** attempt)
            except NonRetriableError as e:
                # Not the provider's fault — no point trying other providers either
                raise

        print(f"[fallback] {provider.name} exhausted, trying next provider")

    raise LLMError(f"all providers failed, last error: {last_error}")
```

### Watch what this does under two different failures

**Provider has a bad-request problem (your fault):** `NonRetriableError` propagates immediately, no fallback attempted. Correct — switching providers won't fix a malformed request.

**Provider is down (their fault):** `RetriableError` triggers retries, then moves to the next provider in the list. If you have two providers and one is down, users don't notice.

**Try it:** Make `OpenAIProvider.call` always raise `RetriableError` (simulate an outage) and confirm `call_with_fallback` seamlessly returns a result from `AnthropicProvider` instead.

**Check yourself before Stage 5:** Why does `NonRetriableError` skip fallback entirely while `RetriableError` triggers it? What would happen if you got this backwards?

---

## Stage 5 — Streaming through a provider-agnostic interface

You measured streaming's mechanics in [Part III Stage 7](part03-llm-fundamentals.md). Now build it into the client so callers get consistent behavior regardless of which provider answers.

```python
from typing import AsyncIterator


async def stream_openai(client: httpx.AsyncClient, api_key: str, prompt: str) -> AsyncIterator[str]:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async with client.stream("POST", "https://api.openai.com/v1/chat/completions",
                              json=payload, headers=headers, timeout=30) as response:
        if response.status_code >= 400:
            raise RetriableError(f"stream failed to start: {response.status_code}")

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = httpx.Response(200, content=data.encode()).json()  # simplified parse
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta
```

Using it is exactly like using any other async generator:

```python
async def main():
    async with httpx.AsyncClient() as client:
        async for piece in stream_openai(client, api_key, "Tell me a short story"):
            print(piece, end="", flush=True)
```

**The catch with streaming + structured output:** you can't validate JSON until it's fully arrived — a half-streamed object isn't valid JSON yet. So for anything you need to validate (Stage 3), buffer the full response before parsing. Stream only for content that's meant to be read as it arrives, like a chat reply.

**Check yourself before Stage 6:** Why can't you validate a Pydantic model against a partially-streamed response?

---

## Stage 6 — Cost, denominated correctly

You built a token counter and a cost calculator on one request in [Part III](part03-llm-fundamentals.md)'s project. Now make it a permanent part of every call.

```python
from dataclasses import dataclass, field


PRICES_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


@dataclass
class CallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float = field(init=False)

    def __post_init__(self):
        prices = PRICES_PER_MILLION.get(self.model, {"input": 0, "output": 0})
        self.cost_usd = (
            self.input_tokens / 1_000_000 * prices["input"]
            + self.output_tokens / 1_000_000 * prices["output"]
        )
```

`__post_init__` runs right after the dataclass's normal `__init__`. It's the right place to compute a value that *derives* from the other fields — here, cost is entirely determined by tokens and model, so it shouldn't be something the caller passes in and might get wrong.

You'll attach a running total of these records to every agent run starting in [Part XXVIII](part28-agent-harness.md), where it becomes a hard budget you enforce, not just a number you log.

**Check yourself:** Why compute `cost_usd` in `__post_init__` instead of accepting it as a constructor argument?

---

## Stage 7 — Assembling the real client

Everything above becomes one class. This is the file you'll import from for the rest of the curriculum.

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class RetriableError(LLMError):
    pass


class NonRetriableError(LLMError):
    pass


class SchemaValidationError(LLMError):
    pass


class Provider(Protocol):
    name: str
    model: str
    async def call(self, prompt: str) -> str: ...


@dataclass
class LLMClient:
    providers: list[Provider]
    attempts_per_provider: int = 2

    async def ask(self, prompt: str) -> tuple[str, str]:
        """Returns (text, provider_name_that_answered)."""
        last_error: Exception | None = None

        for provider in self.providers:
            for attempt in range(1, self.attempts_per_provider + 1):
                start = time.monotonic()
                try:
                    text = await provider.call(prompt)
                    latency = time.monotonic() - start
                    print(f"[ok] {provider.name} answered in {latency:.2f}s "
                          f"(attempt {attempt})")
                    return text, provider.name
                except RetriableError as e:
                    last_error = e
                    if attempt < self.attempts_per_provider:
                        await asyncio.sleep(2 ** attempt)
                except NonRetriableError:
                    raise

            print(f"[fallback] {provider.name} exhausted, trying next")

        raise LLMError(f"all providers failed: {last_error}")

    async def ask_structured(self, prompt: str, schema: type[T]) -> T:
        text, _ = await self.ask(prompt)
        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            raise SchemaValidationError(f"output didn't match {schema.__name__}: {e}") from e
```

Notice `ask_structured` builds on `ask` — it doesn't duplicate the retry/fallback logic, it reuses it and adds one more layer (validation) on top. That's the boundary layer from Part I's mental model, made concrete: `ask` handles layer-3 unreliability (the model/network), `ask_structured` adds the schema check that turns raw output into trusted data.

**Check yourself before the project:** Trace through `ask_structured` in your head. If the first provider's response fails validation, does it try the second provider? Should it? (Think about what validation failure actually tells you — is it the provider's fault, or the prompt's?)

---

# The Project: A Production-Shaped LLM Client

## What you're building

A `LLMClient` package with two real providers, automatic fallback, structured output, streaming, and cost tracking — used as a library, not a script.

```python
from llm_client import LLMClient, OpenAIProvider, AnthropicProvider

client = LLMClient(providers=[
    OpenAIProvider(api_key=OPENAI_KEY),
    AnthropicProvider(api_key=ANTHROPIC_KEY),
])

text, provider = await client.ask("What is the capital of France?")
print(f"{provider} says: {text}")

class Answer(BaseModel):
    answer: str
    confidence: float

result = await client.ask_structured("Answer with JSON: capital of France?", Answer)
```

## Why this project

Every later part imports something like this. Part VI's tool-calling loop needs a reliable client underneath it. Part XXVIII's harness needs cost tracking. Building it carefully now means you're not rebuilding it five times later.

## Build it in stages

---

### Stage A — One provider, real calls, real failures

**Goal:** `OpenAIProvider` works against the real API, with the retriable/non-retriable split correctly implemented.

**Do:**
1. Implement `call()` exactly as in Stage 2
2. Test the happy path: a real question, a real answer
3. Test the error paths **deliberately** — send a request with an invalid model name (should be non-retriable, fails fast) and, if you can, simulate a timeout by setting `timeout=0.001` (should be retriable)

**You're done when:** You've seen both error types actually fire and confirmed which one they are.

---

### Stage B — Second provider, fallback

**Goal:** Two providers, automatic switch-over.

**Do:**
1. Implement `AnthropicProvider` (or a second provider of your choice)
2. Wire both into `LLMClient`
3. **Force a failure test:** temporarily set the OpenAI provider's API key to garbage. Call `client.ask(...)` and confirm it falls through to Anthropic and still returns a good answer

**You're done when:** That forced-failure test passes — you get a correct answer despite one provider being completely broken.

---

### Stage C — Structured output with real validation failures

**Goal:** `ask_structured` works, and you've seen it correctly reject bad output.

**Do:**
1. Define 2 Pydantic schemas for different tasks (e.g., a sentiment classifier, a data extractor)
2. Call `ask_structured` for each against real prompts
3. **Force a validation failure:** define a schema with an impossible constraint (e.g., `confidence: float = Field(ge=2.0)` — nothing can satisfy this) and confirm you get a clean `SchemaValidationError`, not a crash

**You're done when:** Good input validates correctly, and the forced-impossible schema fails with a clear error message naming the exact problem.

---

### Stage D — Cost tracking across a session

**Goal:** Every call updates a running total you can inspect.

**Do:**
1. Add a `total_cost_usd` and `call_history: list[CallRecord]` to `LLMClient`
2. Update both after every successful call (you'll need the provider to report token usage — check your provider's response format for a `usage` field)
3. Add a `client.summary()` method printing total cost, total calls, and cost by provider

**You're done when:** After running 10 varied calls, `client.summary()` gives you a believable total, and you can see which provider/model was used for each.

---

### Stage E — Streaming, integrated

**Goal:** A `client.stream(prompt)` method with the same fallback behavior as `ask`.

**Do:**
1. Add `stream()` to each provider class
2. In `LLMClient.stream()`, try the first provider; if the connection fails to even start, fall back to the next (once streaming has started and is mid-way through, don't try to fall back — that would duplicate partial output)
3. Print the streamed response live to your terminal

**You're done when:** You can watch a real response stream token-by-token, and forcing the primary provider to fail still gets you a streamed response from the fallback.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| A bad prompt gets retried 4 times before failing | Status code check backwards, or catching bare `Exception` | Recheck the 429/5xx vs 4xx split from Stage 2 |
| Fallback never triggers even when you force a failure | Exception type mismatch — code raises something that isn't `RetriableError` | Make sure every provider's error handling maps to your two exception types |
| `SchemaValidationError` triggers fallback to the next provider | `ask_structured` catching `ValidationError` too broadly, inside the retry loop instead of after `ask()` returns | Validation happens once, after you already have a successful raw response — it's not a provider-communication failure |
| Cost totals are all zero | Not reading token usage from the actual response, or reading the wrong field name | Print the raw response once and find the real `usage` field for your provider |

## Going further (optional)

1. **A third provider on a completely different API shape.** Forces you to keep the `Provider` interface genuinely provider-agnostic rather than accidentally OpenAI-shaped.
2. **Retry budget across the whole session**, not just per-call — if you've had 20 failures in the last minute, stop trying and fail fast rather than grinding through retries on every single request. This previews the circuit breaker in [Part VI](part06-structured-outputs-tool-calling.md).
3. **Per-call correlation ID.** Add a `request_id` to every `CallRecord` and print it with every log line. You won't see the payoff until [Part XXXIX](part39-observability.md), but wiring it in now costs nothing and saves rework later.

---

## Checkpoint

**Knowledge:**
1. List the six places a request can fail between your code and the model's answer.
2. Why do you validate structured output even when using the provider's strongest schema-enforcement mode?
3. Why does `NonRetriableError` skip fallback to other providers, while `RetriableError` triggers it?
4. Why can't you validate a partially-streamed JSON response?

**Implementation:** Build `LLMClient.ask` and `LLMClient.ask_structured` from memory, including the retriable/non-retriable split and fallback.

**Debugging:** A teammate reports that your agent occasionally returns a `SchemaValidationError` even though the prompt clearly asks for the right JSON shape. Walk through how you'd figure out whether the problem is the prompt, the schema, or an occasional model mistake — and what you'd log to tell these apart next time.

**Architecture:** You're adding a third LLM provider to `LLMClient` purely for cost reasons (it's much cheaper for simple tasks). Where would you add the logic that decides which provider to try *first*, and what would that decision need to know about the request?

---

**Next:** [Part V — Prompt Engineering](part05-prompt-engineering.md), where you learn what to actually put inside the `prompt` string you've been passing to this client.
