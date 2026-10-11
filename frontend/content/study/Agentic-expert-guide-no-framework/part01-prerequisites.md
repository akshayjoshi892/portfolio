# Part I — Python and Systems Skills You Actually Need

## Before you start

**What you need already:** You can write basic Python — variables, functions, loops, `if` statements, and simple classes. If you can't, stop here and finish a beginner Python course first. This curriculum will not teach you that.

**What this part gives you:** Not all of Python. Just the specific parts that agent code uses every single day. The big one is `async` — agent code is full of it, and if you don't understand it, every later file will look like noise.

**Time:** About a week if `async` is new to you. Two days if you already write async Python.

**When you finish, you'll be able to:** Write a program that calls an API 50 times at once, retries the calls that fail, waits longer between each retry, and never has more than 5 calls running at the same time. That exact program is the skeleton of every agent you build later.

---

## Stage 1 — Why agent code spends almost all its time waiting

Let's start with a fact about agent programs that explains why they're written the way they are.

Here's what a single agent step does:

1. Send a question to an LLM over the internet. **Wait ~2 seconds.**
2. Read the answer.
3. Call a tool (maybe a web search). **Wait ~0.5 seconds.**
4. Send the tool's result back to the LLM. **Wait ~2 seconds.**

Add that up. The program spends about 4.5 seconds waiting and maybe 0.001 seconds actually computing. Your CPU is idle for 99.98% of the time.

This matters because of what happens when you have 10 users.

### The naive version

```python
import time


def call_llm(question):
    time.sleep(2)          # pretend this is a network call
    return f"answer to {question}"


def handle_all_users(questions):
    answers = []
    for q in questions:
        answers.append(call_llm(q))
    return answers


start = time.time()
handle_all_users([f"question {i}" for i in range(10)])
print(f"took {time.time() - start:.1f} seconds")
```

Run it. You get:

```
took 20.0 seconds
```

Ten users, two seconds each, twenty seconds total. The tenth user waited 20 seconds for a 2-second request. And your CPU did nothing the whole time.

The problem isn't that the calls are slow. The problem is that your program *stands still* while waiting, instead of starting the next call.

### What we want instead

Start all 10 calls. Then wait. All 10 come back around the same time. Total: ~2 seconds, not 20.

That's what `async` gives you. Stage 2 shows how.

> **Key idea:** Agent code is *I/O-bound* — "I/O" means input/output, i.e. talking to things outside your program (networks, disks, databases). I/O-bound means your program's speed is limited by waiting for those things, not by how fast your CPU computes. Async is the tool for I/O-bound work.

**Check yourself before Stage 2:** Can you explain, without looking, why 10 LLM calls take 20 seconds in the code above even though the computer isn't busy?

---

## Stage 2 — Your first async program

Async adds three things to Python. Let's meet them one at a time.

### `async def` — marks a function as pausable

```python
async def call_llm(question):
    ...
```

A function defined with `async def` is called a **coroutine**. The difference from a normal function: a coroutine is allowed to pause in the middle, let other code run, and resume later.

### `await` — the pause point

```python
result = await something_slow()
```

`await` means: "This will take a while. Pause me here, go run other code, and wake me up when the result is ready."

You can only use `await` inside an `async def` function.

### `asyncio.run()` — starts the whole thing

```python
asyncio.run(main())
```

Async code needs a manager running underneath it — something that keeps track of which coroutines are paused and wakes them up when their results arrive. That manager is called the **event loop**. `asyncio.run()` starts it, runs your coroutine, and shuts it down when finished.

### Putting it together

```python
import asyncio
import time


async def call_llm(question):
    await asyncio.sleep(2)        # async version of time.sleep
    return f"answer to {question}"


async def main():
    start = time.time()
    answer = await call_llm("what is 2+2?")
    print(answer)
    print(f"took {time.time() - start:.1f} seconds")


asyncio.run(main())
```

Output:

```
answer to what is 2+2?
took 2.0 seconds
```

Still 2 seconds — we haven't made anything faster yet. One call is one call. The speedup comes in Stage 3, when we run several at once.

### The one trap that catches everyone

Notice I used `asyncio.sleep(2)`, not `time.sleep(2)`.

`time.sleep()` blocks. It freezes the entire event loop — every paused coroutine stops too, because they all share one thread. `asyncio.sleep()` yields: it tells the loop "wake me in 2 seconds, go do other things meanwhile."

This generalizes beyond sleeping. Inside async code:

| Don't use | Use instead |
|---|---|
| `time.sleep()` | `await asyncio.sleep()` |
| `requests.get()` | `await httpx.AsyncClient().get()` |
| A slow CPU-heavy loop | `await asyncio.to_thread(slow_function, args)` |

If you call a blocking function inside `async def`, your program still works — it just silently loses all the benefit of async. Nothing warns you. This is the single most common async bug.

**Try breaking it:** Change `await asyncio.sleep(2)` to `time.sleep(2)` in the Stage 3 code below (after you write it) and watch the runtime jump from 2 seconds back to 20. Do this. Seeing it is worth more than reading about it.

**Check yourself before Stage 3:** What does `await` actually do? What is the event loop? Why does `time.sleep()` break async code?

---

## Stage 3 — Running many calls at once

Now the payoff. We want to start all 10 calls before waiting for any of them.

The tool is `asyncio.gather()`. You hand it several coroutines, it runs them all at the same time, and it gives you back a list of results in the same order you passed them in.

```python
import asyncio
import time


async def call_llm(question):
    await asyncio.sleep(2)
    return f"answer to {question}"


async def main():
    questions = [f"question {i}" for i in range(10)]

    start = time.time()
    answers = await asyncio.gather(*(call_llm(q) for q in questions))
    print(f"got {len(answers)} answers in {time.time() - start:.1f} seconds")


asyncio.run(main())
```

Output:

```
got 10 answers in 2.0 seconds
```

**20 seconds → 2 seconds.** Same work, same machine, same network. The only change is that we stopped standing still.

### What that `*` is doing

```python
asyncio.gather(*(call_llm(q) for q in questions))
```

The part in parentheses creates 10 coroutine objects. The `*` unpacks them into 10 separate arguments, because `gather` wants `gather(coro1, coro2, coro3, ...)`, not `gather([coro1, coro2, coro3])`.

If that's unfamiliar, this longer version does exactly the same thing:

```python
coroutines = []
for q in questions:
    coroutines.append(call_llm(q))
answers = await asyncio.gather(*coroutines)
```

### An important detail about creating coroutines

Calling `call_llm("hi")` does **not** run the function. It creates a coroutine object that is ready to run but hasn't started. Nothing happens until you `await` it or hand it to `gather`.

```python
c = call_llm("hi")
print(c)        # <coroutine object call_llm at 0x...>  — not a result!
result = await c   # NOW it runs
```

This trips people up: they call an async function, forget to `await`, and wonder why they got a weird object instead of their answer. If you see `<coroutine object ...>` in your output, you forgot an `await`.

**Check yourself before Stage 4:** Why did the runtime drop from 20 seconds to 2? What does calling an `async def` function return if you don't `await` it?

---

## Stage 4 — Handling failure: retry

Real API calls fail. Not occasionally — constantly, at scale. The server is overloaded, the network hiccups, you hit a rate limit. If 5% of calls fail and your agent makes 20 calls, most runs fail somewhere.

So we retry. Let's add failure to our fake call first, so there's something real to handle:

```python
import asyncio
import random


class RetriableError(Exception):
    """A failure that might succeed if we just try again."""
    pass


async def call_llm(question):
    await asyncio.sleep(0.2)
    if random.random() < 0.3:              # fails 30% of the time
        raise RetriableError("server busy (simulated)")
    return f"answer to {question}"
```

Now the retry wrapper:

```python
async def call_with_retry(question, max_attempts=4):
    for attempt in range(1, max_attempts + 1):
        try:
            return await call_llm(question)
        except RetriableError as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                raise                       # out of chances, give up
    # unreachable, but makes the function's contract obvious


async def main():
    answer = await call_with_retry("what is 2+2?")
    print(answer)


asyncio.run(main())
```

Sample output:

```
  attempt 1 failed: server busy (simulated)
  attempt 2 failed: server busy (simulated)
answer to what is 2+2?
```

### Why the custom exception class matters

Notice we only catch `RetriableError`, not `Exception`. That's deliberate, and it's a distinction you'll use constantly:

| Kind of failure | Example | Retry? |
|---|---|---|
| **Retriable** | Timeout, "server busy" (HTTP 429), server error (HTTP 500) | Yes — the same request might work in a second |
| **Not retriable** | "Your request is malformed" (HTTP 400), "invalid API key" (401) | No — it'll fail identically every time |

Retrying a non-retriable error wastes time and money and hides the real bug. If you catch bare `Exception` here, a typo in your request gets retried 4 times before failing with a confusing message.

**Try breaking it:** Change `except RetriableError` to `except Exception`, then make `call_llm` raise `ValueError("bad request")` instead. Watch it pointlessly retry a bug 4 times.

**Check yourself before Stage 5:** Give an example of an error you should retry and one you shouldn't. Why does the difference matter?

---

## Stage 5 — Backoff: don't hammer a struggling server

There's a problem with the Stage 4 retry. Look at the timing: it retries *immediately*.

If the server said "I'm overloaded," and you instantly ask again — and so does every other client doing the same thing — you make the overload worse. Everyone retries at once, the server falls over harder, everyone retries again. This is called a **thundering herd**, and it turns a small hiccup into an outage.

The fix is to wait longer between each attempt:

```python
async def call_with_retry(question, max_attempts=4):
    for attempt in range(1, max_attempts + 1):
        try:
            return await call_llm(question)
        except RetriableError as e:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt                    # 2s, 4s, 8s
            print(f"  attempt {attempt} failed, waiting {wait}s")
            await asyncio.sleep(wait)
```

`2 ** attempt` doubles the wait each time: 2, 4, 8, 16 seconds. This is **exponential backoff**. Each failure makes you back off harder, giving the server room to recover.

### One more fix: jitter

Exponential backoff alone still has a flaw. If 100 clients all fail at the same moment, they all wait exactly 2 seconds, then all retry at exactly the same moment. You've made the herd slower, not smaller.

So you add a little randomness:

```python
import random

wait = (2 ** attempt) + random.random()    # 2.4s, 4.9s, 8.1s — all different
```

That random fraction of a second is called **jitter**. It spreads the retries out so they don't land together.

### Final version with a ceiling

```python
wait = min(2 ** attempt, 30) + random.random()
```

The `min(..., 30)` caps the wait so attempt 10 doesn't wait 17 minutes.

> **The complete rule:** exponential backoff + jitter + a maximum cap. You will write this same pattern for LLM calls, tool calls, database calls, and every external service in this curriculum.

**Check yourself before Stage 6:** What's a thundering herd? Why isn't exponential backoff alone enough to prevent it?

---

## Stage 6 — Limiting how many run at once

We fixed hammering the server on *retries*. But look back at Stage 3 — we fired 10 calls at once. What if it were 10,000?

You'd instantly hit your API rate limit, exhaust your machine's network connections, and possibly get your API key temporarily suspended. "Run everything at once" is as wrong as "run everything one at a time."

We want a middle ground: **at most N at a time.** The tool is `asyncio.Semaphore`.

A semaphore is a counter with N slots. To proceed, a coroutine takes a slot. If no slots are free, it waits until someone releases one.

```python
import asyncio


async def run_all(questions, concurrency=5):
    sem = asyncio.Semaphore(concurrency)     # 5 slots

    async def one_call(q):
        async with sem:                       # take a slot (wait if none free)
            return await call_with_retry(q)
        # slot is automatically released here, even if an error was raised

    return await asyncio.gather(*(one_call(q) for q in questions))
```

The `async with sem:` line is the whole mechanism. It takes a slot on the way in and gives it back on the way out — including if the code inside raises an exception, which is exactly the behavior you want. (That automatic cleanup is what `with` blocks are for in Python generally. Here it guarantees a crashed call can't permanently leak a slot.)

### See it work

```python
import asyncio
import time


async def slow_task(n):
    await asyncio.sleep(1)
    print(f"  finished task {n} at {time.time() - START:.1f}s")


async def main():
    global START
    START = time.time()
    sem = asyncio.Semaphore(3)

    async def bounded(n):
        async with sem:
            await slow_task(n)

    await asyncio.gather(*(bounded(i) for i in range(9)))


asyncio.run(main())
```

Output:

```
  finished task 0 at 1.0s
  finished task 1 at 1.0s
  finished task 2 at 1.0s      <- first 3 together
  finished task 3 at 2.0s
  finished task 4 at 2.0s
  finished task 5 at 2.0s      <- next 3
  finished task 6 at 3.0s
  finished task 7 at 3.0s
  finished task 8 at 3.0s      <- last 3
```

Nine tasks, three at a time, three seconds. Exactly what we asked for.

**Check yourself before Stage 7:** Why is "run everything at once" a problem? What does `async with sem:` do on entry and exit?

---

## Stage 7 — Putting it all together

Here is everything from Stages 2–6 in one working program. This is the file you'll copy the shape of for the rest of the curriculum.

```python
import asyncio
import random
import time
from dataclasses import dataclass


class RetriableError(Exception):
    """A failure that might succeed on another try."""


@dataclass
class CallResult:
    prompt: str
    text: str
    latency_s: float
    attempts: int


async def fake_llm_call(prompt: str) -> str:
    """Stands in for a real HTTP call to an LLM provider."""
    await asyncio.sleep(0.2 + random.random() * 0.3)
    if random.random() < 0.25:
        raise RetriableError("429 rate limited (simulated)")
    return f"response to: {prompt}"


async def call_with_retry(prompt: str, max_attempts: int = 4) -> CallResult:
    start = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        try:
            text = await fake_llm_call(prompt)
            return CallResult(
                prompt=prompt,
                text=text,
                latency_s=time.monotonic() - start,
                attempts=attempt,
            )
        except RetriableError:
            if attempt == max_attempts:
                raise
            await asyncio.sleep(min(2 ** attempt, 30) + random.random())
    raise RuntimeError("unreachable")


async def run_batch(prompts: list[str], concurrency: int = 5) -> list[CallResult]:
    sem = asyncio.Semaphore(concurrency)

    async def bound_call(p: str) -> CallResult:
        async with sem:
            return await call_with_retry(p)

    return await asyncio.gather(*(bound_call(p) for p in prompts))


async def main() -> None:
    prompts = [f"task-{i}" for i in range(10)]
    start = time.monotonic()
    results = await run_batch(prompts, concurrency=3)

    for r in results:
        print(f"{r.prompt}: {r.latency_s:.2f}s, {r.attempts} attempt(s)")
    print(f"\ntotal: {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

### Two small things worth noticing

**`time.monotonic()` instead of `time.time()`.** `time.time()` returns wall-clock time, which can jump backwards if the system clock syncs mid-measurement. `time.monotonic()` only ever moves forward. Always use it for measuring durations.

**`@dataclass`.** This decorator writes the boring `__init__` and `__repr__` for you. `CallResult(prompt=..., text=..., ...)` just works, and printing one shows all its fields. You'll see dataclasses everywhere in this curriculum for small data containers like this.

**Check yourself before Stage 8:** Without looking at the file, write `run_batch` from memory. Then compare.

---

## Stage 8 — The mental model to carry forward

Now step back from the code. Here's the way to think about every agent program you'll write.

Agent code has three layers:

```
┌─────────────────────────────────────────────┐
│  1. YOUR CODE (deterministic)               │
│     Loops, if-statements, parsing.          │
│     Same input → same output. Testable.     │
├─────────────────────────────────────────────┤
│  2. THE BOUNDARY (where you do the work)    │
│     Validate. Retry. Handle errors.         │
│     Turn unreliable output into trusted data│
├─────────────────────────────────────────────┤
│  3. THE MODEL (unpredictable)               │
│     Same input → possibly different output. │
│     Can fail, be slow, or return nonsense.  │
└─────────────────────────────────────────────┘
```

Layer 3 is not like calling a normal function. A normal function either returns what it promised or raises an error. An LLM can return something that *looks* right, parses fine, and is completely wrong. It can also just be slow, or rate-limited, or down.

Layer 2 is where you absorb all of that. Everything you built in Stages 4–6 — retries, backoff, concurrency limits — lives here. Later you'll add schema validation, timeouts, and permission checks to the same layer.

> **Remember this:** Almost every agent bug you will ever debug lives in layer 2. Not in the model, not in your business logic — in the code that's supposed to catch the model misbehaving and doesn't.

---

## Stage 9 — The systems around your code

Agent code doesn't run alone. Here's what surrounds it and why. You don't need to master these now — you need to know what each one is *for*, so later files make sense.

| Thing | What it is | Why agents need it | Covered in |
|---|---|---|---|
| **Docker** | Runs software in an isolated box | Start a database in one command; safely run code an LLM wrote | [Part XLVIII](part48-deployment.md) |
| **PostgreSQL** | A relational database (tables, rows, SQL) | Stores agent progress so a crash doesn't lose work | [Part IX](part09-agent-state.md), [XLII](part42-databases-for-agents.md) |
| **Redis** | Very fast in-memory key-value store | Caching, rate limit counters, locks between processes | [Part IX](part09-agent-state.md), [XLI](part41-performance-engineering.md) |
| **HTTP / REST** | How programs talk over the web | Every LLM API and most tools are HTTP | [Part IV](part04-llm-application-engineering.md) |
| **Git** | Tracks changes to code | Agents that edit code produce diffs you must review | [Part XLIV](part44-code-agents.md) |
| **Message queue** | A durable to-do list between programs | Lets many workers share agent tasks | [Part XLIX](part49-distributed-agents.md) |

### Get Postgres and Redis running now

You'll need both starting in Part IX. Two commands:

```bash
docker run -d --name agent-pg -e POSTGRES_PASSWORD=devpass -p 5432:5432 postgres:16
```

```bash
docker run -d --name agent-redis -p 6379:6379 redis:7
```

Check they're alive:

```bash
docker ps
```

You should see both listed as `Up`. If Docker isn't installed, install Docker Desktop first.

### One idea worth absorbing early

Even a "simple" agent talks to an LLM API, a database, and some tools. The moment your program depends on things across a network, you inherit the hard parts of distributed systems:

- Any call can fail, hang, or *succeed while appearing to fail* (the response gets lost on the way back)
- Multi-step operations can finish halfway
- Two copies of your program can trip over each other

Treat agent code with the seriousness you'd give a payment service, not a script. Timeouts everywhere. Assume any step can be interrupted.

---

# The Project: Build a Resilient Batch Caller

Everything above was practice. This is the thing you build and keep — later parts extend it rather than starting over.

## What you're building

A command-line program that takes a list of prompts, sends them all to an LLM API, and prints the results plus statistics.

Concretely:

```bash
$ python batch_caller.py prompts.txt --concurrency 5
```

```
[1/10] ✓ "Summarize photosynthesis"         1.82s  1 attempt   142 tokens
[2/10] ✓ "What is the capital of France?"   0.94s  1 attempt    18 tokens
[3/10] ✗ "Explain quantum tunneling"        —      4 attempts  gave up: rate limited
...

Done: 9 succeeded, 1 failed in 4.21s
Average latency: 1.34s   p95 latency: 2.90s
Total tokens: 1,204      Estimated cost: $0.0031
Max concurrent calls observed: 5
```

## Why this project

It forces you to use every stage above for real, against a real API that really fails sometimes. And the `run_batch` shape you end up with is reused in Part IV (multi-provider client), Part VI (parallel tool calls), and Part XLI (performance benchmarks).

## Before you start

- You need an API key from an LLM provider, in an environment variable — **never typed into your code**
- Install: `pip install httpx`
- Have a `prompts.txt` with one prompt per line, ~10 lines

## Build it in stages

Do these in order. Each one should fully work before you start the next. Resist the urge to write the whole thing at once — that's how you end up with a bug you can't locate.

---

### Stage A — One real call

**Goal:** Send one prompt to a real LLM API and print the answer.

**Do:**
1. Read your API key from an environment variable with `os.environ`
2. Use `httpx.AsyncClient` to POST to your provider's chat endpoint
3. Print the response text

**You're done when:** You run it and see a real model's answer in your terminal.

**Watch out for:** If you get a 401, your key isn't reaching the code — print `bool(os.environ.get("YOUR_KEY_NAME"))` to check it's actually set in *this* shell.

---

### Stage B — Many calls, one at a time

**Goal:** Read `prompts.txt`, loop over it, call each prompt, print each answer. Sequential, no async concurrency yet.

**Do:**
1. Read the file, split into lines, drop blank lines
2. Loop and call Stage A's function for each
3. Print total elapsed time at the end

**You're done when:** 10 prompts produce 10 answers, and your printed total time is roughly `10 × single-call-time`.

**Write that number down.** It's your baseline for Stage D.

---

### Stage C — Retry with backoff

**Goal:** A failing call retries instead of crashing the program.

**Do:**
1. Define a `RetriableError` exception
2. Inspect the HTTP status code. Raise `RetriableError` for 429 and 5xx. For 4xx (other than 429), raise a *different*, non-retriable error
3. Wrap the call in the Stage 5 retry loop: exponential backoff + jitter + cap
4. Track how many attempts each call took

**You're done when:** You can force a failure and see it recover. To force one: point the URL at a nonexistent path to get a 404 (should fail immediately, no retries), then temporarily make your code raise `RetriableError` on the first two attempts artificially (should retry, then succeed).

**Watch out for:** A 400 "bad request" retried 4 times means your retriable/non-retriable split is wrong.

---

### Stage D — Concurrency with a limit

**Goal:** All prompts run at once, capped at N, and it's dramatically faster.

**Do:**
1. Add `asyncio.Semaphore(concurrency)`
2. Wrap each call in `async with sem:`
3. Use `asyncio.gather` to launch them all
4. Reuse **one** `httpx.AsyncClient` across all calls — create it once in `main`, pass it in. Creating one per call re-does the TLS handshake every time and is much slower

**You're done when:** Your total time is close to `(number of prompts ÷ concurrency) × single-call-time`, and much lower than your Stage B baseline. Compare against that number you wrote down.

**Watch out for:** If it's no faster, you've probably got a blocking call in there (`time.sleep`, `requests`) — that's the Stage 2 trap.

---

### Stage E — Statistics

**Goal:** The summary block from the example output.

**Do:**
1. Record per call: latency (with `time.monotonic()`), attempt count, token usage from the API response, success/failure
2. Compute: success/failure counts, average latency, p95 latency, total tokens
3. Estimate cost using your provider's published per-token prices
4. Print the summary

**How to compute p95:** Sort the latencies, then take the value at index `int(len(sorted_latencies) * 0.95)`. p95 means "95% of calls were at least this fast." It matters because averages hide the slow tail — an average of 1s can hide 5% of users waiting 8 seconds.

**You're done when:** Your output matches the example format, and re-running gives slightly different but sensible numbers.

---

### Stage F — Don't lose work on interruption

**Goal:** Pressing Ctrl+C mid-run doesn't silently abandon in-flight calls.

**Do:**
1. Wrap `asyncio.run(main())` in `try/except KeyboardInterrupt`
2. On interrupt, print how many calls had completed
3. Write completed results to a JSON file so a long run isn't wasted

**You're done when:** You start a 50-prompt run, hit Ctrl+C partway, and see a clean message plus a JSON file with the finished results.

**Watch out for:** This is your first taste of a big theme — [Part IX](part09-agent-state.md) makes "don't lose work when interrupted" a central concern.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| No speedup from concurrency | A blocking call inside async code | Search for `time.sleep`, `requests.`, `open()` on big files |
| `<coroutine object ...>` printed | Forgot `await` | Add it |
| Works for 5 prompts, fails at 100 | Concurrency limit too high, hitting rate limits | Lower it; check that 429s are retried |
| Retries on a typo'd request | Catching bare `Exception` | Split retriable vs. non-retriable |
| Timings are wild / occasionally negative | Using `time.time()` | Use `time.monotonic()` |
| Slow even with concurrency | New HTTP client per call | Create one client, reuse it |

## Going further (optional)

When Stages A–F all work, try these:

1. **Clean cancellation.** On Ctrl+C, cancel in-flight tasks properly and verify none are left running:

   ```python
   async def cancel_all(tasks: set[asyncio.Task]) -> None:
       for t in tasks:
           t.cancel()
       await asyncio.gather(*tasks, return_exceptions=True)
       assert all(t.cancelled() or t.done() for t in tasks)
   ```

2. **Find your optimal concurrency.** Run with concurrency 1, 2, 5, 10, 20, 50. Plot total time. You'll find a point where more concurrency stops helping (you're rate-limited) and eventually hurts (retry storms). Note that number — it's a property of your API tier, and you'll want it again.

3. **Adaptive concurrency.** Automatically reduce concurrency when you see 429s, and slowly increase it when calls succeed. This is a simplified version of what production API clients do.

---

## Checkpoint — do not skip

Answer these from memory. If you can't, reread that stage. Later parts assume all of this is automatic for you.

**Knowledge:**
1. Why does async help agent code but not a program that's crunching numbers?
2. What exactly does the event loop do?
3. Why does `time.sleep()` inside async code break things, with no error message?
4. Name a retriable error and a non-retriable one. Why does confusing them cause real problems?
5. What is jitter, and what goes wrong without it?
6. Why is p95 latency more useful than average latency?

**Implementation:** Write `run_batch` — semaphore, gather, retry with backoff — on a blank page, no reference.

**Debugging:** Your agent program hangs. How do you determine whether it's (a) a blocked event loop, (b) a genuinely slow API call, or (c) a deadlock? What would you print or measure to distinguish them?

**Architecture:** You have 5 agent programs sharing one Postgres and one Redis. Each can have 20 concurrent operations. Postgres allows 100 connections total. Sketch how you'd prevent them from exhausting the connection limit.

---

**Next:** [Part II — Machine Learning Foundations](part02-ml-foundations.md), where you'll build the math behind every search and retrieval system in this curriculum.
