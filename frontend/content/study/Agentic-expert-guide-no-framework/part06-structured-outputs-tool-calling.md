# Part VI — Structured Outputs and Tool Calling

## Before you start

**What you need already:** [Part IV](part04-llm-application-engineering.md) (a working client, Pydantic validation), [Part V](part05-prompt-engineering.md) (clear instructions, the untrusted-content pattern).

**What this part gives you:** The single most important mechanism in this entire curriculum. Everything from here on — every agent, every RAG system, every multi-agent team — is built on the model being able to *do things*, not just generate text. This part is where that becomes real.

**Time:** 4–5 days.

**When you finish, you'll be able to:** Give a model a set of real capabilities (calculator, file access, web search), have it choose and use them correctly, and handle every way that can go wrong — because it will go wrong, regularly, and your code needs to survive that.

---

## Stage 1 — The problem tool calling solves

Ask an LLM "what's 847 × 293?" without any help, and it might get it right — or it might not. Not because it's bad at math conceptually, but because of what you learned in [Part III](part03-llm-fundamentals.md): it's predicting likely-looking tokens, not running an arithmetic algorithm. Big multiplications are exactly the kind of thing that's easy to get subtly wrong this way.

Now ask it "what's the weather in Tokyo right now?" It has no way to know — its knowledge is frozen at training time ([Part III Stage 8](part03-llm-fundamentals.md)), and weather changes by the hour.

Both problems have the same shape: **there's a real answer, obtainable by a normal, reliable, deterministic method (a calculator, a weather API) — and the model should use that method instead of guessing.**

Tool calling is the mechanism that lets it. The model doesn't compute 847 × 293 itself; it says "please run `calculator(847 * 293)` and tell me the result." Your code runs the actual calculation. The model just decides *when* to ask for it.

**Check yourself before Stage 2:** Why is "the model should compute 847 × 293 itself" the wrong approach, given what you know about how the model generates text?

---

## Stage 2 — What a tool call actually looks like

Strip away the framework magic. Here's the real mechanism, in three steps.

**Step 1: you describe the tool.** Not code — a description of what it does and what arguments it takes, sent alongside your prompt:

```json
{
  "name": "calculator",
  "description": "Evaluate a simple arithmetic expression.",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {"type": "string", "description": "e.g. '847 * 293'"}
    },
    "required": ["expression"]
  }
}
```

That block, `parameters`, is a **JSON Schema** — a way to describe the shape of data. This is the same idea as a Pydantic model, just expressed as JSON instead of Python. In fact, you'll almost never write JSON Schema by hand — Pydantic generates it for you, which Stage 4 shows.

**Step 2: the model responds with a request to call it**, instead of plain text:

```json
{
  "name": "calculator",
  "arguments": {"expression": "847 * 293"}
}
```

**Step 3: your code runs the real function, and sends the result back**, so the model can use it in its final answer:

```
"The result of the calculator call was: 248171"
```

The model then writes something like "847 × 293 is 248,171" — using the *real* computed number, not a guessed one.

**Check yourself before Stage 3:** What are the three steps? Which step is your code's responsibility, and which is the model's?

---

## Stage 3 — Build it by hand, once, before using a shortcut

You're about to see Pydantic generate the schema automatically. Before that, build one call by hand so the mechanism isn't a mystery.

```python
def calculator(expression: str) -> float:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ValueError(f"disallowed characters in expression: {expression!r}")
    return eval(expression, {"__builtins__": {}}, {})


TOOL_SCHEMA = {
    "name": "calculator",
    "description": "Evaluate a simple arithmetic expression.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"],
    },
}


async def ask_with_tool(client, prompt: str) -> str:
    # A real provider call would include TOOL_SCHEMA in the request and the
    # provider would return a structured tool_call instead of plain text
    # when it decides to use it. Here we simulate that decision to see the
    # shape clearly first.
    model_wants_to_call = {"name": "calculator", "arguments": {"expression": "847 * 293"}}

    if model_wants_to_call["name"] == "calculator":
        result = calculator(**model_wants_to_call["arguments"])
        return f"The calculator returned: {result}"
    return "model answered without a tool"


import asyncio
print(asyncio.run(ask_with_tool(None, "What is 847 times 293?")))
```

```
The calculator returned: 248171
```

### A dangerous line to notice

`eval(expression, ...)` runs arbitrary Python. That's terrifying if `expression` could be anything — `eval("__import__('os').system('rm -rf /')")` would happily destroy your filesystem. The `allowed_chars` check is doing real, load-bearing work: it guarantees the string can *only* contain digits, basic operators, parentheses, and spaces before it ever reaches `eval`. This is your first taste of a theme that gets serious in [Part XXXV](part35-security.md): **never trust what a model hands you as an argument. Validate it before you act on it, every time.**

**Try breaking it:** Remove the `allowed_chars` check and call `calculator("__import__('os').listdir('.')")`. Watch it work — meaning watch it run arbitrary code. Put the check back.

**Check yourself before Stage 4:** What does `allowed_chars` protect against? What would happen if you removed it and a malicious or confused model passed a Python expression instead of arithmetic?

---

## Stage 4 — Doing it properly with Pydantic

Hand-writing JSON Schema is tedious and error-prone. Pydantic — which you already used in [Part IV](part04-llm-application-engineering.md) for validating *responses* — does the same job for tool *arguments*, in both directions: it generates the schema to send, and validates what comes back.

```python
from pydantic import BaseModel, Field, ValidationError


class CalculatorArgs(BaseModel):
    expression: str = Field(description="A simple arithmetic expression, e.g. '2 + 2 * 3'")


def calculator_tool(args: CalculatorArgs) -> float:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(args.expression) <= allowed_chars:
        raise ValueError(f"disallowed characters: {args.expression!r}")
    return eval(args.expression, {"__builtins__": {}}, {})


# Pydantic generates the JSON Schema for you:
print(CalculatorArgs.model_json_schema())
```

```python
{
    'description': 'A simple arithmetic expression...',
    'properties': {'expression': {'title': 'Expression', 'type': 'string', 'description': '...'}},
    'required': ['expression'],
    'title': 'CalculatorArgs',
    'type': 'object',
}
```

That's the exact shape you hand-wrote in Stage 3, generated automatically from a class definition. Add a field, change a type — the schema updates itself. This is also why using Pydantic here matters beyond convenience: your argument *validation* and your argument *schema* can never drift out of sync, because they're the same class.

Now handle the model's response, validating before running anything:

```python
def execute_tool_call(name: str, raw_args: dict) -> dict:
    if name != "calculator":
        return {"error": f"unknown tool: {name}"}

    try:
        parsed = CalculatorArgs.model_validate(raw_args)
    except ValidationError as e:
        return {"error": f"invalid arguments: {e}"}

    try:
        result = calculator_tool(parsed)
    except ValueError as e:
        return {"error": str(e)}

    return {"result": result}


print(execute_tool_call("calculator", {"expression": "847 * 293"}))
# {'result': 248171}

print(execute_tool_call("calculator", {"expression": "rm -rf /"}))
# {'error': 'disallowed characters: ...'}

print(execute_tool_call("calculator", {"wrong_field": "oops"}))
# {'error': 'invalid arguments: ...'}
```

### Notice what every failure path has in common

None of them raise an exception that would crash your program. They all return `{"error": "..."}`. This is deliberate, and it's the subject of the next stage.

**Check yourself before Stage 5:** What does `CalculatorArgs.model_json_schema()` give you, and why does generating it from the same class you validate against matter?

---

## Stage 5 — Why errors go back to the model instead of crashing your program

Here's a design decision that looks strange until you think it through: when a tool call fails, you don't raise an exception up through your program. You package the failure as data and hand it *back to the model*, as if it were just another piece of information.

### Why not just crash on a bad call?

Because the model can often *recover*. If it called `calculator(expression="6 + ")` — a malformed expression — and you crash your whole program, the conversation is over and the user gets nothing. If instead you return `{"error": "invalid expression: unexpected end of input"}`, the model sees that error, understands what went wrong, and can often just... try again with a corrected expression. You're not the one fixing the mistake. The model is, because you gave it the information to.

```python
async def agent_step_with_tool_recovery(client, user_question: str, max_tool_attempts=3):
    conversation = [{"role": "user", "content": user_question}]

    for attempt in range(max_tool_attempts):
        # In a real call, the provider returns either plain text or a tool_call.
        # Simulated here: the model gets a malformed expression on attempt 0.
        if attempt == 0:
            decision = {"type": "tool_call", "name": "calculator",
                        "arguments": {"expression": "847 *"}}
        elif attempt == 1:
            decision = {"type": "tool_call", "name": "calculator",
                        "arguments": {"expression": "847 * 293"}}
        else:
            decision = {"type": "final", "content": "unreachable in this demo"}

        if decision["type"] == "tool_call":
            outcome = execute_tool_call(decision["name"], decision["arguments"])
            conversation.append({"role": "tool", "content": outcome})

            if "error" in outcome:
                print(f"attempt {attempt}: tool error, feeding back to model: {outcome['error']}")
                continue   # model gets another chance, informed by the error
            else:
                print(f"attempt {attempt}: success: {outcome['result']}")
                return f"The answer is {outcome['result']}"

    return "couldn't complete the calculation after retries"


import asyncio
print(asyncio.run(agent_step_with_tool_recovery(None, "What is 847 * 293?")))
```

```
attempt 0: tool error, feeding back to model: invalid expression: unexpected end of input
attempt 1: success: 248171
The answer is 248171
```

This mirrors the retry pattern from [Part I](part01-prerequisites.md) almost exactly — bounded attempts, informative failure — except the thing "retrying" isn't your code blindly repeating the same call. It's the model, using the error message to try something *different*. That only works because you handed it real information instead of just a crash.

**Check yourself before Stage 6:** Why does returning `{"error": ...}` instead of raising an exception let the model recover from its own mistake? What would happen to the conversation if you crashed the program on the first bad call instead?

---

## Stage 6 — Tools that can hang, and tools you shouldn't overuse

Two more real-world problems before you build the full loop.

### Problem 1: a tool that never comes back

A calculator finishes instantly. A web search tool might hang for 30 seconds, or forever, if the remote service is having a bad day. Without a limit, one slow tool call freezes your entire agent.

The fix is exactly the timeout pattern from [Part I](part01-prerequisites.md), applied to tool execution:

```python
import asyncio


async def execute_with_timeout(fn, args: dict, timeout_s: float = 5.0) -> dict:
    try:
        result = await asyncio.wait_for(asyncio.to_thread(fn, args), timeout=timeout_s)
        return {"result": result}
    except asyncio.TimeoutError:
        return {"error": f"tool timed out after {timeout_s}s"}
```

`asyncio.to_thread` matters here: most real tool functions (a database query, a file read) are ordinary synchronous Python, not `async def`. Running one directly inside your async agent loop would block the entire event loop — the exact trap from [Part I Stage 2](part01-prerequisites.md). `to_thread` runs it in a separate thread instead, so a slow tool blocks only itself, and `asyncio.wait_for` can still cut it off on schedule.

### Problem 2: a tool that keeps failing

If a tool has failed 5 times in the last minute, calling it a 6th time is probably pointless — the underlying service is likely down. Retrying anyway wastes time and can make an already-struggling service worse (the thundering herd idea from [Part I Stage 5](part01-prerequisites.md), but now it's *your* agent hammering a broken dependency instead of a server hammering back).

The fix: after N consecutive failures, stop calling that tool for a cooldown period.

```python
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_s: float = 30.0
    _consecutive_failures: int = field(default=0, init=False)
    _open_until: float = field(default=0.0, init=False)

    def is_open(self) -> bool:
        """True means: don't even try, we're in cooldown."""
        return time.monotonic() < self._open_until

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.cooldown_s
            print(f"circuit breaker OPEN for {self.cooldown_s}s "
                  f"after {self._consecutive_failures} consecutive failures")
```

The name "circuit breaker" is a literal electrical metaphor: a household circuit breaker trips and cuts power after detecting a fault, rather than letting the wiring keep drawing current and catch fire. Same idea here — trip after repeated failure, stop drawing on a broken dependency, let it cool down.

**Check yourself before Stage 7:** Why does a hanging tool call threaten the *entire* agent, not just that one call? Why keep retrying a tool that's failed 5 times in a row make things worse, not better?

---

## Stage 7 — The full loop: multiple tools, one model choosing between them

Everything comes together now. A registry of tools, a model that picks which one to use (if any), and execution wrapped in everything you just built.

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel, ValidationError


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def available_schemas(self) -> list[dict]:
        # Skip tools whose circuit breaker is open — don't even offer them
        return [t.to_schema() for t in self.tools.values() if not t.breaker.is_open()]

    async def call(self, name: str, raw_args: dict, timeout_s: float = 5.0) -> dict:
        spec = self.tools.get(name)
        if spec is None:
            return {"error": f"unknown tool: {name!r}"}

        if spec.breaker.is_open():
            return {"error": f"{name} is temporarily unavailable (circuit breaker open)"}

        try:
            parsed_args = spec.args_model.model_validate(raw_args)
        except ValidationError as e:
            return {"error": f"invalid arguments for {name}: {e}"}

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(spec.fn, parsed_args), timeout=timeout_s
            )
            spec.breaker.record_success()
            return {"result": result}
        except asyncio.TimeoutError:
            spec.breaker.record_failure()
            return {"error": f"{name} timed out after {timeout_s}s"}
        except Exception as e:
            spec.breaker.record_failure()
            return {"error": f"{name} failed: {e}"}
```

Register two tools:

```python
class CalculatorArgs(BaseModel):
    expression: str


class WeatherArgs(BaseModel):
    city: str


def calculator_fn(args: CalculatorArgs) -> float:
    allowed = set("0123456789+-*/(). ")
    if not set(args.expression) <= allowed:
        raise ValueError("disallowed characters")
    return eval(args.expression, {"__builtins__": {}}, {})


def weather_fn(args: WeatherArgs) -> str:
    # stand-in for a real HTTP call to a weather API
    return f"72°F and sunny in {args.city}"


registry = ToolRegistry()
registry.register(ToolSpec("calculator", "Evaluate arithmetic expressions.",
                            CalculatorArgs, calculator_fn))
registry.register(ToolSpec("weather", "Get current weather for a city.",
                            WeatherArgs, weather_fn))
```

And the loop that ties it to a real (or simulated) model:

```python
async def run_agent_turn(registry: ToolRegistry, user_question: str, max_steps: int = 4) -> str:
    print(f"available tools: {[s['name'] for s in registry.available_schemas()]}")

    # A real call would send registry.available_schemas() to the model and
    # get back either a tool_call or a final answer. Simulated for clarity:
    model_decision = {"type": "tool_call", "name": "weather", "arguments": {"city": "Tokyo"}}

    for step in range(max_steps):
        if model_decision["type"] == "final":
            return model_decision["content"]

        outcome = await registry.call(model_decision["name"], model_decision["arguments"])
        print(f"step {step}: called {model_decision['name']} -> {outcome}")

        if "error" in outcome:
            model_decision = {"type": "final", "content": f"couldn't complete: {outcome['error']}"}
        else:
            model_decision = {"type": "final", "content": f"It's {outcome['result']}"}

    return "reached max steps without a final answer"


print(asyncio.run(run_agent_turn(registry, "What's the weather in Tokyo?")))
```

```
available tools: ['calculator', 'weather']
step 0: called weather -> {'result': '72°F and sunny in Tokyo'}
It's 72°F and sunny in Tokyo
```

Every safety mechanism from Stages 3–6 is present: validated arguments, a timeout, a circuit breaker, and structured errors fed back rather than crashes. This exact shape — a registry, schema-validated calls, timeout + circuit breaker wrapping — is what you'll extend into a full agent loop in [Part VII](part07-what-is-an-agent.md), and it's what real MCP tool servers implement underneath their protocol in [Part XXIII](part23-mcp.md).

**Check yourself before the project:** Trace `registry.call()` for a tool name that doesn't exist, for valid arguments that time out, and for a tool whose breaker is open. What does each path return, and why does none of them raise an exception?

---

# The Project: A Multi-Tool Agent That Survives Failure

## What you're building

A command-line agent with three real tools that a real LLM chooses between, with every failure mode from this part handled — not simulated, actually tested against real failures.

```bash
$ python tool_agent.py "What's 15% of 2,847, and what's the weather in Paris?"
```

```
[tool: calculator] expression="2847 * 0.15" -> 427.05
[tool: weather] city="Paris" -> 68°F, partly cloudy

15% of 2,847 is 427.05. The current weather in Paris is 68°F and partly cloudy.
```

## Why this project

This is the foundation every later agent builds on. The reliability patterns you implement here — validation, timeouts, circuit breakers, structured errors — don't change as agents get more sophisticated; they get *reused*. Get them right now.

## Build it in stages

---

### Stage A — One real tool, one real model call

**Goal:** A real LLM sees a real tool schema, decides to call it, and your code executes it correctly.

**Do:**
1. Implement `calculator` with Pydantic args, as in Stage 4
2. Send its schema to a real provider (check your provider's tool-calling API docs for the exact request format — it varies by provider)
3. Parse the model's tool-call response and execute it
4. Feed the result back and get the model's final answer

**You're done when:** You ask a real arithmetic question through the model and get the *actual* computed answer in the final response, not a guessed one. Verify by asking something the model would likely get wrong unaided (a large multiplication) and confirming the tool result is exact.

---

### Stage B — Three tools, model chooses

**Goal:** Add `weather` (mocked HTTP is fine) and a third tool of your choice (a unit converter, a word counter — anything simple). The model picks the right one per question.

**Do:**
1. Build the `ToolRegistry` from Stage 7
2. Register all three tools
3. Ask a question that clearly needs one specific tool; confirm the model picks correctly
4. Ask a question needing two tools in one turn (as in the example output above); confirm both get called

**You're done when:** Three different single-tool questions each trigger the correct tool, and the combined question triggers both.

**Watch out for:** If the model calls the wrong tool, check your tool `description` fields first — vague descriptions are the most common cause, echoing [Part V Stage 2](part05-prompt-engineering.md)'s lesson about specificity.

---

### Stage C — Force real failures, verify real recovery

**Goal:** Not simulate failure — actually cause it, and watch your code handle it.

**Do:**
1. **Malformed arguments:** temporarily make `WeatherArgs` require a field the model won't naturally provide (e.g., `country_code: str` with no default), and confirm you get a validation error fed back, not a crash
2. **Timeout:** add `await asyncio.sleep(10)` inside `weather_fn` temporarily, set your timeout to 2 seconds, confirm you get a timeout error, not a hang
3. **Circuit breaker:** make `weather_fn` always raise an exception, call it 4 times in a row, confirm the breaker opens on the 3rd and the 4th call is rejected immediately without even trying

**You're done when:** You've watched all three failure types happen for real and confirmed your program kept running and gave a sensible response each time — including the case where the model just needs to fall back to answering without the failed tool.

**Revert your temporary breakages before moving on.**

---

### Stage D — Bounded recovery loop

**Goal:** When a tool call fails with a fixable problem (bad arguments), let the model see the error and retry — bounded, like Stage 5.

**Do:**
1. Wrap tool execution in a loop with `max_tool_attempts`
2. On an error, feed it back to the model as a tool result and let it try again
3. On success, or after exhausting attempts, produce a final answer

**You're done when:** You can force a malformed first attempt (e.g., ask an ambiguous question that leads the model to guess a slightly wrong argument format) and watch it self-correct on attempt 2, without you intervening.

---

### Stage E — Two tools at once, safely

**Goal:** When the model requests two independent tool calls in one turn, run them concurrently — reusing Part I's `asyncio.gather` — instead of one after another.

**Do:**
1. Detect when the model's response contains multiple tool calls
2. Run them with `asyncio.gather`, each through your existing `registry.call()` (which already has its own timeout)
3. Feed all results back together

**You're done when:** A question needing both `calculator` and `weather` completes in roughly the time of the *slower* tool, not the sum of both — measure it and confirm.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Model calls a tool with a nonexistent name | Hallucinated tool call — happens occasionally with any model | `registry.call` already returns `{"error": "unknown tool"}` — confirm your loop feeds this back rather than crashing |
| Arguments come back with wrong types (e.g., a number as a string) | Some providers are looser about type coercion than others | Pydantic will often coerce automatically; if not, this is a real validation failure — feed it back as an error, don't silently "fix" it yourself |
| Circuit breaker never opens even after repeated failures | `record_failure()` not being called on every error path | Check every `except` branch in `registry.call` calls it |
| Concurrent tool calls take as long as sequential ones | Blocking call inside a tool function without `asyncio.to_thread` | Revisit [Part I Stage 2](part01-prerequisites.md)'s blocking-call trap |

## Going further (optional)

1. **A fourth tool with real side effects** (e.g., writing to a local file). Notice this feels different from the read-only tools — you'll formalize *why* in [Part XXXIII](part33-tool-engineering.md), where side-effecting tools get a stricter contract.
2. **Tool call logging.** Record every tool call — name, arguments, result, duration — to a file. This is the seed of the tracing system you'll build properly in [Part XXXIX](part39-observability.md).
3. **Deliberately hostile arguments.** Try to get the calculator to execute something dangerous through creative expressions. If you succeed, tighten the validation. This is a small, hands-on preview of [Part XXXV](part35-security.md).

---

## Checkpoint

**Knowledge:**
1. Walk through the three steps of a tool call, start to finish.
2. Why does a failed tool call return `{"error": ...}` instead of raising an exception?
3. Why does a hanging tool call threaten more than just that one call, and what fixes it?
4. What does a circuit breaker do, and why does retrying a broken dependency make things worse, not better?

**Implementation:** Build `ToolRegistry.call` — validation, timeout, circuit breaker, structured error return — from memory.

**Debugging:** Your agent occasionally calls a tool with an argument type that doesn't match the schema (e.g., a number where a string was expected). Walk through exactly what happens in your code, step by step, from the moment that malformed call arrives to the moment the user sees a response.

**Architecture:** You're adding a 20th tool to an agent that already has 19. Some are read-only, some can delete data. Sketch how you'd organize the registry so a *new* tool automatically gets safe defaults (timeout, circuit breaker) without every tool author having to remember to add them.

---

**Next:** [Part VII — What Is an Agent?](part07-what-is-an-agent.md), where this tool-calling mechanism becomes part of a real loop that decides its own next steps.
