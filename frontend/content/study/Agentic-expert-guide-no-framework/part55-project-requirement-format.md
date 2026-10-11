# Part LV — Project Requirement Format

## What this page is for, in plain terms

This is a fill-in-the-blanks worksheet. The instinct when starting a project is to open an editor and start typing code — this template exists specifically to interrupt that instinct for anything bigger than a quick exercise. Spending 15 minutes filling this out *before* writing a line of code routinely saves hours of rework, because it forces you to notice a missing piece of the design (a failure mode you hadn't considered, a metric you have no way to measure) while it's still just a sentence to write, not code to rewrite.

**Use the full template for:** anything from Part LIII level 3 onward, and every capstone in Part LIV.
**Skip it for:** the quick warm-up exercises in Part LIII levels 1–2, where the scope is small enough to hold in your head without a worksheet.

A fully worked example is included below — read it once before filling out your own, so you can see what "good" looks like at each section (specific enough to be useful, short enough to actually write in 15 minutes).

## Template

```markdown
## Project name


## Objective
(One sentence: what does this system do and for whom?)

## Concepts learned
(List the specific curriculum Parts/mechanisms this project exercises.)

## Prerequisites
(What must already work before starting — prior projects, infrastructure.)

## Architecture
(Diagram — mermaid flowchart or sequence diagram — of the major components.)

## Component breakdown
(Table: component -> responsibility -> which Part's pattern it reuses.)

## Data flow
(What data moves between components, in what format, at each step.)

## State flow
(What state exists, where it lives, how it's checkpointed — Part IX.)

## Tool flow
(Which tools exist, their contracts — Part XXXIII — and permission requirements.)

## Expected behavior
(Concrete example: given input X, the system should produce output Y via steps Z.)

## Failure scenarios
(Table: failure -> detection -> handling -> prevention — Part LVII's format.)

## Testing strategy
(Which layers from Part XXXVII apply: unit/tool/integration/e2e/adversarial/chaos.)

## Evaluation metrics
(Which metrics from Part XVIII/XXXVIII apply, and target thresholds.)

## Security considerations
(Which Part XXXV threats apply: injection surface, permission scope, secrets.)

## Performance considerations
(Latency budget breakdown per Part XLI; where parallelization applies.)

## Cost considerations
(Estimated cost per run per Part XL; budget ceiling and enforcement point.)

## Suggested implementation stages
(Ordered list: what to build first, second, third.)

## Beginner version
(Minimal scope: mocked externals, no distribution, single happy path.)

## Intermediate version
(Real externals, basic error handling, single-node reliability.)

## Advanced version
(Full reliability/security/observability per production standards.)

## Expert extension
(One additional hard problem to solve after the advanced version works.)

## What this project teaches that tutorials usually miss
(The non-obvious lesson — usually a failure mode or trade-off, not a feature.)
```

## Practical Example — Filled Template for Project 19 (ReAct Agent)

```markdown
## Project name
ReAct Tool-Using Agent

## Objective
An agent that answers multi-step factual questions by interleaving reasoning and tool calls, with transparent step-by-step traces.

## Concepts learned
Part VII (agent loop), Part VIII §2 (ReAct), Part VI (tool calling), Part XXVIII (harness basics).

## Prerequisites
A working LLM client (Part IV) and at least one real tool (e.g., a search API).

## Architecture
(sequence diagram: User -> Agent Loop -> Model (Thought/Action) -> Tool -> Observation -> loop)

## Component breakdown
| Component | Responsibility | Reuses |
|---|---|---|
| Loop controller | Bounds steps, tracks termination | Part VII §5 |
| Tool registry | Validates/executes tool calls | Part VI §5 |
| Tracer | Logs each Thought/Action/Observation | Part XXXIX §3 |

## Data flow
User query -> prompt with tool schemas -> model emits Thought+Action -> tool executes -> Observation appended to history -> repeat.

## State flow
In-memory `AgentState.messages` list, checkpointed after each step per Part IX.

## Tool flow
One real tool (search), read-only, low risk (Part XXXIII), no approval gate needed.

## Expected behavior
Given "What is the population of the capital of France?", the agent should: (1) search for France's capital, (2) search for that city's population, (3) answer with both facts cited.

## Failure scenarios
| Failure | Detection | Handling | Prevention |
|---|---|---|---|
| Model never emits a final answer | max_steps reached | Return "terminated: max_steps" | Bounded loop (Part VII) |
| Tool timeout | asyncio.TimeoutError | Structured error fed back to model | Per-Part VI §7 |

## Testing strategy
Unit tests for the loop controller; integration test with a mocked search tool; property-based e2e test checking the final answer contains both expected facts.

## Evaluation metrics
Task success rate (Part XXXVIII) on 10 held-out multi-hop questions; average step count.

## Security considerations
Search results are untrusted content — wrap per Part XXXV §3 before re-injecting into context.

## Performance considerations
Single tool call per step, sequential (no parallelization opportunity here since each search depends on the prior result).

## Cost considerations
~3 model calls per run at low token count; no special budget concerns at this scope.

## Suggested implementation stages
1. Loop controller with a fake model. 2. Real model integration. 3. Real search tool. 4. Tracing. 5. Eval harness.

## Beginner version
Fake model and fake tool, verifying loop mechanics only.

## Intermediate version
Real model, real search tool, no tracing/eval yet.

## Advanced version
Full tracing, untrusted-content wrapping, bounded cost, eval harness with 10 test cases.

## Expert extension
Add self-consistency (Part V §6) across 3 independent runs and compare accuracy vs. single-run.

## What this project teaches that tutorials usually miss
That "ReAct" is not magic — it's a specific, bounded prompt-and-loop pattern that fails exactly like any other unbounded process if you skip the termination and error-feedback discipline built in Parts VII and VI.
```
