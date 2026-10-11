# DSA for Big-Tech Interviews — Instinct Over Memorization

Read in order. Every chapter follows the same template: **Core insight → Trigger/Anti-signals → Mental model → Templates → 3 Worked examples → Mistakes → Connections → Practice ladder → Recognition drill.**

## Table of Contents

| # | File | Topic |
|---|------|-------|
| 0 | [00-meta-framework.md](00-meta-framework.md) | Meta-framework: procedure, constraint table, **master recognition cheat sheet**, optimization levers |
| 1 | [01-complexity.md](01-complexity.md) | Complexity analysis (amortized, recursion trees, Master theorem) |
| 2 | [02-arrays-hashing.md](02-arrays-hashing.md) | Arrays & hashing |
| 3 | [03-two-pointers.md](03-two-pointers.md) | Two pointers |
| 4 | [04-sliding-window.md](04-sliding-window.md) | Sliding window (fixed & variable) |
| 5 | [05-prefix-sums.md](05-prefix-sums.md) | Prefix sums & difference arrays |
| 6 | [06-binary-search.md](06-binary-search.md) | Binary search (index & answer space) |
| 7 | [07-stack-monotonic.md](07-stack-monotonic.md) | Stack, monotonic stack/queue |
| 8 | [08-linked-lists.md](08-linked-lists.md) | Linked lists |
| 9 | [09-backtracking.md](09-backtracking.md) | Recursion & backtracking |
| 10 | [10-trees.md](10-trees.md) | Trees |
| 11 | [11-heaps.md](11-heaps.md) | Heaps / priority queues |
| 12 | [12-greedy.md](12-greedy.md) | Greedy (+ exchange-argument proofs) |
| 13 | [13-intervals.md](13-intervals.md) | Intervals |
| 14 | [14-graphs.md](14-graphs.md) | Graphs: BFS/DFS, components, cycles, topo sort |
| 15 | [15-union-find.md](15-union-find.md) | Union-Find |
| 16 | [16-shortest-paths-mst.md](16-shortest-paths-mst.md) | Shortest paths & MST |
| 17 | [17-dynamic-programming.md](17-dynamic-programming.md) | Dynamic programming (all families) |
| 18 | [18-tries.md](18-tries.md) | Tries |
| 19 | [19-bit-manipulation.md](19-bit-manipulation.md) | Bit manipulation |
| 20 | [20-math.md](20-math.md) | Math for interviews |
| 21 | [21-advanced.md](21-advanced.md) | Fenwick / segment tree, KMP / rolling hash, line sweep, design |
| 22 | [22-instinct-training.md](22-instinct-training.md) | Instinct training: mixed sets, "looks like X", flowchart, study plan, communication |

## How to use
1. Read Part 0 until you can reproduce the cheat sheet from memory.
2. For each chapter, read the insight and templates, **re-derive** the worked examples yourself before reading the solution, then do the practice ladder.
3. Do the recognition drill cold. If you miss one, re-read that chapter's trigger signals.
4. From week 4 on, interleave the mixed sets in Part 22. Topic-labelled practice builds skill; unlabelled practice builds recognition.

Build a single file: `cat 00-*.md 01-*.md ... 22-*.md > DSA.md` (or in PowerShell: `Get-Content (Get-ChildItem 0*.md,1*.md,2*.md | Sort-Object Name) | Set-Content DSA.md`).
