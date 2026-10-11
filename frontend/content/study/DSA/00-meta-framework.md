# Part 0 — The Meta-Framework

The goal of this part: turn "I've never seen this problem" into a **ranked list of 2–3 candidate techniques within 2–3 minutes**, using signals that are in the problem statement.

---

## 0.1 The Interview Procedure

A 45-minute coding round has roughly 35 minutes of real work. Budget:

| Step | Time | What you do | Failure mode it prevents |
|---|---|---|---|
| **1. Clarify** | 2–3 min | Input size and value ranges, negatives, duplicates, sorted?, empty input, output format (indices vs values, one vs all, order matters?), mutate input allowed?, online/streaming? | Solving the wrong problem; missing that input is sorted (a free hint) |
| **2. Examples & edge cases** | 2 min | Build one *normal* and one *adversarial* example by hand. Solve it by hand; watch what your brain does. | Code that passes the given example only |
| **3. Brute force** | 1–2 min | State it in one sentence with its complexity. Don't code it unless asked. | Freezing; having no baseline |
| **4. Bottleneck** | 2–3 min | Ask: *which loop dominates, and what does it recompute?* | Optimizing the wrong part |
| **5. Optimize** | 3–8 min | Apply a lever (§0.4). Check against the constraint table (§0.2). Get interviewer buy-in before coding. | Coding a solution that's still too slow |
| **6. Code** | 10–15 min | Templates, meaningful names, helper functions for sub-steps. | Messy code, off-by-one bugs |
| **7. Test** | 3–5 min | Trace your hand example line by line through the *code* (not your intent). Then edge cases. | Bugs the interviewer finds before you |
| **8. Complexity** | 1 min | Time and space, with a one-line justification each. | "O(n) I think" |

**Standard edge-case checklist:** empty, single element, two elements, all equal, all negative / zero, already sorted / reverse sorted, max size (performance), duplicates, disconnected graph, self-loops, cycles, `k = 0`, `k > n`, target not present, integer overflow (irrelevant in Python, but mention it if the interviewer uses Java/C++).

---

## 0.2 Constraint → Complexity Table

Budget: about **10⁸ simple operations** per second in C++ and roughly 10⁷ in Python. Interviewers think in C++ scale. Use this table **before** thinking about the algorithm. It rules out whole families of approaches.

| Max `n` | Max acceptable complexity | What that implies |
|---|---|---|
| ≤ 10–12 | O(n!), O(n · n!) | Permutations, brute-force all orderings |
| ≤ 20–25 | O(2ⁿ), O(n · 2ⁿ) | Subsets, bitmask enumeration, bitmask DP |
| ≤ 40 | O(2^(n/2)) | Meet-in-the-middle |
| ≤ 100 | O(n⁴) | 4 nested loops, some DP |
| ≤ 500 | O(n³) | Interval DP, Floyd-Warshall, triple loops |
| ≤ 5 000 | O(n²) | 2D DP (LCS, edit distance), all pairs |
| ≤ 10⁵ – 10⁶ | O(n log n) or O(n) | Sort, heap, binary search, two pointers, sliding window, hashing, monotonic stack, prefix sums, BFS/DFS, DP with O(1) transitions |
| ≤ 10⁹ – 10¹⁸ | O(log n), O(√n), O(1) | Binary search on answer, math, fast exponentiation |

### Using it to eliminate
- `n = 10⁵` → O(n²) = 10¹⁰, **dead**. You need O(n log n): sorting, a heap, binary search, or a linear sweep. Don't spend time on nested loops.
- `n ≤ 20` is **a hint, not a limit**. The setter is *telling* you exponential is intended (bitmask/backtracking). Nobody writes `n ≤ 20` for an O(n) problem.
- `n ≤ 1000` or `n ≤ 5000` → O(n²) DP is likely intended.
- Answer range up to 10⁹ with a "minimum X such that…" phrasing → binary search on the answer (≈30 iterations).
- **Secondary constraints matter too:**
  - `values ≤ 10⁴` → you can index arrays by value (counting sort, DP over values).
  - `k ≤ 100` → k can be a DP dimension.
  - `q` queries up to 10⁵ → each query must be O(log n) or O(1), so precompute (prefix sums, sparse tables, Fenwick).
  - `m·n ≤ 10⁵` grid → BFS/DFS over cells is fine.
  - `sum(nums) ≤ 10⁴` → DP over sums (knapsack) is fine.

---

## 0.3 Master Recognition Cheat Sheet

**The most important table in this document.** Read the problem, then scan for these signals. Several may fire. Keep the 2–3 strongest candidates and test each against the constraint table.

### A. By input shape and phrasing

| Signal in the problem | Candidate technique | Why it fits (the exploited property) |
|---|---|---|
| Array is **sorted** (or sorting loses nothing) | Binary search; two pointers | Order lets one comparison discard half, or discard a whole row of pairs |
| "Pair / triplet with sum = target" | Unsorted and indices needed → **hash map**. Values only or dedupe needed → **sort + two pointers** | Complement lookup in O(1); sorting makes the pair sum monotone in the pointers |
| "Contains duplicate", "seen before", "first unique" | Hash set / Counter | O(1) membership |
| "Anagram", "permutation of", "same characters" | Frequency count (26-array or Counter); if it's a substring → fixed sliding window | Multiset equality = equal counts |
| "Group things that are equivalent" | Hash map with a **canonical key** (sorted string, count tuple, normalized shape) | Equivalent items map to the same key |
| **Contiguous** subarray/substring + longest/shortest + a constraint | **Sliding window** (if the constraint is monotone as the window grows) | Each endpoint only moves forward → O(n) |
| Contiguous subarray + **sum = k / divisible by k** / negatives present | **Prefix sum + hash map** | sum(i..j) = P[j+1] − P[i]; count pairs of prefixes |
| Many **range sum queries**, static array | Prefix sums (1D/2D) | O(1) per query after O(n) build |
| Range queries **with updates** | Fenwick tree / segment tree | O(log n) per query and per update |
| Many **range increments**, then read final values | Difference array | O(1) per update, one prefix pass at the end |
| "Next greater / smaller element", "days until warmer", "span" | **Monotonic stack** | Each element is resolved by the first one that dominates it |
| "Largest rectangle", "sum of subarray minimums", "how far can X extend" | Monotonic stack (contribution technique) | Previous/next smaller gives each element's range of influence |
| "Max / min of every window of size k" | **Monotonic deque** | Dominated elements can never be the max again |
| Brackets, nesting, "evaluate expression", undo, "remove adjacent" | Stack | Most-recent-unresolved-first (LIFO) structure |
| Linked list: cycle, middle, k-th from end, intersection | **Fast/slow pointers** | The speed difference gives distance information in O(1) space |
| Linked list: reverse part, reorder, delete with possible head change | **Dummy node** + in-place reversal | Removes head special-cases |
| "All subsets / permutations / combinations", "generate all valid" | **Backtracking** | Output size itself is exponential |
| `n ≤ 20` + subsets or assignments | Bitmask enumeration / **bitmask DP** | 2ⁿ states fit |
| Sudoku, N-Queens, "place such that no conflicts" | Backtracking + constraint sets + pruning | Prune early violations |
| Tree: height, diameter, balanced, path sum, subtree property | **Postorder DFS** that returns info to the parent | Answer at a node = f(answers of children) |
| Tree: level, "right side view", min depth, zigzag | **BFS** by levels | Levels = BFS layers |
| BST: k-th smallest, validate, range, successor | Inorder traversal / pass bounds down | Inorder of a BST is sorted |
| "Lowest common ancestor" | Postorder recursion (binary tree); split-point walk (BST) | The first node where p and q diverge |
| "Top k", "k-th largest", "k closest", "k most frequent" | **Heap of size k**; quickselect; bucket sort | Only k items need to be kept ordered |
| "Merge k sorted", "smallest range across k lists" | **k-way merge** with heap | Heap holds the frontier of k streams |
| "Median of a stream", "balance two halves" | **Two heaps** | Max-heap for the low half, min-heap for the high half |
| Repeatedly "pick the best available option" as time moves | Heap (often after sorting by time) | Dynamic min/max |
| "Min number of X to cover", "max non-overlapping", "can reach end" | **Greedy** (usually sort by end, or furthest reach) | Exchange argument holds |
| Intervals: merge, overlap, insert | **Sort by start + sweep** | After sorting, overlaps are only with the running interval |
| "Max concurrent", "min meeting rooms", "at any time" | Sweep events (+1/−1) or min-heap of end times | Count of active intervals |
| Grid islands / regions / "connected" | DFS / BFS / Union-Find | Connected components |
| "Minimum steps / moves / transformations", unweighted | **BFS** | BFS layer = distance |
| "Spreads simultaneously from several sources" | **Multi-source BFS** | Seed the queue with all sources at distance 0 |
| Prerequisites, dependencies, "order of tasks", "alien dictionary" | **Topological sort** (Kahn's) | DAG ordering; in-degree 0 = ready |
| Detect cycle, **directed** | 3-color DFS or Kahn's (processed < n) | A back edge means a cycle |
| Detect cycle, **undirected** / "redundant edge" | Union-Find (or DFS tracking the parent) | An edge inside one component closes a cycle |
| "Are X and Y connected" with edges arriving over time; equivalence classes | **Union-Find** | Near-O(1) merge and query |
| Weighted shortest path, non-negative weights | **Dijkstra** | Greedy settling of the closest node is safe |
| Edge weights only 0 or 1 | **0-1 BFS** (deque) | Dijkstra without the heap |
| "At most k stops / edges", negative weights | **Bellman-Ford** (k rounds) | Round i = shortest path using ≤ i edges |
| "Minimize the max edge along a path" | Dijkstra with max instead of +, or BS + BFS, or UF | Minimax path |
| "Connect all points / cities with min total cost" | **MST** (Kruskal / Prim) | Cut property |
| "**Minimize the maximum**" / "**maximize the minimum**" / "min capacity/speed/days such that…" | **Binary search on the answer** + greedy feasibility check | Feasibility is monotone in the answer |
| "Number of ways", "min/max cost", choices at each step, subproblems repeat | **DP** | Optimal substructure + overlapping subproblems |
| Two strings: align, edit, common subsequence, interleave | **2D DP** over (i, j) prefixes | The last characters decide the transition |
| "Longest increasing subsequence" / chains / envelopes | LIS DP; O(n log n) patience sorting | Tail array is sorted, so binary search on it |
| "Choose items subject to capacity / target sum" | **Knapsack** DP | Item × capacity state |
| "Split / merge / burst a range, cost depends on the split point" | **Interval DP** | dp[i][j] over sub-ranges, try every split k |
| States like holding/sold/cooldown, "at most k transactions" | **State-machine DP** | Finite states × position |
| Many words, shared prefixes, autocomplete, "starts with" | **Trie** | Prefix sharing |
| "Maximum XOR of two numbers" | Bitwise trie or greedy bit-by-bit with a set | Decide the highest bit first |
| "Every element appears twice except one" | **XOR** | a ^ a = 0 |
| "Return answer modulo 10⁹+7" | Counting DP or combinatorics | The count is huge, so it's a counting problem |
| Find a pattern in text, periodicity | KMP / Z / rolling hash | Reuse partial matches |
| "Longest duplicate substring", compare many substrings | **Rolling hash** + binary search on length | O(1) substring hash |
| O(1) get/put with eviction (LRU/LFU) | Hash map + doubly linked list | O(1) locate + O(1) reorder |
| "Count smaller after self", inversions, rank queries | **Fenwick tree** / merge sort | Order statistics under insertion |
| Values in `[1..n]`, O(1) extra space | Index-as-hash / cyclic sort / Floyd on indices | The array is its own hash table |
| Palindrome check / substrings / subsequence | Two pointers / expand around center / interval DP | Symmetry |
| Stream too large to store, random sample | Reservoir sampling | Uniform with O(k) memory |

### B. By question type

| The question asks… | First candidates |
|---|---|
| **Does there exist / is it possible** | BFS/DFS reachability, greedy, two pointers, DP (bool), binary search check |
| **Count the number of** | DP, combinatorics, prefix sum + hash map, "at most K − at most K−1" |
| **Min / max value** | DP, greedy, binary search on answer, heap, shortest path |
| **Enumerate all** | Backtracking (the output size forces exponential time) |
| **k-th something** | Heap, quickselect, binary search on the value ("how many ≤ x?") |
| **Longest / shortest contiguous** | Sliding window, prefix sums + hash map (first occurrence) |
| **Design a data structure** | Hash map plus one ordered structure (DLL, heap, sorted list, stack) |

### C. By constraint (from §0.2)
`n ≤ 20` → bitmask / backtracking · `n ≤ 500` → O(n³) DP · `n ≤ 5000` → O(n²) DP · `n ≥ 10⁵` → O(n log n) or better · answer range ≥ 10⁹ → binary search on answer / math.

---

## 0.4 Optimization Levers

Almost every brute force → optimal transformation is one of these seven moves. When stuck, go down this list and ask "does this lever remove the waste in my brute force?"

| Lever | Waste it kills | Canonical transformation |
|---|---|---|
| **1. Precompute** | Recomputing the same aggregate for overlapping ranges | Range sum O(n) per query → prefix sums O(1). Product except self: prefix × suffix. |
| **2. Hash for O(1) lookup** | An inner loop that *searches* for something | Two Sum O(n²) → O(n): "have I seen `target − x`?" |
| **3. Sort first** | Unstructured search; pairs that could be pruned by order | 3Sum O(n³) → O(n²). Intervals: sort by start and only neighbors overlap. |
| **4. Maintain an invariant while sweeping** | Re-scanning a window or history every step | Sliding window, monotonic stack, running min ("best buy price so far"), Kadane |
| **5. Cache repeated subproblems** | Exponential recursion that solves the same state repeatedly | Fibonacci O(φⁿ) → O(n). Any recursion with repeated arguments → memo → DP. |
| **6. Prune the search space** | Exploring branches that can't win | Binary search (discard half), backtracking pruning, two-pointer elimination, branch-and-bound |
| **7. Change what you iterate over** | Iterating over a huge or wrong dimension | Iterate over the answer (BS on answer), over values instead of indices, over gaps, over the smaller set, over the complement (count total − bad), reverse time (process deletions as unions) |

**The bottleneck question:** look at the innermost loop of your brute force and name *exactly* what it computes. That name usually tells you the lever:
- "the sum of this range" → prefix sums
- "whether this value exists" → hash set
- "the max of the last k" → monotonic deque
- "the answer for this smaller input" → DP
- "the first element bigger than x" → monotonic stack / binary search
- "all pairs" → sort + two pointers or hashing

---

## 0.5 The 2-Minute Hypothesis Routine

For every new problem, run this in your head:

1. **What is the input shape?** Array / string / tree / graph / grid / intervals / stream / design.
2. **What is the question type?** Exists / count / optimize / enumerate / k-th.
3. **What does `n` allow?** Use §0.2.
4. **Which signals fire?** Use §0.3.
5. **Brute force in one sentence, then name the bottleneck.** Use §0.4.
6. **Pick the best candidate and sanity-check it on your adversarial example.** If it fails, the counterexample usually points to the next candidate (e.g., greedy fails → DP).

Next: Part 1 — Complexity Analysis
