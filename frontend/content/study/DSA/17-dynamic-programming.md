# Part 17 — Dynamic Programming

## 1. Core insight
DP applies when a problem has **optimal substructure** and **overlapping subproblems**. Optimal substructure: the best answer is built from best answers to smaller versions. Overlapping subproblems: a naive recursion solves the same smaller versions repeatedly. DP solves each distinct subproblem **once** and stores it, collapsing an exponential recursion tree into (number of states) × (cost per transition). The hard part is never the table. It's **choosing the state**: the *minimum information about the past needed to make every future decision optimally*. Everything else follows mechanically from the state.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Number of ways" (often "mod 10⁹+7"), "minimum/maximum cost/length/profit", "is it possible".
- Choices at each step (take/skip, which coin, which split point, edit operation) whose effects **interact** with later choices.
- A brute-force recursion whose arguments repeat (draw the tree; if you see `f(3)` twice, it's DP).
- Sequences/strings with "subsequence" (non-contiguous), two strings compared, "palindromic subsequence".
- Constraints n ≤ 1000–5000 (O(n²) DP), n·target ≤ 10⁷ (knapsack), n ≤ 20 (bitmask DP).
- Grid with moves restricted to right/down (acyclic) → grid DP.

**Anti-signals**
- A greedy choice is provably safe → greedy is simpler (Jump Game, Gas Station).
- "List all solutions" → backtracking (the output is exponential; DP counts, it doesn't enumerate cheaply).
- State would need to remember an unbounded history (e.g., the full set of used items with n = 10⁵) → rethink: sort + greedy, or a different state.
- Movement in all 4 directions on a grid (cycles) → shortest path (BFS/Dijkstra), not DP.
- Contiguous subarray with a simple constraint → sliding window / prefix sums are usually simpler (Kadane *is* DP, but a tiny one).

## 3. Mental model — the framework

**Five questions. Answer them in words before writing code.**

| Step | Question | Example: House Robber |
|---|---|---|
| **State** | What does `dp[...]` mean, *in one sentence*? | `dp[i]` = max loot from houses `0..i` |
| **Transition** | How is a state computed from smaller states? (Enumerate the *last decision*.) | Rob i: `dp[i−2] + a[i]`. Skip i: `dp[i−1]`. Take the max. |
| **Base case** | Which states are known without recursion? | `dp[−1] = 0`, `dp[0] = a[0]` |
| **Order** | In which order do dependencies become available? | i increasing |
| **Answer** | Which state (or combination) is the answer? | `dp[n−1]` |

**How to find the state:** write the brute-force recursion first. Its **parameters** are the state. If it takes `(i, remaining, holding)`, then dp has those three dimensions. To shrink it, ask which parameters are determined by others.

**The four-stage derivation (do this in interviews):**
```
1. Recursion        (correct but exponential)       -> shows the state and transition
2. Memoization      (@cache on the recursion)       -> O(states × transition)
3. Tabulation       (loops in dependency order)     -> no recursion-depth limit
4. Space optimize   (keep only the rows you read)   -> O(n) or O(1)
```

**Complexity = number of states × cost per transition.** Always say both factors.

## 4. Canonical templates — the four stages on House Robber (LC 198)

```python
from functools import cache

# Stage 1: plain recursion. f(i) = best loot from houses i..n-1. O(2^n).
def rob_rec(nums: list[int]) -> int:
    def f(i: int) -> int:
        if i >= len(nums):
            return 0
        return max(f(i + 1), nums[i] + f(i + 2))      # skip i | rob i
    return f(0)

# Stage 2: memoization. n states × O(1) = O(n) time, O(n) space (+ recursion stack).
def rob_memo(nums: list[int]) -> int:
    @cache
    def f(i: int) -> int:
        if i >= len(nums):
            return 0
        return max(f(i + 1), nums[i] + f(i + 2))
    return f(0)

# Stage 3: tabulation. f(i) depends on f(i+1), f(i+2) -> fill i from n-1 down to 0.
def rob_tab(nums: list[int]) -> int:
    n = len(nums)
    dp = [0] * (n + 2)                                 # dp[n] = dp[n+1] = 0 (base cases)
    for i in range(n - 1, -1, -1):
        dp[i] = max(dp[i + 1], nums[i] + dp[i + 2])
    return dp[0]

# Stage 4: space optimization. Only the two most recent values are read.
def rob(nums: list[int]) -> int:
    prev2 = prev1 = 0                                  # dp[i-2], dp[i-1] (forward version)
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1
```
**Trace** [2,7,9,3,1]: (0,0) → (0,2) → (2,7) → (7,11) → (11,11) → (11,12). Answer 12 ✓ (2+9+1).

---

## 5. The DP families (each with insight, template, and a worked example)

### 17.A — 1D DP (linear sequences)
**Insight:** the state is a prefix (or suffix) position. The transition looks back a bounded number of steps, or over all earlier positions. **Signals:** a sequence, a decision per element, "ways to reach step i", "can the prefix be segmented".

| Problem | State | Transition |
|---|---|---|
| Climbing Stairs | ways to reach i | `dp[i−1] + dp[i−2]` |
| House Robber | best loot through i | `max(dp[i−1], dp[i−2] + a[i])` |
| Decode Ways | decodings of s[:i] | `dp[i−1]` (if the single digit is valid) + `dp[i−2]` (if the 2 digits are in 10..26) |
| Word Break | can s[:i] be segmented | `any(dp[j] and s[j:i] in words)` |
| Coin Change | min coins for amount a | `min(dp[a − c] + 1)` over c |
| LIS (O(n²)) | LIS ending at i | `1 + max(dp[j])` for j < i with a[j] < a[i] |

**Worked example — Coin Change (LC 322), Medium**
**Brute force:** try every coin at each step, recursively → O(k^(amount/min_coin)).
**Why not greedy:** coins [1,3,4], amount 6 → greedy 4+1+1 = 3 coins, optimal 3+3 = 2.
**State:** `dp[a]` = the min number of coins to make amount a. **Transition:** the last coin used is some c → `dp[a − c] + 1`. **Base:** `dp[0] = 0`. **Order:** a increasing. **Answer:** `dp[amount]`.
```python
def coin_change(coins: list[int], amount: int) -> int:
    INF = float('inf')
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == INF else dp[amount]
```
**Complexity:** O(amount · k) time, O(amount) space.
**Edge cases:** amount 0 → 0. Unreachable amounts → −1.

```python
# Two more 1D templates worth memorizing
def word_break(s: str, word_dict: list[str]) -> bool:
    words = set(word_dict)
    max_len = max(map(len, words), default=0)
    dp = [True] + [False] * len(s)                     # dp[i]: s[:i] can be segmented
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - max_len), i):        # only the last word's possible starts
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[-1]

def num_decodings(s: str) -> int:
    prev2, prev1 = 1, (1 if s[0] != '0' else 0)        # dp[0], dp[1]
    for i in range(2, len(s) + 1):
        cur = prev1 if s[i - 1] != '0' else 0          # one-digit decode
        if 10 <= int(s[i - 2:i]) <= 26:                # two-digit decode
            cur += prev2
        prev2, prev1 = prev1, cur
    return prev1
```

---

### 17.B — 2D / Grid DP
**Insight:** the state is a cell (r, c). With moves restricted (right/down), the grid is a DAG, and row-major order satisfies all dependencies. **Signals:** "paths from top-left to bottom-right", "only right or down", "largest square/rectangle of 1s", "min path sum".

**Worked example — Maximal Square (LC 221), Medium**
**Brute force:** for each cell as the top-left corner, grow the square while all cells are '1' → O((RC)²).
**Aha:** `dp[r][c]` = side of the largest all-1 square with its **bottom-right corner** at (r, c). The square is limited by the smallest of the three neighbors that must also be squares:
```
dp[r][c] = 1 + min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])   if matrix[r][c] == '1'
   up  diag
   ┌──┬──┐
   │ 2│ 2│
   ├──┼──┤      min(2, 2, 1) + 1 = 2  -> the diagonal limits it
   │ 1│ ?│
   └──┴──┘
  left
```
```python
def maximal_square(matrix: list[list[str]]) -> int:
    R, C = len(matrix), len(matrix[0])
    dp = [0] * (C + 1)                   # one rolling row; dp[c] ↔ column c-1
    best = 0
    for r in range(R):
        diag = 0                         # dp[r-1][c-1] before it gets overwritten
        for c in range(1, C + 1):
            up = dp[c]
            if matrix[r][c - 1] == '1':
                dp[c] = 1 + min(up, dp[c - 1], diag)
                best = max(best, dp[c])
            else:
                dp[c] = 0
            diag = up
    return best * best
```
**Complexity:** O(R·C) time, O(C) space.
**Space-optimization rule for 2D:** if a cell reads only the previous row and the current row, keep one row. Save the "diagonal" before overwriting it.

```python
# Unique Paths (LC 62) in one row: dp[c] (from above) += dp[c-1] (from the left)
def unique_paths(m: int, n: int) -> int:
    dp = [1] * n
    for _ in range(1, m):
        for c in range(1, n):
            dp[c] += dp[c - 1]
    return dp[-1]
```

---

### 17.C — Knapsack Family
**Insight:** items × capacity. For each item decide how many to take. The state is `(items considered, capacity used / sum reached)`. **Signals:** "subset with sum", "partition into two equal", "choose items under a weight limit", "number of ways to make amount", "assign + or −".

| Variant | Loop order (1D array) | Why |
|---|---|---|
| **0/1** (each item once) | items outer, capacity **descending** | descending reads the *old* row, so an item isn't reused |
| **Unbounded** (reuse allowed) | items outer, capacity **ascending** | ascending reads the *updated* row, so reuse is allowed |
| **Count combinations** (order doesn't matter) | **items outer**, amount inner | each combination is counted once, in item order |
| **Count permutations** (order matters) | **amount outer**, items inner | every order is counted |

```python
# 0/1 feasibility: Partition Equal Subset Sum (LC 416)
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [True] + [False] * target                 # dp[s]: some subset sums to s
    for x in nums:
        for s in range(target, x - 1, -1):         # DESCENDING: each x used at most once
            dp[s] = dp[s] or dp[s - x]
    return dp[target]

# Unbounded count of combinations: Coin Change II (LC 518)
def change(amount: int, coins: list[int]) -> int:
    dp = [1] + [0] * amount
    for c in coins:                                # coins OUTER -> combinations
        for a in range(c, amount + 1):             # ASCENDING -> reuse allowed
            dp[a] += dp[a - c]
    return dp[amount]
```

**Worked example — Target Sum (LC 494), Medium**
**Problem:** assign + or − to each number so the total equals target. Count the ways.
**Brute force:** 2ⁿ assignments. That's OK for n = 20, but the DP insight is expected.
**Aha (transform to a known family):** let P = the sum of the + numbers and N = the sum of the − numbers. Then `P − N = target` and `P + N = total`, so **P = (total + target) / 2**. Now it's "count subsets with sum P" = 0/1 knapsack counting.
```python
def find_target_sum_ways(nums: list[int], target: int) -> int:
    total = sum(nums)
    if abs(target) > total or (total + target) % 2:
        return 0
    P = (total + target) // 2
    dp = [1] + [0] * P                             # dp[s]: number of subsets with sum s
    for x in nums:
        for s in range(P, x - 1, -1):
            dp[s] += dp[s - x]
    return dp[P]
```
**Trace** [1,1,1,1,1], target 3 → P = 4 → C(5,4) = 5 ✓.
**Complexity:** O(n · P) time, O(P) space.
**Edge cases:** zeros (each zero doubles the count, and the descending loop with x=0 handles it: `dp[s] += dp[s]`). Parity or impossible targets → 0.

---

### 17.D — LIS / LCS / Edit Distance (sequence alignment)
**Insight:** for **two sequences**, the state is a pair of prefixes `(i, j)`, and the transition looks at the **last characters**: match them, or drop one. For **one sequence with order constraints** (LIS), the state is "best ending at i".

```python
from bisect import bisect_left

# LIS in O(n log n): tails[k] = smallest possible tail of an increasing subsequence of length k+1
def length_of_lis(nums: list[int]) -> int:
    tails: list[int] = []
    for x in nums:
        i = bisect_left(tails, x)          # the first tail >= x (strictly increasing LIS)
        if i == len(tails):
            tails.append(x)                # extends the longest subsequence
        else:
            tails[i] = x                   # a smaller tail for length i+1: better for the future
    return len(tails)                      # tails is NOT an actual LIS, only its length is right

# LCS (LC 1143)
def longest_common_subsequence(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]    # dp[i][j]: LCS of a[:i], b[:j]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```
**Trace LIS** [10,9,2,5,3,7,101,18]: tails → [10] → [9] → [2] → [2,5] → [2,3] → [2,3,7] → [2,3,7,101] → [2,3,7,18] → 4 ✓.

**Worked example — Edit Distance (LC 72), Medium**
**Problem:** the minimum number of insert/delete/replace operations to turn `a` into `b`.
**Brute force:** recursion on (i, j) with 3 branches → O(3^(m+n)).
**State:** `dp[i][j]` = the edit distance between `a[:i]` and `b[:j]`. **Transition (last characters):**
- `a[i−1] == b[j−1]` → `dp[i−1][j−1]` (free).
- Otherwise 1 + min of: **delete** `a[i−1]` → `dp[i−1][j]`; **insert** `b[j−1]` → `dp[i][j−1]`; **replace** → `dp[i−1][j−1]`.

**Base:** `dp[i][0] = i` (delete all), `dp[0][j] = j` (insert all).
```python
def min_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))                      # row i-1 (initially row 0)
    for i in range(1, m + 1):
        cur = [i] + [0] * n                        # dp[i][0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j],          # delete
                                 cur[j - 1],       # insert
                                 prev[j - 1])      # replace
        prev = cur
    return prev[n]
```
**Trace** "horse" → "ros" = 3 ✓ (replace h→r, delete r, delete e).
**Complexity:** O(m·n) time, O(n) space.
**Edge cases:** an empty string on either side (the base rows handle it).

---

### 17.E — Interval DP
**Insight:** the state is a **range** `[i, j]`. The transition picks a **split point k** inside the range (or peels the ends). Fill by **increasing length** so sub-ranges are ready. Typically O(n³). **Signals:** "burst / merge / cut / remove, and the cost depends on the neighbors or the range", "palindromic subsequence", "optimal game on both ends", "matrix-chain order", n ≤ 500.

```python
# Longest Palindromic Subsequence (LC 516): peel the ends
def longest_palindrome_subseq(s: str) -> int:
    n = len(s)
    dp = [[0] * n for _ in range(n)]              # dp[i][j]: LPS of s[i..j]
    for i in range(n - 1, -1, -1):                # i descending so dp[i+1][*] is ready
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2   # dp[i+1][i] = 0 when j = i+1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

**Worked example — Burst Balloons (LC 312), Hard**
**Problem:** bursting balloon i earns `left · nums[i] · right` (its current neighbors). Maximize the total.
**Naive state (fails):** "choose the first balloon to burst" → the neighbors of the rest change, so the subproblems aren't independent.
**Aha (think about the LAST balloon):** if k is the **last** balloon burst in the open interval (i, j), then while bursting everything else between i and j, k is still present. So it cleanly separates (i, k) and (k, j), and when k finally bursts, its neighbors are exactly i and j.
```python
def max_coins(nums: list[int]) -> int:
    a = [1] + nums + [1]                          # virtual boundary balloons
    n = len(a)
    dp = [[0] * n for _ in range(n)]              # dp[i][j]: best for the open interval (i, j)
    for length in range(2, n):                    # j - i = length
        for i in range(0, n - length):
            j = i + length
            best = 0
            for k in range(i + 1, j):             # k = last burst in (i, j)
                best = max(best, dp[i][k] + dp[k][j] + a[i] * a[k] * a[j])
            dp[i][j] = best
    return dp[0][n - 1]
```
**Trace** [3,1,5,8] → 167 ✓ (the LC example).
**Complexity:** O(n³) time, O(n²) space. n ≤ 300 fits.
**Transferable insight:** when a "first choice" breaks independence, try the **last choice**.

---

### 17.F — DP on Trees
**Insight:** the state is a node plus a small status. The postorder recursion **returns a tuple of answers**, one per status (e.g., "robbed / not robbed", "has camera / covered / not covered"). **Signals:** a tree plus a constraint between parent and child ("adjacent", "cover neighbors"), "max independent set on a tree".

**Worked example — House Robber III (LC 337), Medium**
**Brute force:** at each node try rob / skip and recurse into the grandchildren → exponential re-computation.
**Aha:** return two values per node: the best if this node is **robbed**, and the best if it's **not robbed**. The parent combines them in O(1).
```python
from typing import Optional

def rob_tree(root: Optional['TreeNode']) -> int:
    def dfs(node: Optional['TreeNode']) -> tuple[int, int]:   # (rob_this, skip_this)
        if not node:
            return 0, 0
        l_rob, l_skip = dfs(node.left)
        r_rob, r_skip = dfs(node.right)
        rob_this = node.val + l_skip + r_skip           # children must be skipped
        skip_this = max(l_rob, l_skip) + max(r_rob, r_skip)
        return rob_this, skip_this
    return max(dfs(root))
```
**Complexity:** O(n) time, O(h) space.
**Edge cases:** an empty tree → 0. A skewed tree (recursion depth).

---

### 17.G — Bitmask DP
**Insight:** when the state must remember **which subset** of a small set (n ≤ 20) has been used, encode it as an integer mask. There are 2ⁿ masks (× n for "current position"). **Signals:** n ≤ 20, "visit all", "assign each to exactly one", "partition into groups", TSP-like.

```python
# TSP-style template: dp[mask][u] = min cost to visit exactly `mask`, ending at u
def tsp(dist: list[list[int]]) -> int:
    n = len(dist)
    INF = float('inf')
    FULL = 1 << n
    dp = [[INF] * n for _ in range(FULL)]
    dp[1][0] = 0                                      # start at node 0
    for mask in range(FULL):
        for u in range(n):
            if dp[mask][u] == INF or not (mask >> u) & 1:
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                nm = mask | (1 << v)
                dp[nm][v] = min(dp[nm][v], dp[mask][u] + dist[u][v])
    return min(dp[FULL - 1][u] + dist[u][0] for u in range(n))
# O(2^n · n^2) time, O(2^n · n) space. Masks increase, and every transition adds a bit -> order is valid.
```

**Worked example — Partition to K Equal Sum Subsets (LC 698), Medium**
**Brute force:** assign each number to one of k buckets → kⁿ.
**Aha (signal: n ≤ 16):** the state is the **set of used numbers**. Numbers are placed into buckets one after another, so the current bucket's fill is determined by the mask: `sum(mask) % target`. We only need to know whether the mask is reachable.
```python
def can_partition_k_subsets(nums: list[int], k: int) -> bool:
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    nums.sort()
    if nums[-1] > target:
        return False
    n = len(nums)
    fill = [-1] * (1 << n)            # fill[mask]: current bucket's sum if reachable, else -1
    fill[0] = 0
    for mask in range(1 << n):
        if fill[mask] == -1:
            continue
        for i in range(n):
            if (mask >> i) & 1:
                continue
            if fill[mask] + nums[i] > target:
                break                  # sorted: larger numbers won't fit either
            nxt = mask | (1 << i)
            if fill[nxt] == -1:
                fill[nxt] = (fill[mask] + nums[i]) % target   # a full bucket resets to 0
    return fill[-1] == 0
```
**Complexity:** O(2ⁿ · n) time, O(2ⁿ) space. With n = 16 that's about 10⁶.
**Edge cases:** an element > target → False. k = 1 → True.

---

### 17.H — State-Machine DP
**Insight:** at each position you're in one of a few **modes** (holding / not holding / cooldown, or transactions used). Write one variable per mode and update them all from the *previous* position's values. **Signals:** stock problems, "cooldown", "fee", "at most k transactions", "can't do X twice in a row".

```
          buy (−p)
   rest ───────────▶ hold
    ▲  ▲               │
    │  └──(wait)       │ sell (+p)
    │                  ▼
    └────(cooldown)── sold
```

**Worked example — Best Time to Buy and Sell Stock with Cooldown (LC 309), Medium**
**States at the end of day i:** `hold` (own a share), `sold` (sold today, so tomorrow is a cooldown), `rest` (no share, free to buy).
**Transitions:** `hold' = max(hold, rest − p)`, `sold' = hold + p`, `rest' = max(rest, sold)`.
```python
def max_profit_cooldown(prices: list[int]) -> int:
    hold, sold, rest = float('-inf'), 0, 0
    for p in prices:
        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)   # all use old values
    return max(sold, rest)
```
**Trace** [1,2,3,0,2] → buy 1, sell 2, cooldown, buy 0, sell 2 → 3 ✓.
**Complexity:** O(n) time, O(1) space.

```python
# At most k transactions (LC 188): buy[t]/sell[t] = best with t-th buy/sell done
def max_profit_k(k: int, prices: list[int]) -> int:
    buy = [float('-inf')] * (k + 1)
    sell = [0] * (k + 1)
    for p in prices:
        for t in range(1, k + 1):
            buy[t] = max(buy[t], sell[t - 1] - p)
            sell[t] = max(sell[t], buy[t] + p)
    return sell[k]
```

---

## 6. Common mistakes & traps
- Defining the state vaguely ("dp[i] is the answer for i"). Write a precise sentence: *ending at i* vs *within the first i*. These are different DPs.
- Wrong loop direction in 0/1 vs unbounded knapsack.
- Combinations vs permutations loop order (Coin Change II vs Combination Sum IV).
- Off-by-one with prefix-length indexing: `dp[i]` about `s[:i]` means character `s[i−1]`.
- Space optimization that overwrites a value still needed (the diagonal in 2D, the old values in state machines → use tuple assignment).
- `@cache` on a function that closes over mutable data that changes between calls, or recursion deeper than 1000 → use tabulation.
- Forgetting impossible states (use −∞/∞, not 0, for max/min problems where some states are unreachable).
- Interval DP filled in the wrong order (must be by increasing length, or i descending).
- LIS `tails` returned as the actual subsequence. It isn't one.
- Answer location: `dp[n]`, `max(dp)`, `dp[0][n−1]`, or a max over the final states. Decide explicitly.

## 7. Connections
- **Backtracking** (Ch 9): memoized backtracking *is* top-down DP. Write the recursion, notice repeated arguments, add `@cache`.
- **Greedy** (Ch 12): greedy = DP where one choice always dominates. A greedy counterexample tells you the missing DP state.
- **Graphs** (Ch 14/16): DP = shortest/longest path in the DAG of states. Bellman-Ford and Floyd-Warshall are DPs.
- **Binary search** (Ch 6): LIS in O(n log n). Weighted job scheduling (DP + bisect for the last compatible job).
- **Monotonic deque** (Ch 7): optimizes `max over a sliding range of dp`.
- **Prefix sums** (Ch 5): O(1) range sums inside transitions (stone game, merge stones).
- **Bitmask** (Ch 19): subset states. **Trees** (Ch 10): postorder tuple returns.

## 8. Practice ladder (grouped by family, Easy → Hard within the group)

| # | Problem | Diff | Family | Signal hint |
|---|---|---|---|---|
| 70 | [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | Easy | 1D | Ways to reach i from the last 1–2 steps |
| 746 | [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) | Easy | 1D | Min cost version of the same |
| 198 | [House Robber](https://leetcode.com/problems/house-robber/) | Medium | 1D | Take/skip with an adjacency ban |
| 213 | [House Robber II](https://leetcode.com/problems/house-robber-ii/) | Medium | 1D | Circle → run twice, excluding the first or last |
| 91 | [Decode Ways](https://leetcode.com/problems/decode-ways/) | Medium | 1D | Count parses, 1- or 2-digit steps |
| 139 | [Word Break](https://leetcode.com/problems/word-break/) | Medium | 1D | Can the prefix be segmented |
| 322 | [Coin Change](https://leetcode.com/problems/coin-change/) | Medium | 1D/unbounded | Min coins, greedy fails |
| 152 | [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) | Medium | 1D | Negatives flip → track the max and min ending here |
| 62 | [Unique Paths](https://leetcode.com/problems/unique-paths/) | Medium | Grid | Right/down only → DAG |
| 64 | [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) | Medium | Grid | Min over up/left |
| 221 | [Maximal Square](https://leetcode.com/problems/maximal-square/) | Medium | Grid | Bottom-right corner, min of three |
| 416 | [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) | Medium | 0/1 knapsack | Subset sum = total/2 |
| 494 | [Target Sum](https://leetcode.com/problems/target-sum/) | Medium | 0/1 knapsack | ± signs → count subsets with P = (total+target)/2 |
| 518 | [Coin Change II](https://leetcode.com/problems/coin-change-ii/) | Medium | Unbounded | Count combinations → coins outer |
| 377 | [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/) | Medium | Unbounded | Order matters → amount outer |
| 300 | [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) | Medium | LIS | Ending at i; O(n log n) tails |
| 1143 | [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) | Medium | LCS | Two strings → prefix pair |
| 72 | [Edit Distance](https://leetcode.com/problems/edit-distance/) | Medium | LCS | Three operations on the last characters |
| 516 | [Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/) | Medium | Interval | Peel both ends |
| 309 | [Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | Medium | State machine | hold / sold / rest |
| 337 | [House Robber III](https://leetcode.com/problems/house-robber-iii/) | Medium | Tree | Return (rob, skip) per node |
| 698 | [Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/) | Medium | Bitmask | n ≤ 16 → used-set mask |
| 97 | [Interleaving String](https://leetcode.com/problems/interleaving-string/) | Medium | LCS-like | Prefix pair (i, j) determines the position in s3 |
| 354 | [Russian Doll Envelopes](https://leetcode.com/problems/russian-doll-envelopes/) | Hard | LIS | Sort w asc, h desc, then LIS on h |
| 115 | [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) | Hard | LCS-like | Count matches of t in s |
| 10 | [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/) | Hard | LCS-like | `*` = zero or more of the previous → two transitions |
| 312 | [Burst Balloons](https://leetcode.com/problems/burst-balloons/) | Hard | Interval | Choose the LAST balloon |
| 188 | [Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) | Hard | State machine | k transactions → buy[t], sell[t] |
| 968 | [Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/) | Hard | Tree | Three states per node |
| 1235 | [Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/) | Hard | 1D + BS | Sort by end, bisect for the last compatible job |
| 847 | [Shortest Path Visiting All Nodes](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) | Hard | Bitmask | (node, visited mask) with BFS |

## 9. Recognition drill
1. Given a string, return the minimum number of cuts so every piece is a palindrome.
2. Given a triangle of numbers, find the minimum path sum from top to bottom, moving to adjacent numbers on the row below.
3. You have `n ≤ 10` people and 40 hat types; each person likes some hats. Count the ways to give distinct hats so everyone gets one they like.
4. Two players take coins from either end of a row; both play optimally. Can player 1 win?
5. Given an array, find the length of the longest subsequence where consecutive elements differ by exactly `d`.

<details><summary>Answers</summary>

1. **1D DP over prefixes + interval precomputation**: `is_pal[i][j]` (interval DP), then `cuts[j] = min(cuts[i−1] + 1)` over palindromic s[i..j] (LC 132).
2. **Grid-like DP bottom-up**: `dp[j] = row[j] + min(dp[j], dp[j+1])`, processed from the bottom row, O(n) space (LC 120).
3. **Bitmask DP over people** (mask of people already assigned), iterating over hats: `dp[hat][mask]`. People are the small dimension (n ≤ 10), so they're the mask, not the 40 hats (LC 1434).
4. **Interval DP (game)**: `dp[i][j]` = the best score difference the current player can force on a[i..j] = `max(a[i] − dp[i+1][j], a[j] − dp[i][j−1])` (LC 486 / 877).
5. **1D DP with a hash map keyed by value**: `dp[x] = dp[x − d] + 1` (LC 1218). Subsequence + a fixed-difference predecessor → a map instead of an O(n²) scan.
</details>

Next: Part 18 — Tries
