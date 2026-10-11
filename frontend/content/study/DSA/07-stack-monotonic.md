# Part 7 — Stack & Monotonic Stack / Queue

## 1. Core insight
A stack holds **unresolved items, most recent first**. It fits whenever the next thing to resolve is the most recently opened one: brackets, nested structures, expression evaluation, undo. A **monotonic stack** adds one rule: before pushing x, pop every element x dominates. Each popped element has just found its answer (x is its "next greater"), and it can never be anyone's answer again, because x blocks it. Every index is pushed once and popped at most once → O(n) for "next/previous greater/smaller" problems. Naively those are O(n²). A **monotonic deque** does the same for a sliding window, additionally evicting indices that fall out of the window from the front.

## 2. Trigger signals / Anti-signals
**Triggers (plain stack)**
- Matching brackets, "valid parentheses", nested encodings (`3[a2[c]]`), paths (`/a/../b`).
- Evaluate expressions (RPN, calculators).
- "Remove adjacent duplicates", "asteroid collision". A new item interacts with the most recent survivor.
- Iterative DFS / simulating recursion.

**Triggers (monotonic stack)**
- "Next greater / smaller element", "days until warmer", "stock span".
- "For each element, how far can it extend left/right while it's the min/max" → largest rectangle, sum of subarray minimums/ranges, maximal rectangle.
- "Remove k digits to make the smallest number", "smallest subsequence with distinct characters". Greedy with a monotonic stack.
- "Car fleet", "buildings with an ocean view". Dominance from one side.

**Triggers (monotonic deque)**
- "Max/min of every window of size k".
- "Longest subarray where max − min ≤ limit".
- DP optimization: `dp[i] = max(dp[j]) + … for j in [i−k, i)`.

**Anti-signals**
- Need the k-th largest overall, not the nearest dominating element → heap.
- Queries on arbitrary ranges (not "nearest") → segment tree / sparse table.
- "Next greater" in a *circular* array still works: iterate 2n with `i % n`. Don't reject the pattern.

## 3. Mental model
**People in a line looking right, tallest blocks the view.** Scan left to right, keeping a stack of people still "waiting" for someone taller. When a taller person arrives, everyone shorter at the top of the stack has found their answer and leaves.

```
temps:  73  74  75  71  69  72  76
i=0 push 73               stack: [73]
i=1 74 > 73: 73 -> 1 day  stack: [74]
i=2 75 > 74: 74 -> 1      stack: [75]
i=3 push 71               stack: [75,71]
i=4 push 69               stack: [75,71,69]     (decreasing = still waiting)
i=5 72 pops 69 (1), 71 (2) stack: [75,72]
i=6 76 pops 72 (1), 75 (4) stack: [76]
```
The stack is always **decreasing** from bottom to top. That's the "monotonic" invariant.

**Which monotonicity?** To find the next **greater**, keep a **decreasing** stack (pop smaller ones). To find the next **smaller**, keep an **increasing** stack.

## 4. Canonical templates

```python
from collections import deque

# 4.1 Next greater element (to the right). Store INDICES, not values.
def next_greater(nums: list[int]) -> list[int]:
    res = [-1] * len(nums)
    stack: list[int] = []                    # indices; nums[stack] strictly decreasing
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            res[stack.pop()] = x             # x is the first greater element to its right
        stack.append(i)
    return res                               # unresolved indices stay -1

# 4.2 Previous smaller element (to the left): answer is whatever survives on top
def prev_smaller_index(nums: list[int]) -> list[int]:
    res = [-1] * len(nums)
    stack: list[int] = []                    # nums[stack] strictly increasing
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] >= x:
            stack.pop()
        res[i] = stack[-1] if stack else -1
        stack.append(i)
    return res

# 4.3 Contribution technique: sum of min over all subarrays (LC 907)
def sum_subarray_mins(arr: list[int]) -> int:
    MOD = 10**9 + 7
    a = arr + [float('-inf')]                # sentinel flushes the stack at the end
    stack: list[int] = []                    # strictly increasing values
    res = 0
    for i, x in enumerate(a):
        while stack and a[stack[-1]] >= x:   # pop on >=: right boundary is "next <="
            j = stack.pop()
            left = stack[-1] if stack else -1   # previous strictly smaller
            res += a[j] * (j - left) * (i - j)  # a[j] is the min of (j-left)*(i-j) subarrays
        stack.append(i)
    return res % MOD

# 4.4 Monotonic deque: max of each window of size k (Example 3)

# 4.5 Plain stack: bracket matching
def is_valid(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for c in s:
        if c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
        else:
            stack.append(c)
    return not stack

# 4.6 Min stack: store (value, min_so_far) so getMin is O(1)
class MinStack:
    def __init__(self) -> None:
        self.st: list[tuple[int, int]] = []
    def push(self, val: int) -> None:
        self.st.append((val, min(val, self.st[-1][1]) if self.st else val))
    def pop(self) -> None:
        self.st.pop()
    def top(self) -> int:
        return self.st[-1][0]
    def getMin(self) -> int:
        return self.st[-1][1]
```
**Trace 4.3** arr=[3,1,2,4]: i=1 (1) pops 3: left=−1 → 3·1·1=3. i=4 (sentinel) pops 4: 4·1·1=4; pops 2: left=1 → 2·1·2=4; pops 1: left=−1 → 1·2·3=6. Total 17 ✓.
**Duplicates:** pop on `>=` makes the right boundary "next ≤" while the left boundary is "previous <". Asymmetric boundaries count each subarray exactly once.

## 5. Worked examples

### Example 1 — Daily Temperatures (LC 739) — Medium
**Brute force:** for each day scan forward for a warmer one → O(n²) on a decreasing sequence.
**Bottleneck:** a cold day gets re-scanned past by every earlier colder day.
**Aha (signal: "next warmer" = next greater):** a decreasing stack of unresolved days. A warmer day resolves all colder days on top.
```python
def daily_temperatures(temps: list[int]) -> list[int]:
    res = [0] * len(temps)
    stack: list[int] = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```
**Complexity:** O(n) time (each index pushed and popped at most once), O(n) space.
**Edge cases:** strictly decreasing → all 0. Equal temperatures are *not* warmer (strict `<`).

### Example 2 — Largest Rectangle in Histogram (LC 84) — Hard
**Brute force:** for each bar i as the limiting height, expand left and right while bars ≥ h[i] → O(n²).
**Bottleneck:** the expansion is exactly "previous smaller" and "next smaller" for each bar.
**Aha (signal: "how far can each bar extend while it's the minimum"):** an increasing stack. When bar j is popped by i (h[i] < h[j]), i is j's next smaller and the new stack top is j's previous smaller. So its width is known at that moment.
```python
def largest_rectangle_area(heights: list[int]) -> int:
    hs = heights + [0]                         # sentinel pops everything at the end
    stack: list[int] = []                      # indices, heights increasing
    best = 0
    for i, h in enumerate(hs):
        while stack and hs[stack[-1]] >= h:
            height = hs[stack.pop()]
            left = stack[-1] if stack else -1  # previous smaller index
            best = max(best, height * (i - left - 1))
        stack.append(i)
    return best
```
**Trace** [2,1,5,6,2,3]: i=1 pops 2 → 2·1=2. i=4 (2) pops 6 → 6·1=6, pops 5 → 5·2=10. i=6 (sentinel) pops 3 → 3·1, pops 2 (idx 4, left=1) → 2·4=8, pops 1 → 1·6=6. Best 10 ✓.
**Complexity:** O(n) time, O(n) space.
**Edge cases:** empty → 0. All equal (the `>=` pop handles duplicates: the last equal bar gets the full width). Strictly increasing (everything is popped by the sentinel).
**Extension:** Maximal Rectangle (LC 85) = run this per row on the accumulated column heights → O(R·C).

### Example 3 — Sliding Window Maximum (LC 239) — Hard
**Brute force:** max of each window → O(n·k). **Heap:** O(n log n) with lazy deletion of expired indices.
**Aha (signal: window max, and a new element makes older smaller ones useless):** if `nums[j] ≤ nums[i]` with j < i, then j can **never** be a window max again. i is bigger and leaves the window later. So keep a deque of indices with *decreasing* values. The front is the current max. Evict the front when it leaves the window.
```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    dq: deque[int] = deque()                  # indices; nums[dq] strictly decreasing
    res: list[int] = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:       # dominated -> useless forever
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                    # front fell out of the window [i-k+1, i]
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```
**Trace** [1,3,−1,−3,5,3,6,7], k=3: windows end at i=2..7 → 3,3,5,5,6,7 ✓.
**Complexity:** O(n) time (each index enters and leaves the deque once), O(k) space.
**Edge cases:** k = 1 → the array itself. k = n → one value. Equal values: `<=` keeps only the newest, which is also correct.

## 6. Common mistakes & off-by-one traps
- Storing values instead of indices. You usually need distances or window checks.
- Wrong direction: "next greater" needs a *decreasing* stack. Say the invariant out loud before coding.
- Strict vs non-strict comparisons with duplicates (next greater *or equal*? contribution double counting?).
- Forgetting the leftover stack at the end → use a sentinel or post-process (leftovers = "no answer").
- Monotonic deque: checking expiry with `<` instead of `<=` (`dq[0] <= i − k`), or appending results before the first full window.
- Circular arrays: iterate `for i in range(2n)` using `nums[i % n]`, and only push during the first pass (or just push, since it's harmless).
- Using `list.pop(0)` as a queue. Use `deque`.

## 7. Connections
- **Greedy** (Ch 12): Remove K Digits and Remove Duplicate Letters are greedy decisions implemented with a monotonic stack ("pop the bigger earlier digit if we can afford it").
- **DP optimization**: a monotonic deque turns `max over a sliding range of dp` from O(k) to O(1) amortized (LC 1696 Jump Game VI, LC 862).
- **Prefix sums** (Ch 5): LC 862 = a monotonic deque over prefix sums.
- **Trees**: an iterative inorder traversal is a stack. Building a Cartesian tree / "max tree" is a monotonic stack.
- **Histogram trick**: Maximal Rectangle = histogram per row (Ch 5's "reduce 2D to 1D").
- Mistaken for: heap (window max, and the deque is strictly better); DP (Trapping Rain Water also has a monotonic-stack solution that fills water by layers).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 20 | [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) | Easy | Most recent open bracket must close first |
| 496 | [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) | Easy | "Next greater" → decreasing stack + map |
| 155 | [Min Stack](https://leetcode.com/problems/min-stack/) | Medium | O(1) min under push/pop → store min alongside |
| 150 | [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/) | Medium | Operators apply to the two most recent operands |
| 739 | [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) | Medium | Days until warmer = next greater |
| 503 | [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) | Medium | Circular → iterate twice |
| 901 | [Online Stock Span](https://leetcode.com/problems/online-stock-span/) | Medium | Previous greater, streaming → collapse spans |
| 394 | [Decode String](https://leetcode.com/problems/decode-string/) | Medium | Nested `k[...]` → stack of (prefix, k) |
| 735 | [Asteroid Collision](https://leetcode.com/problems/asteroid-collision/) | Medium | A new item fights the most recent survivor |
| 853 | [Car Fleet](https://leetcode.com/problems/car-fleet/) | Medium | Sort by position; arrival times dominated from the front |
| 402 | [Remove K Digits](https://leetcode.com/problems/remove-k-digits/) | Medium | Smallest number → pop larger earlier digits greedily |
| 907 | [Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/) | Medium | Sum over all subarrays of min → contribution |
| 84 | [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) | Hard | Each bar's extent = previous/next smaller |
| 239 | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | Hard | Window max → decreasing deque |
| 85 | [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/) | Hard | Binary matrix rectangle → histogram per row |

## 9. Recognition drill
1. For each building, count how many buildings to its right it can see (a view is blocked by any building at least as tall).
2. Given a string of digits and k, remove k digits to make the largest possible number.
3. Given an array, compute Σ (max − min) over all subarrays.
4. Given nums and k, you start at index 0 and can jump 1..k steps; the score is the sum of visited values. Maximize the score at the last index (n up to 10⁵).
5. Simplify a Unix path like `/a/./b/../../c/`.

<details><summary>Answers</summary>

1. **Monotonic stack scanning from the right** (decreasing): count pops + 1 if a blocker remains (LC 1944 variant).
2. **Greedy + monotonic stack** (pop a smaller earlier digit while k > 0 and the new digit is bigger). Mirror of Remove K Digits.
3. **Contribution technique twice**: Σ max via next/previous greater, minus Σ min via next/previous smaller (LC 2104).
4. **DP + monotonic deque**: `dp[i] = nums[i] + max(dp[i−k..i−1])`, sliding-window max over dp (LC 1696).
5. **Plain stack** of directory names: `..` pops, `.` and empty are ignored.
</details>

Next: Part 8 — Linked Lists
