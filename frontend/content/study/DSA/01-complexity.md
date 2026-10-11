# Part 1 — Complexity Analysis

## 1. Core insight
Complexity measures how the count of the **dominant operation** grows with input size, ignoring constants and lower-order terms. There are three tools. **Loop counting**: multiply nested ranges, add sequential blocks. **Recursion trees**: sum the work over all nodes. **Amortization**: total cost over a sequence of operations divided by the count. Amortization is why "a while loop inside a for loop" is often still O(n). In interviews the skill is spotting *hidden* costs and *proving* amortized bounds in one sentence.

## 2. Trigger signals / Anti-signals
**Use loop counting when:** loops have independent ranges.
**Use amortized reasoning when:**
- An inner `while` moves a pointer that **never resets** (sliding window, two pointers). Total inner iterations ≤ n.
- Each element is pushed and popped **at most once** (monotonic stack, queue). Total work is O(n).
- A dynamic array doubles its capacity. Each append is O(1) amortized.
- Union-Find with path compression and union by rank gives O(α(n)) per operation.

**Use recursion trees / Master theorem when:** the function calls itself on smaller inputs.

**Anti-signals (traps that look O(1) but aren't):**
- `x in list` → O(n). `x in set/dict` → O(1) average.
- `list.pop(0)`, `list.insert(0, x)` → O(n). Use `collections.deque`.
- `s[i:j]`, `arr[:]`, `list(arr)`, `sorted()` → O(length) or O(k log k), **inside a loop they multiply**.
- `s += c` in a loop → O(n²) worst case in general (CPython sometimes optimizes it; don't rely on that). Use `''.join(parts)`.
- Hashing a tuple or string of length k is O(k), not O(1).
- `max(arr)`, `sum(arr)`, `arr.index(x)`, `arr.count(x)` → O(n).
- Recursion uses **stack space** O(depth). Python's default recursion limit is 1000.
- `heapq.heapify` is O(n), not O(n log n).

## 3. Mental model

A recursion tree: each level's work, summed over all levels.

```
T(n) = 2T(n/2) + n          (merge sort)

level 0:            n                      -> n
level 1:       n/2     n/2                 -> n
level 2:    n/4 n/4  n/4 n/4               -> n
...                                         ...
level log n: 1 1 1 ... 1 (n leaves)        -> n
                                  total  = n · log n
```

```
T(n) = 2T(n-1) + 1          (naive subsets / fib-like)

depth n, branching 2 -> 2^n nodes, O(1) each -> O(2^n)
```

**Rule for recursion:** total = (number of nodes) × (work per node), or sum level by level when the work differs by level.
**Rule for memoized recursion / DP:** total = (number of distinct states) × (work per transition).

## 4. Canonical templates

### 4.1 Master theorem
For `T(n) = a·T(n/b) + O(n^d)` with a ≥ 1 and b > 1:

| Case | Condition | Result | Intuition |
|---|---|---|---|
| 1 | d > log_b a | O(n^d) | Root work dominates |
| 2 | d = log_b a | O(n^d log n) | Every level does equal work |
| 3 | d < log_b a | O(n^(log_b a)) | Leaves dominate |

### 4.2 Recurrences to know by heart

| Recurrence | Result | Example |
|---|---|---|
| T(n) = T(n/2) + O(1) | O(log n) | Binary search, fast power |
| T(n) = T(n/2) + O(n) | O(n) | Quickselect (average), halving scans |
| T(n) = 2T(n/2) + O(1) | O(n) | Tree traversal (balanced) |
| T(n) = 2T(n/2) + O(n) | O(n log n) | Merge sort, quicksort (average) |
| T(n) = T(n−1) + O(1) | O(n) | Linear recursion |
| T(n) = T(n−1) + O(n) | O(n²) | Quicksort worst case, selection sort |
| T(n) = 2T(n−1) + O(1) | O(2ⁿ) | Naive subsets, Tower of Hanoi |
| T(n) = n·T(n−1) | O(n!) | Permutations |

### 4.3 Summations to know

| Sum | Value | Where it appears |
|---|---|---|
| 1 + 2 + … + n | O(n²) | `for i: for j in range(i)` |
| 1 + 2 + 4 + … + n | O(n) | Doubling array, heap build |
| n/1 + n/2 + n/3 + … + n/n | O(n log n) | `for i in 1..n: for j in range(i, n, i)` (harmonic) |
| Σ over primes p ≤ n of n/p | O(n log log n) | Sieve of Eratosthenes |

### 4.4 Python operation costs (average case)

| Operation | list | dict / set | deque | heapq |
|---|---|---|---|---|
| index / get | O(1) | O(1) | O(n) middle, O(1) ends | — |
| append / add | O(1) amortized | O(1) | O(1) both ends | push O(log n) |
| pop from end | O(1) | — | O(1) both ends | pop O(log n) |
| pop(0) / insert(0) | **O(n)** | — | O(1) | — |
| `in` | **O(n)** | O(1) | O(n) | O(n) |
| sort | O(n log n) | — | — | heapify O(n) |

## 5. Worked examples

### Example 1 — Pow(x, n) (LC 50) — Easy/Medium
**Problem:** compute xⁿ, where n can be up to 2³¹ and may be negative.

**Brute force:** multiply n times → O(n). With n = 2³¹ that's about 2·10⁹ operations. Too slow.

**Bottleneck:** we recompute the same products. x⁸ = x⁴·x⁴, but the brute force builds x⁴ twice.

**Aha (signal: n ≈ 10⁹ → need O(log n)):** halve the exponent each step. Recurrence T(n) = T(n/2) + O(1) → O(log n).

```python
def my_pow(x: float, n: int) -> float:
    if n < 0:
        x, n = 1 / x, -n
    result = 1.0
    # Invariant: result * x**n == original answer
    while n:
        if n & 1:          # odd exponent: peel off one factor
            result *= x
        x *= x             # square the base
        n >>= 1            # halve the exponent
    return result
```
**Trace** (x=2, n=10 = 0b1010): n=10 even → x=4, n=5 · odd → result=4, x=16, n=2 · even → x=256, n=1 · odd → result=1024, n=0. ✓

**Complexity:** O(log n) time (one iteration per bit), O(1) space.
**Edge cases:** n=0 → 1. n negative. x=0 with n<0 is undefined (clarify with the interviewer).

### Example 2 — Proving amortized O(n) on a sliding window — Medium
**Problem:** analyze this code (longest subarray with sum ≤ k, non-negative numbers):

```python
def longest_at_most_k(nums: list[int], k: int) -> int:
    left = total = best = 0
    for right in range(len(nums)):
        total += nums[right]
        while total > k:            # nested loop!
            total -= nums[left]
            left += 1
        best = max(best, right - left + 1)
    return best
```

**Naive analysis:** outer loop n × inner loop up to n → "O(n²)". **Wrong.**

**Aha (signal: the inner pointer never resets):** `left` only increases and never exceeds n. So across the *entire* run, the inner body executes at most n times in total. Total work = n (outer) + n (inner, summed over all iterations) = O(n).

**The general proof pattern (potential/aggregate argument):** find a quantity that (a) each inner iteration changes monotonically, and (b) is bounded. Here `left` goes 0 → n. For a monotonic stack: each index is pushed once and popped at most once, so there are ≤ 2n operations total.

**Complexity:** O(n) time, O(1) space.

### Example 3 — Recursion tree with non-constant work per node — Medium/Hard
**Problem:** what is the complexity of generating all subsets like this?

```python
def subsets(nums: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []
    def dfs(i: int) -> None:
        if i == len(nums):
            res.append(path[:])      # copy: O(len(path))
            return
        path.append(nums[i]); dfs(i + 1); path.pop()   # take nums[i]
        dfs(i + 1)                                       # skip nums[i]
    dfs(0)
    return res
```

**Thinking:** the tree has depth n and branching 2, so there are 2ⁿ leaves and 2ⁿ⁺¹ − 1 nodes. Internal nodes do O(1) work. Each leaf copies a path of length up to n.
- Time: 2ⁿ leaves × O(n) copy = **O(n · 2ⁿ)**. The internal-node cost O(2ⁿ) is dominated.
- Space: O(n) recursion depth + path, **excluding the output**. The output itself is O(n · 2ⁿ).

**Aha:** the output size is a lower bound. When you must return 2ⁿ subsets, O(n·2ⁿ) is optimal. Always say this in interviews: "this is output-sensitive; we can't beat the output size."

**Contrast with memoized recursion** (the DP rule): naive `fib(n) = fib(n-1) + fib(n-2)` has a tree with O(φⁿ) ≈ O(1.618ⁿ) nodes. With memoization there are n+1 distinct states and O(1) work each → O(n). The complexity of memoized recursion is **states × transition cost**, not the size of the tree.

## 6. Common mistakes & traps
- Calling a nested loop O(n²) when the inner pointer is amortized (Example 2), or calling it O(n) when the inner pointer *does* reset.
- Forgetting the cost of copying (`path[:]`, slicing, `str` building) at the leaves.
- Forgetting the recursion stack in space complexity (DFS on a skewed tree is O(n) space, not O(1)).
- Treating hash operations as O(1) worst case. They're O(1) *average*. Say "expected O(1)" if pushed.
- Sorting inside a loop: `for q in queries: sorted(...)` → O(q · n log n).
- Two-variable inputs: say O(m + n) or O(m · n), not O(n). Graphs are **O(V + E)**, grids are **O(R · C)**.
- Log base doesn't matter in big-O (log₂ n = log₃ n / log₃ 2), but the **exponential base does**: 3ⁿ is *not* O(2ⁿ) (the ratio (3/2)ⁿ is unbounded).
- Heap: `heappush` n times → O(n log n). `heapify` → O(n).
- String concatenation in a loop.

## 7. Connections
- **DP:** complexity = states × transition. This is how you *choose* DP state definitions (fewer states → faster).
- **Binary search on answer:** O(log(range) × cost of check). Say both factors.
- **Backtracking:** O(branching^depth × leaf work); pruning improves the practical bound, not always the worst case.
- **Amortized** analysis is the backbone of sliding window (Ch 4), monotonic stack (Ch 7), and Union-Find (Ch 15).
- **Master theorem** explains merge sort / count-inversions (Ch 21) and quickselect (Ch 11).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 509 | [Fibonacci Number](https://leetcode.com/problems/fibonacci-number/) | Easy | Exponential recursion tree; count distinct states instead |
| 70 | [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | Easy | Same recurrence as Fibonacci; states × transition |
| 1 | [Two Sum](https://leetcode.com/problems/two-sum/) | Easy | The inner loop is a *search*; what makes search O(1)? |
| 69 | [Sqrt(x)](https://leetcode.com/problems/sqrtx/) | Easy | x up to 2³¹ → linear is too slow |
| 367 | [Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/) | Easy | Monotone predicate over a huge range |
| 50 | [Pow(x, n)](https://leetcode.com/problems/powx-n/) | Medium | n ≈ 2³¹ forces logarithmic time |
| 912 | [Sort an Array](https://leetcode.com/problems/sort-an-array/) | Medium | Derive T(n) = 2T(n/2) + O(n) yourself |
| 204 | [Count Primes](https://leetcode.com/problems/count-primes/) | Medium | Harmonic-style sum over multiples |
| 215 | [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) | Medium | T(n) = T(n/2) + O(n) on average |
| 372 | [Super Pow](https://leetcode.com/problems/super-pow/) | Medium | Exponent given digit by digit; modular fast power |
| 239 | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | Hard | Each index enters and leaves a structure once (amortized) |
| 4 | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | Hard | Required O(log(m+n)) → you must discard half each step |

## 9. Recognition drill
State time and space for each.

1. `for i in range(n): for j in range(i, n, i + 1): work()` (O(1) work).
2. `def f(n): return 1 if n <= 1 else f(n // 2) + f(n // 2)` (note: two calls).
3. A loop over n strings, each of length L, inserting each string into a set.
4. `for i in range(n): stack-based loop where each index is pushed once and popped at most once`.
5. Memoized `dp(i, j)` with 0 ≤ i < n, 0 ≤ j < m, where each call loops over k in range(j).

<details><summary>Answers</summary>

1. The inner loop runs about n/(i+1) times → Σ n/(i+1) = O(n log n) (harmonic). Space O(1).
2. T(n) = 2T(n/2) + O(1) → Master case 3 (d=0 < log₂2=1) → O(n) time; O(log n) stack space. (With one call cached it would be O(log n).)
3. Hashing a string costs O(L) → O(n·L) time, O(n·L) space.
4. Amortized O(n) time (≤ 2n push/pop operations), O(n) space.
5. States n·m, transition O(m) → O(n·m²) time, O(n·m) space for the memo.
</details>

Next: Part 2 — Arrays & Hashing
