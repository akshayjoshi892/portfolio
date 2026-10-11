# Part 5 — Prefix Sums & Difference Arrays

## 1. Core insight
Precompute `P[i] = a[0] + … + a[i−1]`. Then any range sum is a **difference of two prefixes**: `sum(i..j) = P[j+1] − P[i]`. This does two things. Range queries drop from O(n) to O(1). And "find a subarray with property X" becomes "find a **pair of prefixes** with property X". That pair search is Two Sum, solved with a hash map in one pass, **and it works with negative numbers**, which sliding window can't handle. The difference array is the inverse operation: to add v to a whole range in O(1), mark `+v` at the start and `−v` after the end, and recover the values with one prefix pass. The idea generalizes to any **invertible** associative operation: XOR, counts, and products (when there are no zeros).

## 2. Trigger signals / Anti-signals
**Triggers**
- Many range-sum queries on a **static** array/grid.
- "Subarray with sum = k", "sum divisible by k", "equal number of 0s and 1s", "count subarrays with sum …" — especially with **negatives**.
- "Longest subarray with sum k" → prefix + first-occurrence map.
- Many **range updates** followed by final reads ("booking", "flights", "car pooling", "shift letters in ranges").
- "Pivot / equilibrium index", "left sum equals right sum".
- 2D: "sum of submatrix", repeated rectangle queries.
- XOR of a subarray → prefix XOR.

**Anti-signals**
- Values **change between queries** → Fenwick/segment tree (Ch 21).
- Range **min/max** queries → not invertible, so prefixes don't work. Use a sparse table, segment tree, or monotonic structures.
- Non-negative values + longest/shortest with a threshold (not equality) → sliding window is O(1) space and simpler.
- Updates interleaved with reads → a difference array needs a Fenwick tree (range update, point query).

## 3. Mental model
```
index:    0   1   2   3   4
a:        3  -1   4   2  -2
P:     0  3   2   6   8   6        (P has n+1 entries; P[0] = 0 = empty prefix)
          ^--- sum(1..3) = P[4] - P[1] = 8 - 3 = 5  (= -1 + 4 + 2)
```
**Subarray-sum-equals-k as Two Sum:** we need i < j with `P[j] − P[i] = k`, i.e. `P[i] = P[j] − k`. Sweep j, and ask the hash map "how many earlier prefixes equal `P[j] − k`?"

**Difference array** — a range update touches only two points:
```
add 5 to [1..3] in length 6:
diff:  0  +5   0   0  -5   0
prefix: 0  5   5   5   0   0
```

## 4. Canonical templates

```python
from collections import defaultdict
from itertools import accumulate

# 4.1 1D prefix sums
def build_prefix(a: list[int]) -> list[int]:
    P = [0] * (len(a) + 1)
    for i, x in enumerate(a):
        P[i + 1] = P[i] + x
    return P            # sum(a[i..j]) inclusive = P[j+1] - P[i]
# One-liner: P = [0, *accumulate(a)]

# 4.2 2D prefix sums (inclusion-exclusion)
class NumMatrix:
    def __init__(self, M: list[list[int]]):
        R, C = len(M), len(M[0])
        self.P = [[0] * (C + 1) for _ in range(R + 1)]
        for r in range(R):
            for c in range(C):
                self.P[r + 1][c + 1] = M[r][c] + self.P[r][c + 1] + self.P[r + 1][c] - self.P[r][c]

    def sum_region(self, r1: int, c1: int, r2: int, c2: int) -> int:
        P = self.P
        return P[r2 + 1][c2 + 1] - P[r1][c2 + 1] - P[r2 + 1][c1] + P[r1][c1]

# 4.3 Count subarrays with sum == k (works with negatives)
def subarray_sum(nums: list[int], k: int) -> int:
    count: dict[int, int] = defaultdict(int)
    count[0] = 1                  # empty prefix: lets subarrays starting at index 0 count
    s = res = 0
    for x in nums:
        s += x
        res += count[s - k]       # earlier prefixes P[i] with P[j] - P[i] = k
        count[s] += 1             # record AFTER querying (a subarray must be non-empty)
    return res

# 4.4 Longest subarray with sum == k: store FIRST occurrence of each prefix
def longest_subarray_sum_k(nums: list[int], k: int) -> int:
    first = {0: -1}               # prefix sum 0 "occurs" before index 0
    s = best = 0
    for j, x in enumerate(nums):
        s += x
        if s - k in first:
            best = max(best, j - first[s - k])
        if s not in first:        # keep the earliest index to maximize length
            first[s] = j
    return best

# 4.5 Divisibility: sum(i..j) % k == 0  <=>  P[j+1] % k == P[i] % k
def subarrays_div_by_k(nums: list[int], k: int) -> int:
    count = defaultdict(int); count[0] = 1
    s = res = 0
    for x in nums:
        s = (s + x) % k           # Python's % is non-negative for k > 0
        res += count[s]
        count[s] += 1
    return res

# 4.6 Difference array: many range adds, then read
def range_add(n: int, updates: list[tuple[int, int, int]]) -> list[int]:
    diff = [0] * (n + 1)
    for l, r, v in updates:       # add v to a[l..r] inclusive
        diff[l] += v
        diff[r + 1] -= v
    return list(accumulate(diff[:n]))
```
**Trace 4.3** nums=[1,2,3], k=3: s=1: count[−2]=0 → {0:1,1:1}. s=3: count[0]=1 → res=1 ([1,2]) → {…,3:1}. s=6: count[3]=1 → res=2 ([3]). ✓

**Trace 4.6** (Car Pooling style): trips (2 passengers, 1→5), (3 passengers, 3→7), capacity 4. diff[1]+=2, diff[5]−=2, diff[3]+=3, diff[7]−=3 → load over time: 0,2,2,5 at t=3 > 4 → infeasible. ✓ (In Car Pooling, `to` is the drop-off point, so the range is [from, to) and you subtract at `to`, not `to+1`.)

## 5. Worked examples

### Example 1 — Find Pivot Index (LC 724) — Easy
**Problem:** index where the sum of elements to its left equals the sum to its right.
**Brute force:** for each i, sum both sides → O(n²).
**Bottleneck:** recomputing left and right sums from scratch.
**Aha (lever: precompute):** `right = total − left − nums[i]`. So one running `left` sum suffices.
```python
def pivot_index(nums: list[int]) -> int:
    total, left = sum(nums), 0
    for i, x in enumerate(nums):
        if left == total - left - x:
            return i
        left += x
    return -1
```
**Complexity:** O(n) time, O(1) space.
**Edge cases:** pivot at index 0 (left sum is 0). Negative numbers. No pivot → −1.

### Example 2 — Subarray Sum Equals K (LC 560) — Medium
**Brute force:** all (i, j) with a running sum → O(n²). Too slow for n = 2·10⁴ in Python and not the expected answer.
**Why not a sliding window?** Values can be negative, so extending the window can decrease the sum and validity isn't monotone. That's the anti-signal.
**Aha (signal: "subarray sum = k" + negatives):** `sum(i..j) = P[j+1] − P[i] = k` ⇒ count earlier prefixes equal to `P[j+1] − k`. This is Two Sum on prefix sums. Code: template 4.3.
**Complexity:** O(n) time, O(n) space.
**Edge cases:** k = 0 with zeros `[0,0,0]` → 6 (every non-empty subarray). The `count[0] = 1` seed handles subarrays that start at index 0. Querying before inserting prevents counting empty subarrays.

### Example 3 — Number of Submatrices That Sum to Target (LC 1074) — Hard
**Problem:** count submatrices (any rectangle) whose sum equals target. R, C ≤ 100.

**Brute force:** O(R²C²) rectangles × O(1) via 2D prefix → about 10⁸. That's borderline or too slow in Python.
**Aha (reduce 2D to the 1D problem you already know):** fix a pair of rows (top, bottom). Collapse the rows between them into a 1D array of column sums. Every rectangle between those rows is a contiguous subarray of that array. Now it's exactly LC 560. Cost: O(R²) row pairs × O(C) → O(R²·C) = 10⁶.

```python
from collections import defaultdict

def num_submatrix_sum_target(matrix: list[list[int]], target: int) -> int:
    R, C = len(matrix), len(matrix[0])
    if R > C:                                    # iterate pairs over the smaller dimension
        matrix = [list(row) for row in zip(*matrix)]
        R, C = C, R
    res = 0
    for top in range(R):
        col = [0] * C                            # col[c] = sum of matrix[top..bottom][c]
        for bottom in range(top, R):
            for c in range(C):
                col[c] += matrix[bottom][c]
            count = defaultdict(int); count[0] = 1   # LC 560 on `col`
            s = 0
            for v in col:
                s += v
                res += count[s - target]
                count[s] += 1
    return res
```
**Complexity:** O(min(R,C)² · max(R,C)) time, O(max(R,C)) space.
**Edge cases:** 1×1 matrix. Negative entries (a sliding window would be wrong here too). target = 0.
**Transferable pattern:** *"fix two boundaries in one dimension, reduce to a 1D problem in the other."* The same move solves max-sum rectangle (Kadane on column sums) and Maximal Rectangle (histogram per row).

## 6. Common mistakes & off-by-one traps
- Mixing up conventions. Pick `P` of length n+1 with `P[0] = 0` and **always** use `P[j+1] − P[i]` for inclusive [i..j].
- Forgetting `count[0] = 1` (or `first[0] = −1`) → misses subarrays that start at index 0.
- Inserting the current prefix *before* querying → counts empty subarrays when k = 0.
- For "longest", overwriting the first-occurrence index → shorter answers.
- Mod with negatives: Python's `%` already returns a non-negative result for k > 0. In Java/C++ you need `((s % k) + k) % k`.
- 2D inclusion-exclusion: sign errors. Always add back the doubly-subtracted corner `P[r1][c1]`.
- Difference array bounds: allocate n+1 so `diff[r+1]` is safe. Know whether the range end is inclusive.
- "Equal 0s and 1s" (LC 525): map 0 → −1 first, then it's "longest subarray with sum 0".

## 7. Connections
- **Hash map** (Ch 2): prefix + hash is the canonical combination.
- **Sliding window** (Ch 4): prefer it for non-negative values with inequality constraints. Use prefix sums for equality or negatives.
- **Binary search** (Ch 6): the prefix sums of positive arrays are sorted → binary search for "first prefix ≥ x" (Random Pick with Weight, LC 528).
- **Monotonic deque** on prefix sums: shortest subarray with sum ≥ k with negatives (LC 862).
- **Kadane's algorithm** is "max over j of P[j] − min over i<j of P[i]" in disguise.
- **Fenwick tree** (Ch 21) = prefix sums with updates. **Line sweep** (Ch 13/21) = difference array over sorted events.
- **Tree path sums** (LC 437 Path Sum III) = prefix sum + hash map along a root-to-node path, with backtracking.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 1480 | [Running Sum of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/) | Easy | Definition of a prefix sum |
| 303 | [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) | Easy | Many static range queries |
| 724 | [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/) | Easy | Left sum vs right sum → total − left − x |
| 238 | [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) | Medium | Prefix and suffix products, no division |
| 304 | [Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/) | Medium | Rectangle queries → 2D inclusion-exclusion |
| 560 | [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) | Medium | Equality + negatives → prefix pairs in a hash map |
| 525 | [Contiguous Array](https://leetcode.com/problems/contiguous-array/) | Medium | Equal 0s and 1s → map 0→−1, longest sum-0 |
| 523 | [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/) | Medium | Multiple of k → equal prefix remainders, length ≥ 2 |
| 974 | [Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/) | Medium | Count equal remainders |
| 930 | [Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/) | Medium | Count with sum = goal → prefix counts (or atMost trick) |
| 1094 | [Car Pooling](https://leetcode.com/problems/car-pooling/) | Medium | Range adds, then check max → difference array |
| 1109 | [Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/) | Medium | Many range bookings → difference array |
| 437 | [Path Sum III](https://leetcode.com/problems/path-sum-iii/) | Medium | Downward path sum = k → prefix sums along the root path |
| 1074 | [Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/) | Hard | 2D count → fix two rows, reduce to 560 |
| 862 | [Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/) | Hard | Negatives + "at least" → monotonic deque over prefixes |

## 9. Recognition drill
1. Given an array and many queries (l, r), return the XOR of `a[l..r]` for each.
2. Given a binary string, find the longest substring with more 1s than 0s.
3. Given n lamps and m operations "toggle brightness +v for lamps l..r", return the final brightness of each lamp.
4. Given an integer array, count subarrays whose sum lies in [lower, upper].
5. Given an array, for each index return the product of all other elements, without division.

<details><summary>Answers</summary>

1. **Prefix XOR**: `X[r+1] ^ X[l]` (XOR is its own inverse).
2. **Prefix sum with 1→+1, 0→−1**: longest j−i with `P[j] > P[i]`. If `P[j] > 0` the whole prefix works; otherwise use the first occurrence of `P[j] − 1` (values change by ±1, so if any smaller prefix exists, `P[j] − 1` occurred first) → hash map O(n).
3. **Difference array**: O(1) per operation, one prefix pass.
4. **Prefix sums + counting pairs with P[j] − P[i] in [lower, upper]** → needs order statistics: merge sort count or Fenwick tree over compressed prefixes (LC 327). A hash map alone can't handle ranges.
5. **Prefix × suffix products** (two passes, O(1) extra space if you reuse the output array).
</details>

Next: Part 6 — Binary Search
