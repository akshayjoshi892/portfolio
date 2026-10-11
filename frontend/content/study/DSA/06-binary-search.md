# Part 6 — Binary Search (on Index and on Answer Space)

## 1. Core insight
Binary search does not need a sorted array. It needs a **monotone predicate** over an ordered domain: `pred(x)` is False…False, True…True. Then one evaluation at the midpoint discards half the domain, so O(log |domain|) evaluations find the boundary. The domain can be array indices (**search on index**) or the space of possible answers (**search on answer**). In the second case, "find the minimum k such that feasible(k)" turns an optimization problem into O(log range) *decision* problems, and decisions are usually easy (often a greedy O(n) check).

## 2. Trigger signals / Anti-signals
**Triggers (index)**
- Sorted array / matrix, "find the position / insert position / first or last occurrence".
- "Rotated sorted array", "peak element", "bitonic". Some half is always structured.
- Required complexity O(log n).

**Triggers (answer space)**
- "**Minimize the maximum**" / "**maximize the minimum**".
- "Minimum speed / capacity / days / time such that…".
- "k-th smallest" where you can count "how many ≤ x" quickly (multiplication table, pair distances, sorted matrix).
- The answer range is huge (up to 10⁹) while checking one candidate is O(n).
- Feasibility is obviously monotone: "if speed k works, any faster speed works".

**Anti-signals**
- Predicate is not monotone (e.g., "exactly k", or feasibility that goes up and down) → binary search silently gives wrong answers.
- Unsorted data where sorting destroys needed information (original indices / order matters).
- Rotated arrays **with duplicates** → worst case degrades to O(n) (`[1,1,1,0,1]`). Mention it.
- A direct O(n) greedy/DP computes the optimum without guessing. Binary search on the answer is then an unnecessary log factor. It's still acceptable, but say so.

## 3. Mental model
Think of the domain as a row of lights: **off off off off ON ON ON**. You want the first ON. Every probe at `mid` tells you which side of the boundary you're on.

```
index:  0  1  2  3  4  5  6
pred:   F  F  F  F  T  T  T
        lo                  hi (exclusive)
mid=3 -> F -> lo=4
mid=5 -> T -> hi=5
mid=4 -> T -> hi=4  -> lo == hi == 4  (first True)
```
**Search on answer**: the "array" is the virtual list `[feasible(lo), …, feasible(hi)]`. You never build it. You evaluate `feasible(mid)` on demand.

## 4. Canonical templates

Use **one** template everywhere: half-open `[lo, hi)`, find the **first True**. Every variant reduces to it.

```python
from typing import Callable
import bisect

# 4.1 THE template: first index in [lo, hi) where pred is True; returns hi if none.
def first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    # Invariant: every x < lo has pred(x) False; every x >= hi has pred(x) True (or is out of range)
    while lo < hi:
        mid = (lo + hi) // 2          # lo <= mid < hi, so the range always shrinks
        if pred(mid):
            hi = mid                  # mid could be the answer; keep it
        else:
            lo = mid + 1              # mid is False; discard it
    return lo

# 4.2 lower_bound: first i with a[i] >= x   (== bisect.bisect_left(a, x))
def lower_bound(a: list[int], x: int) -> int:
    return first_true(0, len(a), lambda i: a[i] >= x)

# 4.3 upper_bound: first i with a[i] > x    (== bisect.bisect_right(a, x))
def upper_bound(a: list[int], x: int) -> int:
    return first_true(0, len(a), lambda i: a[i] > x)

# Derived:
#   exists x?                 i = lower_bound(a, x); i < len(a) and a[i] == x
#   first occurrence of x     lower_bound (then verify)
#   last occurrence of x      upper_bound - 1 (then verify)
#   count of x                upper_bound - lower_bound
#   largest element < x       lower_bound - 1
#   largest element <= x      upper_bound - 1

# 4.4 Binary search on answer: MINIMUM feasible value in [lo, hi]
def min_feasible(lo: int, hi: int, feasible: Callable[[int], bool]) -> int:
    return first_true(lo, hi + 1, feasible)        # feasible must be F..F T..T

# 4.5 MAXIMUM feasible value (feasible is T..T F..F): last True = first False - 1
def max_feasible(lo: int, hi: int, feasible: Callable[[int], bool]) -> int:
    return first_true(lo, hi + 1, lambda x: not feasible(x)) - 1

# 4.6 Real-valued answers: fixed iteration count (no epsilon bugs)
def min_real(lo: float, hi: float, feasible: Callable[[float], bool]) -> float:
    for _ in range(100):                           # each step halves: 2^-100 precision
        mid = (lo + hi) / 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi
```
**Trace 4.2** a=[1,3,3,5], x=3: lo=0,hi=4 → mid=2 (3≥3 T) hi=2 → mid=1 (3≥3 T) hi=1 → mid=0 (1≥3 F) lo=1 → return 1 ✓. upper_bound → 3, so count = 3 − 1 = 2 ✓.

**How to set the answer range:** `lo` = the smallest value that could possibly be the answer, `hi` = one that is guaranteed feasible (e.g., `max(piles)`, `sum(nums)`). If unsure, widen it. Width costs only log.

## 5. Worked examples

### Example 1 — Search in Rotated Sorted Array (LC 33) — Medium
**Brute force:** linear scan O(n). The problem demands O(log n).
**Aha (signal: "rotated sorted" + O(log n)):** at any `mid`, **at least one half is sorted**. Compare `nums[lo]` with `nums[mid]` to see which. If the target lies inside the sorted half's range, go there. Otherwise go to the other half. The closed-interval form is clearer here because we test equality directly.

```python
def search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                    # left half [lo..mid] is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                        # right half [mid..hi] is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```
**Trace** [4,5,6,7,0,1,2], target 0: mid=3 (7), left sorted [4..7], 0 not in it → lo=4. mid=5 (1), `nums[4]=0 ≤ 1` → left sorted [0..1], 0 in [0,1) → hi=4. mid=4 → found ✓.
**Complexity:** O(log n) time, O(1) space.
**Edge cases:** `<=` in `nums[lo] <= nums[mid]` handles lo == mid (2-element ranges). Not rotated. n = 1.
**Sibling (LC 153, find min):** `while lo < hi: mid; if nums[mid] > nums[hi]: lo = mid+1 else: hi = mid`. Compare with `hi`, not `lo`: `nums[mid] > nums[hi]` proves the drop is to the right of mid.

### Example 2 — Koko Eating Bananas (LC 875) — Medium
**Problem:** minimum integer speed k so all piles are eaten within h hours (each hour, eat up to k from one pile).

**Brute force:** try k = 1, 2, … → O(max(p) · n) = 10⁹ · 10⁴. Dead.
**Aha (signals: "minimum speed such that…" + feasibility is monotone):** if speed k works, k+1 works. So binary search k over [1, max(piles)]. The check is O(n).

```python
def min_eating_speed(piles: list[int], h: int) -> int:
    def feasible(k: int) -> bool:
        return sum((p + k - 1) // k for p in piles) <= h      # ceil(p / k) hours per pile

    lo, hi = 1, max(piles)             # speed max(piles) finishes in n <= h hours
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```
**Trace** piles=[3,6,7,11], h=8: range [1,11]. mid=6 → 1+1+2+2=6 ≤ 8 T → hi=6. mid=3 → 1+2+3+4=10 F → lo=4. mid=5 → 1+2+2+3=8 T → hi=5. mid=4 → 1+2+2+3=8 T → hi=4. Answer 4 ✓.
**Complexity:** O(n log M) time, where M = max(piles). O(1) space.
**Edge cases:** h == n → answer is max(piles). Use integer ceil `(p + k − 1) // k`, not floats.

### Example 3 — Split Array Largest Sum (LC 410) — Hard
**Problem:** split `nums` into k non-empty contiguous parts, minimizing the largest part sum.

**Brute force:** try all split positions → C(n−1, k−1). **DP:** `dp[i][j]` = best for the first i elements in j parts → O(k·n²). That's workable for n = 1000, k = 50, but slow.
**Aha (signal: "minimize the maximum"):** guess the cap `C`. *Can we split into ≤ k parts with every part ≤ C?* Greedy: extend the current part until adding the next element would exceed C, then start a new part. If the greedy uses fewer than k parts, splitting any part further never increases a part sum (k ≤ n), so "≤ k parts" means feasible. Feasibility is monotone in C.

```python
def split_array(nums: list[int], k: int) -> int:
    def parts_needed(cap: int) -> int:
        parts, cur = 1, 0
        for x in nums:
            if cur + x > cap:          # x starts a new part
                parts += 1
                cur = x
            else:
                cur += x
        return parts

    lo, hi = max(nums), sum(nums)      # cap < max(nums) is impossible; sum is always feasible
    while lo < hi:
        mid = (lo + hi) // 2
        if parts_needed(mid) <= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
```
**Why greedy is right for the check:** packing each part as full as possible minimizes the number of parts for a given cap. That's an exchange argument: any valid split can shift a boundary right without breaking the cap on the left part, and the right part only shrinks.
**Trace** [7,2,5,10,8], k=2: range [10,32]. mid=21 → [7,2,5],[10,8] = 2 parts T → hi=21. mid=15 → [7,2,5],[10],[8] = 3 F → lo=16. mid=18 → [7,2,5],[10,8] = 2 T → hi=18. mid=17 → 14, then 10 → new part, then 10+8=18 > 17 → new part → 3 F → lo=18. Answer **18** ✓. (Trace each check element by element; eyeballing "2 parts" at mid=17 is exactly the mistake that gives 17.)
**Complexity:** O(n log S) time, where S = sum(nums). O(1) space.
**Edge cases:** k = 1 → sum. k = n → max. The lower bound must be max(nums), or the greedy check misbehaves on a single oversized element.

## 6. Common mistakes & off-by-one traps
- Mixing templates (closed `lo <= hi` with `hi = mid`) → infinite loop. With `lo < hi` + `hi = mid` + `lo = mid + 1`, the range always shrinks because `mid < hi`.
- Using `lo = mid` (not `mid + 1`) with floor division → infinite loop when hi = lo + 1. If you truly need `lo = mid`, use `mid = (lo + hi + 1) // 2`. Better: rewrite it as a first-True search.
- Wrong answer range: `lo = 0` for a speed (division by zero); `hi` not guaranteed feasible.
- Predicate direction reversed (T..T F..F fed to a first-True template).
- Floating-point: comparing with `==` or looping `while hi − lo > eps` with a bad eps. Use a fixed iteration count.
- Rotated array: comparing `nums[mid]` with `nums[lo]` for find-min (fails on non-rotated input). Compare with `nums[hi]`.
- Returning `lo` without verifying the element exists (`lo == len(a)` or `a[lo] != x`).
- In Python, `(lo + hi) // 2` has no overflow. In Java/C++ write `lo + (hi − lo) / 2`, and say so.

## 7. Connections
- **Greedy** (Ch 12): the most common pairing is binary search on the answer + a greedy feasibility check (410, 1011, 875, 1482, 2226).
- **BFS/DFS/Union-Find** as the check: "minimum effort path" (binary search the threshold + BFS), "swim in rising water".
- **Prefix sums**: binary search over prefix sums of positive arrays (weighted random pick, LC 528).
- **LIS in O(n log n)** (Ch 17): binary search on the patience-sorting tails array.
- **Two pointers**: often replaces "for each i, binary search j" with O(n) total.
- **k-th smallest by counting**: "how many elements ≤ x?" + binary search on x (LC 378, 668, 719).
- Mistaken for: DP (410 has a DP solution, and binary search is faster); heap (k-th smallest in a sorted matrix has both).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 704 | [Binary Search](https://leetcode.com/problems/binary-search/) | Easy | Sorted + find |
| 35 | [Search Insert Position](https://leetcode.com/problems/search-insert-position/) | Easy | "Where would it go" = lower bound |
| 69 | [Sqrt(x)](https://leetcode.com/problems/sqrtx/) | Easy | Largest k with k² ≤ x → max feasible |
| 74 | [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) | Medium | Row-major sorted = one virtual sorted array |
| 34 | [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | Medium | lower_bound and upper_bound − 1 |
| 153 | [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) | Medium | Rotated → compare mid with hi |
| 33 | [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) | Medium | One half is always sorted |
| 162 | [Find Peak Element](https://leetcode.com/problems/find-peak-element/) | Medium | Slope direction is a monotone-enough predicate |
| 875 | [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) | Medium | "Minimum speed such that" |
| 1011 | [Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) | Medium | Minimize the max load → greedy check |
| 981 | [Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/) | Medium | Timestamps increase → bisect per key |
| 1482 | [Minimum Number of Days to Make m Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/) | Medium | Minimum day such that → count adjacent bloomed |
| 378 | [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) | Medium | k-th → binary search the value, count ≤ x |
| 410 | [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) | Hard | Minimize the largest part |
| 4 | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | Hard | Binary search the partition in the shorter array |
| 719 | [Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/) | Hard | Binary search distance, count pairs with two pointers |

## 9. Recognition drill
1. Place c cows in n stalls (positions given) to maximize the minimum distance between any two cows.
2. A sorted array where every element appears twice except one. Find it in O(log n).
3. Given sorted `arr`, x, k: return the k closest elements to x.
4. Given workers' times per task and a total of T tasks, find the minimum time for all tasks when workers run in parallel.
5. An array strictly increases, then strictly decreases. Find the maximum.

<details><summary>Answers</summary>

1. **Binary search on the answer (maximize the minimum)**: feasible(d) = greedy placement left to right; monotone (smaller d is easier).
2. **Binary search on index parity**: before the single element, pairs start at even indices; after it, at odd. The predicate "pair broken at mid" is monotone.
3. **Binary search the window's left edge** in [0, n−k]: compare `x − arr[mid]` vs `arr[mid+k] − x`. (Or lower_bound + two pointers expanding.)
4. **Binary search on time t**: tasks done by time t = Σ ⌊t / time_i⌋ ≥ T is monotone.
5. **Binary search on slope** (`a[mid] < a[mid+1]` → go right). Peak-element style.
</details>

Next: Part 7 — Stack & Monotonic Stack/Queue
