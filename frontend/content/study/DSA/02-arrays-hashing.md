# Part 2 — Arrays & Hashing

## 1. Core insight
A hash map/set trades O(n) memory for **O(1) expected membership and lookup**. That turns any "inner loop that searches for something" into a constant-time check. There are three recurring uses. **Complement lookup:** "have I seen the value that completes me?" **Frequency counting:** multiset equality, majority, duplicates. **Canonical keys:** map equivalent objects to one key so grouping becomes a dictionary insert. When values are bounded in `[1..n]`, the array itself can serve as the hash table (index-as-hash) for O(1) extra space.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Find two elements such that…" in an **unsorted** array, and indices are needed.
- "Duplicate", "seen before", "first unique", "count occurrences", "most frequent".
- "Anagram", "same characters", "isomorphic", "pattern matching between two sequences".
- "Group by …" → canonical key.
- "Consecutive values" (not positions) in O(n) → set + "start of run" check.
- "O(1) extra space" with values in `[1..n]` → index-as-hash / cyclic sort.
- Needing O(1) insert/delete/lookup with an extra property (random access, order) → hash map + another structure.

**Anti-signals**
- Array is **sorted** → two pointers or binary search gives O(1) space. Hashing works but is the weaker answer.
- Need **ordering** (smallest key, predecessor, range) → a hash map has no order. Use sorting, a heap, or a sorted structure.
- Need **all** pairs/triplets without duplicates → sort + two pointers deduplicates more cleanly than hashing.
- Values unbounded and O(1) space required → hashing is disallowed; look for math (XOR, sum) or Floyd's cycle detection.
- Keys are mutable (lists) → convert to a tuple or string first.

## 3. Mental model
A hash map is a **memory of the past** that you query in O(1). Sweep left to right. At each element ask: *"Given what I've seen, can I answer now?"* and then *"What should I remember for the future?"*

```
nums = [2, 7, 11, 15], target = 9
i=0: need 7, seen={}        -> remember 2:0
i=1: need 2, seen={2:0}     -> found! (0, 1)
```

## 4. Canonical templates

```python
from collections import Counter, defaultdict

# 4.1 Complement lookup (one pass). Invariant: seen holds every index < i.
def two_sum(nums: list[int], target: int) -> list[int]:
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        if target - x in seen:          # check BEFORE inserting (avoids using x twice)
            return [seen[target - x], i]
        seen[x] = i
    return []

# 4.2 Frequency counting / multiset equality
def is_anagram(s: str, t: str) -> bool:
    return len(s) == len(t) and Counter(s) == Counter(t)

# 4.3 Grouping by canonical key
def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for s in strs:
        key = [0] * 26
        for c in s:
            key[ord(c) - ord('a')] += 1
        groups[tuple(key)].append(s)    # O(L) key instead of O(L log L) sorting
    return list(groups.values())

# 4.4 Bucket sort by frequency (top-k in O(n))
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]   # freq ≤ n
    for x, c in count.items():
        buckets[c].append(x)
    res: list[int] = []
    for c in range(len(buckets) - 1, 0, -1):
        for x in buckets[c]:
            res.append(x)
            if len(res) == k:
                return res
    return res

# 4.5 Index-as-hash (cyclic sort) for values in [1..n]. See Example 3.

# 4.6 Length-prefix encoding (serialize a list of arbitrary strings)
def encode(strs: list[str]) -> str:
    return ''.join(f'{len(s)}#{s}' for s in strs)

def decode(s: str) -> list[str]:
    res, i = [], 0
    while i < len(s):
        j = s.index('#', i)             # the length field ends at the first '#'
        length = int(s[i:j])
        res.append(s[j + 1: j + 1 + length])
        i = j + 1 + length
    return res
```
**Trace 4.4** (nums=[1,1,1,2,2,3], k=2): count={1:3, 2:2, 3:1}. buckets[3]=[1], [2]=[2], [1]=[3]. Scan from 6 down: c=3 → res=[1], c=2 → res=[1,2] → return. ✓

## 5. Worked examples

### Example 1 — Two Sum (LC 1) — Easy
**Brute force:** check all pairs → O(n²).
**Bottleneck:** for each `x`, the inner loop *searches* for `target − x`.
**Aha (signal: unsorted, indices needed, "search" in the inner loop):** replace the search with a hash lookup. Code is template 4.1.
**Why not sort + two pointers?** Sorting destroys the original indices. You'd need to sort (value, index) pairs: O(n log n) and more code.
**Complexity:** O(n) time (one pass, O(1) expected lookup), O(n) space.
**Edge cases:** `[3,3]` with target 6. Checking before inserting handles it. Negative numbers are fine.

### Example 2 — Longest Consecutive Sequence (LC 128) — Medium
**Problem:** length of the longest run of consecutive integer *values*, in O(n).

**Brute force:** sort and scan → O(n log n). That violates the O(n) requirement.
**Naive hash idea:** for each x, walk x+1, x+2, … in a set. Each walk can be O(n) → O(n²) on `[1..n]`.
**Bottleneck:** the run 1,2,3,4 is walked from 1, from 2, from 3, and from 4. Redundant.
**Aha:** only start walking from the **start** of a run, meaning `x − 1 ∉ set`. Then each element is visited by exactly one walk.

```python
def longest_consecutive(nums: list[int]) -> int:
    s = set(nums)
    best = 0
    for x in s:                 # iterate the SET, not nums (duplicates would re-walk)
        if x - 1 not in s:      # x starts a run
            y = x
            while y + 1 in s:
                y += 1
            best = max(best, y - x + 1)
    return best
```
**Trace** `[100,4,200,1,3,2]`: starts are 100, 200, 1. Walk from 1 → 4, length 4. ✓
**Complexity:** O(n) time (each value is visited once as a start check and at most once inside a walk), O(n) space.
**Edge cases:** empty → 0. Duplicates `[1,2,2,3]` → 3. Negative values work.

### Example 3 — First Missing Positive (LC 41) — Hard
**Problem:** smallest missing positive integer, O(n) time and **O(1) extra space**.

**Brute force 1:** for k = 1, 2, …, check `k in nums` → O(n²).
**Brute force 2:** put everything in a set, then check k = 1… → O(n) time but O(n) space. That violates the constraint.
**Key observation:** the answer is in `[1, n+1]`. With n slots, at most the values 1..n can be present. So only values in `[1..n]` matter, and there are exactly n indices to store them.
**Aha (signal: "O(1) space" + values bounded by n):** use the array as the hash set. Put value v at index v−1 (cyclic sort). Then the first index i with `nums[i] != i+1` gives the answer.

```python
def first_missing_positive(nums: list[int]) -> int:
    n = len(nums)
    for i in range(n):
        # Invariant: keep swapping nums[i] into its home until it's out of range,
        # already home, or its home already holds the same value (duplicate).
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1
```
**Trace** `[3,4,-1,1]`: i=0: 3→idx2 ⇒ [-1,4,3,1]. -1 out of range. i=1: 4→idx3 ⇒ [-1,1,3,4]. 1→idx0 ⇒ [1,-1,3,4]. i=2,3 already home. Scan: idx1 holds -1 → answer 2. ✓
**Complexity:** O(n) time, because each swap puts one value in its final home, so there are ≤ n swaps total (amortized). O(1) extra space.
**Edge cases:** `[1]` → 2. `[2]` → 1. Duplicates `[1,1]`: the `nums[j] != nums[i]` guard prevents an infinite loop.
**Python trap:** `nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]` is **wrong**. The left side assigns `nums[i]` first, which changes the index expression evaluated next. Always compute `j` first.

## 6. Common mistakes & traps
- Two Sum: inserting before checking lets an element pair with itself.
- Using a `list` as a set (`if x in lst`) → silent O(n²).
- Using a list as a dict key → `TypeError`. Use `tuple`.
- Iterating `nums` instead of `set(nums)` in Longest Consecutive → O(n²) on many duplicates.
- `Counter` subtraction drops zero and negative counts. `c1 == c2` is safe for equality checks.
- `defaultdict` creates keys on *read* (`d[k]` inside an `if`), which inflates `len(d)`. Use `k in d` or `d.get(k)`.
- Cyclic-sort swap order in Python (see Example 3).
- Sorting strings as the anagram key costs O(L log L) per string. A count tuple is O(L + 26).

## 7. Connections
- **Prefix sum + hash map** (Ch 5): the most important combination. "Count subarrays with sum k" is Two Sum on prefix sums.
- **Sliding window + frequency map** (Ch 4): anagram/permutation substrings.
- **Hash map + doubly linked list** → LRU cache (Ch 21).
- **Hash map + array** → Insert Delete GetRandom O(1): swap-with-last delete.
- **Index-as-hash** relates to **Floyd's cycle detection** (Find the Duplicate Number, Ch 8): values in [1..n] define a functional graph.
- Mistaken for: sorting problems (Longest Consecutive looks like "sort then scan"); two pointers (Two Sum on an *unsorted* array).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 217 | [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) | Easy | "Seen before?" |
| 242 | [Valid Anagram](https://leetcode.com/problems/valid-anagram/) | Easy | Multiset equality |
| 1 | [Two Sum](https://leetcode.com/problems/two-sum/) | Easy | Unsorted + indices → complement lookup |
| 205 | [Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/) | Easy | A bijection needs maps in *both* directions |
| 169 | [Majority Element](https://leetcode.com/problems/majority-element/) | Easy | Counting; follow-up: O(1) space via a voting invariant |
| 49 | [Group Anagrams](https://leetcode.com/problems/group-anagrams/) | Medium | Group equivalent items → canonical key |
| 347 | [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) | Medium | Frequency ≤ n → bucket by frequency |
| 238 | [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) | Medium | No division → precompute from both sides |
| 36 | [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) | Medium | Seen-sets per row/col/box; box key = (r//3, c//3) |
| 271 | [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) (Premium) | Medium | Arbitrary characters → length prefix |
| 128 | [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) | Medium | O(n) without sorting → only start at run beginnings |
| 454 | [4Sum II](https://leetcode.com/problems/4sum-ii/) | Medium | Four arrays → split into two halves, hash pair sums |
| 380 | [Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) | Medium | O(1) random pick needs an array; O(1) delete needs index lookup |
| 41 | [First Missing Positive](https://leetcode.com/problems/first-missing-positive/) | Hard | O(1) space + answer in [1..n+1] → array as hash |

## 9. Recognition drill
1. Given an array of strings, return groups of strings that are rotations of each other ("abc", "bca", "cab").
2. Given an unsorted array and k, count pairs (i<j) with `nums[i] + nums[j] == k`.
3. An array of n+1 integers in [1..n] contains exactly one repeated value. Find it without modifying the array, in O(1) space.
4. Design a structure supporting insert(x), remove(x), and getRandom() in O(1).
5. Determine whether string `s` follows `pattern` (e.g., "abba" ↔ "dog cat cat dog").

<details><summary>Answers</summary>

1. **Canonical key**: the lexicographically smallest rotation (or a sorted gap signature). Equivalent strings share the key, so group in a hash map.
2. **Hash map of counts, one pass**: at each x, add `count[k − x]`, then increment `count[x]`. This is Two Sum's counting variant.
3. **Floyd's cycle detection on index→value** (see Ch 8). O(1) space and read-only rule out hashing and cyclic sort.
4. **Hash map value→index + array**. Delete by swapping with the last element and popping.
5. **Two hash maps (bijection)**: pattern char → word and word → pattern char must both be consistent.
</details>

Next: Part 3 — Two Pointers
