# Part 3 — Two Pointers

## 1. Core insight
Two pointers exploit **monotonicity** to discard candidates without examining them. In a sorted array, if `a[l] + a[r]` is too small, then `a[l] + a[j]` is too small for *every* `j ≤ r`, so `l` can be eliminated in O(1). One comparison removes a whole row of the O(n²) pair matrix, and O(n²) pairs collapse to O(n) steps. There are three families. **Opposite ends** (pair search, palindromes, container). **Same direction / read-write** (in-place compaction, partitioning). **Two sequences** (merge, subsequence check). Fast/slow pointers (linked lists) are covered in Ch 8.

## 2. Trigger signals / Anti-signals
**Triggers**
- **Sorted** array (or sorting is allowed) + pair/triplet with target sum/difference.
- "In-place", "O(1) extra space", "remove/move elements", "partition by a predicate".
- Palindrome check, "reverse", symmetric comparison.
- Two sorted sequences: merge, intersect, "is subsequence".
- "Maximize area/width between two lines", where width shrinks as pointers move inward.
- After fixing one element, the rest is a 2-pointer problem (3Sum, 4Sum).

**Anti-signals**
- Unsorted input and **original indices** are required → sorting loses them; use hashing.
- Subarray *sum* with negative numbers → moving a pointer doesn't change the sum monotonically; use prefix sums.
- The relationship between pointer movement and the objective isn't monotone. If you can't state *why* moving `l` can never skip the answer, it's not two pointers.
- Need all pairs with a property that isn't order-compatible (e.g., XOR) → use a hash or trie.

## 3. Mental model
The **elimination matrix**. Rows are `l`, columns are `r`, and a cell is `a[l] + a[r]`. With a sorted array, values increase rightward along rows and downward along columns.

```
        r=0  1  2  3  4
 l=0  [ .   .  .  .  S ]   start at top-right corner (l=0, r=n-1)
 l=1  [     .  .  .  . ]   sum < target -> row l is all too small  -> l++ (drop row)
 l=2  [        .  .  . ]   sum > target -> column r is all too big -> r-- (drop col)
 l=3  [           .  . ]   each step drops a full row or column
                           -> at most n steps
```
Same-direction (read/write): `read` scans everything; `write` marks the end of the "kept" prefix.
```
[ kept ... | garbage ... | unread ... ]
           ^write        ^read
```

## 4. Canonical templates

```python
# 4.1 Opposite ends: pair with target sum in a sorted array
def pair_sum_sorted(a: list[int], target: int) -> tuple[int, int] | None:
    l, r = 0, len(a) - 1
    # Invariant: if a valid pair exists, it lies within [l, r]
    while l < r:
        s = a[l] + a[r]
        if s == target:
            return l, r
        if s < target:
            l += 1      # a[l] + (anything ≤ a[r]) is too small -> l is useless
        else:
            r -= 1      # a[r] + (anything ≥ a[l]) is too big -> r is useless
    return None

# 4.2 Same direction (read/write): remove duplicates from a sorted array in place
def remove_duplicates(nums: list[int]) -> int:
    write = 0
    # Invariant: nums[:write] holds the unique values seen so far, in order
    for read in range(len(nums)):
        if write == 0 or nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1
    return write

# 4.3 Merge from the back (in place, no overwriting of unread data)
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:                       # once nums2 is exhausted, nums1's rest is in place
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]; i -= 1
        else:
            nums1[k] = nums2[j]; j -= 1
        k -= 1

# 4.4 Three-way partition (Dutch national flag): 0s | 1s | unknown | 2s
def sort_colors(nums: list[int]) -> None:
    lo, mid, hi = 0, 0, len(nums) - 1
    # Invariant: [0,lo) = 0, [lo,mid) = 1, [mid,hi] unknown, (hi,end] = 2
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]; lo += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]; hi -= 1   # don't advance mid: the swapped-in value is unknown

# 4.5 Two sequences: is s a subsequence of t?
def is_subsequence(s: str, t: str) -> bool:
    i = 0
    for c in t:
        if i < len(s) and s[i] == c:
            i += 1
    return i == len(s)
```
**Trace 4.4** `[2,0,1]`: mid=0 sees 2 → swap with hi → [1,0,2], hi=1. mid=0 sees 1 → mid=1. mid=1 sees 0 → swap with lo=0 → [0,1,2], lo=1, mid=2 > hi → stop. ✓

## 5. Worked examples

### Example 1 — Two Sum II, sorted input (LC 167) — Medium
**Brute force:** all pairs → O(n²). **Hash map:** O(n) time and O(n) space, but the problem requires O(1) space.
**Aha (signal: sorted + pair + O(1) space):** the elimination matrix. Template 4.1 with 1-indexed output:
```python
def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    l, r = 0, len(numbers) - 1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target:
            return [l + 1, r + 1]
        if s < target:
            l += 1
        else:
            r -= 1
    return []
```
**Complexity:** O(n) time (each step moves a pointer inward), O(1) space.
**Edge cases:** negative numbers are fine. A duplicated value pair (e.g., [3,3]) is found at l≠r.

### Example 2 — 3Sum (LC 15) — Medium
**Problem:** all *unique* triplets summing to 0.

**Brute force:** triple loop + set of sorted tuples → O(n³).
**Better:** fix i, then Two Sum with a hash set → O(n²) time, but deduplication is messy.
**Aha (signals: "unique triplets" + values only → sort; after fixing one element it's sorted Two Sum):** sort, fix `i`, run opposite pointers on `i+1..n−1`. Skip duplicates at both levels.

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    n, res = len(nums), []
    for i in range(n - 2):
        if nums[i] > 0:                         # smallest is positive -> no zero sum possible
            break
        if i > 0 and nums[i] == nums[i - 1]:    # same anchor -> same triplets
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:   # skip duplicate second elements
                    lo += 1
    return res
```
**Trace** `[-1,0,1,2,-1,-4]` → sorted `[-4,-1,-1,0,1,2]`. i=0 (−4): no pair sums to 4. i=1 (−1): lo=2,hi=5: −1−1+2=0 ✓ → [−1,−1,2]. Then lo=3,hi=4: −1+0+1=0 ✓ → [−1,0,1]. i=2 (−1) is a duplicate, skip. i=3 (0): lo=4,hi=5 → 3>0. Result: 2 triplets. ✓
**Complexity:** O(n²) time (n anchors × O(n) scan; the sort is dominated). O(1) extra space, or O(n) if you count the sort.
**Edge cases:** `[0,0,0,0]` → exactly one [0,0,0]. Fewer than 3 elements → [].

### Example 3 — Trapping Rain Water (LC 42) — Hard
**Problem:** total trapped water given bar heights.

**Brute force:** for each i, water = `min(max_left(i), max_right(i)) − h[i]`. Scanning both sides per i → O(n²).
**Better (lever: precompute):** prefix-max and suffix-max arrays → O(n) time, O(n) space.
**Aha (two pointers, O(1) space):** we don't need the *exact* max on both sides, only the *smaller* one. If `h[l] < h[r]`, the right side has a wall at least `h[r]`, and that wall exceeds the running `left_max` (proof below). So the water at `l` is determined by `left_max` alone.

```python
def trap(height: list[int]) -> int:
    l, r = 0, len(height) - 1
    left_max = right_max = water = 0
    while l < r:
        if height[l] < height[r]:
            left_max = max(left_max, height[l])
            water += left_max - height[l]       # left_max is the binding wall
            l += 1
        else:
            right_max = max(right_max, height[r])
            water += right_max - height[r]
            r -= 1
    return water
```
**Why it's correct:** suppose we are moving `l`, and `left_max` was set by position p < l. When the loop was at p, it moved left, so `h[p] < h[r']` for the r' at that time. Since r only decreases, r' ≥ current r, so `max(h[r:]) ≥ h[r'] > left_max`. The true right wall is therefore taller than `left_max`, and `min(left_max, true_right_max) = left_max`. The case for moving `r` is symmetric.
**Trace** `[0,1,0,2,1,0,1,3,2,1,2,1]` → 6. ✓ (Check it by hand once: the pairs are resolved from the lower side each time.)
**Complexity:** O(n) time, O(1) space.
**Edge cases:** n < 3 → 0. Monotonic heights → 0. Plateaus: using `<` vs `<=` both work.

## 6. Common mistakes & off-by-one traps
- `while l <= r` for pair search → pairs an element with itself. Use `l < r`.
- 3Sum: dedupe the anchor with `nums[i] == nums[i-1]` (not `nums[i+1]`, which would skip valid first occurrences).
- 3Sum: forgetting to move *both* pointers after a match → infinite loop.
- Dutch flag: advancing `mid` after swapping with `hi`. The value swapped in hasn't been examined yet.
- Merge in place from the front overwrites unread data. Merge from the back.
- Container With Most Water: moving the *taller* side can never help (width shrinks, height is capped by the shorter side). Always move the shorter one.
- Sorting when the problem needs original indices.

## 7. Connections
- **Binary search**: each pair search can also be done by fixing l and binary searching r (O(n log n)). Two pointers is the O(n) upgrade.
- **Sliding window** (Ch 4) is same-direction two pointers with a maintained window aggregate.
- **Fast/slow pointers** (Ch 8) for cycles and middles.
- **Greedy**: Boats to Save People is two pointers justified by an exchange argument (pair the heaviest with the lightest).
- **Quicksort partition / quickselect** (Ch 11) = read/write or Hoare two pointers.
- Mistaken for: hashing (Two Sum on unsorted input), DP (Trapping Rain Water looks like "compute something per index").

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 125 | [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) | Easy | Symmetric comparison from both ends |
| 344 | [Reverse String](https://leetcode.com/problems/reverse-string/) | Easy | In-place, O(1) space |
| 283 | [Move Zeroes](https://leetcode.com/problems/move-zeroes/) | Easy | In-place compaction → read/write |
| 26 | [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) | Easy | Sorted + in-place → write pointer |
| 88 | [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) | Easy | Free space at the end → fill from the back |
| 977 | [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/) | Easy | The largest squares are at the extremes |
| 680 | [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/) | Easy | One deletion allowed → branch once at the first mismatch |
| 167 | [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | Medium | Sorted + pair + O(1) space |
| 15 | [3Sum](https://leetcode.com/problems/3sum/) | Medium | Unique triplets → sort, fix one, pair search |
| 11 | [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) | Medium | Width shrinks, so only moving the shorter side can help |
| 75 | [Sort Colors](https://leetcode.com/problems/sort-colors/) | Medium | Three values, one pass → three regions |
| 16 | [3Sum Closest](https://leetcode.com/problems/3sum-closest/) | Medium | Same structure, track the best distance |
| 881 | [Boats to Save People](https://leetcode.com/problems/boats-to-save-people/) | Medium | Pair the heaviest with the lightest if possible |
| 18 | [4Sum](https://leetcode.com/problems/4sum/) | Medium | Fix two, then pair search; generalize to k-sum |
| 42 | [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) | Hard | Water depends on min(left max, right max) → resolve the smaller side |

## 9. Recognition drill
1. Given a sorted array and k, count pairs with `a[i] + a[j] < k`.
2. Rearrange an array so all negatives come before all positives, in place, order not required.
3. Given two sorted arrays, find their intersection (with multiplicity).
4. Given an array of heights, choose two lines maximizing `min(h[i], h[j]) × (j − i)`.
5. Given an unsorted array of integers, find the number of unique pairs with difference exactly k.

<details><summary>Answers</summary>

1. **Opposite pointers**: if `a[l] + a[r] < k`, then all pairs (l, l+1..r) qualify → add `r − l`, then l++. Otherwise r--.
2. **Read/write partition** (Lomuto-style). It's one predicate, so a single write boundary suffices.
3. **Two sequences, two pointers**: advance the smaller one; on equal, record and advance both.
4. **Opposite pointers, move the shorter side** (Container With Most Water).
5. **Sort + same-direction two pointers** (or a hash set of values: check x+k). Sorting enables easy dedupe; hashing is O(n).
</details>

Next: Part 4 — Sliding Window
