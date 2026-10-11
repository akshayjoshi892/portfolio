# Part 4 — Sliding Window (Fixed & Variable)

## 1. Core insight
For **contiguous** subarrays/substrings, a brute force enumerates O(n²) windows and recomputes each aggregate from scratch. Sliding window fixes both kinds of waste. The aggregate is updated **incrementally** (add the entering element, remove the leaving one). And, *if validity is monotone*, both endpoints only move forward. Monotone means: shrinking a valid window keeps it valid, or growing an invalid window keeps it invalid. Each index enters and leaves once, so the total is O(n). Monotonicity is the whole game: without it, advancing `left` might skip the answer.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Contiguous subarray / substring" + "longest / shortest / maximum / minimum / count".
- "At most k distinct", "at most k replacements/flips/zeros", "without repeating", "sum ≥ target" (with non-negative values).
- "Of size k" / "every window of length k" → fixed window.
- "Permutation / anagram of p inside s" → fixed window of length |p| with frequency counts.
- "Number of subarrays with exactly K …" → atMost(K) − atMost(K−1).

**Anti-signals**
- **Negative numbers** with a sum constraint → adding an element can decrease the sum, so validity isn't monotone. Use prefix sums + hash map (sum = k) or prefix sums + monotonic deque (LC 862).
- **Subsequence** (not contiguous) → DP or greedy.
- Constraint is "exactly k" and not decomposable → try atMost differences, or prefix sums.
- The window aggregate can't be un-done cheaply (e.g., max of window) → need a monotonic deque (Ch 7) or a heap with lazy deletion.

## 3. Mental model
A caterpillar crawling right. The head (`right`) always advances. The tail (`left`) advances only as much as needed to restore validity.

```
s = a b c a b c b b
    L   R            window "abc" valid
    L     R          add 'a' -> invalid (dup 'a')
      L   R          shrink until valid: "bca"
```
The question "which index do I remove?" always has the answer "the oldest one", which is why a window (a FIFO) works.

## 4. Canonical templates

```python
from collections import Counter, defaultdict

# 4.1 Fixed-size window
def max_sum_window(nums: list[int], k: int) -> int:
    window = sum(nums[:k])
    best = window
    for r in range(k, len(nums)):
        window += nums[r] - nums[r - k]     # add entering, remove leaving
        best = max(best, window)
    return best

# 4.2 Variable window: LONGEST valid window
def longest_valid(s: str, k: int) -> int:     # example: at most k distinct chars
    count: dict[str, int] = defaultdict(int)
    left = best = 0
    for right, c in enumerate(s):
        count[c] += 1                               # 1) expand
        while len(count) > k:                       # 2) shrink until valid
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        best = max(best, right - left + 1)          # 3) window [left, right] is valid
    return best

# 4.3 Variable window: SHORTEST valid window (non-negative nums, sum >= target)
def min_subarray_len(target: int, nums: list[int]) -> int:
    left = total = 0
    best = float('inf')
    for right, x in enumerate(nums):
        total += x
        while total >= target:                      # while valid, record and shrink
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float('inf') else best

# 4.4 Counting subarrays: every valid window ending at `right` counts
def count_at_most_k_distinct(nums: list[int], k: int) -> int:
    count: dict[int, int] = defaultdict(int)
    left = res = 0
    for right, x in enumerate(nums):
        count[x] += 1
        while len(count) > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                del count[nums[left]]
            left += 1
        res += right - left + 1          # subarrays [left..right], [left+1..right], ..., [right..right]
    return res

def count_exactly_k_distinct(nums: list[int], k: int) -> int:
    return count_at_most_k_distinct(nums, k) - count_at_most_k_distinct(nums, k - 1)
```
**Longest vs shortest:** for *longest*, shrink while **invalid** and record after the loop. For *shortest*, shrink while **valid** and record inside the loop.

**Trace 4.3** target=7, nums=[2,3,1,2,4,3]: r=3 total=8≥7 → best=4, drop 2 → 6. r=4 total=10 → best=4, drop 3 → 7 → best=3, drop 1 → 6. r=5 total=9 → best=3, drop 2 → 7 → best=2, drop 4 → 3. Answer 2. ✓

## 5. Worked examples

### Example 1 — Longest Substring Without Repeating Characters (LC 3) — Medium
**Brute force:** all O(n²) substrings, check uniqueness in O(n) → O(n³). With incremental sets: O(n²).
**Bottleneck:** when `s[i..j]` has a duplicate, every longer substring starting at i also has it (monotone!). So there's no need to restart from i+1 and re-scan.
**Aha (signals: contiguous + longest + "without repeating" is monotone):** variable window. Optimization: store each character's last index and jump `left` directly past the duplicate.

```python
def length_of_longest_substring(s: str) -> int:
    last: dict[str, int] = {}
    left = best = 0
    for right, c in enumerate(s):
        if c in last and last[c] >= left:   # duplicate inside the current window
            left = last[c] + 1
        last[c] = right
        best = max(best, right - left + 1)
    return best
```
**Trace** "abba": r=0 a → best 1. r=1 b → best 2. r=2 b (last=1 ≥ 0) → left=2. r=3 a (last=0 < left=2, stale, ignore) → window "ba" → best 2. ✓ The `>= left` check exists exactly for this stale case.
**Complexity:** O(n) time. O(min(n, Σ)) space, where Σ is the alphabet size.
**Edge cases:** "" → 0. All the same → 1. The stale-index case above.

### Example 2 — Longest Repeating Character Replacement (LC 424) — Medium
**Problem:** replace at most k characters to get the longest block of one letter.

**Brute force:** for each window, cost = `len − max_count`. O(n²) windows × O(26) → too slow for n = 10⁵.
**Aha (signal: "at most k changes" in a contiguous block):** a window is valid iff `len − max_freq ≤ k`, and validity is monotone in shrinking. So: variable window, longest.

```python
def character_replacement(s: str, k: int) -> int:
    count = [0] * 26
    left = max_freq = best = 0
    for right, c in enumerate(s):
        count[ord(c) - ord('A')] += 1
        max_freq = max(max_freq, count[ord(c) - ord('A')])
        while (right - left + 1) - max_freq > k:
            count[ord(s[left]) - ord('A')] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```
**Subtle point:** `max_freq` is never decreased when shrinking, so it may be stale (too high). This is safe. The answer only improves when a window of length `max_freq + k` or more becomes valid, which requires a *true* max_freq larger than any before. A stale value can only keep the window from shrinking. It never produces a larger `best` than some real valid window.
**Complexity:** O(n) time, O(1) space (26 counters).
**Edge cases:** k ≥ n → n. Single character.

### Example 3 — Minimum Window Substring (LC 76) — Hard
**Problem:** the shortest substring of `s` containing all characters of `t` (with multiplicity).

**Brute force:** all substrings × a count check → O(n² · Σ).
**Aha (signals: contiguous + shortest + "contains all" is monotone, since growing keeps it valid):** shortest-window template. Keep `need[c]` = how many more c's we need (it can go negative for surplus), plus a scalar `missing` so the validity check is O(1).

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not s or not t:
        return ""
    need = Counter(t)
    missing = len(t)                 # total chars still needed
    left = 0
    best_len, best_l = float('inf'), 0
    for right, c in enumerate(s):
        if need[c] > 0:              # c was actually needed
            missing -= 1
        need[c] -= 1
        while missing == 0:          # window valid -> record and shrink
            if right - left + 1 < best_len:
                best_len, best_l = right - left + 1, left
            need[s[left]] += 1
            if need[s[left]] > 0:    # we just gave up a needed char
                missing += 1
            left += 1
    return "" if best_len == float('inf') else s[best_l:best_l + best_len]
```
**Trace** s="ADOBECODEBANC", t="ABC": the first valid window is at r=5 "ADOBEC" (len 6). Shrinking drops A → invalid. Valid again at r=10 ("DOBECODEBA" → shrinks to "CODEBA", len 6). At r=12 it shrinks to "BANC" (len 4). ✓
**Complexity:** O(|s| + |t|) time (each index enters and leaves once), O(Σ) space.
**Edge cases:** t longer than s → "". Duplicate chars in t ("AA") are handled via counts. No valid window → "".

## 6. Common mistakes & off-by-one traps
- Window length is `right − left + 1`, not `right − left`.
- Recording the answer at the wrong place (longest: after shrinking; shortest: inside the shrink loop).
- Not deleting zero-count keys when you use `len(count)` as "distinct".
- Applying sliding window to sums with negatives. Test `[2, -1, 2]`, k=3 mentally: the shortest subarray with sum ≥ 3 is the whole array, but a window would shrink wrongly.
- "Exactly k" solved directly with a window. Usually wrong. Use atMost(k) − atMost(k−1).
- In the last-index jump variant (LC 3), forgetting `last[c] >= left` → `left` moves backward.
- Fixed window: off-by-one on the first window (`nums[:k]`) and the leaving index `r − k`.

## 7. Connections
- **Two pointers** (Ch 3): sliding window is same-direction two pointers plus an aggregate.
- **Hash map** (Ch 2) for counts; **fixed-size array[26]** when the alphabet is small.
- **Monotonic deque** (Ch 7): window max/min (LC 239), and "longest subarray with |max − min| ≤ limit" (LC 1438).
- **Prefix sums** (Ch 5): the replacement when negatives break monotonicity. Also "shortest subarray with sum ≥ k" with negatives (LC 862) = prefix sums + monotonic deque.
- **Binary search on answer**: some "longest window with property" problems can be binary searched on length with a fixed-window check, O(n log n). A variable window is the O(n) upgrade.
- **Rolling hash** (Ch 21) is a fixed window over hash values.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 643 | [Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/) | Easy | "Of length k" → fixed window |
| 121 | [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | Easy | Running minimum as the window's left edge |
| 3 | [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Medium | Contiguous + longest + monotone uniqueness |
| 567 | [Permutation in String](https://leetcode.com/problems/permutation-in-string/) | Medium | Permutation of p in s → fixed window of counts |
| 438 | [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/) | Medium | Same as above, report all positions |
| 424 | [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) | Medium | "At most k changes" → len − max_freq ≤ k |
| 209 | [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) | Medium | Positive values + shortest with sum ≥ target |
| 1004 | [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/) | Medium | "Flip at most k zeros" → at most k zeros in the window |
| 904 | [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/) | Medium | At most 2 distinct |
| 1423 | [Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/) | Medium | Take from both ends = leave a contiguous middle window |
| 1248 | [Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/) | Medium | "Exactly k" count → atMost difference |
| 76 | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | Hard | Shortest window containing a multiset |
| 239 | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | Hard | Window max can't be un-done → monotonic deque |
| 992 | [Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) | Hard | Exactly k distinct → atMost(k) − atMost(k−1) |
| 30 | [Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/) | Hard | Equal-length words → run `len(word)` windows with word-level steps |

## 9. Recognition drill
1. Given a binary array, find the longest subarray of 1s after deleting exactly one element.
2. Given an integer array (may contain negatives) and k, count subarrays with sum equal to k.
3. Given a string, count substrings that contain at least one of each of 'a', 'b', 'c'.
4. For each window of size k, report the number of distinct elements.
5. Given positive integers and k, count subarrays whose product is strictly less than k.

<details><summary>Answers</summary>

1. **Variable window with at most one zero** (answer = window length − 1). "Delete one" = "allow one zero".
2. **Not a sliding window** (negatives break monotonicity) → **prefix sum + hash map** (Ch 5).
3. **Variable window counting**: once [left..right] contains all three, every extension to the right also does → add `n − right`, or count "at most" complements.
4. **Fixed window + frequency map**: distinct = number of keys with count > 0; update on enter/leave.
5. **Variable window, counting variant**: positive numbers make the product monotone; after shrinking while product ≥ k, add `right − left + 1`.
</details>

Next: Part 5 — Prefix Sums & Difference Arrays
