# Part 21 — Advanced Topics (What Actually Appears at Big Tech)

Four families: **range-query trees** (Fenwick, segment tree), **string matching** (KMP, rolling hash), **line sweep**, and **design** (LRU, LFU, iterators). Design questions are the most frequent of these in real interviews. Segment trees appear mostly at Google and in hard follow-ups.

## 1. Core insight
- **Fenwick / segment trees** answer range aggregates **with updates** in O(log n). Prefix sums are O(1) to query but O(n) to update. These trees store precomputed aggregates over O(log n) *canonical blocks*, so a query combines a few blocks and an update touches a few blocks. Fenwick is shorter but limited to invertible operations on prefixes (sum, count). A segment tree handles any associative operation (min, max, gcd) and, with lazy propagation, range updates.
- **KMP** never re-reads text characters. On a mismatch it falls back using the **longest proper prefix of the pattern that is also a suffix** of what matched (the LPS/failure table), giving O(n + m). A **rolling hash** makes the hash of any fixed-length window O(1) to update, so "compare substrings" becomes "compare numbers". Combined with binary search on the length, that answers "longest repeated substring"-type questions.
- **Line sweep** turns 2D geometry into sorted 1D events plus a data structure for the "currently active" set (a heap, a sorted list, a segment tree).
- **Design problems** combine a hash map (O(1) locate) with one ordering structure (a doubly linked list for recency, frequency buckets, a heap for priority, a stack for iteration state).

## 2. Trigger signals / Anti-signals
**Triggers**
- "Update an element and query a range sum/min/max" repeatedly → Fenwick / segment tree.
- "Count smaller elements after self", "count inversions", "reverse pairs", "count range sums" → Fenwick over compressed values (or merge sort).
- "Range add + range query" → lazy segment tree (or two Fenwicks).
- "Find the pattern in text" with O(n+m) required, "shortest palindrome by adding a prefix", "repeated substring pattern" → KMP / prefix function.
- "Longest duplicate substring", "compare many substrings", "distinct substrings of length L" → rolling hash (+ binary search).
- "Skyline", "area of union of rectangles", "max overlapping at a point in 2D" → line sweep.
- "Design a cache with O(1) get/put", "LRU", "LFU", "iterator", "snapshot", "time-based store", "stock price with corrections" → hash map + an ordering structure.

**Anti-signals**
- No updates between queries → plain prefix sums / sparse table.
- Only point updates + prefix sums → Fenwick is enough; don't write a segment tree.
- A single substring search in an interview with no complexity demand → Python's `str.find` (say it's O(nm) worst case, then offer KMP).
- Small n (≤ 10³) → an O(n²) approach may be the expected answer. Don't over-engineer.

## 3. Mental model
**Fenwick tree:** index i is responsible for the range of length `i & −i` ending at i.
```
i (binary)   covers
1  (0001)    [1..1]
2  (0010)    [1..2]
3  (0011)    [3..3]
4  (0100)    [1..4]
6  (0110)    [5..6]
8  (1000)    [1..8]
prefix(7) = t[7] + t[6] + t[4]     (7 -> 6 -> 4 -> 0 by removing the lowest bit)
add(3)    updates t[3], t[4], t[8] (3 -> 4 -> 8 by adding the lowest bit)
```
**KMP failure table:** "if I've matched k characters and the next one fails, which shorter match is still alive?"
```
pattern: a b a b a c a
lps:     0 0 1 2 3 0 1
matched "ababa" then mismatch -> fall back to lps[4] = 3 ("aba") without moving in the text
```

## 4. Canonical templates

```python
import heapq
import random
from collections import OrderedDict, defaultdict

# ---------- 4.1 Fenwick tree (1-indexed): point add, prefix sum ----------
class BIT:
    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.t[i] += delta
            i += i & -i                  # move to the next block that covers i

    def prefix(self, i: int) -> int:     # sum of [1..i]
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i                  # jump to the block before this one
        return s

    def range_sum(self, l: int, r: int) -> int:
        return self.prefix(r) - self.prefix(l - 1)

# ---------- 4.2 Iterative segment tree: point update, range query [l, r) ----------
class SegTree:
    def __init__(self, arr: list[int], op=min, identity=float('inf')):
        self.n, self.op, self.id = len(arr), op, identity
        self.t = [identity] * self.n + list(arr)          # leaves live at [n, 2n)
        for i in range(self.n - 1, 0, -1):
            self.t[i] = op(self.t[2 * i], self.t[2 * i + 1])

    def update(self, i: int, val: int) -> None:
        i += self.n
        self.t[i] = val
        while i > 1:
            i //= 2
            self.t[i] = self.op(self.t[2 * i], self.t[2 * i + 1])

    def query(self, l: int, r: int):                      # op over arr[l:r]
        res_l = res_r = self.id
        l += self.n; r += self.n
        while l < r:
            if l & 1:
                res_l = self.op(res_l, self.t[l]); l += 1
            if r & 1:
                r -= 1; res_r = self.op(self.t[r], res_r)
            l //= 2; r //= 2
        return self.op(res_l, res_r)

# ---------- 4.3 Lazy segment tree: range add, range sum (inclusive, 0-indexed) ----------
class LazySegTree:
    def __init__(self, n: int):
        self.n = n
        self.sum = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def _apply(self, node: int, l: int, r: int, v: int) -> None:
        self.sum[node] += v * (r - l + 1)
        self.lazy[node] += v

    def _push(self, node: int, l: int, r: int) -> None:
        if self.lazy[node]:
            m = (l + r) // 2
            self._apply(2 * node, l, m, self.lazy[node])
            self._apply(2 * node + 1, m + 1, r, self.lazy[node])
            self.lazy[node] = 0

    def update(self, ql: int, qr: int, v: int, node: int = 1, l: int = 0, r: int | None = None) -> None:
        if r is None:
            r = self.n - 1
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self._apply(node, l, r, v)
            return
        self._push(node, l, r)
        m = (l + r) // 2
        self.update(ql, qr, v, 2 * node, l, m)
        self.update(ql, qr, v, 2 * node + 1, m + 1, r)
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

    def query(self, ql: int, qr: int, node: int = 1, l: int = 0, r: int | None = None) -> int:
        if r is None:
            r = self.n - 1
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.sum[node]
        self._push(node, l, r)
        m = (l + r) // 2
        return self.query(ql, qr, 2 * node, l, m) + self.query(ql, qr, 2 * node + 1, m + 1, r)

# ---------- 4.4 KMP ----------
def build_lps(p: str) -> list[int]:
    lps = [0] * len(p)
    k = 0                                  # length of the current matched prefix
    for i in range(1, len(p)):
        while k and p[i] != p[k]:
            k = lps[k - 1]                 # fall back to a shorter border
        if p[i] == p[k]:
            k += 1
        lps[i] = k
    return lps

def kmp_search(text: str, p: str) -> int:  # first index of p in text, or -1
    if not p:
        return 0
    lps, k = build_lps(p), 0
    for i, c in enumerate(text):
        while k and c != p[k]:
            k = lps[k - 1]
        if c == p[k]:
            k += 1
        if k == len(p):
            return i - len(p) + 1          # (for all matches: record, then k = lps[k-1])
    return -1

# ---------- 4.5 Rolling hash over windows of length L ----------
MOD61 = (1 << 61) - 1                      # a large Mersenne prime -> collisions ~ n^2 / 2^61
def window_hashes(a: list[int], L: int, base: int) -> list[int]:
    h = 0
    for i in range(L):
        h = (h * base + a[i]) % MOD61
    out = [h]
    top = pow(base, L, MOD61)              # weight of the character that leaves
    for i in range(L, len(a)):
        h = (h * base - a[i - L] * top + a[i]) % MOD61
        out.append(h)
    return out                             # out[i] = hash of a[i:i+L]
```
**Trace 4.4** `build_lps("ababaca")` → [0,0,1,2,3,0,1]. At i=5 ('c'), k falls 3 → lps[2]=1 → lps[0]=0 with no match → 0 ✓.
**Trace 4.1** add(3, 5) on n=8 touches t[3], t[4], t[8]. prefix(7) reads t[7], t[6], t[4], and the 5 is counted via t[4] ✓.

### Design templates

```python
# ---------- 4.6 LRU cache: hash map + doubly linked list with sentinels ----------
class _Node:
    __slots__ = ('key', 'val', 'prev', 'next')
    def __init__(self, key: int = 0, val: int = 0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map: dict[int, _Node] = {}
        self.head, self.tail = _Node(), _Node()      # head.next = most recent, tail.prev = least recent
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node: _Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _add_front(self, node: _Node) -> None:
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            self._remove(self.map[key])
        node = _Node(key, value)
        self.map[key] = node
        self._add_front(node)
        if len(self.map) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]                    # this is why nodes store their key
# Python shortcut (mention it, but be ready for the manual version):
# OrderedDict with move_to_end(key) and popitem(last=False).

# ---------- 4.7 LFU cache: key -> (value, freq), freq -> ordered keys, plus min_freq ----------
class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.val: dict[int, int] = {}
        self.freq: dict[int, int] = {}
        self.buckets: defaultdict[int, OrderedDict] = defaultdict(OrderedDict)  # freq -> keys (LRU order)
        self.min_freq = 0

    def _touch(self, key: int) -> None:              # bump the frequency, keep recency within a bucket
        f = self.freq[key]
        del self.buckets[f][key]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq = f + 1
        self.freq[key] = f + 1
        self.buckets[f + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.val:
            return -1
        self._touch(key)
        return self.val[key]

    def put(self, key: int, value: int) -> None:
        if self.cap <= 0:
            return
        if key in self.val:
            self.val[key] = value
            self._touch(key)
            return
        if len(self.val) >= self.cap:
            old, _ = self.buckets[self.min_freq].popitem(last=False)   # least recent among least frequent
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.val[old], self.freq[old]
        self.val[key], self.freq[key] = value, 1
        self.buckets[1][key] = None
        self.min_freq = 1                           # a new key always has the minimum frequency

# ---------- 4.8 Iterator designs: keep the traversal state on a stack ----------
class BSTIterator:                                  # inorder, O(h) memory, amortized O(1) next
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    def _push_left(self, node) -> None:
        while node:
            self.stack.append(node)
            node = node.left
    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val
    def hasNext(self) -> bool:
        return bool(self.stack)

class NestedIterator:                               # LC 341; NestedInteger API: isInteger/getInteger/getList
    def __init__(self, nested_list):
        self.stack = nested_list[::-1]
    def next(self) -> int:
        return self.stack.pop().getInteger()        # hasNext() guarantees an integer on top
    def hasNext(self) -> bool:
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            self.stack.pop()
            self.stack.extend(top.getList()[::-1])  # expand lazily, only when needed
        return False
```

## 5. Worked examples

### Example 1 — Count of Smaller Numbers After Self (LC 315) — Hard (Fenwick)
**Brute force:** for each i, scan to the right → O(n²) with n = 10⁵. Dead.
**Aha (signal: "how many seen so far are smaller" = a rank query under insertion):** scan **right to left**. Maintain a Fenwick tree of counts indexed by value rank. For each element, query the count of ranks below it, then insert it. Compress the values to ranks 1..k first.
```python
def count_smaller(nums: list[int]) -> list[int]:
    ranks = {v: i + 1 for i, v in enumerate(sorted(set(nums)))}   # coordinate compression
    bit = BIT(len(ranks))
    res = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = ranks[nums[i]]
        res[i] = bit.prefix(r - 1)          # strictly smaller values already inserted (to the right)
        bit.add(r, 1)
    return res
```
**Trace** [5,2,6,1] → ranks {1:1, 2:2, 5:3, 6:4}. i=3 (1): 0. i=2 (6): 1. i=1 (2): 1. i=0 (5): 2 → [2,1,1,0] ✓.
**Complexity:** O(n log n) time, O(n) space.
**Alternative:** merge sort counting how many right-half elements jump ahead of each left element (same complexity, trickier indexing).

### Example 2 — Longest Duplicate Substring (LC 1044) — Hard (rolling hash + binary search)
**Brute force:** all substrings in a hash set → O(n²) substrings × O(n) hashing = O(n³).
**Two observations:**
1. **Monotone in the length.** If a duplicate of length L exists, one of length L−1 exists (drop a character). So binary search on L.
2. For fixed L, "is any length-L window repeated?" = hash all windows in O(n) with a rolling hash.
```python
def longest_dup_substring(s: str) -> str:
    a = [ord(c) - 97 for c in s]
    n = len(s)
    base = random.randint(256, 10**9)                # a random base defeats adversarial inputs

    def find_dup(L: int) -> int:                     # start of a duplicated length-L window, or -1
        seen: dict[int, list[int]] = {}
        for i, h in enumerate(window_hashes(a, L, base)):
            if h in seen and any(s[j:j + L] == s[i:i + L] for j in seen[h]):   # verify: no false positives
                return i
            seen.setdefault(h, []).append(i)
        return -1

    lo, hi = 1, n                                    # first L in [1, n) with no duplicate
    while lo < hi:
        mid = (lo + hi) // 2
        if find_dup(mid) == -1:
            hi = mid
        else:
            lo = mid + 1
    best = lo - 1
    if best == 0:
        return ""
    start = find_dup(best)
    return s[start:start + best]
```
**Trace** "banana": L=3 → "ana" repeats (1, 3) ✓. L=4 → none. Answer "ana".
**Complexity:** O(n log n) expected time (O(n) per check × log n checks; verification runs only on hash hits). O(n) space.
**Edge cases:** no duplicate → "". All the same character → n−1.

### Example 3 — The Skyline Problem (LC 218) — Hard (line sweep)
**Brute force:** for every x-coordinate, take the max height over all covering buildings → O(n²).
**Aha (signal: the outline changes only at building edges):** sweep the edges left to right, keeping the **active buildings in a max-heap** keyed by height. Lazily remove buildings whose right edge ≤ x. Record a key point whenever the current max height changes.
```python
def get_skyline(buildings: list[list[int]]) -> list[list[int]]:
    events = []
    for l, r, h in buildings:
        events.append((l, -h, r))         # start: taller buildings first at the same x
        events.append((r, 0, 0))          # end marker (the actual removal is lazy)
    events.sort()
    res: list[list[int]] = []
    heap = [(0, float('inf'))]            # (-height, right); ground level never expires
    for x, neg_h, r in events:
        if neg_h:
            heapq.heappush(heap, (neg_h, r))
        while heap[0][1] <= x:            # buildings that ended at or before x
            heapq.heappop(heap)
        height = -heap[0][0]
        if not res or res[-1][1] != height:
            res.append([x, height])
    return res
```
**Trace** [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]] → [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]] ✓.
**Complexity:** O(n log n) time, O(n) space.
**Edge cases:** adjacent buildings of equal height (no duplicate key point, thanks to the `!=` check). Several buildings starting at the same x (taller first, so no spurious intermediate point).

### Example 4 — LRU Cache (LC 146) — Medium (design)
**Requirements:** `get` and `put` in O(1), evicting the least recently used key.
**Naive:** a list ordered by recency → O(n) to move a key. A heap of timestamps → O(log n) plus stale entries.
**Aha:** **O(1) locate** needs a hash map. **O(1) reorder** needs a doubly linked list (unlink/relink given a node pointer). Store node pointers in the map. Sentinel head/tail nodes remove the edge cases. Code: template 4.6.
**Complexity:** O(1) per operation, O(capacity) space.
**Interview script:** (1) state the two requirements and the structure each implies, (2) draw the list with sentinels, (3) write the `_remove` and `_add_front` helpers first, then get/put in terms of them, (4) mention the OrderedDict version.
**LFU extension (LC 460, Hard):** add frequency buckets (freq → an OrderedDict of keys in LRU order) and a `min_freq` pointer. It only increases by 1 on a touch and resets to 1 on an insert, so eviction is O(1). Code: template 4.7.

## 6. Common mistakes & traps
- Fenwick: 0-indexed input fed to a 1-indexed tree (`add(0, …)` loops forever since `0 & −0 = 0`).
- Segment tree: inconsistent half-open vs closed ranges; forgetting to push lazy tags before recursing.
- KMP: advancing `i` on a fallback (only `k` changes), or building the LPS with `k` reset to 0 instead of `lps[k−1]`.
- Rolling hash: a small modulus (collisions), a fixed base against adversarial tests, a negative intermediate (Python's `%` fixes it, other languages don't), and forgetting that equal hashes ≠ equal strings.
- Line sweep: the tie order at equal x (starts vs ends, tall vs short).
- LRU: not storing the key in the node (you can't delete from the map on eviction), updating an existing key without moving it to the front.
- LFU: forgetting to update `min_freq` when its bucket empties on a touch, or not resetting it to 1 on insert.
- Iterators: doing all the work in the constructor (O(n) memory) when the interviewer asked for lazy iteration.

## 7. Connections
- **Prefix sums** (Ch 5) → Fenwick is prefix sums with updates. **Difference arrays** → range updates on a Fenwick.
- **Binary search** (Ch 6): rolling hash + binary search on the length; the Fenwick "lower bound" descent for k-th order statistics.
- **Merge sort** (Ch 1): inversion counting as an alternative to Fenwick.
- **Heaps** (Ch 11): the skyline, lazy deletion.
- **Hashing + linked lists** (Ch 2, Ch 8): LRU. **Stacks** (Ch 7): iterators.
- **Intervals** (Ch 13): line sweep generalizes the meeting-rooms event sweep.
- **Tries** (Ch 18): multi-pattern matching; KMP is the single-pattern tool.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 28 | [Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/) | Easy | Pattern in text → KMP |
| 232 | [Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/) | Easy | Amortized transfer between two stacks |
| 303 | [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) | Easy | No updates → prefix sums (the baseline) |
| 459 | [Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/) | Easy | Period from the LPS: lps[−1] > 0 and n % (n − lps[−1]) == 0 |
| 146 | [LRU Cache](https://leetcode.com/problems/lru-cache/) | Medium | O(1) get/put with recency → map + DLL |
| 173 | [Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/) | Medium | Paused inorder → stack of left spines |
| 341 | [Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/) | Medium | Lazy flattening → stack |
| 284 | [Peeking Iterator](https://leetcode.com/problems/peeking-iterator/) | Medium | Cache the next element |
| 981 | [Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/) | Medium | Increasing timestamps → bisect per key |
| 1146 | [Snapshot Array](https://leetcode.com/problems/snapshot-array/) | Medium | Per-index history + bisect on snap_id |
| 307 | [Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | Medium | Updates + range sums → Fenwick |
| 187 | [Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/) | Medium | Fixed-length repeated windows → rolling hash / set |
| 2034 | [Stock Price Fluctuation](https://leetcode.com/problems/stock-price-fluctuation/) | Medium | Corrections → map + heaps with lazy deletion |
| 895 | [Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/) | Hard | Frequency → stack of stacks |
| 460 | [LFU Cache](https://leetcode.com/problems/lfu-cache/) | Hard | Frequency buckets + min_freq |
| 315 | [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | Hard | Rank queries under insertion → Fenwick |
| 493 | [Reverse Pairs](https://leetcode.com/problems/reverse-pairs/) | Hard | Inversions with a factor → merge sort / Fenwick |
| 327 | [Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/) | Hard | Prefix sums in a window → Fenwick over compressed prefixes |
| 214 | [Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/) | Hard | Longest palindromic prefix → KMP on s + '#' + reversed(s) |
| 1044 | [Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/) | Hard | Rolling hash + binary search on the length |
| 218 | [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/) | Hard | Edges as events + max-heap |
| 715 | [Range Module](https://leetcode.com/problems/range-module/) | Hard | Dynamic interval set → sorted boundaries or a segment tree |

## 9. Recognition drill
1. An array with point updates and queries "minimum on [l, r]".
2. Design a structure: `inc(key)`, `dec(key)`, `getMaxKey()`, `getMinKey()` all in O(1).
3. Given a string, find the longest prefix that is also a suffix (not the whole string).
4. Given n rectangles, compute the total area covered.
5. Given an array, count pairs i < j with `nums[i] > 2 * nums[j]`.

<details><summary>Answers</summary>

1. **Segment tree** (min isn't invertible, so Fenwick doesn't apply directly).
2. **Hash map + a doubly linked list of count buckets** (All O`one Data Structure, LC 432). The LFU idea with ±1 moves.
3. **KMP prefix function**: `lps[-1]` (LC 1392 Longest Happy Prefix).
4. **Line sweep over x with a segment tree (or a sorted list) over y coverage** (LC 850). Coordinate-compress y.
5. **Merge sort with a two-pointer count before merging, or a Fenwick tree over compressed values** (LC 493).
</details>

Next: Part 22 — Instinct Training
