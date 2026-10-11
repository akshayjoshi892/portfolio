# Part 11 — Heaps / Priority Queues

## 1. Core insight
A heap keeps the **min (or max) of a changing set** available in O(1), with O(log n) insert and extract. Use it when you repeatedly need "the best current candidate" while candidates arrive or leave. Sorting can't do that, because the set changes. There are three signature patterns. **Top-k**: a heap of size k with the *opposite* polarity evicts the worst of the best, giving O(n log k). **k-way merge**: the heap holds the frontier of k sorted streams. **Two heaps**: a max-heap for the lower half and a min-heap for the upper half, for running medians or balanced splits. Many greedy algorithms (scheduling, Dijkstra) are "sort by one key, heap by another".

## 2. Trigger signals / Anti-signals
**Triggers**
- "k-th largest / smallest", "top k", "k closest", "k most frequent".
- "Merge k sorted lists/arrays", "smallest range covering k lists", "k-th smallest in a sorted matrix", "k pairs with smallest sums".
- "Median of a data stream", "sliding window median".
- Simulation with time: "process tasks by earliest availability", "CPU scheduling", "meeting rooms", "IPO: pick the most profitable project you can afford".
- "Repeatedly take the two largest / smallest and combine" (Last Stone Weight, connect ropes).
- Greedy with regret: "take it now, drop the worst later if needed" (Course Schedule III, Furthest Building).

**Anti-signals**
- k-th element of a static array, with average O(n) acceptable → **quickselect** beats the heap.
- Frequencies bounded by n → **bucket sort** gives O(n) top-k.
- Window max/min → **monotonic deque** (O(n)) beats a heap (O(n log n)).
- You need to delete arbitrary elements or search by key → a heap has no O(log n) arbitrary delete. Use lazy deletion or a sorted container.
- Static data, all queries at the end → just sort.

## 3. Mental model
**A bouncer at a VIP room of size k** (top-k largest): the room holds the k largest seen so far, and the bouncer watches the **weakest** member (a min-heap). A newcomer stronger than the weakest replaces them.
```
stream: 5 1 9 3 7, k = 3, min-heap of size 3
5       -> [5]
1       -> [1,5]
9       -> [1,5,9]
3 > 1   -> pop 1, push 3 -> [3,5,9]
7 > 3   -> pop 3, push 7 -> [5,7,9]    top is the 3rd largest = 5
```
**Two heaps for the median:**
```
   lower half (max-heap)  |  upper half (min-heap)
      ... 3  5  [7]       |  [8]  10  12 ...
                 ^ max           ^ min
   invariant: every lower ≤ every upper; len(lower) - len(upper) ∈ {0, 1}
```

## 4. Canonical templates

```python
import heapq
import random

# Python's heapq is a MIN-heap on a list.
# heappush / heappop: O(log n). heapify: O(n). h[0] is the min.
# Max-heap: push -x. Ties/uncomparable objects: push (key, tiebreak_id, obj).

# 4.1 Top-k largest with a min-heap of size k: O(n log k)
def top_k_largest(nums: list[int], k: int) -> list[int]:
    h: list[int] = []
    for x in nums:
        if len(h) < k:
            heapq.heappush(h, x)
        elif x > h[0]:
            heapq.heapreplace(h, x)          # pop the min and push x in one O(log k) step
    return h                                 # h[0] is the k-th largest
# Library: heapq.nlargest(k, nums) / heapq.nsmallest(k, nums, key=...)

# 4.2 k-way merge: the heap holds one frontier element per stream
def merge_sorted_arrays(arrays: list[list[int]]) -> list[int]:
    h = [(arr[0], i, 0) for i, arr in enumerate(arrays) if arr]
    heapq.heapify(h)
    out: list[int] = []
    while h:
        val, i, j = heapq.heappop(h)
        out.append(val)
        if j + 1 < len(arrays[i]):
            heapq.heappush(h, (arrays[i][j + 1], i, j + 1))
    return out

# 4.3 Sort by one key, heap by another (IPO pattern: unlock by capital, pick by profit)
def find_maximized_capital(k: int, w: int, profits: list[int], capital: list[int]) -> int:
    projects = sorted(zip(capital, profits))
    avail: list[int] = []                    # max-heap of profits we can afford
    i = 0
    for _ in range(k):
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(avail, -projects[i][1])
            i += 1
        if not avail:
            break
        w -= heapq.heappop(avail)            # add the best available profit
    return w

# 4.4 Lazy deletion: mark removed items and skip them when they surface at the top
#     (used when an element must be "removed" but isn't at the top, e.g. sliding window median)
```
**Trace 4.3** k=2, w=0, profits=[1,2,3], capital=[0,1,1]: afford (0,1) → take 1, w=1. Now afford (1,2),(1,3) → take 3, w=4 ✓.

## 5. Worked examples

### Example 1 — Kth Largest Element in an Array (LC 215) — Medium
**Brute force:** sort and index → O(n log n).
**Heap (signal: "k-th largest"):** template 4.1 → O(n log k) time, O(k) space. That's good when k ≪ n or the data is a stream.
**Aha for the static array:** we don't need order among the other elements, only the element *at rank* n−k. **Quickselect**: partition around a random pivot and recurse into only one side. T(n) = T(n/2) + O(n) → O(n) on average. Use a **3-way partition** so arrays with many duplicates don't degrade.
```python
def find_kth_largest(nums: list[int], k: int) -> int:
    target = len(nums) - k                   # index in ascending order
    lo, hi = 0, len(nums) - 1
    while True:
        pivot = nums[random.randint(lo, hi)]
        lt, i, gt = lo, lo, hi               # [lo,lt) < pivot, [lt,i) == pivot, (gt,hi] > pivot
        while i <= gt:
            if nums[i] < pivot:
                nums[lt], nums[i] = nums[i], nums[lt]; lt += 1; i += 1
            elif nums[i] > pivot:
                nums[i], nums[gt] = nums[gt], nums[i]; gt -= 1
            else:
                i += 1
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            return pivot                     # target lies in the == pivot block
```
**Trace** [3,2,1,5,6,4], k=2 → target index 4 in sorted [1,2,3,4,5,6] → 5 ✓ (the path depends on the random pivots).
**Complexity:** O(n) average, O(n²) worst case (made vanishingly unlikely by the random pivot). O(1) space. Heap version: O(n log k), O(k).
**Edge cases:** all duplicates (the 3-way partition returns at once). k = 1, k = n.
**Interview script:** give the heap version first (safe and simple), then offer quickselect with the average vs worst-case trade-off.

### Example 2 — Merge k Sorted Lists (LC 23) — Hard
**Brute force 1:** concatenate and sort → O(N log N), N = total nodes.
**Brute force 2:** merge lists one by one → O(N·k), because early nodes get re-merged k times.
**Aha (signal: k sorted streams):** the next output is always one of the k current heads. Keep the heads in a min-heap → each of the N nodes costs O(log k).
```python
import heapq
from typing import Optional

def merge_k_lists(lists: list[Optional['ListNode']]) -> Optional['ListNode']:
    h = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapq.heapify(h)                         # i breaks ties (ListNode isn't comparable)
    dummy = tail = ListNode()
    while h:
        _, i, node = heapq.heappop(h)
        tail.next = tail = node
        if node.next:
            heapq.heappush(h, (node.next.val, i, node.next))
    return dummy.next
```
(`ListNode` as in Ch 8. `tail.next = tail = node` assigns left to right: first `tail.next = node`, then `tail = node`.)
**Complexity:** O(N log k) time, O(k) heap space.
**Alternative:** divide and conquer, pairwise merging in rounds → also O(N log k), with O(1) extra space iteratively.
**Edge cases:** empty `lists`, lists containing `None`, one list.

### Example 3 — Find Median from Data Stream (LC 295) — Hard
**Brute force:** keep a sorted list with `bisect.insort` → O(n) insert (shifting), O(1) median.
**Aha (signal: the median = the boundary between two halves):** we only ever need the max of the lower half and the min of the upper half. Two heaps give O(log n) insert and O(1) median.
```python
import heapq

class MedianFinder:
    def __init__(self) -> None:
        self.lo: list[int] = []              # max-heap (stored negated): lower half
        self.hi: list[int] = []              # min-heap: upper half

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lo, -num)                        # 1) add to lower
        heapq.heappush(self.hi, -heapq.heappop(self.lo))     # 2) move lower's max up (keeps order)
        if len(self.hi) > len(self.lo):                      # 3) rebalance sizes
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self) -> float:
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2
```
**Trace** add 1 → lo=[1], hi=[]. Add 2 → lo gets 2, moves 2 up → lo=[1], hi=[2] → median 1.5. Add 3 → lo gets 3, moves 3 up → hi=[2,3] → rebalance moves 2 down → lo=[2,1], hi=[3] → median 2 ✓.
**Complexity:** O(log n) per add, O(1) per median. O(n) space.
**Follow-ups:** values in [0,100] → a counting array with O(100) median. Sliding window median → two heaps + lazy deletion, or a sorted list.

## 6. Common mistakes & traps
- Forgetting that `heapq` is min-only → negate for max, and **negate back** on read.
- Pushing uncomparable objects (ListNode, dicts) → TypeError on ties. Add a unique tiebreaker.
- Top-k largest with a **max**-heap of all n elements → O(n log n). Use a min-heap of size k.
- `heapq.heapify` returns None (it's in place).
- Reading `h[0]` on an empty heap.
- Assuming the heap list is sorted. Only `h[0]` is guaranteed.
- Lazy deletion: forgetting to clean the top before reading it, or keeping the size counters wrong.
- Using a heap where a monotonic deque or bucket sort gives a better complexity. Mention the alternative.

## 7. Connections
- **Greedy** (Ch 12): scheduling (Task Scheduler, Single-Threaded CPU, IPO), regret greedy (Course Schedule III, Furthest Building You Can Reach).
- **Intervals** (Ch 13): Meeting Rooms II = a min-heap of end times.
- **Graphs** (Ch 16): Dijkstra and Prim are "BFS with a heap".
- **Binary search** (Ch 6): k-th smallest in a sorted matrix / pair sums can use a heap *or* binary search on the value.
- **Hash map + heap**: top k frequent (or bucket sort), design Twitter (merge k feeds).
- **Sorting + heap**: "sort by time, heap by priority" is the core of most simulation problems.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 703 | [Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) | Easy | Stream + k-th largest → size-k min-heap |
| 1046 | [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/) | Easy | Repeatedly take the two largest |
| 973 | [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) | Medium | Top-k by distance (heap or quickselect) |
| 215 | [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) | Medium | Static k-th → quickselect; heap as a baseline |
| 347 | [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) | Medium | Counts + top-k (heap or bucket) |
| 692 | [Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/) | Medium | Top-k with a lexicographic tiebreak |
| 621 | [Task Scheduler](https://leetcode.com/problems/task-scheduler/) | Medium | Always run the most frequent available task (or the math formula) |
| 767 | [Reorganize String](https://leetcode.com/problems/reorganize-string/) | Medium | No two adjacent equal → alternate the two most frequent |
| 355 | [Design Twitter](https://leetcode.com/problems/design-twitter/) | Medium | News feed = merge k recent lists |
| 373 | [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/) | Medium | Implicit sorted matrix → k-way merge frontier |
| 1834 | [Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/) | Medium | Sort by arrival, heap by (duration, index) |
| 1642 | [Furthest Building You Can Reach](https://leetcode.com/problems/furthest-building-you-can-reach/) | Medium | Ladders for the largest climbs → regret min-heap |
| 23 | [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) | Hard | k sorted streams |
| 295 | [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) | Hard | Median of a stream → two heaps |
| 502 | [IPO](https://leetcode.com/problems/ipo/) | Hard | Unlock by capital (sort), choose by profit (heap) |
| 632 | [Smallest Range Covering Elements from K Lists](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/) | Hard | Track max while popping the min of k frontiers |
| 480 | [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/) | Hard | Two heaps + lazy deletion |

## 9. Recognition drill
1. Given ropes of various lengths, connecting two costs their sum. Find the minimum total cost to connect all of them.
2. Given a stream of (timestamp, price) updates that may correct earlier prices, report the current max price.
3. Hire exactly k workers where each is paid proportionally to quality at the same wage ratio, and each worker's minimum wage must be met. Minimize the total cost.
4. Return the k-th smallest sum among all pairs (a[i], b[j]) of two sorted arrays.
5. Given n tasks with durations and deadlines, maximize the number of tasks completed on time (one at a time).

<details><summary>Answers</summary>

1. **Min-heap, repeatedly combine the two smallest** (Huffman-style greedy).
2. **Hash map timestamp→price + max-heap with lazy deletion** (skip heap tops whose price no longer matches the map) (LC 2034).
3. **Sort by wage/quality ratio + max-heap of the k smallest qualities** (LC 857). The current ratio is the captain, and the heap keeps the sum of qualities minimal.
4. **k-way merge frontier** starting from (a[i], b[0]) for each i (or binary search on the sum value).
5. **Sort by deadline + max-heap of durations (regret greedy)**: add each task; if the total time exceeds the deadline, drop the longest (LC 630).
</details>

Next: Part 12 — Greedy
