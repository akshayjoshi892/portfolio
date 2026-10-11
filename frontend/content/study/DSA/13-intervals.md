# Part 13 — Intervals (Sort + Sweep, Merging, Meeting Rooms)

## 1. Core insight
Unsorted intervals can overlap in O(n²) ways. **Sorted by start**, an interval can only overlap the "currently open" group, so a single left-to-right sweep that tracks the running end merges everything in O(n log n). There are three recurring sweeps. **Merge sweep** (sort by start, extend or close the current block). **Selection sweep** (sort by *end*, greedily keep compatible intervals; see Ch 12). **Event sweep** (split each interval into +1 at the start and −1 at the end, sort the events, and the running sum is the number of active intervals at each moment). A min-heap of end times is the event sweep's dynamic cousin, and it tells you *which* intervals are active.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Merge overlapping", "insert an interval", "union / total covered length".
- "Minimum meeting rooms", "maximum concurrent", "max overlap at any point", "car pooling capacity".
- "Minimum removals so none overlap", "maximum number of non-overlapping", "arrows to burst balloons" → sort by end.
- "Intersection of two interval lists" → two pointers over both sorted lists.
- "For each query point, the best interval containing it" → sort queries offline + heap.
- Calendar booking with incremental inserts → sorted structure or brute-force check (n small).

**Anti-signals**
- Coordinates are small integers (≤ 10⁶) with many range adds → a **difference array** (Ch 5) may be simpler.
- Weighted interval scheduling (maximize the *sum of weights*, not the count) → **DP + binary search** (LC 1235), not greedy.
- Online dynamic insert/delete with range queries → segment tree / balanced BST (Ch 21).

## 3. Mental model
A timeline, scanned left to right:
```
sorted by start:
[1-----3]
   [2------6]
                [8--10]
                     [9---12]
                                [15-18]
sweep: cur=[1,3] -> 2<=3 extend to [1,6] -> 8>6 close, cur=[8,10] -> 9<=10 extend [8,12] -> close -> [15,18]
result: [1,6] [8,12] [15,18]
```
Event sweep: `+1` at each start, `−1` at each end → the running count = the rooms in use:
```
time:    1   2   3   6   8   9   10  12
event:  +1  +1  -1  -1  +1  +1  -1  -1
active:  1   2   1   0   1   2   1   0     -> max = 2 rooms
```

## 4. Canonical templates

```python
import heapq

# 4.1 Merge overlapping intervals
def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    res: list[list[int]] = []
    for s, e in intervals:
        if res and s <= res[-1][1]:            # overlaps the open block (touching counts)
            res[-1][1] = max(res[-1][1], e)    # max: the new interval may be nested inside
        else:
            res.append([s, e])
    return res

# 4.2 Insert into sorted, non-overlapping intervals: three phases
def insert(intervals: list[list[int]], new: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    s, e = new
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < s:       # 1) entirely before
        res.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= e:      # 2) overlapping: absorb
        s, e = min(s, intervals[i][0]), max(e, intervals[i][1]); i += 1
    res.append([s, e])
    res.extend(intervals[i:])                  # 3) entirely after
    return res

# 4.3 Event sweep: max concurrent intervals ([start, end) semantics)
def max_concurrent(intervals: list[list[int]]) -> int:
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()                              # (t, -1) sorts before (t, +1): an end frees the room first
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best

# 4.4 Intersection of two sorted interval lists
def interval_intersection(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    i = j = 0
    res: list[list[int]] = []
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if lo <= hi:
            res.append([lo, hi])
        if A[i][1] < B[j][1]:                  # drop the one that ends first
            i += 1
        else:
            j += 1
    return res

# 4.5 Min removals to make non-overlapping = n - max kept (sort by end)
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    kept, end = 0, float('-inf')
    for s, e in sorted(intervals, key=lambda x: x[1]):
        if s >= end:
            kept += 1
            end = e
    return len(intervals) - kept
```
**Trace 4.2** [[1,3],[6,9]], new [2,5]: phase 1 nothing ([1,3] ends at 3 ≥ 2). Phase 2 absorbs [1,3] → [1,5]. [6,9] starts at 6 > 5 → stop. Result [[1,5],[6,9]] ✓.

## 5. Worked examples

### Example 1 — Merge Intervals (LC 56) — Medium
**Brute force:** repeatedly find any overlapping pair and merge them until none remain → O(n²) per pass, up to O(n³).
**Bottleneck:** without order, any interval might overlap any other.
**Aha (lever: sort first):** after sorting by start, if the next interval doesn't overlap the current block, **no later interval can** either (they start even later). So each interval is compared only with the open block. Code: template 4.1.
**Complexity:** O(n log n) time for the sort, O(n) output space (O(log n)–O(n) sort space).
**Edge cases:** nested `[1,10],[2,3]` (the `max` matters), touching `[1,2],[2,3]` → merged under `<=`, a single interval.

### Example 2 — Meeting Rooms II (LC 253, Premium) — Medium
**Problem:** the minimum number of rooms for all meetings `[start, end)`.
**Brute force:** for each time point, count the meetings covering it → O(n · T), or O(n²) over the start points.
**Aha (signal: "at the same time"):** rooms needed = the max number of simultaneously active meetings. Either use an event sweep (4.3), or process by start time with a **min-heap of end times**. If the earliest-ending room is free by the time this meeting starts, reuse it.
```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda x: x[0])
    ends: list[int] = []                    # end times of rooms in use
    for s, e in intervals:
        if ends and ends[0] <= s:           # the earliest-freeing room is available
            heapq.heapreplace(ends, e)
        else:
            heapq.heappush(ends, e)
    return len(ends)
```
**Trace** [[0,30],[5,10],[15,20]]: push 30. 5 < 30 → push 10 (2 rooms). 15 ≥ 10 → replace 10 with 20. Answer 2 ✓.
**Complexity:** O(n log n) time, O(n) space.
**Edge cases:** back-to-back meetings `[1,5],[5,10]` need 1 room (`<=`). Empty → 0.
**Why the heap version is useful:** it generalizes to "assign actual room IDs" (LC 2402 Meeting Rooms III) where the plain count doesn't.

### Example 3 — Minimum Interval to Include Each Query (LC 1851) — Hard
**Problem:** for each query point q, the size of the smallest interval `[l, r]` with `l ≤ q ≤ r`, or −1. n, q up to 10⁵.
**Brute force:** for each query scan all intervals → O(n·q) = 10¹⁰.
**Aha (signal: many point queries → process offline in sorted order):** sort the queries. Sweep them in increasing order. Push every interval with `l ≤ q` into a min-heap keyed by size. Lazily pop intervals with `r < q`. They're expired for this query **and every later one** (queries only increase), so lazy deletion is safe.
```python
import heapq

def min_interval(intervals: list[list[int]], queries: list[int]) -> list[int]:
    intervals.sort()
    h: list[tuple[int, int]] = []            # (size, right_end)
    ans: dict[int, int] = {}
    i = 0
    for q in sorted(set(queries)):
        while i < len(intervals) and intervals[i][0] <= q:   # starts at or before q
            l, r = intervals[i]
            heapq.heappush(h, (r - l + 1, r))
            i += 1
        while h and h[0][1] < q:                             # ended before q: dead forever
            heapq.heappop(h)
        ans[q] = h[0][0] if h else -1
    return [ans[q] for q in queries]                         # restore the original order
```
**Trace** intervals [[1,4],[2,4],[3,6],[4,4]], queries [2,3,4,5] → 3, 3, 1, 4 ✓. At q=5: [1,4],[2,4],[4,4] have expired, [3,6] (size 4) remains.
**Complexity:** O((n + q) log n) time, O(n + q) space.
**Transferable pattern:** **offline queries** — sort the queries alongside the data so each data item is inserted once and expired once.

## 6. Common mistakes & traps
- Overlap definition: `s <= end` (touching overlaps) vs `s < end`. Read the problem's semantics (closed vs half-open).
- Forgetting `max()` when extending (a nested interval shrinks the block).
- Sorting by the wrong key: merge → start. Max non-overlapping / arrows → end.
- Event tie-breaking: at equal times, should an end come before a start? For `[s, e)` meetings, yes: `(t, −1) < (t, +1)` sorts that way naturally.
- Mutating the input list while iterating over it (in insert/merge).
- Returning query answers in sorted order instead of the original order (offline processing).
- Using a greedy for **weighted** interval scheduling. That's DP + binary search.

## 7. Connections
- **Greedy** (Ch 12): activity selection, arrows, Non-overlapping Intervals.
- **Heap** (Ch 11): meeting rooms, offline queries, Employee Free Time (k-way merge of schedules).
- **Difference array** (Ch 5): the event sweep over integer coordinates.
- **Line sweep** (Ch 21): the skyline problem, rectangle area union. Events + a heap or segment tree.
- **Binary search / DP**: weighted job scheduling (LC 1235) = sort by end, `dp[i] = max(dp[i−1], w_i + dp[last compatible])`, with the last compatible found by bisect.
- **Sorted containers / segment tree**: My Calendar I/II/III, Range Module.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 252 | [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) (Premium) | Easy | Any overlap after sorting by start? |
| 56 | [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | Medium | Sort by start, extend the open block |
| 57 | [Insert Interval](https://leetcode.com/problems/insert-interval/) | Medium | Already sorted → three phases, O(n) |
| 435 | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | Medium | Min removals = n − max kept (sort by end) |
| 452 | [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/) | Medium | Shoot at the earliest end; touching counts |
| 986 | [Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/) | Medium | Two sorted lists → two pointers, drop the earlier end |
| 253 | [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) (Premium) | Medium | Max simultaneous → heap of ends / events |
| 1094 | [Car Pooling](https://leetcode.com/problems/car-pooling/) | Medium | Load over time vs capacity → events |
| 1288 | [Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/) | Medium | Sort by start asc, end desc; track the max end |
| 729 | [My Calendar I](https://leetcode.com/problems/my-calendar-i/) | Medium | Incremental insert with an overlap check |
| 1353 | [Maximum Number of Events That Can Be Attended](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/) | Medium | Each day, attend the open event ending soonest |
| 759 | [Employee Free Time](https://leetcode.com/problems/employee-free-time/) (Premium) | Hard | Merge all schedules, report the gaps |
| 1851 | [Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/) | Hard | Offline sorted queries + heap with lazy expiry |
| 1235 | [Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/) | Hard | Weighted → DP + binary search, not greedy |
| 218 | [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/) | Hard | Critical points + max-heap of active heights |

## 9. Recognition drill
1. Given a list of closed intervals, return the total length covered by their union.
2. Given flight bookings `(first, last, seats)` on n flights, return the seats booked per flight.
3. Given meetings with start, end, and value, attend non-overlapping meetings to maximize the total value.
4. Given people's work intervals, find the maximum number of people working at the same moment and *when* it occurs.
5. Given intervals, find the minimum number of points such that every interval contains at least one point.

<details><summary>Answers</summary>

1. **Merge after sorting by start**, then sum the lengths (or an event sweep counting coverage > 0).
2. **Difference array** (range add, one prefix pass). Coordinates are small integers.
3. **Weighted interval scheduling → DP + binary search** (sort by end; greedy fails with weights).
4. **Event sweep** (+1/−1, sorted with ends before starts at equal times if half-open); record the time when the running count peaks.
5. **Greedy by end** (the arrows problem): place a point at the earliest end, skip everything it covers.
</details>

Next: Part 14 — Graphs
