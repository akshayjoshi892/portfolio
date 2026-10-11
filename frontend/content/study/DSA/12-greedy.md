# Part 12 — Greedy (and How to Prove or Disprove It)

## 1. Core insight
A greedy algorithm commits to the locally best choice and **never revisits it**. It's correct only when the problem has the **greedy-choice property**: some optimal solution begins with the greedy choice, so committing to it loses nothing. What remains is a smaller instance of the same problem. Greedy is fast and short, but it's the most error-prone technique in interviews, because wrong greedies *look* right. The expert habit is: **propose greedy → try to break it with a tiny counterexample → if it survives, prove it with an exchange argument or "stays ahead" → then code it.** If it breaks, the counterexample usually shows which state DP needs to track.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Maximum number of non-overlapping / compatible …" → sort by **end**, take the earliest finishing.
- "Minimum number of X to cover / reach / burst …" → furthest reach, or sort by end.
- "Can you reach the end", "minimum jumps" → track the furthest reach.
- A circular route with a gas/cost balance → running balance with reset.
- Assignment/matching with sorted sizes (cookies to children, boats) → sort both, match two pointers.
- "Rearrange to satisfy a local constraint" (no two adjacent equal, candy ratings) → most-frequent-first, or a two-pass sweep.
- Scheduling with deadlines/profits → sort by deadline + heap (regret).
- A problem so constrained that DP would be O(n²) with n = 10⁵ → the setter expects greedy.

**Anti-signals (greedy likely fails → DP)**
- Coin systems in general (`coins = [1,3,4]`, amount 6: greedy 4+1+1 = 3 coins, optimal 3+3 = 2).
- 0/1 knapsack (taking by value density fails; fractional knapsack is greedy).
- The choice affects **future options in a non-monotone way** (House Robber: taking the max house can block two good neighbors).
- "Count the number of ways". Greedy picks one solution; it doesn't count.
- The problem asks for an optimum over *subsequences* with interacting elements (LIS, edit distance).

## 3. Mental model
**How to prove greedy (pick one):**
1. **Exchange argument.** Take any optimal solution O that differs from greedy G at the first decision. Swap O's choice for G's choice, and show the result is still valid and no worse. Repeat until O = G. So G is optimal.
2. **Greedy stays ahead.** Show by induction that after each step, greedy's partial solution is at least as good as any other solution's (e.g., greedy's k-th interval ends no later than the optimal's k-th).
3. **Invariant / bound.** Show a lower bound every solution must meet, and that greedy meets it exactly (Task Scheduler's idle formula).

**How to disprove greedy fast:** try inputs of size 2–4 designed to punish the rule. Ties, one huge element, an item that's locally best but blocks two others, sort orders reversed. Spend 60 seconds on this before coding.

```
Activity selection: why sort by END, not START or LENGTH?
by start:   [0----------10]  would block
                [1-2] [3-4] [5-6]     -> greedy 1, optimal 3
by length:  [0---5] [4-6] [5---10]    -> short [4-6] blocks both -> greedy 1, optimal 2
by end:     earliest end leaves the most room -> exchange argument works
```

## 4. Canonical templates

```python
import heapq

# 4.1 Interval scheduling: max non-overlapping (sort by end)
def max_non_overlapping(intervals: list[list[int]]) -> int:
    count, end = 0, float('-inf')
    for s, e in sorted(intervals, key=lambda x: x[1]):
        if s >= end:                    # compatible (touching allowed here)
            count += 1
            end = e
    return count

# 4.2 Furthest reach (Jump Game I)
def can_jump(nums: list[int]) -> bool:
    reach = 0
    for i, x in enumerate(nums):
        if i > reach:                   # i is unreachable
            return False
        reach = max(reach, i + x)
    return True

# 4.3 Two-pass for two-sided local constraints (Candy)
def candy(ratings: list[int]) -> int:
    n = len(ratings)
    c = [1] * n
    for i in range(1, n):               # satisfy the left neighbor
        if ratings[i] > ratings[i - 1]:
            c[i] = c[i - 1] + 1
    for i in range(n - 2, -1, -1):      # satisfy the right neighbor without breaking the left
        if ratings[i] > ratings[i + 1]:
            c[i] = max(c[i], c[i + 1] + 1)
    return sum(c)

# 4.4 Regret greedy: take everything, undo the worst choice when a constraint breaks (Example 3)

# 4.5 Sort both sides, match greedily (Assign Cookies)
def find_content_children(greed: list[int], cookies: list[int]) -> int:
    greed.sort(); cookies.sort()
    child = 0
    for cookie in cookies:              # the smallest cookie that satisfies the least greedy child
        if child < len(greed) and cookie >= greed[child]:
            child += 1
    return child
```
**Trace 4.3** [1,0,2]: left pass [1,1,2]; right pass i=0: 1>0 → max(1, 2) = 2 → [2,1,2] → 5 ✓.

## 5. Worked examples

### Example 1 — Jump Game II (LC 45) — Medium
**Problem:** minimum jumps to reach the last index (always reachable).
**Brute force:** BFS over indices, or DP `dp[i] = min(dp[j]) + 1` over j that reach i → O(n²).
**Aha (implicit BFS):** the indices reachable with exactly `jumps` jumps form a **contiguous window** `[start, cur_end]`. The next window extends to the furthest reach from within the current one. So count the BFS levels with O(1) state.
```python
def jump(nums: list[int]) -> int:
    jumps = cur_end = farthest = 0
    for i in range(len(nums) - 1):          # never "jump" from the last index
        farthest = max(farthest, i + nums[i])
        if i == cur_end:                    # exhausted this level -> must jump
            jumps += 1
            cur_end = farthest
    return jumps
```
**Proof (stays ahead):** after k jumps, greedy's reachable frontier `cur_end` is the maximum possible over all k-jump strategies, because it takes the furthest reach over *every* index reachable in k jumps.
**Trace** [2,3,1,1,4]: i=0: far=2, level end → jumps=1, end=2. i=1: far=4. i=2: end → jumps=2, end=4. Stop (i < 4). Answer 2 ✓.
**Complexity:** O(n) time, O(1) space.
**Edge cases:** n = 1 → 0 (the loop doesn't run).

### Example 2 — Gas Station (LC 134) — Medium
**Problem:** the start index to complete a circuit, or −1. The answer is unique if it exists.
**Brute force:** simulate from every start → O(n²).
**Two facts:**
1. If `sum(gas) < sum(cost)`, it's impossible. Otherwise a solution exists (a classic result, proven via the prefix sum minimum).
2. **If starting at s you first run dry at i, then no start in (s, i] works.** You arrived at every intermediate j with tank ≥ 0. Starting at j means starting with tank 0 ≤ that, so you'd run dry at i or earlier.
**Aha:** fact 2 lets you **skip** all those starts. Reset the start to i+1. One pass.
```python
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    if sum(gas) < sum(cost):
        return -1
    tank = start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:            # can't reach i+1 from `start` (or anything in between)
            start = i + 1
            tank = 0
    return start
```
**Trace** gas=[1,2,3,4,5], cost=[3,4,5,1,2]: diffs −2,−2,−2,3,3. i=0..2 reset → start=3. Then tank=3, 6 → answer 3 ✓.
**Complexity:** O(n) time, O(1) space.
**Why the returned start works for the wrap-around:** the segment [start, n) has non-negative running sums. The total is ≥ 0, so the wrapped part [0, start) has a sum ≥ −(sum of [start, n)), and every prefix of the wrap stays ≥ 0 because the minimum prefix sum occurs right before `start`.

### Example 3 — Course Schedule III (LC 630) — Hard
**Problem:** courses `(duration, last_day)`, taken one at a time. Maximize the number of courses finished by their deadlines.
**Naive greedies (break them):**
- **Shortest first:** `[(2,2), (1,3)]`. It takes (1,3) at t=1, then (2,2) ends at t=3 > 2 ✗ → 1 course. Optimal: (2,2) then (1,3) → 2. Duration ignores urgency.
- **Earliest deadline first, never undo:** `[(4,4), (1,5), (1,5), (1,5)]`. It takes (4,4) at t=4 and (1,5) at t=5. The rest don't fit → 2 courses. Optimal: skip the long course and take the three 1s → 3. EDF is the right *order*, but an early long course can block several later short ones.
**Aha (sort by deadline + regret heap):** process courses by deadline and tentatively take each. If the total time exceeds the current deadline, **drop the longest course taken so far**. That keeps the count the same while freeing the most time. Exchange argument: among schedules with k courses from this prefix, the one with minimum total duration is best for the future, and dropping the longest achieves that minimum.
```python
import heapq

def schedule_course(courses: list[list[int]]) -> int:
    courses.sort(key=lambda c: c[1])        # by deadline
    taken: list[int] = []                   # max-heap of durations (negated)
    t = 0
    for dur, last in courses:
        heapq.heappush(taken, -dur)
        t += dur
        if t > last:
            t += heapq.heappop(taken)       # pops -max_dur -> subtracts the longest course
    return len(taken)
```
**Trace** [(4,4),(1,5),(1,5),(1,5)]: take 4 (t=4). Take 1 (t=5). Take 1 (t=6 > 5) → drop 4 → t=2. Take 1 (t=3). Taken = 3 ✓.
**Complexity:** O(n log n) time, O(n) space.

## 6. Common mistakes & traps
- Coding a greedy without trying to break it. Always test 2–3 adversarial mini-inputs.
- Sorting by the wrong key (start vs end, ascending vs descending). Derive the key from the exchange argument.
- Touching intervals: is `[1,2]` and `[2,3]` an overlap? Read the problem (LC 435 says no; LC 452 says yes, balloons touching at x=2 burst together).
- Two-sided constraints handled in one pass (Candy needs two).
- Jump Game II: iterating to `n` instead of `n − 1` adds an extra jump.
- Assuming greedy works because it passes the examples. LC examples rarely contain the counterexample.

## 7. Connections
- **DP** (Ch 17): greedy is DP where only one choice per state is ever optimal. When greedy fails, the counterexample tells you what state DP needs. Jump Game has an O(n²) DP and an O(n) greedy.
- **Heap** (Ch 11): regret greedy, scheduling, "best available now".
- **Sorting + intervals** (Ch 13): most interval greedies are "sort by end".
- **Binary search on answer** (Ch 6): the feasibility check is usually greedy.
- **Monotonic stack** (Ch 7): Remove K Digits / Remove Duplicate Letters are greedy with a stack.
- **MST / Dijkstra** (Ch 16): greedy algorithms with cut-property / settled-distance proofs.
- Mistaken for DP: Jump Game, Gas Station, Partition Labels, Task Scheduler. Mistaken for greedy: Coin Change, House Robber, 0/1 knapsack.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 455 | [Assign Cookies](https://leetcode.com/problems/assign-cookies/) | Easy | Sort both, match smallest-sufficient |
| 122 | [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) | Medium | Unlimited trades → sum every positive diff |
| 53 | [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) | Medium | A negative running sum can never help → reset (Kadane) |
| 55 | [Jump Game](https://leetcode.com/problems/jump-game/) | Medium | Reachability → furthest reach |
| 45 | [Jump Game II](https://leetcode.com/problems/jump-game-ii/) | Medium | Minimum jumps → BFS levels as windows |
| 134 | [Gas Station](https://leetcode.com/problems/gas-station/) | Medium | Failing at i rules out all starts before i |
| 763 | [Partition Labels](https://leetcode.com/problems/partition-labels/) | Medium | Extend the part to the last occurrence of each char |
| 846 | [Hand of Straights](https://leetcode.com/problems/hand-of-straights/) | Medium | The smallest card must start a group |
| 678 | [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/) | Medium | Track the range [min_open, max_open] |
| 1899 | [Merge Triplets to Form Target Triplet](https://leetcode.com/problems/merge-triplets-to-form-target-triplet/) | Medium | Discard any triplet exceeding the target |
| 435 | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | Medium | Max kept = activity selection by end |
| 452 | [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/) | Medium | Shoot at the earliest end |
| 1029 | [Two City Scheduling](https://leetcode.com/problems/two-city-scheduling/) | Medium | Sort by the cost difference (exchange argument) |
| 135 | [Candy](https://leetcode.com/problems/candy/) | Hard | Two-sided neighbor constraint → two passes |
| 630 | [Course Schedule III](https://leetcode.com/problems/course-schedule-iii/) | Hard | Deadlines → sort + drop the longest (regret) |
| 871 | [Minimum Number of Refueling Stops](https://leetcode.com/problems/minimum-number-of-refueling-stops/) | Hard | Refuel retroactively from the best station passed |

## 9. Recognition drill
1. Given coin denominations [1, 5, 10, 25] and an amount, minimize the number of coins. Then: the same with arbitrary denominations.
2. Given a string, partition it into as many parts as possible so each letter appears in at most one part.
3. Each person has a weight; boats carry at most 2 people and a weight limit. Minimize the number of boats.
4. Given tasks with enqueue times and processing times, a single CPU always picks the shortest available task. Output the processing order.
5. Given houses in a row with values, you can't rob adjacent houses. Maximize the loot.

<details><summary>Answers</summary>

1. **US coins: greedy works** (a canonical coin system). **Arbitrary coins: DP** (Coin Change). Counterexample for greedy: [1,3,4], amount 6.
2. **Greedy**: extend the current part's end to the last occurrence of each letter seen; cut when i == end.
3. **Greedy + two pointers** on sorted weights: pair the heaviest with the lightest if they fit.
4. **Sort by enqueue time + min-heap by (duration, index)**: a simulation greedy.
5. **Not greedy → DP** (House Robber). Taking the local max can block better neighbors: [2, 3, 2] → greedy takes 3, optimal takes 2+2=4.
</details>

Next: Part 13 — Intervals
