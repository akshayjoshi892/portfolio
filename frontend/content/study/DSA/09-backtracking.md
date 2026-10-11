# Part 9 — Recursion & Backtracking

## 1. Core insight
Backtracking is **DFS over an implicit decision tree**. Each level makes one choice, each root-to-leaf path is a candidate, and valid leaves are answers. The template is always **choose → explore → un-choose**, with shared mutable state (path, used-sets) restored on the way back so siblings see a clean state. It's the right tool when the *output itself* is exponential ("all subsets / permutations / placements"), or when constraints are complex enough that you must search. Performance comes from **pruning**: cut a branch the moment it can't lead to a valid answer (sorted input + `break`, constraint sets, bounds).

## 2. Trigger signals / Anti-signals
**Triggers**
- "Return **all** subsets / permutations / combinations / partitions / valid strings".
- "Generate all valid parentheses / IP addresses / palindrome partitions".
- Constraint satisfaction: N-Queens, Sudoku, word search in a grid, placing items without conflicts.
- Small n (≤ 10 for permutations, ≤ 20 for subsets) with no obvious polynomial structure.
- "Find **one** valid configuration" in a hard search space (return on the first success).

**Anti-signals**
- Asked for the **count** or the **best** value, not the list → subproblems likely overlap → **DP** (memoize the recursion). E.g., "number of ways to reach target" (Target Sum), "min coins".
- Large n (≥ 30) with a count/optimize question → exponential search is dead; look for DP, greedy, or math.
- The choice at each step never needs revisiting → greedy.
- Only "does a subset exist with sum S" and sum is small → knapsack DP, not backtracking.

## 3. Mental model
**The decision tree.** Subsets of [1,2,3] with the "loop from start" formulation. Every node (not just leaves) is an answer:
```
                     []
         /           |          \
       [1]          [2]         [3]
      /   \          |
  [1,2]  [1,3]     [2,3]
    |
 [1,2,3]
```
Permutations: each level picks any unused element:
```
              []
     1/       2|       \3
    [1]       [2]       [3]
   2/ \3     1/ \3     1/ \2
 [1,2][1,3] [2,1][2,3] [3,1][3,2]
   |    |     |    |     |    |
  +3   +2    +3   +1    +2   +1
```
**Complexity rule:** (number of nodes) × (work per node). Subsets: 2ⁿ nodes × O(n) copy. Permutations: about e·n! nodes × O(n) copy at the leaves → O(n · n!).

## 4. Canonical templates

```python
# 4.1 Subsets: every node is an answer; `start` prevents reusing earlier elements
def subsets(nums: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []
    def dfs(start: int) -> None:
        res.append(path[:])                         # copy! path keeps mutating
        for i in range(start, len(nums)):
            path.append(nums[i])                    # choose
            dfs(i + 1)                              # explore (i+1: no reuse)
            path.pop()                              # un-choose
    dfs(0)
    return res

# 4.2 Combinations of size k, with a counting prune
def combine(n: int, k: int) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []
    def dfs(start: int) -> None:
        if len(path) == k:
            res.append(path[:])
            return
        need = k - len(path)
        for i in range(start, n - need + 2):        # prune: leave room for `need` numbers in [i..n]
            path.append(i)
            dfs(i + 1)
            path.pop()
    dfs(1)
    return res

# 4.3 Permutations: a `used` array instead of `start`
def permute(nums: list[int]) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []
    used = [False] * len(nums)
    def dfs() -> None:
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True; path.append(nums[i])
            dfs()
            used[i] = False; path.pop()
    dfs()
    return res

# 4.4 Permutations with duplicates: sort + skip a duplicate whose twin is NOT in use
def permute_unique(nums: list[int]) -> list[list[int]]:
    nums.sort()
    res: list[list[int]] = []
    path: list[int] = []
    used = [False] * len(nums)
    def dfs() -> None:
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue                            # equal values must be used in index order
            used[i] = True; path.append(nums[i])
            dfs()
            used[i] = False; path.pop()
    dfs()
    return res

# 4.5 Grid search (Word Search): mark visited in place, restore on return
def exist(board: list[list[str]], word: str) -> bool:
    R, C = len(board), len(board[0])
    def dfs(r: int, c: int, i: int) -> bool:
        if i == len(word):
            return True
        if not (0 <= r < R and 0 <= c < C) or board[r][c] != word[i]:
            return False
        board[r][c] = '#'                           # choose (mark visited)
        found = (dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1) or
                 dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1))
        board[r][c] = word[i]                       # un-choose
        return found
    return any(dfs(r, c, 0) for r in range(R) for c in range(C))
```
**Trace 4.4** [1,1,2]: at the root, i=1 (second 1) is skipped because used[0] is False. So we only start with index 0 or index 2 → [1,1,2], [1,2,1], [2,1,1] ✓ (3 = 3!/2!).

## 5. Worked examples

### Example 1 — Subsets II (LC 90) — Medium
**Problem:** all subsets of an array with duplicates, with no duplicate subsets.
**Brute force:** generate all 2ⁿ subsets, sort each, dedupe with a set of tuples → O(n · 2ⁿ log n) and wasteful.
**Aha (signal: "duplicates" + "unique results"):** sort, then at each tree level **don't start a branch with the same value twice**. The condition `i > start and nums[i] == nums[i−1]` skips duplicate siblings but still allows `[1,1]` (the second 1 is chosen at a deeper level, where i == start).
```python
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    nums.sort()
    res: list[list[int]] = []
    path: list[int] = []
    def dfs(start: int) -> None:
        res.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue                            # same value already tried at this level
            path.append(nums[i])
            dfs(i + 1)
            path.pop()
    dfs(0)
    return res
```
**Trace** [1,2,2]: [], [1], [1,2], [1,2,2], (skip the second [1,2]), [2], [2,2], (skip the second [2]) → 6 subsets ✓.
**Complexity:** O(n · 2ⁿ) time worst case (no duplicates), O(n) auxiliary space excluding the output.
**Edge cases:** all equal [2,2,2] → 4 subsets. Empty → [[]].

### Example 2 — Combination Sum (LC 39) — Medium
**Problem:** all unique combinations (unlimited reuse) of `candidates` that sum to `target`.
**Brute force thinking:** try every multiset. Unbounded, so it needs structure.
**Aha:** (1) To avoid permutations of the same combination, enforce **non-decreasing index order** (`start`), and pass `i` (not `i+1`) to allow reuse. (2) Sort, and **break** as soon as a candidate exceeds the remainder. All later ones are bigger, so this prunes whole subtrees.
```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    candidates.sort()
    res: list[list[int]] = []
    path: list[int] = []
    def dfs(start: int, remain: int) -> None:
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            c = candidates[i]
            if c > remain:
                break                               # sorted -> nothing further fits
            path.append(c)
            dfs(i, remain - c)                      # i: the same element can be reused
            path.pop()
    dfs(0, target)
    return res
```
**Trace** [2,3,6,7], target 7 → [2,2,3], [7] ✓.
**Complexity:** O(N^(T/m + 1)) worst case, where T = target and m = min candidate (tree depth ≤ T/m, branching ≤ N). Space O(T/m) for recursion.
**Contrast:** if the question were "**how many** combinations", that's Coin Change II, a DP (Ch 17). Counting doesn't need enumeration.

### Example 3 — N-Queens (LC 51) — Hard
**Problem:** all placements of n queens on an n×n board with no attacks.
**Brute force:** choose n cells out of n² → C(n², n) (≈ 4.4·10⁹ for n=8). Checking each is hopeless.
**Aha (constraint search + pruning):** exactly one queen per row, so recurse **row by row** (n choices per level instead of n²). Keep O(1) conflict checks with three sets: columns, diagonals `r − c`, anti-diagonals `r + c`.
```python
def solve_n_queens(n: int) -> list[list[str]]:
    res: list[list[str]] = []
    cols: set[int] = set(); diag: set[int] = set(); anti: set[int] = set()
    board = [['.'] * n for _ in range(n)]
    def dfs(r: int) -> None:
        if r == n:
            res.append([''.join(row) for row in board])
            return
        for c in range(n):
            if c in cols or (r - c) in diag or (r + c) in anti:
                continue                            # prune: attacked square
            cols.add(c); diag.add(r - c); anti.add(r + c); board[r][c] = 'Q'
            dfs(r + 1)
            cols.remove(c); diag.remove(r - c); anti.remove(r + c); board[r][c] = '.'
    dfs(0)
    return res
```
**Why `r − c` and `r + c`:** cells on the same "\" diagonal share r − c; cells on the same "/" diagonal share r + c.
**Complexity:** O(n!) time upper bound (row r has ≤ n − r free columns), plus O(n²) to build each solution. O(n) space for the sets and recursion (plus the board).
**Edge cases:** n=1 → [["Q"]]. n=2, 3 → [].
**Upgrade:** use bitmasks for cols/diag/anti (Ch 19) for N-Queens II counting. It's the same search, with faster constant factors.

## 6. Common mistakes & traps
- Appending `path` instead of `path[:]` → every result is the same (empty) list at the end.
- Forgetting to un-choose (pop / unmark / restore the board cell).
- Using `start = i` vs `i + 1` incorrectly (reuse allowed vs not).
- Dedupe condition errors. Subsets/combinations: `i > start and nums[i] == nums[i−1]`. Permutations: `i > 0 and nums[i] == nums[i−1] and not used[i−1]`. Both require **sorting first**.
- `continue` where `break` is valid after sorting (a lost pruning opportunity), or `break` where the input isn't sorted (wrong answers).
- Python recursion limit (1000) for deep recursions. Backtracking depth is usually ≤ n, which is fine.
- Mutating a shared set/list while iterating over it.
- Doing expensive validation only at the leaves instead of pruning at each step.

## 7. Connections
- **DP** (Ch 17): if the recursion's arguments repeat and you need a count or an optimum, memoize. Backtracking is the *brute force* of DP, and writing it first often reveals the DP state.
- **Bitmask** (Ch 19): subsets of n ≤ 20 can be enumerated iteratively with masks. Used-sets can be bitmasks (bitmask DP = memoized backtracking over a mask).
- **Trie** (Ch 18): Word Search II = backtracking on the grid + trie pruning (stop when the prefix is absent).
- **Graphs** (Ch 14): DFS with a visited set is backtracking where visited is *not* un-marked (you want each node once, not each path).
- **Palindrome partitioning**: backtracking + DP precomputation of `is_pal[i][j]`.
- Mistaken for: DP (Combination Sum *listing* vs Coin Change II *counting*); greedy.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 78 | [Subsets](https://leetcode.com/problems/subsets/) | Medium | "All subsets" → every node is an answer |
| 77 | [Combinations](https://leetcode.com/problems/combinations/) | Medium | Fixed size k → prune by remaining room |
| 46 | [Permutations](https://leetcode.com/problems/permutations/) | Medium | Order matters → used[] |
| 17 | [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) | Medium | Cartesian product of choices per position |
| 22 | [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) | Medium | Valid prefixes only: open < n, close < open |
| 39 | [Combination Sum](https://leetcode.com/problems/combination-sum/) | Medium | Unlimited reuse → pass i, sort + break |
| 40 | [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/) | Medium | Each used once + duplicates → sort, skip siblings |
| 90 | [Subsets II](https://leetcode.com/problems/subsets-ii/) | Medium | Duplicates → skip equal siblings |
| 47 | [Permutations II](https://leetcode.com/problems/permutations-ii/) | Medium | Duplicates in permutations → use equal values in order |
| 79 | [Word Search](https://leetcode.com/problems/word-search/) | Medium | Path in a grid without reuse → mark/unmark |
| 131 | [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/) | Medium | All partitions → choose the next cut |
| 93 | [Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/) | Medium | Four segments with validity rules → bounded depth |
| 698 | [Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/) | Medium | n ≤ 16 → fill buckets with pruning or bitmask DP |
| 51 | [N-Queens](https://leetcode.com/problems/n-queens/) | Hard | One per row + diagonal sets |
| 37 | [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) | Hard | Constraint search; pick the most constrained cell first |
| 212 | [Word Search II](https://leetcode.com/problems/word-search-ii/) | Hard | Many words on one grid → trie-guided DFS |

## 9. Recognition drill
1. Given a string with letters and digits, return all strings obtained by toggling the case of each letter.
2. Given n ≤ 15 matchsticks, decide whether they can form a square.
3. Return the number of distinct ways to assign + or − to each number so the total equals target (n ≤ 20, sum ≤ 1000).
4. Given a dictionary and a string, return **all** sentences formed by inserting spaces so every word is in the dictionary.
5. Find all combinations of k distinct numbers from 1–9 summing to n.

<details><summary>Answers</summary>

1. **Backtracking / iterative doubling**: two choices per letter → O(2^L · L).
2. **Backtracking with pruning** (sort descending, fill 4 sides, skip equal side values) or bitmask DP. n ≤ 15 invites exponential search.
3. **Count, not list → DP** (subset-sum transform: (sum+target)/2). Backtracking at 2²⁰ ≈ 10⁶ also passes, but DP is the intended answer.
4. **Backtracking + memo (DP of lists)**. Listing all sentences is output-exponential; memoize suffix results to avoid recomputation.
5. **Combinations with a sum constraint**: start index + prune when the remaining sum < 0 or the count is exceeded.
</details>

Next: Part 10 — Trees
