# Part 15 — Union-Find (Disjoint Set Union)

## 1. Core insight
Union-Find maintains a partition of elements into groups under two operations. `union(a, b)` merges two groups. `find(x)` returns the group's representative (root). It answers "are a and b connected?" **as edges arrive**, which BFS/DFS can't do without re-traversing. Each group is a tree of parent pointers. **Path compression** flattens trees during `find`, and **union by size/rank** keeps them shallow. Together they give O(α(n)) amortized per operation, where α is the inverse Ackermann function (≤ 4 for any practical n). A union that finds both elements already in the same group means **this edge closes a cycle**.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Connected components" when edges are given as a list (especially incrementally, or with queries interleaved).
- "Redundant connection", "does adding this edge create a cycle", "is this graph a valid tree".
- Equivalence relations: "accounts with a shared email", "similar strings", "a == b constraints", "synonyms".
- "Number of islands II" (cells turned into land over time), "minimum edges to connect".
- **Kruskal's MST** (Ch 16): sort edges and union if it's not a cycle.
- Offline tricks: "process deletions in reverse as unions", "sort edges/queries by threshold and union progressively".

**Anti-signals**
- Need the actual **path** or **distance** between nodes → BFS/DFS/Dijkstra. UF only knows "same group or not".
- Edge **deletions** online → UF can't split. Reverse time if the problem is offline.
- **Directed** reachability / cycles → UF is for undirected connectivity. Use DFS/Kahn.
- A one-time component count on an adjacency list → DFS is equally good. UF shines with edge lists and streaming.

## 3. Mental model
Clubs with presidents. Each person points to someone in their club, and following the pointers leads to the president (the root). Merging clubs: the smaller club's president now reports to the bigger club's president. Path compression: after you walk up to find the president, you (and everyone on the way) remember a shortcut closer to the top.

```
before find(4):       after find(4) with path halving:
    1                        1
    |                      /   \
    2                     2     3
    |                           |
    3                           4
    |
    4
```

## 4. Canonical templates

```python
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n                                   # number of components

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving (iterative compression)
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False                                 # already connected: this edge is a cycle
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra                             # attach the smaller tree under the larger
        self.size[ra] += self.size[rb]
        self.count -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


# Keys that aren't 0..n-1 (strings, coordinates): a dict-based DSU
class DictDSU:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra                             # (add union by size if needed)
        return True

# Grid cell (r, c) -> integer id: r * C + c
```
**Trace** DSU(4): union(0,1) → parent[1]=0, count 3. union(2,3) → parent[3]=2, count 2. union(1,3): find(1)=0, find(3)=2, equal sizes → parent[2]=0, count 1. union(0,3) → both root 0 → False (cycle) ✓.

## 5. Worked examples

### Example 1 — Number of Provinces (LC 547) — Medium
**Problem:** count the connected components given an n×n adjacency matrix.
**DFS works** (O(n²)). **UF framing:** start with n components and each successful union reduces the count by 1.
```python
def find_circle_num(is_connected: list[list[int]]) -> int:
    n = len(is_connected)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):                  # upper triangle: the matrix is symmetric
            if is_connected[i][j]:
                dsu.union(i, j)
    return dsu.count
```
**Complexity:** O(n² · α(n)) time (reading the matrix dominates), O(n) space.
**Edge cases:** n = 1. Fully connected. Identity matrix (n provinces).

### Example 2 — Redundant Connection (LC 684) — Medium
**Problem:** a tree with one extra edge added. Return the edge that can be removed (the last one in the input if there are several).
**Brute force:** for each edge, remove it and check whether the rest is a tree (DFS) → O(n²).
**Aha (signal: "which edge creates a cycle", undirected):** process edges in order. The first edge whose endpoints are **already connected** closes the cycle. Since exactly one extra edge exists, it's the last edge of the cycle in input order.
```python
def find_redundant_connection(edges: list[list[int]]) -> list[int]:
    dsu = DSU(len(edges) + 1)                       # nodes are labeled 1..n
    for u, v in edges:
        if not dsu.union(u, v):
            return [u, v]
    return []
```
**Complexity:** O(n · α(n)) time, O(n) space.
**Edge cases:** 1-indexed nodes (allocate n+1). The directed version (LC 685) is much harder, because a node can have two parents. Don't confuse them.

### Example 3 — Accounts Merge (LC 721) — Medium (hard modeling)
**Problem:** accounts `[name, email1, email2, …]`. Two accounts belong to the same person if they share any email. Merge them, and return the emails sorted.
**Brute force:** compare every pair of accounts for a shared email, then find components → O(A² · E).
**Aha (signal: "same if they share X" = equivalence, transitively):** union accounts through their emails. Map each email to the first account that owned it, and union any later account containing that email with the owner.
```python
from collections import defaultdict

def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    dsu = DSU(len(accounts))
    owner: dict[str, int] = {}                          # email -> first account index
    for i, acc in enumerate(accounts):
        for email in acc[1:]:
            if email in owner:
                dsu.union(i, owner[email])
            else:
                owner[email] = i
    groups: dict[int, list[str]] = defaultdict(list)
    for email, i in owner.items():
        groups[dsu.find(i)].append(email)
    return [[accounts[root][0]] + sorted(emails) for root, emails in groups.items()]
```
**Complexity:** O(E log E) time for sorting emails (the unions are near-linear), O(E) space, where E = total emails.
**Edge cases:** the same name but different people (not merged unless they share an email). Duplicate emails within one account.

## 6. Common mistakes & traps
- Comparing `parent[a] == parent[b]` instead of `find(a) == find(b)`.
- Unioning the elements instead of their roots (`parent[b] = a` without `find`) → breaks the trees.
- No path compression and no union by size → O(n) per find on chains.
- Recursive `find` with path compression on 10⁵ elements in Python → recursion depth issues. Use the iterative halving above.
- Off-by-one with 1-indexed nodes.
- Counting components by `len(set(parent))` → wrong without calling `find` on each element first. Maintain `count`.
- Applying UF to directed graphs.

## 7. Connections
- **Graphs** (Ch 14): an alternative to DFS for components and undirected cycles. Valid tree ⇔ n−1 edges and no union fails.
- **MST — Kruskal** (Ch 16): sort edges by weight and union if they're in different components.
- **Offline processing**: sort queries by a threshold and add edges progressively (LC 1697 Checking Existence of Edge Length Limited Paths). Reverse deletions into unions.
- **Binary search / Dijkstra alternative**: Swim in Rising Water = add cells in increasing height order until start and end connect.
- **Weighted Union-Find**: track the ratio to the parent (Evaluate Division, LC 399).
- **Hashing**: DictDSU for strings/coordinates; Most Stones Removed (union rows with columns).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 1971 | [Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/) | Easy | Connectivity query → find(u) == find(v) |
| 547 | [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) | Medium | Components from an adjacency matrix |
| 323 | [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) (Premium) | Medium | Components from an edge list |
| 261 | [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/) (Premium) | Medium | Tree ⇔ n−1 edges and no cycle |
| 684 | [Redundant Connection](https://leetcode.com/problems/redundant-connection/) | Medium | The edge that closes a cycle |
| 990 | [Satisfiability of Equality Equations](https://leetcode.com/problems/satisfiability-of-equality-equations/) | Medium | Union all ==, then check the != pairs |
| 721 | [Accounts Merge](https://leetcode.com/problems/accounts-merge/) | Medium | Shared email = equivalence |
| 1319 | [Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/) | Medium | Components − 1, if there are enough edges |
| 947 | [Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/) | Medium | Stones − components; union rows with columns |
| 1202 | [Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/) | Medium | Swaps are transitive → sort within components |
| 399 | [Evaluate Division](https://leetcode.com/problems/evaluate-division/) | Medium | Ratios → weighted UF (or graph DFS) |
| 778 | [Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/) | Hard | Add cells by height until the corners connect |
| 1579 | [Remove Max Number of Edges to Keep Graph Fully Traversable](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/) | Hard | Shared edges first, then two DSUs |
| 952 | [Largest Component Size by Common Factor](https://leetcode.com/problems/largest-component-size-by-common-factor/) | Hard | Union each number with its prime factors |

## 9. Recognition drill
1. Cells of a grid become land one by one. After each addition, report the number of islands.
2. Given edges with weights and queries (u, v, limit), answer whether a path exists using only edges with weight < limit.
3. Given a list of synonym pairs and a sentence, generate all equivalent sentences.
4. Given n computers and a cable list, determine the minimum cable moves to connect all, or −1.
5. Given a directed graph where one extra edge was added to a rooted tree, find the edge to remove.

<details><summary>Answers</summary>

1. **Online Union-Find** on cell ids (LC 305): each new land cell adds a component, and unions with its land neighbors decrement the count.
2. **Offline**: sort the edges and the queries by limit, add edges with weight < limit progressively, and answer with `find` (LC 1697).
3. **Union-Find (or DFS) to group synonyms** + backtracking over word choices within each group (LC 1258).
4. **Union-Find**: needs ≥ n−1 cables; answer = components − 1.
5. **Not plain UF: a directed tree** (LC 685). Handle the node-with-two-parents case, then use UF for cycle detection on the remaining edges.
</details>

Next: Part 16 — Shortest Paths & MST
