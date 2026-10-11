# Part 14 — Graphs: Representation, BFS/DFS, Components, Cycles, Topological Sort

## 1. Core insight
Many problems are graphs in disguise. Whenever there are **states** and **moves between states**, you have nodes and edges: cells and adjacency, words and one-letter edits, courses and prerequisites, lock combinations and turns. Once modeled, a small toolkit covers most interview questions. **BFS** explores in order of edge count, so it gives shortest paths in unweighted graphs. **DFS** explores deeply, giving reachability, components, cycle detection via the recursion stack, and postorder for topological order. **Kahn's algorithm** repeatedly removes nodes with no remaining prerequisites, which yields a topological order *and* detects cycles. Every one runs in O(V + E) because each node and edge is processed a constant number of times, guaranteed by a `visited` set.

## 2. Trigger signals / Anti-signals
**Triggers**
- Grid with "islands / regions / connected / flood fill / surrounded" → DFS/BFS over cells.
- "Minimum number of steps / moves / transformations", every move costs the same → **BFS**.
- "Spreads every minute from all X simultaneously" (rotting oranges, walls and gates, 0-1 matrix) → **multi-source BFS**.
- "Prerequisites", "dependencies", "build order", "alien dictionary", "can all be finished" → **topological sort**.
- "Is there a cycle" in a directed graph → 3-color DFS / Kahn. Undirected → Union-Find or DFS with a parent.
- "Two groups with no internal edges", "possible bipartition" → **BFS/DFS 2-coloring**.
- "Clone a graph", "all paths from source to target" → DFS with a map / backtracking.
- Implicit state graphs: lock combinations, sliding puzzle, (position, keys-held) → BFS over states.

**Anti-signals**
- Weighted edges (non-uniform cost) → Dijkstra / 0-1 BFS (Ch 16), not plain BFS.
- Dynamic edge additions with connectivity queries → Union-Find (Ch 15).
- Longest path in a general graph → NP-hard. In a **DAG** it's DP over the topological order.
- A grid DP where moves only go right/down → no cycles, so it's DP (Ch 17), not BFS.

## 3. Mental model
**BFS = ripples in a pond**: layer d contains exactly the nodes at distance d. **DFS = a maze explorer with chalk**: go as deep as possible, backtrack at dead ends, and chalk marks (visited) prevent loops.

```
Directed-cycle detection with 3 colors:
WHITE = unvisited, GRAY = on the current DFS path, BLACK = fully explored
edge u -> v with v GRAY  => back edge => CYCLE
edge u -> v with v BLACK => already finished, safe (cross/forward edge)

Kahn's algorithm:
indegree:  A:0  B:1  C:1  D:2        A -> B -> D
queue: [A]                            A -> C -> D
pop A -> B:0, C:0 -> queue [B, C]
pop B -> D:1; pop C -> D:0 -> queue [D]
order: A B C D   (if order has < V nodes -> a cycle exists)
```

## 4. Canonical templates

```python
from collections import defaultdict, deque

# 4.0 Build an adjacency list
def build_graph(n: int, edges: list[list[int]], directed: bool = False) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    return adj

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# 4.1 BFS shortest path (unweighted). Mark visited WHEN ENQUEUING.
def bfs_dist(adj: list[list[int]], src: int) -> list[int]:
    dist = [-1] * len(adj)
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:                 # doubles as `visited`
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

# 4.2 Multi-source BFS on a grid (distance to the nearest source)
def multi_source_bfs(grid: list[list[int]], sources: list[tuple[int, int]]) -> list[list[int]]:
    R, C = len(grid), len(grid[0])
    dist = [[-1] * C for _ in range(R)]
    q = deque()
    for r, c in sources:                       # all sources at distance 0
        dist[r][c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and dist[nr][nc] == -1 and grid[nr][nc] != -1:  # -1 = wall
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist

# 4.3 Iterative DFS (avoids Python's recursion limit)
def count_components(n: int, adj: list[list[int]]) -> int:
    seen = [False] * n
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        seen[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps

# 4.4 Directed cycle detection (3 colors)
def has_cycle_directed(n: int, adj: list[list[int]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:              # back edge to the current path
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False
    return any(color[u] == WHITE and dfs(u) for u in range(n))

# 4.5 Topological sort (Kahn). Returns [] if there is a cycle.
def topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int]:   # (a, b): a before b
    adj: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    q = deque(u for u in range(n) if indeg[u] == 0)
    order: list[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []

# 4.6 Bipartite check (2-coloring with BFS; handles disconnected graphs)
def is_bipartite(adj: list[list[int]]) -> bool:
    color = [-1] * len(adj)
    for s in range(len(adj)):
        if color[s] != -1:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    q.append(v)
                elif color[v] == color[u]:    # odd cycle
                    return False
    return True
```
**Trace 4.5** n=4, edges (0,1),(0,2),(1,3),(2,3): indeg [0,1,1,2]. q=[0] → order 0, indeg 1,2 → 0 → q=[1,2] → pop 1: indeg[3]=1 → pop 2: indeg[3]=0 → pop 3. Order [0,1,2,3] ✓.

## 5. Worked examples

### Example 1 — Number of Islands (LC 200) — Medium
**Brute force framing:** "for each land cell, is it connected to an already counted island?" → needs connectivity → graph.
**Aha (signal: grid + connected regions):** each unvisited land cell starts a new component. Flood-fill it (sink it) so it's never counted again.
```python
def num_islands(grid: list[list[str]]) -> int:
    R, C = len(grid), len(grid[0])
    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] != '1':
                continue
            count += 1
            grid[r][c] = '0'                          # mark visited by sinking
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == '1':
                        grid[nr][nc] = '0'
                        stack.append((nr, nc))
    return count
```
**Complexity:** O(R·C) time (each cell is sunk once), O(R·C) worst-case stack.
**Edge cases:** all water, all land, a single row or column. If mutating the input isn't allowed, use a `visited` set (ask the interviewer!).

### Example 2 — Course Schedule II (LC 210) — Medium
**Problem:** return an order to take all courses given `[a, b]` = "b before a", or [] if impossible.
**Brute force:** try permutations → O(n!).
**Aha (signals: "prerequisites" + "order"):** a topological order of a DAG. Kahn's algorithm produces it, and fewer than n processed nodes means a cycle.
```python
from collections import deque

def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    adj: list[list[int]] = [[] for _ in range(num_courses)]
    indeg = [0] * num_courses
    for a, b in prerequisites:                   # edge b -> a
        adj[b].append(a)
        indeg[a] += 1
    q = deque(i for i in range(num_courses) if indeg[i] == 0)
    order: list[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == num_courses else []
```
**Complexity:** O(V + E) time and space.
**Edge cases:** no prerequisites (any order), a self-loop `[0,0]` (a cycle → []), disconnected courses. **Edge direction** is the #1 bug. Say it out loud: "[a, b] means edge b→a."

### Example 3 — Word Ladder (LC 127) — Hard
**Problem:** shortest transformation length from `begin` to `end`, changing one letter at a time, where every intermediate word must be in the list.
**Graph model:** nodes = words, edges = differ by one letter. The question is the shortest path in an unweighted graph → **BFS**.
**Bottleneck:** building edges by comparing all pairs → O(N²·L) with N = 5000.
**Aha:** generate neighbors through **wildcard patterns** (`h*t` connects hot, hit, hat). Each word has L patterns, and words sharing a pattern are neighbors. Clearing a pattern bucket after its first expansion prevents re-scanning it.
```python
from collections import defaultdict, deque

def ladder_length(begin: str, end: str, word_list: list[str]) -> int:
    words = set(word_list)
    if end not in words:
        return 0
    buckets: dict[str, list[str]] = defaultdict(list)
    for w in words:
        for i in range(len(w)):
            buckets[w[:i] + '*' + w[i + 1:]].append(w)
    q = deque([(begin, 1)])
    seen = {begin}
    while q:
        w, d = q.popleft()
        if w == end:
            return d
        for i in range(len(w)):
            p = w[:i] + '*' + w[i + 1:]
            for nb in buckets[p]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
            buckets[p] = []                     # every word in this bucket is now queued
    return 0
```
**Trace** hit → hot → dot → dog → cog = 5 ✓ (standard example).
**Complexity:** O(N·L²) time (N·L patterns, each built by O(L) slicing), O(N·L²) space for the buckets.
**Upgrade:** **bidirectional BFS** (expand the smaller frontier from both ends) cuts the explored states dramatically. Mention it as a follow-up.

## 6. Common mistakes & traps
- Marking visited on **dequeue** instead of on **enqueue** in BFS → the same node is enqueued many times (exponential blowup on grids).
- Edge direction reversed in prerequisite problems.
- Forgetting disconnected components (loop over all nodes as start points).
- Recursive DFS on a 10⁵-node graph or a 1000×1000 grid → RecursionError. Use an iterative stack.
- Undirected cycle detection using the 3-color method without skipping the parent edge (every undirected edge looks like a back edge).
- Directed cycle detection with a single `visited` set (it can't distinguish "on the current path" from "done").
- Using BFS on weighted graphs.
- Grid bounds checks in the wrong order (index before bounds check).

## 7. Connections
- **Union-Find** (Ch 15): an alternative for components, undirected cycles, and online connectivity.
- **Shortest paths** (Ch 16): BFS → 0-1 BFS → Dijkstra as the edge weights generalize.
- **DP on DAGs** (Ch 17): longest path / number of paths = DP in topological order. Longest Increasing Path in a Matrix = DFS + memo on an implicit DAG.
- **Backtracking** (Ch 9): all paths from source to target, Word Search.
- **Trees** (Ch 10): all nodes distance K = tree → graph + BFS.
- **Binary search on answer** + BFS check: Swim in Rising Water, Path With Minimum Effort.
- Mistaken for: DP (grid problems where movement goes in all 4 directions have cycles, so they need BFS/Dijkstra, not DP).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 1971 | [Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/) | Easy | Reachability → BFS/DFS/UF |
| 733 | [Flood Fill](https://leetcode.com/problems/flood-fill/) | Easy | Connected same-colored region |
| 200 | [Number of Islands](https://leetcode.com/problems/number-of-islands/) | Medium | Count grid components |
| 695 | [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) | Medium | Component size |
| 133 | [Clone Graph](https://leetcode.com/problems/clone-graph/) | Medium | Traverse + old→new map |
| 994 | [Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) | Medium | Simultaneous spread → multi-source BFS |
| 542 | [01 Matrix](https://leetcode.com/problems/01-matrix/) | Medium | Distance to the nearest 0 → multi-source BFS from all 0s |
| 417 | [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/) | Medium | Reverse the flow: search uphill from each ocean |
| 130 | [Surrounded Regions](https://leetcode.com/problems/surrounded-regions/) | Medium | Mark border-connected cells first (complement) |
| 207 | [Course Schedule](https://leetcode.com/problems/course-schedule/) | Medium | Prerequisites feasible = no directed cycle |
| 210 | [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) | Medium | Return the order → Kahn |
| 785 | [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/) | Medium | Two groups → 2-coloring |
| 752 | [Open the Lock](https://leetcode.com/problems/open-the-lock/) | Medium | Implicit state graph, min moves → BFS |
| 802 | [Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/) | Medium | Nodes not reaching a cycle → 3-color or reverse Kahn |
| 310 | [Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/) | Medium | Peel leaves layer by layer → centers |
| 269 | [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) (Premium) | Hard | Order constraints from adjacent words → topo sort |
| 127 | [Word Ladder](https://leetcode.com/problems/word-ladder/) | Hard | Min transformations → BFS with wildcard buckets |
| 329 | [Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) | Hard | Strictly increasing edges = a DAG → DFS + memo |

## 9. Recognition drill
1. A grid has 0 = empty, 1 = building, 2 = obstacle. Find the empty cell minimizing the total walking distance to all buildings.
2. Given `n` tasks and pairs "a must finish before b", each taking 1 month, with unlimited parallelism, find the minimum number of months.
3. Given a directed graph, return all nodes from which every path eventually ends at a terminal node.
4. Given an equation list like `a / b = 2.0`, answer queries `x / y`.
5. On a grid, find the shortest path from the top-left to the bottom-right if you may remove at most k obstacles.

<details><summary>Answers</summary>

1. **BFS from each building** (accumulating distance and reach counts per empty cell). Multiple BFS runs, because the objective is a sum, not a min.
2. **Topological sort by levels** (Kahn's, counting BFS layers) = the longest path in the DAG.
3. **Reverse graph + Kahn** from the terminal nodes, or 3-color DFS (safe = BLACK without reaching GRAY) (LC 802).
4. **Weighted graph + DFS/BFS** multiplying ratios along the path (or weighted Union-Find) (LC 399).
5. **BFS over states (r, c, remaining_k)** (LC 1293). The state includes the resource, which is the key modeling step.
</details>

Next: Part 15 — Union-Find
