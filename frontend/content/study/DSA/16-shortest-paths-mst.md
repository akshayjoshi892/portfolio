# Part 16 — Shortest Paths (BFS, Dijkstra, 0-1 BFS, Bellman-Ford) & MST (Kruskal, Prim)

## 1. Core insight
All shortest-path algorithms repeatedly **relax** edges (`dist[v] = min(dist[v], dist[u] + w)`). They differ in the *order* of relaxation, which depends on what the weights allow. **Unweighted**: BFS order is distance order, O(V+E). **Non-negative weights**: Dijkstra. The closest unsettled node can't be improved later (any detour adds ≥ 0), so settling it greedily is safe. A heap gives O(E log V). **0/1 weights**: 0-1 BFS, a deque replacing the heap. **Negative weights or a hop limit**: Bellman-Ford. After round i, `dist` is optimal over paths with ≤ i edges, O(V·E). **MST** is a different objective: connect everything with minimum *total* weight. It rests on the **cut property**: the lightest edge crossing any cut belongs to some MST. Kruskal applies it globally (sorted edges + Union-Find). Prim grows one tree (the lightest edge leaving it).

## 2. Trigger signals / Anti-signals
**Triggers**
- "Minimum cost / time / distance from source", weighted, non-negative → **Dijkstra**.
- "Network delay", "time for a signal to reach all nodes" → Dijkstra, then take the max of dist.
- Weights only 0 and 1 ("free moves vs costly moves", "remove obstacle costs 1") → **0-1 BFS**.
- "At most k stops / edges" → **Bellman-Ford with k+1 rounds** (or BFS/Dijkstra over (node, stops) states).
- Negative edge weights, "detect a negative cycle" → Bellman-Ford.
- All-pairs shortest paths, n ≤ 400 → **Floyd-Warshall** O(n³).
- "Minimize the maximum edge weight on the path" (effort, water level) → Dijkstra with `max` instead of `+`, or binary search + BFS, or Kruskal-style UF.
- "Maximum probability path" → Dijkstra with a max-heap on the product (probabilities ≤ 1 act like non-negative weights).
- "Connect all points/cities with minimum total cost" → **MST**.

**Anti-signals**
- Unit weights → Dijkstra works but BFS is simpler and O(V+E). Say that.
- Negative weights → **Dijkstra is wrong** (a settled node can later be improved).
- Hop limit + weights → vanilla Dijkstra is wrong (the cheapest path may use too many stops). The state must include stops.
- "Shortest path that visits all nodes" → not Dijkstra. That's BFS over (node, visited_mask) (bitmask BFS/DP).
- MST vs shortest-path tree: an MST minimizes the *total* edge weight, not the distance from a source. They're different trees.

## 3. Mental model
**Dijkstra = a fire spreading along fuses of different lengths.** The next node to catch fire is always the one with the smallest arrival time. Once it's on fire, its time is final.
```
      A --1-- B
      |       |
      4       1          pop order: A(0) → B(1) → D(2) → C(3)
      |       |          C's tentative 4 (direct) is improved to 3 via D
      C --1-- D          before C is popped, so the pop value is final
```
**Bellman-Ford = rounds of gossip.** In round i, everyone updates using their neighbors' round-(i−1) knowledge, so information travels one hop per round.

**MST cut property:** split the nodes into two sides (any split). The cheapest edge across the split is safe to include.

## 4. Canonical templates

```python
import heapq
from collections import deque

INF = float('inf')

# 4.1 Dijkstra (lazy deletion). adj[u] = [(v, w), ...], w >= 0
def dijkstra(adj: list[list[tuple[int, int]]], src: int) -> list[float]:
    dist = [INF] * len(adj)
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                        # stale entry: u was already settled with a smaller d
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# 4.2 0-1 BFS: weight-0 edges go to the front, weight-1 edges to the back
def zero_one_bfs(adj: list[list[tuple[int, int]]], src: int) -> list[float]:
    dist = [INF] * len(adj)
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist

# 4.3 Bellman-Ford limited to at most k edges (copy per round!)
def bellman_ford_k(n: int, edges: list[tuple[int, int, int]], src: int, k: int) -> list[float]:
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k):                      # round i: paths with <= i edges
        new = dist[:]                       # read only the previous round's values
        for u, v, w in edges:
            if dist[u] + w < new[v]:
                new[v] = dist[u] + w
        dist = new
    return dist
# Negative cycle check: run n-1 rounds; if round n still relaxes something -> a negative cycle.

# 4.4 Floyd-Warshall (all pairs): the k loop must be OUTERMOST
def floyd_warshall(n: int, edges: list[tuple[int, int, int]]) -> list[list[float]]:
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        d[u][v] = min(d[u][v], w)
    for k in range(n):                      # allow k as an intermediate node
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d

# 4.5 Kruskal (uses DSU from Ch 15). edges = [(w, u, v)]
def kruskal(n: int, edges: list[tuple[int, int, int]]) -> float:
    dsu = DSU(n)
    total = used = 0
    for w, u, v in sorted(edges):
        if dsu.union(u, v):                 # the lightest edge joining two components
            total += w
            used += 1
            if used == n - 1:
                break
    return total if used == n - 1 else INF  # INF: graph disconnected

# 4.6 Prim with a heap (sparse graphs): O(E log V)
def prim(adj: list[list[tuple[int, int]]]) -> float:
    n = len(adj)
    seen = [False] * n
    pq = [(0, 0)]                           # (edge weight into the tree, node)
    total = 0
    while pq:
        w, u = heapq.heappop(pq)
        if seen[u]:
            continue
        seen[u] = True
        total += w
        for v, wv in adj[u]:
            if not seen[v]:
                heapq.heappush(pq, (wv, v))
    return total if all(seen) else INF
```
**Trace 4.2** grid edges "follow the arrow = 0, change direction = 1": nodes reached at cost 0 are processed before any cost-1 node. The deque stays sorted as [d, d, …, d+1, d+1], which is exactly Dijkstra's order.

## 5. Worked examples

### Example 1 — Network Delay Time (LC 743) — Medium
**Problem:** a signal starts at k and travels directed weighted edges. When does every node receive it? (−1 if never.)
**Brute force:** enumerate paths (exponential), or Bellman-Ford O(V·E) → works but is slower.
**Aha (signals: single source + non-negative weights + "time to reach"):** Dijkstra, then the answer = the max of dist.
```python
import heapq
from collections import defaultdict

def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))
    dist: dict[int, int] = {}
    pq = [(0, k)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in dist:
            continue                          # already settled
        dist[u] = d                           # first pop = final distance
        for v, w in adj[u]:
            if v not in dist:
                heapq.heappush(pq, (d + w, v))
    return max(dist.values()) if len(dist) == n else -1
```
This is the "settle on first pop" variant, equivalent to the lazy-deletion template.
**Complexity:** O(E log E) time (each edge pushes at most once), O(V + E) space.
**Edge cases:** unreachable nodes → −1. A single node → 0. Parallel edges (fine).

### Example 2 — Cheapest Flights Within K Stops (LC 787) — Medium
**Problem:** the cheapest price from src to dst with at most k stops (i.e., ≤ k+1 flights).
**Naive Dijkstra (wrong):** it settles each city at its cheapest cost regardless of stops. A cheap path with too many stops can "block" a pricier path with fewer stops that is needed to continue:
```
src -1-> A -1-> B -1-> dst        (3 flights, cost 3)
src ------5-----> B               (B via direct: 5, 1 flight)
k = 1 (≤ 2 flights): answer = src->B (5) -> dst (1) = 6.
Dijkstra settles B at cost 2 (via A, 2 flights) and reaches dst at 3 with 3 flights (invalid);
if it rejects that, B is already settled, so the 1-flight route to B is never retried -> fails.
```
**Aha (signal: "at most k edges"):** Bellman-Ford restricted to k+1 rounds, where round i = best cost using ≤ i flights. **Copy the array each round** so one round can't chain several edges.
```python
def find_cheapest_price(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):                    # at most k+1 flights
        new = dist[:]
        for u, v, w in flights:
            if dist[u] != INF and dist[u] + w < new[v]:
                new[v] = dist[u] + w
        dist = new
    return -1 if dist[dst] == INF else dist[dst]
```
**Trace** (the example above, with k=1): round 1: A=1, B=5. Round 2: B=min(5, A+1=2)=2, dst = B_prev(5)+1 = 6. Answer 6 ✓. (Without the copy, round 1 could already chain src→A→B→… and break the hop limit.)
**Complexity:** O(k · E) time, O(n) space.
**Alternative:** Dijkstra/BFS over states (city, flights_used), pruning when the flight count exceeds k+1.

### Example 3 — Min Cost to Connect All Points (LC 1584) — Medium
**Problem:** points in the plane. The cost between two points is their Manhattan distance. Minimum cost to connect all (n ≤ 1000).
**Aha (signals: "connect all" + "minimum total cost"):** MST on a **complete graph** with about n²/2 = 5·10⁵ edges.
- Kruskal: sort 5·10⁵ edges → O(n² log n). OK.
- **Prim with an array** (no heap): O(n²), the best choice for dense graphs. Keep `best[v]` = the cheapest edge from the tree to v, and pick the min each round.
```python
def min_cost_connect_points(points: list[list[int]]) -> int:
    n = len(points)
    in_tree = [False] * n
    best = [float('inf')] * n
    best[0] = 0
    total = 0
    for _ in range(n):
        u = -1
        for i in range(n):                    # the cheapest node to attach next: O(n)
            if not in_tree[i] and (u == -1 or best[i] < best[u]):
                u = i
        in_tree[u] = True
        total += best[u]
        ux, uy = points[u]
        for v in range(n):                    # update attachment costs via u
            if not in_tree[v]:
                d = abs(ux - points[v][0]) + abs(uy - points[v][1])
                if d < best[v]:
                    best[v] = d
    return total
```
**Trace** [[0,0],[2,2],[3,10],[5,2],[7,0]] → 20 ✓ (the LC example).
**Complexity:** O(n²) time, O(n) space. It never materializes the edges.
**Edge cases:** n = 1 → 0. Duplicate points (distance 0).

## 6. Common mistakes & traps
- Dijkstra without the stale check (`if d > dist[u]: continue`) → still correct but slower. Missing it *and* relaxing from stale entries with the wrong `d` → wrong.
- Using Dijkstra with negative edges.
- Bellman-Ford with k limits **without copying** `dist` per round.
- Floyd-Warshall with the `k` loop not outermost.
- Marking visited on push (BFS-style) in Dijkstra → wrong. Nodes are finalized on **pop**.
- 0-1 BFS: pushing all nodes to the back → becomes a plain (wrong) BFS.
- MST on a disconnected graph → check that you used n−1 edges.
- Confusing an MST with the shortest-path tree.
- Heap entries with equal distances and uncomparable payloads → include an int tiebreaker.

## 7. Connections
- **BFS** (Ch 14) is Dijkstra with unit weights. **0-1 BFS** is the in-between.
- **Heap** (Ch 11) powers Dijkstra and Prim.
- **Union-Find** (Ch 15) powers Kruskal, plus "minimax path" via sorted edges.
- **Binary search on answer** (Ch 6): Path With Minimum Effort / Swim in Rising Water have BS + BFS alternatives.
- **DP**: Bellman-Ford is DP over (edges used, node). Shortest paths in a DAG = DP in topological order. Floyd-Warshall = DP over allowed intermediate nodes.
- **State-space search**: Dijkstra/BFS over (node, extra_state): stops used, keys held, obstacles removed, visited mask.

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 1091 | [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/) | Medium | Unit weights, 8 directions → BFS |
| 743 | [Network Delay Time](https://leetcode.com/problems/network-delay-time/) | Medium | Single source, non-negative → Dijkstra |
| 1514 | [Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/) | Medium | Multiplicative weights ≤ 1 → max-heap Dijkstra |
| 1631 | [Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) | Medium | Minimize the max step → minimax Dijkstra |
| 787 | [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) | Medium | Hop limit → Bellman-Ford rounds |
| 1584 | [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/) | Medium | Connect all, dense → Prim O(n²) |
| 1976 | [Number of Ways to Arrive at Destination](https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/) | Medium | Count shortest paths → Dijkstra + a ways array |
| 2290 | [Minimum Obstacle Removal to Reach Corner](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/) | Hard | Costs 0/1 → 0-1 BFS |
| 1368 | [Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/) | Hard | Follow the arrow = 0, change = 1 → 0-1 BFS |
| 778 | [Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/) | Hard | Minimax path → Dijkstra on max / UF / BS |
| 1293 | [Shortest Path in a Grid with Obstacles Elimination](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/) | Hard | BFS over (cell, removals left) |
| 1489 | [Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/) | Hard | Kruskal with an edge excluded / forced |
| 847 | [Shortest Path Visiting All Nodes](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) | Hard | Visit all, n ≤ 12 → BFS over (node, mask) |

## 9. Recognition drill
1. A city's roads have travel times. Find the shortest travel time from A to B if you may make at most one road's time zero (a coupon).
2. A grid where moving to an equal-height cell is free and to a different height costs 1. Min cost from the corner to the corner.
3. n ≤ 100 cities; answer 10⁴ queries of the shortest distance between any two.
4. Airline routes with prices, some possibly negative (rebates), no negative cycles. Cheapest route.
5. Lay cable to connect all offices; some links are pre-built (free). Minimum cost.

<details><summary>Answers</summary>

1. **Dijkstra on a layered state graph (node, coupon_used ∈ {0,1})**: a free edge moves you from layer 0 to layer 1.
2. **0-1 BFS** (weights in {0,1}).
3. **Floyd-Warshall** once (O(n³) = 10⁶), then O(1) queries.
4. **Bellman-Ford** (negative weights → no Dijkstra). SPFA works in practice.
5. **Kruskal with the pre-built edges unioned first** (weight 0), then the usual sorted edges.
</details>

Next: Part 17 — Dynamic Programming
