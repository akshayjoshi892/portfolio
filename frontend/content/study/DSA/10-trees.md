# Part 10 — Trees

## 1. Core insight
A tree is a recursive structure: a node plus two smaller trees. So almost every tree problem is answered by one question: **"If I magically had the answer for my left and right subtrees, how would I compute mine?"** That gives a **bottom-up (postorder)** recursion that returns information to the parent. If instead the node needs information *from above* (a path so far, bounds, depth), pass it **top-down (preorder)** as a parameter. BFS covers anything about **levels**. A BST adds the invariant that the inorder traversal is sorted, and that every node constrains a *range* of valid values for its subtree.

## 2. Trigger signals / Anti-signals
**Triggers**
- Height, depth, diameter, balanced, "max path sum", subtree sums/properties → **postorder, return a value to the parent**, possibly updating a global best.
- Root-to-leaf paths, "path sum from root", "good nodes" (compare with the max so far) → **preorder, pass state down**.
- "Level order", "right side view", "zigzag", "minimum depth", "average per level", "width" → **BFS by levels**.
- BST: "k-th smallest", "validate", "range sum", "successor", "convert sorted array" → inorder = sorted, or bounds.
- "Lowest common ancestor" → postorder "found p or q" signal (binary tree) or a split-point walk (BST).
- "Serialize / construct from traversals" → preorder + null markers; preorder + inorder with an index map.
- "Distance k from a node", "burn the tree" → convert to a graph (parent pointers), then BFS.

**Anti-signals**
- Very deep trees (skewed, n = 10⁵) in Python → recursion depth limit. Use an iterative approach or `sys.setrecursionlimit` (and say so).
- A "tree" given as edges with no root, possibly with cycles → it's a graph problem (Ch 14).
- When a path can go **up and down** (any node to any node), a pure top-down pass fails. You need a bottom-up "best downward path" + a combine step at each node.

## 3. Mental model
**Each node is a manager.** It asks each child for a report (a return value), combines the reports with its own value, possibly updates a company-wide record (global), and sends a report upward. The key design decision: **what does a node report to its parent?** It's often *different* from what the problem asks. For max path sum, the node reports "the best path going *down* from me", while the global tracks "the best path *bending* at me".

```
        1                 traversal orders:
       / \                preorder  (N L R): 1 2 4 5 3
      2   3               inorder   (L N R): 4 2 5 1 3
     / \                  postorder (L R N): 4 5 2 3 1
    4   5                 level     (BFS)  : 1 | 2 3 | 4 5
```

## 4. Canonical templates

```python
from collections import deque
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: 'Optional[TreeNode]' = None,
                 right: 'Optional[TreeNode]' = None):
        self.val, self.left, self.right = val, left, right

# 4.1 Bottom-up: return info to the parent (height), update a global (diameter)
def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    best = 0
    def height(node: Optional[TreeNode]) -> int:
        nonlocal best
        if not node:
            return 0
        l, r = height(node.left), height(node.right)
        best = max(best, l + r)          # the longest path bending at node (in edges)
        return 1 + max(l, r)             # report: the longest downward chain (in nodes)
    height(root)
    return best

# 4.2 Top-down: pass state down (count nodes that are >= every ancestor)
def good_nodes(root: TreeNode) -> int:
    def dfs(node: Optional[TreeNode], max_so_far: int) -> int:
        if not node:
            return 0
        good = 1 if node.val >= max_so_far else 0
        m = max(max_so_far, node.val)
        return good + dfs(node.left, m) + dfs(node.right, m)
    return dfs(root, root.val)

# 4.3 BFS by level: snapshot the queue length
def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    if not root:
        return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):          # exactly the nodes of the current level
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res

# 4.4 Iterative inorder (BST -> sorted order; stop early for k-th smallest)
def kth_smallest(root: TreeNode, k: int) -> int:
    stack: list[TreeNode] = []
    cur = root
    while cur or stack:
        while cur:                       # go as far left as possible
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()                # the next node in sorted order
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right
    raise ValueError("k out of range")

# 4.5 Serialize / deserialize (preorder + null markers)
def serialize(root: Optional[TreeNode]) -> str:
    out: list[str] = []
    def dfs(node: Optional[TreeNode]) -> None:
        if not node:
            out.append('#')
            return
        out.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ','.join(out)

def deserialize(data: str) -> Optional[TreeNode]:
    tokens = iter(data.split(','))
    def build() -> Optional[TreeNode]:
        v = next(tokens)
        if v == '#':
            return None
        node = TreeNode(int(v))
        node.left = build()              # preorder: consume left subtree fully, then right
        node.right = build()
        return node
    return build()

# 4.6 Build from preorder + inorder: the root splits the inorder array
def build_tree(preorder: list[int], inorder: list[int]) -> Optional[TreeNode]:
    idx = {v: i for i, v in enumerate(inorder)}   # O(1) root lookup (values unique)
    pre = iter(preorder)
    def build(lo: int, hi: int) -> Optional[TreeNode]:   # inorder[lo..hi]
        if lo > hi:
            return None
        root = TreeNode(next(pre))
        mid = idx[root.val]
        root.left = build(lo, mid - 1)   # left first: preorder is N L R
        root.right = build(mid + 1, hi)
        return root
    return build(0, len(inorder) - 1)

# 4.7 LCA in a BST: walk until p and q split
def lca_bst(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    node = root
    while node:
        if p.val < node.val and q.val < node.val:
            node = node.left
        elif p.val > node.val and q.val > node.val:
            node = node.right
        else:
            return node
    raise ValueError("p or q not in tree")
```
**Trace 4.5** tree 1(2, 3(4,5)) → "1,2,#,#,3,4,#,#,5,#,#". Deserialize reads 1, then builds left: 2, #, # → leaf 2. Then right: 3 … ✓.

## 5. Worked examples

### Example 1 — Validate Binary Search Tree (LC 98) — Medium
**Naive idea (wrong):** check `node.left.val < node.val < node.right.val` at each node. It fails on:
```
    5
   / \
  4   6
     / \
    3   7      <- 3 < 5 but sits in the right subtree
```
**Why it fails:** the BST property is about *entire subtrees*, not children.
**Aha (top-down state):** each node inherits an **open interval (lo, hi)** from its ancestors. Going left tightens `hi`; going right tightens `lo`.
```python
def is_valid_bst(root: Optional[TreeNode]) -> bool:
    def ok(node: Optional[TreeNode], lo: float, hi: float) -> bool:
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return ok(node.left, lo, node.val) and ok(node.right, node.val, hi)
    return ok(root, float('-inf'), float('inf'))
```
**Alternative:** an inorder traversal must be strictly increasing (track `prev`).
**Complexity:** O(n) time, O(h) space (h = height; O(n) for a skewed tree).
**Edge cases:** duplicates (strict `<` makes them invalid, per the LC definition). Values equal to ±2³¹ (the infinite bounds handle them).

### Example 2 — Lowest Common Ancestor of a Binary Tree (LC 236) — Medium
**Brute force:** find the root→p and root→q paths (O(n) each), then take the last common node. That's O(n) time with extra path storage and two passes.
**Aha (bottom-up signal):** ask each subtree "did you find p or q?". The first node where **both** sides report something is the LCA. If a node *is* p (or q), it can return itself immediately: either the other target is below it (so it's the LCA), or the other is elsewhere and an ancestor will combine the reports.
```python
def lowest_common_ancestor(root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
    if not root or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root            # p and q are on different sides -> root is the split point
    return left or right       # propagate whichever was found (or None)
```
**Trace:** in tree 3(5(6, 2(7,4)), 1(0,8)), p=5, q=4 → the left subtree returns 5 immediately (4 is below it), the right returns None → answer 5 ✓.
**Complexity:** O(n) time, O(h) space.
**Edge cases:** p is an ancestor of q. This code assumes both exist. If they might not, count the found targets explicitly.

### Example 3 — Binary Tree Maximum Path Sum (LC 124) — Hard
**Problem:** the max sum of any path (node to node, any direction, at least one node). Values may be negative.
**Brute force:** for every pair of nodes, sum the path between them → O(n²) pairs × O(n).
**Aha (split what you *report* from what you *record*):** any path has a unique highest node where it bends. At that node, the path = node + best downward chain on the left (if positive) + best downward chain on the right (if positive). The parent can only extend a *straight* chain, so report `node + max(left, right)` upward and record the bent path globally.
```python
def max_path_sum(root: TreeNode) -> int:
    best = float('-inf')
    def gain(node: Optional[TreeNode]) -> int:
        nonlocal best
        if not node:
            return 0
        l = max(gain(node.left), 0)            # drop negative chains
        r = max(gain(node.right), 0)
        best = max(best, node.val + l + r)     # record: the path bending here
        return node.val + max(l, r)            # report: a straight chain for the parent
    gain(root)
    return best
```
**Trace** −10(9, 20(15, 7)): gain(15)=15, gain(7)=7, at 20: best=42, report 35. At −10: l=9, r=35 → path −10+9+35=34 < 42 → answer 42 ✓.
**Complexity:** O(n) time, O(h) space.
**Edge cases:** all negative (e.g., [−3]) → answer −3. That's why `best` starts at −∞, not 0, and the chains are clamped at 0 but the node itself is always included.

## 6. Common mistakes & traps
- Checking only children for BST validity (Example 1).
- Confusing "what I return" with "the answer" (diameter, max path sum). Write both down before coding.
- Depth vs height, and edges vs nodes (diameter is in *edges*).
- Min depth: a node with one child is **not** a leaf. `min(left, right)` with a 0 from a missing child is wrong.
- BFS without snapshotting `len(q)` → levels mix.
- Using `==` on nodes with duplicate values when identity is meant → use `is`.
- Forgetting `nonlocal` for a global accumulator in a nested function.
- Recursion depth on skewed trees in Python (default limit 1000).
- Path Sum problems: "root-to-leaf" (must end at a leaf) vs "any downward path" (prefix sums + hash map, LC 437).

## 7. Connections
- **DFS / BFS on graphs** (Ch 14): a tree is a connected acyclic graph. "All nodes distance K" converts the tree to a graph via parent pointers.
- **DP on trees** (Ch 17): House Robber III and Binary Tree Cameras return *tuples of states* per node. This is the bottom-up pattern with richer reports.
- **Prefix sum + hash map** along the root path with backtracking (Path Sum III).
- **Binary search**: a BST lookup is binary search. Count Complete Tree Nodes uses binary search on the last level.
- **Stack**: iterative traversals; BST Iterator (LC 173) = a paused inorder traversal.
- **Heap**: k-th largest in a BST (reverse inorder is better).
- **Serialization + hashing**: Find Duplicate Subtrees / Subtree of Another Tree (serialize each subtree to a canonical key).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 104 | [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | Easy | Answer = f(children answers) |
| 226 | [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) | Easy | A local operation applied recursively |
| 100 | [Same Tree](https://leetcode.com/problems/same-tree/) | Easy | Parallel recursion on two trees |
| 572 | [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) | Easy | Same Tree at every node (or serialize) |
| 543 | [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) | Easy | Report height, record the bend |
| 110 | [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) | Easy | Report height or −1 for "unbalanced" |
| 235 | [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | Medium | BST → walk to the split point |
| 102 | [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/) | Medium | "Level" → BFS with a snapshot |
| 199 | [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) | Medium | Last node per level |
| 1448 | [Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) | Medium | Depends on ancestors → pass the max down |
| 98 | [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) | Medium | Subtree-wide constraint → pass bounds |
| 230 | [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) | Medium | BST inorder = sorted; stop early |
| 105 | [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | Medium | The root splits inorder → index map |
| 236 | [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | Medium | "Found p or q" reports combine |
| 863 | [All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/) | Medium | Distance in any direction → parent map + BFS |
| 124 | [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | Hard | Report the straight chain, record the bent path |
| 297 | [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | Hard | Preorder + null markers is unambiguous |
| 968 | [Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/) | Hard | Greedy/DP bottom-up with 3 states per node |

## 9. Recognition drill
1. Return the sum of all numbers formed by root-to-leaf digit paths.
2. Return the maximum width of a binary tree, where width counts the null positions between the end nodes of a level.
3. Given a BST and a range [lo, hi], return the sum of values in the range, visiting as few nodes as possible.
4. Find the length of the longest path where every node has the same value.
5. Given a binary tree, flatten it into a "linked list" in preorder, in place.

<details><summary>Answers</summary>

1. **Top-down DFS** passing `cur = cur*10 + val`; add at leaves.
2. **BFS with position indices** (left child 2i, right child 2i+1); width = last − first + 1 per level. Normalize per level to keep the numbers small.
3. **BST-pruned DFS**: skip the left subtree if val < lo, skip the right if val > hi.
4. **Bottom-up**: report the longest same-value downward chain; record left+right chains at each node (the diameter pattern with a condition).
5. **Reverse postorder (right, left, node) with a `prev` pointer**, or a Morris-style rewire: attach the right subtree to the rightmost node of the left subtree.
</details>

Next: Part 11 — Heaps / Priority Queues
