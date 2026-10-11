# Part 18 — Tries

## 1. Core insight
A trie stores strings character by character along a tree, so **all strings with a common prefix share one path**. A prefix query costs O(L) (its length), independent of how many words are stored. It also lets a search **prune**: if the current prefix isn't in the trie, no word can extend it, so stop immediately. That's the difference between Word Search II being feasible or not. The same structure over **bits** (a binary trie) answers "which stored number maximizes XOR with x" greedily from the highest bit down.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Starts with", "prefix", "autocomplete", "search suggestions".
- Many words searched in one grid/text at once (multi-pattern search) → trie + DFS with pruning.
- Wildcard search (`.` matches any letter) over a dictionary.
- "Replace words with their shortest root", "longest word built one character at a time".
- "Maximum XOR of two numbers", "XOR queries with limits" → a binary trie.
- Counting distinct prefixes / sharing between strings.

**Anti-signals**
- Only exact-match lookups → a hash set is simpler and just as fast.
- A single pattern in a single text → KMP / rolling hash (Ch 21).
- Very few words, or a large alphabet with sparse children → a sorted list + binary search (`bisect`) for prefix ranges may be simpler (LC 1268 has a clean sort + bisect solution).
- Suffix queries → a trie of reversed strings (still a trie, but note it).

## 3. Mental model
A filing cabinet where each drawer is a letter:
```
insert: "cat", "car", "cart", "dog"
root
├── c
│   └── a
│       ├── t*          (* = end of word)
│       └── r*
│           └── t*
└── d
    └── o
        └── g*
search("car")      -> path exists, node marked *  -> True
startsWith("ca")   -> path exists                 -> True
search("ca")       -> path exists, not marked     -> False
```

## 4. Canonical templates

```python
class TrieNode:
    __slots__ = ('children', 'is_end')
    def __init__(self) -> None:
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end = False

class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.is_end = True

    def _walk(self, s: str) -> TrieNode | None:
        node = self.root
        for c in s:
            node = node.children.get(c)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

# Lightweight dict-of-dicts trie (handy in interviews): '$' marks the end of a word
def build_trie(words: list[str]) -> dict:
    root: dict = {}
    for w in words:
        node = root
        for c in w:
            node = node.setdefault(c, {})
        node['$'] = w                    # store the word itself at its terminal node
    return root

# Binary trie for max XOR (LC 421)
def find_maximum_xor(nums: list[int]) -> int:
    L = max(nums).bit_length()
    root: dict = {}
    for x in nums:
        node = root
        for b in range(L - 1, -1, -1):
            node = node.setdefault((x >> b) & 1, {})
    best = 0
    for x in nums:
        node, cur = root, 0
        for b in range(L - 1, -1, -1):
            bit = (x >> b) & 1
            if 1 - bit in node:          # the opposite bit makes this XOR bit 1: take it greedily
                cur |= 1 << b
                node = node[1 - bit]
            else:
                node = node[bit]
        best = max(best, cur)
    return best
```
**Trace (max XOR)** [3,10,5,25,2,8], L = 5: for x = 5 (00101), the greedy path follows bits 1,1,0,0,1 (opposite bits where they exist; at bit 0 only 25's 1 exists) → 25 (11001) → 5 ^ 25 = 28 ✓. The greedy is correct because a higher bit outweighs all lower bits combined.

## 5. Worked examples

### Example 1 — Implement Trie (LC 208) — Medium
**Brute force:** store the words in a list; `startsWith` scans all → O(N·L) per query.
**Hash-set alternative:** also store every prefix of every word in a set → O(1) queries but O(Σ L²) memory and time.
**Aha:** shared prefixes share nodes, and every operation is O(L). Code: the `Trie` class above.
**Complexity:** O(L) time per operation. O(total characters) space (× the alphabet factor if you use arrays of 26).
**Edge cases:** searching a prefix that is itself not a word (`is_end` False). An empty string prefix → True.

### Example 2 — Design Add and Search Words Data Structure (LC 211) — Medium
**Problem:** support `addWord` and `search(word)` where `.` matches any single letter.
**Brute force:** compare against every stored word → O(N·L) per search.
**Aha (signal: wildcard over a dictionary):** walk the trie. On `.`, branch into **all** children (DFS). Other letters follow one path. Pruning comes for free: absent prefixes stop immediately.
```python
class WordDictionary:
    def __init__(self) -> None:
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, i: int) -> bool:
            if i == len(word):
                return node.is_end
            c = word[i]
            if c == '.':
                return any(dfs(child, i + 1) for child in node.children.values())
            child = node.children.get(c)
            return child is not None and dfs(child, i + 1)
        return dfs(self.root, 0)
```
**Complexity:** add O(L). Search O(L) without dots, O(26^d · L) worst case with d dots. Space O(total characters).
**Edge cases:** all dots ("..."), words of different lengths (checked by `is_end` at the end).

### Example 3 — Word Search II (LC 212) — Hard
**Problem:** find all dictionary words present in a letter grid (adjacent cells, no reuse within a word).
**Brute force:** run Word Search (Ch 9) for each word → O(W · R·C · 4·3^(L−1)). With W = 3·10⁴ words, that's dead.
**Bottleneck:** words sharing prefixes re-explore the same grid paths, and DFS continues down paths that match no word.
**Aha (signal: many words, one grid):** one DFS from each cell, guided by the trie. Stop as soon as the path isn't a trie prefix. **Prune** found words and emptied branches so later searches get cheaper.
```python
def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    root = build_trie(words)
    R, C = len(board), len(board[0])
    res: list[str] = []

    def dfs(r: int, c: int, parent: dict) -> None:
        ch = board[r][c]
        node = parent[ch]
        if '$' in node:
            res.append(node.pop('$'))            # record once; pop avoids duplicates
        board[r][c] = '#'                        # mark visited
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and board[nr][nc] in node:
                dfs(nr, nc, node)
        board[r][c] = ch                         # restore
        if not node:                             # leaf exhausted: prune it from the trie
            parent.pop(ch)

    for r in range(R):
        for c in range(C):
            if board[r][c] in root:
                dfs(r, c, root)
    return res
```
**Complexity:** O(R·C · 4·3^(L−1)) worst case (L = max word length), heavily pruned in practice. O(total characters) space for the trie.
**Edge cases:** duplicate words in the input (the trie merges them). A word found via multiple paths (pop '$' once). A 1×1 board.

## 6. Common mistakes & traps
- Forgetting the end-of-word marker → `search("ca")` returns True after inserting "cat".
- Creating nodes during `search` (using `setdefault` in a read path) → pollutes the trie.
- Word Search II: not un-marking the cell, not deduplicating found words, not pruning (TLE).
- Using a 26-array per node for sparse data → memory blowup. Use a dict unless profiling says otherwise.
- Binary trie: iterating bits in the wrong order (must be high → low for greedy XOR) or with an inconsistent bit length between insert and query.
- Recursion depth: fine for words (L ≤ 100s), but watch out for pathological inputs.

## 7. Connections
- **Backtracking** (Ch 9): Word Search II = grid backtracking + trie pruning.
- **Bit manipulation** (Ch 19): binary trie for XOR problems; greedy from the highest bit.
- **Hashing** (Ch 2): an exact-match set vs a prefix trie. Rolling hash (Ch 21) for substring matching.
- **DP**: Word Break can use a trie to enumerate the dictionary words starting at i in O(L) instead of trying every length.
- **Sorting + binary search**: autocomplete over sorted products (all words with prefix p form a contiguous range).
- **Heap**: autocomplete ranked by frequency (store top-k per node, or a heap at query time).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 14 | [Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/) | Easy | Shared prefix (vertical scan or trie) |
| 208 | [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/) | Medium | "startsWith" → trie |
| 211 | [Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | Medium | Wildcard over a dictionary → DFS in the trie |
| 648 | [Replace Words](https://leetcode.com/problems/replace-words/) | Medium | Shortest root prefix of each word |
| 677 | [Map Sum Pairs](https://leetcode.com/problems/map-sum-pairs/) | Medium | Sum over all keys with a prefix → store sums on nodes |
| 720 | [Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/) | Medium | Built one char at a time → every prefix node is_end |
| 1268 | [Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/) | Medium | Autocomplete per typed prefix (trie or sort + bisect) |
| 421 | [Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) | Medium | Max XOR → binary trie, high bit first |
| 212 | [Word Search II](https://leetcode.com/problems/word-search-ii/) | Hard | Many words on one grid → trie-guided DFS |
| 472 | [Concatenated Words](https://leetcode.com/problems/concatenated-words/) | Hard | A word made of other words → Word Break per word |
| 745 | [Prefix and Suffix Search](https://leetcode.com/problems/prefix-and-suffix-search/) | Hard | Insert `suffix + '#' + word` for every suffix |
| 336 | [Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/) | Hard | Reversed-word trie + palindrome checks on the remainders |
| 1707 | [Maximum XOR With an Element From Array](https://leetcode.com/problems/maximum-xor-with-an-element-from-array/) | Hard | XOR with a limit → offline sort + binary trie |

## 9. Recognition drill
1. Given a stream of characters, after each character report whether some suffix of the stream is a dictionary word.
2. Given a list of phone numbers, determine whether any number is a prefix of another.
3. Given a list of product names and a search word, return up to 3 lexicographically smallest suggestions after each character typed.
4. Given integers and queries (x, m), return the max `x XOR a` over elements a ≤ m.
5. Count the number of distinct substrings of a string of length ≤ 1000.

<details><summary>Answers</summary>

1. **Trie of reversed words**; walk backwards from the newest character over the recent stream (LC 1032).
2. **Trie (or sort + check adjacent pairs)**: during insertion, hitting an existing end marker or ending on a non-leaf node means yes.
3. **Trie with sorted suggestions per node** or **sort + bisect** (the prefix range is contiguous in sorted order).
4. **Offline: sort the queries by m and the numbers ascending**, insert numbers ≤ m into a binary trie, then run the greedy XOR query (LC 1707).
5. **Insert every suffix into a trie**; the number of nodes (excluding the root) = distinct substrings. O(n²). (A suffix automaton or suffix array is the advanced route.)
</details>

Next: Part 19 — Bit Manipulation
