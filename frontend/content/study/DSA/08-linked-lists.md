# Part 8 — Linked Lists (Fast/Slow, Reversal, Dummy Node)

## 1. Core insight
A linked list gives O(1) splice/insert/delete **once you hold the right pointer**, but no random access. So problems reduce to two skills. **Positioning** a pointer without knowing the length: fast/slow pointers use a speed difference to find the middle, the k-th from the end, cycles, and cycle entries in one pass and O(1) space. **Rewiring** pointers safely: in-place reversal, merging, partitioning, where a **dummy head** removes every "what if the head changes?" special case. Most "hard" list problems are compositions of 2–3 of these primitives.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Cycle", "loop", "where does the cycle begin" → Floyd's tortoise and hare.
- "Middle", "k-th from the end", "split in half", "palindrome list" → fast/slow.
- "Reverse" (whole, sublist, k-group), "reorder", "rotate", "swap pairs" → reversal primitive.
- "Remove / insert / partition" where the head might change → dummy node.
- "Merge sorted lists", "sort a list in O(n log n) with O(1) space" → merge primitive / merge sort.
- Array with values in [1..n] + one duplicate + O(1) space + read-only → Floyd on the functional graph `i → nums[i]`.

**Anti-signals**
- Random access needed repeatedly → copy into a Python list first (O(n) space) if allowed. It's often acceptable, and saying so shows judgment.
- Deep copy with random pointers → a hash map old→new, or the interleaving trick (not fast/slow).
- LRU-type O(1) reordering needs a **doubly** linked list + hash map (Ch 21), not these singly-linked tricks.

## 3. Mental model
**Reversal: three hands.** `prev`, `cur`, `nxt`. Save the rest, flip one arrow, step forward.
```
prev   cur    nxt
None   1 -> 2 -> 3 -> None
       1 -> None      (cur.next = prev)
       prev=1, cur=2  ...  finally prev = new head
```
**Fast/slow:** fast moves 2, slow moves 1. When fast hits the end, slow is at the middle. In a cycle, fast gains 1 step per move on slow, so they must meet within one lap.

**Floyd's cycle entry proof.** Let `a` = head→entry, `b` = entry→meeting point, `c` = meeting point→entry (lap length L = b + c).
```
slow walked a + b,  fast walked 2(a + b) = a + b + kL
=> a + b = kL  => a = kL - b = (k-1)L + c
```
So a pointer from the head and a pointer from the meeting point, both moving 1 step, meet exactly at the entry.

## 4. Canonical templates

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'Optional[ListNode]' = None):
        self.val, self.next = val, next

# 4.1 Reverse a list
def reverse(head: Optional[ListNode]) -> Optional[ListNode]:
    prev, cur = None, head
    while cur:
        nxt = cur.next      # save the rest
        cur.next = prev     # flip
        prev, cur = cur, nxt
    return prev             # Invariant: prev is the head of the reversed prefix

# 4.2 Middle (for even length returns the SECOND middle)
def middle(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow
# For the FIRST middle (useful for splitting): loop on `fast.next and fast.next.next`.

# 4.3 Dummy node + remove n-th from the end (gap of n between two pointers)
def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):          # fast is n+1 ahead so slow stops BEFORE the target
        fast = fast.next
    while fast:
        slow, fast = slow.next, fast.next
    slow.next = slow.next.next
    return dummy.next               # correct even when the head was removed

# 4.4 Merge two sorted lists
def merge_two(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:          # <= keeps it stable
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next

# 4.5 Reverse the sublist between positions left..right (1-indexed), "front insertion"
def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    before = dummy
    for _ in range(left - 1):
        before = before.next
    cur = before.next               # first node of the segment; it becomes the segment tail
    for _ in range(right - left):
        nxt = cur.next              # move nxt to the front of the segment
        cur.next = nxt.next
        nxt.next = before.next
        before.next = nxt
    return dummy.next
```
**Trace 4.5** 1→2→3→4→5, left=2, right=4: before=1, cur=2. Move 3 to the front: 1→3→2→4→5. Move 4 to the front: 1→4→3→2→5 ✓.

## 5. Worked examples

### Example 1 — Linked List Cycle II (LC 142) — Medium
**Brute force:** a hash set of visited nodes. The first repeat is the entry. O(n) time, O(n) space.
**Aha (signal: cycle + "O(1) memory" follow-up):** Floyd. Detect the meeting point, then reset one pointer to the head. By the proof in §3, they meet at the entry.
```python
def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:                     # inside the cycle
            p = head
            while p is not slow:
                p, slow = p.next, slow.next
            return p
    return None
```
**Complexity:** O(n) time (fast meets slow within one lap after slow enters), O(1) space.
**Edge cases:** empty, single node without a cycle, a self-loop (node.next = node), the cycle entry is the head.
**Transfer:** LC 287 Find the Duplicate Number. Treat index i → nums[i] as a `next` pointer. The duplicate value is the cycle entry.

### Example 2 — Reorder List (LC 143) — Medium
**Problem:** L0→Ln→L1→Ln−1→… in place.
**Brute force:** copy the nodes into an array and relink with two pointers → O(n) space. That's acceptable to mention.
**Aha (compose primitives):** the second half is consumed in *reverse* order → (1) find the middle, (2) reverse the second half, (3) merge alternately.
```python
def reorder_list(head: Optional[ListNode]) -> None:
    if not head or not head.next:
        return
    slow = fast = head                                   # 1) first middle
    while fast.next and fast.next.next:
        slow, fast = slow.next, fast.next.next
    second, slow.next = slow.next, None                  # cut
    prev = None                                          # 2) reverse second half
    while second:
        nxt = second.next
        second.next = prev
        prev, second = second, nxt
    first, second = head, prev                           # 3) weave
    while second:
        n1, n2 = first.next, second.next
        first.next, second.next = second, n1
        first, second = n1, n2
```
**Trace** 1→2→3→4→5: middle=3, second half 4→5 reversed to 5→4, first half 1→2→3. Weave: 1→5→2→4→3 ✓. With 4 nodes: 1→4→2→3 ✓.
**Complexity:** O(n) time, O(1) space.
**Edge cases:** 1 or 2 nodes (no change). Forgetting to cut (`slow.next = None`) creates a cycle.

### Example 3 — Reverse Nodes in k-Group (LC 25) — Hard
**Problem:** reverse every consecutive group of k nodes. A leftover group (< k) stays as is.
**Brute force:** push k nodes onto a stack and relink → O(k) extra space.
**Aha (compose primitives):** a dummy node + "is there a full group?" check + reverse the group in place, with the reversal's `prev` initialized to the node *after* the group so the tail connects automatically.
```python
def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev                       # find the k-th node of this group
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next              # fewer than k left: done
        group_next = kth.next
        prev, cur = group_next, group_prev.next   # reverse [group start .. kth]
        while cur is not group_next:
            nxt = cur.next
            cur.next = prev
            prev, cur = cur, nxt
        old_first = group_prev.next            # becomes the group's tail
        group_prev.next = kth                  # kth is the new group head
        group_prev = old_first
```
**Trace** 1→2→3→4→5, k=2: group [1,2] → 2→1→3…, group_prev=1. Group [3,4] → 2→1→4→3→5. The next group has only 5 → return 2→1→4→3→5 ✓.
**Complexity:** O(n) time (each node is visited twice), O(1) space.
**Edge cases:** k = 1 (no change), k = n, n not divisible by k.

## 6. Common mistakes & traps
- Losing the rest of the list: always save `nxt = cur.next` before rewiring.
- No dummy node → special-case bugs when the head is removed or changes.
- Fast/slow loop condition: `while fast and fast.next` (check both before `fast.next.next`).
- Choosing the wrong middle: for splitting, you usually want the **first** middle so the halves are [≤ half] and [rest].
- Forgetting to **cut** the list after splitting → cycles and infinite loops.
- Comparing nodes with `==`. If `__eq__` is overridden this breaks. Use `is`.
- Remove n-th from end: advance fast n+1 steps from the *dummy* so slow stops at the predecessor.
- Recursion on long lists (10⁵ nodes) → RecursionError in Python. Prefer iteration.

## 7. Connections
- **Two pointers** (Ch 3): fast/slow is the list version; a gap of k gives k-th from end.
- **Merge sort** on lists (LC 148): middle + recursive sort + merge_two → O(n log n) time, O(log n) stack.
- **Heap** (Ch 11): Merge k Sorted Lists = k-way merge.
- **Hash map**: Copy List with Random Pointer (old→new map), LRU cache (map + doubly linked list, Ch 21).
- **Cycle detection in functional graphs**: Happy Number (LC 202), Find the Duplicate Number (LC 287).
- **Stack**: palindrome linked list can use a stack (O(n) space) vs reverse-second-half (O(1)).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 206 | [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) | Easy | The reversal primitive; do both iterative and recursive |
| 21 | [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) | Easy | Dummy + tail pointer |
| 141 | [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) | Easy | O(1) space cycle → speed difference |
| 876 | [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) | Easy | Unknown length → fast/slow |
| 234 | [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) | Easy | O(1) space → reverse the second half |
| 160 | [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/) | Easy | Switch heads to equalize path lengths |
| 19 | [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) | Medium | "From the end" in one pass → gap pointers + dummy |
| 2 | [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) | Medium | Digit-by-digit with carry; dummy head |
| 142 | [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) | Medium | Cycle entry → Floyd phase 2 |
| 143 | [Reorder List](https://leetcode.com/problems/reorder-list/) | Medium | Consume the tail in reverse → middle + reverse + weave |
| 92 | [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) | Medium | Sublist reversal → dummy + front insertion |
| 138 | [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/) | Medium | Arbitrary pointers → old→new map (or interleave) |
| 287 | [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) | Medium | Values in [1..n], read-only, O(1) space → functional-graph cycle |
| 148 | [Sort List](https://leetcode.com/problems/sort-list/) | Medium | O(n log n) on a list → merge sort |
| 25 | [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) | Hard | Repeated sublist reversal with a full-group check |
| 23 | [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) | Hard | k sorted streams → heap or divide and conquer |

## 9. Recognition drill
1. Rotate a linked list to the right by k places (k can exceed the length).
2. Determine whether repeatedly replacing a number by the sum of squares of its digits reaches 1.
3. Partition a list so nodes < x come before nodes ≥ x, preserving relative order.
4. Given the head of a singly linked list, return the node where two lists merge, in O(1) space.
5. Swap every two adjacent nodes without modifying values.

<details><summary>Answers</summary>

1. **Length + make it circular + cut** at `len − k % len`. Fast/slow with a gap also works once k is reduced mod len.
2. **Cycle detection on a functional sequence** (Floyd, or a hash set). The sequence either reaches 1 or loops.
3. **Two dummy heads** (less / greater-equal) and stitch them together. Remember to terminate the second list.
4. **Two pointers switching heads**: each walks a+c+b, so they meet at the intersection (or both reach None).
5. **Dummy node + pairwise rewiring** (k-group reversal with k = 2).
</details>

Next: Part 9 — Recursion & Backtracking
