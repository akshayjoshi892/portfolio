# Part 19 — Bit Manipulation

## 1. Core insight
An integer is a vector of independent bits, and bitwise operators act on all of them at once in O(1). That gives three kinds of leverage. **Algebraic identities**: XOR is its own inverse (`a ^ a = 0`, `a ^ 0 = a`, commutative), so pairs cancel and only the odd one survives. **Per-bit decomposition**: a problem that's hard on whole numbers is often easy per bit (count the 1s at each position, decide the highest bit greedily). **Sets as integers**: a subset of n ≤ 20–30 items is a mask, and union/intersection/membership are single operations. That's what bitmask enumeration and bitmask DP run on.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Every element appears twice/three times except one" → XOR / bit counting.
- "Missing number in 0..n" → XOR of indices and values (or the sum formula).
- "O(1) extra space" + a pairing/cancellation structure.
- "Maximum XOR", "XOR of a subarray" → binary trie / prefix XOR.
- "Power of two", "count set bits", "Hamming distance", "reverse bits".
- n ≤ 20 with subsets → mask enumeration. States with a used-set → bitmask DP.
- "Add without + / −", "divide without / or *" → bit-level arithmetic.

**Anti-signals**
- Large n with subset *sums* → knapsack DP, not 2ⁿ masks.
- Readability matters more than micro-speed → use sets, and mention the bitmask optimization.
- Python's arbitrary-precision ints: "32-bit overflow" semantics must be **simulated with masks**. Don't assume C behavior.

## 3. Mental model
```
x          = 0b 1011 0100
x - 1      = 0b 1011 0011     (the lowest 1 becomes 0, the bits below it become 1)
x & (x-1)  = 0b 1011 0000     -> clears the lowest set bit
-x         = ~x + 1 = ...0100 1100
x & -x     = 0b 0000 0100     -> isolates the lowest set bit
```
**XOR as "toggle"**: XOR every number into an accumulator. Each number toggles its bits, pairs toggle twice (no effect), and the unpaired number remains.

## 4. Canonical templates

| Operation | Expression |
|---|---|
| Is bit k set | `(x >> k) & 1` |
| Set / clear / toggle bit k | `x \| (1 << k)`, `x & ~(1 << k)`, `x ^ (1 << k)` |
| Clear the lowest set bit | `x & (x - 1)` |
| Isolate the lowest set bit | `x & -x` |
| Power of two (x > 0) | `x & (x - 1) == 0` |
| Popcount | `x.bit_count()` (3.10+) or `bin(x).count('1')` |
| Number of bits | `x.bit_length()` |
| All n bits set | `(1 << n) - 1` |
| Is mask a subset of full | `mask & full == mask` |

```python
# 4.1 Enumerate all subsets of n items
def all_subsets(nums: list[int]) -> list[list[int]]:
    n = len(nums)
    return [[nums[i] for i in range(n) if (mask >> i) & 1] for mask in range(1 << n)]

# 4.2 Enumerate all submasks of `mask` (descending), used in subset DP: O(3^n) total over all masks
def submasks(mask: int):
    s = mask
    while s:
        yield s
        s = (s - 1) & mask
    yield 0

# 4.3 Brian Kernighan popcount: loops once per set bit
def popcount(x: int) -> int:
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

# 4.4 32-bit simulation in Python (e.g., Sum of Two Integers, LC 371)
def get_sum(a: int, b: int) -> int:
    MASK, MAX = 0xFFFFFFFF, 0x7FFFFFFF
    while b:
        a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK   # sum without carry, carry
    return a if a <= MAX else ~(a ^ MASK)              # reinterpret as signed 32-bit

# 4.5 Reverse 32 bits
def reverse_bits(n: int) -> int:
    res = 0
    for _ in range(32):
        res = (res << 1) | (n & 1)
        n >>= 1
    return res

# 4.6 Two unique numbers, all others paired (LC 260): split by a differing bit
def single_number_iii(nums: list[int]) -> list[int]:
    xor_all = 0
    for x in nums:
        xor_all ^= x                    # = a ^ b (nonzero since a != b)
    low = xor_all & -xor_all            # a bit where a and b differ
    a = 0
    for x in nums:
        if x & low:
            a ^= x                      # the group containing a (pairs still cancel)
    return [a, xor_all ^ a]
```
**Trace 4.4** get_sum(1, 2): a=3, b=0 → 3 ✓. get_sum(−1, 1): the carries ripple until they fall off the 32-bit mask, then a=0 → 0 ✓.

## 5. Worked examples

### Example 1 — Single Number (LC 136) — Easy
**Brute force:** a hash counter → O(n) time, O(n) space. Sorting → O(n log n).
**Aha (signals: "twice except one" + O(1) space):** XOR everything. Pairs cancel (`a ^ a = 0`), and order doesn't matter (commutative, associative).
```python
from functools import reduce
from operator import xor

def single_number(nums: list[int]) -> int:
    return reduce(xor, nums, 0)
```
**Complexity:** O(n) time, O(1) space.
**Edge cases:** negative numbers work in Python (XOR on two's-complement semantics). A single element.

### Example 2 — Counting Bits (LC 338) — Easy
**Problem:** for every i in [0, n], return popcount(i).
**Brute force:** popcount each → O(n log n) (or O(n · bits)).
**Aha (DP over bits):** `i >> 1` drops the lowest bit, and it's already computed. `popcount(i) = popcount(i >> 1) + (i & 1)`. Alternatively `popcount(i) = popcount(i & (i−1)) + 1`.
```python
def count_bits(n: int) -> list[int]:
    res = [0] * (n + 1)
    for i in range(1, n + 1):
        res[i] = res[i >> 1] + (i & 1)
    return res
```
**Trace** n=5 → [0,1,1,2,1,2] ✓.
**Complexity:** O(n) time, O(n) output.

### Example 3 — Single Number II (LC 137) — Medium
**Problem:** every element appears three times except one. O(n) time, O(1) space.
**Why XOR fails:** three copies XOR to the number itself, not 0.
**Aha (per-bit decomposition):** at each bit position, the count of 1s across all numbers is `3k + (bit of the single number)`. So `count % 3` recovers that bit.
```python
def single_number_ii(nums: list[int]) -> int:
    res = 0
    for b in range(32):
        cnt = sum((x >> b) & 1 for x in nums)   # Python >> on negatives sign-extends: correct bits
        if cnt % 3:
            res |= 1 << b
    return res - (1 << 32) if res >= 1 << 31 else res   # restore the sign (32-bit)
```
**The O(1)-per-element state machine** (each bit counts mod 3 using two bitmasks):
```python
def single_number_ii_fsm(nums: list[int]) -> int:
    ones = twos = 0
    for x in nums:
        ones = (ones ^ x) & ~twos     # bits seen once (mod 3)
        twos = (twos ^ x) & ~ones     # bits seen twice (mod 3)
    return ones
```
**Complexity:** O(32·n) = O(n) time, O(1) space.
**Edge cases:** a negative single number (the sign restoration in the counting version). Python's infinite-width ints make the FSM version work directly.

## 6. Common mistakes & traps
- Operator precedence: `x & 1 == 0` parses as `x & (1 == 0)`. Always parenthesize: `(x & 1) == 0`.
- Python ints are unbounded: `~x` is `−x−1`, never "flip 32 bits". Use `x ^ 0xFFFFFFFF` and reinterpret the sign explicitly.
- Right-shifting negatives in Python is arithmetic (it floors). Loops like `while x: x >>= 1` never end for negative x.
- `1 << n` for the subset count vs `(1 << n) − 1` for the full mask.
- Using `x & -x` and forgetting that it's a *value* (a power of two), not a bit index. Use `.bit_length() − 1` for the index.
- The submask enumeration must also include 0 if needed (the loop stops before 0).

## 7. Connections
- **Bitmask DP** (Ch 17): TSP, assignment, partition into k subsets. **Bitmask BFS** (Ch 16): visit all nodes.
- **Backtracking** (Ch 9): N-Queens with bitmasks (cols/diag/anti as ints).
- **Trie** (Ch 18): a binary trie for max XOR.
- **Prefix sums** (Ch 5): prefix XOR for range XOR queries, and "subarrays with XOR = k" (prefix XOR + hash map).
- **Math** (Ch 20): fast exponentiation iterates over the exponent's bits. Gray code.
- **Hashing**: bitmask signatures for "shared letters" (Maximum Product of Word Lengths: mask of letters, AND == 0).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 136 | [Single Number](https://leetcode.com/problems/single-number/) | Easy | Pairs + one → XOR |
| 191 | [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/) | Easy | `x & (x−1)` clears one bit |
| 338 | [Counting Bits](https://leetcode.com/problems/counting-bits/) | Easy | popcount(i) from popcount(i >> 1) |
| 268 | [Missing Number](https://leetcode.com/problems/missing-number/) | Easy | XOR indices with values |
| 190 | [Reverse Bits](https://leetcode.com/problems/reverse-bits/) | Easy | Shift out / shift in 32 times |
| 231 | [Power of Two](https://leetcode.com/problems/power-of-two/) | Easy | A single set bit |
| 461 | [Hamming Distance](https://leetcode.com/problems/hamming-distance/) | Easy | popcount(x ^ y) |
| 137 | [Single Number II](https://leetcode.com/problems/single-number-ii/) | Medium | Thrice except one → count bits mod 3 |
| 260 | [Single Number III](https://leetcode.com/problems/single-number-iii/) | Medium | Two singles → split by a differing bit |
| 371 | [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/) | Medium | No + → XOR for the sum, AND << 1 for the carry |
| 201 | [Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/) | Medium | Common high-bit prefix of left and right |
| 477 | [Total Hamming Distance](https://leetcode.com/problems/total-hamming-distance/) | Medium | Per bit: ones × zeros |
| 318 | [Maximum Product of Word Lengths](https://leetcode.com/problems/maximum-product-of-word-lengths/) | Medium | No shared letters → letter masks, AND == 0 |
| 1239 | [Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/) | Medium | Small n + unique chars → masks + backtracking |
| 29 | [Divide Two Integers](https://leetcode.com/problems/divide-two-integers/) | Medium | No division → subtract shifted divisors |

## 9. Recognition drill
1. Given an array of n distinct numbers in [0, n], find the missing one in O(1) space.
2. Given a list of integers, find the pair whose XOR is maximal.
3. Given words, count the pairs of words that share no common letters.
4. Given `left` and `right`, compute AND of all numbers in [left, right] (up to 2³¹).
5. Given n ≤ 15 people with skill sets, find the smallest team covering all required skills (≤ 16 skills).

<details><summary>Answers</summary>

1. **XOR of all indices 0..n and all values** (or n(n+1)/2 − sum).
2. **Binary trie, greedy from the high bit** (or the prefix-set greedy on bits).
3. **Letter bitmask per word**; a pair is valid iff `mask_a & mask_b == 0`. Group by mask to speed it up.
4. **Common prefix of left and right**: shift both right until equal, then shift back (any differing bit position has a 0 somewhere in the range).
5. **Bitmask DP over skills** (dp[skill_mask] = smallest team), LC 1125. Skills ≤ 16 → 2¹⁶ states.
</details>

Next: Part 20 — Math for Interviews
