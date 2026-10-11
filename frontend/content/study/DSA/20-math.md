# Part 20 — Math for Interviews

## 1. Core insight
Interview math is a small toolkit for replacing enumeration with a formula or a structure. **GCD** captures shared structure between numbers (periodicity, LCM, reducing fractions and slopes). **Modular arithmetic** keeps huge counts representable, and it's closed under +, −, ×; division requires a modular inverse. **Primes** via the sieve (many queries) or trial division up to √n (one number). **Combinatorics** counts arrangements directly instead of generating them (C(n, k), stars and bars, inclusion-exclusion). And **counting functions + binary search** find the k-th element of an implicitly defined sorted set. When you see "return modulo 10⁹+7" or n up to 10⁹, suspect math.

## 2. Trigger signals / Anti-signals
**Triggers**
- "Return the answer modulo 10⁹+7" → counting; apply mod at every step; use an inverse for division.
- "k-th number divisible by a or b", "k-th ugly number" → counting function (inclusion-exclusion) + binary search, or a heap/DP.
- "Count primes below n", "prime factors", "smallest prime factor for many numbers" → sieve.
- "Number of paths in a grid", "ways to choose", "distribute identical items" → binomial coefficients / stars and bars.
- "Slope", "same line", "fractions" → normalize with GCD and avoid floats.
- "Random", "uniform", "weighted pick", "sample from a stream" → prefix sums + bisect, reservoir sampling, rejection sampling.
- Digit manipulation: "reverse integer", "palindrome number", "digit sum" → `% 10`, `// 10`.
- n ≤ 10¹⁸ with a clear formula → O(1) or O(log n).

**Anti-signals**
- A formula exists but is fragile (floating point, overflow) → prefer exact integer arithmetic. Python's big ints help.
- Counting with complex constraints → DP is often cleaner than a closed form.
- Division under a modulus that isn't prime → Fermat's inverse doesn't apply (use extended Euclid, and only when gcd = 1).

## 3. Mental model
```
Euclid:  gcd(48, 18) -> gcd(18, 12) -> gcd(12, 6) -> gcd(6, 0) = 6
         (each step at least halves the larger number every two steps -> O(log min))

Sieve (n = 30): cross out multiples starting at p*p
2: 4 6 8 10 ... 30
3: 9 15 21 27 (12, 18, 24, 30 already gone)
5: 25 (the others are gone)
stop when p*p > n  -> primes: 2 3 5 7 11 13 17 19 23 29

Inclusion-exclusion: |A ∪ B| = |A| + |B| − |A ∩ B|
   multiples of a or b up to x = x//a + x//b − x//lcm(a, b)
```

## 4. Canonical templates

```python
import math
import random
from bisect import bisect_left
from itertools import accumulate

MOD = 10**9 + 7

# 4.1 GCD / LCM
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a                          # also: math.gcd

def lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b    # divide first to keep the numbers small (math.lcm in 3.9+)

# 4.2 Modular arithmetic
# (a + b) % M, (a * b) % M, (a - b) % M (Python keeps it non-negative)
# Fast power: pow(a, e, M) in O(log e)
# Inverse (M prime, a % M != 0): pow(a, M - 2, M)  (Fermat)  or pow(a, -1, M) in 3.8+

# 4.3 nCr mod p with precomputed factorials: O(n) build, O(1) per query
def build_factorials(n: int) -> tuple[list[int], list[int]]:
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], MOD - 2, MOD)
    for i in range(n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD
    return fact, inv_fact

def ncr(n: int, r: int, fact: list[int], inv_fact: list[int]) -> int:
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD
# Without a modulus: math.comb(n, r) (exact big ints).

# 4.4 Sieve of Eratosthenes: O(n log log n)
def sieve(n: int) -> list[bool]:       # is_prime[0..n]
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for p in range(2, math.isqrt(n) + 1):
        if is_prime[p]:
            for m in range(p * p, n + 1, p):   # smaller multiples were crossed out by smaller primes
                is_prime[m] = False
    return is_prime

# 4.5 Prime factorization by trial division: O(sqrt n)
def factorize(n: int) -> dict[int, int]:
    f: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:                           # the leftover is a prime > sqrt(original n)
        f[n] = f.get(n, 0) + 1
    return f

# 4.6 Weighted random pick (LC 528): prefix sums + binary search
class WeightedPicker:
    def __init__(self, w: list[int]):
        self.prefix = list(accumulate(w))
    def pick_index(self) -> int:
        r = random.randint(1, self.prefix[-1])       # uniform over the total weight
        return bisect_left(self.prefix, r)           # the first prefix >= r

# 4.7 Reservoir sampling (k = 1): uniform choice from a stream of unknown length
def reservoir_sample(stream) -> object:
    chosen = None
    for i, x in enumerate(stream, start=1):
        if random.randint(1, i) == 1:               # replace with probability 1/i
            chosen = x
    return chosen
```
**Trace 4.6** w=[1,3] → prefix [1,4]. r=1 → index 0 (prob 1/4). r ∈ {2,3,4} → index 1 (prob 3/4) ✓.
**Why 4.7 is uniform:** item i is chosen at step i with probability 1/i and survives each later step j with probability (j−1)/j. The product telescopes to 1/n.

**Facts worth knowing cold**
- Grid paths from (0,0) to (m−1, n−1) with right/down moves: C(m+n−2, m−1).
- Stars and bars: ways to put n identical items into k boxes = C(n+k−1, k−1).
- Catalan numbers C(2n, n)/(n+1): valid parentheses strings, BST shapes (LC 96), triangulations.
- Trailing zeros of n! = Σ ⌊n/5ⁱ⌋.
- The number of divisors of n ≤ 10⁹ is at most 1344. Divisor enumeration up to √n is fine.
- Sum 1..n = n(n+1)/2. Sum of squares = n(n+1)(2n+1)/6.
- The pigeonhole principle: n+1 values in [1..n] must repeat (Find the Duplicate Number).

## 5. Worked examples

### Example 1 — Count Primes (LC 204) — Medium
**Problem:** the number of primes strictly less than n (n ≤ 5·10⁶).
**Brute force:** trial-divide each number up to √k → O(n√n) ≈ 10¹⁰. Dead.
**Bottleneck:** each composite is re-tested from scratch. Instead, *generate* the composites from the primes.
**Aha (sieve):** cross out the multiples of each prime, starting at p². Total work Σ n/p over primes = O(n log log n).
```python
import math

def count_primes(n: int) -> int:
    if n < 3:
        return 0
    is_prime = bytearray([1]) * n                    # indices 0..n-1
    is_prime[0] = is_prime[1] = 0
    for p in range(2, math.isqrt(n - 1) + 1):
        if is_prime[p]:
            is_prime[p * p::p] = bytes(len(range(p * p, n, p)))   # slice assignment: fast in Python
    return sum(is_prime)
```
**Trace** n=10 → primes 2,3,5,7 → 4 ✓.
**Complexity:** O(n log log n) time, O(n) space.
**Edge cases:** n ≤ 2 → 0 (strictly less than n).

### Example 2 — Unique Paths via combinatorics (LC 62) — Medium
**DP solution:** O(m·n) (Ch 17).
**Aha (counting arrangements):** every path is a sequence of (m−1) downs and (n−1) rights in some order. The count of orderings is C(m+n−2, m−1).
```python
import math

def unique_paths(m: int, n: int) -> int:
    return math.comb(m + n - 2, m - 1)
```
**Complexity:** O(min(m, n)) multiplications on big ints. O(1) space.
**When DP is still needed:** obstacles (LC 63), weights, or mod constraints with forbidden cells. Say this in the interview: "the closed form breaks as soon as the grid is not uniform."

### Example 3 — Nth Magical Number (LC 878) — Hard
**Problem:** the n-th positive integer divisible by a or b, mod 10⁹+7. n ≤ 10⁹, a, b ≤ 4·10⁴.
**Brute force:** walk the integers counting magical ones → O(n · min(a,b)). Dead.
**Aha (counting function + binary search on the value):** `count(x)` = the number of magical numbers ≤ x = `x//a + x//b − x//lcm(a,b)` (inclusion-exclusion). It's monotone in x, so binary search for the smallest x with count ≥ n.
```python
import math

def nth_magical_number(n: int, a: int, b: int) -> int:
    MOD = 10**9 + 7
    l = a // math.gcd(a, b) * b
    lo, hi = min(a, b), n * min(a, b)                # the n-th multiple of min(a,b) is always magical
    while lo < hi:
        mid = (lo + hi) // 2
        if mid // a + mid // b - mid // l >= n:
            hi = mid
        else:
            lo = mid + 1
    return lo % MOD                                  # take the mod only at the end
```
**Trace** n=4, a=2, b=3 → magical: 2,3,4,6 → 6. The search finds the smallest x with count(x) ≥ 4: count(6) = 3+2−1 = 4 ✓, count(5) = 2+1−0 = 3.
**Complexity:** O(log(n · min(a,b))) time, O(1) space.
**Edge cases:** a == b (lcm = a, handled). Applying the mod *during* the binary search would break the ordering. Only take it at the end.

## 6. Common mistakes & traps
- Taking `% MOD` inside comparisons or binary searches (it destroys the ordering). Apply it to the final value or to accumulated counts only.
- Division under a modulus: `(a / b) % M` is wrong. Use `a * pow(b, M−2, M) % M` (M prime).
- Negative mods in other languages. Python's `%` is non-negative for a positive modulus.
- Floating-point slopes/equality (`dy/dx`) → store `(dy/g, dx/g)` with a normalized sign.
- `math.sqrt` precision for large n → use `math.isqrt`.
- Sieve bounds: "less than n" vs "≤ n".
- `lcm(a, b) = a*b // gcd` can be huge in other languages. Divide first.
- Reservoir sampling with `random.random() < 1/i` is fine; `randint(0, i)` is off by one.

## 7. Connections
- **Binary search** (Ch 6): k-th element via a counting function (magical numbers, multiplication table, ugly number III).
- **DP** (Ch 17): counting problems mod 10⁹+7; Pascal's triangle is DP for C(n, k).
- **Bit manipulation** (Ch 19): fast exponentiation iterates over the exponent's bits; subsets ↔ 2ⁿ.
- **Hashing**: normalized-slope keys (Max Points on a Line), GCD-reduced fraction keys.
- **Prefix sums**: weighted random. **Heap**: Ugly Number II (or a 3-pointer DP).
- **Union-Find**: union numbers by shared prime factors (Largest Component Size by Common Factor).

## 8. Practice ladder

| # | Problem | Diff | Signal hint |
|---|---|---|---|
| 9 | [Palindrome Number](https://leetcode.com/problems/palindrome-number/) | Easy | Digit manipulation; reverse half |
| 1071 | [Greatest Common Divisor of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings/) | Easy | s+t == t+s ⇒ the answer has length gcd(len) |
| 202 | [Happy Number](https://leetcode.com/problems/happy-number/) | Easy | A digit function iterated → cycle detection |
| 263 | [Ugly Number](https://leetcode.com/problems/ugly-number/) | Easy | Divide out 2, 3, 5 |
| 7 | [Reverse Integer](https://leetcode.com/problems/reverse-integer/) | Medium | Digits + 32-bit overflow check |
| 50 | [Pow(x, n)](https://leetcode.com/problems/powx-n/) | Medium | Huge exponent → binary exponentiation |
| 204 | [Count Primes](https://leetcode.com/problems/count-primes/) | Medium | All primes below n → sieve |
| 172 | [Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/) | Medium | Count factors of 5 |
| 43 | [Multiply Strings](https://leetcode.com/problems/multiply-strings/) | Medium | Grade-school multiplication into a digits array |
| 264 | [Ugly Number II](https://leetcode.com/problems/ugly-number-ii/) | Medium | Merge three multiplied sequences (3 pointers) |
| 528 | [Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/) | Medium | Weights → prefix sums + bisect |
| 382 | [Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/) | Medium | Unknown length, uniform → reservoir sampling |
| 470 | [Implement Rand10() Using Rand7()](https://leetcode.com/problems/implement-rand10-using-rand7/) | Medium | Build uniform 1..49, reject > 40 |
| 372 | [Super Pow](https://leetcode.com/problems/super-pow/) | Medium | Exponent given as a digit array → modular power per digit |
| 149 | [Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/) | Hard | Slopes as GCD-normalized pairs |
| 878 | [Nth Magical Number](https://leetcode.com/problems/nth-magical-number/) | Hard | k-th in an implicit set → inclusion-exclusion + BS |

## 9. Recognition drill
1. Count the number of ways to climb n stairs with steps of 1 or 2, n ≤ 10⁵, mod 10⁹+7.
2. Given a and b up to 10⁹, find the number of integers in [1, N] divisible by neither a nor b.
3. Given an array of up to 10⁵ numbers ≤ 10⁶, answer, for each, its number of distinct prime factors.
4. Pick a uniformly random point inside a circle.
5. Count the number of structurally unique BSTs with n nodes.

<details><summary>Answers</summary>

1. **1D DP with mod** (Fibonacci), or matrix exponentiation for n ≤ 10¹⁸.
2. **Inclusion-exclusion**: N − (N//a + N//b − N//lcm(a,b)).
3. **Smallest-prime-factor sieve** up to 10⁶, then factor each number in O(log v).
4. **Rejection sampling** in the bounding square (or r = R·√U, θ = 2πV). The √ is needed for uniformity.
5. **Catalan number** via the DP `G(n) = Σ G(i−1)·G(n−i)` or the closed form C(2n, n)/(n+1) (LC 96).
</details>

Next: Part 21 — Advanced Topics
