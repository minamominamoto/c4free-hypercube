#!/usr/bin/env python3
"""
solve_field_ip.py -- exact solution of the field-bound integer programme.

The paper's Section "Exact values for n=6,7,8" bounds g(n) above by

    maximise   S = sum_{v in U} h(v)
    subject to     sum_{v in U} h(v)^2 = n * 2^(n-1),
                   h(v) integer, h(v) = n (mod 2), |h(v)| <= n,
                   |U| = 2^(n-1)

and solves it in closed form via the quantised two-point inequality
(Lemma "Quantised two-point inequality"): for a, b = a+2 of the same
parity as n, (h-a)(h-b) is a nonnegative multiple of 8.

This script solves the same programme exhaustively and independently, by
dynamic programming over the sum-of-squares budget, and enumerates every
optimal h-distribution. It is a check on the closed-form argument, not a
substitute for it. No sign restriction is imposed: negative values of the
correct parity are in the domain throughout.

Standard library only (functools, sys). Python 3.8+.

Usage:
    python3 solve_field_ip.py

Expected output (also asserted, so the exit code is 0 iff it holds):

    n=6:  S <= 72   -> g(6) <= 132     3 optimal distributions
    n=7:  S <= 160  -> g(7) <= 304     1 optimal distribution
    n=8:  S <= 340  -> g(8) <= 682     2 optimal distributions
"""

import sys
from functools import lru_cache

NEG = -10 ** 9

# n -> (|E(Q_n)|, closed-form optimum from the paper, number of optimal
#       distributions, the two-point pair (a, b) used in the paper)
EXPECTED = {
    6: (192, 72, 3, (2, 4)),
    7: (448, 160, 1, (1, 3)),
    8: (1024, 340, 2, (2, 4)),
}


def solve(n):
    """Return (optimum S, list of optimal distributions as {h: count})."""
    slots = 1 << (n - 1)                       # |U|
    budget = n * (1 << (n - 1))                # required sum of squares
    vals = [h for h in range(-n, n + 1) if (h - n) % 2 == 0]

    sys.setrecursionlimit(10000)

    @lru_cache(maxsize=None)
    def best(i, k, s):
        """Max sum of h using values vals[i:], k slots left, s budget left."""
        if i == len(vals):
            return 0 if (k == 0 and s == 0) else NEG
        h = vals[i]
        sq = h * h
        top = k if sq == 0 else min(k, s // sq)
        out = NEG
        for c in range(top + 1):
            r = best(i + 1, k - c, s - c * sq)
            if r > NEG and r + c * h > out:
                out = r + c * h
        return out

    opt = best(0, slots, budget)
    sols = []

    def walk(i, k, s, acc, chosen):
        if i == len(vals):
            if k == 0 and s == 0 and acc == opt:
                sols.append(dict(chosen))
            return
        h = vals[i]
        sq = h * h
        top = k if sq == 0 else min(k, s // sq)
        for c in range(top + 1):
            if best(i + 1, k - c, s - c * sq) + acc + c * h == opt:
                if c:
                    chosen.append((h, c))
                walk(i + 1, k - c, s - c * sq, acc + c * h, chosen)
                if c:
                    chosen.pop()

    walk(0, slots, budget, 0, [])
    return opt, sols


def two_point_real_bound(n, a, b):
    """The real (non-quantised) bound S <= (Q + a*b*m) / (a+b)."""
    m = 1 << (n - 1)
    q = n * m
    return (q + a * b * m) / (a + b)


def main():
    ok = True
    for n in (6, 7, 8):
        edges, want_opt, want_count, (a, b) = EXPECTED[n]
        opt, sols = solve(n)
        real = two_point_real_bound(n, a, b)
        delta = n * (1 << (n - 1)) - (a + b) * opt + a * b * (1 << (n - 1))
        print("n=%d   |U|=%d   sum h^2 = %d   pair (a,b)=(%d,%d)"
              % (n, 1 << (n - 1), n * (1 << (n - 1)), a, b))
        print("   real two-point bound      S <= %.4f" % real)
        print("   EXACT integer optimum     S  = %d   -> g(%d) <= %d"
              % (opt, n, (edges + opt) // 2))
        print("   residual Delta at optimum    = %d  (must be a "
              "nonnegative multiple of 8)" % delta)
        print("   optimal h-distributions: %d" % len(sols))
        for d in sols:
            print("      %s" % dict(sorted(d.items())))
        if opt != want_opt or len(sols) != want_count:
            print("   MISMATCH against the paper's stated values")
            ok = False
        if delta < 0 or delta % 8:
            print("   MISMATCH: Delta is not a nonnegative multiple of 8")
            ok = False
        print()
    print("RESULT:", "MATCHES THE PAPER" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
