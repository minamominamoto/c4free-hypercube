#!/usr/bin/env python3
"""
q6_parity_obstruction.py -- a linear-algebra proof, over GF(2), that the
optimal field distributions A and B for n=6 are not realisable.

This supersedes the exhaustive 2^32 search of q6_decide_realizability.py for
the two negative cases: the conclusion follows from a rank computation and a
weight enumeration, both small enough to check by hand-waving-free machine
arithmetic in under a second.

THE ARGUMENT
------------
With s_v = (-1)^{t_v} and b(v,w) = 1 exactly when J(v,w) = -1, an edge vw is
positive iff t_v XOR t_w = b(v,w). For v in U write

    k(v) = #{ i : t_{v XOR e_i} XOR b(v,i) = 1 },

which depends only on the spins on U^c. Then deg(v) is k(v) or n - k(v)
according to t_v, so h(v) = 2 deg(v) - n equals 2k(v) - n or n - 2k(v). In
either case

    k(v) = (h(v) + n)/2   or   (n - h(v))/2,

and those two differ by h(v), which is even; so the PARITY of k(v) is
determined by h(v) alone, independently of the free spin t_v:

    k(v) = (h(v) + n)/2   (mod 2).

For n = 6 this reads: k(v) is odd exactly when h(v) is in {0, 4}.
(For n = 8 it reads: k(v) is odd exactly when h(v) is in {2, 6}.)

Reducing the definition of k(v) mod 2,

    k(v) = sum_i t_{v XOR e_i} + sum_i b(v,i)   (mod 2),

so the parity vector p in GF(2)^U is the affine image p = M t + B, where M is
the U x U^c bipartite incidence matrix of Q_n over GF(2) and B(v) = sum_i
b(v,i). The reachable parity vectors are therefore exactly the coset
B + Im(M), and a target h-histogram is realisable only if the number of its
vertices with k odd -- a number the histogram determines -- occurs as a weight
in that coset.

For n = 6, M has GF(2) rank 16, and the coset B + Im(M) has weight
distribution {8: 640, 12: 13824, 16: 36608, 20: 13824, 24: 640}. Distribution
A = {2:30, 6:2} has no vertex with h in {0,4}, so it needs weight 0;
B = {0:1, 2:27, 4:3, 6:1} needs weight 4. Neither occurs. C = {0:2, 2:24, 4:6}
needs weight 8, which does occur -- and the released witness realises it.

Standard library only. Usage:
    python3 q6_parity_obstruction.py
"""
import sys
from collections import Counter

EXPECTED_RANK = 16
EXPECTED_WEIGHTS = {8: 640, 12: 13824, 16: 36608, 20: 13824, 24: 640}
TARGETS = {
    'A': {2: 30, 6: 2},
    'B': {0: 1, 2: 27, 4: 3, 6: 1},
    'C': {0: 2, 2: 24, 4: 6},
}


def coupling(x, dim):
    return 1 if bin(x & ((1 << dim) - 1)).count('1') % 2 == 0 else -1


def bbit(v, i):
    return 0 if coupling(min(v, v ^ (1 << i)), i) == 1 else 1


def build(n):
    V = 1 << n
    U = [v for v in range(V) if bin(v).count('1') % 2 == 0]
    W = [v for v in range(V) if bin(v).count('1') % 2 == 1]
    widx = {w: i for i, w in enumerate(W)}
    cols = [0] * len(W)          # each column a bitmask over U
    const = 0
    for r, v in enumerate(U):
        acc = 0
        for i in range(n):
            cols[widx[v ^ (1 << i)]] ^= 1 << r
            acc ^= bbit(v, i)
        if acc:
            const |= 1 << r
    return U, cols, const


def image_basis(cols):
    basis = []
    for c in cols:
        x = c
        for bb in basis:
            x = min(x, x ^ bb)
        if x:
            basis.append(x)
            basis.sort(reverse=True)
    return basis


def parity_weight(hist, n):
    """Number of vertices whose k is odd, i.e. with (h+n)/2 odd."""
    return sum(c for h, c in hist.items() if ((h + n) // 2) % 2 == 1)


def main():
    n = 6
    U, cols, const = build(n)
    basis = image_basis(cols)
    rank = len(basis)
    print(f'n={n}: |U|={len(U)}, GF(2) rank of the U x U^c incidence map '
          f'= {rank}')

    coset = [0]
    for bb in basis:
        coset += [x ^ bb for x in coset]
    wd = Counter(bin(const ^ x).count('1') for x in coset)
    print(f'coset B + Im(M): {len(coset)} vectors, weight distribution '
          f'{dict(sorted(wd.items()))}')

    ok = (rank == EXPECTED_RANK and dict(wd) == EXPECTED_WEIGHTS)
    for name, hist in TARGETS.items():
        need = parity_weight(hist, n)
        reachable = need in wd
        print(f'  {name} = {hist}: needs weight {need} -> '
              f'{"reachable" if reachable else "IMPOSSIBLE"}')
        if name in ('A', 'B') and reachable:
            ok = False
        if name == 'C' and not reachable:
            ok = False

    # Rank sweep n=2..9: the sequence 1,4,4,16,16,64,64,256 is stated in
    # the paper's structural discussion of the incidence map; computed here
    # with the same generic machinery so the full sequence is re-checkable
    # from the released code (closes a re-checkability gap noted in review).
    SWEEP_EXPECTED = [1, 4, 4, 16, 16, 64, 64, 256]
    sweep = []
    for m in range(2, 10):
        _, cols_m, _ = build(m)
        sweep.append(len(image_basis(cols_m)))
    print(f'rank sweep n=2..9: {sweep} (expected {SWEEP_EXPECTED})')
    if sweep != SWEEP_EXPECTED:
        ok = False

    print('RESULT:', 'matches the paper' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
