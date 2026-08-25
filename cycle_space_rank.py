#!/usr/bin/env python3
"""cycle_space_rank.py -- deterministic GF(2) verification that the squares
(4-cycles) of Q_n generate its cycle space, for every n used in the paper.

For each n in 3..8 this script:
  * enumerates the edges of Q_n and assigns each a bit position;
  * enumerates all C(n,2) * 2^(n-2) squares of Q_n and encodes each as the
    GF(2) indicator vector of its four edges (a Python integer bitmask);
  * computes the GF(2) rank of the resulting square-vector family by
    Gaussian elimination on bitmasks;
  * asserts that this rank equals the cycle-space dimension
    |E| - |V| + 1 = n*2^(n-1) - 2^n + 1 = (n-2)*2^(n-1) + 1.

Since the span of the square vectors is a subspace of the cycle space
(each square is a cycle), rank equality proves that the squares generate
the entire cycle space of Q_n over GF(2) -- the fact used in the proof of
the switching lemma.

Standard library only; no randomness; runs in seconds.
Exit status 0 iff every n in 3..8 passes.
"""
import sys


def edge_slots(n):
    slot = {}
    for d in range(n):
        for v in range(1 << n):
            u = v ^ (1 << d)
            if v < u:
                slot[(v, u)] = len(slot)
    return slot


def square_vectors(n, slot):
    vecs = []
    for i in range(n):
        for j in range(i + 1, n):
            for v in range(1 << n):
                if (v >> i & 1) or (v >> j & 1):
                    continue
                a, b = v ^ (1 << i), v ^ (1 << j)
                z = a ^ (1 << j)
                mask = 0
                for x, y in ((v, a), (v, b), (a, z), (b, z)):
                    p, q = (x, y) if x < y else (y, x)
                    mask |= 1 << slot[(p, q)]
                vecs.append(mask)
    return vecs


def gf2_rank(vecs):
    pivots = []  # basis rows, each with a distinct leading bit
    for v in vecs:
        for p in pivots:
            v = min(v, v ^ p)
        if v:
            pivots.append(v)
            pivots.sort(reverse=True)
    return len(pivots)


def main():
    ok = True
    for n in range(3, 9):
        slot = edge_slots(n)
        m = len(slot)                      # n * 2^(n-1)
        nv = 1 << n                        # 2^n
        expected = m - nv + 1              # cycle-space dimension
        vecs = square_vectors(n, slot)
        nsq = (n * (n - 1) // 2) * (1 << (n - 2))
        assert len(vecs) == nsq, (n, len(vecs), nsq)
        r = gf2_rank(vecs)
        status = "OK" if r == expected else "MISMATCH"
        print(f"n={n}: |E|={m}, |V|={nv}, squares={nsq}, "
              f"rank(squares)={r}, dim cycle space=|E|-|V|+1={expected}  [{status}]")
        if r != expected:
            ok = False
    if ok:
        print("RESULT: squares generate the cycle space of Q_n for every "
              "n=3..8 (rank equals E-V+1 in every case)")
        return 0
    print("RESULT: MISMATCH against the cycle-space dimension")
    return 1


if __name__ == "__main__":
    sys.exit(main())
