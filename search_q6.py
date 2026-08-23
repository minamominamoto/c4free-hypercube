#!/usr/bin/env python3
"""Search for a 132-edge odd-square subgraph of Q6.

Method: the canonical fully frustrated coupling J(x, x^2^dim) =
(-1)^popcount(x & (2^dim - 1)) makes every square of Q_n have edge-sign
product -1.  For ANY spin assignment s, the set of edges on which
J(x,y) s_x s_y = +1 then meets every square in an odd number of edges
(switching preserves frustration), i.e. it is odd-square.  Maximising
that edge count is exactly the fully frustrated ground-state problem.

We solve it by seeded simulated annealing over the 64 spins.  A fixed
seed is used so the run is exactly reproducible.

Standard library only.
"""

import json
import random
from collections import Counter
from itertools import combinations

N = 6
V = 1 << N
E_TOTAL = N * (1 << (N - 1))          # 192
TARGET = 132
SEED = 20260823


def coupling(x, dim):
    """Canonical fully frustrated coupling on edge x--(x xor 2**dim)."""
    return 1 if bin(x & ((1 << dim) - 1)).count('1') % 2 == 0 else -1


def check_fully_frustrated():
    """Every square must have coupling product -1."""
    bad = 0
    for d1, d2 in combinations(range(N), 2):
        m1, m2 = 1 << d1, 1 << d2
        for base in range(V):
            if base & (m1 | m2):
                continue
            p = (coupling(base, d1) * coupling(base, d2)
                 * coupling(base | m1, d2) * coupling(base | m2, d1))
            if p != -1:
                bad += 1
    return bad


# adjacency with couplings, precomputed
NBR = [[(x ^ (1 << d), coupling(min(x, x ^ (1 << d)), d)) for d in range(N)]
       for x in range(V)]


def satisfied(s):
    """Number of edges with J*s_x*s_y = +1."""
    t = 0
    for x in range(V):
        for y, j in NBR[x]:
            if x < y and j * s[x] * s[y] == 1:
                t += 1
    return t


def local_field(s, x):
    """Sum over neighbours of J*s_y ; flipping x changes satisfied count by -h*s_x."""
    return sum(j * s[y] for y, j in NBR[x])


def anneal(rng, iters=200000):
    s = [rng.choice((1, -1)) for _ in range(V)]
    cur = satisfied(s)
    best, best_s = cur, s[:]
    for it in range(iters):
        T = max(0.02, 2.0 * (1.0 - it / iters))
        x = rng.randrange(V)
        delta = -s[x] * local_field(s, x)      # change in satisfied count
        if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
            s[x] = -s[x]
            cur += delta
            if cur > best:
                best, best_s = cur, s[:]
                if best == TARGET:
                    return best, best_s
    return best, best_s


def main():
    bad = check_fully_frustrated()
    print(f'fully frustrated check (n={N}): {bad} squares with product != -1'
          f'  -> {"OK" if bad == 0 else "FAIL"}')

    rng = random.Random(SEED)
    for trial in range(1, 200):
        best, s = anneal(rng)
        if best >= TARGET:
            print(f'trial {trial}: reached {best} satisfied edges')
            break
        if trial % 20 == 0:
            print(f'  trial {trial}: best so far {best}')
    else:
        print('target not reached')
        return

    # normalise: global flip is a symmetry; fix s[0] = +1
    if s[0] == -1:
        s = [-t for t in s]

    spins = ''.join('1' if t == 1 else '0' for t in s)
    print('SPINS =', spins)
    with open('q6_spins.txt', 'w') as f:
        f.write(spins + '\n')


if __name__ == '__main__':
    main()
