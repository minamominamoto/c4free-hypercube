#!/usr/bin/env python3
"""
q6_decide_realizability.py -- EXHAUSTIVE decision, not a heuristic search.

Decides, for each of the three optimal field distributions on U for n=6
returned by solve_field_ip.py, whether it is realisable by some spin
configuration under the canonical fully frustrated coupling.

    A = {2:30, 6:2}
    B = {0:1, 2:27, 4:3, 6:1}
    C = {0:2, 2:24, 4:6}          (realised by q6_odd_square_132.json)

THE REDUCTION
-------------
Write s_v = (-1)^{t_v}, t_v in {0,1}, and let b(v,w)=1 iff J(v,w)=-1.
An edge (v,w) is positive iff  t_v XOR t_w = b(v,w).

Q6 is bipartite: every neighbour of v in U lies in U^c. Define, for v in U,

    k(v) = #{ i : t_{v XOR e_i} XOR b(v,i) = 1 }

which depends ONLY on the 32 spins on U^c. Then

    deg(v) = k(v)      if t_v = 1
    deg(v) = 6 - k(v)  if t_v = 0

so each t_v (v in U) is a free, independent choice. Hence the achievable
degree at v is exactly one of {k(v), 6-k(v)}, and with h = 2*deg - 6 the
achievable NON-NEGATIVE field is h = 6 - 2*m(v) where

    m(v) = min(k(v), 6 - k(v))  in {0,1,2,3}
    m=0 <-> h=6,  m=1 <-> h=4,  m=2 <-> h=2,  m=3 <-> h=0

Therefore a target h-histogram on U is realisable IF AND ONLY IF the
multiset { m(v) : v in U } equals the corresponding m-multiset, for some
assignment of the 32 U^c spins. That is a 2^32 decision problem, made
tractable by DFS with counter-overflow pruning: as soon as all six
neighbours of some v in U are assigned, m(v) is final and can be tallied.

Global spin flip is a symmetry (it fixes every h), so one U^c spin is
pinned; the script also reports the unpinned node count for comparison.

Standard library only.
"""

import argparse
import sys
import time
from collections import Counter

N = 6
V = 1 << N
U = [v for v in range(V) if bin(v).count('1') % 2 == 0]
UC = [v for v in range(V) if bin(v).count('1') % 2 == 1]
UC_INDEX = {v: i for i, v in enumerate(UC)}

# h-histograms of the three optimal distributions (from solve_field_ip.py)
TARGETS_H = {
    'A': {2: 30, 6: 2},
    'B': {0: 1, 2: 27, 4: 3, 6: 1},
    'C': {0: 2, 2: 24, 4: 6},
}


def coupling(x, dim):
    """Canonical fully frustrated coupling on edge x--(x xor 2**dim)."""
    return 1 if bin(x & ((1 << dim) - 1)).count('1') % 2 == 0 else -1


def bbit(v, i):
    """b(v, v xor e_i): 1 iff the coupling on that edge is -1."""
    return 0 if coupling(min(v, v ^ (1 << i)), i) == 1 else 1


def h_to_m(hist_h):
    """Convert a non-negative h-histogram to the m-multiset it forces."""
    out = Counter()
    for h, c in hist_h.items():
        if h < 0 or h % 2 or h > N:
            raise ValueError(f'bad h={h}')
        out[(N - h) // 2] += c
    return dict(out)


# Precompute, for each v in U, the list of (uc_index, b_bit) over its 6 edges.
U_NBR = []
for v in U:
    U_NBR.append([(UC_INDEX[v ^ (1 << i)], bbit(v, i)) for i in range(N)])

# For each U^c position, which U-vertices become FINAL once it is assigned
# (i.e. that position is the last of their six neighbours in our ordering).
LAST_POS = [[] for _ in range(len(UC))]
for ui, nbrs in enumerate(U_NBR):
    LAST_POS[max(p for p, _ in nbrs)].append(ui)


def decide(target_m, pin_first=True):
    """Exhaustive DFS. Returns (realisable, witness_t_or_None, nodes)."""
    need = [target_m.get(m, 0) for m in range(4)]
    if sum(need) != len(U):
        raise ValueError('target multiset size != |U|')

    t = [0] * len(UC)
    have = [0, 0, 0, 0]
    nodes = 0
    found = [None]

    def rec(pos):
        nonlocal nodes
        if found[0] is not None:
            return True
        if pos == len(UC):
            return have == need
        for val in ((0,) if (pin_first and pos == 0) else (0, 1)):
            t[pos] = val
            nodes += 1
            finalized = []
            ok = True
            for ui in LAST_POS[pos]:
                k = 0
                for p, bb in U_NBR[ui]:
                    k += t[p] ^ bb
                m = min(k, N - k)
                have[m] += 1
                finalized.append(m)
                if have[m] > need[m]:
                    ok = False
                    break
            if ok and rec(pos + 1):
                if found[0] is None:
                    found[0] = t[:]
                return True
            for m in finalized:
                have[m] -= 1
        return False

    res = rec(0)
    return res, found[0], nodes


def spins_to_hist(t_uc, target_m):
    """Rebuild full spins (choosing h>=0 at each U vertex) and report the
    h-histogram, edge count and square histogram -- an independent check
    that a reported witness really realises the target."""
    t_full = [0] * V
    for i, v in enumerate(UC):
        t_full[v] = t_uc[i]
    for ui, v in enumerate(U):
        k = sum(t_uc[p] ^ bb for p, bb in U_NBR[ui])
        # choose t_v so that deg = max(k, 6-k)  =>  h >= 0
        t_full[v] = 1 if k >= N - k else 0
    s = [1 if x == 0 else -1 for x in t_full]
    edges = []
    deg = [0] * V
    for x in range(V):
        for d in range(N):
            y = x ^ (1 << d)
            if x < y and coupling(x, d) * s[x] * s[y] == 1:
                edges.append((x, y))
                deg[x] += 1
                deg[y] += 1
    hU = Counter(2 * deg[v] - N for v in U)
    return dict(sorted(hU.items())), len(edges), t_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='also run the unpinned search (doubles the work; '
                         'the node count is exactly twice the pinned one)')
    args = ap.parse_args()
    full = args.full
    print('Exhaustive realizability decision for the n=6 optimal field '
          'distributions')
    print('(canonical fully frustrated coupling; 2^32 spin space on U^c, '
          'DFS with pruning)\n')
    results = {}
    for name in ('A', 'B', 'C'):
        hist_h = TARGETS_H[name]
        target_m = h_to_m(hist_h)
        t0 = time.time()
        ok, witness, nodes = decide(target_m, pin_first=True)
        elapsed = time.time() - t0
        if full:
            _, _, nodes_full = decide(target_m, pin_first=False)
        else:
            nodes_full = None
        verdict = 'REALISABLE' if ok else 'NOT REALISABLE'
        print(f'{name} h={hist_h}')
        print(f'   m-multiset {target_m}')
        print(f'   nodes visited (one spin pinned): {nodes}  '
              f'[{elapsed:.1f}s]')
        if nodes_full is not None:
            print(f'   nodes visited (no symmetry reduction): {nodes_full}')
        else:
            print(f'   (unpinned cross-check skipped; use --full. Expected '
                  f'node count: {2*nodes})')
        print(f'   VERDICT: {verdict}')
        if ok:
            hh, ne, _ = spins_to_hist(witness, target_m)
            print(f'   witness check: h-histogram {hh}, {ne} edges')
            assert hh == {k: v for k, v in sorted(hist_h.items())}, 'mismatch'
        print()
        results[name] = ok

    print('SUMMARY:',
          ', '.join(f'{k}={"yes" if v else "no"}' for k, v in results.items()))
    # Expected: only C realisable
    if results == {'A': False, 'B': False, 'C': True}:
        print('RESULT: only distribution C is realisable (A and B are '
              'proved NOT realisable by exhaustive search).')
        return 0
    print('RESULT: UNEXPECTED -- differs from the paper.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
