#!/usr/bin/env python3
"""
q7_oddsquare_orbits.py -- decomposes the 389 odd-square solutions of the
released Q_7 catalogue into Aut(Q_7)-orbits.

An automorphism of Q_n is v -> pi(v) XOR a with pi a coordinate permutation, so
|Aut(Q_7)| = 7! * 2^7 = 645,120. Orbits are found by taking an unassigned
solution, applying every group element to it, and recording which other
catalogue members appear among the images.

Membership is tested on the exact 448-byte incidence vector, NOT on a hash: a
random-projection signature is much faster but can collide, and a single false
positive silently merges two orbits. (An earlier signature-based run of this
computation reported orbit sizes 128/48 where the exact test gives 127/49.)

Requires numpy. Runtime: about a minute. Usage:
    python3 q7_oddsquare_orbits.py
"""
import json
import sys
from itertools import permutations

import numpy as np

N = 7
V = 1 << N
PARTS = ['q7_edges_304.jsonl.part1',
         'q7_edges_304.jsonl.part2',
         'q7_edges_304.jsonl.part3']
ODD_CSV = 'q7_odd_square_389.csv'

EXPECTED_ORBITS = 6
EXPECTED_SIZES = [127, 83, 55, 49, 46, 29]

SLOT = {}
for _d in range(N):
    for _v in range(V):
        _u = _v ^ (1 << _d)
        if _v < _u:
            SLOT[(_v, _u)] = len(SLOT)
NSLOT = len(SLOT)


def slot_perm(pi, a):
    """Column permutation of the 448 edge slots induced by v -> pi(v) XOR a."""
    vmap = [0] * V
    for v in range(V):
        w = 0
        for d in range(N):
            if v >> d & 1:
                w |= 1 << pi[d]
        vmap[v] = w ^ a
    out = np.empty(NSLOT, dtype=np.int32)
    for (x, y), s in SLOT.items():
        p, q = vmap[x], vmap[y]
        out[SLOT[(p, q) if p < q else (q, p)]] = s
    return out


def load_oddsquare():
    idx = set()
    with open(ODD_CSV, encoding='utf-8') as f:
        f.readline()
        for line in f:
            idx.add(int(line.split(',')[0]))
    rows, gi = [], 0
    for part in PARTS:
        with open(part, encoding='utf-8') as f:
            for line in f:
                if gi in idx:
                    es = json.loads(line)['edges']
                    vec = np.zeros(NSLOT, dtype=np.uint8)
                    for u, v in es:
                        a, b = (u, v) if u < v else (v, u)
                        vec[SLOT[(a, b)]] = 1
                    rows.append(vec)
                gi += 1
    return np.array(rows)


def main():
    M = load_oddsquare()
    n = M.shape[0]
    print(f'odd-square solutions loaded: {n}')
    exact = {M[i].tobytes(): i for i in range(n)}
    assert len(exact) == n, 'duplicate solutions in the catalogue'

    trans = np.array([slot_perm(tuple(range(N)), a) for a in range(V)])
    perms = [slot_perm(pi, 0) for pi in permutations(range(N))]
    print(f'group elements: {len(perms) * V}')

    orbit = [-1] * n
    sizes = []
    cur = 0
    for start in range(n):
        if orbit[start] != -1:
            continue
        v = M[start]
        found = {start}
        for pp in perms:
            imgs = v[pp[trans]]
            for r in range(V):
                j = exact.get(imgs[r].tobytes())
                if j is not None:
                    found.add(j)
        for j in found:
            orbit[j] = cur
        sizes.append(len(found))
        cur += 1

    ordered = sorted(sizes, reverse=True)
    print(f'ORBITS: {cur}')
    print(f'sizes within the catalogue: {ordered}')
    print(f'total: {sum(sizes)}')
    ok = (cur == EXPECTED_ORBITS and ordered == EXPECTED_SIZES)
    print('RESULT:', 'matches the paper' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
