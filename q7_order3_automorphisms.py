#!/usr/bin/env python3
"""
q7_order3_automorphisms.py -- verifies that EVERY one of the 19,866 released
304-edge C4-free Q_7 solutions admits an automorphism of order 3.

Method (the same block reduction already used by type18_automorphisms.py,
generalised to all 20 dimension profiles): an automorphism of Q_n is
v -> pi(v) XOR a with pi a coordinate permutation. Such a map sends an edge in
direction d to an edge in direction pi(d), so it preserves the multiset of
per-direction edge counts. Hence only permutations preserving the solution's
own direction-count vector can stabilise it, and for a profile with repeated
counts those form the product of symmetric groups on the equal-count blocks.
The search is therefore exhaustive over

    (block-preserving permutations) x (128 translations)

which is 18,432 elements for profile [48,48,48,40,40,40,40] and far fewer for
profiles with distinct counts, instead of the full |Aut(Q_7)| = 645,120.

Everything is vectorised: solutions of one profile are held as a
(k x 448) uint8 matrix, and applying a group element is a single column
permutation, so testing one element against every solution of that profile is
one numpy comparison.

An automorphism of order 3 must have pi of order 3 (if pi were the identity
the map is v -> v XOR a, of order 2), so only order-3 coordinate permutations
preserving the solution's own direction-count vector need be tried. Solutions
are grouped by that vector, and for each group the candidate elements are
tested as vectorised column permutations.

Requires numpy. Runtime: about a minute. Usage:
    python3 q7_order3_automorphisms.py
"""
import argparse
import json
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np

N = 7
V = 1 << N
PARTS = ['q7_edges_304.jsonl.part1',
         'q7_edges_304.jsonl.part2',
         'q7_edges_304.jsonl.part3']


def edge_slots():
    idx = {}
    for d in range(N):
        for v in range(V):
            u = v ^ (1 << d)
            if v < u:
                idx[(v, u)] = len(idx)
    return idx


SLOT = edge_slots()
NSLOT = len(SLOT)


def slot_perm(pi, a):
    """Column permutation induced by v -> pi(v) XOR a.

    pi is a tuple with pi[d] = image direction of direction d.
    """
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


def load():
    sols, dirs = [], []
    for part in PARTS:
        with open(part, encoding='utf-8') as f:
            for line in f:
                es = json.loads(line)['edges']
                vec = np.zeros(NSLOT, dtype=np.uint8)
                dc = [0] * N
                for u, v in es:
                    a, b = (u, v) if u < v else (v, u)
                    vec[SLOT[(a, b)]] = 1
                    dc[(u ^ v).bit_length() - 1] += 1
                sols.append(vec)
                dirs.append(tuple(dc))
    return np.array(sols), dirs


def block_perms(dc):
    """Permutations of the 7 directions preserving the count vector dc."""
    groups = defaultdict(list)
    for d, c in enumerate(dc):
        groups[c].append(d)
    blocks = list(groups.values())
    out = []

    def rec(i, partial):
        if i == len(blocks):
            pi = [0] * N
            for src, dst in partial:
                pi[src] = dst
            out.append(tuple(pi))
            return
        blk = blocks[i]
        for perm in permutations(blk):
            rec(i + 1, partial + list(zip(blk, perm)))

    rec(0, [])
    return out



def order3_perms(dc):
    """Order-3 permutations preserving the direction-count vector dc."""
    groups = defaultdict(list)
    for d, c in enumerate(dc):
        groups[c].append(d)
    blocks = list(groups.values())
    ident = tuple(range(N))
    out = []

    def rec(i, partial):
        if i == len(blocks):
            pi = [0] * N
            for src, dst in partial:
                pi[src] = dst
            pi = tuple(pi)
            p2 = tuple(pi[pi[d]] for d in range(N))
            p3 = tuple(pi[p2[d]] for d in range(N))
            if p3 == ident and pi != ident:
                out.append(pi)
            return
        blk = blocks[i]
        for perm in permutations(blk):
            rec(i + 1, partial + list(zip(blk, perm)))

    rec(0, [])
    return out


def main():
    M, dirs = load()
    n = M.shape[0]
    print(f'loaded {n} solutions')

    by_dc = defaultdict(list)
    for i, d in enumerate(dirs):
        by_dc[d].append(i)
    print(f'distinct direction-count vectors: {len(by_dc)}')

    has3 = np.zeros(n, dtype=bool)
    no_candidate = 0
    for dc, idxs in by_dc.items():
        cands = order3_perms(dc)
        if not cands:
            no_candidate += len(idxs)
            continue
        rows = np.array(idxs)
        block = M[rows]
        acc = np.zeros(len(idxs), dtype=bool)
        for pi in cands:
            for a in range(V):
                acc |= (block[:, slot_perm(pi, a)] == block).all(axis=1)
            if acc.all():
                break
        has3[rows] = acc

    print(f'count vectors admitting no order-3 permutation '
          f'(solutions): {no_candidate}')
    print(f'solutions WITH an order-3 automorphism: {int(has3.sum())}/{n}')
    print(f'solutions WITHOUT: {int((~has3).sum())}')
    ok = bool(has3.all())
    print('RESULT:', 'every released solution has an order-3 automorphism'
          if ok else 'MISMATCH - some solution has none')
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
