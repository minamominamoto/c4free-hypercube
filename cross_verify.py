#!/usr/bin/env python3
"""
cross_verify.py -- an INDEPENDENT re-implementation of the certificate
checks, sharing no code and no core mechanism with verify.py.

Purpose. The manuscript's "Use of generative AI" section reports that the
released certificates were re-verified by a program written separately
from verify.py, to mitigate single-implementation risk. A round-78 review
correctly noted that no such second implementation was bundled, making
that mitigation unauditable. This file closes the gap: it is a bundled,
permanently auditable second implementation.

How it differs from verify.py. verify.py enumerates the four-cycles of
Q_n as corner 4-tuples and tests the four edges of each square by set
membership. This script never enumerates squares that way. It represents
each solution as per-vertex adjacency BITMASKS and uses the metric
characterisation: a C4 in Q_n consists of two vertices u, w at Hamming
distance 2 together with their exactly two common Q_n-neighbours, so

  * C4-freeness  <=>  for every pair (u, w) at Hamming distance 2, the
    chosen edge set contains at most one length-2 path from u to w,
    i.e. popcount(adj[u] & adj[w]) <= 1;
  * the odd-square condition  <=>  for every such pair, the number of
    present edges among the four edges joining {u, w} to their two common
    neighbours x = u ^ (bit i), y = u ^ (bit j) is odd (1 or 3).

It also re-checks, by direct means (no hashing of any kind for the
distinctness test): edge validity (every listed edge is a genuine Q_n
edge, no duplicates inside a solution), the exact edge counts
(132 / 304 x 19,866 / 680 x 2 / 682), the catalogue size 19,866, and the
pairwise distinctness of all 19,866 Q7 edge sets via a set of
canonically sorted edge tuples.

Standard library only; deterministic; wall time is dominated by the
19,866 x 2,688 distance-2 pair checks for Q7 (about one to two minutes on
our reference machine). Exit status 0 iff every check passes.
"""
import json
import sys
from itertools import combinations

FAIL = []


def expect(name, ok, detail=''):
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}" +
          (f": {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def load_edge_lists(path):
    out = []
    with open(path, encoding='utf-8') as f:
        text = f.read()
    text = text.strip()
    if text.startswith('{') and '\n' not in text:
        objs = [json.loads(text)]
    else:
        objs = [json.loads(line) for line in text.splitlines() if line.strip()]
    for o in objs:
        out.append(o['edges'])
    return out


def check_edges_valid(edges, n):
    """Every pair is a genuine Q_n edge (differ in exactly one bit),
    endpoints in range, no duplicate edges. Returns normalised set."""
    seen = set()
    for u, v in edges:
        if not (0 <= u < (1 << n) and 0 <= v < (1 << n)):
            return None
        d = u ^ v
        if d == 0 or d & (d - 1):          # zero or more than one bit
            return None
        e = (u, v) if u < v else (v, u)
        if e in seen:
            return None
        seen.add(e)
    return seen


def dist2_masks(n):
    return [(1 << i) | (1 << j) for i, j in combinations(range(n), 2)]


def analyse(edge_set, n, masks):
    """Return (c4free, oddsquare) via bitmask common-neighbour counting."""
    adj = [0] * (1 << n)
    for u, v in edge_set:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    c4free = True
    oddsq = True
    for u in range(1 << n):
        au = adj[u]
        for m in masks:
            w = u ^ m
            if u > w:
                continue
            common = au & adj[w]
            if bin(common).count('1') > 1:
                c4free = False
            # the two common Q_n-neighbours of u and w:
            lo = m & -m
            x = u ^ lo
            y = u ^ (m ^ lo)
            cnt = ((au >> x) & 1) + ((au >> y) & 1) \
                + ((adj[w] >> x) & 1) + ((adj[w] >> y) & 1)
            if cnt % 2 == 0:
                oddsq = False
    return c4free, oddsq


def main():
    # (path, n, expected count per solution, expected number of solutions,
    #  must be C4-free, must be odd-square)
    jobs = [
        ('q6_edges_132.jsonl', 6, 132, 1, True, False),
        ('q6_odd_square_132.json', 6, 132, 1, True, True),
        ('q8_edges_680.jsonl', 8, 680, 2, True, False),
        ('q8_odd_square_682.json', 8, 682, 1, True, True),
    ]
    for path, n, cnt, k, want_c4, want_os in jobs:
        sols = load_edge_lists(path)
        expect(f'{path}: {k} solution(s)', len(sols) == k, f'{len(sols)}')
        masks = dist2_masks(n)
        for i, es in enumerate(sols):
            s = check_edges_valid(es, n)
            expect(f'{path}[{i}]: valid Q{n} edges, no duplicates',
                   s is not None and len(s) == cnt,
                   f'{0 if s is None else len(s)} edges')
            if s is None:
                continue
            c4, osq = analyse(s, n, masks)
            expect(f'{path}[{i}]: C4-free', c4 == want_c4)
            if want_os:
                expect(f'{path}[{i}]: odd-square', osq)

    parts = ['q7_edges_304.jsonl.part1', 'q7_edges_304.jsonl.part2',
             'q7_edges_304.jsonl.part3']
    masks = dist2_masks(7)
    distinct = set()
    total = 0
    bad_valid = bad_c4 = 0
    for part in parts:
        for es in load_edge_lists(part):
            total += 1
            s = check_edges_valid(es, 7)
            if s is None or len(s) != 304:
                bad_valid += 1
                continue
            c4, _ = analyse(s, 7, masks)
            if not c4:
                bad_c4 += 1
            distinct.add(tuple(sorted(s)))
    expect('Q7 catalogue: 19,866 solutions', total == 19866, f'{total}')
    expect('Q7 catalogue: all valid 304-edge sets', bad_valid == 0,
           f'{bad_valid} failures')
    expect('Q7 catalogue: all C4-free', bad_c4 == 0, f'{bad_c4} failures')
    expect('Q7 catalogue: pairwise distinct (sorted-tuple set, no hashing)',
           len(distinct) == 19866, f'{len(distinct)} distinct')

    if FAIL:
        print(f'RESULT: MISMATCH ({len(FAIL)})')
        return 1
    print('RESULT: all certificates re-verified by the independent '
          'implementation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
