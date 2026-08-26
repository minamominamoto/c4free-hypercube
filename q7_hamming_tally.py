#!/usr/bin/env python3
"""
q7_hamming_tally.py -- reproduces the Section "Solution landscape" pairwise
Hamming statistics for the 19,866 released 304-edge Q7 solutions, including
the minimum-distance breakdown that analyze_q7_structure.py does not emit.

For two 304-edge sets, |E xor E'| = 2*(304 - |E and E'|), so all
C(19866,2) = 197,319,045 pairwise distances follow from one matrix product
of the 19866 x 448 incidence matrix with its transpose, computed in blocks.
With numpy this takes a few seconds.

Outputs: min, max, mean, median over all pairs; the number of pairs attaining
the minimum; and the breakdown of those pairs by dimension-profile type
(the ranks of Table "Dimension-profile classification").

Requires numpy (see requirements.txt). Usage:

    python3 q7_hamming_tally.py
    python3 q7_hamming_tally.py --csv q7_min_distance_pairs.csv
"""
import argparse
import json
from collections import Counter

import numpy as np

N = 7
EDGE_COUNT = 304
PARTS = ['q7_edges_304.jsonl.part1',
         'q7_edges_304.jsonl.part2',
         'q7_edges_304.jsonl.part3']

# Dimension profiles, in the rank order of the paper's classification table.
PROFILES = {
    1: (44, 44, 44, 44, 43, 43, 42),   2: (45, 45, 45, 43, 43, 42, 41),
    3: (45, 45, 45, 43, 43, 43, 40),   4: (44, 44, 44, 44, 44, 43, 41),
    5: (46, 46, 46, 42, 42, 42, 40),   6: (46, 46, 46, 42, 42, 41, 41),
    7: (44, 44, 44, 44, 44, 44, 40),   8: (45, 45, 45, 43, 42, 42, 42),
    9: (45, 45, 44, 43, 43, 43, 41),  10: (44, 44, 44, 44, 44, 42, 42),
    11: (47, 47, 47, 41, 41, 41, 40), 12: (45, 45, 43, 43, 43, 43, 42),
    13: (45, 44, 44, 43, 43, 43, 42), 14: (46, 45, 45, 42, 42, 42, 42),
    15: (44, 44, 44, 43, 43, 43, 43), 16: (47, 47, 44, 43, 41, 41, 41),
    17: (46, 46, 44, 42, 42, 42, 42), 18: (48, 48, 48, 40, 40, 40, 40),
    19: (46, 46, 43, 43, 42, 42, 42), 20: (46, 46, 45, 42, 42, 42, 41),
}
RANK_OF = {v: k for k, v in PROFILES.items()}


def edge_index():
    idx, edges = {}, []
    for i in range(N):
        for v in range(1 << N):
            u = v ^ (1 << i)
            if v < u:
                idx[(v, u)] = len(edges)
                edges.append((v, u))
    return idx


def load():
    idx = edge_index()
    rows, ranks = [], []
    for part in PARTS:
        with open(part, encoding='utf-8') as f:
            for line in f:
                es = json.loads(line)['edges']
                vec = np.zeros(len(idx), dtype=np.float32)
                dirs = [0] * N
                for u, v in es:
                    a, b = (u, v) if u < v else (v, u)
                    vec[idx[(a, b)]] = 1.0
                    dirs[(u ^ v).bit_length() - 1] += 1
                rows.append(vec)
                ranks.append(RANK_OF[tuple(sorted(dirs, reverse=True))])
    return np.array(rows), ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None,
                    help='write the minimum-distance pairs to this CSV')
    ap.add_argument('--block', type=int, default=1500)
    args = ap.parse_args()

    M, ranks = load()
    n = M.shape[0]
    print(f'loaded {n} solutions, {M.shape[1]} edge slots')

    tally = Counter()
    total = 0
    weighted = 0
    minimum = None
    min_pairs = []
    for start in range(0, n, args.block):
        end = min(start + args.block, n)
        inter = M[start:end] @ M.T
        for r in range(end - start):
            i = start + r
            d = (2 * (EDGE_COUNT - inter[r, i + 1:])).astype(np.int32)
            if d.size == 0:
                continue
            vals, counts = np.unique(d, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                tally[v] += c
            total += d.size
            weighted += int(d.sum())
            m = int(d.min())
            if minimum is None or m < minimum:
                minimum = m
                min_pairs = []
            if m == minimum:
                for off in np.where(d == minimum)[0].tolist():
                    min_pairs.append((i, i + 1 + off))

    assert total == n * (n - 1) // 2, (total, n * (n - 1) // 2)
    srt = sorted(tally.items())
    cum, median = 0, None
    for v, c in srt:
        cum += c
        if median is None and cum >= total / 2:
            median = v
    print(f'pairs        : {total}')
    print(f'min          : {minimum}')
    print(f'max          : {max(tally)}')
    print(f'mean         : {weighted}/{total} = {weighted / total:.9f}')
    print(f'median       : {median}')
    print(f'pairs at min : {len(min_pairs)}')
    print(f'pairs at max : {tally[max(tally)]}')

    combos = Counter(tuple(sorted((ranks[i], ranks[j]))) for i, j in min_pairs)
    print(f'profile-type combinations at min : {len(combos)}')
    for combo, c in sorted(combos.items(), key=lambda x: (-x[1], x[0])):
        print(f'   types {combo[0]}/{combo[1]}: {c}')

    expected = {'min': 6, 'max': 274, 'median': 196,
                'num': 38555373486, 'den': 197319045,
                'min_pairs': 636, 'combos': 28, 'max_pairs': 85}
    ok = (minimum == expected['min'] and max(tally) == expected['max']
          and median == expected['median'] and weighted == expected['num']
          and total == expected['den'] and len(min_pairs) == expected['min_pairs']
          and tally[max(tally)] == expected['max_pairs']
          and len(combos) == expected['combos'])
    print('RESULT:', 'matches the paper' if ok
          else 'MISMATCH against the values stated in the paper')
    if not ok:
        raise SystemExit(1)

    if args.csv:
        with open(args.csv, 'w', encoding='utf-8', newline='\n') as f:
            f.write('index_a,index_b,rank_a,rank_b,distance\n')
            for i, j in min_pairs:
                f.write(f'{i},{j},{ranks[i]},{ranks[j]},{minimum}\n')
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
