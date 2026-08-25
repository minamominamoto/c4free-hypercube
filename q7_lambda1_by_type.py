#!/usr/bin/env python3
"""
q7_lambda1_by_type.py -- reproduce the manuscript's Table of 20
dimension-profile types (counts, profile entropy H, and the per-type MEAN
of the spectral radius lambda_1) directly from the released Q7 catalogue,
and assert every printed figure against the values stated in the paper.

A round-77 review noted that the per-type lambda_1 means in that table,
while independently re-derivable, were not directly printed by any bundled
script. This script closes that gap. For each of the 19,866 solutions in
q7_edges_304.jsonl.part1..3 it computes

  - the dimension profile (edge counts per direction, sorted decreasingly),
  - lambda_1, the largest adjacency eigenvalue, obtained as the largest
    singular value of the 64x64 bipartite biadjacency block B (rows: even-
    parity vertices ascending; columns: odd-parity vertices ascending);
    for a bipartite graph the adjacency spectrum is {+/- sigma_i(B)}, so
    lambda_1 = sqrt(lambda_max(B B^T)),

then aggregates by profile and checks, against the paper's table:
the 20 profiles and their exact solution counts, the profile Shannon
entropy H = -sum_d (c_d/304) log2 (c_d/304) to four decimal places, the
per-type mean lambda_1 to four decimal places (each mean is rounded
independently), and the exhaustive individual-solution range
lambda_1 in [4.78543, 4.79129] to five decimal places.

Requires numpy (like analyze_q7_structure.py); run from the release
directory, or pass the three part files as arguments:

    python3 q7_lambda1_by_type.py [part1 part2 part3]

Read-only; writes nothing. Exit status 0 iff everything matches.
"""
import json
import math
import sys

import numpy as np

N = 7
V = 1 << N
DEFAULT_PARTS = ['q7_edges_304.jsonl.part1',
                 'q7_edges_304.jsonl.part2',
                 'q7_edges_304.jsonl.part3']

# The paper's table: profile -> (solutions, H to 4dp, mean lambda_1 to 4dp).
EXPECTED = {
    (44, 44, 44, 44, 43, 43, 42): (3155, '2.8072', '4.7869'),
    (45, 45, 45, 43, 43, 42, 41): (2913, '2.8065', '4.7880'),
    (45, 45, 45, 43, 43, 43, 40): (2200, '2.8063', '4.7870'),
    (44, 44, 44, 44, 44, 43, 41): (2116, '2.8069', '4.7875'),
    (46, 46, 46, 42, 42, 42, 40): (1756, '2.8053', '4.7874'),
    (46, 46, 46, 42, 42, 41, 41): (1181, '2.8054', '4.7887'),
    (44, 44, 44, 44, 44, 44, 40): (1085, '2.8066', '4.7868'),
    (45, 45, 45, 43, 42, 42, 42): (1064, '2.8066', '4.7877'),
    (45, 45, 44, 43, 43, 43, 41): (974, '2.8067', '4.7868'),
    (44, 44, 44, 44, 44, 42, 42): (931, '2.8070', '4.7866'),
    (47, 47, 47, 41, 41, 41, 40): (902, '2.8037', '4.7880'),
    (45, 45, 43, 43, 43, 43, 42): (439, '2.8069', '4.7879'),
    (45, 44, 44, 43, 43, 43, 42): (307, '2.8070', '4.7880'),
    (46, 45, 45, 42, 42, 42, 42): (236, '2.8063', '4.7891'),
    (44, 44, 44, 43, 43, 43, 43): (163, '2.8073', '4.7866'),
    (47, 47, 44, 43, 41, 41, 41): (153, '2.8050', '4.7875'),
    (46, 46, 44, 42, 42, 42, 42): (107, '2.8062', '4.7888'),
    (48, 48, 48, 40, 40, 40, 40): (101, '2.8014', '4.7881'),
    (46, 46, 43, 43, 42, 42, 42): (44, '2.8063', '4.7884'),
    (46, 46, 45, 42, 42, 42, 41): (39, '2.8058', '4.7878'),
}
EXPECTED_RANGE = ('4.78543', '4.79129')
EXPECTED_TOTAL = 19866

FAIL = []


def expect(name, ok, detail):
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


def main():
    parts = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PARTS
    if len(parts) != 3:
        raise SystemExit('expected exactly three part files')

    U = [v for v in range(V) if bin(v).count('1') % 2 == 0]
    ROW = {v: i for i, v in enumerate(U)}
    UC = [v for v in range(V) if bin(v).count('1') % 2 == 1]
    COL = {v: i for i, v in enumerate(UC)}

    stats = {}   # profile -> [count, sum_lambda1]
    lam_min, lam_max = float('inf'), float('-inf')
    total = 0
    for path in parts:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                edges = json.loads(line)['edges']
                cnt = [0] * N
                B = np.zeros((len(U), len(UC)))
                for u, v in edges:
                    cnt[(u ^ v).bit_length() - 1] += 1
                    if u in ROW:
                        B[ROW[u], COL[v]] = 1.0
                    else:
                        B[ROW[v], COL[u]] = 1.0
                lam1 = math.sqrt(float(np.linalg.eigvalsh(B @ B.T)[-1]))
                lam_min = min(lam_min, lam1)
                lam_max = max(lam_max, lam1)
                prof = tuple(sorted(cnt, reverse=True))
                s = stats.setdefault(prof, [0, 0.0])
                s[0] += 1
                s[1] += lam1
                total += 1

    expect('total solutions', total == EXPECTED_TOTAL,
           f'{total} vs {EXPECTED_TOTAL}')
    expect('number of profile types', len(stats) == len(EXPECTED),
           f'{len(stats)} vs {len(EXPECTED)}')

    print(f"{'rank':>4} {'profile':<30} {'n':>5} {'H':>7} "
          f"{'mean lambda_1':>13}")
    ranked = sorted(stats.items(), key=lambda kv: -kv[1][0])
    for rank, (prof, (n, sl)) in enumerate(ranked, 1):
        h = -sum((c / 304) * math.log2(c / 304) for c in prof)
        mean = sl / n
        print(f"{rank:>4} {str(list(prof)):<30} {n:>5} {h:>7.4f} "
              f"{mean:>13.4f}")
        exp = EXPECTED.get(prof)
        if exp is None:
            expect(f'profile {list(prof)}', False, 'not in the paper table')
            continue
        en, eh, em = exp
        expect(f'type {rank} {list(prof)}',
               n == en and f'{h:.4f}' == eh and f'{mean:.4f}' == em,
               f'count {n} vs {en}, H {h:.4f} vs {eh}, '
               f'mean {mean:.4f} vs {em}')

    expect('individual lambda_1 range',
           (f'{lam_min:.5f}', f'{lam_max:.5f}') == EXPECTED_RANGE,
           f'[{lam_min:.5f}, {lam_max:.5f}] vs '
           f'[{EXPECTED_RANGE[0]}, {EXPECTED_RANGE[1]}]')

    if FAIL:
        print(f'RESULT: MISMATCH against the paper table ({len(FAIL)})')
        return 1
    print('RESULT: matches the paper (all 20 per-type counts, entropies and '
          'mean lambda_1 values, and the individual lambda_1 range)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
