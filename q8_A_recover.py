#!/usr/bin/env python3
"""
q8_A_recover.py -- fixed-seed search for a 682-edge odd-square subgraph of Q8
whose local-field histogram on U is {0:1, 2:84, 4:43}.

Why this exists. The released 682-edge witness q8_odd_square_682.json was
obtained from a spin configuration communicated in dated private
correspondence; its search seeds and intermediate records were not archived,
so that one artefact -- the headline lower-bound certificate -- was the only
item in the release whose *derivation* could not be re-executed. (The
certificate itself has always been machine-checkable, and verify.py checks it.)
This script closes that gap: it finds, from a fixed seed and with no input
from the released witness, a spin configuration realising the same optimal
field distribution.

It does NOT claim to reproduce the released edge set, nor any configuration
Marinari-Parisi-Ritort may have used. Distinct spin configurations realising
the same distribution are expected; what is reproducible here is the
*existence proof*, not the particular artefact.

Method is identical to q8_B_recover.py: canonical fully frustrated coupling,
simulated annealing on the L1 distance between the current h-histogram on U
and the target.

Standard library only. Usage:
    python3 q8_A_recover.py

The defaults (seeds 90000.., --iters 250000) reproduce the released
q8_A_witness.json exactly: seed 90008 hits at iteration 207579. The iteration
budget is NOT merely a stopping rule -- it enters the annealing temperature
schedule T = max(0.05, 3.0*(1 - it/iters)), so a different --iters traverses a
different trajectory and finds a different (equally valid) witness.

Output goes to q8_A_witness_run.json by default, NOT to the released
q8_A_witness.json, which is covered by ODDSQUARE_BRIDGE_SHA256SUMS.txt.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from itertools import combinations

N = 8
V = 1 << N
U = [v for v in range(V) if bin(v).count('1') % 2 == 0]
TARGET = {0: 1, 2: 84, 4: 43}
DEFAULT_SEEDS = [90000 + i for i in range(12)]
DEFAULT_ITERS = 250000   # the value used for the released witness


def coupling(x, dim):
    return 1 if bin(x & ((1 << dim) - 1)).count('1') % 2 == 0 else -1


NBR = [[(x ^ (1 << d), coupling(min(x, x ^ (1 << d)), d)) for d in range(N)]
       for x in range(V)]


def h_of(s, v):
    return s[v] * sum(j * s[y] for y, j in NBR[v])


def hist_on_U(s):
    return Counter(h_of(s, v) for v in U)


def l1(hist, target):
    keys = set(hist) | set(target)
    return sum(abs(hist.get(k, 0) - target.get(k, 0)) for k in keys)


def anneal(rng, iters):
    s = [rng.choice((1, -1)) for _ in range(V)]
    hist = hist_on_U(s)
    cur = l1(hist, TARGET)
    best = cur
    for it in range(iters):
        if cur == 0:
            return 0, s, it
        T = max(0.05, 3.0 * (1.0 - it / iters))
        x = rng.randrange(V)
        s[x] = -s[x]
        nh = hist_on_U(s)
        nd = l1(nh, TARGET)
        if nd <= cur or rng.random() < pow(2.718281828, (cur - nd) / T):
            cur, hist = nd, nh
        else:
            s[x] = -s[x]
        best = min(best, cur)
    return best, s, None


def certify(s):
    """Recompute everything from the spins, independently of the search."""
    edges, deg = [], [0] * V
    for x in range(V):
        for d in range(N):
            y = x ^ (1 << d)
            if x < y and coupling(x, d) * s[x] * s[y] == 1:
                edges.append([x, y])
                deg[x] += 1
                deg[y] += 1
    eset = {tuple(e) for e in edges}
    sq = Counter()
    for d1, d2 in combinations(range(N), 2):
        m1, m2 = 1 << d1, 1 << d2
        for base in range(V):
            if base & (m1 | m2):
                continue
            c = ((base, base | m1), (base, base | m2),
                 (base | m1, base | m1 | m2), (base | m2, base | m1 | m2))
            sq[sum(tuple(sorted(e)) in eset for e in c)] += 1
    hU = Counter(2 * deg[v] - N for v in U)
    return edges, dict(sorted(hU.items())), dict(sorted(sq.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--iters', type=int, default=DEFAULT_ITERS)
    ap.add_argument('--output', default='q8_A_witness_run.json',
                    help='output path. NOTE: this deliberately does NOT default '
                         'to the released q8_A_witness.json, which is listed in '
                         'ODDSQUARE_BRIDGE_SHA256SUMS.txt; overwriting it would '
                         'break that manifest.')
    args = ap.parse_args()

    print(f'target h-histogram on U: {TARGET}')
    for seed in args.seeds:
        rng = random.Random(seed)
        best, s, hit = anneal(rng, args.iters)
        if best == 0:
            print(f'seed {seed}: HIT at iteration {hit}')
            if s[0] == -1:
                s = [-t for t in s]
            edges, hU, sq = certify(s)
            spins = ''.join('1' if t == 1 else '0' for t in s)
            ok = (len(edges) == 682 and hU == {0: 1, 2: 84, 4: 43}
                  and set(sq) <= {1, 3})
            print(f'  edges={len(edges)} h|U={hU} squares={sq}')
            print(f'  odd-square: {set(sq) <= {1, 3}}')
            rec = {'n': N, 'seed': seed, 'hit_iteration': hit,
                   'iterations_budget': args.iters,
                   'coupling': 'J(x,x xor 2^dim)=(-1)^(popcount(x & ((1<<dim)-1)))',
                   'spin_encoding': '1 for +1, 0 for -1; vertex order 0..255',
                   'spins': spins,
                   'num_edges': len(edges),
                   'local_field_histogram_U': hU,
                   'square_positive_edge_histogram': sq,
                   'edges': edges,
                   'provenance_note': (
                       'Found by this script from the stated seed with no input '
                       'from q8_odd_square_682.json. Not claimed to reproduce '
                       'that edge set or any configuration of MPR95.')}
            with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(rec, f, sort_keys=True, separators=(',', ':'))
                f.write('\n')
            print(f'  wrote {args.output}, sha256='
                  f'{hashlib.sha256(open(args.output, "rb").read()).hexdigest()}')
            return 0 if ok else 1
        print(f'seed {seed}: miss (best L1 distance {best})')
    print('no hit within the given budget')
    return 1


if __name__ == '__main__':
    sys.exit(main())
