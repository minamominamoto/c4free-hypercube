#!/usr/bin/env python3
"""
q7_orbit_witness_check.py -- CHECKER for the orbit-witness certificate
q7_orbit_witnesses.json (see q7_orbit_witnesses_gen.py for the generator
and for the encoding conventions, which this file shares verbatim).

Default mode (standard library only, well under a minute): verifies

  1. every one of the 19,866 witnesses -- applying the recorded group
     element (k, a) to the orbit representative reproduces exactly that
     catalogue solution's edge set (this certifies the entire orbit
     assignment and hence "at most 180 orbits" without redoing the group
     sweep);
  2. the witness file's orbit assignment is IDENTICAL to the released
     census artefact q7_orbit_census.json, its orbit sizes match
     catalogue_sizes there, and each representative is the smallest
     catalogue index in its orbit (the census's representative rule);
  3. each orbit's stored canonical form is really attained by its stored
     canonical witness element, the 180 canonical forms are pairwise
     distinct, and the catalogue SHA-256 recorded in the certificate
     matches the catalogue on disk.

--canonical mode (requires numpy; census-scale wall time, budget a few
hundred seconds): re-verifies MINIMALITY -- for each orbit it sweeps all
645,120 images of the representative and asserts that the stored canonical
form is the lexicographic minimum.  Because the canonical form is an orbit
invariant, minimality plus pairwise distinctness certifies "exactly 180
orbits" independently of the census scan.  Use --orbits A:B to check a
slice (e.g. --orbits 0:45) if your environment enforces per-command time
limits; the slices together cover the full claim.

Usage:
    python3 q7_orbit_witness_check.py
    python3 q7_orbit_witness_check.py --canonical [--orbits A:B]

Read-only; writes nothing.  Exit status 0 iff every check passes.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import permutations

N = 7
V = 1 << N
NPERM = 5040
PARTS = ['q7_edges_304.jsonl.part1',
         'q7_edges_304.jsonl.part2',
         'q7_edges_304.jsonl.part3']
WIT = 'q7_orbit_witnesses.json'
CENSUS = 'q7_orbit_census.json'

SLOT = {}
for _d in range(N):
    for _v in range(V):
        _u = _v ^ (1 << _d)
        if _v < _u:
            SLOT[(_v, _u)] = len(SLOT)
NSLOT = len(SLOT)
PERMS = list(permutations(range(N)))

FAIL = []


def expect(name, ok, detail=''):
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}" +
          (f": {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def vmap(pi, a):
    out = [0] * V
    for v in range(V):
        w = 0
        for d in range(N):
            if v >> d & 1:
                w |= 1 << pi[d]
        out[v] = w ^ a
    return out


def apply_elem(edges, pi, a):
    vm = vmap(pi, a)
    res = set()
    for x, y in edges:
        p, q = vm[x], vm[y]
        res.add((p, q) if p < q else (q, p))
    return res


def pack56(edge_set):
    bits = bytearray(56)
    for e in edge_set:
        s = SLOT[e]
        bits[s >> 3] |= 128 >> (s & 7)
    return bytes(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', action='store_true',
                    help='re-verify each canonical form is the true '
                         'lexicographic minimum over all 645,120 images '
                         '(numpy; census-scale wall time)')
    ap.add_argument('--orbits', default=None, metavar='A:B',
                    help='with --canonical: check only orbits A..B-1')
    args = ap.parse_args()

    sols = []
    for part in PARTS:
        with open(part, encoding='utf-8') as f:
            for line in f:
                es = json.loads(line)['edges']
                sols.append({(u, v) if u < v else (v, u) for u, v in es})

    with open(WIT, encoding='utf-8') as f:
        wit = json.load(f)
    with open(CENSUS, encoding='utf-8') as f:
        census = json.load(f)

    n = len(sols)
    K = wit['orbits']
    reps = wit['reps']
    canon = wit['canonical']
    cwit = wit['canonical_witness']
    triples = wit['witness']
    expect('catalogue size', n == 19866, f'{n}')
    expect('orbit count', K == 180 and len(reps) == K and
           len(canon) == K and len(cwit) == K, f'{K}')
    expect('witness rows', len(triples) == n, f'{len(triples)}')

    h = hashlib.sha256()
    for s in sols:
        vec = bytearray(NSLOT)
        for e in s:
            vec[SLOT[e]] = 1
        h.update(bytes(vec))
    expect('catalogue SHA-256 matches certificate',
           h.hexdigest() == wit['catalogue_sha256'],
           h.hexdigest()[:12] + '...')

    assign = [t[0] for t in triples]
    expect('assignment identical to released census',
           assign == census['assignment'])
    sizes = Counter(assign)
    expect('orbit sizes match census catalogue_sizes',
           [sizes[i] for i in range(K)] == census['catalogue_sizes'])
    first = {}
    for j, o in enumerate(assign):
        first.setdefault(o, j)
    expect('representatives are the minimal index of their orbit',
           all(first[o] == reps[o] for o in range(K)))
    expect('representative witnesses are the identity',
           all(triples[reps[o]] == [o, 0, 0] for o in range(K)))

    bad = 0
    for j, (o, k, a) in enumerate(triples):
        if not (0 <= o < K and 0 <= k < NPERM and 0 <= a < V):
            bad += 1
            continue
        if apply_elem(sols[reps[o]], PERMS[k], a) != sols[j]:
            bad += 1
    expect('all 19,866 witnesses map representative to solution',
           bad == 0, f'{bad} failures')

    cbad = 0
    for o in range(K):
        k, a = cwit[o]
        img = apply_elem(sols[reps[o]], PERMS[k], a)
        if pack56(img).hex() != canon[o]:
            cbad += 1
    expect('all 180 canonical forms attained by their witness element',
           cbad == 0, f'{cbad} failures')
    expect('canonical forms pairwise distinct',
           len(set(canon)) == K)

    if args.canonical:
        import numpy as np
        lo, hi = 0, K
        if args.orbits:
            lo, hi = (int(x) for x in args.orbits.split(':'))
        M = {o: np.zeros(NSLOT, dtype=np.uint8) for o in range(lo, hi)}
        for o in range(lo, hi):
            for e in sols[reps[o]]:
                M[o][SLOT[e]] = 1

        def slot_perm(pi, a):
            vm = vmap(pi, a)
            out = np.empty(NSLOT, dtype=np.int32)
            for (x, y), s in SLOT.items():
                p, q = vm[x], vm[y]
                out[SLOT[(p, q) if p < q else (q, p)]] = s
            return out

        trans = np.array([slot_perm(tuple(range(N)), a) for a in range(V)])
        perms_np = [slot_perm(pi, 0) for pi in PERMS]
        chunk = 64
        mbad = 0
        for o in range(lo, hi):
            v = M[o]
            best = None
            for base in range(0, NPERM, chunk):
                blk = perms_np[base:base + chunk]
                idx = np.concatenate([pp[trans] for pp in blk])
                words = np.packbits(v[idx], axis=1).view('>u8')
                cand = np.arange(words.shape[0])
                for c in range(7):
                    col = words[cand, c]
                    cand = cand[col == col.min()]
                    if len(cand) == 1:
                        break
                w = tuple(words[int(cand[0])])
                if best is None or w < best:
                    best = w
            got = b''.join(int(x).to_bytes(8, 'big') for x in best).hex()
            if got != canon[o]:
                mbad += 1
        expect(f'canonical minimality re-verified for orbits {lo}..{hi - 1}',
               mbad == 0, f'{mbad} failures')

    if FAIL:
        print(f'RESULT: MISMATCH ({len(FAIL)})')
        return 1
    print('RESULT: certificate verifies' +
          ('' if args.canonical else
           ' (run --canonical to re-verify minimality of the canonical '
           'forms as well)'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
