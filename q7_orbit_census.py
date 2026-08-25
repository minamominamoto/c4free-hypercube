#!/usr/bin/env python3
"""
q7_orbit_census.py -- full Aut(Q_7)-orbit decomposition of the released
19,866-solution catalogue.

An automorphism of Q_n is v -> pi(v) XOR a, so |Aut(Q_7)| = 7! * 2^7 = 645,120.
Orbits are found by taking an unassigned solution, applying every group element
to it, and recording which catalogue members appear among the images.

Membership uses a two-stage test: a 64-bit random-projection signature to
prefilter (vectorised, fast), then an exact comparison of the 448-byte
incidence vector on every signature hit. The signature alone is NOT safe -- a
single collision silently merges two orbits, which happened in an earlier
signature-only run of the odd-square subset (reporting sizes 128/48 where the
exact test gives 127/49). The exact confirmation removes that risk while
keeping the speed.

Writes a checkpoint after each orbit so the run can be resumed.

The scan order over solutions, the seeded random-projection signature, and
the orbit numbering are all deterministic, so a census run to completion
reproduces the released q7_orbit_census.json byte for byte (the JSON is
written with a fixed key order for exactly this reason).

--fresh runs a from-scratch census whose checkpoint and JSON live in
separate files (q7_orbit_census_fresh_ckpt.npz / q7_orbit_census_fresh.json):
it never reads the released checkpoint and never writes the released
artefacts, so the released, manifest-listed files cannot be overwritten by
it. A fresh run resumes its own checkpoint across invocations; delete the
two _fresh files to force a new from-scratch start.

Requires numpy. Usage:
    python3 q7_orbit_census.py               # run (resumes if checkpoint exists)
    python3 q7_orbit_census.py --report      # summarise a finished run
    python3 q7_orbit_census.py --fresh       # from-scratch run, separate files
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import permutations

import numpy as np

N = 7
V = 1 << N
GROUP = 5040 * V
PARTS = ['q7_edges_304.jsonl.part1',
         'q7_edges_304.jsonl.part2',
         'q7_edges_304.jsonl.part3']
CKPT = 'q7_orbit_census_ckpt.npz'
OUT = 'q7_orbit_census.json'
FRESH_CKPT = 'q7_orbit_census_fresh_ckpt.npz'
FRESH_OUT = 'q7_orbit_census_fresh.json'

# Fixed key order for the JSON artefact, matching the released file, so that a
# regenerated artefact is byte-identical to the released one whenever their
# contents agree. Keys absent from the data (e.g. before the --stabilisers
# pass) are simply skipped.
CANON_KEYS = ('stabiliser_orders', 'orbit_lengths', 'labelled_total',
              'orbits', 'catalogue_sizes', 'catalogue_sizes_sorted',
              'assignment')


def write_out(path, data):
    ordered = {k: data[k] for k in CANON_KEYS if k in data}
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(ordered, f)

SLOT = {}
for _d in range(N):
    for _v in range(V):
        _u = _v ^ (1 << _d)
        if _v < _u:
            SLOT[(_v, _u)] = len(SLOT)
NSLOT = len(SLOT)


def slot_perm(pi, a):
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
    rows, dirs = [], []
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
                rows.append(vec)
                dirs.append(tuple(dc))
    return np.array(rows), dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--stabilisers', action='store_true',
                    help='compute stabiliser orders per orbit (needs a '
                         'completed census); prunes to permutations that '
                         'preserve the direction-count vector')
    ap.add_argument('--budget', type=float, default=250.0,
                    help='seconds to work before saving and exiting')
    ap.add_argument('--chunk', type=int, default=64,
                    help='permutations per batched matmul')
    ap.add_argument('--fresh', action='store_true',
                    help='from-scratch census using separate files '
                         f'({FRESH_CKPT} / {FRESH_OUT}); never reads or '
                         'writes the released artefacts')
    args = ap.parse_args()
    ckpt = FRESH_CKPT if args.fresh else CKPT
    out = FRESH_OUT if args.fresh else OUT

    import time
    t0 = time.time()

    M, dirs = load()
    n = M.shape[0]
    rng = np.random.default_rng(20260824)
    W = rng.integers(1, 2 ** 62, size=NSLOT, dtype=np.int64)
    sig = (M.astype(np.int64) @ W)
    assert len(set(sig.tolist())) == n, 'signature collision inside catalogue'
    order = np.argsort(sig)
    sig_sorted = sig[order]

    cat_sig = hashlib.sha256(M.tobytes()).hexdigest()
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        stored = str(z['catalogue_sha256']) if 'catalogue_sha256' in z else ''
        if stored != cat_sig:
            print('checkpoint does not match this catalogue '
                  f'(stored {stored[:12]}..., computed {cat_sig[:12]}...); '
                  f'delete {ckpt} and rerun')
            return 3
        orbit = z['orbit'].tolist()
        sizes = z['sizes'].tolist()
        print(f'resumed: {sum(1 for x in orbit if x >= 0)}/{n} assigned, '
              f'{len(sizes)} orbits so far')
    else:
        orbit = [-1] * n
        sizes = []

    if args.report:
        report(orbit, sizes, dirs, n, out)
        return 0

    if args.stabilisers:
        assigned = sum(1 for x in orbit if x >= 0)
        if assigned != n or any(x < 0 for x in orbit):
            print(f'ERROR: --stabilisers requires a completed census; '
                  f'only {assigned}/{n} solutions are assigned. '
                  f'Run the census to completion first.', file=sys.stderr)
            return 2
        stabilisers(M, dirs, orbit, out)
        return 0

    trans = np.array([slot_perm(tuple(range(N)), a) for a in range(V)])
    perms = [slot_perm(pi, 0) for pi in permutations(range(N))]

    done_here = 0
    for start in range(n):
        if orbit[start] != -1:
            continue
        # The budget is only honoured between orbits, and only once this
        # invocation has completed at least one: a partially processed orbit
        # cannot be checkpointed, so stopping mid-orbit would lose the work
        # and, if a single orbit exceeded the budget, no invocation would ever
        # make progress. Each invocation therefore always finishes the orbit
        # it has started.
        if done_here and time.time() - t0 > args.budget:
            np.savez(ckpt, orbit=np.array(orbit), sizes=np.array(sizes),
                 catalogue_sha256=np.array(cat_sig))
            print(f'budget reached: {sum(1 for x in orbit if x >= 0)}/{n} '
                  f'assigned, {len(sizes)} orbits; checkpoint saved')
            return 2
        v = M[start]
        found = {start}
        for base in range(0, len(perms), args.chunk):
            blk = perms[base:base + args.chunk]
            idx = np.concatenate([pp[trans] for pp in blk])   # (k*128, NSLOT)
            imgs = v[idx]
            s = imgs.astype(np.int64) @ W
            pos = np.searchsorted(sig_sorted, s)
            pos = np.clip(pos, 0, n - 1)
            hit = sig_sorted[pos] == s
            for r in np.nonzero(hit)[0]:
                j = int(order[pos[r]])
                if np.array_equal(imgs[r], M[j]):      # exact confirmation
                    found.add(j)
        cur = len(sizes)
        for j in found:
            orbit[j] = cur
        sizes.append(len(found))
        done_here += 1

    if done_here:
        np.savez(ckpt, orbit=np.array(orbit), sizes=np.array(sizes),
                 catalogue_sha256=np.array(cat_sig))
    report(orbit, sizes, dirs, n, out)
    return 0


def report(orbit, sizes, dirs, n, out=OUT):
    assert all(x >= 0 for x in orbit), 'decomposition incomplete'
    print(f'ORBITS: {len(sizes)}')
    print(f'catalogue sizes histogram: '
          f'{dict(sorted(Counter(sizes).items(), reverse=True))}')
    print(f'largest orbit covers {max(sizes)} of {n} '
          f'({100 * max(sizes) / n:.1f}%)')
    top = sorted(sizes, reverse=True)[:10]
    print(f'top ten orbits cover {sum(top)} ({100 * sum(top) / n:.1f}%)')
    ok = (len(sizes) == 180 and max(sizes) == 2048 and sum(top) == 5199)
    print('RESULT:', 'matches the paper' if ok
          else 'MISMATCH against the values stated in the paper')
    if not ok:
        raise SystemExit(1)
    # Preserve stabiliser data from a previous --stabilisers pass if it is
    # still consistent with this census; drop it otherwise, so a stale pass
    # can never masquerade as belonging to a fresh decomposition.
    data = {}
    if os.path.exists(out):
        try:
            with open(out, encoding='utf-8') as f:
                old = json.load(f)
        except (json.JSONDecodeError, OSError):
            old = {}
        st = old.get('stabiliser_orders')
        if (isinstance(st, list) and len(st) == len(sizes)
                and all(isinstance(s, int) and s > 0 and GROUP % s == 0
                        for s in st)):
            data['stabiliser_orders'] = st
            data['orbit_lengths'] = [GROUP // s for s in st]
            data['labelled_total'] = sum(GROUP // s for s in st)
    data.update({'orbits': len(sizes),
                 # Aligned by orbit id: catalogue_sizes[k] is the number of
                 # released catalogue members assigned to orbit k.
                 'catalogue_sizes': list(sizes),
                 # Convenience ranking only; intentionally not indexed by
                 # orbit id.
                 'catalogue_sizes_sorted': sorted(sizes, reverse=True),
                 'assignment': orbit})
    write_out(out, data)
    print(f'wrote {out}')




def stabilisers(M, dirs, orbit, out=OUT):
    """Stabiliser order of one representative per orbit, and the implied total
    number of labelled solutions in the orbits represented.

    Only permutations preserving the representative's direction-count vector
    can fix it (an automorphism permutes directions), so the scan is over that
    subgroup times the 128 translations, not the full 645,120 elements.
    A negative orbit id denotes an incomplete census and is rejected rather
    than being treated as a real orbit.
    """
    if any(x < 0 for x in orbit):
        raise ValueError("stabiliser computation requires a completed orbit census")
    import time
    from collections import defaultdict
    reps = {}
    for i, o in enumerate(orbit):
        if o not in reps:
            reps[o] = i
    trans = np.array([slot_perm(tuple(range(N)), a) for a in range(V)])

    def preserving(dc):
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

    t0 = time.time()
    res = {}
    for o in sorted(reps):
        i = reps[o]
        v = M[i]
        cnt = 0
        for pi in preserving(dirs[i]):
            idx = slot_perm(pi, 0)[trans]
            cnt += int((v[idx] == v).all(axis=1).sum())
        res[o] = cnt
    hist = Counter(res.values())
    lengths = Counter(GROUP // c for c in res.values())
    total = sum(GROUP // c for c in res.values())
    print(f'stabiliser orders (per orbit): {dict(sorted(hist.items()))}')
    print(f'orbit lengths: {dict(sorted(lengths.items(), reverse=True))}')
    print(f'total labelled solutions in these orbits: {total}')
    print(f'catalogue is {100 * len(orbit) / total:.4f}% of that')
    print(f'all stabiliser orders divisible by 3: '
          f'{all(c % 3 == 0 for c in res.values())}')
    ok = (dict(hist) == {3: 142, 6: 32, 12: 4, 24: 1, 72: 1}
          and total == 34227200
          and all(c % 3 == 0 for c in res.values()))
    print('RESULT:', 'matches the paper' if ok
          else 'MISMATCH against the values stated in the paper')
    if not ok:
        raise SystemExit(1)
    # Persist the stabiliser data into the census artefact, so the completed
    # run is distributable and checkable (q7_orbit_census_check.py) without
    # re-executing the census or this pass.
    data = {}
    if os.path.exists(out):
        with open(out, encoding='utf-8') as f:
            data = json.load(f)
    ids = sorted(res)
    data['stabiliser_orders'] = [res[o] for o in ids]     # aligned by orbit id
    data['orbit_lengths'] = [GROUP // res[o] for o in ids]  # aligned by orbit id
    data['labelled_total'] = total
    write_out(out, data)
    print(f'updated {out} with stabiliser data')
    print(f'[{time.time() - t0:.0f}s]')
    return res

if __name__ == '__main__':
    sys.exit(main())
