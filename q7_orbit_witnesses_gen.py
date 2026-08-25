#!/usr/bin/env python3
"""
q7_orbit_witnesses_gen.py -- one-time GENERATOR for the orbit-witness
certificate q7_orbit_witnesses.json.

Motivation (round-78 review): the full Aut(Q_7)-orbit census is the paper's
heaviest computation, and two independent review environments could not
finish a from-scratch rerun within their per-command time limits. This
script converts the census's conclusions into a *certificate* that a
standard-library checker (q7_orbit_witness_check.py) can verify in seconds:

  - for every one of the 19,866 catalogue solutions, a concrete group
    element g = (perm k, xor a) with  g(representative) = that solution
    (so "at most 180 orbits" and the entire orbit assignment become a
    seconds-checkable claim), and
  - for every orbit, the lexicographically minimal 448-bit incidence
    vector over ALL 645,120 images of its representative (the orbit's
    canonical form), together with a group element achieving it.  The 180
    canonical forms are pairwise distinct; since each is an orbit
    invariant, their distinctness certifies "exactly 180 orbits" once the
    checker's --canonical mode has re-verified minimality.

Conventions (shared verbatim with the checker):
  * A group element is v -> pi(v) XOR a, encoded as (k, a) where
    pi = the k-th tuple of itertools.permutations(range(7)) in its native
    lexicographic order (k in 0..5039, pi maps bit d to bit pi[d]) and
    a in 0..127.  (k, a) = (0, 0) is the identity.
  * Edge slots: for d in 0..6, for v in 0..127, the edge (v, v^2^d) with
    v < v^2^d gets the next slot index; this fixed enumeration defines the
    448-position incidence vector.  The canonical form is the
    lexicographically smallest incidence vector (equivalently, of its
    56-byte big-endian bit-packing) over the orbit.
  * Witnesses are deterministic: for each solution the recorded (k, a) is
    the smallest value of k*128 + a that maps the orbit representative to
    it, and likewise for the canonical-form witness.
  * Orbit numbering, scan order and representative choice (first
    unassigned catalogue index) are exactly the census's, so the witness
    file's assignment must equal q7_orbit_census.json's -- the generator
    asserts this on completion.

Requires numpy; budgeted and checkpointed like q7_orbit_census.py (the
budget takes effect only between orbits).  Wall-clock for the whole
generation is census-scale (a few hundred seconds on our reference
machine).  Usage:

    python3 q7_orbit_witnesses_gen.py [--budget SECONDS]

Writes q7_orbit_witnesses.json (and its checkpoint
q7_orbit_witnesses_ckpt.npz while running); never touches the released
census artefacts.
"""
import argparse
import hashlib
import json
import os
import time
from itertools import permutations

import numpy as np

N = 7
V = 1 << N
NPERM = 5040
PARTS = ['q7_edges_304.jsonl.part1',
         'q7_edges_304.jsonl.part2',
         'q7_edges_304.jsonl.part3']
CKPT = 'q7_orbit_witnesses_ckpt.npz'
OUT = 'q7_orbit_witnesses.json'
CENSUS = 'q7_orbit_census.json'

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
    rows = []
    for part in PARTS:
        with open(part, encoding='utf-8') as f:
            for line in f:
                es = json.loads(line)['edges']
                vec = np.zeros(NSLOT, dtype=np.uint8)
                for u, v in es:
                    a, b = (u, v) if u < v else (v, u)
                    vec[SLOT[(a, b)]] = 1
                rows.append(vec)
    return np.array(rows)


def pack_rows(imgs):
    """(rows, 448) uint8 0/1 -> (rows, 7) big-endian uint64 words whose
    lexicographic order equals that of the incidence vectors."""
    return np.packbits(imgs, axis=1).view('>u8')


def lexmin_of(words):
    """Return (index_of_lexicographic_minimum, its_row) for (rows,7) '>u8'.
    Ties broken by smallest index (row order = ascending k*128+a)."""
    cand = np.arange(words.shape[0])
    for c in range(words.shape[1]):
        col = words[cand, c]
        cand = cand[col == col.min()]
        if len(cand) == 1:
            break
    i = int(cand[0])
    return i, words[i].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=250.0)
    ap.add_argument('--chunk', type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()

    M = load()
    n = M.shape[0]
    rng = np.random.default_rng(20260824)
    W = rng.integers(1, 2 ** 62, size=NSLOT, dtype=np.int64)
    sig = (M.astype(np.int64) @ W)
    assert len(set(sig.tolist())) == n, 'signature collision inside catalogue'
    order = np.argsort(sig)
    sig_sorted = sig[order]
    cat_sig = hashlib.sha256(M.tobytes()).hexdigest()

    if os.path.exists(CKPT):
        z = np.load(CKPT)
        assert str(z['catalogue_sha256']) == cat_sig, \
            f'checkpoint does not match this catalogue; delete {CKPT}'
        orbit = z['orbit'].tolist()
        wit_k = z['wit_k'].tolist()
        wit_a = z['wit_a'].tolist()
        reps = z['reps'].tolist()
        canon = [bytes(b) for b in z['canon']]
        canon_k = z['canon_k'].tolist()
        canon_a = z['canon_a'].tolist()
        print(f'resumed: {sum(1 for x in orbit if x >= 0)}/{n} assigned, '
              f'{len(reps)} orbits so far')
    else:
        orbit = [-1] * n
        wit_k = [-1] * n
        wit_a = [-1] * n
        reps, canon, canon_k, canon_a = [], [], [], []

    def save():
        np.savez(CKPT, orbit=np.array(orbit), wit_k=np.array(wit_k),
                 wit_a=np.array(wit_a), reps=np.array(reps, dtype=np.int64),
                 canon=np.array([np.frombuffer(c, dtype=np.uint8)
                                 for c in canon], dtype=np.uint8
                                ).reshape(len(canon), 56),
                 canon_k=np.array(canon_k, dtype=np.int64),
                 canon_a=np.array(canon_a, dtype=np.int64),
                 catalogue_sha256=np.array(cat_sig))

    trans = np.array([slot_perm(tuple(range(N)), a) for a in range(V)])
    perms = [slot_perm(pi, 0) for pi in permutations(range(N))]

    done_here = 0
    for start in range(n):
        if orbit[start] != -1:
            continue
        if done_here and time.time() - t0 > args.budget:
            save()
            print(f'budget reached: {sum(1 for x in orbit if x >= 0)}/{n} '
                  f'assigned, {len(reps)} orbits; checkpoint saved')
            return 2
        v = M[start]
        cur = len(reps)
        best_words = None
        best_ka = None
        for base in range(0, len(perms), args.chunk):
            blk = perms[base:base + args.chunk]
            idx = np.concatenate([pp[trans] for pp in blk])   # (k*128, NSLOT)
            imgs = v[idx]
            # canonical-form running minimum over ALL images of this rep
            words = pack_rows(imgs)
            i, w = lexmin_of(words)
            if best_words is None or tuple(w) < tuple(best_words):
                best_words = w
                best_ka = (base + i // V, i % V)
            # catalogue matching (signature prefilter + exact confirmation)
            s = imgs.astype(np.int64) @ W
            pos = np.searchsorted(sig_sorted, s)
            pos = np.clip(pos, 0, n - 1)
            hit = sig_sorted[pos] == s
            for r in np.nonzero(hit)[0]:
                j = int(order[pos[r]])
                if orbit[j] == -1 and np.array_equal(imgs[r], M[j]):
                    orbit[j] = cur
                    wit_k[j] = base + int(r) // V
                    wit_a[j] = int(r) % V
        assert orbit[start] == cur and wit_k[start] == 0 and wit_a[start] == 0
        reps.append(start)
        canon.append(best_words.astype('>u8').tobytes())
        canon_k.append(best_ka[0])
        canon_a.append(best_ka[1])
        done_here += 1

    save()
    assert all(x >= 0 for x in orbit)
    assert len(set(canon)) == len(canon), 'canonical forms not distinct'

    # tie the certificate to the released census
    with open(CENSUS, encoding='utf-8') as f:
        census = json.load(f)
    assert census['assignment'] == orbit, \
        'witness orbit assignment differs from the released census'
    assert census['orbits'] == len(reps)

    data = {
        'orbits': len(reps),
        'catalogue_sha256': cat_sig,
        'reps': reps,
        'canonical': [c.hex() for c in canon],
        'canonical_witness': [[canon_k[i], canon_a[i]]
                              for i in range(len(reps))],
        'witness': [[orbit[j], wit_k[j], wit_a[j]] for j in range(n)],
    }
    keys = ('orbits', 'catalogue_sha256', 'reps', 'canonical',
            'canonical_witness', 'witness')
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump({k: data[k] for k in keys}, f)
    print(f'wrote {OUT}: {len(reps)} orbits, {n} witnesses, '
          f'sha256={hashlib.sha256(open(OUT, "rb").read()).hexdigest()}')
    print(f'[{time.time() - t0:.0f}s]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
