#!/usr/bin/env python3
"""
q7_orbit_census_check.py -- lightweight consistency checker for the completed
orbit-census artefact q7_orbit_census.json (produced by q7_orbit_census.py,
census run to completion followed by --stabilisers).

This does NOT recompute the census (that is q7_orbit_census.py's job, budgeted
and resumable); it verifies, in well under a second and with the standard
library only, that the bundled artefact is internally consistent and matches
every census-derived number stated in the paper (Proposition `Orbit census'):

  * 180 orbits; assignment covers all 19,866 catalogue members exactly once;
  * per-orbit catalogue counts agree with the assignment and sum to 19,866;
  * largest orbit meets the catalogue in 2,048 members (10.3%), the ten
    largest in 5,199 (26.2%);
  * stabiliser orders per orbit have histogram {3:142, 6:32, 12:4, 24:1, 72:1},
    all divisible by 3;
  * orbit lengths are 645,120/stabiliser order, sum to 34,227,200 (of which
    the catalogue is 0.0580%), and each orbit meets the catalogue in at most
    its own length;
  * the two orbits of stabiliser order 24 and 72 meet the catalogue in 55 and
    46 members respectively (the Type-18 orbits).

Usage: python3 q7_orbit_census_check.py [path-to-json]
Exit status 0 iff every check passes.
"""
import json
import sys
from collections import Counter

N_CAT = 19866
GROUP = 645120


def fail(msg):
    print(f'FAIL: {msg}')
    raise SystemExit(1)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'q7_orbit_census.json'
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    for key in ('orbits', 'catalogue_sizes', 'catalogue_sizes_sorted',
                'assignment', 'stabiliser_orders', 'orbit_lengths',
                'labelled_total'):
        if key not in d:
            fail(f'missing key {key!r} (census or --stabilisers pass '
                 f'not run to completion?)')

    n_orb = d['orbits']
    if n_orb != 180:
        fail(f'orbits = {n_orb}, expected 180')

    a = d['assignment']
    if len(a) != N_CAT:
        fail(f'assignment length {len(a)}, expected {N_CAT}')
    if any((not isinstance(x, int)) or x < 0 or x >= n_orb for x in a):
        fail('assignment contains an id outside 0..179')

    sizes = d['catalogue_sizes']
    if len(sizes) != n_orb:
        fail(f'catalogue_sizes length {len(sizes)}, expected {n_orb}')
    cnt = Counter(a)
    if any(cnt.get(k, 0) != sizes[k] for k in range(n_orb)):
        fail('catalogue_sizes disagrees with the assignment')
    if sum(sizes) != N_CAT:
        fail(f'catalogue_sizes sums to {sum(sizes)}, expected {N_CAT}')
    if d['catalogue_sizes_sorted'] != sorted(sizes, reverse=True):
        fail('catalogue_sizes_sorted is not the sorted catalogue_sizes')
    if max(sizes) != 2048:
        fail(f'largest orbit meets catalogue in {max(sizes)}, expected 2048')
    top10 = sum(sorted(sizes, reverse=True)[:10])
    if top10 != 5199:
        fail(f'ten largest meet catalogue in {top10}, expected 5199')

    st = d['stabiliser_orders']
    if len(st) != n_orb:
        fail(f'stabiliser_orders length {len(st)}, expected {n_orb}')
    hist = dict(Counter(st))
    if hist != {3: 142, 6: 32, 12: 4, 24: 1, 72: 1}:
        fail(f'stabiliser histogram {hist}, expected '
             '{3: 142, 6: 32, 12: 4, 24: 1, 72: 1}')
    if any(s % 3 != 0 for s in st):
        fail('a stabiliser order is not divisible by 3')

    ln = d['orbit_lengths']
    if len(ln) != n_orb:
        fail(f'orbit_lengths length {len(ln)}, expected {n_orb}')
    if any(ln[k] * st[k] != GROUP for k in range(n_orb)):
        fail('orbit_lengths[k] * stabiliser_orders[k] != 645120 for some k')
    total = sum(ln)
    if total != 34227200 or d['labelled_total'] != 34227200:
        fail(f'orbit lengths sum to {total} (recorded '
             f'{d["labelled_total"]}), expected 34227200')
    if any(sizes[k] > ln[k] for k in range(n_orb)):
        fail('an orbit meets the catalogue in more members than its length')

    t18 = sorted(sizes[k] for k in range(n_orb) if st[k] in (24, 72))
    if t18 != [46, 55]:
        fail(f'orbits of stabiliser order 24/72 meet catalogue in {t18}, '
             'expected [46, 55] (Type-18)')

    print(f'orbits: {n_orb}; assignment covers {N_CAT} exactly once')
    print(f'stabiliser histogram: {dict(sorted(hist.items()))}')
    print(f'labelled total: {total}; catalogue is '
          f'{100 * N_CAT / total:.4f}% of that')
    print('Type-18 orbits (stabiliser 24, 72) meet catalogue in 55 and 46')
    print('RESULT: matches the paper')
    return 0


if __name__ == '__main__':
    sys.exit(main())
