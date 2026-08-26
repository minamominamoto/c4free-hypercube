#!/usr/bin/env python3
"""
verify_q6_ilp_map.py -- machine-check the provenance claim that the
released Q6 ILP artefacts are what the paper says they are:

  * q6_ilp_edge_map.csv lists exactly 192 variables mapping bijectively
    onto the 192 edges of Q_6 (every listed pair is a genuine Q_6 edge,
    no duplicates, none missing);
  * q6_ilp.mps contains exactly those 192 variables and exactly 240
    C4-constraint rows, and the four variables appearing in each
    constraint are precisely the four edges of one square of Q_6, with
    the 240 constraints hitting the 240 squares bijectively.

Standard library only; read-only; exit status 0 iff every check passes.
Added in response to a pre-release review noting these bijections were
asserted in the text but not packaged as a dedicated verifier.
"""
import csv
import sys
from itertools import combinations

FAIL = []


def expect(name, ok, detail=''):
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}" +
          (f": {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def main():
    # --- CSV: variables <-> edges of Q6, bijectively -------------------
    var2edge = {}
    with open('q6_ilp_edge_map.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            u, v = int(row['vertex_u']), int(row['vertex_v'])
            var2edge[row['variable']] = (u, v) if u < v else (v, u)
    q6_edges = {(v, v ^ (1 << d)) for v in range(64) for d in range(6)
                if v < v ^ (1 << d)}
    expect('CSV: 192 distinct variables', len(var2edge) == 192,
           f'{len(var2edge)}')
    expect('CSV: all pairs are genuine Q6 edges',
           set(var2edge.values()) <= q6_edges)
    expect('CSV: edges pairwise distinct and cover all 192',
           set(var2edge.values()) == q6_edges,
           f'{len(set(var2edge.values()))} distinct')

    # --- MPS: rows and columns ----------------------------------------
    section = None
    rows = set()
    row_vars = {}
    with open('q6_ilp.mps', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            if not line[0].isspace():
                section = line.split()[0]
                continue
            parts = line.split()
            if section == 'ROWS':
                if parts[0] in ('L', 'G', 'E'):
                    rows.add(parts[1])
            elif section == 'COLUMNS':
                if "'MARKER'" in line or 'MARKER' in parts[1:2]:
                    continue          # integrality marker lines, not variables
                var = parts[0]
                for rname, _coef in zip(parts[1::2], parts[2::2]):
                    if rname == 'OBJ':
                        continue
                    row_vars.setdefault(rname, set()).add(var)
    expect('MPS: exactly 240 C4 constraint rows',
           len(rows) == 240, f'{len(rows)}')
    expect('MPS: exactly the 192 CSV variables appear in constraints',
           set().union(*row_vars.values()) == set(var2edge), '')

    squares = set()
    for i, j in combinations(range(6), 2):
        for b in range(64):
            if b >> i & 1 or b >> j & 1:
                continue
            c1, c2, c3 = b ^ (1 << i), b ^ (1 << j), b ^ (1 << i) ^ (1 << j)
            e = lambda x, y: (x, y) if x < y else (y, x)
            squares.add(frozenset(
                {e(b, c1), e(b, c2), e(c1, c3), e(c2, c3)}))
    expect('Q6 has 240 squares (sanity)', len(squares) == 240)

    bad = 0
    seen = set()
    for rname, vs in row_vars.items():
        if len(vs) != 4:
            bad += 1
            continue
        sq = frozenset(var2edge[v] for v in vs)
        if sq not in squares:
            bad += 1
        else:
            seen.add(sq)
    expect('MPS: every constraint is exactly the 4 edges of one square',
           bad == 0, f'{bad} failures')
    expect('MPS: the 240 constraints hit the 240 squares bijectively',
           len(seen) == 240, f'{len(seen)} distinct squares')

    if FAIL:
        print(f'RESULT: MISMATCH ({len(FAIL)})')
        return 1
    print('RESULT: matches the paper (variable and constraint maps are '
          'the stated bijections)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
