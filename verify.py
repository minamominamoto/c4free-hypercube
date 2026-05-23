#!/usr/bin/env python3
"""
verify.py — dependency-free verifier for the C4-free hypercube constructions.

Standard library only (json, hashlib, itertools, sys); Python 3.8+.
No third-party packages, no network access.

For each n in {6, 7, 8} this script:
  - reads every solution (JSON, key "edges": a list of [u, v] integer pairs,
    vertices in 0 .. 2^n - 1; one object per line for the multi-solution files);
  - checks every edge is a valid Q_n edge (endpoints differ in exactly one bit),
    with no loops, no duplicates, and exactly the claimed edge count;
  - certifies C4-freeness by EXHAUSTIVELY enumerating all four-cycles of Q_n
    ( C(n,2) * 2^(n-2) of them: 240 for Q6, 672 for Q7, 1792 for Q8 )
    and confirming none of them is fully present;
  - prints the SHA-256 of each data file as a fixed-version certificate.

Usage:
    python3 verify.py

Exit code is 0 iff every solution passes every check.
"""

import json
import hashlib
import sys
from itertools import combinations

# (n, expected_edges_per_solution, expected_number_of_solutions, data_files)
TARGETS = [
    (6, 132, 1, ["q6_edges_132.jsonl"]),
    (7, 304, 19866, [
        "q7_edges_304.jsonl.part1",
        "q7_edges_304.jsonl.part2",
        "q7_edges_304.jsonl.part3",
    ]),
    (8, 680, 2, ["q8_edges_680.jsonl"]),
]


def four_cycle_corners(n):
    """Yield (a, b, c, d) for every potential 4-cycle of Q_n.

    A 4-cycle is fixed by a base vertex with two 'free' dimensions d1 < d2
    (both bits 0 in base); its vertices are base, base|d1, base|d2, base|d1|d2.
    There are C(n,2) * 2^(n-2) such cycles.
    """
    for d1, d2 in combinations(range(n), 2):
        m1, m2 = 1 << d1, 1 << d2
        for base in range(1 << n):
            if base & m1 or base & m2:
                continue
            yield base, base | m1, base | m2, base | m1 | m2


def build_edge_set(edges):
    es = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            raise ValueError("self-loop at vertex %d" % u)
        es.add((u, v) if u < v else (v, u))
    if len(es) != len(edges):
        raise ValueError("duplicate edges present")
    return es


def is_hypercube_edge(u, v):
    x = u ^ v
    return x != 0 and (x & (x - 1)) == 0  # exactly one bit differs


def verify_solution(edges, n, expected_edges):
    es = build_edge_set(edges)
    if len(es) != expected_edges:
        return False, "edge count %d != %d" % (len(es), expected_edges)
    N = 1 << n
    for u, v in es:
        if not (0 <= u < N and 0 <= v < N):
            return False, "vertex out of range in edge (%d,%d)" % (u, v)
        if not is_hypercube_edge(u, v):
            return False, "(%d,%d) is not a Q_%d edge" % (u, v, n)
    for a, b, c, d in four_cycle_corners(n):
        if (a, b) in es and (a, c) in es and (b, d) in es and (c, d) in es:
            return False, "C4 found on vertices %d,%d,%d,%d" % (a, b, d, c)
    return True, "ok"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_solutions(paths):
    """Return a list of edge-lists. Tolerates both a single-object file
    ({"edges": [...]}) and newline-delimited JSON (one object per line)."""
    sols = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        try:
            obj = json.loads(text)  # whole file is one JSON value
            if isinstance(obj, dict) and "edges" in obj:
                sols.append(obj["edges"])
                continue
            if isinstance(obj, list):  # a JSON array of objects
                sols.extend(o["edges"] for o in obj)
                continue
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():  # JSON-lines fallback
            line = line.strip()
            if line:
                sols.append(json.loads(line)["edges"])
    return sols


def main():
    all_ok = True
    print("C4-free hypercube verification")
    print("=" * 56)
    for n, ec, nsol, paths in TARGETS:
        ncycles = sum(1 for _ in four_cycle_corners(n))
        print("\nQ%d: expecting %d solution(s), %d edges each; "
              "%d four-cycles checked per solution"
              % (n, nsol, ec, ncycles))
        missing = False
        for p in paths:
            try:
                print("  sha256  %s  %s" % (sha256(p), p))
            except FileNotFoundError:
                print("  [MISSING] %s" % p)
                missing = True
                all_ok = False
        if missing:
            continue
        sols = load_solutions(paths)
        if len(sols) != nsol:
            print("  [FAIL] found %d solution(s), expected %d"
                  % (len(sols), nsol))
            all_ok = False
        bad = 0
        for i, edges in enumerate(sols):
            ok, msg = verify_solution(edges, n, ec)
            if not ok:
                bad += 1
                if bad <= 5:
                    print("  [FAIL] solution %d: %s" % (i, msg))
        if bad == 0 and len(sols) == nsol:
            print("  [OK] all %d solution(s) are C4-free with exactly %d edges"
                  % (len(sols), ec))
        else:
            all_ok = False
    print("\n" + "=" * 56)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "FAILURES DETECTED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
