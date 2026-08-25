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

Additionally, for the two dedicated odd-square witnesses (q6_odd_square_132.json and
q8_odd_square_682.json) this script independently checks the stronger
odd-square condition: every square must meet the edge set in exactly 1 or 3
edges (not merely "not 4"). These checks are independent of the self-checks
performed by generate_q6_132.py and generate_q8_682.py; this script does not
import or reuse either script's logic. For the Q7 catalogue it also checks the
two inexpensive structural assertions stated in the paper: every one of
the 19,866 solutions is locally maximal in Q7, and their union covers all 448
Q7 edges. It also asserts catalogue-level pairwise distinctness: the 19,866
normalised edge sets are all different from one another (checked via SHA-256
over each sorted edge list; this is a different check from the per-solution
"no duplicates" above, which only forbids a repeated edge INSIDE one
solution). It also independently confirms that the first released 680-edge Q8
solution (Solution A) is odd-square.

Note that q6_edges_132.jsonl and q6_odd_square_132.json are two DIFFERENT
132-edge C4-free subgraphs of Q6. Only the latter is odd-square; the former
is checked for C4-freeness alone, above.

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

# Odd-square targets are checked separately (stronger condition than C4-free).
ODDSQUARE_TARGETS = [
    (6, 132, 1, ["q6_odd_square_132.json"]),
    (8, 682, 1, ["q8_odd_square_682.json"]),
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


def all_hypercube_edges(n):
    """Return every undirected edge of Q_n as a normalized (u,v) pair."""
    out = set()
    for u in range(1 << n):
        for d in range(n):
            v = u ^ (1 << d)
            if u < v:
                out.add((u, v))
    return out


def completion_triples_by_edge(n):
    """For each Q_n edge e, list the triples of other edges that with e
    complete a square.  An absent edge can be added C4-freely iff none of its
    completion triples is fully present."""
    out = {e: [] for e in all_hypercube_edges(n)}
    for a, b, c, d in four_cycle_corners(n):
        square = [(a, b), (a, c), (b, d), (c, d)]
        square = [(u, v) if u < v else (v, u) for u, v in square]
        for i, e in enumerate(square):
            out[e].append(tuple(square[:i] + square[i + 1:]))
    return out


def locally_maximal_in_cube(es, cube_edges, completion):
    """Return (True,None) iff every missing cube edge completes a C4."""
    for e in cube_edges - es:
        if not any(all(x in es for x in triple) for triple in completion[e]):
            return False, e
    return True, None


def nonedge_violation_distribution(es, cube_edges, completion):
    """Histogram: for each missing cube edge, how many C4s its addition completes."""
    hist = {}
    for e in cube_edges - es:
        k = sum(all(x in es for x in triple) for triple in completion[e])
        hist[k] = hist.get(k, 0) + 1
    return dict(sorted(hist.items()))


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


def verify_odd_square(edges, n, expected_edges):
    """Independent check of the odd-square condition: every square of Q_n
    must meet the edge set in exactly 1 or 3 edges. This subsumes the
    plain C4-free check (a square with 4 edges present fails here too)
    and additionally rejects squares with 0 or 2 edges present."""
    es = build_edge_set(edges)
    if len(es) != expected_edges:
        return False, "edge count %d != %d" % (len(es), expected_edges)
    N = 1 << n
    for u, v in es:
        if not (0 <= u < N and 0 <= v < N):
            return False, "vertex out of range in edge (%d,%d)" % (u, v)
        if not is_hypercube_edge(u, v):
            return False, "(%d,%d) is not a Q_%d edge" % (u, v, n)
    bad = 0
    for a, b, c, d in four_cycle_corners(n):
        cnt = ((a, b) in es) + ((a, c) in es) + ((b, d) in es) + ((c, d) in es)
        if cnt not in (1, 3):
            bad += 1
            if bad <= 3:
                print("    [FAIL] square %d,%d,%d,%d has %d edges present "
                      "(neither 1 nor 3)" % (a, b, d, c, cnt))
    if bad:
        return False, "%d square(s) violate the odd-square condition" % bad
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
        cube_edges = all_hypercube_edges(n)
        completion = completion_triples_by_edge(n)
        q7_union = set() if n == 7 else None
        q7_nonmax = 0
        q7_keys = set()
        q7_dupes = 0
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
                continue
            if n == 7:
                es = build_edge_set(edges)
                q7_union.update(es)
                # Catalogue-level pairwise distinctness: hash the sorted,
                # normalised edge list; a repeated digest is reported as a
                # duplicate. This compares solutions to EACH OTHER, unlike the
                # per-solution "no duplicates" check above, which only forbids
                # a repeated edge inside one solution.
                key = hashlib.sha256(
                    ";".join("%d,%d" % e for e in sorted(es)).encode()
                ).digest()
                if key in q7_keys:
                    q7_dupes += 1
                    if q7_dupes <= 5:
                        print("  [FAIL] solution %d duplicates an earlier "
                              "solution's edge set" % i)
                else:
                    q7_keys.add(key)
                local_ok, addable_edge = locally_maximal_in_cube(
                    es, cube_edges, completion
                )
                if not local_ok:
                    q7_nonmax += 1
                    if q7_nonmax <= 5:
                        print("  [FAIL] solution %d is not locally maximal; "
                              "edge %r can be added C4-freely" % (i, addable_edge))
        if n == 7:
            if q7_nonmax:
                print("  [FAIL] %d Q7 solution(s) are not locally maximal" % q7_nonmax)
                all_ok = False
            elif len(sols) == nsol:
                print("  [OK] all %d Q7 solutions are locally maximal" % nsol)
            if q7_union != cube_edges:
                print("  [FAIL] Q7 catalogue union covers %d/%d cube edges"
                      % (len(q7_union), len(cube_edges)))
                all_ok = False
            elif len(sols) == nsol:
                print("  [OK] Q7 catalogue union covers all %d cube edges"
                      % len(cube_edges))
            if q7_dupes:
                print("  [FAIL] %d duplicate edge set(s) across the catalogue"
                      % q7_dupes)
                all_ok = False
            elif len(sols) == nsol:
                print("  [OK] all %d Q7 edge sets are pairwise distinct "
                      "across the catalogue" % len(q7_keys))
        if n == 8 and ec == 680 and len(sols) >= 1:
            odd_ok, odd_msg = verify_odd_square(sols[0], 8, 680)
            if odd_ok:
                print("  [OK] Q8 Solution A (first 680-edge record) is odd-square")
            else:
                print("  [FAIL] Q8 Solution A odd-square check: %s" % odd_msg)
                all_ok = False
            expected_v = [
                {2: 3, 3: 48, 4: 144, 5: 136, 6: 13},
                {1: 1, 2: 1, 3: 49, 4: 153, 5: 124, 6: 16},
            ]
            for j, exp in enumerate(expected_v[:len(sols)]):
                got = nonedge_violation_distribution(
                    build_edge_set(sols[j]), cube_edges, completion
                )
                if got != exp:
                    print("  [FAIL] Q8 680 solution %d non-edge violation "
                          "distribution %r != %r" % (j, got, exp))
                    all_ok = False
                else:
                    print("  [OK] Q8 680 solution %d non-edge violation %r"
                          % (j, got))
        if bad == 0 and len(sols) == nsol:
            print("  [OK] all %d solution(s) are C4-free with exactly %d edges"
                  % (len(sols), ec))
        else:
            all_ok = False

    print("\n" + "-" * 56)
    print("Odd-square condition (independent of the C4-free check above)")
    for n, ec, nsol, paths in ODDSQUARE_TARGETS:
        ncycles = sum(1 for _ in four_cycle_corners(n))
        print("\nQ%d odd-square: expecting %d solution(s), %d edges each; "
              "%d squares checked per solution"
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
        cube_edges = all_hypercube_edges(n)
        completion = completion_triples_by_edge(n)
        for i, edges in enumerate(sols):
            ok, msg = verify_odd_square(edges, n, ec)
            if not ok:
                bad += 1
                print("  [FAIL] solution %d: %s" % (i, msg))
                continue
            expected_margin = None
            if n == 6 and ec == 132:
                expected_margin = {2: 2, 3: 29, 4: 26, 5: 3}
            elif n == 8 and ec == 682:
                expected_margin = {3: 41, 4: 159, 5: 120, 6: 22}
            if expected_margin is not None:
                got = nonedge_violation_distribution(
                    build_edge_set(edges), cube_edges, completion
                )
                if got != expected_margin:
                    bad += 1
                    print("  [FAIL] solution %d non-edge violation "
                          "distribution %r != %r" %
                          (i, got, expected_margin))
                else:
                    print("  [OK] solution %d non-edge violation %r" %
                          (i, got))
        if bad == 0 and len(sols) == nsol:
            print("  [OK] all %d solution(s) satisfy the odd-square "
                  "condition with exactly %d edges" % (len(sols), ec))
        else:
            all_ok = False

    print("\n" + "=" * 56)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "FAILURES DETECTED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
