#!/usr/bin/env python3
"""Exact Type-18 automorphism audit for the released Q7 solutions.

This script verifies the two stronger Type-18 claims in the manuscript that
``analyze_q7_structure.py`` previously did not check:

  1. every one of the 101 solutions with dimension profile
     [48,48,48,40,40,40,40] has an Aut(Q7) automorphism whose coordinate
     action nontrivially permutes the three 48-edge directions;
  2. exactly 46 of the 101 have an order-3 automorphism that cycles all three
     48-edge directions.

Completeness of the enumeration
-------------------------------
Every automorphism of Q7 has the form

    g(v) = P(v) xor a,

where ``a`` is one of the 2^7 translations and ``P`` permutes the seven
coordinate directions.  If g fixes an edge set, P must preserve the number of
selected edges in each direction.  For Type 18 the direction counts are three
48s and four 40s, so P is necessarily in S3 x S4: only 3!*4! = 144 coordinate
permutations, not all 7!, need to be tested.  For each of those 144
permutations this script checks all compatible XOR translations exactly.
Thus the search is exhaustive over the full Aut(Q7), not heuristic.

Permutation convention
----------------------
A permutation is stored as a tuple p with ``p[d] = new direction of old
direction d``.  The affine automorphism is then ``v -> P(v) xor a``.

Standard library only. Python 3.9+.

Usage:
    python3 type18_automorphisms.py

or explicitly:
    python3 type18_automorphisms.py q7_edges_304.jsonl.part1 \
        q7_edges_304.jsonl.part2 q7_edges_304.jsonl.part3

By default a detailed CSV is written to ``q7_type18_automorphisms_101.csv``.
Use ``--no-csv`` to suppress it or ``--csv PATH`` to choose a path.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

N = 7
NUM_VERTICES = 1 << N
TYPE18_PROFILE = (48, 48, 48, 40, 40, 40, 40)
EXPECTED_TYPE18 = 101
EXPECTED_ORDER3 = 46
IDENTITY = tuple(range(N))


def normalize_edges(raw_edges):
    edges = frozenset(
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in raw_edges
    )
    if len(edges) != len(raw_edges):
        raise ValueError("duplicate edge in solution")
    return edges


def load_solutions(paths):
    records = []
    global_index = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(
                    {
                        "global_index": global_index,
                        "source_file": Path(path).name,
                        "line_number": line_number,
                        "edges": normalize_edges(obj["edges"]),
                    }
                )
                global_index += 1
    return records


def direction_counts(edges):
    counts = [0] * N
    for u, v in edges:
        x = u ^ v
        if x == 0 or x & (x - 1):
            raise ValueError(f"non-Q7 edge {(u, v)}")
        counts[x.bit_length() - 1] += 1
    return tuple(counts)


def sorted_profile(edges):
    return tuple(sorted(direction_counts(edges), reverse=True))


def permute_vertex(v, p):
    out = 0
    for old_direction in range(N):
        if (v >> old_direction) & 1:
            out |= 1 << p[old_direction]
    return out


def permute_edges(edges, p):
    vertex_map = [permute_vertex(v, p) for v in range(NUM_VERTICES)]
    return frozenset(
        (min(vertex_map[u], vertex_map[v]), max(vertex_map[u], vertex_map[v]))
        for u, v in edges
    )


def translate_edges(edges, mask):
    return frozenset(
        (min(u ^ mask, v ^ mask), max(u ^ mask, v ^ mask)) for u, v in edges
    )


def permute_mask(mask, p):
    out = 0
    for old_direction in range(N):
        if (mask >> old_direction) & 1:
            out |= 1 << p[old_direction]
    return out


def compose_permutations(p, q):
    """Return p after q under the old-direction -> new-direction convention."""
    return tuple(p[q[d]] for d in range(N))


def permutation_power(p, exponent):
    out = IDENTITY
    for _ in range(exponent):
        out = compose_permutations(p, out)
    return out


def affine_has_order_three(p, mask):
    """Exact test for g(v)=P(v) xor mask having order exactly 3."""
    if p == IDENTITY and mask == 0:
        return False
    if permutation_power(p, 3) != IDENTITY:
        return False
    p_mask = permute_mask(mask, p)
    p2_mask = permute_mask(p_mask, p)
    # g^3(v) = P^3(v) xor mask xor P(mask) xor P^2(mask)
    return (mask ^ p_mask ^ p2_mask) == 0


def profile_preserving_permutations(high_directions, low_directions):
    """Enumerate exactly S3 x S4 for the Type-18 direction-count partition."""
    for high_image in itertools.permutations(high_directions):
        for low_image in itertools.permutations(low_directions):
            p = list(range(N))
            for old, new in zip(high_directions, high_image):
                p[old] = new
            for old, new in zip(low_directions, low_image):
                p[old] = new
            yield tuple(p)


def stabilizer(edges):
    """Enumerate the complete affine stabilizer of one Type-18 edge set.

    We precompute all 128 translations T_a(E).  For each of the only 144
    direction-count-compatible coordinate permutations P, P(E) is looked up
    in that table.  If P(E)=T_a(E), then T_a P fixes E because T_a is its own
    inverse in (Z_2)^7.  Multiple masks may represent the same translated
    edge set when E itself has translation symmetry, so all masks are retained.
    """
    counts = direction_counts(edges)
    high = tuple(d for d, c in enumerate(counts) if c == 48)
    low = tuple(d for d, c in enumerate(counts) if c == 40)
    if len(high) != 3 or len(low) != 4:
        raise ValueError(f"not Type 18 direction counts: {counts}")

    translated_to_masks = {}
    for mask in range(NUM_VERTICES):
        translated_to_masks.setdefault(translate_edges(edges, mask), []).append(mask)

    automorphisms = []
    for p in profile_preserving_permutations(high, low):
        pe = permute_edges(edges, p)
        for mask in translated_to_masks.get(pe, ()):  # P(E)=T_mask(E)
            automorphisms.append((p, mask))

    # Identity must appear exactly once as an affine group element.
    assert (IDENTITY, 0) in automorphisms
    assert len(set(automorphisms)) == len(automorphisms)
    return counts, high, low, automorphisms


def cycle_notation_on_subset(p, subset):
    """Compact readable cycle notation for the coordinate action on subset."""
    subset = set(subset)
    seen = set()
    cycles = []
    for start in sorted(subset):
        if start in seen:
            continue
        cur = start
        cyc = []
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = p[cur]
        if len(cyc) > 1:
            cycles.append("(" + " ".join(map(str, cyc)) + ")")
    return "".join(cycles) or "()"


def audit(paths, csv_path=None):
    records = load_solutions(paths)
    if len(records) != 19866:
        raise AssertionError(f"expected 19866 total solutions, got {len(records)}")

    type18 = [r for r in records if sorted_profile(r["edges"]) == TYPE18_PROFILE]
    if len(type18) != EXPECTED_TYPE18:
        raise AssertionError(
            f"expected {EXPECTED_TYPE18} Type-18 solutions, got {len(type18)}"
        )

    rows = []
    all_direction_nontrivial = 0
    with_order3_cycle = 0
    stabilizer_histogram = Counter()

    for ordinal, record in enumerate(type18, 1):
        counts, high, low, auts = stabilizer(record["edges"])
        stabilizer_histogram[len(auts)] += 1

        direction_movers = [
            (p, a) for p, a in auts if any(p[d] != d for d in high)
        ]
        order3_cycles = [
            (p, a)
            for p, a in auts
            if all(p[d] != d for d in high)
            and affine_has_order_three(p, a)
        ]

        if direction_movers:
            all_direction_nontrivial += 1
        if order3_cycles:
            with_order3_cycle += 1

        witness_dir = direction_movers[0] if direction_movers else None
        witness_o3 = order3_cycles[0] if order3_cycles else None
        rows.append(
            {
                "type18_ordinal_one_based": ordinal,
                "global_index_zero_based": record["global_index"],
                "source_file": record["source_file"],
                "line_number_one_based": record["line_number"],
                "direction_counts_bits_0_to_6": ";".join(map(str, counts)),
                "high_48_directions": ";".join(map(str, high)),
                "stabilizer_size": len(auts),
                "direction_moving_automorphisms": len(direction_movers),
                "order3_high_cycle_automorphisms": len(order3_cycles),
                "direction_witness_permutation_old_to_new": (
                    ";".join(map(str, witness_dir[0])) if witness_dir else ""
                ),
                "direction_witness_xor_mask": witness_dir[1] if witness_dir else "",
                "order3_witness_permutation_old_to_new": (
                    ";".join(map(str, witness_o3[0])) if witness_o3 else ""
                ),
                "order3_witness_xor_mask": witness_o3[1] if witness_o3 else "",
                "order3_witness_high_cycle": (
                    cycle_notation_on_subset(witness_o3[0], high) if witness_o3 else ""
                ),
            }
        )

    if all_direction_nontrivial != EXPECTED_TYPE18:
        raise AssertionError(
            "direction-permuting automorphism count mismatch: "
            f"{all_direction_nontrivial}/{EXPECTED_TYPE18}"
        )
    if with_order3_cycle != EXPECTED_ORDER3:
        raise AssertionError(
            f"order-3 count mismatch: {with_order3_cycle}/{EXPECTED_TYPE18}, "
            f"expected {EXPECTED_ORDER3}/{EXPECTED_TYPE18}"
        )

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("Type-18 automorphism audit: PASS")
    print(f"total_q7_solutions={len(records)} type18={len(type18)}")
    print(
        "direction-permuting automorphism exists: "
        f"{all_direction_nontrivial}/{EXPECTED_TYPE18}"
    )
    print(
        "order-3 automorphism cycling the three 48-edge directions: "
        f"{with_order3_cycle}/{EXPECTED_TYPE18}"
    )
    print(
        "full stabilizer-size histogram: "
        + str(dict(sorted(stabilizer_histogram.items())))
    )
    if csv_path:
        print(f"details_csv={csv_path}")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Q7 JSONL parts; defaults to q7_edges_304.jsonl.part1-3",
    )
    parser.add_argument(
        "--csv",
        default="q7_type18_automorphisms_101.csv",
        help="write per-solution details here (default: %(default)s)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="do not write a CSV",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = args.paths or [
        "q7_edges_304.jsonl.part1",
        "q7_edges_304.jsonl.part2",
        "q7_edges_304.jsonl.part3",
    ]
    csv_path = None if args.no_csv else args.csv
    audit(paths, csv_path=csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
