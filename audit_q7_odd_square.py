#!/usr/bin/env python3
"""Identify odd-square members of Minamoto's 19,866 published Q7 solutions."""

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import combinations


def squares(n):
    for d1, d2 in combinations(range(n), 2):
        m1, m2 = 1 << d1, 1 << d2
        for base in range(1 << n):
            if not base & (m1 | m2):
                yield ((base, base | m1), (base, base | m2),
                       (base | m1, base | m1 | m2),
                       (base | m2, base | m1 | m2))


def main(paths):
    rows = []
    global_index = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                obj = json.loads(line)
                edges = {tuple(sorted(map(int, e))) for e in obj["edges"]}
                hist = Counter(sum(tuple(sorted(e)) in edges for e in sq)
                               for sq in squares(7))
                profile = [0] * 7
                for u, v in edges:
                    profile[(u ^ v).bit_length() - 1] += 1
                if hist[0] == 0 and hist[2] == 0 and hist[4] == 0:
                    canonical = json.dumps(sorted(edges), separators=(",", ":"))
                    rows.append({
                        "global_index_zero_based": global_index,
                        "source_file": os.path.basename(path),
                        "line_number_one_based": line_number,
                        "solution_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        "direction_counts_bits_0_to_6": ";".join(map(str, profile)),
                        "dimension_profile_sorted": ";".join(map(str, sorted(profile, reverse=True))),
                        "squares_with_1_edge": hist[1],
                        "squares_with_3_edges": hist[3],
                    })
                global_index += 1
    assert global_index == 19866, global_index
    assert len(rows) == 389, len(rows)
    with open("q7_odd_square_389.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    profiles = Counter(r["dimension_profile_sorted"] for r in rows)
    assert len(profiles) == 4
    print("Q7 audit: PASS")
    print("solutions=19866 odd_square=389 non_odd_square=19477")
    print("odd-square profiles:")
    for profile, count in sorted(profiles.items()):
        print("  %s : %d" % (profile, count))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: audit_q7_odd_square.py q7_edges_304.jsonl.part1 ...")
    main(sys.argv[1:])
