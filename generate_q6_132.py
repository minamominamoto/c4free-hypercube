#!/usr/bin/env python3
"""Generate and verify a Q6 odd-square witness attaining g(6)=132.

Same construction as generate_q8_682.py: the canonical fully frustrated
coupling J(x, x xor 2^dim) = (-1)^popcount(x & (2^dim - 1)) on Q6, and a
spin configuration found by seeded simulated annealing (search_q6.py,
seed 20260823, reached on the first trial).  The set of edges on which
J(x,y) s_x s_y = +1 is odd-square for any spin assignment; this one
attains 132 edges, the maximum permitted by the field bound.

Standard library only.
"""

import hashlib
import json
from collections import Counter
from itertools import combinations

N = 6
SPINS = (
    "1110000000000000110010111000111010111111000101101100111111100100"
)


def coupling(x, dim):
    """Canonical fully frustrated coupling on edge x--(x xor 2**dim)."""
    return 1 if bin(x & ((1 << dim) - 1)).count("1") % 2 == 0 else -1


def main():
    assert len(SPINS) == 1 << N
    s = [1 if c == "1" else -1 for c in SPINS]
    edges = []
    direction_counts = [0] * N
    degrees = [0] * (1 << N)
    for x in range(1 << N):
        for dim in range(N):
            y = x ^ (1 << dim)
            if x < y and coupling(x, dim) * s[x] * s[y] == 1:
                edges.append([x, y])
                direction_counts[dim] += 1
                degrees[x] += 1
                degrees[y] += 1

    edge_set = {tuple(e) for e in edges}
    square_hist = Counter()
    for d1, d2 in combinations(range(N), 2):
        m1, m2 = 1 << d1, 1 << d2
        for base in range(1 << N):
            if base & (m1 | m2):
                continue
            square = ((base, base | m1), (base, base | m2),
                      (base | m1, base | m1 | m2),
                      (base | m2, base | m1 | m2))
            square_hist[sum(tuple(sorted(e)) in edge_set for e in square)] += 1

    degree_hist = Counter(degrees)
    local_fields = [2 * d - N for d in degrees]
    field_hist = Counter(local_fields)

    # U = even-popcount half of the bipartition (the paper's convention)
    U = [v for v in range(1 << N) if bin(v).count("1") % 2 == 0]
    hU = [2 * degrees[v] - N for v in U]
    fieldU_hist = Counter(hU)

    assert len(edges) == 132
    assert set(square_hist) <= {1, 3}, "not odd-square"
    assert sum(hU) == 72
    assert sum(h * h for h in hU) == N * (1 << (N - 1)) == 192

    witness = {
        "n": N,
        "vertex_encoding": "integers 0..63; coordinate dimensions are bits 0..5",
        "coupling": "J(x,x xor 2^dim)=(-1)^(popcount(x & ((1<<dim)-1)))",
        "spin_encoding": "character x is 1 for spin +1 and 0 for spin -1",
        "spins": SPINS,
        "num_edges": len(edges),
        "direction_counts": direction_counts,
        "degree_histogram": dict(sorted(degree_hist.items())),
        "local_field_histogram": dict(sorted(field_hist.items())),
        "local_field_histogram_U": dict(sorted(fieldU_hist.items())),
        "square_positive_edge_histogram": dict(sorted(square_hist.items())),
        "edges": edges,
    }
    # Force LF so that the advertised byte-level SHA-256 is identical on Windows.
    with open("q6_odd_square_132.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(witness, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    digest = hashlib.sha256(open("q6_odd_square_132.json", "rb").read()).hexdigest()
    print("Q6 witness: PASS")
    print("edges=132 squares=" + str(dict(sorted(square_hist.items()))))
    print("degrees=" + str(dict(sorted(degree_hist.items()))))
    print("h on U =" + str(dict(sorted(fieldU_hist.items()))) +
          "  sum=" + str(sum(hU)) + "  sum of squares=" + str(sum(h * h for h in hU)))
    print("sha256=" + digest)


if __name__ == "__main__":
    main()
