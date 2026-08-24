#!/usr/bin/env python3
"""Generate and verify Minamo Minamoto's reconstructed Q8 odd-square witness."""

import hashlib
import json
from collections import Counter
from itertools import combinations

N = 8
SPINS = (
    "1111101100010011000000111100010000111010000000110101000110111101"
    "1011000011111011100101011110110011101100111011000000010000001111"
    "1101110010110101101101011011110010111100010000110011000111110001"
    "0011010000010011111111000001010100000011111010100100101001100100"
)


def coupling(x, dim):
    """Canonical fully frustrated coupling on edge x--(x xor 2**dim)."""
    return 1 if (x & ((1 << dim) - 1)).bit_count() % 2 == 0 else -1


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
    fieldU_hist = Counter(2 * degrees[v] - N for v in U)
    assert len(edges) == 682
    assert square_hist == Counter({3: 1491, 1: 301})
    assert degree_hist == Counter({5: 168, 6: 86, 4: 2})
    assert field_hist == Counter({2: 168, 4: 86, 0: 2})
    assert sum(local_fields) == 680
    assert sum(h * h for h in local_fields) == 2048

    witness = {
        "n": N,
        "vertex_encoding": "integers 0..255; coordinate dimensions are bits 0..7",
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
    with open("q8_odd_square_682.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(witness, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    digest = hashlib.sha256(open("q8_odd_square_682.json", "rb").read()).hexdigest()
    print("Q8 witness: PASS")
    print("edges=682 squares={1:301,3:1491} degrees={4:2,5:168,6:86}")
    print("sha256=" + digest)


if __name__ == "__main__":
    main()
