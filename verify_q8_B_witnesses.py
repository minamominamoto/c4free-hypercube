#!/usr/bin/env python3
"""Independent deterministic verifier; contains no search/SA code."""
import argparse
import hashlib
import json
from collections import Counter

N = 8
V = 1 << N
TARGET = {2: 87, 4: 40, 6: 1}
TARGET_OTHER_SIDE = {0: 1, 2: 84, 4: 43}
EXPECTED_VIOLATIONS = {
    20261207: {3: 34, 4: 169, 5: 121, 6: 18},
    20261218: {3: 41, 4: 154, 5: 130, 6: 17},
}


def coupling(x, dim):
    return 1 if bin(x & ((1 << dim) - 1)).count("1") % 2 == 0 else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="q8_B_witnesses.json")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)["witnesses"]

    for record in records:
        spins = record["spins_vertex_order_0_to_255"]
        assert len(spins) == V and all(s in (-1, 1) for s in spins)
        fields = {}
        for x in range(V):
            fields[x] = spins[x] * sum(
                coupling(min(x, x ^ (1 << d)), d) * spins[x ^ (1 << d)]
                for d in range(N))
        u = [x for x in range(V) if bin(x).count("1") % 2 == 0]
        w = [x for x in range(V) if bin(x).count("1") % 2 == 1]
        hist = Counter(fields[x] for x in u)
        hist_w = Counter(fields[x] for x in w)
        assert dict(hist) == TARGET
        assert dict(hist_w) == TARGET_OTHER_SIDE
        assert sum(fields[x] ** 2 for x in u) == 1024
        assert sum(fields[x] for x in u) == 340

        edges = []
        edge_set = set()
        for x in range(V):
            for d in range(N):
                y = x ^ (1 << d)
                if x < y and coupling(x, d) * spins[x] * spins[y] == 1:
                    edge = (x, y)
                    edges.append(edge)
                    edge_set.add(edge)
        assert len(edges) == 682
        assert [list(e) for e in edges] == record["positive_edges"]

        square_hist = Counter()
        for x in range(V):
            for d1 in range(N):
                for d2 in range(d1 + 1, N):
                    if (x >> d1) & 1 or (x >> d2) & 1:
                        continue
                    a, b = x ^ (1 << d1), x ^ (1 << d2)
                    c = x ^ (1 << d1) ^ (1 << d2)
                    boundary = ((min(x, a), max(x, a)),
                                (min(x, b), max(x, b)),
                                (min(a, c), max(a, c)),
                                (min(b, c), max(b, c)))
                    square_hist[sum(e in edge_set for e in boundary)] += 1
        assert sum(square_hist.values()) == 1792
        assert set(square_hist) <= {1, 3}

        # For each absent cube edge, count the number of squares whose other
        # three boundary edges are already present.  This is exactly the number
        # of new C4s created by adding that edge.
        nonedge_viol = Counter()
        all_edges = set()
        for x in range(V):
            for d in range(N):
                y = x ^ (1 << d)
                if x < y:
                    all_edges.add((x, y))
        completion_count = Counter()
        for x in range(V):
            for d1 in range(N):
                for d2 in range(d1 + 1, N):
                    if (x >> d1) & 1 or (x >> d2) & 1:
                        continue
                    a, b = x ^ (1 << d1), x ^ (1 << d2)
                    c = x ^ (1 << d1) ^ (1 << d2)
                    boundary = [(min(x, a), max(x, a)),
                                (min(x, b), max(x, b)),
                                (min(a, c), max(a, c)),
                                (min(b, c), max(b, c))]
                    present = [e in edge_set for e in boundary]
                    if sum(present) == 3:
                        completion_count[boundary[present.index(False)]] += 1
        for e in all_edges - edge_set:
            nonedge_viol[completion_count[e]] += 1
        expected_viol = EXPECTED_VIOLATIONS[record["seed"]]
        assert dict(sorted(nonedge_viol.items())) == expected_viol
        assert min(nonedge_viol) >= 3

        spin_bytes = bytes(1 if s == 1 else 0 for s in spins)
        edge_text = "\n".join(f"{x},{y}" for x, y in edges).encode("ascii")
        assert hashlib.sha256(spin_bytes).hexdigest() == record["spin_bits_sha256"]
        assert hashlib.sha256(edge_text).hexdigest() == record["positive_edges_csv_sha256"]
        print("PASS", record["seed"], "hist_U", dict(sorted(hist.items())),
              "hist_Uc", dict(sorted(hist_w.items())),
              "sum_h2", 1024, "edges", len(edges),
              "squares", dict(sorted(square_hist.items())),
              "nonedge_violations", dict(sorted(nonedge_viol.items())))


if __name__ == "__main__":
    main()
