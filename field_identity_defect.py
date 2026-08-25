#!/usr/bin/env python3
"""field_identity_defect.py -- verify the exact defect formula

    sum_{v in U} h(v)^2  =  n^2 * 2^(n-1)  -  4 * sum_s spl_U(s)

for the released certificates, where U is the even-parity side of the
bipartition, h(v) = 2*deg_E(v) - n, the sum runs over all C(n,2)*2^(n-2)
squares s of Q_n, and spl_U(s) in {0,1,2} counts the even-parity corners
of s incident to exactly one of the two s-edges at that corner in E
(the square is "U-split" there).  The symmetric statement for the
odd-parity side U^c is checked as well.

Checked against the paper's stated values:
  * Q6 odd-square witness:  per-square spl_U histogram {1:240} (pointwise),
    both sides, sums 192 = 6*2^5;
  * released q6_edges_132.jsonl (not odd-square): sums 176 on both sides,
    sum spl_U = sum spl_W = 244 = 240 + 4;
  * Q7 catalogue record 0 (not odd-square): histogram {0:60, 1:552, 2:60},
    sum spl_U = 672 = C(7,2)*2^5 exactly, so the identity holds on average
    without holding pointwise; record 34 (odd-square): histogram {1:672};
  * Q8 682-edge odd-square witness and 680-edge Solution A: {1:1792} pointwise;
  * Q8 680-edge Solution B: sum spl_U = 1794 = 1792 + 2 on the failing side
    (sum h^2 = 1016 = 1024 - 4*2) against exactly 1792 on the complement.

The formula is additionally checked exactly on arbitrary edge sets:
for n=4 and n=5, the empty set, the full edge set E(Q_n), a single edge,
and 200 seeded random subsets each (random.Random(20260825), so the run
is deterministic).

Standard library only; deterministic; runs in seconds.
Exit status 0 iff the formula and every stated value check out.
"""
import json
import sys

FAIL = []


def parity(v):
    return bin(v).count("1") & 1


def analyze(n, edges):
    E = set()
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))
    deg = [0] * (1 << n)
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    sums = [0, 0]  # index by parity: [U, U^c]
    for v in range(1 << n):
        sums[parity(v)] += (2 * deg[v] - n) ** 2

    def has(x, y):
        return ((x, y) if x < y else (y, x)) in E

    hists = [[0, 0, 0], [0, 0, 0]]  # per-square split histograms, U and U^c
    for i in range(n):
        for j in range(i + 1, n):
            for w in range(1 << n):
                if (w >> i & 1) or (w >> j & 1):
                    continue
                a, b = w ^ (1 << i), w ^ (1 << j)
                z = a ^ (1 << j)
                corners = ((w, (w, a), (w, b)), (z, (z, a), (z, b)),
                           (a, (w, a), (z, a)), (b, (w, b), (z, b)))
                spl = [0, 0]
                for c, e1, e2 in corners:
                    if has(*e1) != has(*e2):
                        spl[parity(c)] += 1
                hists[0][spl[0]] += 1
                hists[1][spl[1]] += 1
    nsq = (n * (n - 1) // 2) * (1 << (n - 2))
    out = {"n": n, "m": len(E), "nsq": nsq}
    for side, name in ((0, "U"), (1, "Uc")):
        h = hists[side]
        tot = h[1] + 2 * h[2]
        lhs = sums[side]
        rhs = n * n * (1 << (n - 1)) - 4 * tot
        out[f"sum_h2_{name}"] = lhs
        out[f"spl_{name}"] = tot
        out[f"hist_{name}"] = h
        out[f"formula_{name}"] = (lhs == rhs)
    return out


def expect(label, cond, detail):
    status = "OK" if cond else "MISMATCH"
    print(f"  [{status}] {label}: {detail}")
    if not cond:
        FAIL.append(label)


def jsonl_records(path, wanted):
    got = {}
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k in wanted:
                got[k] = json.loads(line)["edges"]
            if len(got) == len(wanted):
                break
    return got


def main():
    # Q6 odd-square witness
    E = json.load(open("q6_odd_square_132.json", encoding="utf-8"))["edges"]
    r = analyze(6, E)
    print("q6_odd_square_132.json:", r)
    expect("Q6 witness formula", r["formula_U"] and r["formula_Uc"], "defect formula both sides")
    expect("Q6 witness pointwise", r["hist_U"] == [0, 240, 0] == r["hist_Uc"],
           f"histograms {r['hist_U']} / {r['hist_Uc']} vs {{1:240}}")
    expect("Q6 witness sums", r["sum_h2_U"] == 192 == r["sum_h2_Uc"], "192 = 6*2^5 both sides")

    # released (non-odd-square) q6_edges_132.jsonl
    E = jsonl_records("q6_edges_132.jsonl", {0})[0]
    r = analyze(6, E)
    print("q6_edges_132.jsonl:", r)
    expect("q6_edges_132 formula", r["formula_U"] and r["formula_Uc"], "defect formula both sides")
    expect("q6_edges_132 values", r["sum_h2_U"] == 176 == r["sum_h2_Uc"]
           and r["spl_U"] == 244 == r["spl_Uc"],
           "sums 176, split totals 244 = 240 + 4")

    # Q7 catalogue records 0 (not odd-square) and 34 (odd-square)
    recs = jsonl_records("q7_edges_304.jsonl.part1", {0, 34})
    r = analyze(7, recs[0])
    print("q7 record 0:", r)
    expect("Q7 rec0 formula", r["formula_U"] and r["formula_Uc"], "defect formula both sides")
    expect("Q7 rec0 average-not-pointwise",
           r["hist_U"] == [60, 552, 60] and r["spl_U"] == 672 and r["sum_h2_U"] == 448,
           f"histogram {r['hist_U']}, split total {r['spl_U']} = C(7,2)*2^5, sum 448")
    r = analyze(7, recs[34])
    print("q7 record 34:", r)
    expect("Q7 rec34 pointwise", r["hist_U"] == [0, 672, 0] == r["hist_Uc"],
           f"histograms {r['hist_U']} / {r['hist_Uc']} vs {{1:672}}")

    # Q8: 682-edge odd-square witness; 680-edge Solutions A and B
    E = json.load(open("q8_odd_square_682.json", encoding="utf-8"))["edges"]
    r = analyze(8, E)
    print("q8_odd_square_682.json:", r)
    expect("Q8 682 pointwise", r["hist_U"] == [0, 1792, 0] == r["hist_Uc"]
           and r["sum_h2_U"] == 1024 == r["sum_h2_Uc"],
           f"histograms {r['hist_U']} / {r['hist_Uc']}, sums 1024 = 8*2^7")

    recs = jsonl_records("q8_edges_680.jsonl", {0, 1})
    r = analyze(8, recs[0])
    print("q8 Solution A (record 0):", r)
    expect("Q8 A pointwise", r["hist_U"] == [0, 1792, 0] == r["hist_Uc"],
           f"histograms {r['hist_U']} / {r['hist_Uc']} vs {{1:1792}}")
    r = analyze(8, recs[1])
    print("q8 Solution B (record 1):", r)
    expect("Q8 B formula", r["formula_U"] and r["formula_Uc"], "defect formula both sides")
    expect("Q8 B one-sided failure",
           r["sum_h2_U"] == 1016 and r["spl_U"] == 1794
           and r["sum_h2_Uc"] == 1024 and r["spl_Uc"] == 1792,
           f"U: sum {r['sum_h2_U']}, split {r['spl_U']} = 1792+2; "
           f"Uc: sum {r['sum_h2_Uc']}, split {r['spl_Uc']} = 1792")

    # Exact formula on arbitrary edge sets: empty, full, single edge, and
    # 200 seeded random subsets of E(Q_n) for n=4,5 (deterministic seed).
    import random
    rng = random.Random(20260825)
    for n in (4, 5):
        all_edges = [(v, v | (1 << i)) for v in range(1 << n)
                     for i in range(n) if not (v >> i) & 1]
        cases = [[], all_edges, [all_edges[0]]]
        for _ in range(200):
            k = rng.randrange(len(all_edges) + 1)
            cases.append(rng.sample(all_edges, k))
        bad = sum(1 for E in cases
                  if not ((r := analyze(n, E))["formula_U"] and r["formula_Uc"]))
        expect(f"Q{n} arbitrary-set formula", bad == 0,
               f"{len(cases)} edge sets (empty, full, single edge, "
               f"200 seeded random): {bad} failures")

    if FAIL:
        print(f"RESULT: MISMATCH against the values stated in the paper ({len(FAIL)})")
        return 1
    print("RESULT: matches the paper (exact defect formula verified on all "
          "released certificates listed above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
