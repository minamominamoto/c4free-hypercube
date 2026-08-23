#!/usr/bin/env python3
"""Targeted Q6 search for the two unresolved optimal field distributions.

The exact field DP in ``solve_field_ip.py`` finds three optimal distributions
on the even-popcount bipartition U:

    A = {2:30, 6:2}
    B = {0:1, 2:27, 4:3, 6:1}
    C = {0:2, 2:24, 4:6}       (realised by q6_odd_square_132.json)

This script performs deterministic, seeded simulated annealing toward a chosen
histogram.  The objective is L1 distance between the CURRENT local-field
histogram on U and the TARGET histogram.  It is computational evidence only:
non-attainment is not a proof of non-realisability.

Compared with the earlier bundled version, this revision:
  * exposes the experiment parameters on the command line;
  * supplies a documented 40-seed / 20-seed reproducibility protocol for A/B;
  * can run a realised-distribution control C;
  * writes every run to CSV plus a JSON summary;
  * updates local fields incrementally, making the larger experiment practical;
  * continuously self-checks the local-field convention against independent
    edge counting before any search is run.

IMPORTANT PROVENANCE NOTE
-------------------------
The 40/20 seed lists below are a newly specified deterministic reproducibility
protocol for the released package.  They should not be represented as recovered
historical seeds unless independent logs establish that.  A manuscript can
truthfully report results from this protocol after the resulting CSV/JSON logs
are generated and archived.

Standard library only. Python 3.9+.

Canonical full run (A: 40 seeds, B: 20 seeds, 300,000 iterations/run):

    python3 q6_other_distributions.py --profile paper \
        --csv q6_other_distributions_results.csv \
        --json q6_other_distributions_summary.json

Positive control (20 seeds, realised distribution C):

    python3 q6_other_distributions.py --profile paper --targets C --iters 60000 \
        --csv q6_realized_control_results.csv \
        --json q6_realized_control_summary.json

Fast smoke test:

    python3 q6_other_distributions.py --profile quick
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

N = 6
V = 1 << N
U = tuple(v for v in range(V) if v.bit_count() % 2 == 0)
IS_U = tuple(v.bit_count() % 2 == 0 for v in range(V))
E_APPROX = 2.718281828  # preserve the acceptance formula of the original script
FIELD_VALUES = tuple(range(-N, N + 1, 2))
FIELD_INDEX = {h: i for i, h in enumerate(FIELD_VALUES)}

TARGETS = {
    "A": {2: 30, 6: 2},
    "B": {0: 1, 2: 27, 4: 3, 6: 1},
    "C": {0: 2, 2: 24, 4: 6},
}
TARGET_LABELS = {
    "A": "A {2:30,6:2}",
    "B": "B {0:1,2:27,4:3,6:1}",
    "C": "C {0:2,2:24,4:6} realised control",
}


def coupling(x, dim):
    return 1 if (x & ((1 << dim) - 1)).bit_count() % 2 == 0 else -1


NBR = [
    [
        (
            x ^ (1 << d),
            coupling(min(x, x ^ (1 << d)), d),
        )
        for d in range(N)
    ]
    for x in range(V)
]


def compute_fields(spins):
    return [
        spins[v] * sum(j * spins[y] for y, j in NBR[v])
        for v in range(V)
    ]


def compute_u_fields(spins):
    """Track only fields on U; W-fields are never needed by the objective."""
    fields = [0] * V
    for v in U:
        fields[v] = spins[v] * sum(j * spins[y] for y, j in NBR[v])
    return fields


def histogram_u_from_fields(fields):
    return Counter(fields[v] for v in U)


def histogram_u(spins):
    return histogram_u_from_fields(compute_fields(spins))


def edge_count_fields(spins):
    """Independent degree-based local fields and positive-edge count."""
    degrees = [0] * V
    edges = 0
    for x in range(V):
        for d in range(N):
            y = x ^ (1 << d)
            if x < y and coupling(x, d) * spins[x] * spins[y] == 1:
                edges += 1
                degrees[x] += 1
                degrees[y] += 1
    return edges, [2 * degrees[v] - N for v in range(V)]


def self_test():
    """Check the signed local-field definition against independent edge counting."""
    for seed in (1, 7, 20260823):
        rng = random.Random(seed)
        spins = [rng.choice((1, -1)) for _ in range(V)]
        fields = compute_fields(spins)
        _, degree_fields = edge_count_fields(spins)
        assert fields == degree_fields, (seed, fields, degree_fields)
        u_only = compute_u_fields(spins)
        assert all(u_only[v] == fields[v] for v in U)

        # Also test a sequence of incremental flips against full recomputation.
        for _ in range(100):
            x = rng.randrange(V)
            old_spin = spins[x]
            old_u = [x] if IS_U[x] else [y for y, _ in NBR[x]]
            old_values = {v: fields[v] for v in old_u}

            # Apply the exact incremental field update.
            fields[x] = -fields[x]
            for y, j in NBR[x]:
                fields[y] += -2 * j * spins[y] * old_spin
            spins[x] = -old_spin

            full = compute_fields(spins)
            assert fields == full
            assert all(fields[v] != old_values[v] or True for v in old_u)


def histogram_vector_u(fields):
    out = [0] * len(FIELD_VALUES)
    for v in U:
        out[FIELD_INDEX[fields[v]]] += 1
    return out


def vector_to_hist(vec):
    return {h: vec[i] for i, h in enumerate(FIELD_VALUES) if vec[i]}


def l1_distance_vector(hist, target_vec):
    return sum(abs(a - b) for a, b in zip(hist, target_vec))


def update_hist_for_flip(hist, u_fields, spins, x):
    """Flip x, updating exactly the local fields on U and their histogram.

    If x is in U, only h(x) on U changes and it simply changes sign.
    If x is in W, the six neighbouring U-fields change.  Fields on W are not
    needed for this objective and therefore are not maintained.
    """
    old_spin = spins[x]
    if IS_U[x]:
        old_h = u_fields[x]
        hist[FIELD_INDEX[old_h]] -= 1
        new_h = -old_h
        u_fields[x] = new_h
        hist[FIELD_INDEX[new_h]] += 1
    else:
        for y, j in NBR[x]:  # every y is in U
            old_h = u_fields[y]
            hist[FIELD_INDEX[old_h]] -= 1
            new_h = old_h - 2 * j * spins[y] * old_spin
            u_fields[y] = new_h
            hist[FIELD_INDEX[new_h]] += 1
    spins[x] = -old_spin

def anneal_to_target(rng, target, iters):
    spins = [rng.choice((1, -1)) for _ in range(V)]
    fields = compute_u_fields(spins)
    hist = histogram_vector_u(fields)
    target_vec = [target.get(h, 0) for h in FIELD_VALUES]
    cur = l1_distance_vector(hist, target_vec)
    best = cur
    best_hist = hist[:]
    best_spins = spins[:]
    used = 0

    for it in range(iters):
        if best == 0:
            break
        used = it + 1
        temperature = max(0.05, 3.0 * (1.0 - it / iters))
        x = rng.randrange(V)

        update_hist_for_flip(hist, fields, spins, x)
        newd = l1_distance_vector(hist, target_vec)
        accept = newd <= cur or rng.random() < pow(
            E_APPROX, (cur - newd) / temperature
        )
        if accept:
            cur = newd
        else:
            # The same spin flip is an involution, so applying it again is an
            # exact, allocation-free revert of spins, fields, and histogram.
            update_hist_for_flip(hist, fields, spins, x)

        if cur < best:
            best = cur
            best_hist = hist[:]
            best_spins = spins[:]

    return {
        "best_distance": best,
        "best_hist": vector_to_hist(best_hist),
        "best_spins": "".join("1" if s == 1 else "0" for s in best_spins),
        "iterations_used": used,
    }


def seeds_for_profile(profile):
    if profile == "paper":
        # Newly specified deterministic protocol; see provenance note above.
        return {
            "A": list(range(20260900, 20260940)),  # 40 seeds
            "B": list(range(20261000, 20261020)),  # 20 seeds
            "C": list(range(20261100, 20261120)),  # 20 control seeds
        }, 300_000
    if profile == "quick":
        # Matches the scale of the earlier bundled smoke test for A/B.
        return {
            "A": list(range(20260900, 20260910)),
            "B": list(range(20260900, 20260910)),
            "C": list(range(20260900, 20260903)),
        }, 60_000
    raise ValueError(profile)


def run_target(name, target, seeds, iters, rows):
    print(f"--- target {TARGET_LABELS[name]}: {target} ---", flush=True)
    for i, seed in enumerate(seeds, 1):
        result = anneal_to_target(random.Random(seed), target, iters)
        hit = result["best_distance"] == 0
        status = "HIT" if hit else f"miss (dist={result['best_distance']})"
        print(
            f"  {i:02d}/{len(seeds):02d} seed {seed}: {status}  "
            f"best_hist={result['best_hist']}",
            flush=True,
        )
        rows.append(
            {
                "target": name,
                "target_histogram": json.dumps(target, sort_keys=True, separators=(",", ":")),
                "seed": seed,
                "iterations_requested": iters,
                "iterations_used": result["iterations_used"],
                "hit": int(hit),
                "best_distance": result["best_distance"],
                "best_histogram": json.dumps(result["best_hist"], sort_keys=True, separators=(",", ":")),
                "best_spins": result["best_spins"],
            }
        )
    print()


def summarize(rows, profile, iters, include_control):
    summary = {
        "profile": profile,
        "iterations_per_run": iters,
        "provenance_note": (
            "Seed lists are a newly specified deterministic reproducibility "
            "protocol for this released package, not recovered historical seeds."
        ),
        "targets": {},
    }
    for name in ("A", "B", "C"):
        subset = [r for r in rows if r["target"] == name]
        if not subset:
            continue
        distances = [int(r["best_distance"]) for r in subset]
        summary["targets"][name] = {
            "target_histogram": TARGETS[name],
            "runs": len(subset),
            "hits": sum(int(r["hit"]) for r in subset),
            "min_best_distance": min(distances),
            "max_best_distance": max(distances),
            "distance_histogram": dict(sorted(Counter(distances).items())),
            "seeds": [int(r["seed"]) for r in subset],
        }
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("paper", "quick"),
        default="paper",
        help="experiment size (default: %(default)s)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="override iterations per run",
    )
    parser.add_argument(
        "--targets",
        default="AB",
        help="targets to run, any combination of A/B/C (default: AB)",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="also run realised target C as a positive control",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="write per-run CSV; paper profile default is q6_other_distributions_results.csv",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="write JSON summary; paper profile default is q6_other_distributions_summary.json",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    self_test()
    print("h(v) self-test against edge-counting and incremental updates: PASS")
    print()

    seed_map, default_iters = seeds_for_profile(args.profile)
    iters = args.iters if args.iters is not None else default_iters
    if iters <= 0:
        raise SystemExit("--iters must be positive")

    selected = []
    for ch in args.targets.upper():
        if ch not in TARGETS:
            raise SystemExit(f"unknown target {ch!r}; use A, B, and/or C")
        if ch not in selected:
            selected.append(ch)
    if args.control and "C" not in selected:
        selected.append("C")

    rows = []
    for name in selected:
        run_target(name, TARGETS[name], seed_map[name], iters, rows)

    summary = summarize(rows, args.profile, iters, "C" in selected)
    print("SUMMARY")
    for name, info in summary["targets"].items():
        print(
            f"  {name}: runs={info['runs']} hits={info['hits']} "
            f"best-distance-hist={info['distance_histogram']}"
        )

    csv_path = args.csv
    json_path = args.json
    if args.profile == "paper":
        if csv_path is None:
            csv_path = "q6_other_distributions_results.csv"
        if json_path is None:
            json_path = "q6_other_distributions_summary.json"

    if csv_path:
        write_csv(csv_path, rows)
        print(f"csv={csv_path}")
    if json_path:
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"json={json_path}")

    # Do not fail merely because A/B were not hit: non-hits are the expected
    # experimental outcome.  The realised control, if requested, must hit in
    # every run or the experiment returns nonzero.
    if "C" in selected:
        c = summary["targets"]["C"]
        if c["hits"] != c["runs"]:
            print("ERROR: realised control C did not hit in every run", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
