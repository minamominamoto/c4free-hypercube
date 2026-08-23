#!/usr/bin/env python3
"""Recover explicit Q8 spin witnesses for target field distribution B."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter

N = 8
V = 1 << N
U = [v for v in range(V) if bin(v).count("1") % 2 == 0]
TARGET = {2: 87, 4: 40, 6: 1}


def coupling(x, dim):
    return 1 if bin(x & ((1 << dim) - 1)).count("1") % 2 == 0 else -1


NBR = [[(x ^ (1 << d), coupling(min(x, x ^ (1 << d)), d))
        for d in range(N)] for x in range(V)]


def h_of(spins, v):
    return spins[v] * sum(j * spins[y] for y, j in NBR[v])


def histogram(spins):
    return Counter(h_of(spins, v) for v in U)


def distance(hist):
    return sum(abs(hist.get(k, 0) - TARGET.get(k, 0))
               for k in set(hist) | set(TARGET))


def anneal(seed, iterations, t_start=3.0, t_end=0.05):
    rng = random.Random(seed)
    spins = [rng.choice((1, -1)) for _ in range(V)]
    hist = histogram(spins)
    current = distance(hist)
    best = current
    best_hist = Counter(hist)
    best_spins = list(spins)
    best_iteration = 0
    for it in range(iterations):
        if best == 0:
            break
        temperature = max(t_end, t_start * (1.0 - it / iterations))
        x = rng.randrange(V)
        spins[x] = -spins[x]
        new_hist = histogram(spins)
        new_distance = distance(new_hist)
        if (new_distance <= current or
                rng.random() < math.exp((current - new_distance) / temperature)):
            current, hist = new_distance, new_hist
        else:
            spins[x] = -spins[x]
        if current < best:
            best = current
            best_hist = Counter(hist)
            best_spins = list(spins)
            best_iteration = it + 1
    return best, best_hist, best_spins, best_iteration


def positive_edges(spins):
    edges = []
    for x in range(V):
        for d in range(N):
            y = x ^ (1 << d)
            if x < y and coupling(x, d) * spins[x] * spins[y] == 1:
                edges.append([x, y])
    return edges


def make_record(seed, iterations):
    best, hist, spins, hit_iteration = anneal(seed, iterations)
    edges = positive_edges(spins)
    spin_bytes = bytes(1 if s == 1 else 0 for s in spins)
    edge_text = "\n".join(f"{x},{y}" for x, y in edges).encode("ascii")
    return {
        "schema": "q8-second-field-distribution-witness-v1",
        "n": N,
        "coupling": "J(x,d)=(-1)^popcount(x & ((1<<d)-1)); x is lower endpoint",
        "bipartition": "U={x: popcount(x) even}",
        "target_histogram": {str(k): v for k, v in sorted(TARGET.items())},
        "seed": seed,
        "iterations_budget": iterations,
        "hit_iteration": hit_iteration if best == 0 else None,
        "best_distance": best,
        "field_histogram_U": {str(k): v for k, v in sorted(hist.items())},
        "spins_vertex_order_0_to_255": spins,
        "positive_edges": edges,
        "positive_edge_count": len(edges),
        "spin_bits_sha256": hashlib.sha256(spin_bytes).hexdigest(),
        "positive_edges_csv_sha256": hashlib.sha256(edge_text).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20261207, 20261218])
    parser.add_argument("--iters", type=int, default=300000)
    parser.add_argument("--output", default="q8_B_witnesses.json")
    args = parser.parse_args()
    records = []
    for seed in args.seeds:
        record = make_record(seed, args.iters)
        records.append(record)
        print(seed, record["best_distance"], record["hit_iteration"],
              record["field_histogram_U"], record["positive_edge_count"])
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"witnesses": records}, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
