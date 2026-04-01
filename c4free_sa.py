"""
c4free_sa.py  --  Two-phase Simulated Annealing for C4-free subgraphs of Q_n
Author : Minamo Minamoto (2026)
GitHub : https://github.com/minamominamoto/c4free-hypercube
License: CC-BY 4.0

Algorithm
---------
Phase 1 (Penalty SA):
    Minimise  f(E) = -|E| + lambda * V(E)
    where V(E) = number of C4 violations.
    Allows temporary violations to escape local optima.

Phase 2 (Swap SA):
    Fix |E| and drive V(E) -> 0 via swap moves (remove one edge, add one).

Diversification:
    Before each trial, apply a random element of Aut(Q_n)
    (random bit permutation + random bit flip) to the current best.

Usage
-----
    python c4free_sa.py --n 7 --target 304 --trials 10 --seed 42
    python c4free_sa.py --n 8 --target 680 --trials 5  --seed 0
"""

import argparse
import json
import math
import random
import time
from itertools import combinations


# ---------------------------------------------------------------------------
# Hypercube construction
# ---------------------------------------------------------------------------

def build_hypercube(n):
    """Return list of edges of Q_n as (u, v) with u < v."""
    N = 1 << n
    edges = []
    for u in range(N):
        for d in range(n):
            v = u ^ (1 << d)
            if u < v:
                edges.append((u, v))
    return edges


def build_c4_list(n):
    """
    Return list of all C4s in Q_n.
    Each C4 is represented as a tuple of 4 edge-indices
    (indices into the edge list returned by build_hypercube).
    """
    edges = build_hypercube(n)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    N = 1 << n
    c4s = []
    seen = set()
    for base in range(N):
        for d1 in range(n):
            for d2 in range(d1 + 1, n):
                if (base >> d1) & 1 or (base >> d2) & 1:
                    continue
                a = base
                b = a | (1 << d1)
                c = a | (1 << d2)
                d = b | (1 << d2)
                key = tuple(sorted([a, b, c, d]))
                if key in seen:
                    continue
                seen.add(key)
                e1 = (min(a, b), max(a, b))
                e2 = (min(a, c), max(a, c))
                e3 = (min(b, d), max(b, d))
                e4 = (min(c, d), max(c, d))
                c4s.append((
                    edge_to_idx[e1],
                    edge_to_idx[e2],
                    edge_to_idx[e3],
                    edge_to_idx[e4],
                ))
    return c4s


# ---------------------------------------------------------------------------
# Violation counting (incremental)
# ---------------------------------------------------------------------------

def build_edge_to_c4s(num_edges, c4s):
    """For each edge index, list of C4 indices that contain it."""
    e2c = [[] for _ in range(num_edges)]
    for ci, (e1, e2, e3, e4) in enumerate(c4s):
        for e in (e1, e2, e3, e4):
            e2c[e].append(ci)
    return e2c


def count_violations(selected, c4s):
    """Count number of C4s fully contained in selected (a set of edge indices)."""
    return sum(1 for (e1, e2, e3, e4) in c4s
               if e1 in selected and e2 in selected
               and e3 in selected and e4 in selected)


def delta_v_toggle(edge_idx, selected, e2c, c4s):
    """
    Change in violation count if we toggle edge_idx.
    Returns dV (negative means fewer violations after toggle).
    """
    dv = 0
    in_sel = edge_idx in selected
    for ci in e2c[edge_idx]:
        e1, e2, e3, e4 = c4s[ci]
        others = [e for e in (e1, e2, e3, e4) if e != edge_idx]
        all_others_in = all(o in selected for o in others)
        if in_sel:
            # removing: if this C4 was fully in, violation decreases
            if all_others_in:
                dv -= 1
        else:
            # adding: if others are in, violation increases
            if all_others_in:
                dv += 1
    return dv


# ---------------------------------------------------------------------------
# Automorphism group of Q_n (bit permutations + bit flips)
# ---------------------------------------------------------------------------

def random_automorphism(n, rng):
    """Return a function that applies a random element of Aut(Q_n)."""
    perm = list(range(n))
    rng.shuffle(perm)
    flip = rng.randint(0, (1 << n) - 1)

    def apply(v):
        w = 0
        for i in range(n):
            if (v >> i) & 1:
                w |= (1 << perm[i])
        return w ^ flip

    return apply


def apply_automorphism_to_edges(edge_indices, edges, apply_fn, edge_to_idx):
    """Transform a set of edge indices under an automorphism."""
    result = set()
    for ei in edge_indices:
        u, v = edges[ei]
        nu, nv = apply_fn(u), apply_fn(v)
        ne = (min(nu, nv), max(nu, nv))
        result.add(edge_to_idx[ne])
    return result


# ---------------------------------------------------------------------------
# Phase 1: Penalty SA
# ---------------------------------------------------------------------------

def phase1_sa(edges, c4s, e2c, target_size, params, rng):
    """
    Penalty SA: minimise -|E| + lambda * V.
    Returns (current_set, violations).
    """
    lam = params.get('lambda', params.get('lambda_', 0.5))
    T0 = params['T0']
    T1 = params['T1']
    steps = params['steps']
    viol_cap = params['viol_cap']

    ne = len(edges)
    # Start from a random set of size target_size
    current = set(rng.sample(range(ne), min(target_size, ne)))
    v_cur = count_violations(current, c4s)
    size_cur = len(current)

    best = set(current)
    best_v = v_cur
    best_size = size_cur

    alpha = (T1 / T0) ** (1.0 / max(steps, 1))
    T = T0

    for _ in range(steps):
        ei = rng.randint(0, ne - 1)
        dv = delta_v_toggle(ei, current, e2c, c4s)

        if ei in current:
            new_size = size_cur - 1
            new_v = v_cur + dv
        else:
            new_size = size_cur + 1
            new_v = v_cur + dv

        if new_v > viol_cap:
            T *= alpha
            continue

        df = (-new_size + lam * new_v) - (-size_cur + lam * v_cur)
        if df < 0 or rng.random() < math.exp(max(-30.0, -df / T)):
            if ei in current:
                current.discard(ei)
            else:
                current.add(ei)
            size_cur, v_cur = new_size, new_v

            if v_cur < best_v or (v_cur == best_v and size_cur > best_size):
                best = set(current)
                best_v = v_cur
                best_size = size_cur

        T *= alpha

    return best, best_v


# ---------------------------------------------------------------------------
# Phase 2: Swap SA
# ---------------------------------------------------------------------------

def phase2_sa(edges, c4s, e2c, current, params, rng):
    """
    Swap SA: fix |E|, minimise V.
    Returns (edge_set, violations).
    """
    T2 = params.get('T2', 0.5)
    T3 = params.get('T3', 0.001)
    steps = params['steps2']

    ne = len(edges)
    current = set(current)
    not_selected = list(set(range(ne)) - current)
    selected_list = list(current)

    v_cur = count_violations(current, c4s)
    best = set(current)
    best_v = v_cur

    if best_v == 0:
        return best, 0

    alpha = (T3 / T2) ** (1.0 / max(steps, 1))
    T = T2

    for _ in range(steps):
        # pick one to remove, one to add
        e_out = rng.choice(selected_list)
        e_in = rng.choice(not_selected)

        # compute delta V
        # remove e_out first
        dv_out = delta_v_toggle(e_out, current, e2c, c4s)
        current.discard(e_out)
        # then add e_in
        dv_in = delta_v_toggle(e_in, current, e2c, c4s)
        current.add(e_out)  # restore

        dv = dv_out + dv_in
        new_v = v_cur + dv

        if dv < 0 or rng.random() < math.exp(max(-30.0, -dv / T)):
            current.discard(e_out)
            current.add(e_in)
            selected_list.remove(e_out)
            selected_list.append(e_in)
            not_selected.remove(e_in)
            not_selected.append(e_out)
            v_cur = new_v

            if v_cur < best_v:
                best = set(current)
                best_v = v_cur
                if best_v == 0:
                    return best, 0

        T *= alpha

    return best, best_v


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_c4free(edge_indices, c4s):
    """Return True iff no C4 is fully contained in edge_indices."""
    s = set(edge_indices)
    for (e1, e2, e3, e4) in c4s:
        if e1 in s and e2 in s and e3 in s and e4 in s:
            return False
    return True


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def search(n, target, trials, seed, verbose=True):
    rng = random.Random(seed)

    if verbose:
        print(f"Building Q_{n}  (N={1<<n} vertices, {n*(1<<(n-1))} edges)")
    edges = build_hypercube(n)
    ne = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    if verbose:
        print(f"Building C4 list...", end=" ", flush=True)
    c4s = build_c4_list(n)
    if verbose:
        print(f"{len(c4s)} four-cycles")

    e2c = build_edge_to_c4s(ne, c4s)

    best_solution = None
    best_size = 0

    for trial in range(trials):
        t0 = time.time()

        # Random parameters
        lam   = rng.uniform(0.30, 0.90)
        T0    = rng.uniform(0.20, 4.00)
        T1    = rng.uniform(0.001, 0.030)
        steps = rng.randint(3_000_000, 15_000_000)
        vcap  = rng.randint(6, 40)

        p1 = dict(T0=T0, T1=T1,
                  steps=steps, viol_cap=vcap)
        p1['lambda'] = lam

        # Diversification: apply random automorphism to best solution
        if best_solution is not None:
            fn = random_automorphism(n, rng)
            start = apply_automorphism_to_edges(
                best_solution, edges, fn, edge_to_idx)
        else:
            start = set(rng.sample(range(ne), target))

        # Phase 1
        sol, v = phase1_sa(edges, c4s, e2c, target, p1, rng)

        # Phase 2 if close
        if v <= 4:
            steps2 = 50_000_000 if v > 1 else 200_000_000
            p2 = dict(T2=0.5, T3=0.001, steps2=steps2)
            sol, v = phase2_sa(edges, c4s, e2c, sol, p2, rng)

        elapsed = time.time() - t0
        status = "✓ C4-FREE" if v == 0 else f"violations={v}"
        if verbose:
            print(f"  trial {trial+1:3d}/{trials}  "
                  f"|E|={len(sol)}  {status}  ({elapsed:.1f}s)")

        if v == 0 and len(sol) >= target:
            if best_solution is None or len(sol) > best_size:
                best_solution = set(sol)
                best_size = len(sol)
                if verbose:
                    print(f"    → New best: {best_size} edges")

    return best_solution, edges


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Two-phase SA for C4-free subgraphs of Q_n")
    ap.add_argument("--n",       type=int, default=7,
                    help="Hypercube dimension (default: 7)")
    ap.add_argument("--target",  type=int, default=304,
                    help="Target edge count (default: 304)")
    ap.add_argument("--trials",  type=int, default=10,
                    help="Number of SA trials (default: 10)")
    ap.add_argument("--seed",    type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--out",     type=str, default=None,
                    help="Output JSON file (optional)")
    args = ap.parse_args()

    print(f"=== C4-free SA: Q_{args.n}, target={args.target}, "
          f"trials={args.trials}, seed={args.seed} ===\n")

    sol, edges = search(args.n, args.target, args.trials, args.seed)

    if sol is not None:
        print(f"\nFound C4-free subgraph with {len(sol)} edges.")
        # degree sequence
        deg = [0] * (1 << args.n)
        for ei in sol:
            u, v = edges[ei]
            deg[u] += 1
            deg[v] += 1
        from collections import Counter
        dc = Counter(deg)
        print(f"Degree sequence: {dict(sorted(dc.items()))}")

        out = args.out or f"q{args.n}_edges_{len(sol)}.json"
        data = {
            "n": args.n,
            "num_edges": len(sol),
            "edges": [list(edges[ei]) for ei in sorted(sol)]
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {out}")
    else:
        print(f"\nNo solution found at target={args.target} "
              f"in {args.trials} trials.")


if __name__ == "__main__":
    main()
