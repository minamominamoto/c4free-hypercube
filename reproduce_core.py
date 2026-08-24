#!/usr/bin/env python3
"""One-command reproducibility driver for the paper's core computational claims.

Default mode uses only the Python standard library and checks:
  1. all released C4-free / odd-square certificates (verify.py);
  2. the exact field integer programme for n=6,7,8 (solve_field_ip.py);
  3. the exhaustive Q6 realizability decision: distributions A and B are not
     realisable, C is (q6_decide_realizability.py);
  4. the two Q8 witnesses for the second optimal field distribution
     {2:87,4:40,6:1} (verify_q8_B_witnesses.py);
  5. byte-for-byte regeneration of the Q6 odd-square witness;
  6. byte-for-byte regeneration of the Q8 odd-square witness;
  7. the exhaustive Q7 odd-square audit and its released CSV;
  8. the exact Type-18 automorphism claims, including 46/101 order-3 cases.

Optional modes:
  --structural   also run analyze_q7_structure.py, q7_hamming_tally.py,
                 q7_order3_automorphisms.py and q7_oddsquare_orbits.py (all require numpy;
                 analyze_q7_structure.py is the expensive one)
  --experiments  rerun the deterministic Q6 A/B targeted SA protocol and compare
                 its logs with the bundled canonical logs
  --all          run both optional modes

Usage:
    python3 reproduce_core.py
    python3 reproduce_core.py --experiments
    python3 reproduce_core.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
Q7_PARTS = [
    ROOT / "q7_edges_304.jsonl.part1",
    ROOT / "q7_edges_304.jsonl.part2",
    ROOT / "q7_edges_304.jsonl.part3",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_step(number, title, cmd, cwd=None):
    print("\n" + "=" * 72, flush=True)
    print(f"[{number}] {title}", flush=True)
    print("$ " + " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run([str(x) for x in cmd], cwd=cwd or ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"FAILED: {title} (exit {proc.returncode})")
    print(f"[PASS] {title}", flush=True)


def require_same(generated, bundled, label):
    generated = Path(generated)
    bundled = Path(bundled)
    if not generated.exists():
        raise SystemExit(f"FAILED: {label}: generated file missing: {generated}")
    if not bundled.exists():
        raise SystemExit(f"FAILED: {label}: bundled file missing: {bundled}")
    g = sha256(generated)
    b = sha256(bundled)
    if g != b:
        raise SystemExit(
            f"FAILED: {label}: SHA-256 mismatch\n"
            f"  generated {g}  {generated}\n"
            f"  bundled   {b}  {bundled}"
        )
    print(f"[MATCH] {label}: {g}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural",
        action="store_true",
        help="run expensive numpy-based full Q7 structural analysis",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="rerun deterministic Q6 targeted SA experiment and compare logs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="equivalent to --structural --experiments",
    )
    args = parser.parse_args(argv)
    if args.all:
        args.structural = True
        args.experiments = True

    step = 1
    run_step(step, "certificate verification", [PYTHON, ROOT / "verify.py"])
    step += 1
    run_step(step, "exact field integer programme", [PYTHON, ROOT / "solve_field_ip.py"])
    step += 1
    run_step(
        step,
        "exhaustive Q6 realizability decision (A/B not realisable, C realisable)",
        [PYTHON, ROOT / "q6_decide_realizability.py"],
    )
    step += 1
    run_step(
        step,
        "Q8 second-distribution witnesses ({2:87,4:40,6:1})",
        [PYTHON, ROOT / "verify_q8_B_witnesses.py"],
    )
    step += 1

    with tempfile.TemporaryDirectory(prefix="c4free-repro-") as td:
        tmp = Path(td)

        run_step(step, "regenerate Q6 odd-square witness", [PYTHON, ROOT / "generate_q6_132.py"], cwd=tmp)
        require_same(tmp / "q6_odd_square_132.json", ROOT / "q6_odd_square_132.json", "Q6 odd-square witness")
        step += 1

        run_step(step, "regenerate Q8 odd-square witness", [PYTHON, ROOT / "generate_q8_682.py"], cwd=tmp)
        require_same(tmp / "q8_odd_square_682.json", ROOT / "q8_odd_square_682.json", "Q8 odd-square witness")
        step += 1

        run_step(
            step,
            "exhaustive Q7 odd-square audit",
            [PYTHON, ROOT / "audit_q7_odd_square.py", *Q7_PARTS],
            cwd=tmp,
        )
        require_same(tmp / "q7_odd_square_389.csv", ROOT / "q7_odd_square_389.csv", "Q7 odd-square audit CSV")
        step += 1

        type18_csv = tmp / "q7_type18_automorphisms_101.csv"
        run_step(
            step,
            "exact Type-18 automorphism audit (101/101 and 46/101)",
            [PYTHON, ROOT / "type18_automorphisms.py", "--csv", type18_csv, *Q7_PARTS],
            cwd=tmp,
        )
        require_same(type18_csv, ROOT / "q7_type18_automorphisms_101.csv", "Type-18 automorphism CSV")
        step += 1

        if args.experiments:
            out_csv = tmp / "q6_other_distributions_results.csv"
            out_json = tmp / "q6_other_distributions_summary.json"
            run_step(
                step,
                "Q6 targeted A/B simulated-annealing protocol",
                [
                    PYTHON,
                    ROOT / "q6_other_distributions.py",
                    "--profile",
                    "paper",
                    "--csv",
                    out_csv,
                    "--json",
                    out_json,
                ],
                cwd=tmp,
            )
            require_same(out_csv, ROOT / "q6_other_distributions_results.csv", "Q6 targeted-search CSV")
            require_same(out_json, ROOT / "q6_other_distributions_summary.json", "Q6 targeted-search JSON")
            step += 1

            control_csv = tmp / "q6_realized_control_results.csv"
            control_json = tmp / "q6_realized_control_summary.json"
            run_step(
                step,
                "Q6 realised-distribution positive control (20/20)",
                [
                    PYTHON,
                    ROOT / "q6_other_distributions.py",
                    "--profile",
                    "paper",
                    "--targets",
                    "C",
                    "--iters",
                    "60000",
                    "--csv",
                    control_csv,
                    "--json",
                    control_json,
                ],
                cwd=tmp,
            )
            require_same(control_csv, ROOT / "q6_realized_control_results.csv", "Q6 realised-control CSV")
            require_same(control_json, ROOT / "q6_realized_control_summary.json", "Q6 realised-control JSON")
            step += 1

    if args.structural:
        run_step(
            step,
            "full Q7 structural/statistical analysis (numpy)",
            [PYTHON, ROOT / "analyze_q7_structure.py", *Q7_PARTS],
        )
        step += 1
        run_step(
            step,
            "Q7 pairwise Hamming tally (636 minimum pairs, 28 type combinations)",
            [PYTHON, ROOT / "q7_hamming_tally.py"],
        )
        step += 1
        run_step(
            step,
            "order-3 automorphism check on all 19,866 solutions",
            [PYTHON, ROOT / "q7_order3_automorphisms.py"],
        )
        step += 1
        run_step(
            step,
            "Aut(Q7)-orbit decomposition of the 389 odd-square solutions",
            [PYTHON, ROOT / "q7_oddsquare_orbits.py"],
        )
        step += 1
        run_step(
            step,
            "full Aut(Q7)-orbit census of the 19,866 solutions (180 orbits)",
            [PYTHON, ROOT / "q7_orbit_census.py", "--budget", "100000"],
        )
        step += 1
        run_step(
            step,
            "stabiliser orders per orbit and the 34,227,200 total",
            [PYTHON, ROOT / "q7_orbit_census.py", "--stabilisers"],
        )
        step += 1

    print("\n" + "=" * 72)
    print("RESULT: ALL REQUESTED REPRODUCIBILITY CHECKS PASSED")
    print(f"root={ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
