# Reproducibility update — 2026-08-23

This directory adds a reproducibility layer without changing the released edge
certificates.

## Added

- `reproduce_core.py`: one-command deterministic driver for certificate checks,
  the field DP, witness regeneration, the Q7 odd-square audit, and Type-18
  automorphisms.
- `type18_automorphisms.py`: exhaustive full-`Aut(Q7)` verification of the
  stronger Type-18 claims. It verifies that all 101 Type-18 solutions have an
  automorphism moving the tied 48-edge directions and exactly 46/101 have an
  order-3 automorphism cycling all three.
- `q7_type18_automorphisms_101.csv`: per-solution witnesses and stabilizer data.
- `requirements.txt`: pinned optional dependencies. Core verification remains
  standard-library-only.
- `q6_other_distributions.py`: revised deterministic experiment harness with
  explicit parameters and CSV/JSON logging.
- `q6_other_distributions_results.csv` and
  `q6_other_distributions_summary.json`: canonical logs for the newly specified
  40-seed/20-seed A/B protocol.
- `q6_realized_control_results.csv` and `q6_realized_control_summary.json`:
  20-seed positive-control logs for the realised distribution C.

## Important Q6 provenance distinction

The new 40/20 seed lists are a **newly specified deterministic reproducibility
protocol**. They are not claimed to be recovered historical seed lists. The
manuscript should describe results from these archived logs as a new reproducible
experiment, rather than implying that the exact historical seeds were recovered.
Non-attainment remains computational evidence, not a proof of non-realisability.

## Commands

Core checks, standard library only:

```bash
python3 reproduce_core.py
```

Also rerun and byte-compare the Q6 targeted-search logs:

```bash
python3 reproduce_core.py --experiments
```

Install optional dependencies and run the expensive Q7 structural analysis too:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce_core.py --all
```
