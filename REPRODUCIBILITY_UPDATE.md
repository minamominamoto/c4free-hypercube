# Reproducibility update — 2026-08-23 (updated 2026-08-25)

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
  explicit parameters and CSV/JSON logging. The consecutive integer seed lists
  are reproducibility labels; no claim of statistically independent random
  sampling is made.
- `q6_other_distributions_results.csv` and
  `q6_other_distributions_summary.json`: byte-reproducible archival logs for the newly specified 40-seed/20-seed A/B protocol; retained for provenance, not as evidence after the exhaustive decision below.
- `q6_realized_control_results.csv` and `q6_realized_control_summary.json`:
  20-seed positive-control logs for the realised distribution C.

## Important Q6 provenance distinction

The 40/20 seed lists are a **newly specified deterministic reproducibility
protocol**. They are not claimed to be recovered historical seed lists.

**Update:** the realisability question these searches addressed is now settled
exactly by `q6_decide_realizability.py`, which reduces it to a finite (2^32)
decision and proves that distributions A and B are not realisable while C is.
The simulated-annealing logs below are retained as a record of the earlier
approach and are superseded as evidence; non-attainment in a heuristic search
was never a proof, and the exhaustive decision now supplies one.


## Round-69 verifier hardening

- `verify.py` now recomputes the non-edge C4-violation distributions used for
  local-maximality margins for both 680-edge Q8 solutions, the 682-edge Q8
  odd-square witness, and the Q6 odd-square witness.
- `verify_q8_B_witnesses.py` now also checks the opposite bipartition's field
  histogram and both witnesses' non-edge violation distributions.
- `q7_orbit_census.py --stabilisers` now rejects an incomplete or absent census
  checkpoint instead of silently treating sentinel orbit id `-1` as a real
  orbit.

## Round-76 additions

- Completed orbit-census **artefact**: `q7_orbit_census.json` now ships with
  the full assignment (19,866 solutions, 180 orbits), per-orbit catalogue
  counts, stabiliser orders/orbit lengths, and the 34,227,200 total; the
  finished checkpoint `q7_orbit_census_ckpt.npz` is bundled, and
  `q7_orbit_census.py --stabilisers` now persists its results into the JSON
  (a census re-run preserves consistent stabiliser data instead of stripping
  it).
- `q7_orbit_census_check.py`: lightweight (<1 s, standard library) checker of
  that artefact against every census-derived number in the paper; run by
  `reproduce_core.py` by default.
- `verify.py` now asserts **catalogue-level pairwise distinctness** of the
  19,866 Q7 edge sets (SHA-256 over sorted edge lists), distinct from its
  per-solution duplicate-edge check.
- `reproduce_core.py` now byte-compares (SHA-256) the regenerated
  `q8_A_witness_check.json` against the bundled `q8_A_witness.json`, turning
  the "exact reproduction" claim into an asserted regression test.
- `sa_q8.py` (historical, recovered): the two runtime print labels that
  called the 674.9 figure a lower bound now label it an out-of-domain
  extrapolation; the change is disclosed in the file header, and the search
  logic is untouched.

## Commands

Core checks, standard library only:

```bash
python3 reproduce_core.py
```

Also rerun and byte-compare the superseded Q6 targeted-search logs (provenance/reproducibility only, not evidence for realisability):

```bash
python3 reproduce_core.py --experiments
```

Install optional dependencies and run the expensive Q7 structural analysis too:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce_core.py --all
```
