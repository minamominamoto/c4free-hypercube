# Reproducibility update — 2026-08-23 (updated 2026-08-26)

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


## Verification-layer inventory (cumulative)

Beyond the items listed under "Added" above, the reproducibility layer has
grown to include the following (all wired into `reproduce_core.py` as
noted; wall times are from the measured environment in `README.md`):

- Verifier hardening: `verify.py` recomputes the non-edge C4-violation
  distributions (local-maximality margins) for both 680-edge Q8 solutions,
  the 682-edge Q8 odd-square witness, and the Q6 odd-square witness, and
  asserts catalogue-level pairwise distinctness of the 19,866 Q7 edge sets
  (SHA-256 over sorted edge lists). `verify_q8_B_witnesses.py` also checks
  the opposite bipartition's field histogram and both witnesses' non-edge
  violation distributions. `q7_orbit_census.py --stabilisers` rejects an
  incomplete or absent census checkpoint instead of treating the sentinel
  orbit id `-1` as a real orbit.
- Completed orbit-census artefact: `q7_orbit_census.json` ships with the
  full assignment (19,866 solutions, 180 orbits), per-orbit catalogue
  counts, stabiliser orders/orbit lengths, and the 34,227,200 total; the
  finished checkpoint `q7_orbit_census_ckpt.npz` is bundled, and the
  stabiliser pass persists its results into the JSON.
  `q7_orbit_census_check.py` (<1 s, standard library) checks that artefact
  against every census-derived number in the paper; in the default path.
- Byte-compare regression: `reproduce_core.py` regenerates
  `q8_A_witness_check.json` (seed 90008) and asserts SHA-256 identity with
  the bundled `q8_A_witness.json`. See `README.md` for the platform scope
  of this byte-identity claim.
- `sa_q8.py` (historical, recovered): two runtime print labels that called
  the 674.9 figure a lower bound now label it an out-of-domain
  extrapolation; the change is disclosed in the file header and the search
  logic is untouched.
- `cycle_space_rank.py`: dependency-free GF(2) elimination showing the
  squares of Q_n span its cycle space for n = 3..8 (rank = E − V + 1 =
  5, 17, 49, 129, 321, 769). Default path.
- `field_identity_defect.py`: verifies the exact defect formula
  `sum_{v in U} h(v)^2 = n^2·2^(n−1) − 4·sum_s spl_U(s)` on every released
  certificate (reproducing each quoted split histogram and total) and on
  203 deterministic edge sets each for n = 4 and n = 5; both bipartition
  sides checked. Default path.
- `q7_orbit_census.py --fresh`: from-scratch census in separate files
  (`q7_orbit_census_fresh_ckpt.npz` / `q7_orbit_census_fresh.json`); it
  never reads the bundled checkpoint and never writes the released
  artefacts, and a completed fresh run must reproduce the released JSON
  byte for byte (canonical key order; checkpoint rewritten only by an
  invocation that computed something).
  `reproduce_core.py --structural` performs the strong check: delete stale
  `_fresh` files, run `--fresh` to completion plus `--fresh
  --stabilisers`, and require SHA-256 identity with the released census.
- `q7_lambda1_by_type.py` (numpy): recomputes the 20-row dimension-profile
  table from the released catalogue and asserts every printed figure —
  per-type counts, profile entropies, per-type mean spectral radius
  (rounded independently to four decimals), and the exhaustive individual
  range [4.78543, 4.79129]; added so the printed type means are produced
  by a bundled script. In `--structural`.
- `reproduce_core.py --experiments` is labelled historical/provenance-only
  (superseded as evidence by the exhaustive realizability decision; slow).
- Orbit-witness certificate: `q7_orbit_witnesses.json` (generated by
  `q7_orbit_witnesses_gen.py`, which asserts its assignment equals the
  released census) + `q7_orbit_witness_check.py`. Default mode (standard
  library, ~16 s) verifies every one of the 19,866 witness group elements,
  the identity of the certified assignment with `q7_orbit_census.json`,
  the attainment of each orbit's stored canonical form, and the pairwise
  distinctness of the 180 canonical forms — certifying the assignment
  ("at most 180 orbits") with no group sweep. `--canonical [--orbits A:B]`
  (numpy, census-scale) re-verifies each canonical form is the true orbit
  minimum; minimality plus distinctness certifies "exactly 180"
  independently of the census scan. Default mode in the default path,
  `--canonical` in `--structural`.
- `verify.py` additionally checks the regenerable second Q8 certificate
  `q8_A_witness.json` (odd-square, 682 edges, margin distribution
  {3:42, 4:151, 5:133, 6:16}), so no released 682-edge claim rests solely
  on the non-regenerable historical artefact. `q7_hamming_tally.py` now
  prints and asserts the count of pairs attaining the maximum distance
  (274: exactly 85 pairs), closing a regression gap a pre-release review
  identified.
- `verify_q6_ilp_map.py`: machine-checks the released Q6 ILP artefacts'
  stated bijections (192 variables <-> 192 edges; 240 constraints <-> 240
  squares, handling MPS integrality MARKER lines). Standard library,
  <1 s; wired into the default path.
- `cross_verify.py`: bundled independent re-implementation of the
  certificate checks (bitmask common-neighbour counting over
  Hamming-distance-2 pairs; sorted-tuple distinctness, no hashing; no code
  shared with `verify.py`). Standard library, ~30 s; default path. Makes
  the manuscript's single-implementation-risk mitigation auditable.

## Commands

Core checks, standard library only:

```bash
python3 reproduce_core.py
```

Also rerun and byte-compare the superseded Q6 targeted-search logs (provenance/reproducibility only, not evidence for realisability):

```bash
python3 reproduce_core.py --experiments
```

Certify "exactly 180 orbits" (the quick default path certifies the
assignment, i.e. "at most 180"; exactness needs one of the two heavy
routes — see README for details):

```bash
python3 q7_orbit_witness_check.py --canonical
```

Install optional dependencies and run the expensive Q7 structural analysis too:

```bash
python3 -m pip install -r requirements.txt
python3 reproduce_core.py --all
```
