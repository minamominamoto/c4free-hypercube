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

## Round-77 additions

- `cycle_space_rank.py`: dependency-free GF(2) elimination showing that the
  squares of Q_n span its cycle space for every n used in the paper
  (n = 3..8): rank = E − V + 1 = (n−2)·2^(n−1) + 1, i.e. 5, 17, 49, 129,
  321, 769. Run by `reproduce_core.py` by default; the manuscript's spanning
  claim now points at this bundled machine check instead of "routine to
  verify".
- `field_identity_defect.py`: verifies the paper's exact defect formula
  `sum_{v in U} h(v)^2 = n^2·2^(n−1) − 4·sum_s spl_U(s)` (new Remark after
  the field-identity lemma) on every released certificate, reproducing each
  quoted split histogram and total (Q7 catalogue record 0:
  {0:60, 1:552, 2:60} with total exactly 672 = C(7,2)·2^5; Q8 Solution B:
  1794 = 1792 + 2 splits on the failing side against exactly 1792 on the
  complement), and additionally on 203 deterministic edge sets each for
  n = 4 and n = 5 (empty, full, single edge, 200 seeded random). Both
  bipartition sides are checked throughout. Run by `reproduce_core.py` by
  default.
- `q7_orbit_census.py --fresh`: from-scratch census kept in separate files
  (`q7_orbit_census_fresh_ckpt.npz` / `q7_orbit_census_fresh.json`). It
  never reads the bundled checkpoint and never writes the released
  artefacts; the computation is deterministic, so a completed fresh run must
  reproduce the released JSON byte for byte. To support that byte identity
  the JSON writer now emits a fixed canonical key order, and the checkpoint
  is rewritten only by an invocation that actually computed something.
- `reproduce_core.py --structural` now performs the strong census check: it
  deletes stale `_fresh` files, runs `q7_orbit_census.py --fresh` to
  completion followed by `--fresh --stabilisers`, and requires SHA-256
  identity between `q7_orbit_census_fresh.json` and the released
  `q7_orbit_census.json`. (The previous driver re-ran the census in the
  release directory, where it resumed the bundled completed checkpoint,
  found nothing left to do, and rewrote the same content — an internal
  consistency check, not an independent recomputation. The paper's
  computational appendix discloses this.)

## Round-78 additions

- `q7_lambda1_by_type.py` (numpy): recomputes the paper's 20-row
  dimension-profile table directly from the released catalogue and asserts
  every printed figure — per-type solution counts, profile entropies, the
  per-type **mean** spectral radius lambda_1 (each rounded independently to
  four decimals), and the exhaustive individual range [4.78543, 4.79129].
  Added because a round-77 review noted the type means, while independently
  re-derivable, were not printed by any bundled script. Wired into
  `reproduce_core.py --structural` immediately after
  `analyze_q7_structure.py`.
- `reproduce_core.py --experiments` is now labelled
  historical/provenance-only (superseded as evidence by the exhaustive
  realizability decision; slow) in both the module docstring and the
  argparse help.
- README now records the measured environment and wall times (single Intel
  Xeon vCPU @ 2.10 GHz, ~4 GB RAM, Python 3.12.3, numpy 2.4.4; no GPU used
  anywhere): `reproduce_core.py` default 85 s; `q8_A_recover.py` 18 s here
  but over 600 s in one review environment (hardware-sensitive);
  `q7_orbit_census.py --fresh` 362 s + 9 s stabilisers, with the note that
  the census budget takes effect only between orbits.
- Manuscript-side round-78 changes (for completeness): the
  DPTV79-plural/MPR95-singular mismatch about "the other solution(s)" is
  now stated explicitly in one sentence; the MPR95 energy conversion shows
  the explicit intermediate step -340/(256*sqrt(8)) = -0.46956...; the
  exhaustive-decision proof now names the fixed assignment order and
  per-value counters of the DFS pruning; the census timing sentence points
  at the README's measured environment; the general-n cycle-space claim now
  carries a literature citation (Hellmuth-Leydold-Stadler 2014, convex
  cycle bases of partial cubes); and the LNSW26 disclosure records the
  preprint access date (2026-08-25), the DOI-record confirmation of the
  published version (2026-08-26), and that the published full text remained
  inaccessible.

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
