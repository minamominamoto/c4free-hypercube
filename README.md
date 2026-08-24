# C4-Free Subgraphs of Hypercubes

Supplementary data, code, and paper accompanying work by **Minamo Minamoto** (2026)
on quadrilateral-free (C4-free) subgraphs of the hypercubes Q6, Q7, and Q8.

Preprint: **arXiv:2603.29127** (v1–v4: *New Lower Bounds for C4-Free Subgraphs of
the Hypercubes Q6, Q7, and Q8: Constructions, Structure, and Computational
Method*). This repository now corresponds to the **revised** manuscript,
*C4-Free Subgraphs of the Hypercubes Q6, Q7, and Q8: Odd Squares, Fully
Frustrated Models, and Computational Structure*, which supersedes v4's
Q8 result — see "What changed" below.

## Main results

| n | \|E(Qn)\| | ex(Qn, C4) | Source |
|---|-----------|------------|--------|
| 6 | 192 | **= 132** | Harborth–Nienborg 1994 (lower-bound side reproduced here independently; upper bound not independently re-verified, see paper) |
| 7 | 448 | **≥ 304** | Not new: follows from Derrida–Pomeau–Toulouse–Vannimenus (1979), reproduced/audited here |
| 8 | 1024 | **≥ 682** | Witness attaining the bound reported by Marinari–Parisi–Ritort (1995); explicit certificate obtained and verified here |

Within the **odd-square subclass** (edge sets meeting every 4-cycle an odd
number of times, equivalent to fully frustrated signed hypercubes), the
paper proves the subclass value g(n) exactly: **g(6)=132, g(7)=304,
g(8)=682**. All three are self-contained here: each upper bound follows
from the paper's field-bound lemma plus an elementary quantisation
argument, and each lower bound has an explicit odd-square edge list in
this repository, machine-checked by `verify.py`. This does not resolve ex(Q7,C4) or ex(Q8,C4) themselves, which
remain open — see the paper's Open Problems.

## What changed from the original (v1–v4) submission

After the original manuscript was circulated, a reader (S. Lai) identified
an overlooked connection to the statistical-physics literature on fully
frustrated hypercubes (Derrida–Pomeau–Toulouse–Vannimenus 1979;
Marinari–Parisi–Ritort 1995). This led to:

- An improved Q8 lower bound: **680 → 682** edges (the 680-edge conjecture
  is withdrawn).
- Withdrawal of the claim that the Q7 bound (304) is novel — it follows
  from the 1979 construction.
- A new theorem determining the odd-square subclass exactly for n=6,7,8.
- An audit of the original 19,866 Q7 solutions: only 389 of them are
  odd-square; the rest are not, so the odd-square construction does not
  explain the bulk of the structural classification.
- A Q6 odd-square witness (`q6_odd_square_132.json`) and an exact
  solution of the field-bound integer programme, which together remove
  the paper's earlier reliance on Harborth–Nienborg and on DPTV79's
  unreconstructed D=6 configuration for the value g(6)=132.

## Reproducibility: one command

**Resource note.** `verify.py` (and hence `reproduce_core.py`) loads all 19,866
Q7 solutions into memory at once; peak RSS is around 700 MB, comparable to
`analyze_q7_structure.py`. Budget roughly 1 GB of free memory and a few
minutes for the default run; `--structural` adds the expensive numpy passes.

Every C4-free claim is independently re-checkable with a dependency-free
script (standard library only; no third-party packages, no network):

```bash
python3 verify.py
```

`verify.py` reads each solution, checks that every edge is a valid Q_n edge
with no loops or duplicates and exactly the claimed edge count, certifies
C4-freeness by **exhaustively enumerating all four-cycles** of Q_n, and
(for the Q6 and Q8 odd-square witnesses) additionally checks the stronger
odd-square condition (every square has exactly 1 or 3 edges, not merely
not-4). It prints the SHA-256 of each data file and exits 0 iff every
check passes.
Four-cycles enumerated per solution ( C(n,2) · 2^(n-2) ): Q6 = 240,
Q7 = 672 (for all 19,866 solutions), Q8 = 1,792.

The data-file hashes are also recorded in two manifests, checkable with
standard tools:

```bash
shasum -a 256 -c ORIGINAL_DATA_SHA256SUMS.txt     # macOS
sha256sum -c ORIGINAL_DATA_SHA256SUMS.txt         # Linux
sha256sum -c ODDSQUARE_BRIDGE_SHA256SUMS.txt      # odd-square material
```

### Dependencies for the search/solver scripts

`verify.py`, `generate_q6_132.py`, `search_q6.py`, `generate_q8_682.py`,
`audit_q7_odd_square.py`, `solve_field_ip.py`, `q6_decide_realizability.py`,
and `reproduce_core.py` use only the Python standard library, as noted above.
`q7_hamming_tally.py`, `q7_order3_automorphisms.py`, `q7_oddsquare_orbits.py`
and `q7_orbit_census.py` additionally need `numpy`. `analyze_q7_structure.py`
and `audit_q7_odd_square.py` take the three `q7_edges_304.jsonl.part*` files as
required arguments. `q7_orbit_census.py` works to a time budget
(`--budget`, default 250 s) and saves a checkpoint: a full census needs two or
three invocations, each resuming automatically, and exits with code 2 while
incomplete. Its `--stabilisers` mode needs a completed census and then takes
seconds. The recovered production-history scripts need further third-party
packages. All of these are pinned in the provided `requirements.txt`
(install as needed):

- `source.py` (bundled): `highspy`
- `source_72h.py` (not bundled): `highspy` (tested with
  1.15.1)
- `source_cbc_origin.py`: `pulp`, `networkx`, `numpy`
- `analyze_q7_structure.py`: `numpy` only
- `sa_search.py`, `sa_collect304.py`, `sa_q8.py`, `c4free_sa.py`:
  standard library only (`random`, `json`, `time`, `math`, `os`,
  `hashlib`)

## Paper and code

| File | Description |
| --- | --- |
| `c4free_hypercube_v5.pdf` | The revised paper (PDF) |
| `c4free_hypercube_v5.tex` | LaTeX source of the revised paper |
| `c4free_sa.py` | Two-phase simulated-annealing search originally reported for the Q7/Q8 lower bounds. **Known bug, documented in the script and in the paper:** the Aut(Q_n)-based diversification restart is computed but never passed into `phase1_sa`, which always initialises from a fresh random sample and gets permanently stuck (see the paper's Computational Method section). This script's relationship to the released Q8 680-edge solutions remains an open provenance question. |
| `sa_search.py`, `sa_collect304.py`, `selected_edges_best.json` | **Recovered production code for the Q7 results** (recovered after `c4free_sa.py` was found not to explain the Q7 provenance). `sa_search.py` is a conventional penalty SA (add/remove/1-swap/kick moves, temperature-scaled Boltzmann acceptance); as released here it loads `selected_edges_best.json` unconditionally (no random/greedy fallback), and that file already contains the final 304-edge solution, so running it as packaged searches for improvements *beyond* 304 edges, not a from-scratch discovery of the first 304-edge solution. `sa_search.py` also has a known implementation defect: its `cur_list`/`mis_list` candidate lists are only rebuilt every 5,000 iterations while the underlying sets are updated every iteration, so a stale list entry can be sampled and the tracked violation count can in principle drift from the true value, which is never re-verified before saving. This does not affect the released certificates (independently re-verified `C_4`-free regardless of which script produced them), but affects trust in the script's own internal bookkeeping. `sa_collect304.py` mass-collects further solutions by removing a few edges from an existing solution (optionally automorphism-transformed) and greedily repairing; it does not share `sa_search.py`'s two defects above, but has its own: its `canonicalize()` function's docstring-level intent (coordinate relabelling, then the lexicographically smallest of all 2^n bit-flip images) is not what's implemented — the code only tries the n single-bit-flip masks (not all 2^n combined masks), and compares candidates with Python's `<` on `frozenset`, which tests proper-subset (not lexicographic order); since all candidates here have equal cardinality (304), this comparison is always False and the flip-selection loop never updates its choice, so no flip-based canonicalisation actually occurs. Verified: running `sa_collect304.py` from the released seed for 30 seconds finds ~2,000 new valid solutions, and applying its hash function *exactly as implemented* (i.e., with this defect) directly to all 19,866 released `q7_edges_304.jsonl.*` edge sets reproduces the `hash` field already stored in each of those same records, with zero mismatches (reproducible from the released files alone). This 19,866/19,866 match is evidence this is the actual code that produced the stored hashes; it is *not* evidence that the 19,866 solutions are de-duplicated across the full translation symmetry group, since the intended canonicalisation is not what runs. Recomputing hashes with a corrected, fully-canonicalising implementation gives a different result for 92 of the first 100 released solutions. Note also that `sa_collect304.py`'s `RUNTIME` constant (36 hours) has no command-line override: to run it briefly, either edit that constant or wrap the invocation in an external `timeout`, as documented for `sa_q8.py`. |
| `source.py` | **Recovered code for the origin of the Q7 304-edge solution chain**, tracing back further than the SA-based scripts above. A from-scratch HiGHS ILP solve of the square-≤3-edges formulation, with no warm start when `selected_edges_best.json` does not yet exist. Verified: run for 60 seconds with no pre-existing warm-start file (a genuine cold start), it produces a valid, growing 295-edge C4-free solution via HiGHS branch-and-bound alone. File timestamps and a recovered 72-hour solver log (not bundled, ~4.7MB) are consistent with a documented progression 289→301→303 edges (the first two steps via this same ILP lineage), with the final 303→304 step apparently made by the `sa_search.py` series above. |
| `source_cbc_origin.py` | **The earliest recovered attempt on this problem**, dated over a week before every other file in this archive: a CBC-solver (not HiGHS) ILP of the same formulation, explicitly commented "Erdős's $\\$100 problem, the n=7 case." This specific file has a confirmed bug: its C4-detection computes common neighbours of adjacent vertices, but Q7 is bipartite, so adjacent vertices never share a common neighbour; it finds 0 of 672 C4s and builds an ILP with no C4 constraints at all. The 289-edge result described in Section 7 was reached by a since-lost corrected version of this script; only a recovered solver log (not bundled) documents that result. This file is released for provenance completeness, not as a functional solver. |
| `sa_q8.py`, `q8_solution_a.txt`, `q8_solution_b.json` | **Verified, working production code for the Q8 680-edge results.** `sa_q8.py` is a penalty-SA hill-climber that, unlike `c4free_sa.py`, always restarts each trial from the current best *valid* (zero-violation) solution rather than from a fresh random sample, so it never has to escape a large initial violation count. `q8_solution_a.txt`/`q8_solution_b.json` are the recovered outputs, verified to match released Solution A and Solution B exactly by edge set. The separate `1,076`-trial statistic at 681 edges (attempts to exceed 680) remains unsubstantiated by any recovered script or log. |
| `q8_checkpoint_670.json` | A recovered 670-edge intermediate solution. To reproduce our finding that the code makes genuine progress (an external 90-second timeout reduces the violation count at the 675-edge target from 22 to 8 in one run — exact numbers won't repeat, since `sa_q8.py` sets no random seed): `cp q8_checkpoint_670.json q8_best.json`, then `timeout 90 python3 sa_q8.py` in the same directory. Use an external timeout, not the script's internal `RUNTIME` variable — `RUNTIME` is only checked once per outer trial, while each trial's inner Phase-1 loop runs 2–12 million steps with no time check inside it, so editing `RUNTIME` alone won't reliably stop execution near that value. Without `q8_best.json` present, `sa_q8.py` instead starts from a fresh greedy construction. |
| `verify.py` | Dependency-free verifier (re-checks every certificate from scratch, including the Q8 odd-square condition) |
| `generate_q6_132.py` | Regenerates and self-checks the 132-edge Q6 odd-square witness from its 64-bit spin configuration, using the same canonical fully frustrated coupling as the Q8 script. |
| `search_q6.py` | The fixed-seed (20260823) simulated-annealing search that produced that spin configuration; reached 132 edges on its first trial, and reproduces exactly because the seed is fixed. Also verifies that the coupling is fully frustrated on Q6 (all 240 squares). |
| `solve_field_ip.py` | Solves the paper's field-bound integer programme exhaustively by dynamic programming (no sign restriction), returning optima 72 / 160 / 340 for n = 6 / 7 / 8 and enumerating the 3 / 1 / 2 optimal local-field distributions. A cross-check on the paper's closed-form argument, not a substitute for it. |
| `generate_q8_682.py` | Regenerates and self-checks the 682-edge Q8 odd-square witness from its 256-bit spin configuration, derived by the author (17-18 Aug 2026) using MPR's canonical fully frustrated coupling and independently cross-checked by S. Lai; see the paper (Section 5.3) for the full account. |
| `audit_q7_odd_square.py` (run as `python3 audit_q7_odd_square.py q7_edges_304.jsonl.part1 q7_edges_304.jsonl.part2 q7_edges_304.jsonl.part3`; the three part files are required arguments) | Audits the 19,866 released Q7 solutions against the odd-square condition (389 satisfy it) |
| `ORIGINAL_DATA_SHA256SUMS.txt` | SHA-256 certificate for the original data files and `c4free_sa.py` |
| `ODDSQUARE_BRIDGE_SHA256SUMS.txt` | SHA-256 certificate for the odd-square reconstruction/audit material |

## Data files

| File | Description |
| --- | --- |
| `q6_edges_132.jsonl` | 132-edge C4-free subgraph of Q6 (lower-bound witness for ex(Q6,C4); *not* odd-square — see `q6_odd_square_132.json`) |
| `q6_ilp.mps` | ILP in MPS format (192 variables, 240 constraints) for the Q6 upper bound. Optimality was not independently closed within a practical runtime with a generic solver (see paper, Section 8.2); the upper bound ex(Q6,C4)≤132 rests on Harborth–Nienborg's combinatorial proof. |
| `q6_ilp_edge_map.csv` | Reconstructed x_i ↔ Q6-edge correspondence for `q6_ilp.mps` (not originally recorded; reconstructed by matching the MPS's variable-constraint incidence against Q6's known edge-square structure, verified to reproduce all 240 constraints exactly). |
| `q7_edges_304.jsonl.part{1,2,3}` | The 19,866 distinct 304-edge C4-free subgraphs of Q7 (split into 3 parts) |
| `analyze_q7_structure.py` | Recomputes the paper's Section 6 structural/statistical claims (degree sequence, dimension-profile classification, spectral-radius range, exhaustive pairwise Hamming-distance stats, Type-18 nontrivial-automorphism existence) directly from `q7_edges_304.jsonl.part1-3`. Standard library + numpy only. Verified: reproduces every reported number exactly. Peak memory is around 600–800 MB for the frozenset solution store alone (the Python `frozenset`-of-`frozenset` representation of 19,866 solutions has substantial per-object overhead); wall time is typically 3–10 minutes depending on hardware. The `CHUNK` constant controls the Hamming-distance batch size and does not materially affect peak memory, which is dominated by the solution store. Does not itself determine automorphism *order* (the 46/101 order-3 figure elsewhere in the paper used a separate, more detailed check). |
| `q8_edges_680.jsonl` | Two distinct 680-edge C4-free subgraphs of Q8 (Solution A and Solution B; see paper for how they differ) |
| `q6_odd_square_132.json` | The 132-edge odd-square witness for Q6. **Not the same edge set as `q6_edges_132.jsonl`** — the two differ in 62 edges; both are 132-edge C4-free subgraphs of Q6, but only this one is odd-square (square histogram `{1:30, 3:210}` vs `{1:8, 2:44, 3:188}`). |
| `q8_odd_square_682.json` | The 682-edge odd-square witness for Q8 (current headline lower bound) |
| `q7_odd_square_389.csv` | The 389 (of 19,866) Q7 solutions that satisfy the odd-square condition, with their dimension profiles |

Each `.jsonl` line is a JSON object `{"edges": [[u, v], ...]}` with vertices
encoded as integers `0 … 2^n − 1`. To reconstruct the full Q7 set:

```bash
cat q7_edges_304.jsonl.part1 q7_edges_304.jsonl.part2 q7_edges_304.jsonl.part3 > q7_solutions_all.jsonl
```

## Upper bound (Q6)

The ILP in `q6_ilp.mps` gives a feasible value matching 132 with any MIP
solver, but did not close to a proven optimum within a practical runtime
in our own tests (HiGHS 1.15.1, default settings: 6–7% gap remaining after
260s); see the paper for details. The upper bound ex(Q6,C4)≤132 itself
rests on Harborth and Nienborg's independent combinatorial proof (1994).

```bash
scip -f q6_ilp.mps          # SCIP
# or in Python (pyscipopt):
#   from pyscipopt import Model
#   m = Model(); m.readProblem("q6_ilp.mps"); m.optimize()
#   print(int(-m.getObjVal()))   # feasible value, not necessarily proven optimal quickly
```

## Citation

```
Minamo Minamoto (2026). C4-Free Subgraphs of the Hypercubes Q6, Q7, and Q8:
Odd Squares, Fully Frustrated Models, and Computational Structure.
(Revision of arXiv:2603.29127.)
```

## License

Released under the MIT License (see `LICENSE`), including `c4free_sa.py`.

## Contact

Minamo Minamoto — ORCID [0009-0002-1201-5704](https://orcid.org/0009-0002-1201-5704)
