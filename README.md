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
g(8)=682**. This does not resolve ex(Q7,C4) or ex(Q8,C4) themselves, which
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

## Reproducibility: one command

Every C4-free claim is independently re-checkable with a dependency-free
script (standard library only; no third-party packages, no network):

```bash
python3 verify.py
```

`verify.py` reads each solution, checks that every edge is a valid Q_n edge
with no loops or duplicates and exactly the claimed edge count, certifies
C4-freeness by **exhaustively enumerating all four-cycles** of Q_n, and
(for the Q8 odd-square witness) additionally checks the stronger odd-square
condition (every square has exactly 1 or 3 edges, not merely not-4). It
prints the SHA-256 of each data file and exits 0 iff every check passes.
Four-cycles enumerated per solution ( C(n,2) · 2^(n-2) ): Q6 = 240,
Q7 = 672 (for all 19,866 solutions), Q8 = 1,792.

The data-file hashes are also recorded in two manifests, checkable with
standard tools:

```bash
shasum -a 256 -c ORIGINAL_DATA_SHA256SUMS.txt     # macOS
sha256sum -c ORIGINAL_DATA_SHA256SUMS.txt         # Linux
sha256sum -c ODDSQUARE_BRIDGE_SHA256SUMS.txt      # odd-square material
```

## Paper and code

| File | Description |
| --- | --- |
| `c4free_hypercube_v5.pdf` | The revised paper (PDF) |
| `c4free_hypercube_v5.tex` | LaTeX source of the revised paper |
| `c4free_sa.py` | Two-phase simulated-annealing search used to obtain the original Q7/Q8 lower bounds. **Known bug, documented in the script and in the paper (Section 7.1):** the Aut(Q_n)-based diversification restart is computed but never passed into `phase1_sa`, which always initialises from a fresh random sample; diversification is dead code as released. |
| `verify.py` | Dependency-free verifier (re-checks every certificate from scratch, including the Q8 odd-square condition) |
| `generate_q8_682.py` | Regenerates and self-checks the 682-edge Q8 odd-square witness from its 256-bit spin configuration, derived by the author (17-18 Aug 2026) using MPR's canonical fully frustrated coupling and independently cross-checked by S. Lai; see the paper (Section 5.3) for the full account. |
| `audit_q7_odd_square.py` | Audits the 19,866 released Q7 solutions against the odd-square condition (389 satisfy it) |
| `ORIGINAL_DATA_SHA256SUMS.txt` | SHA-256 certificate for the original data files and `c4free_sa.py` |
| `ODDSQUARE_BRIDGE_SHA256SUMS.txt` | SHA-256 certificate for the odd-square reconstruction/audit material |

## Data files

| File | Description |
| --- | --- |
| `q6_edges_132.jsonl` | 132-edge C4-free subgraph of Q6 (lower-bound witness) |
| `q6_ilp.mps` | ILP in MPS format (192 variables, 240 constraints) for the Q6 upper bound. Optimality was not independently closed within a practical runtime with a generic solver (see paper, Section 8.2); the upper bound ex(Q6,C4)≤132 rests on Harborth–Nienborg's combinatorial proof. |
| `q7_edges_304.jsonl.part{1,2,3}` | The 19,866 distinct 304-edge C4-free subgraphs of Q7 (split into 3 parts) |
| `q8_edges_680.jsonl` | Two distinct 680-edge C4-free subgraphs of Q8 (Solution A and Solution B; see paper for how they differ) |
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

Released under the MIT License (see `LICENSE`).

**Known inconsistency to resolve:** `c4free_sa.py`'s own header states
`License: CC-BY 4.0`, which conflicts with this repository-wide MIT
license. Pick one and make the two consistent (either update the
script's header to MIT, or mark it as an explicit per-file exception in
this README) before treating the reuse terms as settled.

## Contact

Minamo Minamoto — ORCID [0009-0002-1201-5704](https://orcid.org/0009-0002-1201-5704)
