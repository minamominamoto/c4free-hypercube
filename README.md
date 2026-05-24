# C4-Free Subgraphs of Hypercubes

Supplementary data, code, and paper accompanying work by **Minamo Minamoto** (2026)
on quadrilateral-free (C4-free) subgraphs of the hypercubes Q6, Q7, and Q8.

Preprint: **arXiv:2603.29127** — *New Lower Bounds for C4-Free Subgraphs of the
Hypercubes Q6, Q7, and Q8: Constructions, Structure, and Computational Method.*

## Main results

| n | \|E(Qn)\| | ex(Qn, C4) | Source |
|---|-----------|------------|--------|
| 6 | 192 | **= 132** | Harborth–Nienborg 1994 (reproduced here by independent ILP) |
| 7 | 448 | **≥ 304** | This work (new lower bound) |
| 8 | 1024 | **≥ 680** | This work (new lower bound) |

The lower bounds for Q7 and Q8 are theorems, witnessed by the explicit edge lists
in this repository. The corresponding **equalities** ex(Q7,C4)=304 and ex(Q8,C4)=680
are stated as **conjectures**, supported by simulated-annealing and integer-programming
search; they are experimental evidence, not upper-bound proofs.

## Reproducibility: one command

Every C4-free claim is independently re-checkable with a dependency-free script
(standard library only; no third-party packages, no network):

```bash
python3 verify.py
```

`verify.py` reads each solution, checks that every edge is a valid Q_n edge with no
loops or duplicates and exactly the claimed edge count, certifies C4-freeness by
**exhaustively enumerating all four-cycles** of Q_n, and prints the SHA-256 of each
data file. It exits 0 iff every check passes. Four-cycles enumerated per solution
( C(n,2) · 2^(n-2) ): Q6 = 240, Q7 = 672 (for all 19,866 solutions), Q8 = 1,792.

The data-file hashes are also recorded in `SHA256SUMS` as a fixed-version
certificate, checkable with standard tools:

```bash
shasum -a 256 -c SHA256SUMS     # macOS
sha256sum -c SHA256SUMS         # Linux
```

## Paper and code

| File | Description |
| --- | --- |
| `c4free_hypercube.pdf` | The paper (PDF): new lower bounds, structural classification (Q7: 20 dimension-profile types, 19,866 solutions; Q8: the 680-edge construction and 681-edge barrier), and the computational method |
| `c4free_hypercube.tex` | LaTeX source of the paper |
| `c4free_sa.py` | Two-phase simulated-annealing search used to obtain the lower bounds |
| `verify.py` | Dependency-free verifier (re-checks every certificate from scratch) |
| `SHA256SUMS` | SHA-256 certificate for the data files |

## Data files

| File | Description |
| --- | --- |
| `q6_edges_132.jsonl` | 132-edge C4-free subgraph of Q6 (lower-bound witness) |
| `q6_ilp.mps` | ILP in MPS format (192 variables, 240 constraints) for the Q6 upper bound |
| `q7_edges_304.jsonl.part{1,2,3}` | The 19,866 distinct 304-edge C4-free subgraphs of Q7 (split into 3 parts) |
| `q8_edges_680.jsonl` | Two distinct 680-edge C4-free subgraphs of Q8 |

Each line is a JSON object `{"edges": [[u, v], ...]}` with vertices encoded as
integers `0 … 2^n − 1`. To reconstruct the full Q7 set:

```bash
cat q7_edges_304.jsonl.part1 q7_edges_304.jsonl.part2 q7_edges_304.jsonl.part3 > q7_solutions_all.jsonl
```

## Upper bound (Q6)

The ILP in `q6_ilp.mps` certifies ex(Q6,C4) ≤ 132 with any MIP solver:

```bash
scip -f q6_ilp.mps          # SCIP
# or in Python (pyscipopt):
#   from pyscipopt import Model
#   m = Model(); m.readProblem("q6_ilp.mps"); m.optimize()
#   print(int(-m.getObjVal()))   # 132
```

## Citation

```
Minamo Minamoto (2026). New Lower Bounds for C4-Free Subgraphs of the
Hypercubes Q6, Q7, and Q8: Constructions, Structure, and Computational
Method. arXiv:2603.29127.
```

## License

Released under the MIT License (see `LICENSE`).

## Contact

Minamo Minamoto — ORCID [0009-0002-1201-5704](https://orcid.org/0009-0002-1201-5704)
