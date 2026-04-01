# C4-Free Subgraphs of Hypercubes

Supplementary data, paper, and code for:

**Minamo Minamoto** (2026).
*New Lower Bounds for C4-Free Subgraphs of the Hypercubes Q6, Q7, and Q8:
Constructions, Structure, and Computational Method.*
arXiv:[2603.29127](https://arxiv.org/abs/2603.29127) [math.CO]

## Main Results

| n | \|E(Qn)\| | ex(Qn, C4) | Source |
|---|--------|------------|--------|
| 6 | 192 | = 132 | Harborth–Nienborg 1994 (reproduced) |
| 7 | 448 | ≥ **304** | This work (new) |
| 8 | 1024 | ≥ **680** | This work (new) |

## Paper

| File | Description |
|------|-------------|
| `c4free_hypercube_v2.pdf` | Full paper (7 pages): constructions, 20-type classification, SA method, ILP proof |

## Code

### `c4free_sa.py` — Two-phase Simulated Annealing

Python implementation of the algorithm. Requires only the standard library.

```bash
# Reproduce the Q7 result (304-edge C4-free subgraph)
python c4free_sa.py --n 7 --target 304 --trials 10 --seed 42

# Reproduce the Q8 result (680-edge C4-free subgraph)
python c4free_sa.py --n 8 --target 680 --trials 5 --seed 0

# Q6 (known exact value: 132)
python c4free_sa.py --n 6 --target 132 --trials 5 --seed 1
```

**Algorithm:**
- **Phase 1 (Penalty SA):** Minimise `-|E| + λ·V` where `V` = C4 violations.
- **Phase 2 (Swap SA):** Fix `|E|`, drive violations to zero via swap moves.
- **Diversification:** Apply random elements of Aut(Qn) between trials.

## Data Files

| File | Description |
|------|-------------|
| `q6_edges_132.jsonl` | 132-edge C4-free subgraph of Q6 |
| `q6_ilp.mps` | ILP in MPS format (192 variables, 240 constraints) for ex(Q6,C4)≤132 |
| `q7_edges_304.jsonl.part1` | 304-edge C4-free subgraph of Q7 — part 1 of 3 |
| `q7_edges_304.jsonl.part2` | 19,866 distinct 304-edge solutions for Q7 — part 2 of 3 |
| `q7_edges_304.jsonl.part3` | 19,866 distinct 304-edge solutions for Q7 — part 3 of 3 |
| `q8_edges_680.jsonl` | Two distinct 680-edge C4-free subgraphs of Q8 |

Reconstruct the full Q7 solution set:
```bash
cat q7_edges_304.jsonl.part1 q7_edges_304.jsonl.part2 q7_edges_304.jsonl.part3 > q7_solutions_all.jsonl
```

## Verification

All C4-free claims certified by exhaustive enumeration:

- Q6: C(6,2)×2^4 = 240 four-cycles checked
- Q7: C(7,2)×2^5 = 672 four-cycles checked (all 19,866 solutions)
- Q8: C(8,2)×2^6 = 1,792 four-cycles checked

```python
import json

def verify_c4free(edges, n):
    es = set(tuple(sorted(e)) for e in edges)
    for base in range(2**n):
        for d1 in range(n):
            for d2 in range(d1+1, n):
                if (base>>d1)&1 or (base>>d2)&1:
                    continue
                a, b, c, d = base, base|(1<<d1), base|(1<<d2), base|(1<<d1)|(1<<d2)
                if all(tuple(sorted(e)) in es for e in [(a,b),(a,c),(b,d),(c,d)]):
                    return False
    return True

with open("q6_edges_132.jsonl") as f:
    sol = json.loads(f.read())
print(verify_c4free(sol["edges"], 6))  # True
```

## Upper Bound (Q6)

```bash
# Using SCIP
scip -f q6_ilp.mps

# Using Python (pyscipopt)
from pyscipopt import Model
m = Model()
m.readProblem("q6_ilp.mps")
m.optimize()
print(int(-m.getObjVal()))  # 132
```

## Contact

Minamo Minamoto — minamominamoto4f5683f6@gmail.com

## Citation

```
Minamo Minamoto (2026). New Lower Bounds for C4-Free Subgraphs of the
Hypercubes Q6, Q7, and Q8. arXiv:2603.29127 [math.CO].
```
