# Q6/Q7/Q8 odd-square certificates communicated by Minamo Minamoto

This package accompanies the 18 August 2026 coordination with Shaoheng Lai.
The Q6 material was added later (23 August 2026) and was not part of that
coordination.

## Q8 reconstruction

Run from this directory:

```bash
python3 generate_q8_682.py
```

The script embeds the reconstructed 256-spin vector, generates the positive edges
under the canonical fully frustrated coupling, verifies every one of the 1,792
squares, and writes `q8_odd_square_682.json`.

Conventions: vertices are integers 0 through 255; dimensions are bits 0 through 7;
for the edge from `x` to `x xor (1<<dim)`, with bit `dim` initially zero,

```text
J(x,dim) = (-1)^popcount(x & ((1<<dim)-1)).
```

The output contains 682 positive edges. Its square intersection histogram is
`{1: 301, 3: 1491}`, its degree histogram is `{4: 2, 5: 168, 6: 86}`, and its
local-field histogram is `{0: 2, 2: 168, 4: 86}` over all 256 vertices; on the even-parity side U it is `{0: 1, 2: 84, 4: 43}`.

`generate_q8_682.py` opens its output with `newline="\n"`, so the advertised
byte-level SHA-256 of `q8_odd_square_682.json` is the same on Unix, macOS, and
Windows. This applies to that JSON file only. `audit_q7_odd_square.py` writes
its CSV through the `csv` module with its default CRLF line terminator, so
`q7_odd_square_389.csv` has CRLF line endings on every platform; its advertised
SHA-256 is the hash of the CRLF form.

## Q8 second optimal field distribution

`q8_B_witnesses.json` contains two 682-edge certificates whose even-side field
histogram is `{2:87,4:40,6:1}`. In both certificates the opposite bipartition
side has histogram `{0:1,2:84,4:43}`, so one edge set realises the two optimal
n=8 distributions on its two sides simultaneously. Run:

```bash
python3 verify_q8_B_witnesses.py
```

The verifier checks both field histograms, all 1,792 squares, the stored spin
and edge hashes, and local maximality margins. The non-edge C4-violation
distributions are `{3:34,4:169,5:121,6:18}` for seed `20261207` and
`{3:41,4:154,5:130,6:17}` for seed `20261218`; hence every missing edge creates
at least three new C4s. The `202612xx` integers are RNG seeds, not execution
dates.

## Q6 reconstruction

Run from this directory:

```bash
python3 generate_q6_132.py
```

Same construction as the Q8 case: the canonical fully frustrated coupling on
Q6 (verified fully frustrated -- all 240 squares have coupling product -1),
plus a 64-spin configuration. The script regenerates the positive edges,
verifies all 240 squares, and writes `q6_odd_square_132.json`.

The output contains 132 positive edges. Its square intersection histogram is
`{1: 30, 3: 210}` (so it is odd-square, hence C4-free), its degree histogram is
`{3: 4, 4: 48, 5: 12}`, and its local-field histogram on U is
`{0: 2, 2: 24, 4: 6}` -- giving sum h = 72 and sum h^2 = 192, attaining the
field bound for n=6. Like the Q8 script, it writes with `newline="\n"`.

The spin configuration was found by `search_q6.py`, a fixed-seed (20260823)
simulated annealing (here the seed was chosen as a date-like mnemonic; by
contrast the Q8 second-distribution seeds such as 20261207 are arbitrary RNG
integers and are not execution dates) over the 64 spins that reached 132 edges on its first
trial:

```bash
python3 search_q6.py
```

Because the seed is fixed, this run reproduces exactly. It now writes
`q6_spins.txt` to the current working directory (an earlier version wrote to
a hardcoded absolute path and failed outside the original environment).

Note: `q6_odd_square_132.json` is NOT the same edge set as the separately
released `q6_edges_132.jsonl`. Their symmetric difference has size
`|E △ E'| = 62`; both are 132-edge C4-free subgraphs of Q6, but only this one
is odd-square. Adding any missing edge to the odd-square witness creates at
least two new C4s, with non-edge violation distribution
`{2:2, 3:29, 4:26, 5:3}`.

## Q6: are the other two optimal distributions realisable?

The field-bound optimisation for n=6 has exactly three optimal local-field
distributions on U: A = `{2:30,6:2}`, B = `{0:1,2:27,4:3,6:1}`, and
C = `{0:2,2:24,4:6}` (the one realised by `q6_odd_square_132.json`).
Derrida et al. (1979) listed multiple solutions in their Table I for d=6 and
stated they did not know whether any but the first is realisable;
Marinari-Parisi-Ritort (1995) reported by simulated annealing that "the other
solution of the Diophantine equation *seems not to* correspond to any spin
configuration".

**This is now decided, not estimated.** Run:

```bash
python3 q6_decide_realizability.py
```

Because Q6 is bipartite, every neighbour of a vertex in U lies in U^c, so the
quantity k(v) depends only on the 32 spins on U^c while the spin at v itself is
free. The achievable non-negative field at v is therefore determined by
m(v) = min(k(v), 6-k(v)), and a target histogram is realisable exactly when the
multiset {m(v) : v in U} matches. That is a finite (2^32) decision, settled by
depth-first search with counter pruning:

| target | nodes visited | verdict |
| --- | ---: | --- |
| A `{2:30,6:2}` | 655,359 | NOT REALISABLE |
| B `{0:1,2:27,4:3,6:1}` | 22,129,151 | NOT REALISABLE |
| C `{0:2,2:24,4:6}` | 501 | REALISABLE (132 edges reconstructed) |

Wall time is about 19 seconds total on one modern core; the C branch also
rebuilds a full spin vector and re-derives the histogram and edge count as an
end-to-end check. `--full` additionally repeats each search without the
global-spin-flip symmetry reduction, exactly doubling every node count.

By the switching lemma the odd-square edge sets of Q6 are precisely the
positive-edge sets of spin configurations under any fixed fully frustrated
coupling, so this verdict does not depend on the particular coupling used.

### Superseded heuristic experiment

`q6_other_distributions.py` and its CSV/JSON logs record an earlier simulated
annealing study of the same question (40 deterministically specified RNG seeds
for A, 20 for B, 300,000 iterations each, all plateauing at L1 distance 16 and
8; these consecutive integer seed lists are reproducibility labels, not a claim
of statistically independent random draws; a positive control
reached C in 20/20 runs at 60,000 iterations). These files remain in the release
as a record of how the question was first approached, but they are **superseded
by `q6_decide_realizability.py` and are no longer offered as evidence** — a
plateau is not a proof, and the iteration budgets for the targets and the
control were not equal.

## Q7 audit

Place the three published Q7 JSONL parts in this directory or pass their paths:

```bash
python3 audit_q7_odd_square.py \
  q7_edges_304.jsonl.part1 \
  q7_edges_304.jsonl.part2 \
  q7_edges_304.jsonl.part3
```

The script audits all 19,866 solutions and writes `q7_odd_square_389.csv`, which
identifies all 389 odd-square solutions by source file, line number, zero-based
global index, content hash, raw direction counts, sorted dimension profile, and
square histogram.

Both scripts use only the Python standard library.
