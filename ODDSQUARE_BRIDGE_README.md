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
local-field histogram is `{0: 2, 2: 168, 4: 86}`.

`generate_q8_682.py` opens its output with `newline="\n"`, so the advertised
byte-level SHA-256 of `q8_odd_square_682.json` is the same on Unix, macOS, and
Windows. This applies to that JSON file only. `audit_q7_odd_square.py` writes
its CSV through the `csv` module with its default CRLF line terminator, so
`q7_odd_square_389.csv` has CRLF line endings on every platform; its advertised
SHA-256 is the hash of the CRLF form.

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
simulated annealing over the 64 spins that reached 132 edges on its first
trial:

```bash
python3 search_q6.py
```

Because the seed is fixed, this run reproduces exactly. It now writes
`q6_spins.txt` to the current working directory (an earlier version wrote to
a hardcoded absolute path and failed outside the original environment).

Note: `q6_odd_square_132.json` is NOT the same edge set as the separately
released `q6_edges_132.jsonl`. The two differ in 62 edges; both are 132-edge
C4-free subgraphs of Q6, but only this one is odd-square.

## Q6: are the other two optimal distributions realisable?

The field-bound optimisation for n=6 (Section "Exact values" of the paper)
has exactly three optimal local-field distributions on U:
`{2:30,6:2}`, `{0:1,2:27,4:3,6:1}`, and `{0:2,2:24,4:6}` -- the last being
the one realised above. Derrida et al. (1979) list multiple solutions in
their Table I for d=6 and state they do not know if the others besides the
first are realisable; Marinari-Parisi-Ritort (1995) report that "the other
solution of the Diophantine equation *seems not to* correspond to any spin configuration".

Run:

```bash
python3 q6_other_distributions.py
```

This targets each of the other two distributions directly (simulated
annealing that minimises L1 distance from the target histogram, not "reach
132 edges by any route") across 20-40 independent seeds and up to 300,000
iterations each. As a control, the same method reaches the realised
distribution `{0:2,2:24,4:6}` immediately from any seed. Neither of the
other two is ever reached; both plateau at an identical nonzero distance
(16 and 8 respectively) regardless of seed. This is computational evidence,
not a proof, that only distribution C is reachable at the budgets tested.
We do not map MPR95's formulation onto the A/B/C labels used here, so we
make no claim about whether this corroborates their specific remark.

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
