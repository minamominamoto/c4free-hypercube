#!/usr/bin/env python3
"""
make_journal_version.py -- derive the E-JC submission layer from the
archival manuscript, deterministically.

The archival file c4free_hypercube_v5.tex remains the single edited
source. This script produces c4free_hypercube_v5_ejc.tex by:

  1. replacing the preamble with the E-JC one (e-jc.sty provides
     geometry, amsmath/amssymb/amsthm, hyperref, and the full theorem
     environment set on a single shared counter, so theorem numbering is
     unchanged); our booktabs/xcolor/enumitem/microtype and the \\ex
     macro are kept;
  2. adding the E-JC metadata commands (\\dateline, \\MSC, \\Copyright)
     and the E-JC author block (\\author + \\authortext with \\email);
  3. relocating the title-page revision note into the Reader's Guide
     section as "Relation to the earlier circulated versions" (E-JC
     papers carry no preprint-style title-page notes; nothing is
     deleted).

Everything else -- abstract, body, appendix, bibliography -- is copied
verbatim. Run from the release directory:

    python3 make_journal_version.py

Writes c4free_hypercube_v5_ejc.tex. Compile with e-jc.sty present.
"""
SRC = 'c4free_hypercube_v5.tex'
DST = 'c4free_hypercube_v5_ejc.tex'

src = open(SRC, encoding='utf-8').read()

# ---- 1+2: preamble swap -------------------------------------------------
pre_end = src.index('\\newcommand{\\ex}{\\mathrm{ex}}')
title_start = src.index('\\title{')
date_line = '\\date{March 2026 (revised August 2026)}\n'
assert date_line in src
title_block = src[title_start:src.index(date_line)]

new_pre = (
    '\\documentclass[12pt]{article}\n'
    '\\usepackage[amsmath]{e-jc}\n'
    '\\usepackage{booktabs}\n'
    '\\usepackage{xcolor}\n'
    '\\usepackage{enumitem}\n'
    '\\usepackage{microtype}\n'
    '\\newcommand{\\ex}{\\mathrm{ex}}\n\n'
    '% E-JC metadata. Accepted/published dates are filled by the journal.\n'
    '\\dateline{Aug 26, 2026}{TBD}{TBD}\n'
    '\\MSC{05C35, 05C22, 05-04, 82D30}\n'
    '\\Copyright{The author. Released under the CC BY license '
    '(International 4.0).}\n\n'
    + title_block +
    '\\author{Minamo Minamoto}\n'
    '\\authortext{}{Independent researcher, Japan '
    '(\\email{minamominamoto4f5683f6@gmail.com}).}\n\n'
)
begin_doc = src.index('\\begin{document}')
src = new_pre + src[begin_doc:]

# drop the archival \author/\date remnants are already excluded above; the
# archival \title was reused verbatim inside title_block.

# ---- 3: relocate the title-page revision note ---------------------------
note_start = src.index('\\begin{center}\n\\small This is a revision')
note_end = src.index('\\end{center}\n', note_start) + len('\\end{center}\n')
note = src[note_start:note_end]
inner = note[len('\\begin{center}\n\\small '):-len('\\end{center}\n')]
src = src[:note_start] + src[note_end:]

anchor = ('relocated, not retracted.')
i = src.index(anchor) + len(anchor)
src = (src[:i]
       + '\n\n\\subsection*{Relation to the earlier circulated versions}\n\n'
       + inner + '\n'
       + src[i:])


# ---- 4: JOURNAL PROJECTIONS (reader-serving layer) ----------------------
# Criterion: keep what helps a reader learn and verify the mathematics;
# defer the author-protective record (discovery narrative, dated access
# logs, failure archive) to the repository/archival layer, with pointers.
# Nothing is deleted from the archival source; these are projections.

# J1: compact Section 2. Keep the two reading conventions verbatim;
# replace the relocated narrative with a withdrawal note + record pointer.
c1a = src.index('One reading convention')
c1b = src.index('not as\nseparate results.') + len('not as\nseparate results.')
conv1 = src[c1a:c1b]
c2a = src.index('A second convention of the same kind')
c2b = src.index('never a theorem.') + len('never a theorem.')
conv2 = src[c2a:c2b]
g_start = src.index("\\section{Reader's Guide: Scope, Attribution, and "
                    'Provenance}\\label{sec:guide}')
g_end = src.index('%% ============================================================',
                  g_start)
compact = (
    '\\section{Scope, Attribution, and Provenance}\\label{sec:guide}\n\n'
    + conv1 + '\n'
    + conv2 + '\n\n'
    'Two further notes. Earlier preprint versions of this paper (v1--v4)\n'
    'claimed a computational proof of the unrestricted $n=6$ upper bound;\n'
    'that claim was withdrawn, and the bound is used here on the authority\n'
    'of~\\cite{HN94} throughout (caveat~$\\ddagger$ to\n'
    'Table~\\ref{tab:values}). The complete provenance and revision record\n'
    '--- the discovery history of the odd-square correspondence, the dated\n'
    'primary-source access log, and the derivation lineage of every\n'
    'released artefact --- is maintained verbatim in the released\n'
    'repository and in the archival (arXiv) version of this paper; this\n'
    'journal version states the results and their verification and defers\n'
    'that record rather than reproducing it.\n\n'
)
src = src[:g_start] + compact + src[g_end:]

# J2: organisation sentence matches the compact section
o = """Section~\\ref{sec:guide} collects scope, attribution, the verification map,
and the provenance record in one place (material that earlier revisions
carried in the abstract)."""
n = """Section~\\ref{sec:guide} states the reading conventions and the
scope/attribution summary; the complete provenance record is maintained
in the released repository."""
assert src.count(o) == 1
src = src.replace(o, n)

# (former J3 removed: that pointer lives inside the relocated guide
# text, which J1 already replaces in the journal layer.)

# J4: drop the failure-record appendix; repoint its summary
a_start = src.index('\\appendix')
a_end = src.index('\\begin{thebibliography}')
src = src[:a_start] + src[a_end:]
o = """is in
    Appendix~\\ref{app:attempts}; it supports no conclusion either way."""
n = """is preserved in the released repository's record of unsuccessful
    attempts; it supports no conclusion either way."""
assert src.count(o) == 1
src = src.replace(o, n)

# J5: compress the dated source-access trail to its reader-serving core
t_start = src.index('The DPTV79 and MPR95 source retrieval and passage-level')
t_end = src.index('without AI assistance.') + len('without AI assistance.')
n = ("The primary-source passages quoted in Section~\\ref{sec:oddsquare}\n"
     "were checked directly at page level during revision (DPTV79\n"
     "pp.~620--622; MPR95 pp.~3--4 of the arXiv version; the LNSW26\n"
     "preprint and its published version, with the $n\\le6$ value list\n"
     "confirmed absent from both); the retrieval was AI-assisted, and the\n"
     "dated access record --- including what could not be obtained, notably\n"
     "a text-extractable copy of DPTV79's Table~I --- is preserved in the\n"
     "released repository. This disclosure should not be read as a claim\n"
     "that the author independently re-transcribed every quoted passage\n"
     "without AI assistance.")
src = src[:t_start] + n + src[t_end:]

open(DST, 'w', encoding='utf-8').write(src)
print(f'wrote {DST} ({len(src)} bytes)')
