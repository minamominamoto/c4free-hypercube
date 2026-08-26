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

open(DST, 'w', encoding='utf-8').write(src)
print(f'wrote {DST} ({len(src)} bytes)')
