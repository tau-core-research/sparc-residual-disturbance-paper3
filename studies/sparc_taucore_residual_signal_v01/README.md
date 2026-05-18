# Paper 3 regeneration study

This directory contains the single public regeneration script for Paper 3:

```bash
python studies/sparc_taucore_residual_signal_v01/make_paper3_submission_source_v01.py
```

The script rebuilds:

- `studies/sparc_taucore_residual_signal_v01/packet_v01_seed/`
- root-level SVG figures in `figures/`
- LaTeX/PDF figure copies in `paper3_submission_source/figures/`
- `paper3_submission_source/main.tex`
- `paper3_submission_source/references.bib`
- `paper3_submission_source/main.pdf` when `tectonic` is available
- `arxiv_submission_source.zip`

Only files required for public reproduction are kept in this repository. Exploratory scripts and intermediate research artifacts from development are intentionally excluded from the public package.

The manuscript treats `TPG/projection` as a frozen operational residual baseline. The outputs are candidate-support and validation-gate artifacts, not physical validation claims.
