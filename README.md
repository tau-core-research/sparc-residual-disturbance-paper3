# SPARC Tau Core Residual Signal Paper 3

This is the slim working reproducibility package for:

**Searching for Tau Core signal candidates in SPARC residual structure: a cautious continuation of the residual-disturbance audit**

Paper 3 extends the Paper 1 and Paper 2 audit line. It asks whether the residual differences between TPG/projection, MOND-simple, empirical RAR-like baselines, and observed SPARC rotation curves contain a candidate signature compatible with the Tau Core observer/environment weighting idea.

The package is intentionally conservative. It does not claim Tau Core validation, gravity-model selection, or a completed independent external replication.

The current RMOND status is intentionally explicit: the local TPG-RMOND bridge supports theory motivation, but a unique frozen pointwise RMOND velocity law is not yet available in this public seed packet. RMOND is therefore audited as a blocker, not reported as a completed numeric comparator.

## Main Files

```text
LICENSE
CITATION.cff
DATA_NOTICE.md
requirements.txt
paper3_submission_source/main.tex
paper3_submission_source/references.bib
paper3_submission_source/main.pdf
paper3_submission_source/figures/
arxiv_submission_source.zip
figures/
tests/test_public_reproducibility_package.py
studies/sparc_taucore_residual_signal_v01/make_paper3_submission_source_v01.py
studies/sparc_taucore_residual_signal_v01/packet_v01_seed/
```

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the Paper 3 source, figures, derived candidate tables, arXiv source ZIP, and PDF:

```bash
python studies/sparc_taucore_residual_signal_v01/make_paper3_submission_source_v01.py
python -m pytest -q
```

The script uses `tectonic` to build `paper3_submission_source/main.pdf`. If `tectonic` is not installed, the TeX source and arXiv ZIP still regenerate, but the PDF readiness gate will report a compile blocker.

## Included Derived Inputs

The retained inherited inputs are derived tables, not raw survey products:

```text
studies/sparc_residual_coherence_test_v01/paper_packet_v06_distance_balanced/
studies/sparc_residual_disturbance_inference_v01/packet_v01_seed/
studies/sparc_radial_s_tau_pilot_v01/packet_v01_seed/
```

These inputs provide residual-family scores, pointwise residual maps, distance/observability/environment summaries, and small radial pilot controls.

## Data Boundary

The repository excludes:

- raw SPARC rotmod files,
- raw FITS cubes or moment maps,
- raw THINGS/HALOGAS/LITTLE THINGS products,
- private notes,
- local caches,
- API keys or tokens.

## arXiv Source Package

The repository includes:

```text
arxiv_submission_source.zip
```

The ZIP contains only:

```text
main.tex
references.bib
figures/*.pdf
```
