# SPARC projection-sensitive residual candidate paper 3

This repository is the public reproducibility package for:

**A reproducible candidate framework for projection-sensitive residual structure in SPARC rotation curves**

The manuscript is an exploratory candidate-methodology preprint. It defines a reproducible framework for comparing fixed SPARC residual diagnostics, selecting candidate and control objects, and specifying the predictive validation gate needed before any stronger physical claim can be made.

The package intentionally does **not** claim a new-gravity detection, model selection, or physical validation of the TPG/projection baseline. In this repository, `TPG/projection` is a frozen operational residual baseline retained as a historical label, not a validated gravity model.

## Author And Research Workflow

I am an independent researcher using an AI-assisted workflow to develop reproducible diagnostic tests around projection-sensitive residual hypotheses. I am not claiming expert-level validation. I would value criticism on whether the proposed gate/falsification structure is scientifically meaningful.

AI systems are used for drafting, mathematical organization, code generation, literature triage, and internal consistency checks. Numerical and symbolic audits can support reproducibility and error-finding, but they do not replace independent expert review or physical validation.

## Theory Context

The broader Tau Core / projection-theory background is maintained separately at:

```text
https://github.com/tau-core-research/tau-core-theory
```

This Paper 3 repository is a standalone reproducibility package. It does not require accepting the Tau Core theory hub; the manuscript should be read as an exploratory candidate/control framework with a predictive validation gate.

## Repository contents

```text
paper3_submission_source/                         LaTeX source, bibliography, figures, and compiled PDF
figures/                                          Regenerated SVG figures used by the manuscript
studies/sparc_taucore_residual_signal_v01/        Paper 3 regeneration script and seed packet
studies/sparc_residual_coherence_test_v01/        Minimal derived Paper 1 input tables
studies/sparc_residual_disturbance_inference_v01/ Minimal derived Paper 2 input tables
studies/sparc_radial_s_tau_pilot_v01/             Minimal derived radial-pilot input tables
tests/                                            Public reproducibility checks
arxiv_submission_source.zip                       arXiv-ready source package
```

Raw SPARC, LITTLE THINGS, or other survey data are not redistributed here. The repository contains only derived tables and paper artifacts needed to regenerate the manuscript package.

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the paper source, derived tables, figures, arXiv source ZIP, and PDF:

```bash
python studies/sparc_taucore_residual_signal_v01/make_paper3_submission_source_v01.py
```

Run the public reproducibility checks:

```bash
python -m pytest -q
```

The generator uses `tectonic` for PDF compilation when available. If `tectonic` is missing, the TeX source, figures, derived tables, and arXiv ZIP are still regenerated, but the PDF readiness row records the compiler blocker.

## Main outputs

- `paper3_submission_source/main.tex`
- `paper3_submission_source/main.pdf`
- `paper3_submission_source/references.bib`
- `paper3_submission_source/figures/*.pdf`
- `figures/*.svg`
- `arxiv_submission_source.zip`
- `studies/sparc_taucore_residual_signal_v01/packet_v01_seed/*.csv`

## Citation

Use `CITATION.cff` for repository citation metadata. The data-use scope and redistribution boundary are documented in `DATA_NOTICE.md`.
