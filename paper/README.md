# QIML 2026 submission package

This directory contains the manuscript source and compiled PDFs for:

**A Feasibility-Aware HUBO/QUBO Benchmark for UAV Obstacle-Avoidance with Visibility Cost**

The manuscript uses the official AAAI-27 LaTeX style and bibliography files (`aaai2027.sty` and `aaai2027.bst`). It presents a **single-instance feasibility-aware benchmark** and does not claim quantum speedup, quantum advantage, or performance across an instance suite.

## Build

From this directory, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For the stand-alone abstract:

```bash
pdflatex abstract.tex
```

A current TeX Live or MiKTeX installation with the packages required by the AAAI-27 style is needed.

## Files

- `main.tex`: full manuscript source
- `QIML_2026_UAV_HUBO_Submission.pdf`: compiled full manuscript
- `abstract.tex`: stand-alone single-column abstract source
- `QIML_2026_UAV_HUBO_Abstract.pdf`: compiled stand-alone abstract
- `references.bib`: bibliography database
- `figures/`: manuscript figures
- `aaai2027.sty` and `aaai2027.bst`: AAAI-27 style and bibliography files

## Consistency rules

- The registered/submitted title is exactly: **A Feasibility-Aware HUBO/QUBO Benchmark for UAV Obstacle-Avoidance with Visibility Cost**
- The paper-level framing is **benchmark**, not "evaluation protocol" as the title/name of the contribution.
- "Benchmark" refers to the reproducible single-instance study currently reported; it does not imply an instance suite.
- The reported exact reference is **CP-SAT**.
- The QUAV-style code is a candidate-path reference, not official QUAV code.
- `neal` is classical simulated annealing.
- `L=20` means 20 discrete planning layers, not 20 physical seconds.
- Submission status does not imply acceptance or publication.
