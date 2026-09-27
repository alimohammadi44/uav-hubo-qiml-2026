# QIML 2026 camera-ready package

This directory contains the author-visible camera-ready manuscript for the Second AAAI Symposium on Quantum Information & Machine Learning (QIML 2026). It uses the official AAAI-27 LaTeX style and bibliography files (`aaai2027.sty` and `aaai2027.bst`).

The paper presents an evaluation protocol on one primary and two fixed obstacle-layout sensitivity instances. It does not claim a general instance-suite benchmark or quantum speedup.

## Build

Run:

```bash
python3 ../src/generate_publication_tables.py --check
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

A current TeX Live or MiKTeX installation with the `newtx` and TeX Gyre font packages is required by the AAAI-27 style.

Citations use the AAAI author--year style, for example `(Qiu et al. 2025)`. The camera-ready source does not use `\nocopyright`, and all fonts in the supplied PDF are embedded Type 1 fonts.

## Files

- `main.tex`: camera-ready manuscript
- `abstract_body.tex`: abstract text shared by both compiled PDFs
- `generated_primary_table.tex`: primary comparison generated from the committed CSV
- `generated_sensitivity_table.tex`: result table generated from the committed JSON logs
- `QIML_2026_UAV_HUBO_Submission.pdf`: compiled camera-ready paper
- `abstract.tex` and `QIML_2026_UAV_HUBO_Abstract.pdf`: stand-alone, single-column abstract for convenience; not the CRC manuscript
- `references.bib`: bibliography database
- `figures/`: paper figures
- `aaai2027.sty` and `aaai2027.bst`: unmodified official AAAI-27 files

## Final author checks

- Confirm author order, spelling, affiliations, and email addresses.
- Confirm every numerical result and bibliography entry against the experiment logs and cited sources.
- Upload the CRC only through the AAAI proceedings platform specified in the acceptance email.
- Follow any additional CRC invitation instructions, including any reproducibility-checklist requirement.
