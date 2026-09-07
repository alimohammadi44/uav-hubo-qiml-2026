# QIML 2026 submission package

This package contains an author-visible AAAI 2024-format manuscript for the
Second AAAI Symposium on Quantum Information & Machine Learning (QIML 2026).
The symposium specifies single-blind review, so the author names and
affiliations remain visible.

## Build

Run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The supplied `aaai24.sty` and `aaai24.bst` files are unmodified copies of the
AAAI 2024 author-kit files linked by the symposium.

Citations use the AAAI author--year style (for example,
`(Qiu et al. 2025)`), not numbered citations such as `[12]`.

The two PNG figure files are the original supplied color images and have not
been edited or converted.

## Review versus camera-ready

`main.tex` uses the normal author-visible AAAI style and calls `\nocopyright`
for the review manuscript. For a camera-ready version, follow the instructions
sent by AAAI Press, remove `\nocopyright`, add any requested copyright-year or
rights information, and complete the required publication form.

## Final checks for the authors

- Confirm the author order, spelling, affiliations, and all email addresses.
- Confirm every numerical result, path, caption, and bibliography entry against
  the experiment logs and cited sources.
- Confirm that the repository URL is public before submission.
- Upload the generated PDF as a Full Paper; the compiled manuscript is six
  US-letter pages.
