#!/usr/bin/env python3
"""Synchronize the QIML paper wording, documentation, and Colab wrappers.

This script intentionally does not change numerical benchmark results or citations.
It only keeps the repository aligned with the registered/submitted paper title and
single-instance benchmark framing, and generates thin Google Colab entry points
for every Python program under src/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

TITLE = "A Feasibility-Aware HUBO/QUBO Benchmark for UAV Obstacle-Avoidance with Visibility Cost"

ABSTRACT = r"""Quantum-assisted UAV path planning has recently been studied using QAOA-style obstacle-avoidance formulations, but a low-energy binary sample is not necessarily a valid UAV trajectory. This paper presents a feasibility-aware benchmark for a discrete $8\times8$, $L=20$ UAV grid-planning problem with obstacle avoidance, a line-of-sight visibility cost, and a higher-order temporal buffer-risk term. The horizon $L$ denotes discrete planning layers, not physical seconds. The model uses binary variables $x_{t,v}$ indicating whether the UAV occupies cell $v$ at time layer $t$ and forms a HUBO objective with one-hot, start, motion, obstacle, visibility, terminal-goal, and sustained near-obstacle proximity terms. The instance has 1,280 original binary variables and 118,344 polynomial terms, including 4,392 cubic monomials. We compare A*, RRT-best, a QUAV-style QAOA candidate-path selector, native trajectory-level simulated annealing (SA), the classical \texttt{neal} BQM sampler after HUBO-to-QUBO reduction, and an exact CP-SAT baseline that minimizes the native HUBO soft objective over hard-feasible paths. The CP-SAT solver returns OPTIMAL and certifies that the best observed SA path is globally optimal for the enforced feasible-path objective, with implementation HUBO energy $-20752.43$, soft cost 247.57, path length 14, six turns, and 85\% visibility. On the tested instance, the results show why feasibility-aware decoding, exact baselines, and quadratization effects matter; they do not establish quantum speedup or performance across an instance suite."""

KEYWORDS = r"HUBO, QUBO, feasibility-aware evaluation, UAV path planning, obstacle avoidance, visibility-aware planning, CP-SAT, QAOA, quantum-compatible optimization."


def replace_abstract(text: str) -> str:
    block = "\\begin{abstract}\n" + ABSTRACT + "\n\\end{abstract}"
    return re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        lambda _: block,
        text,
        count=1,
        flags=re.S,
    )


def sync_main_tex() -> None:
    path = PAPER / "main.tex"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\title\{.*?\}", lambda _: f"\\title{{{TITLE}}}", text, count=1)
    text = replace_abstract(text)

    # Keep 'benchmark' as the paper-level framing while retaining evaluation terminology
    # for metrics and checks where it is technically appropriate.
    replacements = {
        "Rather than claiming quantum speedup, it defines a reproducible HUBO/QUBO-compatible protocol on one fully specified instance,":
            "Rather than claiming quantum speedup, it defines a reproducible HUBO/QUBO-compatible benchmark on one fully specified instance,",
        "Third, we introduce a feasibility-aware evaluation protocol that reports structural feasibility, goal arrival, and full-task success separately.":
            "Third, we introduce a feasibility-aware benchmark that reports structural feasibility, goal arrival, and full-task success separately.",
        "These works reinforce two lessons that directly motivate our evaluation protocol:":
            "These works reinforce two lessons that directly motivate our benchmark:",
        "The evaluation protocol therefore reports feasibility rates and representative decoded trajectories, not energy alone.":
            "The benchmark therefore reports feasibility rates and representative decoded trajectories, not energy alone.",
        "The evaluation protocol exposes these failure modes by reporting structural feasibility, goal arrival, and full-task success separately.":
            "The benchmark exposes these failure modes by reporting structural feasibility, goal arrival, and full-task success separately.",
        "Here, ``benchmark'' denotes the reproducible evaluation protocol demonstrated on this single instance, not an instance suite.":
            "Here, ``benchmark'' denotes the reproducible single-instance evaluation used in this study, not an instance suite.",
        "This paper demonstrates a feasibility-aware HUBO/QUBO evaluation protocol for UAV obstacle-avoidance with visibility cost on one fully specified instance.":
            "This paper demonstrates a feasibility-aware HUBO/QUBO benchmark for UAV obstacle-avoidance with visibility cost on one fully specified instance.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Keep the keyword line synchronized without disturbing the surrounding LaTeX.
    text = re.sub(
        r"\\noindent\\textbf\{Keywords:\}\s*.*?\n",
        lambda _: f"\\noindent\\textbf{{Keywords:}} {KEYWORDS}\n",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def sync_abstract_tex() -> None:
    path = PAPER / "abstract.tex"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\title\{.*?\}", lambda _: f"\\title{{{TITLE}}}", text, count=1)
    text = replace_abstract(text)
    text = re.sub(
        r"\\noindent\\textbf\{Keywords:\}\s*.*?\n",
        lambda _: f"\\noindent\\textbf{{Keywords:}} {KEYWORDS}\n",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def sync_readmes() -> None:
    root_readme = f"""# UAV HUBO/QUBO Feasibility-Aware Benchmark for QIML 2026

This repository contains the code, figures, result table, LaTeX source, PDFs, and reproducibility material for:

**{TITLE}**

The work is a **single-instance feasibility-aware benchmark**. It does **not** claim quantum speedup, quantum advantage, or performance over an instance suite. The benchmark separates decoded structural feasibility, goal arrival, full-task success, native HUBO energy, and path-quality measures.

## QIML submission

The manuscript is the QIML 2026 submission associated with **Paper #4902**. Repository wording such as "submission" or "under review" must not be interpreted as acceptance or publication.

Current manuscript files:

- `paper/main.tex` - full manuscript source
- `paper/QIML_2026_UAV_HUBO_Submission.pdf` - compiled full manuscript
- `paper/abstract.tex` - stand-alone abstract source
- `paper/QIML_2026_UAV_HUBO_Abstract.pdf` - compiled stand-alone abstract

## Repository layout

```text
paper/       Full LaTeX source, bibliography, AAAI style files, and PDFs
figures/     Benchmark figures
src/         Reproducible Python implementations
colab/       Google Colab wrappers, one notebook per Python program in src/
results/     Main benchmark result tables
release/     ZIP snapshot of the GitHub package
```

## Main benchmark

The tested problem is an `8 x 8` grid with horizon `L = 20`, obstacle cells, a line-of-sight visibility cost, and a cubic temporal buffer-risk term. It has 1,280 original binary variables and 118,344 polynomial terms, including 4,392 cubic monomials.

The compared methods are:

- A* baseline
- RRT-best baseline
- QUAV-style QAOA candidate-path selector
- trajectory-level simulated annealing over the native HUBO energy
- HUBO-to-QUBO/BQM sampling with `neal`
- exact CP-SAT baseline over the hard-feasible native-HUBO soft objective

The QUAV-style implementation is a same-platform candidate-path reference and is **not** official QUAV code. `neal` is a classical simulated-annealing BQM sampler, not quantum hardware.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Google Colab

Open any notebook in `colab/`. Each notebook clones this repository, installs `requirements.txt`, changes into the repository directory, and provides a cell for the matching `src/*.py` program.

## Apple M2 Max runtime benchmark

From the repository root:

```bash
./run_m2max_benchmark.command
```

The first run creates an isolated `.venv-m2max` environment and installs the requirements. The complete workflow may take 30--65 minutes because it runs the full 50-run trajectory-SA experiment. It records solver-level wall-clock times while excluding plotting and file-serialization time.

Expected outputs are:

```text
results/m2max_runtime_benchmark.json
results/m2max_runtime_benchmark.csv
results/unified_comparison_m2max.csv
```

## Reproducibility notes

- Feasibility, goal arrival, and full-task success are intentionally reported separately.
- CP-SAT is the exact correctness anchor used in the reported benchmark table.
- A low-energy QUBO/BQM sample is not automatically a valid UAV trajectory.
- The horizon `L` denotes discrete planning layers, not physical seconds.
- The repository intentionally describes this as a **benchmark**, while making clear that the reported numerical study uses one fully specified instance.
"""
    (ROOT / "README.md").write_text(root_readme, encoding="utf-8")

    paper_readme = f"""# QIML 2026 submission package

This directory contains the manuscript source and compiled PDFs for:

**{TITLE}**

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

- The registered/submitted title is exactly: **{TITLE}**
- The paper-level framing is **benchmark**, not "evaluation protocol" as the title/name of the contribution.
- "Benchmark" refers to the reproducible single-instance study currently reported; it does not imply an instance suite.
- The reported exact reference is **CP-SAT**.
- The QUAV-style code is a candidate-path reference, not official QUAV code.
- `neal` is classical simulated annealing.
- `L=20` means 20 discrete planning layers, not 20 physical seconds.
- Submission status does not imply acceptance or publication.
"""
    (PAPER / "README.md").write_text(paper_readme, encoding="utf-8")

    clean = """# Clean GitHub package instructions

The repository already contains the synchronized paper source, Colab wrappers, figures, results, and compiled PDFs.

A generated ZIP snapshot is stored at:

```text
release/uav-hubo-qiml-2026-github-package.zip
```

To clone the live repository instead:

```bash
git clone https://github.com/alimohammadi44/uav-hubo-qiml-2026.git
cd uav-hubo-qiml-2026
```

The ZIP and repository are generated from the same synchronized source. Do not manually rename the paper contribution from **benchmark** to **evaluation protocol** in the title, abstract, or repository documentation.
"""
    (ROOT / "GITHUB_CLEAN_UPLOAD.md").write_text(clean, encoding="utf-8")


def make_colab_notebook(src: Path) -> dict:
    rel = src.relative_to(ROOT).as_posix()
    stem = src.stem
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {stem} - Google Colab wrapper\n",
                "\n",
                f"This notebook runs `{rel}` from the QIML UAV HUBO/QUBO benchmark repository.\n",
                "The underlying Python source remains the authoritative implementation.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git clone -q https://github.com/alimohammadi44/uav-hubo-qiml-2026.git\n",
                "%cd /content/uav-hubo-qiml-2026\n",
                "!pip -q install -r requirements.txt\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run\n",
                "If the program accepts command-line options, edit the next cell to add the required arguments.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"%run {rel}\n"],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": f"{stem}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate_colab() -> None:
    out = ROOT / "colab"
    out.mkdir(exist_ok=True)
    expected = set()
    for src in sorted((ROOT / "src").glob("*.py")):
        dst = out / f"{src.stem}.ipynb"
        expected.add(dst.name)
        dst.write_text(json.dumps(make_colab_notebook(src), indent=2) + "\n", encoding="utf-8")
    for old in out.glob("*.ipynb"):
        if old.name not in expected:
            old.unlink()


def audit() -> None:
    checks = {
        "registered title": TITLE,
        "single-instance scope": "single-instance",
        "Paper #4902": "Paper #4902",
    }
    readmes = [ROOT / "README.md", PAPER / "README.md"]
    merged = "\n".join(p.read_text(encoding="utf-8") for p in readmes)
    for label, needle in checks.items():
        if needle not in merged:
            raise RuntimeError(f"Documentation audit failed: missing {label}: {needle}")

    for tex in [PAPER / "main.tex", PAPER / "abstract.tex"]:
        text = tex.read_text(encoding="utf-8")
        if TITLE not in text:
            raise RuntimeError(f"Title mismatch in {tex}")
        if ABSTRACT not in text:
            raise RuntimeError(f"Abstract mismatch in {tex}")

    forbidden_doc_phrases = [
        "camera-ready package",
        "exact CP-SAT/MILP baseline",
        "A Feasibility-Aware Evaluation Protocol for HUBO/QUBO UAV Obstacle-Avoidance with Visibility Cost",
    ]
    for md in ROOT.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for phrase in forbidden_doc_phrases:
            if phrase in text:
                raise RuntimeError(f"Inconsistent phrase in {md}: {phrase}")


def main() -> None:
    sync_main_tex()
    sync_abstract_tex()
    sync_readmes()
    generate_colab()
    audit()
    print("QIML repository text, documentation, and Colab wrappers synchronized.")


if __name__ == "__main__":
    main()
