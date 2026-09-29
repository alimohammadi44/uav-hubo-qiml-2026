# UAV HUBO/QUBO Feasibility-Aware Benchmark for QIML 2026

This repository contains the code, figures, result table, LaTeX source, PDFs, and reproducibility material for:

**A Feasibility-Aware HUBO/QUBO Benchmark for UAV Obstacle-Avoidance with Visibility Cost**

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
