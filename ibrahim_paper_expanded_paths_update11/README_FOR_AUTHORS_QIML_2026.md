# QIML 2026 Author-Feedback Package

This package contains the revised Ibrahim/QIML UAV HUBO/QUBO paper draft prepared for rapid coauthor review before the Sept. 1 submission deadline supplied by Ali.

## Main updated files

- `paper/qiml_paper_author_feedback_qiskit.pdf`
- `paper/qiml_paper_author_feedback_qiskit.tex`
- `src/qiskit_hubo_pauli_mapping.py`
- `src/task3_qiskit_qaoa.py`

## Main changes in this revision

1. Sharpened the title to avoid overclaiming full occlusion-aware target tracking.
2. Added a dedicated problem-statement/scope paragraph:
   - static discrete grid planning;
   - fixed target/goal;
   - visibility is a soft line-of-sight cost;
   - not continuous quadrotor control;
   - not full moving-target tracking;
   - time horizon `L` means discrete planning layers, not seconds.
3. Removed informal wording such as “uploaded CP-SAT run.”
4. Added an IBM/Qiskit Hamiltonian-compatibility subsection.
5. Added the mathematical HUBO-to-Pauli mapping:
   `x_i = (I - Z_i)/2` and monomial-to-Pauli expansion.
6. Explained why the full 8x8, L=20 model is not yet a practical IBM/QAOA hardware run:
   direct mapping requires 1280 qubits.
7. Clarified D-Wave vs IBM/Qiskit roles:
   - D-Wave BQM/QPU requires quadratization of cubic terms.
   - D-Wave CQM can express constraints more naturally but still needs auxiliary modeling for cubic terms.
   - IBM/Qiskit can represent the HUBO as a Pauli-Z Hamiltonian for reduced-instance QAOA.
8. Added a reduced Qiskit check explanation:
   2x2, L=3, 12 variables, 38 QUBO terms.
9. Reframed Qiskit result as a compatibility and feasibility-decoding test, not as a performance result.
10. Updated code availability to include the HUBO-to-Pauli Qiskit mapping script.

## Suggested coauthor review questions

1. Is the problem statement now clear enough for reviewers?
2. Is the title appropriately conservative?
3. Should the reduced Qiskit QAOA result remain in the main paper or move to appendix/supplement?
4. Should the Qiskit mapping be emphasized more strongly as the quantum contribution?
5. Which final author names, affiliations, and repository link should be inserted?
6. What is the exact QIML 2026 page limit and formatting requirement to enforce in the next revision?

## How to run the new Qiskit mapping script

```bash
cd <package root>
python src/qiskit_hubo_pauli_mapping.py
```

This does not require an IBM token. It only maps the reduced HUBO/QUBO example into a Pauli-Z Hamiltonian representation and writes:

```text
outputs/qiskit_pauli_mapping_summary.json
```

An IBM token is required only when submitting real circuits to IBM hardware/runtime.
