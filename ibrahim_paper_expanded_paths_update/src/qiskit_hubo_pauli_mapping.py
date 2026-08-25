"""
HUBO-to-Pauli-Z Hamiltonian mapping for IBM/Qiskit compatibility.

Purpose
-------
This script documents the gate-based quantum path for the Ibrahim/QIML
UAV HUBO formulation. It does not try to run the full 8x8, L=20 instance
on IBM hardware because the direct state-variable mapping would require
1280 qubits. Instead, it provides reusable functions and a tiny reduced
example showing how HUBO monomials are converted into Pauli-Z strings.

Mathematics
-----------
For a binary variable x_i in {0,1}, use

    x_i = (I - Z_i) / 2.

For a HUBO monomial c * prod_{i in S} x_i,

    c prod_{i in S} x_i
      -> c / 2^|S| * sum_{R subset S} (-1)^|R| prod_{i in R} Z_i.

This supports linear, quadratic, cubic, and higher-order HUBO terms. The
output dictionary can be converted to qiskit's SparsePauliOp when Qiskit
is installed.

Run
---
    python src/qiskit_hubo_pauli_mapping.py

Outputs
-------
    outputs/qiskit_pauli_mapping_summary.json

No IBM token is needed for this script. A token is required only when
submitting circuits to IBM hardware/runtime.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

PauliKey = Tuple[int, ...]       # tuple of qubit indices carrying Z operators
HUBOKey = Tuple[int, ...]        # tuple of binary variable indices
Polynomial = Dict[HUBOKey, float]
PauliPolynomial = Dict[PauliKey, float]


def add_term(poly: Dict[Tuple[int, ...], float], key: Iterable[int], coeff: float, tol: float = 1e-12) -> None:
    """Accumulate a polynomial term and remove near-zero coefficients."""
    key = tuple(sorted(set(key)))
    if abs(coeff) < tol:
        return
    poly[key] = poly.get(key, 0.0) + float(coeff)
    if abs(poly[key]) < tol:
        del poly[key]


def monomial_to_pauli(key: HUBOKey, coeff: float) -> PauliPolynomial:
    """Map c * prod_i x_i to a Pauli-Z polynomial.

    The empty Pauli key () is the constant/identity term.
    """
    key = tuple(sorted(set(key)))
    m = len(key)
    out: PauliPolynomial = {}
    scale = coeff / (2.0 ** m)
    # Expand prod_i (I - Z_i) = sum_R (-1)^|R| prod_{i in R} Z_i
    for r in range(m + 1):
        for subset in itertools.combinations(key, r):
            add_term(out, subset, scale * ((-1.0) ** r))
    return out


def hubo_to_pauli(hubo: Polynomial) -> PauliPolynomial:
    """Convert a whole HUBO polynomial to a Pauli-Z polynomial."""
    out: PauliPolynomial = {}
    for key, coeff in hubo.items():
        mapped = monomial_to_pauli(key, coeff)
        for pkey, pc in mapped.items():
            add_term(out, pkey, pc)
    return out


def pauli_key_to_label(key: PauliKey, num_qubits: int) -> str:
    """Convert a Pauli key to a Qiskit-style label.

    Qiskit labels are ordered left-to-right from qubit n-1 to qubit 0.
    For readability we mark Z at the requested qubit indices.
    """
    chars = ["I"] * num_qubits
    for q in key:
        chars[num_qubits - 1 - q] = "Z"
    return "".join(chars)


def pauli_to_sparse_pauli_op(pauli: PauliPolynomial, num_qubits: int):
    """Create a qiskit.quantum_info.SparsePauliOp if Qiskit is installed."""
    try:
        from qiskit.quantum_info import SparsePauliOp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Qiskit is not installed in this environment") from exc

    labels = []
    coeffs = []
    constant = pauli.get((), 0.0)
    for key, coeff in sorted(pauli.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if key == ():
            continue
        labels.append(pauli_key_to_label(key, num_qubits))
        coeffs.append(coeff)
    op = SparsePauliOp(labels, coeffs) if labels else SparsePauliOp(["I" * num_qubits], [0.0])
    return op, constant


def build_tiny_uav_qubo() -> Polynomial:
    """A very small 2x2, L=3 QUBO-style UAV example.

    Variables are x_{t,v}, encoded by index t*4+v. It includes one-hot,
    start, movement, and terminal-target penalties. This is intentionally
    reduced for Hamiltonian inspection, not for performance claims.
    """
    grid = 2
    V = grid * grid
    L = 3
    start = 0          # (0,0)
    target = 3         # (1,1)
    lam_onehot = 50.0
    lam_start = 50.0
    lam_move = 50.0
    lam_term = 25.0
    lam_goal = 4.0

    def idx(t: int, v: int) -> int:
        return t * V + v

    def rc(v: int):
        return divmod(v, grid)

    def neighbors(v: int):
        r, c = rc(v)
        nbs = {v}
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < grid and 0 <= cc < grid:
                nbs.add(rr * grid + cc)
        return nbs

    hubo: Polynomial = {}

    # one-hot: (sum_v x_t,v - 1)^2 = -sum_v x + 2 sum_{u<v} x_u x_v + const
    for t in range(L):
        for v in range(V):
            add_term(hubo, (idx(t, v),), -lam_onehot)
        for u in range(V):
            for v in range(u + 1, V):
                add_term(hubo, (idx(t, u), idx(t, v)), 2 * lam_onehot)

    # start preference. The one-hot term already penalizes multiple active cells;
    # this adds an energy reward for the correct start cell, matching the
    # reduced Qiskit QAOA runner used in the paper package.
    add_term(hubo, (idx(0, start),), -lam_start)

    # invalid movement penalties
    for t in range(L - 1):
        for u in range(V):
            nb = neighbors(u)
            for v in range(V):
                if v not in nb:
                    add_term(hubo, (idx(t, u), idx(t + 1, v)), lam_move)

    # terminal target penalty
    for v in range(V):
        if v != target:
            add_term(hubo, (idx(L - 1, v),), lam_term)

    # soft distance-to-goal cost
    tr, tc = rc(target)
    for t in range(L):
        weight = (t + 1) / L
        for v in range(V):
            r, c = rc(v)
            dist = ((r - tr) ** 2 + (c - tc) ** 2) ** 0.5
            add_term(hubo, (idx(t, v),), lam_goal * weight * dist)

    return hubo


def main() -> None:
    hubo = build_tiny_uav_qubo()
    num_qubits = 12
    pauli = hubo_to_pauli(hubo)
    degree_counts = {}
    for key in hubo:
        degree_counts[len(key)] = degree_counts.get(len(key), 0) + 1
    pauli_counts = {}
    for key in pauli:
        pauli_counts[len(key)] = pauli_counts.get(len(key), 0) + 1

    qiskit_available = False
    sparse_pauli_terms = None
    constant = pauli.get((), 0.0)
    try:
        op, constant_from_op = pauli_to_sparse_pauli_op(pauli, num_qubits)
        qiskit_available = True
        sparse_pauli_terms = len(op)
        constant = float(constant_from_op)
    except Exception:
        pass

    first_terms = []
    for key, coeff in sorted(pauli.items(), key=lambda kv: (len(kv[0]), kv[0]))[:20]:
        first_terms.append({
            "pauli_key": list(key),
            "label": pauli_key_to_label(key, num_qubits) if key else "I" * num_qubits,
            "coeff": coeff,
        })

    summary = {
        "purpose": "HUBO-to-Pauli-Z mapping for reduced IBM/Qiskit QAOA compatibility tests",
        "full_instance_warning": "The full 8x8, L=20 state-variable model has 1280 binary variables, so direct QAOA would require about 1280 qubits.",
        "tiny_instance": {"grid": [2, 2], "L": 3, "num_qubits": num_qubits},
        "hubo_terms": len(hubo),
        "hubo_degree_counts": degree_counts,
        "pauli_terms_including_constant": len(pauli),
        "pauli_string_order_counts": pauli_counts,
        "constant_identity_coeff": constant,
        "qiskit_available_in_this_run": qiskit_available,
        "sparse_pauli_terms_excluding_constant": sparse_pauli_terms,
        "first_terms": first_terms,
    }

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "qiskit_pauli_mapping_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
