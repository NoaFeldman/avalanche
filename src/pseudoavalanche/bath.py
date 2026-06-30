from __future__ import annotations

import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eigh

try:
    from qiskit.quantum_info import random_clifford
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


def _single_qubit_gate_matrix(gate: str) -> np.ndarray:
    if gate == "H":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if gate == "S":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if gate == "T":
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    raise ValueError(f"Unknown gate {gate}")


def _cnot_matrix(n_qubits: int, control: int, target: int) -> np.ndarray:
    dim = 1 << n_qubits
    U = np.eye(dim, dtype=complex)
    for basis in range(dim):
        if (basis >> control) & 1:
            flipped = basis ^ (1 << target)
            U[basis, basis] = 0
            U[flipped, basis] = 1
    return U


def _single_qubit_operator(n_qubits: int, qubit: int, gate: np.ndarray) -> np.ndarray:
    result = np.array([[1.0 + 0j]])
    for q in range(n_qubits):
        if q == qubit:
            result = np.kron(result, gate)
        else:
            result = np.kron(result, np.eye(2, dtype=complex))
    return result


def random_clifford_unitary(n_qubits: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    U = np.eye(dim, dtype=complex)
    n_gates = max(10, 20 * n_qubits)
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.25:
            control, target = rng.choice(n_qubits, size=2, replace=False)
            U = _cnot_matrix(n_qubits, control, target) @ U
        else:
            qubit = int(rng.integers(0, n_qubits))
            gate = _single_qubit_gate_matrix(rng.choice(["H", "S"]))
            U = _single_qubit_operator(n_qubits, qubit, gate) @ U
    return U


def random_bath_unitary(n_qubits: int, lambda_dope: int | str, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if lambda_dope == "haar":
        return unitary_group.rvs(1 << n_qubits, random_state=rng)
    if lambda_dope == 0:
        if _HAS_QISKIT:
            return random_clifford(n_qubits).to_matrix()
        return random_clifford_unitary(n_qubits, seed=seed)
    if isinstance(lambda_dope, int) and lambda_dope > 0:
        if _HAS_QISKIT:
            base = random_clifford(n_qubits).to_matrix()
        else:
            base = random_clifford_unitary(n_qubits, seed=seed)
        U = base.copy()
        for _ in range(lambda_dope):
            qubit = int(rng.integers(0, n_qubits))
            gate = _single_qubit_operator(n_qubits, qubit, _single_qubit_gate_matrix("T"))
            U = gate @ U
        return U
    raise ValueError(f"Invalid lambda_dope {lambda_dope}")


def sample_spectrum(n_levels: int, spec_mode: str, W: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if spec_mode == "gue":
        a = rng.standard_normal((n_levels, n_levels)) + 1j * rng.standard_normal((n_levels, n_levels))
        H = (a + a.conj().T) / np.sqrt(2 * n_levels)
        evals = eigh(H, eigvals_only=True)
    elif spec_mode == "semicircle":
        values = []
        while len(values) < n_levels:
            x = rng.uniform(-1.0, 1.0)
            if rng.uniform(0.0, 1.0) < np.sqrt(max(0.0, 1.0 - x * x)):
                values.append(x)
        evals = np.array(values, dtype=float)
    elif spec_mode == "poisson":
        evals = rng.uniform(-0.5, 0.5, size=n_levels)
    elif spec_mode == "picket":
        evals = np.linspace(-0.5, 0.5, num=n_levels)
    else:
        raise ValueError(f"Unknown spec_mode {spec_mode}")
    evals = np.asarray(evals, dtype=float)
    evals -= np.mean(evals)
    span = np.ptp(evals)
    if span == 0:
        return np.zeros_like(evals)
    return evals * (W / span)


def build_bath_hamiltonian(n_qubits: int, lambda_dope: int | str, spec_mode: str, W: float, seed: int | None = None) -> np.ndarray:
    U = random_bath_unitary(n_qubits, lambda_dope, seed=seed)
    diag = sample_spectrum(1 << n_qubits, spec_mode, W, seed=seed)
    return (U * diag[np.newaxis, :]) @ U.conj().T


def edge_operator(n_qubits: int) -> np.ndarray:
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return _single_qubit_operator(n_qubits, 0, z)


class Bath:
    def __init__(self, config, seed: int | None = None):
        self.config = config
        self.seed = seed
        self.H_B = build_bath_hamiltonian(config.N0, config.lambda_dope, config.spec_mode, config.W, seed=seed)
        self.edge_op = edge_operator(config.N0)
