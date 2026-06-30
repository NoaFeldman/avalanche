from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .config import AvalancheConfig
from .bath import Bath


def _basis_z_values(num_bits: int) -> np.ndarray:
    indices = np.arange(1 << num_bits)
    return 1.0 - 2.0 * ((indices[:, None] >> np.arange(num_bits)) & 1)


def _spin_flip_indices(L: int, qubit: int) -> np.ndarray:
    dim = 1 << L
    indices = np.arange(dim, dtype=np.int64)
    return indices ^ (1 << qubit)


class AvalancheHamiltonian(LinearOperator):
    def __init__(self, config: AvalancheConfig, bath: Bath):
        self.config = config
        self.bath = bath
        self.dim_bath = config.dim_bath
        self.dim_spins = config.dim_spins
        self.shape = (config.dim_total, config.dim_total)
        self.dtype = np.complex128
        self._spin_z = _basis_z_values(config.L)
        self._spin_flip = [_spin_flip_indices(config.L, j) for j in range(config.L)]
        self._h_values = self._sample_h_fields(config)
        self._zz_diag = self._compute_zz_diagonal(config)
        self._g_values = self._compute_g_values(config)

    def _sample_h_fields(self, config: AvalancheConfig) -> np.ndarray:
        kind, low, high = config.h_dist
        rng = np.random.default_rng(config.seed)
        if kind == "uniform":
            return rng.uniform(low, high, size=config.L)
        if kind == "normal":
            return rng.normal(low, high, size=config.L)
        raise ValueError(f"Unknown h_dist kind {kind}")

    def _compute_zz_diagonal(self, config: AvalancheConfig) -> np.ndarray:
        zvals = self._spin_z
        diag = np.zeros(self.dim_spins, dtype=np.float64)
        if config.J == 0.0:
            return diag
        for j in range(config.L - 1):
            diag += config.J * zvals[:, j] * zvals[:, j + 1]
        return diag

    def _compute_g_values(self, config: AvalancheConfig) -> np.ndarray:
        j = np.arange(1, config.L + 1, dtype=np.float64)
        return config.g * np.exp(-j / config.xi)

    def _apply_spin_coupling(self, psi: np.ndarray, coupling: float, flipped: np.ndarray) -> np.ndarray:
        flipped_psi = psi[:, flipped]
        return coupling * flipped_psi

    def _matvec(self, vec: np.ndarray) -> np.ndarray:
        psi = vec.reshape((self.dim_bath, self.dim_spins))
        result = self.bath.H_B.dot(psi)
        diag_fields = np.sum(self._h_values * self._spin_z, axis=1)
        result += psi * diag_fields[np.newaxis, :]
        result += psi * self._zz_diag[np.newaxis, :]
        for coupling, flipped in zip(self._g_values, self._spin_flip):
            result += self.bath.edge_op.dot(self._apply_spin_coupling(psi, coupling, flipped))
        return result.ravel()

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        return self._matvec(vec)

    def _matmat(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat)
        if mat.ndim == 1:
            return self.matvec(mat)
        return np.column_stack([self.matvec(mat[:, i]) for i in range(mat.shape[1])])

    def matmat(self, mat: np.ndarray) -> np.ndarray:
        return self._matmat(mat)

    def rmatvec(self, vec: np.ndarray) -> np.ndarray:
        return self._matvec(vec)

    def _rmatmat(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat)
        if mat.ndim == 1:
            return self.rmatvec(mat)
        return np.column_stack([self.rmatvec(mat[:, i]) for i in range(mat.shape[1])])

    def rmatmat(self, mat: np.ndarray) -> np.ndarray:
        return self._rmatmat(mat)
