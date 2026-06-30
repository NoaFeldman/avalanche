from __future__ import annotations

import numpy as np

from .config import AvalancheConfig


def _spin_z_expectation(state: np.ndarray, config: AvalancheConfig, spin_index: int) -> float:
    psi = state.reshape((config.dim_bath, config.dim_spins))
    weights = np.abs(psi) ** 2
    bit = (np.arange(config.dim_spins) >> spin_index) & 1
    z = 1.0 - 2.0 * bit
    return np.sum(weights * z[np.newaxis, :])


def _single_spin_entropy(state: np.ndarray, config: AvalancheConfig, spin_index: int) -> float:
    psi = state.reshape((config.dim_bath, config.dim_spins))
    mask = ((np.arange(config.dim_spins) >> spin_index) & 1) == 0
    psi0 = psi[:, mask].ravel()
    psi1 = psi[:, ~mask].ravel()
    rho = np.array([
        [np.vdot(psi0, psi0), np.vdot(psi0, psi1)],
        [np.vdot(psi1, psi0), np.vdot(psi1, psi1)],
    ], dtype=complex)
    evals = np.linalg.eigvalsh(rho.real)
    evals = np.clip(evals, 0.0, 1.0)
    entropy = -np.sum([p * np.log2(p) for p in evals if p > 0.0])
    return float(entropy)


def _measure_state(state: np.ndarray, config: AvalancheConfig) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(config.L, dtype=float)
    entropies = np.zeros(config.L, dtype=float)
    for ell in range(config.L):
        z = _spin_z_expectation(state, config, ell)
        s[ell] = 0.5 * (1.0 - z)
        entropies[ell] = _single_spin_entropy(state, config, ell)
    return s, entropies


def compute_observables(times: np.ndarray, trajectories: np.ndarray, config: AvalancheConfig) -> dict[str, np.ndarray]:
    n_steps = len(times)
    s = np.zeros((n_steps, config.L), dtype=float)
    entropies = np.zeros((n_steps, config.L), dtype=float)
    for step in range(n_steps):
        s[step], entropies[step] = _measure_state(trajectories[step], config)
    front = np.maximum(0, np.argmax(s > 0.5, axis=1) + 1)
    front[np.all(s <= 0.5, axis=1)] = 0
    t_ell = np.full(config.L, np.nan)
    for ell in range(config.L):
        above = np.where(s[:, ell] > 0.5)[0]
        if above.size > 0:
            t_ell[ell] = times[above[0]]
    return {
        "times": times,
        "s": s,
        "entropies": entropies,
        "front": front,
        "t_ell": t_ell,
    }
