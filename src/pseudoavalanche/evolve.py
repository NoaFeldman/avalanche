from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .bath import Bath
from .config import AvalancheConfig
from .hamiltonian import AvalancheHamiltonian
from .observables import _measure_state


def evolve(config: AvalancheConfig, state: np.ndarray, bath: Bath, record_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = AvalancheHamiltonian(config, bath)
    n_steps = int(np.ceil(config.t_max / config.dt)) + 1
    times = np.linspace(0.0, config.t_max, n_steps)
    trajectories = np.empty((n_steps, state.shape[0]), dtype=complex)
    trajectories[0] = state
    for step in range(1, n_steps):
        t0 = times[step - 1]
        t1 = times[step]
        psi = expm_multiply((-1j * (t1 - t0)) * hamiltonian, trajectories[step - 1])
        trajectories[step] = psi / np.linalg.norm(psi)
    return times, trajectories


def evolve_observables(config: AvalancheConfig, state: np.ndarray, bath: Bath) -> dict[str, np.ndarray]:
    hamiltonian = AvalancheHamiltonian(config, bath)
    n_steps = int(np.ceil(config.t_max / config.dt)) + 1
    times = np.linspace(0.0, config.t_max, n_steps)
    s = np.zeros((n_steps, config.L), dtype=float)
    entropies = np.zeros((n_steps, config.L), dtype=float)
    front = np.zeros(n_steps, dtype=int)
    t_ell = np.full(config.L, np.nan)

    current = state.astype(complex, copy=True)
    s[0], entropies[0] = _measure_state(current, config)
    for step in range(1, n_steps):
        t0 = times[step - 1]
        t1 = times[step]
        current = expm_multiply((-1j * (t1 - t0)) * hamiltonian, current)
        current = current / np.linalg.norm(current)
        s[step], entropies[step] = _measure_state(current, config)

    front = np.maximum(0, np.argmax(s > 0.5, axis=1) + 1)
    front[np.all(s <= 0.5, axis=1)] = 0
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
