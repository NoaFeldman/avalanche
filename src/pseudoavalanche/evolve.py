from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .config import AvalancheConfig
from .hamiltonian import AvalancheHamiltonian
from .bath import Bath


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
