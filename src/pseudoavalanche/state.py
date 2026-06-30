from __future__ import annotations

import numpy as np

from .config import AvalancheConfig


def typical_bath_state(dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(dim,)) + 1j * rng.normal(size=(dim,))
    state /= np.linalg.norm(state)
    return state


def initial_state(config: AvalancheConfig, seed: int | None = None) -> np.ndarray:
    bath_state = typical_bath_state(config.dim_bath, seed=seed)
    spin_zero = np.zeros(config.dim_spins, dtype=complex)
    spin_zero[0] = 1.0
    return np.kron(bath_state, spin_zero)


def typical_states(config: AvalancheConfig, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []
    for sample in range(config.n_typicality):
        states.append(initial_state(config, seed=int(rng.integers(0, 2**31))))
    return np.stack(states, axis=0)
