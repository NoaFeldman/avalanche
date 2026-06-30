import numpy as np
from pseudoavalanche.analysis import fit_v_log
from pseudoavalanche.bath import Bath
from pseudoavalanche.config import AvalancheConfig
from pseudoavalanche.evolve import evolve
from pseudoavalanche.observables import compute_observables


def test_evolve_conserves_norm():
    config = AvalancheConfig(N0=1, L=1, xi=2.0, g=0.1, dt=0.1, t_max=0.5, seed=3)
    bath = Bath(config, seed=config.seed)
    psi0 = np.zeros(config.dim_total, dtype=complex)
    psi0[0] = 1.0
    times, traj = evolve(config, psi0, bath)
    norms = np.linalg.norm(traj, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_observables_range_and_monotonic_front():
    config = AvalancheConfig(N0=1, L=3, xi=0.5, g=0.2, dt=0.1, t_max=0.4, seed=4)
    bath = Bath(config, seed=config.seed)
    psi0 = np.zeros(config.dim_total, dtype=complex)
    psi0[0] = 1.0
    times, traj = evolve(config, psi0, bath)
    obs = compute_observables(times, traj, config)
    assert np.all((obs["s"] >= 0.0) & (obs["s"] <= 1.0))
    assert np.all(np.diff(obs["front"]) >= 0)
    assert isinstance(fit_v_log(obs["times"], obs["front"]), float)
