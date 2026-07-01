import numpy as np
from pseudoavalanche.analysis import fit_v_log
from pseudoavalanche.bath import Bath
from pseudoavalanche.config import AvalancheConfig
from pseudoavalanche.evolve import evolve, evolve_observables
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


def test_streaming_observables_shape():
    config = AvalancheConfig(N0=1, L=2, xi=1.0, g=0.1, dt=0.1, t_max=0.3, seed=5)
    bath = Bath(config, seed=config.seed)
    psi0 = np.zeros(config.dim_total, dtype=complex)
    psi0[0] = 1.0
    obs = evolve_observables(config, psi0, bath)
    assert obs["times"].shape[0] == int(np.ceil(config.t_max / config.dt)) + 1
    assert obs["s"].shape == (obs["times"].shape[0], config.L)


def test_large_g_depolarizes_single_spin():
    config = AvalancheConfig(N0=2, L=1, xi=0.5, g=2.0, dt=0.1, t_max=1.0, seed=6)
    bath = Bath(config, seed=config.seed)
    psi0 = np.zeros(config.dim_total, dtype=complex)
    psi0[0] = 1.0
    times, traj = evolve(config, psi0, bath)
    zexp = []
    for state in traj:
        weights = np.abs(state.reshape((config.dim_bath, config.dim_spins))) ** 2
        zexp.append(float(np.sum(weights * np.array([1.0, -1.0])[np.newaxis, :])))
    assert zexp[-1] < 0.9


def test_front_threshold_is_named_and_controls_front():
    config = AvalancheConfig(N0=1, L=1, xi=0.5, g=2.0, dt=0.1, t_max=2.0, seed=7, thermalization_threshold=0.4)
    state0 = np.zeros(config.dim_total, dtype=complex)
    state0[0] = 1.0
    state1 = np.zeros(config.dim_total, dtype=complex)
    state1[1] = 1.0
    trajectories = np.stack([state0, state1], axis=0)
    obs = compute_observables(np.array([0.0, 1.0]), trajectories, config)
    assert obs["front"].tolist() == [0, 1]
