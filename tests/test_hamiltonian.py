import numpy as np
from scipy.sparse.linalg import LinearOperator

from pseudoavalanche.bath import Bath
from pseudoavalanche.config import AvalancheConfig
from pseudoavalanche.hamiltonian import AvalancheHamiltonian


def test_linear_operator_matches_dense():
    config = AvalancheConfig(N0=2, L=2, xi=1.0, g=0.3, dt=0.1, t_max=0.2, seed=1)
    bath = Bath(config, seed=config.seed)
    op = AvalancheHamiltonian(config, bath)
    assert isinstance(op, LinearOperator)
    psi = np.arange(config.dim_total, dtype=complex)
    dense = np.asarray(op.matmat(np.eye(config.dim_total, dtype=complex)))
    dense2 = np.vstack([op.matvec(np.eye(config.dim_total, dtype=complex)[:, i]) for i in range(config.dim_total)]).T
    assert np.allclose(dense, dense2)


def test_norm_conservation():
    config = AvalancheConfig(N0=1, L=1, xi=1.0, g=0.5, dt=0.1, t_max=0.3, seed=2)
    bath = Bath(config, seed=config.seed)
    psi = np.zeros(config.dim_total, dtype=complex)
    psi[0] = 1.0
    vec = op = AvalancheHamiltonian(config, bath).matvec(psi)
    assert np.isfinite(np.linalg.norm(psi))
    assert np.isfinite(np.linalg.norm(op))
