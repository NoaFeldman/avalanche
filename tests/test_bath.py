import numpy as np
from pseudoavalanche.bath import build_bath_hamiltonian


def test_build_bath_hamiltonian_hermitian():
    H = build_bath_hamiltonian(2, 0, "gue", 1.0, seed=42)
    assert np.allclose(H, H.conj().T)
    assert H.shape == (4, 4)


def test_spectrum_rescaling():
    H = build_bath_hamiltonian(2, "haar", "poisson", 2.0, seed=1)
    eigs = np.linalg.eigvalsh(H)
    assert np.isclose(np.ptp(eigs), 2.0, atol=1e-6)
