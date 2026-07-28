import numpy as np

from modforms.qexpansion import QExpansion


def test_evaluate_matches_direct_sum():
    coeffs = [1, -24, 252, -1472]
    f = QExpansion(coeffs, start=1)
    z = np.array([0.3 + 0.7j, -0.2 + 1.1j])
    q = np.exp(2j * np.pi * z)
    expected = sum(c * q ** (n + 1) for n, c in enumerate(coeffs))
    np.testing.assert_allclose(f(z), expected)


def test_start_offset():
    f = QExpansion([1, 2, 3], start=0)
    z = np.array([0.1 + 0.5j])
    q = np.exp(2j * np.pi * z)
    expected = 1 + 2 * q + 3 * q**2
    np.testing.assert_allclose(f(z), expected)
