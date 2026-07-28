import numpy as np

from modforms.forms import delta_coeffs, eisenstein


def test_ramanujan_tau_known_values():
    # tau(1)..tau(10), see e.g. OEIS A000594.
    expected = [1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920]
    tau = delta_coeffs(10)
    assert list(tau) == expected


def test_eisenstein_e4_constant_and_first_coeff():
    e4 = eisenstein(4, n_terms=5)
    # E4 = 1 + 240 q + 2160 q^2 + ...
    assert e4.coeffs[0] == 1
    assert e4.coeffs[1] == 240
    assert e4.coeffs[2] == 2160
