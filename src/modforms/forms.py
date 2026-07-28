"""Built-in example modular forms.

``delta()`` is computed exactly (the Ramanujan tau function, via the
Jacobi/Euler pentagonal number theorem) so it needs no external data.

The other three forms used as running examples in the paper -- a weight 4
cuspform of level 5, a weight 2 cuspform of level 105, and a weight 20
cuspform of level 10 -- have irrational/algebraic Fourier coefficients and
are catalogued on the LMFDB. Fetch their q-expansions from there and load
them with :meth:`modforms.QExpansion.from_list` or ``from_csv``, e.g.

    https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/5/4/a/a/1/1/
    https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/105/2/a/a/1/1/
    https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/10/20/a/a/1/1/

(and Delta itself is also on the LMFDB, at
https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/1/12/a/a/1/1/).
"""

import numpy as np

from .qexpansion import QExpansion

# Coefficients -2k/B_k for the level-1 Eisenstein series
# E_k = 1 - (2k/B_k) * sum_{n>=1} sigma_{k-1}(n) q^n.
_EISENSTEIN_CONST = {4: 240, 6: -504, 8: 480, 10: -264, 14: -24}


def _euler_function_coeffs(n_terms):
    """Coefficients c_0, ..., c_{n_terms} of prod_{n>=1} (1 - q^n), via the
    pentagonal number theorem: prod(1-q^n) = sum_k (-1)^k q^(k(3k-1)/2),
    the sum ranging over all integers k.
    """
    c = np.zeros(n_terms + 1, dtype=np.int64)
    c[0] = 1
    k = 1
    while True:
        g_pos = k * (3 * k - 1) // 2
        g_neg = k * (3 * k + 1) // 2
        sign = -1 if k % 2 else 1
        added = False
        if g_pos <= n_terms:
            c[g_pos] += sign
            added = True
        if g_neg <= n_terms:
            c[g_neg] += sign
            added = True
        if not added:
            break
        k += 1
    return c


def _poly_mult_trunc(a, b, n_terms):
    result = np.zeros(n_terms + 1, dtype=np.int64)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        room = n_terms - i
        if room < 0:
            continue
        result[i : i + room + 1] += ai * b[: room + 1]
    return result


def _poly_power_trunc(base, exponent, n_terms):
    result = np.zeros(n_terms + 1, dtype=np.int64)
    result[0] = 1
    p = base.copy()
    e = exponent
    while e > 0:
        if e & 1:
            result = _poly_mult_trunc(result, p, n_terms)
        e >>= 1
        if e:
            p = _poly_mult_trunc(p, p, n_terms)
    return result


def delta_coeffs(n_terms=400):
    """Fourier coefficients tau(1), ..., tau(n_terms) of the Ramanujan
    Delta function, Delta(z) = q * prod_{n>=1}(1-q^n)^24 = sum tau(n) q^n.
    """
    euler = _euler_function_coeffs(n_terms - 1)
    phi24 = _poly_power_trunc(euler, 24, n_terms - 1)
    return phi24[:n_terms]


def delta(n_terms=400):
    """The Ramanujan Delta function, the unique cuspform of weight 12 on
    SL(2, Z). LMFDB label 1.12.a.a.1.1.
    """
    return QExpansion(
        delta_coeffs(n_terms), start=1, weight=12, level=1, label="1.12.a.a.1.1 (Delta)"
    )


def _sigma(k_minus_1, n_terms):
    """sigma_{k_minus_1}(n) for n = 1, ..., n_terms, via a divisor sieve."""
    sig = np.zeros(n_terms + 1, dtype=object)
    for d in range(1, n_terms + 1):
        sig[d::d] += d**k_minus_1
    return sig[1:]


def eisenstein(k, n_terms=400):
    """The level-1 Eisenstein series E_k for k in {4, 6, 8, 10, 14}, the
    weights for which E_k is not a cusp form but still has a simple,
    exactly computable q-expansion. Not periodic-free in the same sense as
    a cuspform, but a convenient exact example for exercising the plotting
    code without any external data.
    """
    if k not in _EISENSTEIN_CONST:
        raise ValueError(f"k must be one of {sorted(_EISENSTEIN_CONST)}")
    const = _EISENSTEIN_CONST[k]
    coeffs = np.empty(n_terms + 1, dtype=object)
    coeffs[0] = 1
    coeffs[1:] = const * _sigma(k - 1, n_terms)
    return QExpansion(
        coeffs.astype(complex), start=0, weight=k, level=1, label=f"E_{k}"
    )
