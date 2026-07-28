"""Evaluate a modular form from its Fourier (q-)expansion on a grid of
points in the upper halfplane, following Section 2.2 of the paper: the
form is approximated by truncating its q-expansion to some number of terms.
"""

import csv

import numpy as np


class QExpansion:
    """A truncated q-expansion ``f(z) = sum_{n=start}^{start+N-1} a_n q^n``,
    where ``q = exp(2*pi*i*z)``.

    Parameters
    ----------
    coeffs : sequence of complex
        Fourier coefficients a_start, a_start+1, ..., in order.
    start : int
        Exponent of q for coeffs[0]. Cuspforms typically have start=1.
    weight, level : optional
        Metadata only (not used in evaluation), handy for labeling plots.
    label : optional str
        e.g. an LMFDB label, for bookkeeping/titles.
    """

    def __init__(self, coeffs, start=1, weight=None, level=None, label=None):
        self.coeffs = np.asarray(coeffs, dtype=complex)
        self.start = start
        self.weight = weight
        self.level = level
        self.label = label

    def __len__(self):
        return len(self.coeffs)

    def __call__(self, z):
        """Evaluate on a numpy array (or scalar) of complex points z."""
        q = np.exp(2j * np.pi * np.asarray(z, dtype=complex))
        acc = np.zeros_like(q, dtype=complex)
        for a in self.coeffs[::-1]:
            acc = acc * q + a
        return acc * q**self.start

    @classmethod
    def from_list(cls, coeffs, **kwargs):
        return cls(coeffs, **kwargs)

    @classmethod
    def from_csv(cls, source, column=0, **kwargs):
        """Load coefficients from a single column of a CSV file, e.g. a
        q-expansion downloaded from the LMFDB (https://www.lmfdb.org).

        ``source`` may be a path, or a file-like object with a ``read()``
        method (e.g. a Streamlit ``UploadedFile``).
        """
        if hasattr(source, "read"):
            text = source.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            lines = text.splitlines()
        else:
            with open(source, newline="") as f:
                lines = f.read().splitlines()

        coeffs = []
        for row in csv.reader(lines):
            if not row:
                continue
            try:
                coeffs.append(complex(row[column].strip()))
            except ValueError:
                continue  # skip header lines
        return cls(coeffs, **kwargs)
