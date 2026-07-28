"""Grids for the upper halfplane and the Poincare disk, and the Moebius map
between them used throughout the paper (Section 2.1)::

    phi(w) = (1 - i*w) / (w - i)

which sends the disk D = {w : |w| < 1} to the upper halfplane H, with
-i -> 0, 0 -> i, i -> i*infinity.
"""

import numpy as np


def halfplane_grid(box, shape):
    """Return an (rows, cols) complex array Z sampling ``box = ((x0, x1), (y0, y1))``.

    Row 0 corresponds to y0, so that ``imshow(..., origin="lower")`` renders
    the plot with increasing imaginary part upward, as in the paper.
    """
    (x0, x1), (y0, y1) = box
    rows, cols = shape
    xs = np.linspace(x0, x1, cols)
    ys = np.linspace(y0, y1, rows)
    X, Y = np.meshgrid(xs, ys)
    return X + 1j * Y


def phi(w):
    """Moebius transform D -> H from Section 2.1 of the paper."""
    return (1 - 1j * w) / (w - 1j)


def disk_grid(shape, extent=1.02):
    """Return (W, mask): a complex grid over the square [-extent, extent]^2
    and a boolean mask that is True inside the open unit disk.
    """
    rows, cols = shape
    xs = np.linspace(-extent, extent, cols)
    ys = np.linspace(-extent, extent, rows)
    X, Y = np.meshgrid(xs, ys)
    W = X + 1j * Y
    mask = (X**2 + Y**2) < 1.0
    return W, mask
