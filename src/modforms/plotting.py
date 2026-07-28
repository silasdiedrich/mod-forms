"""High-level helpers tying together a grid, a form, a coloring style, and
matplotlib output.
"""

import numpy as np
import matplotlib.pyplot as plt

from .grid import halfplane_grid, disk_grid, phi
from .coloring import STYLES


def evaluate_on_region(form, region="halfplane", box=((-1, 1), (0, 2)), shape=(600, 600), disk_extent=1.02):
    """Evaluate ``form`` on a grid over either the halfplane box or the
    Poincare disk. Points outside the disk are set to NaN (Section 3.1,
    step 2's masking).
    """
    if region == "halfplane":
        z = halfplane_grid(box, shape)
        return form(z)
    if region == "disk":
        w, mask = disk_grid(shape, extent=disk_extent)
        z = phi(w)
        # Points outside the disk can map to extreme values of z (even the
        # lower halfplane), which would otherwise overflow the q-expansion;
        # replace them with a harmless value before evaluating and mask
        # them out afterwards.
        safe_z = np.where(mask, z, 1j)
        vals = form(safe_z)
        return np.where(mask, vals, np.nan + 1j * np.nan)
    raise ValueError(f"unknown region {region!r}, expected 'halfplane' or 'disk'")


def render(vals, style="phase-contour", background=(1.0, 1.0, 1.0), **style_kwargs):
    """Apply a coloring style (see ``modforms.coloring.STYLES``) to
    evaluated form values, filling masked-out (NaN) points with
    ``background``.
    """
    style_fn = STYLES[style] if isinstance(style, str) else style
    rgb = style_fn(vals, **style_kwargs)
    nan_mask = np.isnan(vals.real) | np.isnan(vals.imag)
    if np.any(nan_mask):
        rgb = rgb.copy()
        rgb[nan_mask] = background
    return np.clip(rgb, 0.0, 1.0)


def save_png(rgb, path, dpi=150):
    """Save an (rows, cols, 3) RGB array to ``path`` with no padding/axes."""
    rows, cols = rgb.shape[:2]
    fig = plt.figure(figsize=(cols / dpi, rows / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_form(
    form,
    region="halfplane",
    box=((-1, 1), (0, 2)),
    shape=(600, 600),
    disk_extent=1.02,
    style="phase-contour",
    out=None,
    **style_kwargs,
):
    """Convenience one-shot: evaluate ``form`` and render+save it.

    Returns the RGB array; also writes a PNG to ``out`` if given.
    """
    vals = evaluate_on_region(form, region=region, box=box, shape=shape, disk_extent=disk_extent)
    rgb = render(vals, style=style, **style_kwargs)
    if out is not None:
        save_png(rgb, out)
    return rgb
