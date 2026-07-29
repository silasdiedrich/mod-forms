"""High-level helpers tying together a grid, a form, a coloring style, and
PNG output.
"""

import io

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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


def natural_shape(box, max_dim=600):
    """A (rows, cols) grid shape that keeps pixels square for a given
    halfplane ``box``, with its longer side set to ``max_dim``.

    Without this, a non-square box (e.g. the paper's f_105 box, which is
    twice as wide as it is tall) rendered onto a square (res, res) grid
    comes out visibly stretched, since each pixel no longer represents a
    square region of the plane.
    """
    (x0, x1), (y0, y1) = box
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width <= 0 or height <= 0:
        return max_dim, max_dim
    if width >= height:
        cols = max_dim
        rows = max(1, round(max_dim * height / width))
    else:
        rows = max_dim
        cols = max(1, round(max_dim * width / height))
    return rows, cols


def padded_shape(shape, target_ratio):
    """The (rows, cols) an array of ``shape`` would become after
    :func:`pad_to_aspect` with ``target_ratio`` (width / height), without
    actually allocating/padding anything.
    """
    rows, cols = shape
    current_ratio = cols / rows
    if abs(current_ratio - target_ratio) < 1e-6:
        return rows, cols
    if current_ratio < target_ratio:
        return rows, max(cols, round(rows * target_ratio))
    return max(rows, round(cols / target_ratio)), cols


def pad_to_aspect(rgb, target_ratio, background=(1.0, 1.0, 1.0)):
    """Pad an (rows, cols, 3) RGB array with ``background`` so its
    width/height matches ``target_ratio``, centering the existing content.
    Only ever pads, never crops, so no content is lost.
    """
    rows, cols = rgb.shape[:2]
    new_rows, new_cols = padded_shape((rows, cols), target_ratio)
    if (new_rows, new_cols) == (rows, cols):
        return rgb
    canvas = np.empty((new_rows, new_cols, 3), dtype=rgb.dtype)
    canvas[..., :] = background
    row_off = (new_rows - rows) // 2
    col_off = (new_cols - cols) // 2
    canvas[row_off : row_off + rows, col_off : col_off + cols] = rgb
    return canvas


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


def _build_png_info(metadata):
    if not metadata:
        return None
    info = PngInfo()
    for key, value in metadata.items():
        info.add_text(key, value, zip=len(value) > 1024)
    return info


def rgb_to_png_bytes(rgb, metadata=None):
    """Encode an (rows, cols, 3) float RGB array (values in [0, 1]) as PNG
    bytes, with ``origin="lower"`` (row 0 = bottom) to match the paper's
    orientation, optionally embedding ``metadata`` (a dict of str -> str)
    as PNG text chunks -- see :mod:`modforms.reproduce` for reading it
    back and re-rendering from it.
    """
    arr = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(np.flipud(arr))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=_build_png_info(metadata))
    return buf.getvalue()


def save_png(rgb, path, metadata=None):
    """Save an (rows, cols, 3) RGB array to ``path`` as a PNG, optionally
    embedding ``metadata`` (see :func:`rgb_to_png_bytes`).
    """
    with open(path, "wb") as f:
        f.write(rgb_to_png_bytes(rgb, metadata=metadata))


def plot_form(
    form,
    region="halfplane",
    box=((-1, 1), (0, 2)),
    shape=(600, 600),
    disk_extent=1.02,
    style="phase-contour",
    out=None,
    metadata=None,
    **style_kwargs,
):
    """Convenience one-shot: evaluate ``form`` and render+save it.

    Returns the RGB array; also writes a PNG to ``out`` if given.
    """
    vals = evaluate_on_region(form, region=region, box=box, shape=shape, disk_extent=disk_extent)
    rgb = render(vals, style=style, **style_kwargs)
    if out is not None:
        save_png(rgb, out, metadata=metadata)
    return rgb
