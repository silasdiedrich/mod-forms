"""Domain-coloring / phase-plot styles from the paper.

Each function takes a complex numpy array ``f`` (values of a modular form
on a grid) and returns an RGB array of shape ``f.shape + (3,)`` with values
in ``[0, 1]``. NaNs in ``f`` propagate to NaNs in the output, so callers can
mask them out (see :func:`modforms.plotting.render`).

Section references are to Lowry-Duda, "Visualizing Modular Forms"
(arXiv:2002.05234).
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb

from .hsl import rgb_to_hsl, hsl_to_rgb

try:  # matplotlib >= 3.5
    from matplotlib import colormaps as _mpl_colormaps

    def _lookup_cmap(name):
        return _mpl_colormaps[name]

except ImportError:  # pragma: no cover - older matplotlib
    from matplotlib import cm as _mpl_cm

    def _lookup_cmap(name):
        return _mpl_cm.get_cmap(name)


def _get_cmap(cmap):
    """Resolve ``cmap`` to a matplotlib Colormap. Accepts either a
    built-in colormap name, or an already-built Colormap object (e.g.
    from :func:`custom_colormap`), which is returned as-is.
    """
    if isinstance(cmap, str):
        return _lookup_cmap(cmap)
    return cmap


def custom_colormap(colors, cyclic=True, name="custom"):
    """Build a matplotlib Colormap by interpolating between a list of
    colors (anything matplotlib can parse: hex strings like "#ff8800",
    (r, g, b) tuples in [0, 1], named colors, ...).

    With ``cyclic=True`` (the default), the last color wraps smoothly back
    to the first, which matters for phase-based styles here since phase is
    itself cyclic (see the discussion of the ``twilight`` colormap in
    Section 3.2 of the paper).
    """
    colors = list(colors)
    if len(colors) < 2:
        raise ValueError("need at least 2 colors to build a colormap")
    if cyclic and colors[0] != colors[-1]:
        colors = colors + [colors[0]]
    return LinearSegmentedColormap.from_list(name, colors)


def _phase01(f, offset=0.0, direction=1):
    """arg(f) mapped to [0, 1), optionally reversed/offset (step 3 of 3.1)."""
    theta = np.angle(f) / (2 * np.pi)
    return (direction * theta + offset) % 1.0


def _safe_log(mag, base):
    mag = np.where(mag > 0, mag, np.nan)
    return np.log(mag) / np.log(base)


def _adjust_lightness(rgb, delta_l):
    hsl = rgb_to_hsl(rgb)
    hsl = hsl.copy()
    hsl[..., 2] = np.clip(hsl[..., 2] + delta_l, 0.0, 1.0)
    return hsl_to_rgb(hsl)


def _contour_adjustment(mag, base, strength):
    """A sawtooth in log_base(|f|) mod 1 that jumps sharply at each power of
    ``base``: this is Remark 2's trick for drawing magnitude contours
    without ever computing them explicitly.
    """
    t = _safe_log(mag, base) % 1.0
    return np.nan_to_num((t - 0.5) * 2 * strength, nan=0.0)


# -- Section 2.2.1: standard domain coloring -------------------------------


def standard_domain_coloring(f):
    """Hue = arg(f), brightness = |f| (0 black, infinity white)."""
    hue = _phase01(f)
    val = (2 / np.pi) * np.arctan(np.abs(f))
    hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
    return hsv_to_rgb(np.nan_to_num(hsv))


# -- Section 2.2.2: magnitude only, no color -------------------------------


def magnitude_grayscale(f, alpha=0.25):
    """Grayscale, black = 0 magnitude, via arctan(log(|f|^alpha + 1))."""
    mag = np.abs(f)
    v = np.arctan(np.log(mag**alpha + 1)) / (np.pi / 2)
    v = np.clip(np.nan_to_num(v), 0, 1)
    return np.stack([v, v, v], axis=-1)


# -- Section 2.2.3: magnitude mod 1, linear spacing (LMFDB style) ---------


def magnitude_periodic_linear(f, offset=2 / 3, direction=-1):
    """Hue = |f| mod 1. Defaults mimic the LMFDB (blue at 0, then purple,
    red, orange, yellow as magnitude increases).
    """
    hue = (direction * np.abs(f) + offset) % 1.0
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    return hsv_to_rgb(np.nan_to_num(hsv))


# -- Section 2.2.4: magnitude, logarithmic spacing -------------------------


def magnitude_periodic_log(f, base=7.0, offset=2 / 3, direction=-1):
    """Hue = log_base(|f|) mod 1."""
    t = _safe_log(np.abs(f), base)
    hue = (direction * t + offset) % 1.0
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    return hsv_to_rgb(np.nan_to_num(hsv))


# -- Section 2.2.5: pure phase plots ---------------------------------------


def phase_only(f, offset=0.0):
    """Hue = arg(f)/(2*pi), full saturation and value."""
    hue = _phase01(f, offset=offset)
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    return hsv_to_rgb(np.nan_to_num(hsv))


# -- Section 2.2.6: phase plots with magnitude contours --------------------


def phase_with_contours(f, base=2.0, offset=0.0, strength=0.35):
    """Phase plot with logarithmically-spaced magnitude contours drawn as
    sharp lightness jumps (Remark 2): two consecutive contours indicate a
    doubling of magnitude by default (base=2).
    """
    rgb = phase_only(f, offset=offset)
    adjust = _contour_adjustment(np.abs(f), base, strength)
    return _adjust_lightness(rgb, adjust)


# -- Section 3: matplotlib-colormap variants -------------------------------


def colormap_phase(f, cmap="twilight", offset=0.0):
    """Phase plot using an arbitrary matplotlib colormap (Section 3.1,
    steps 3-4). ``twilight`` is cyclic and a natural fit for phase.
    """
    hue = _phase01(f, offset=offset)
    rgba = _get_cmap(cmap)(np.nan_to_num(hue))
    return rgba[..., :3]


def colormap_phase_with_contours(f, cmap="cividis", base=2.0, offset=0.0, strength=0.35):
    """Colormap phase plot with magnitude contours (Section 3.1, steps 5-8)."""
    rgb = colormap_phase(f, cmap=cmap, offset=offset)
    adjust = _contour_adjustment(np.abs(f), base, strength)
    return _adjust_lightness(rgb, adjust)


def colormap_magnitude(f, cmap="viridis", mode="log", base=7.0, offset=0.0, direction=-1):
    """Colormapped version of the periodic magnitude plots (2.2.3/2.2.4),
    using an arbitrary matplotlib colormap instead of raw hue.
    """
    mag = np.abs(f)
    t = _safe_log(mag, base) if mode == "log" else mag
    t = (direction * t + offset) % 1.0
    rgba = _get_cmap(cmap)(np.nan_to_num(t))
    return rgba[..., :3]


def colormap_domain_coloring(f, cmap="cividis", offset=0.0):
    """Colormapped version of the default (magnitude-as-brightness) domain
    coloring plot (end of Section 3.2 / Figure 10): hue comes from a
    matplotlib colormap applied to phase, brightness from magnitude.
    """
    rgb = colormap_phase(f, cmap=cmap, offset=offset)
    val = (2 / np.pi) * np.arctan(np.abs(f))
    hsl = rgb_to_hsl(rgb)
    hsl = hsl.copy()
    # Rescale so lightness spans [0, 1] with 0 at magnitude 0.
    hsl[..., 2] = np.nan_to_num(val) * (0.5 + 0.5 * hsl[..., 2])
    return hsl_to_rgb(hsl)


STYLES = {
    "standard": standard_domain_coloring,
    "magnitude": magnitude_grayscale,
    "periodic-linear": magnitude_periodic_linear,
    "periodic-log": magnitude_periodic_log,
    "phase": phase_only,
    "phase-contour": phase_with_contours,
    "colormap-phase": colormap_phase,
    "colormap-phase-contour": colormap_phase_with_contours,
    "colormap-magnitude": colormap_magnitude,
    "colormap-standard": colormap_domain_coloring,
}

# Extra keyword arguments each style accepts, beyond the values array.
# Shared by the CLI and the UI so both stay in sync with `STYLES`.
STYLE_PARAMS = {
    "magnitude": ["alpha"],
    "periodic-linear": ["offset"],
    "periodic-log": ["base", "offset"],
    "phase": ["offset"],
    "phase-contour": ["base", "offset"],
    "colormap-phase": ["cmap", "offset"],
    "colormap-phase-contour": ["cmap", "base", "offset"],
    "colormap-magnitude": ["cmap", "base", "offset"],
    "colormap-standard": ["cmap", "offset"],
}

STYLE_LABELS = {
    "standard": "Standard domain coloring (§2.2.1)",
    "magnitude": "Magnitude only, grayscale (§2.2.2)",
    "periodic-linear": "Periodic magnitude, linear — LMFDB style (§2.2.3)",
    "periodic-log": "Periodic magnitude, logarithmic (§2.2.4)",
    "phase": "Pure phase plot (§2.2.5)",
    "phase-contour": "Phase with magnitude contours (§2.2.6)",
    "colormap-phase": "Colormap phase plot (§3)",
    "colormap-phase-contour": "Colormap phase + contours (§3)",
    "colormap-magnitude": "Colormap magnitude (§3)",
    "colormap-standard": "Colormap standard domain coloring (§3)",
}
