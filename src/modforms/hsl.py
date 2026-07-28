"""Vectorized RGB <-> HSL conversion (numpy arrays, shape (..., 3), values in [0, 1]).

``colorsys`` is not actually vectorizable (its branches use scalar ``if``
statements), so we reimplement the standard conversions with ``np.where``.
"""

import numpy as np


def rgb_to_hsl(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    lightness = (maxc + minc) / 2.0

    denom = np.where(lightness <= 0.5, maxc + minc, 2.0 - maxc - minc)
    saturation = np.where(delta < 1e-12, 0.0, delta / np.clip(denom, 1e-12, None))

    safe_delta = np.clip(delta, 1e-12, None)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    hue = np.select(
        [r == maxc, g == maxc, b == maxc],
        [bc - gc, 2.0 + rc - bc, 4.0 + gc - rc],
        default=0.0,
    )
    hue = (hue / 6.0) % 1.0
    hue = np.where(delta < 1e-12, 0.0, hue)

    return np.stack([hue, saturation, lightness], axis=-1)


def _hue_to_channel(p, q, t):
    t = t % 1.0
    out = np.where(t < 1 / 6, p + (q - p) * 6 * t, q)
    out = np.where(t >= 1 / 2, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p), out)
    return out


def hsl_to_rgb(hsl):
    hue, saturation, lightness = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    q = np.where(lightness < 0.5, lightness * (1 + saturation), lightness + saturation - lightness * saturation)
    p = 2 * lightness - q

    r = _hue_to_channel(p, q, hue + 1 / 3)
    g = _hue_to_channel(p, q, hue)
    b = _hue_to_channel(p, q, hue - 1 / 3)

    gray = saturation < 1e-12
    r = np.where(gray, lightness, r)
    g = np.where(gray, lightness, g)
    b = np.where(gray, lightness, b)

    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
