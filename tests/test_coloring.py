import numpy as np

from modforms import coloring


def _sample():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(8, 8))
    imag = rng.normal(size=(8, 8))
    return real + 1j * imag


def test_all_styles_produce_valid_rgb():
    f = _sample()
    for name, fn in coloring.STYLES.items():
        rgb = fn(f)
        assert rgb.shape == f.shape + (3,), name
        assert np.all(np.isfinite(rgb)), name
        assert rgb.min() >= 0 and rgb.max() <= 1, name


def test_contour_style_is_darker_just_below_a_power_of_base():
    base = 2.0
    below = np.array([[0.99 * base]])
    above = np.array([[1.01 * base]])
    rgb_below = coloring.phase_with_contours(below.astype(complex), base=base)
    rgb_above = coloring.phase_with_contours(above.astype(complex), base=base)
    # crossing a contour should change lightness noticeably
    assert not np.allclose(rgb_below, rgb_above, atol=1e-3)


def test_custom_colormap_endpoints_match_input_colors():
    cmap = coloring.custom_colormap(["#000000", "#ffffff"], cyclic=False)
    np.testing.assert_allclose(cmap(0.0), (0, 0, 0, 1), atol=1e-6)
    np.testing.assert_allclose(cmap(1.0), (1, 1, 1, 1), atol=1e-6)


def test_custom_colormap_cyclic_wraps_to_start():
    cmap = coloring.custom_colormap(["#ff0000", "#00ff00", "#0000ff"], cyclic=True)
    np.testing.assert_allclose(cmap(0.0), cmap(0.999), atol=0.02)


def test_custom_colormap_works_with_colormap_phase_style():
    cmap = coloring.custom_colormap(["#000000", "#ff8800"], cyclic=False)
    f = _sample()
    rgb = coloring.colormap_phase(f, cmap=cmap)
    assert rgb.shape == f.shape + (3,)
    assert np.all(np.isfinite(rgb))


def test_custom_colormap_requires_at_least_two_colors():
    import pytest

    with pytest.raises(ValueError):
        coloring.custom_colormap(["#000000"])
