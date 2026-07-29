"""Tests for the PNG reproduce-metadata round trip: build_metadata ->
embed in a PNG -> load_metadata -> reconstruct the exact form/style used.
"""

import io

import numpy as np
import pytest
from PIL import Image

from modforms import forms, plotting, reproduce
from modforms.coloring import custom_colormap


def test_style_kwargs_to_json_builtin_cmap():
    result = reproduce.style_kwargs_to_json({"cmap": "cividis", "base": 2.0}, cmap_fingerprint="cividis")
    assert result == {"base": 2.0, "cmap": {"type": "builtin", "name": "cividis"}}


def test_style_kwargs_to_json_custom_cmap():
    fingerprint = ("custom", ("#000000", "#ffffff"), True)
    result = reproduce.style_kwargs_to_json({"cmap": object()}, cmap_fingerprint=fingerprint)
    assert result == {"cmap": {"type": "custom", "colors": ["#000000", "#ffffff"], "cyclic": True}}


def test_build_and_load_metadata_round_trip():
    form = forms.delta(20)
    metadata = reproduce.build_metadata(
        form,
        "delta",
        region="disk",
        box=None,
        disk_extent=1.02,
        style="phase-contour",
        style_kwargs_json={"base": 2.0},
        content_shape=(100, 100),
        output_shape=(100, 100),
        target_ratio=None,
        background=(0.0, 0.0, 0.0),
    )
    assert "modforms_reproduce" in metadata
    assert metadata["Software"].startswith("modforms")

    rgb = np.zeros((4, 4, 3))
    png_bytes = plotting.rgb_to_png_bytes(rgb, metadata=metadata)
    img = Image.open(io.BytesIO(png_bytes))

    loaded = reproduce.load_metadata(img)
    assert loaded["form"]["label"] == form.label
    assert loaded["form"]["weight"] == 12
    assert loaded["form"]["level"] == 1
    assert loaded["region"] == "disk"
    assert loaded["background"] == [0.0, 0.0, 0.0]


def test_form_from_metadata_reconstructs_exact_coefficients():
    original = forms.delta(15)
    metadata = reproduce.build_metadata(
        original, "delta", region="disk", box=None, disk_extent=1.02, style="standard",
        style_kwargs_json={}, content_shape=(10, 10), output_shape=(10, 10),
        target_ratio=None, background=(1.0, 1.0, 1.0),
    )
    import json

    loaded = json.loads(metadata["modforms_reproduce"])
    rebuilt = reproduce.form_from_metadata(loaded)
    np.testing.assert_allclose(rebuilt.coeffs, original.coeffs)
    assert rebuilt.start == original.start
    assert rebuilt.weight == original.weight
    assert rebuilt.level == original.level


def test_style_kwargs_from_metadata_rebuilds_custom_colormap():
    meta = {"style_kwargs": {"cmap": {"type": "custom", "colors": ["#000000", "#ff0000"], "cyclic": False}, "base": 2.0}}
    kwargs = reproduce.style_kwargs_from_metadata(meta)
    assert kwargs["base"] == 2.0
    assert callable(kwargs["cmap"])
    # sanity: the rebuilt colormap behaves like the one built directly
    expected = custom_colormap(["#000000", "#ff0000"], cyclic=False)
    assert kwargs["cmap"](0.0) == expected(0.0)
    assert kwargs["cmap"](1.0) == expected(1.0)


def test_style_kwargs_from_metadata_builtin_cmap_stays_a_string():
    meta = {"style_kwargs": {"cmap": {"type": "builtin", "name": "viridis"}}}
    kwargs = reproduce.style_kwargs_from_metadata(meta)
    assert kwargs["cmap"] == "viridis"


def test_load_metadata_raises_clear_error_without_reproduce_chunk():
    rgb = np.zeros((4, 4, 3))
    png_bytes = plotting.rgb_to_png_bytes(rgb)  # no metadata
    img = Image.open(io.BytesIO(png_bytes))
    with pytest.raises(ValueError, match="modforms_reproduce"):
        reproduce.load_metadata(img)


def test_end_to_end_replot_matches_original_pixels():
    """The full loop: render -> embed metadata -> load metadata -> re-render
    -> pixel-identical output, without needing the original form object.
    """
    form = forms.delta(50)
    vals = plotting.evaluate_on_region(form, region="disk", shape=(40, 40))
    rgb = plotting.render(vals, style="phase-contour", background=(0.0, 0.0, 0.0), base=2.0)

    metadata = reproduce.build_metadata(
        form, "delta", region="disk", box=None, disk_extent=1.02, style="phase-contour",
        style_kwargs_json={"base": 2.0}, content_shape=(40, 40), output_shape=(40, 40),
        target_ratio=None, background=(0.0, 0.0, 0.0),
    )
    png_bytes = plotting.rgb_to_png_bytes(rgb, metadata=metadata)

    loaded = reproduce.load_metadata(Image.open(io.BytesIO(png_bytes)))
    rebuilt_form = reproduce.form_from_metadata(loaded)
    rebuilt_kwargs = reproduce.style_kwargs_from_metadata(loaded)

    vals2 = plotting.evaluate_on_region(rebuilt_form, region=loaded["region"], shape=tuple(loaded["content_shape"]))
    rgb2 = plotting.render(vals2, style=loaded["style"], background=tuple(loaded["background"]), **rebuilt_kwargs)

    np.testing.assert_array_equal(rgb, rgb2)
