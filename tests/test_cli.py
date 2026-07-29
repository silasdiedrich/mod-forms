"""End-to-end tests for the CLI, focused on the plot -> replot round trip
that backs the "download carries everything needed to reproduce it" story.
"""

import json

import numpy as np
from PIL import Image

from modforms.cli import main


def test_plot_embeds_metadata(tmp_path):
    out = tmp_path / "delta.png"
    main(["plot", "--form", "delta", "--region", "disk", "--style", "standard", "--shape", "40", "40", "--out", str(out)])

    img = Image.open(out)
    assert "modforms_reproduce" in img.text
    meta = json.loads(img.text["modforms_reproduce"])
    assert meta["form"]["source"] == "delta"
    assert meta["form"]["label"] is not None
    assert len(meta["form"]["coeffs"]) == 400  # default --terms


def test_replot_reproduces_pixel_identical_output(tmp_path):
    original = tmp_path / "original.png"
    replotted = tmp_path / "replotted.png"

    main(
        [
            "plot",
            "--form",
            "delta",
            "--region",
            "disk",
            "--style",
            "phase-contour",
            "--base",
            "2.0",
            "--shape",
            "50",
            "50",
            "--out",
            str(original),
        ]
    )
    main(["replot", str(original), "--out", str(replotted)])

    a = np.asarray(Image.open(original).convert("RGB"))
    b = np.asarray(Image.open(replotted).convert("RGB"))
    np.testing.assert_array_equal(a, b)


def test_replot_with_shape_override_changes_resolution(tmp_path):
    original = tmp_path / "original.png"
    resized = tmp_path / "resized.png"

    main(["plot", "--form", "delta", "--region", "disk", "--shape", "50", "50", "--out", str(original)])
    main(["replot", str(original), "--out", str(resized), "--shape", "20", "20"])

    resized_img = Image.open(resized)
    assert resized_img.size == (20, 20)
    # The replot re-embeds metadata reflecting the *new* resolution.
    meta = json.loads(resized_img.text["modforms_reproduce"])
    assert meta["content_shape"] == [20, 20]


def test_replot_on_non_modforms_png_raises_clear_error(tmp_path):
    plain = tmp_path / "plain.png"
    Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(plain)

    try:
        main(["replot", str(plain), "--out", str(tmp_path / "out.png")])
        assert False, "expected an error for a PNG with no modforms_reproduce metadata"
    except ValueError as e:
        assert "modforms_reproduce" in str(e)
