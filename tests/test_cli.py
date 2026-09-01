"""End-to-end tests for the CLI, focused on the plot -> replot round trip
that backs the "download carries everything needed to reproduce it" story,
plus the video subcommand.
"""

import json
import shutil
import textwrap

import numpy as np
import pytest
from PIL import Image

from modforms.cli import main

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


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


def test_inspect_prints_gui_mapped_sections_for_builtin_delta(tmp_path, capsys):
    out = tmp_path / "delta.png"
    main(
        [
            "plot", "--form", "delta", "--region", "disk", "--style", "colormap-phase-contour",
            "--cmap", "cividis", "--base", "2.0", "--shape", "40", "40", "--out", str(out),
        ]
    )
    capsys.readouterr()  # discard the "wrote ..." line

    main(["inspect", str(out)])
    output = capsys.readouterr().out

    assert "=== 1. Form ===" in output
    assert "Source: Built-in" in output
    assert "Form: Delta — weight 12, level 1" in output
    assert "=== 2. Region ===" in output
    assert "Region: Poincaré disk" in output
    assert "=== 3. Style ===" in output
    assert "Colormap: Built-in -- cividis" in output
    assert "Base: 2.0" in output
    assert "=== 4. Resolution & Aspect Ratio ===" in output
    assert "Detail (long edge): 40" in output
    assert "=== 5. Background ===" in output
    assert "White" in output


def test_inspect_shows_eisenstein_weight_and_halfplane_box(tmp_path, capsys):
    out = tmp_path / "e6.png"
    main(
        [
            "plot", "--form", "e6", "--region", "halfplane", "--box=-2,2,0,3",
            "--style", "colormap-phase", "--cmap", "twilight", "--shape", "30", "20", "--out", str(out),
        ]
    )
    capsys.readouterr()

    main(["inspect", str(out)])
    output = capsys.readouterr().out
    assert "Form: E6 — Eisenstein, weight 6" in output
    assert "Region: Upper halfplane" in output
    assert "x0=-2.0  x1=2.0  y0=0.0  y1=3.0" in output


def test_inspect_shows_custom_colormap_and_black_background(tmp_path, capsys):
    from modforms import forms, plotting, reproduce
    from modforms.coloring import custom_colormap

    form = forms.delta(20)
    cmap = custom_colormap(["#000000", "#ff0088"], cyclic=True)
    vals = plotting.evaluate_on_region(form, region="disk", shape=(20, 20))
    rgb = plotting.render(vals, style="colormap-phase-contour", background=(0.0, 0.0, 0.0), base=2.0, cmap=cmap)
    meta = reproduce.build_metadata(
        form, "delta", region="disk", box=None, disk_extent=1.02, style="colormap-phase-contour",
        style_kwargs_json=reproduce.style_kwargs_to_json(
            {"cmap": cmap, "base": 2.0}, ("custom", ("#000000", "#ff0088"), True)
        ),
        content_shape=(20, 20), output_shape=(20, 20), target_ratio=None, background=(0.0, 0.0, 0.0),
    )
    out = tmp_path / "custom.png"
    plotting.save_png(rgb, str(out), metadata=meta)

    main(["inspect", str(out)])
    output = capsys.readouterr().out
    assert "Colormap: Custom -- colors=['#000000', '#ff0088']  cyclic=True" in output
    assert "Outside the disk (or masked region): Black" in output


def test_inspect_on_non_modforms_png_raises_clear_error(tmp_path):
    plain = tmp_path / "plain.png"
    Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(plain)
    with pytest.raises(ValueError, match="modforms_reproduce"):
        main(["inspect", str(plain)])


def _write_tiny_timeline(path):
    path.write_text(
        textwrap.dedent(
            """
            from modforms.video import Keyframe

            DELTA = {"source": "delta", "n_terms": 20}

            TIMELINE = [
                Keyframe(time=0.0, form=DELTA, region="disk", scale=1.02),
                Keyframe(time=0.5, form=DELTA, region="disk", scale=0.5),
            ]
            """
        )
    )


def test_video_dry_run_prints_estimate_without_rendering(tmp_path, capsys):
    timeline_path = tmp_path / "timeline.py"
    _write_tiny_timeline(timeline_path)
    out = tmp_path / "out.mp4"

    main(["video", str(timeline_path), "--out", str(out), "--dry-run", "--fps", "4", "--width", "20", "--height", "20"])

    captured = capsys.readouterr()
    assert "frames" in captured.out
    assert "estimated total" in captured.out
    assert not out.exists()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_video_renders_mp4(tmp_path):
    timeline_path = tmp_path / "timeline.py"
    _write_tiny_timeline(timeline_path)
    out = tmp_path / "out.mp4"

    main(["video", str(timeline_path), "--out", str(out), "--fps", "4", "--width", "24", "--height", "24"])

    assert out.exists() and out.stat().st_size > 0


def test_video_preview_uses_low_res_defaults(tmp_path, capsys):
    timeline_path = tmp_path / "timeline.py"
    _write_tiny_timeline(timeline_path)
    out = tmp_path / "out.mp4"

    main(["video", str(timeline_path), "--out", str(out), "--preview", "--dry-run"])

    output = capsys.readouterr().out
    assert "[preview]" in output
    assert "480x270" in output
    assert "15fps" in output


def test_video_preview_allows_individual_overrides(tmp_path, capsys):
    timeline_path = tmp_path / "timeline.py"
    _write_tiny_timeline(timeline_path)
    out = tmp_path / "out.mp4"

    main(["video", str(timeline_path), "--out", str(out), "--preview", "--width", "640", "--dry-run"])

    output = capsys.readouterr().out
    assert "640x270" in output  # width overridden, height/fps still preview defaults


def test_video_without_preview_uses_full_defaults(tmp_path, capsys):
    timeline_path = tmp_path / "timeline.py"
    _write_tiny_timeline(timeline_path)
    out = tmp_path / "out.mp4"

    main(["video", str(timeline_path), "--out", str(out), "--dry-run"])

    output = capsys.readouterr().out
    assert "[full]" in output
    assert "1280x720" in output
    assert "30fps" in output


def test_video_timeline_without_TIMELINE_or_build_timeline_raises(tmp_path):
    timeline_path = tmp_path / "bad_timeline.py"
    timeline_path.write_text("x = 1\n")
    out = tmp_path / "out.mp4"

    with pytest.raises(ValueError, match="TIMELINE"):
        main(["video", str(timeline_path), "--out", str(out), "--dry-run"])
