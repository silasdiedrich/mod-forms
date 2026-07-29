"""Tests for the ambient zoom video pipeline: keyframe interpolation,
crossfade logic, safety warnings, and (if ffmpeg is available) an actual
end-to-end render.
"""

import shutil
import subprocess

import numpy as np
import pytest

from modforms import video

HAVE_FFMPEG = shutil.which("ffmpeg") is not None

DELTA = {"source": "delta", "n_terms": 30}
E4 = {"source": "eisenstein", "weight": 4, "n_terms": 30}


def test_smootherstep_endpoints_and_monotonic():
    assert video.smootherstep(0.0) == 0.0
    assert video.smootherstep(1.0) == 1.0
    xs = np.linspace(0, 1, 50)
    ys = [video.smootherstep(x) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))  # monotonically increasing
    assert video.smootherstep(-1.0) == 0.0  # clamps below 0
    assert video.smootherstep(2.0) == 1.0  # clamps above 1


def test_timeline_requires_at_least_one_keyframe():
    with pytest.raises(ValueError):
        video.Timeline([])


def test_timeline_bracket_before_first_and_after_last():
    kfs = [
        video.Keyframe(time=1.0, form=DELTA),
        video.Keyframe(time=3.0, form=DELTA),
    ]
    tl = video.Timeline(kfs)
    a, b, p = tl.bracket(0.0)
    assert a is b is kfs[0] and p == 0.0
    a, b, p = tl.bracket(5.0)
    assert a is b is kfs[1] and p == 0.0


def test_timeline_bracket_between_keyframes():
    kfs = [
        video.Keyframe(time=0.0, form=DELTA),
        video.Keyframe(time=10.0, form=DELTA),
    ]
    tl = video.Timeline(kfs)
    a, b, p = tl.bracket(2.5)
    assert a is kfs[0] and b is kfs[1]
    assert p == pytest.approx(0.25)


def test_resolve_form_caches_by_spec():
    a = video.resolve_form(DELTA)
    b = video.resolve_form(dict(DELTA))  # equal but distinct dict object
    assert a is b  # same cache entry
    c = video.resolve_form(E4)
    assert c is not a


def test_render_frame_shape_and_range():
    tl = video.Timeline([video.Keyframe(time=0.0, form=DELTA, region="disk", scale=1.02)])
    rgb = video.render_frame(tl, 0.0, width=40, height=30)
    assert rgb.shape == (30, 40, 3)
    assert np.nanmin(rgb) >= 0.0 and np.nanmax(rgb) <= 1.0


def test_render_frame_pure_zoom_no_crossfade_when_params_identical():
    # Same form/style/colormap at both keyframes, only center/scale differ:
    # rendering should use a single evaluation (no blending needed), and
    # should exactly match rendering that same view directly.
    kf_a = video.Keyframe(time=0.0, form=DELTA, region="disk", center=(0, 0), scale=1.0)
    kf_b = video.Keyframe(time=10.0, form=DELTA, region="disk", center=(0, 0), scale=0.5)
    tl = video.Timeline([kf_a, kf_b])
    assert video._same_render_params(kf_a, kf_b)

    mid = video.render_frame(tl, 5.0, width=20, height=20)
    # Reproduce the same interpolated box directly and confirm it matches.
    p = video.smootherstep(0.5)
    scale = kf_a.scale * (kf_b.scale / kf_a.scale) ** p
    box = ((-scale, scale), (-scale, scale))
    from modforms import plotting

    vals = plotting.evaluate_on_region(video.resolve_form(DELTA), region="disk", disk_box=box, shape=(20, 20))
    expected = plotting.render(vals, style="phase-contour", background=(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(mid, expected)


def test_render_frame_crossfades_between_different_forms():
    kf_a = video.Keyframe(time=0.0, form=DELTA, region="disk", scale=1.0, background=(0.0, 0.0, 0.0))
    kf_b = video.Keyframe(time=10.0, form=E4, region="disk", scale=1.0, background=(1.0, 1.0, 1.0))
    tl = video.Timeline([kf_a, kf_b])

    start = video.render_frame(tl, 0.0, width=10, height=10)
    end = video.render_frame(tl, 10.0, width=10, height=10)
    mid = video.render_frame(tl, 5.0, width=10, height=10)

    # Midpoint should differ from both endpoints (it's a genuine blend).
    assert not np.allclose(mid, start)
    assert not np.allclose(mid, end)


def test_check_safety_flags_deep_halfplane_zoom():
    kf = video.Keyframe(time=0.0, form=DELTA, region="halfplane", center=(0.0, 1e-8), scale=1e-8)
    tl = video.Timeline([kf])
    warnings = tl.check_safety()
    assert len(warnings) == 1
    assert "halfplane" in warnings[0]


def test_check_safety_flags_deep_disk_zoom():
    kf = video.Keyframe(time=0.0, form=DELTA, region="disk", center=(0.0, 0.9), scale=1e-6)
    tl = video.Timeline([kf])
    warnings = tl.check_safety()
    assert len(warnings) == 1
    assert "disk" in warnings[0]


def test_check_safety_silent_for_safe_keyframes():
    kf = video.Keyframe(time=0.0, form=DELTA, region="disk", center=(0.0, 0.0), scale=1.0)
    tl = video.Timeline([kf])
    assert tl.check_safety() == []


def test_builtin_and_custom_cmap_helpers():
    assert video.builtin_cmap("viridis") == {"type": "builtin", "name": "viridis"}
    c = video.custom_cmap(["#000000", "#ffffff"], cyclic=False)
    assert c == {"type": "custom", "colors": ["#000000", "#ffffff"], "cyclic": False}


def test_estimate_render_time_returns_plausible_numbers():
    tl = video.Timeline(
        [video.Keyframe(time=0.0, form=DELTA, region="disk"), video.Keyframe(time=2.0, form=DELTA, region="disk")]
    )
    n_frames, per_frame, total = video.estimate_render_time(tl, fps=10, width=20, height=20, sample_frames=2)
    assert n_frames == 20
    assert per_frame > 0
    assert total == pytest.approx(n_frames * per_frame)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_render_video_end_to_end(tmp_path):
    kf_a = video.Keyframe(time=0.0, form=DELTA, region="disk", scale=1.02)
    kf_b = video.Keyframe(time=0.5, form=DELTA, region="disk", scale=0.5)
    tl = video.Timeline([kf_a, kf_b])

    out = tmp_path / "out.mp4"
    video.render_video(tl, str(out), fps=4, width=32, height=32, progress=False)

    assert out.exists() and out.stat().st_size > 0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "32,32" in probe.stdout
    assert "h264" in probe.stdout
