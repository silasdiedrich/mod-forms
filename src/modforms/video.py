"""Ambient "infinite zoom" video generation, in the style of fractal zoom
videos (continuous exponential zoom, soothing crossfades between forms and
colormaps), built from the same evaluate/render pipeline as still images.

This is a scripting API, not a point-and-click UI: a real multi-minute
render is thousands of frames and can take anywhere from minutes to hours,
which doesn't fit a synchronous web request. Write a small Python script
building a :class:`Timeline` (see ``examples/video_ambient_delta_zoom.py``
for a complete one) and render it with

    python -m modforms.cli video my_timeline.py --out video.mp4

A Timeline is just a sorted list of Keyframes (time, form, region, view
center/scale, style). Between two consecutive keyframes, the view zooms
geometrically (constant perceived zoom speed, like a Mandelbrot zoom) with
an eased (smootherstep) pace; if the two keyframes differ in form, style,
or colormap, their renders are also crossfaded across the same span. A
"held" view is just two adjacent keyframes with identical parameters.

Honesty note: unlike Mandelbrot's escape-time iteration, a modular form's
truncated q-expansion doesn't generate open-ended new detail forever --
past a certain depth (governed by float64 precision and how many Fourier
terms you evaluated with) you'll see truncation artifacts rather than
finer structure. :func:`check_safety` gives a rough, non-blocking warning
when a keyframe's zoom looks too deep for its term count. This still
covers a solid zoom range (comparable to a moderate, not extreme,
Mandelbrot zoom clip) and the disk model's cusps genuinely do show
self-similar, Ford-circle-like nested structure as you approach them.
"""

import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

from . import forms, lmfdb, plotting, reproduce
from .qexpansion import QExpansion

_FORM_CACHE = {}


def _form_cache_key(form_spec):
    return tuple(sorted(form_spec.items()))


def resolve_form(form_spec):
    """Turn a form spec dict into a QExpansion, caching by spec so a
    (possibly expensive, possibly network-fetched) form is only resolved
    once per render even though every frame asks for it.

    Recognized ``source`` values: "delta", "eisenstein" (needs "weight"),
    "lmfdb" (needs "label"), "csv" (needs "path").
    """
    key = _form_cache_key(form_spec)
    if key in _FORM_CACHE:
        return _FORM_CACHE[key]

    source = form_spec["source"]
    n_terms = form_spec.get("n_terms", 400)
    if source == "delta":
        result = forms.delta(n_terms)
    elif source == "eisenstein":
        result = forms.eisenstein(form_spec["weight"], n_terms)
    elif source == "lmfdb":
        result = lmfdb.fetch_qexpansion(form_spec["label"], n_terms=n_terms)
    elif source == "csv":
        result = QExpansion.from_csv(
            form_spec["path"],
            start=form_spec.get("start", 1),
            weight=form_spec.get("weight"),
            level=form_spec.get("level"),
        )
    else:
        raise ValueError(f"unknown form source {source!r}")

    _FORM_CACHE[key] = result
    return result


def builtin_cmap(name):
    """Convenience for a Keyframe's ``style_kwargs["cmap"]``: a built-in
    matplotlib colormap by name.
    """
    return {"type": "builtin", "name": name}


def custom_cmap(colors, cyclic=True):
    """Convenience for a Keyframe's ``style_kwargs["cmap"]``: a custom
    colormap interpolated between ``colors`` (hex strings work).
    """
    return {"type": "custom", "colors": list(colors), "cyclic": cyclic}


@dataclass
class Keyframe:
    """A single point on a video's timeline.

    ``center``/``scale`` describe the view: a box centered at ``center``
    (in halfplane (x, y) or disk w-plane coordinates, depending on
    ``region``) with half-height ``scale`` (half-width follows from the
    output video's aspect ratio, so frames are always full-bleed, no
    letterboxing).
    """

    time: float
    form: dict
    region: str = "disk"
    center: tuple = (0.0, 0.0)
    scale: float = 1.0
    style: str = "phase-contour"
    style_kwargs: dict = field(default_factory=dict)
    background: tuple = (0.0, 0.0, 0.0)


def smootherstep(p):
    """Ken Perlin's improved smoothstep: zero first *and* second
    derivative at both ends, so eased motion never has a velocity or
    acceleration discontinuity at a keyframe -- the "soothing" part.
    """
    p = min(1.0, max(0.0, p))
    return p * p * p * (p * (p * 6 - 15) + 10)


class Timeline:
    def __init__(self, keyframes):
        self.keyframes = sorted(keyframes, key=lambda k: k.time)
        if not self.keyframes:
            raise ValueError("a Timeline needs at least one Keyframe")

    @property
    def duration(self):
        return self.keyframes[-1].time

    def bracket(self, t):
        """The two keyframes surrounding time ``t``, and the raw (unedited)
        progress in [0, 1] between them. Clamps to the first/last keyframe
        outside the timeline's range.
        """
        kfs = self.keyframes
        if t <= kfs[0].time:
            return kfs[0], kfs[0], 0.0
        if t >= kfs[-1].time:
            return kfs[-1], kfs[-1], 0.0
        for a, b in zip(kfs, kfs[1:]):
            if a.time <= t <= b.time:
                span = b.time - a.time
                raw_p = 0.0 if span <= 0 else (t - a.time) / span
                return a, b, raw_p
        return kfs[-1], kfs[-1], 0.0  # pragma: no cover - unreachable given sorted keyframes

    def check_safety(self):
        """Rough, non-blocking warnings about keyframes whose zoom looks
        too deep for their term count to render cleanly (see module
        docstring). Returns a list of warning strings (empty if none).
        """
        warnings = []
        for kf in self.keyframes:
            n_terms = kf.form.get("n_terms", 400)
            if kf.region == "halfplane":
                y_min = kf.center[1] - kf.scale
                safe_y = 6 / (2 * np.pi * n_terms)
                if y_min < safe_y:
                    warnings.append(
                        f"t={kf.time}s: halfplane keyframe approaches y={y_min:.2e}, below the "
                        f"~{safe_y:.2e} floor for n_terms={n_terms} -- expect truncation "
                        f"artifacts. Increase n_terms or don't zoom this deep."
                    )
            elif kf.scale < 1e-4:
                warnings.append(
                    f"t={kf.time}s: disk keyframe scale={kf.scale:.1e} is a very deep zoom -- "
                    f"float64 precision and q-expansion truncation will likely show artifacts "
                    f"below roughly 1e-4 to 1e-5."
                )
        return warnings


def _same_render_params(a, b):
    return (
        _form_cache_key(a.form) == _form_cache_key(b.form)
        and a.region == b.region
        and a.style == b.style
        and a.style_kwargs == b.style_kwargs
        and tuple(a.background) == tuple(b.background)
    )


def _render_keyframe_content(kf, box, shape):
    form = resolve_form(kf.form)
    style_kwargs = reproduce.style_kwargs_from_metadata({"style_kwargs": kf.style_kwargs})
    if kf.region == "disk":
        vals = plotting.evaluate_on_region(form, region="disk", disk_box=box, shape=shape)
    else:
        vals = plotting.evaluate_on_region(form, region="halfplane", box=box, shape=shape)
    return plotting.render(vals, style=kf.style, background=kf.background, **style_kwargs)


def render_frame(timeline, t, width, height):
    """Render a single (height, width, 3) RGB frame at time ``t``."""
    aspect = width / height
    a, b, raw_p = timeline.bracket(t)
    p = smootherstep(raw_p)

    cx = a.center[0] + (b.center[0] - a.center[0]) * p
    cy = a.center[1] + (b.center[1] - a.center[1]) * p
    scale = a.scale * (b.scale / a.scale) ** p if a.scale != b.scale else a.scale

    half_h = scale
    half_w = scale * aspect
    box = ((cx - half_w, cx + half_w), (cy - half_h, cy + half_h))

    rgb_a = _render_keyframe_content(a, box, (height, width))
    if raw_p <= 0.0 or _same_render_params(a, b):
        return rgb_a
    rgb_b = _render_keyframe_content(b, box, (height, width))
    return rgb_a * (1 - p) + rgb_b * p


def estimate_render_time(timeline, fps=30, width=1280, height=720, sample_frames=3):
    """Render a few sample frames to estimate total time for the full
    video. Returns (n_frames, seconds_per_frame, estimated_total_seconds).
    """
    n_frames = max(1, round(timeline.duration * fps))
    # Warm the form cache first (resolving a form, e.g. an LMFDB fetch, is
    # a one-time cost that would otherwise skew a small sample).
    for kf in timeline.keyframes:
        resolve_form(kf.form)
    sample_times = np.linspace(0, timeline.duration, min(sample_frames, n_frames))
    t0 = time.time()
    for t in sample_times:
        render_frame(timeline, float(t), width, height)
    elapsed = time.time() - t0
    per_frame = elapsed / len(sample_times)
    return n_frames, per_frame, n_frames * per_frame


def render_video(
    timeline, out_path, fps=30, width=1280, height=720, crf=18, preset="medium", progress=True, progress_callback=None
):
    """Render ``timeline`` to an H.264 MP4 at ``out_path`` by piping raw
    RGB24 frames into ffmpeg (requires ``ffmpeg`` on PATH).

    ``progress_callback``, if given, is called after every frame as
    ``progress_callback(frames_done, total_frames)`` -- e.g. to drive a
    ``st.progress`` bar in a UI, independent of the ``progress=True``
    textual output (meant for a terminal).
    """
    n_frames = max(1, round(timeline.duration * fps))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found on PATH -- install it (e.g. `apt install ffmpeg`, "
            "`brew install ffmpeg`, or on Windows via winget/choco or ffmpeg.org) to render video."
        ) from e

    t0 = time.time()
    for i in range(n_frames):
        rgb = render_frame(timeline, i / fps, width, height)
        arr = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        arr = np.flipud(arr)  # increasing y upward, matching the paper/PNG convention
        proc.stdin.write(arr.tobytes())
        if progress_callback is not None:
            progress_callback(i + 1, n_frames)
        if progress and (i % max(1, n_frames // 200) == 0 or i == n_frames - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (n_frames - i - 1) / rate / 60 if rate > 0 else float("inf")
            print(f"\rframe {i + 1}/{n_frames}  ({rate:.2f} fps, ETA {eta_min:.1f} min)", end="", flush=True)
    if progress:
        print()

    proc.stdin.close()
    stderr = proc.stderr.read().decode(errors="replace")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed (exit {ret}):\n{stderr[-4000:]}")
    return out_path
