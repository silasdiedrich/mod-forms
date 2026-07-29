"""Example ambient zoom video: a continuous zoom into the Ramanujan Delta
function's cusp structure on the Poincare disk, crossfading through
colormaps as it goes, then switching forms (Delta -> E4) partway through
while still zooming.

Quick low-res preview:

    python -m modforms.cli video examples/video_ambient_delta_zoom.py \\
        --out preview.mp4 --fps 24 --width 640 --height 360

Check timing before committing to a full render:

    python -m modforms.cli video examples/video_ambient_delta_zoom.py \\
        --out video.mp4 --dry-run

Full render, e.g. 1080p:

    python -m modforms.cli video examples/video_ambient_delta_zoom.py \\
        --out video.mp4 --width 1920 --height 1080 --fps 30

To make a genuinely long (YouTube-length) video, add more keyframes
further out in time -- the zoom speed between two keyframes is set by how
much `scale` shrinks over that time span, so spreading the same zoom
across more seconds slows it down; add more keyframes to keep going
deeper (mind modforms.video's safety warnings -- see the module
docstring for why zoom depth isn't literally infinite the way a
Mandelbrot escape-time zoom is).

Requires ffmpeg on PATH.
"""

from modforms.video import Keyframe, builtin_cmap

DELTA = {"source": "delta", "n_terms": 400}
E4 = {"source": "eisenstein", "weight": 4, "n_terms": 400}

TIMELINE = [
    # Hold the full disk briefly so viewers get oriented before the zoom starts.
    Keyframe(
        time=0.0,
        form=DELTA,
        region="disk",
        center=(0.0, 0.0),
        scale=1.02,
        style="colormap-phase-contour",
        style_kwargs={"base": 2.0, "cmap": builtin_cmap("cividis")},
    ),
    Keyframe(
        time=4.0,
        form=DELTA,
        region="disk",
        center=(0.0, 0.0),
        scale=1.02,
        style="colormap-phase-contour",
        style_kwargs={"base": 2.0, "cmap": builtin_cmap("cividis")},
    ),
    # Zoom toward a cusp, crossfading cividis -> twilight along the way.
    Keyframe(
        time=16.0,
        form=DELTA,
        region="disk",
        center=(0.05, 0.85),
        scale=0.05,
        style="colormap-phase-contour",
        style_kwargs={"base": 2.0, "cmap": builtin_cmap("twilight")},
    ),
    # Keep zooming, twilight -> viridis.
    Keyframe(
        time=28.0,
        form=DELTA,
        region="disk",
        center=(0.07, 0.88),
        scale=0.006,
        style="colormap-phase-contour",
        style_kwargs={"base": 2.0, "cmap": builtin_cmap("viridis")},
    ),
    # Switch forms (Delta -> E4), viridis -> plasma. Note this jumps to a
    # *wider* view, not a continued deep zoom: E4 is an Eisenstein series,
    # not a cuspform, so it approaches a nonzero constant near cusps (flat,
    # boring) rather than vanishing like Delta does -- the rich structure
    # near (0.3, 0.3) here comes from E4's actual zeros (visible as the
    # spiral "eyes" from phase winding around each one), not from a cusp.
    Keyframe(
        time=40.0,
        form=E4,
        region="disk",
        center=(0.3, 0.3),
        scale=0.3,
        style="colormap-phase-contour",
        style_kwargs={"base": 2.0, "cmap": builtin_cmap("plasma")},
    ),
]
