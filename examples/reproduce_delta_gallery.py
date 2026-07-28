"""Reproduce, for the Ramanujan Delta function, the six base visualization
styles surveyed in Section 2 of the paper plus a few of the matplotlib
colormap variants from Section 3, on both the disk and the halfplane.

Run with:

    python examples/reproduce_delta_gallery.py

Writes PNGs into ./output/.
"""

import os

from modforms import forms, plotting, presets

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    delta = forms.delta(400)

    jobs = [
        # (name, region, box, style, style_kwargs)
        ("01_standard_disk", "disk", None, "standard", {}),
        ("01_standard_halfplane", "halfplane", presets.DELTA_BOX, "standard", {}),
        ("02_magnitude_disk", "disk", None, "magnitude", {"alpha": 0.25}),
        ("03_periodic_linear_disk", "disk", None, "periodic-linear", {}),
        ("04_periodic_log_disk", "disk", None, "periodic-log", {"base": 7.0}),
        ("05_phase_disk", "disk", None, "phase", {}),
        ("06_phase_contour_disk", "disk", None, "phase-contour", {"base": 2.0}),
        ("07_colormap_phase_twilight", "disk", None, "colormap-phase", {"cmap": "twilight"}),
        ("07_colormap_phase_viridis", "disk", None, "colormap-phase", {"cmap": "viridis"}),
        ("07_colormap_phase_cividis", "disk", None, "colormap-phase", {"cmap": "cividis"}),
        ("08_colormap_phase_contour_cividis", "disk", None, "colormap-phase-contour", {"cmap": "cividis"}),
        ("09_colormap_standard_cividis", "disk", None, "colormap-standard", {"cmap": "cividis"}),
        ("06_phase_contour_zoom", "halfplane", presets.DELTA_ZOOM, "phase-contour", {"base": 2.0}),
    ]

    for name, region, box, style, style_kwargs in jobs:
        call_kwargs = dict(region=region, style=style, shape=(500, 500), out=os.path.join(OUT_DIR, f"{name}.png"))
        if box is not None:
            call_kwargs["box"] = box
        plotting.plot_form(delta, **call_kwargs, **style_kwargs)

    print(f"wrote {len(jobs)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
