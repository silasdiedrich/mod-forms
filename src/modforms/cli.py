"""Command-line interface.

    python -m modforms.cli plot --form delta --region disk --style phase-contour --out delta.png
    python -m modforms.cli plot --lmfdb 5.4.a.a --region disk --out g.png
    python -m modforms.cli search --level 105 --weight 2
    python -m modforms.cli replot delta.png --out delta_again.png
    python -m modforms.cli video my_timeline.py --out video.mp4

Every PNG written by ``plot`` (and by the Streamlit app) embeds a
``modforms_reproduce`` metadata chunk describing exactly how to
reproduce it (form, region, style, resolution, background, ...);
``replot`` reads that back and re-renders from scratch, no need to
remember or re-type any of the original settings.

``video`` renders an ambient zoom video (see modforms.video and
examples/video_ambient_delta_zoom.py) from a Python script defining a
TIMELINE (a list of Keyframe, or a Timeline) or a build_timeline()
function.
"""

import argparse
import importlib.util
import json
import os

import numpy as np

from . import forms, lmfdb, plotting, reproduce, video
from .coloring import STYLES, STYLE_PARAMS
from .plotting import plot_form
from .qexpansion import QExpansion

BUILTIN_FORMS = {
    "delta": lambda n: forms.delta(n),
    "e4": lambda n: forms.eisenstein(4, n),
    "e6": lambda n: forms.eisenstein(6, n),
    "e8": lambda n: forms.eisenstein(8, n),
    "e10": lambda n: forms.eisenstein(10, n),
    "e14": lambda n: forms.eisenstein(14, n),
}


def _parse_box(s):
    x0, x1, y0, y1 = (float(v) for v in s.split(","))
    return ((x0, x1), (y0, y1))


def _add_plot_args(p):
    source = p.add_mutually_exclusive_group()
    source.add_argument("--form", choices=sorted(BUILTIN_FORMS), default="delta", help="Built-in form to plot.")
    source.add_argument("--csv", help="Load Fourier coefficients from a CSV file.")
    source.add_argument(
        "--lmfdb",
        metavar="LABEL",
        help="Fetch a newform's q-expansion from the LMFDB, e.g. '5.4.a.a' or '5.4.a.a.1.1'.",
    )
    p.add_argument("--csv-start", type=int, default=1, help="Exponent of q for the first coefficient in --csv.")
    p.add_argument("--terms", type=int, default=400, help="Number of Fourier coefficients to use.")
    p.add_argument("--region", choices=["halfplane", "disk"], default="halfplane")
    p.add_argument("--box", type=_parse_box, default=((-1, 1), (0, 2)), help="x0,x1,y0,y1 for --region halfplane.")
    p.add_argument("--shape", type=int, nargs=2, default=(600, 600), metavar=("ROWS", "COLS"))
    p.add_argument("--style", choices=sorted(STYLES), default="phase-contour")
    p.add_argument("--cmap", default="cividis", help="Matplotlib colormap, for colormap-* styles.")
    p.add_argument("--alpha", type=float, default=0.25, help="Exponent for the 'magnitude' style.")
    p.add_argument("--base", type=float, default=2.0, help="Base for logarithmic styles / contours.")
    p.add_argument("--offset", type=float, default=0.0, help="Hue offset in [0, 1) for phase-based styles.")
    p.add_argument("--out", required=True, help="Output PNG path.")


def build_parser():
    p = argparse.ArgumentParser(description="Plot and browse modular forms using the techniques of arXiv:2002.05234.")
    sub = p.add_subparsers(dest="command", required=True)

    plot_p = sub.add_parser("plot", help="Render a modular form to a PNG.")
    _add_plot_args(plot_p)

    search_p = sub.add_parser("search", help="Browse newforms on the LMFDB.")
    search_p.add_argument("--level", type=int)
    search_p.add_argument("--weight", type=int)
    search_p.add_argument("--limit", type=int, default=20)

    replot_p = sub.add_parser(
        "replot", help="Re-render a PNG from its embedded modforms_reproduce metadata."
    )
    replot_p.add_argument("input", help="A PNG previously written by `plot` or the UI.")
    replot_p.add_argument("--out", required=True, help="Output PNG path.")
    replot_p.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        help="Override the content resolution (defaults to what's stored in the metadata).",
    )

    inspect_p = sub.add_parser(
        "inspect",
        help="Print a PNG's embedded settings in a human-readable form, e.g. to re-enter them in the UI by hand.",
    )
    inspect_p.add_argument("input", help="A PNG previously written by `plot` or the UI.")

    video_p = sub.add_parser(
        "video", help="Render an ambient zoom video from a Python timeline script (requires ffmpeg)."
    )
    video_p.add_argument(
        "timeline", help="Python file defining TIMELINE (a list of Keyframe, or a Timeline) or build_timeline()."
    )
    video_p.add_argument("--out", required=True, help="Output MP4 path.")
    video_p.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Quick low-res/low-fps draft (480x270 @ 15fps, fastest x264 preset) to check the "
            "composition before committing to a full render. --fps/--width/--height/--preset "
            "still override individually if given."
        ),
    )
    video_p.add_argument("--fps", type=int, default=None, help="Default: 30, or 15 with --preview.")
    video_p.add_argument("--width", type=int, default=None, help="Default: 1280, or 480 with --preview.")
    video_p.add_argument("--height", type=int, default=None, help="Default: 720, or 270 with --preview.")
    video_p.add_argument("--crf", type=int, default=18, help="x264 quality; lower = higher quality/bigger file.")
    video_p.add_argument(
        "--preset", default=None, help="x264 encoding speed/efficiency preset. Default: medium, or ultrafast with --preview."
    )
    video_p.add_argument(
        "--dry-run", action="store_true", help="Print the frame count and a time estimate, then exit."
    )
    video_p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Frames render in parallel across this many processes (default: all CPU cores; pass 1 for sequential).",
    )
    video_p.add_argument(
        "--precision",
        choices=["double", "single"],
        default="double",
        help=(
            "'single' (complex64) is roughly 3x faster but has a much shallower safe zoom depth "
            "than the default 'double' (complex128) -- see Timeline.check_safety."
        ),
    )

    return p


def _load_form(args):
    if args.lmfdb:
        return lmfdb.fetch_qexpansion(args.lmfdb, n_terms=args.terms)
    if args.csv:
        return QExpansion.from_csv(args.csv, start=args.csv_start)
    return BUILTIN_FORMS[args.form](args.terms)


def _source_kind(args):
    if args.lmfdb:
        return "lmfdb"
    if args.csv:
        return "csv"
    return "delta" if args.form == "delta" else "eisenstein"


def _run_plot(args):
    form = _load_form(args)
    style_kwargs = {p: getattr(args, p) for p in STYLE_PARAMS.get(args.style, [])}
    box = args.box if args.region == "halfplane" else None
    background = (1.0, 1.0, 1.0)

    metadata = reproduce.build_metadata(
        form,
        _source_kind(args),
        region=args.region,
        box=box,
        disk_extent=1.02,
        style=args.style,
        style_kwargs_json=reproduce.style_kwargs_to_json(style_kwargs, style_kwargs.get("cmap")),
        content_shape=tuple(args.shape),
        output_shape=tuple(args.shape),
        target_ratio=None,
        background=background,
    )

    plot_form(
        form,
        region=args.region,
        box=args.box,
        shape=tuple(args.shape),
        style=args.style,
        out=args.out,
        metadata=metadata,
        background=background,
        **style_kwargs,
    )
    print(f"wrote {args.out}")


def _run_search(args):
    lmfdb.print_newforms(level=args.level, weight=args.weight, limit=args.limit)


def _run_replot(args):
    meta = reproduce.load_metadata(args.input)
    form = reproduce.form_from_metadata(meta)
    style_kwargs = reproduce.style_kwargs_from_metadata(meta)

    shape = tuple(args.shape) if args.shape else tuple(meta["content_shape"])
    box = (tuple(meta["box"][0]), tuple(meta["box"][1])) if meta.get("box") else ((-1, 1), (0, 2))
    disk_extent = meta.get("disk_extent") or 1.02
    background = tuple(meta["background"])

    vals = plotting.evaluate_on_region(form, region=meta["region"], box=box, shape=shape, disk_extent=disk_extent)
    rgb = plotting.render(vals, style=meta["style"], background=background, **style_kwargs)
    target_ratio = meta.get("target_ratio")
    if target_ratio:
        rgb = plotting.pad_to_aspect(rgb, target_ratio, background=background)

    # Re-embed the same recipe, with the shape fields updated to match this run.
    new_meta = dict(meta)
    new_meta["content_shape"] = list(shape)
    new_meta["output_shape"] = list(rgb.shape[:2])
    out_metadata = {
        "Software": reproduce.SOFTWARE_TAG,
        "Description": f"replotted from {args.input}",
        reproduce.REPRODUCE_KEY: json.dumps(new_meta),
    }
    plotting.save_png(rgb, args.out, metadata=out_metadata)
    print(f"wrote {args.out}")


def _run_inspect(args):
    """Print a PNG's embedded settings in a form that maps directly onto
    the UI's sidebar sections, for re-entering them by hand (e.g. to keep
    tweaking a shot interactively rather than using `replot`).
    """
    meta = reproduce.load_metadata(args.input)
    f = meta["form"]

    print("=== 1. Form ===")
    source = f["source"]
    if source == "delta":
        print("  Source: Built-in")
        print("  Form: Delta — weight 12, level 1")
    elif source == "eisenstein":
        k = int(f["weight"])
        print("  Source: Built-in")
        print(f"  Form: E{k} — Eisenstein, weight {k}")
    elif source == "lmfdb":
        print("  Source: LMFDB")
        print(f"  LMFDB label: {f['label']}")
    else:
        print("  Source: Upload CSV")
        print("  (the original CSV file is needed to re-upload through the UI --")
        print("   the coefficients are embedded here but not re-uploadable as a file;")
        print(f"   use `modforms.cli replot {args.input} --out new.png` instead)")
    print(f"  Terms: {len(f['coeffs'])}")
    if f.get("weight") is not None and source != "eisenstein":
        print(f"  (weight: {f['weight']})")
    if f.get("level") is not None:
        print(f"  (level: {f['level']})")

    print()
    print("=== 2. Region ===")
    if meta["region"] == "disk":
        print("  Region: Poincaré disk")
        print(f"  Disk plot extent: {meta.get('disk_extent', 1.02)}")
    else:
        print("  Region: Upper halfplane")
        print("  Preset box: Custom")
        (x0, x1), (y0, y1) = meta["box"]
        print(f"  x0={x0}  x1={x1}  y0={y0}  y1={y1}")

    print()
    print("=== 3. Style ===")
    print(f"  Visualization style: {meta['style']}")
    for key, value in meta.get("style_kwargs", {}).items():
        if key == "cmap":
            if isinstance(value, dict) and value.get("type") == "custom":
                print(f"  Colormap: Custom -- colors={value['colors']}  cyclic={value['cyclic']}")
            elif isinstance(value, dict):
                print(f"  Colormap: Built-in -- {value['name']}")
            else:
                print(f"  Colormap: Built-in -- {value}")
        elif key == "alpha":
            print(f"  Alpha: {value}")
        elif key == "base":
            print(f"  Base: {value}")
        elif key == "offset":
            print(f"  Hue offset: {value}")

    print()
    print("=== 4. Resolution & Aspect Ratio ===")
    rows, cols = meta["content_shape"]
    print(f"  Detail (long edge): {max(rows, cols)}")
    target_ratio = meta.get("target_ratio")
    if target_ratio:
        print(f"  Output aspect ratio: Custom -- Width/Height = {target_ratio:.4g} (e.g. Width={target_ratio:.4g}, Height=1)")
    else:
        print("  Output aspect ratio: Native (undistorted)")

    print()
    print("=== 5. Background ===")
    bg = tuple(meta["background"])
    hexcode = "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in bg))
    if bg == (1.0, 1.0, 1.0):
        print("  Outside the disk (or masked region): White")
    elif bg == (0.0, 0.0, 0.0):
        print("  Outside the disk (or masked region): Black")
    else:
        print(f"  Outside the disk (or masked region): Custom -- {hexcode}")


def _load_timeline(path):
    spec = importlib.util.spec_from_file_location("modforms_timeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "TIMELINE"):
        timeline = module.TIMELINE
    elif hasattr(module, "build_timeline"):
        timeline = module.build_timeline()
    else:
        raise ValueError(f"{path} must define a TIMELINE (list of Keyframe, or a Timeline) or a build_timeline()")
    if isinstance(timeline, list):
        timeline = video.Timeline(timeline)
    return timeline


_VIDEO_DEFAULTS = {"fps": 30, "width": 1280, "height": 720, "preset": "medium"}
_VIDEO_PREVIEW_DEFAULTS = {"fps": 15, "width": 480, "height": 270, "preset": "ultrafast"}


def _resolve_video_settings(args):
    defaults = _VIDEO_PREVIEW_DEFAULTS if args.preview else _VIDEO_DEFAULTS
    return {name: getattr(args, name) if getattr(args, name) is not None else default for name, default in defaults.items()}


def _run_video(args):
    timeline = _load_timeline(args.timeline)
    settings = _resolve_video_settings(args)
    dtype = np.complex64 if args.precision == "single" else None
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    for warning in timeline.check_safety(dtype=dtype):
        print(f"warning: {warning}")

    n_frames, per_frame, total_est = video.estimate_render_time(
        timeline, fps=settings["fps"], width=settings["width"], height=settings["height"], dtype=dtype
    )
    mode = "preview" if args.preview else "full"
    print(
        f"[{mode}] {settings['width']}x{settings['height']} @ {settings['fps']}fps, "
        f"{args.precision} precision, {workers} worker(s) -- "
        f"{n_frames} frames ({timeline.duration:.1f}s of video), "
        f"~{per_frame * 1000:.0f}ms/frame (single-process estimate), "
        f"estimated total: {total_est / workers / 60:.1f} min with {workers} worker(s)"
    )
    if args.dry_run:
        return

    video.render_video(
        timeline,
        args.out,
        fps=settings["fps"],
        width=settings["width"],
        height=settings["height"],
        crf=args.crf,
        preset=settings["preset"],
        dtype=dtype,
        workers=workers,
    )
    print(f"wrote {args.out}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plot":
        _run_plot(args)
    elif args.command == "search":
        _run_search(args)
    elif args.command == "replot":
        _run_replot(args)
    elif args.command == "inspect":
        _run_inspect(args)
    elif args.command == "video":
        _run_video(args)


if __name__ == "__main__":
    main()
