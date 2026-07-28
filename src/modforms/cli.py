"""Command-line interface: ``python -m modforms.cli --form delta ...``"""

import argparse

from . import forms
from .coloring import STYLES
from .plotting import plot_form

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


def build_parser():
    p = argparse.ArgumentParser(description="Plot a modular form using the techniques of arXiv:2002.05234.")
    p.add_argument("--form", choices=sorted(BUILTIN_FORMS), default="delta", help="Built-in form to plot.")
    p.add_argument("--csv", help="Load Fourier coefficients from a CSV file instead of a built-in form.")
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
    return p


_STYLE_KWARGS = {
    "magnitude": {"alpha": "alpha"},
    "periodic-linear": {"offset": "offset"},
    "periodic-log": {"base": "base", "offset": "offset"},
    "phase": {"offset": "offset"},
    "phase-contour": {"base": "base", "offset": "offset"},
    "colormap-phase": {"cmap": "cmap", "offset": "offset"},
    "colormap-phase-contour": {"cmap": "cmap", "base": "base", "offset": "offset"},
    "colormap-magnitude": {"cmap": "cmap", "base": "base", "offset": "offset"},
    "colormap-standard": {"cmap": "cmap", "offset": "offset"},
}


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.csv:
        from .qexpansion import QExpansion

        form = QExpansion.from_csv(args.csv, start=args.csv_start)
    else:
        form = BUILTIN_FORMS[args.form](args.terms)

    style_kwargs = {
        dest: getattr(args, src) for dest, src in _STYLE_KWARGS.get(args.style, {}).items()
    }

    plot_form(
        form,
        region=args.region,
        box=args.box,
        shape=tuple(args.shape),
        style=args.style,
        out=args.out,
        **style_kwargs,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
