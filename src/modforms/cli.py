"""Command-line interface.

    python -m modforms.cli plot --form delta --region disk --style phase-contour --out delta.png
    python -m modforms.cli plot --lmfdb 5.4.a.a --region disk --out g.png
    python -m modforms.cli search --level 105 --weight 2
"""

import argparse

from . import forms, lmfdb
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

    return p


def _load_form(args):
    if args.lmfdb:
        return lmfdb.fetch_qexpansion(args.lmfdb, n_terms=args.terms)
    if args.csv:
        return QExpansion.from_csv(args.csv, start=args.csv_start)
    return BUILTIN_FORMS[args.form](args.terms)


def _run_plot(args):
    form = _load_form(args)
    style_kwargs = {p: getattr(args, p) for p in STYLE_PARAMS.get(args.style, [])}
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


def _run_search(args):
    lmfdb.print_newforms(level=args.level, weight=args.weight, limit=args.limit)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plot":
        _run_plot(args)
    elif args.command == "search":
        _run_search(args)


if __name__ == "__main__":
    main()
