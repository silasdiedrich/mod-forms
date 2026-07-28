"""
modforms: tools for visualizing modular forms.

Implements the visualization techniques surveyed in

    David Lowry-Duda, "Visualizing Modular Forms", arXiv:2002.05234.

including standard domain coloring, magnitude-only plots, periodic
magnitude-to-hue plots, pure phase plots, phase plots with magnitude
contours, and matplotlib-colormap based variants of all of the above.
"""

from .qexpansion import QExpansion
from .grid import halfplane_grid, disk_grid, phi
from .plotting import evaluate_on_region, render, save_png, plot_form
from . import lmfdb

__all__ = [
    "QExpansion",
    "halfplane_grid",
    "disk_grid",
    "phi",
    "evaluate_on_region",
    "render",
    "save_png",
    "plot_form",
    "lmfdb",
]

__version__ = "0.1.0"
