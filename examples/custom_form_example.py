"""How to plot a modular form that isn't built in, using Fourier
coefficients you supply yourself (e.g. downloaded from the LMFDB).

The paper's other three running examples are LMFDB newforms:

    g      -- weight 4,  level 5,   label 5.4.a.a.1.1
    f_105  -- weight 2,  level 105, label 105.2.a.a.1.1
    f_10   -- weight 20, level 10,  label 10.20.a.a.1.1

To reproduce those figures: open the LMFDB page for a label (e.g.
https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/5/4/a/a/1/1/), copy its
q-expansion coefficients into a one-per-line CSV file, and either load it
with QExpansion.from_csv(...) or paste the values into a Python list and
use QExpansion.from_list(...), as sketched below.
"""

from modforms import QExpansion, plotting

# Toy example: coefficients don't need to come from an actual modular form
# to exercise the plotting code -- here's a short truncated q-expansion.
coeffs = [1, -4, 2, 8, -5, -8, 0, -16, 6, 20]
f = QExpansion.from_list(coeffs, start=1, label="toy example")

rgb = plotting.plot_form(
    f,
    region="halfplane",
    box=((-1, 1), (0, 1)),
    shape=(400, 400),
    style="colormap-phase-contour",
    cmap="cividis",
    out="output/custom_form_example.png",
)
print("wrote output/custom_form_example.png")

# Loading from a CSV of coefficients, one per line, in order starting at q^1:
#
#   f = QExpansion.from_csv("data/my_form.csv", start=1, weight=4, level=5)
