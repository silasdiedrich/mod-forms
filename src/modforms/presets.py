"""Example regions used in the paper (Section 2.2), for the running example
of Delta. Adjust similarly for your own forms based on their level/weight.
"""

DELTA_BOX = ((-1, 1), (0, 2))
DELTA_ZOOM = ((0.1, 0.4), (0, 0.25))

# The paper's other three examples (g on level 5, f_105 on level 105, f_10
# on level 10) use these boxes; plug in their q-expansions (see forms.py)
# to reproduce the corresponding figures.
G_BOX = ((-2.5, 2.5), (0, 2))
F105_BOX = ((-1, 1), (0, 1))
F10_BOX = ((-1, 1), (0, 2))
ZOOM = ((0.1, 0.4), (0, 0.25))
