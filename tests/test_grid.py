import numpy as np

from modforms.grid import phi, halfplane_grid, disk_grid


def test_phi_boundary_points():
    assert abs(phi(-1j) - 0) < 1e-9
    assert abs(phi(0) - 1j) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        result = phi(np.complex128(1j))
    assert np.isinf(result) or np.isnan(result)


def test_halfplane_grid_orientation():
    z = halfplane_grid(((-1, 1), (0, 2)), (5, 5))
    assert z.shape == (5, 5)
    assert z[0, 0].imag == 0
    assert z[-1, 0].imag == 2
    assert z[0, 0].real == -1
    assert z[0, -1].real == 1


def test_disk_grid_mask():
    w, mask = disk_grid((100, 100), extent=1.0)
    assert mask[50, 50]  # center is inside
    assert not mask[0, 0]  # corner is outside
