import numpy as np

from modforms.plotting import natural_shape, pad_to_aspect, padded_shape


def test_natural_shape_preserves_wide_box_ratio():
    # f_105 box is twice as wide as it is tall.
    rows, cols = natural_shape(((-1, 1), (0, 1)), max_dim=600)
    assert cols == 600
    assert rows == 300


def test_natural_shape_preserves_tall_box_ratio():
    rows, cols = natural_shape(((0, 1), (-2, 2)), max_dim=600)
    assert rows == 600
    assert cols == 150


def test_natural_shape_square_box():
    rows, cols = natural_shape(((-1, 1), (0, 2)), max_dim=500)
    assert (rows, cols) == (500, 500)


def test_padded_shape_widens_when_target_is_wider():
    rows, cols = padded_shape((512, 512), 16 / 9)
    assert rows == 512
    assert cols == round(512 * 16 / 9)


def test_padded_shape_heightens_when_target_is_taller():
    rows, cols = padded_shape((512, 512), 9 / 16)
    assert cols == 512
    assert rows == round(512 * 16 / 9)


def test_padded_shape_noop_when_already_matching():
    assert padded_shape((300, 600), 2.0) == (300, 600)


def test_pad_to_aspect_centers_content_with_background():
    content = np.zeros((10, 10, 3))
    content[..., :] = (0.0, 0.0, 0.0)  # all black content
    padded = pad_to_aspect(content, target_ratio=2.0, background=(1.0, 1.0, 1.0))
    assert padded.shape == (10, 20, 3)
    # centered: columns 5..15 are the original content (black), rest is background (white)
    np.testing.assert_array_equal(padded[:, :5], 1.0)
    np.testing.assert_array_equal(padded[:, 5:15], 0.0)
    np.testing.assert_array_equal(padded[:, 15:], 1.0)


def test_pad_to_aspect_noop_when_ratio_matches():
    content = np.random.default_rng(0).random((10, 10, 3))
    padded = pad_to_aspect(content, target_ratio=1.0)
    assert padded is content
