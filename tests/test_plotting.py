import io

import numpy as np
from PIL import Image

from modforms.plotting import natural_shape, pad_to_aspect, padded_shape, rgb_to_png_bytes, save_png


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


def test_rgb_to_png_bytes_orientation_matches_increasing_y_upward():
    # grid.py's convention: array row 0 = smallest y (bottom of the math
    # region). The saved PNG must show that at the BOTTOM of the image
    # (increasing y upward, per the paper), i.e. the *last* image row.
    rgb = np.zeros((5, 5, 3))
    rgb[0] = (1.0, 1.0, 1.0)  # row 0 = y_min -> should end up white at the bottom
    png_bytes = rgb_to_png_bytes(rgb)
    arr = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    assert arr[-1].mean() > 200  # bottom image row is white
    assert arr[0].mean() < 10  # top image row is black


def test_rgb_to_png_bytes_embeds_metadata():
    rgb = np.zeros((4, 4, 3))
    png_bytes = rgb_to_png_bytes(rgb, metadata={"Description": "hello", "modforms_reproduce": "{}"})
    img = Image.open(io.BytesIO(png_bytes))
    assert img.text["Description"] == "hello"
    assert img.text["modforms_reproduce"] == "{}"


def test_rgb_to_png_bytes_no_metadata_by_default():
    rgb = np.zeros((4, 4, 3))
    png_bytes = rgb_to_png_bytes(rgb)
    img = Image.open(io.BytesIO(png_bytes))
    assert getattr(img, "text", {}) == {}


def test_save_png_writes_readable_file_with_metadata(tmp_path):
    rgb = np.ones((3, 3, 3)) * 0.5
    path = tmp_path / "out.png"
    save_png(rgb, str(path), metadata={"Description": "test"})
    img = Image.open(path)
    assert img.size == (3, 3)
    assert img.text["Description"] == "test"
