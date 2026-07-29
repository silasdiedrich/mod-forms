"""Smoke tests for the Streamlit UI (app.py), using Streamlit's headless
AppTest harness. Skipped entirely if streamlit isn't installed, since it's
an optional extra (``pip install -e ".[ui]"``).
"""

import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_app_loads_without_exception():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_render_button_produces_an_image():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    assert len(at.main.image) == 1


def test_colormap_style_reveals_extra_controls_and_renders():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    style_select = [s for s in at.sidebar.selectbox if s.label == "Visualization style"][0]
    style_select.set_value("colormap-phase-contour").run(timeout=30)
    assert not at.exception
    assert any(r.label == "Colormap" for r in at.sidebar.radio)
    assert any(s.label == "Choose colormap" for s in at.sidebar.selectbox)

    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    assert len(at.main.image) == 1
    assert len(at.main.download_button) == 1


def test_custom_colormap_shows_color_pickers_and_renders():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    style_select = [s for s in at.sidebar.selectbox if s.label == "Visualization style"][0]
    style_select.set_value("colormap-phase").run(timeout=30)

    cmap_source_radio = [r for r in at.sidebar.radio if r.label == "Colormap"][0]
    cmap_source_radio.set_value("Custom").run(timeout=30)
    assert not at.exception
    assert len(at.sidebar.color_picker) >= 2

    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    assert len(at.main.image) == 1


def test_aspect_ratio_pads_output_to_target_ratio():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    aspect_select = [s for s in at.sidebar.selectbox if s.label == "Output aspect ratio"][0]
    aspect_select.set_value("16:9").run(timeout=30)

    res_slider = [s for s in at.sidebar.slider if "Detail" in s.label][0]
    res_slider.set_value(256).run(timeout=30)

    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception

    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(at.session_state["last_png"]))
    width, height = img.size
    assert abs(width / height - 16 / 9) < 0.02


def test_csv_upload_renders():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    source_radio = at.sidebar.radio[0]
    source_radio.set_value("Upload CSV").run(timeout=30)

    uploader = at.sidebar.file_uploader[0]
    uploader.upload("coeffs.csv", b"1\n-4\n2\n8\n-5\n", "text/csv").run(timeout=30)
    assert not at.exception

    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    assert len(at.main.image) == 1


def test_lmfdb_network_failure_shown_as_error_not_crash():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    source_radio = at.sidebar.radio[0]
    source_radio.set_value("LMFDB").run(timeout=30)

    load_button = [b for b in at.sidebar.button if b.label == "Load form"][0]
    load_button.click().run(timeout=30)
    assert not at.exception
    assert len(at.error) == 1


def test_render_shows_latex_description_for_delta():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    res_slider = [s for s in at.sidebar.slider if "Detail" in s.label][0]
    res_slider.set_value(200).run(timeout=30)
    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    assert len(at.main.latex) == 2
    form_latex, domain_latex = (l.value for l in at.main.latex)
    assert r"\Delta(z)" in form_latex
    assert r"\tau(n)" in form_latex
    assert "k = 12" in form_latex
    assert "N = 1" in form_latex
    assert r"\mathbb{D}" in domain_latex  # disk is the default region


def test_render_shows_latex_description_for_eisenstein():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    form_select = [s for s in at.sidebar.selectbox if s.label == "Form"][0]
    form_select.set_value("E4 — Eisenstein, weight 4").run(timeout=30)
    res_slider = [s for s in at.sidebar.slider if "Detail" in s.label][0]
    res_slider.set_value(200).run(timeout=30)
    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    form_latex = at.main.latex[0].value
    assert r"E_{4}(z)" in form_latex
    assert "N = 1" in form_latex


def test_csv_upload_weight_level_appear_in_latex():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    source_radio = at.sidebar.radio[0]
    source_radio.set_value("Upload CSV").run(timeout=30)
    uploader = at.sidebar.file_uploader[0]
    uploader.upload("coeffs.csv", b"1\n-4\n2\n8\n-5\n", "text/csv").run(timeout=30)

    weight_input = [n for n in at.sidebar.number_input if n.label == "Weight (optional)"][0]
    weight_input.set_value(6).run(timeout=30)

    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    form_latex = at.main.latex[0].value
    assert "k = 6" in form_latex


def test_settings_caption_reflects_style_and_background():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    bg_radio = [r for r in at.sidebar.radio if "Outside the disk" in r.label][0]
    bg_radio.set_value("Black").run(timeout=30)
    res_slider = [s for s in at.sidebar.slider if "Detail" in s.label][0]
    res_slider.set_value(200).run(timeout=30)
    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    caption = at.session_state["last_settings_caption"]
    assert "background: black" in caption
    assert "200×200" in caption


def test_downloaded_png_embeds_reproduce_metadata():
    import io
    import json

    from PIL import Image

    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    res_slider = [s for s in at.sidebar.slider if "Detail" in s.label][0]
    res_slider.set_value(150).run(timeout=30)
    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception

    img = Image.open(io.BytesIO(at.session_state["last_png"]))
    assert "modforms_reproduce" in img.text
    meta = json.loads(img.text["modforms_reproduce"])
    assert meta["form"]["label"] is not None
    assert len(meta["form"]["coeffs"]) == 400
    assert meta["content_shape"] == [150, 150]

    # Filename is descriptive, not the old generic "modform.png".
    assert at.session_state["last_filename"].endswith(".png")
    assert at.session_state["last_filename"] != "modform.png"
