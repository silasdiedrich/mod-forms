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
    assert any(s.label == "Colormap" for s in at.sidebar.selectbox)

    render_button = [b for b in at.sidebar.button if "Render" in b.label][0]
    render_button.click().run(timeout=60)
    assert not at.exception
    assert len(at.main.image) == 1
    assert len(at.main.download_button) == 1


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
