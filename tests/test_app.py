"""Smoke tests for the Streamlit UI (app.py), using Streamlit's headless
AppTest harness. Skipped entirely if streamlit isn't installed, since it's
an optional extra (``pip install -e ".[ui]"``).
"""

import os
import shutil

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def _switch_to_video_mode(at):
    at.radio[0].set_value("🎬 Video").run(timeout=30)
    return at


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


def _keyframe_expanders(at):
    return [e for e in at.main.expander if e.label.startswith("Keyframe")]


def test_video_mode_loads_with_default_keyframes():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)
    assert not at.exception
    kf_expanders = _keyframe_expanders(at)
    assert len(kf_expanders) == 2
    assert kf_expanders[0].label.startswith("Keyframe 1")
    assert kf_expanders[1].label.startswith("Keyframe 2")


def test_video_add_and_remove_keyframe():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    add_button = [b for b in at.main.button if "Add keyframe" in b.label][0]
    add_button.click().run(timeout=30)
    assert not at.exception
    assert len(_keyframe_expanders(at)) == 3

    remove_buttons = [b for b in at.main.button if "Remove this keyframe" in b.label]
    assert len(remove_buttons) == 3
    remove_buttons[0].click().run(timeout=30)
    assert not at.exception
    assert len(_keyframe_expanders(at)) == 2


def test_video_keyframe_style_change_is_scoped_to_that_keyframe():
    """A regression check for the per-keyframe widget-key scheme: changing
    keyframe 1's style must not affect keyframe 2's widgets/state.
    """
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    style_selects = [s for s in at.main.selectbox if s.label == "Visualization style"]
    assert len(style_selects) == 2
    style_selects[0].set_value("colormap-phase").run(timeout=30)
    assert not at.exception

    style_selects = [s for s in at.main.selectbox if s.label == "Visualization style"]
    assert style_selects[0].value == "colormap-phase"
    # Keyframe 2 keeps its own (different) style, untouched.
    assert style_selects[1].value != "colormap-phase" or style_selects[1] is style_selects[0]
    assert style_selects[1].value == "colormap-phase-contour"


def test_video_preview_frame_renders_image():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    preview_button = [b for b in at.main.button if "Preview this frame" in b.label][0]
    preview_button.click().run(timeout=30)
    assert not at.exception
    assert len(at.main.image) == 1


def test_video_estimate_render_time_shows_info():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    estimate_button = [b for b in at.main.button if "Estimate render time" in b.label][0]
    estimate_button.click().run(timeout=30)
    assert not at.exception
    assert len(at.main.info) == 1
    assert "frames" in at.main.info[0].value


def test_video_render_button_disabled_without_ffmpeg(monkeypatch):
    import shutil as shutil_mod

    monkeypatch.setattr(shutil_mod, "which", lambda name: None)
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)
    assert not at.exception
    assert len(at.error) == 1
    render_button = [b for b in at.main.button if "Render Video" in b.label][0]
    assert render_button.disabled


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_video_full_render_produces_playable_mp4():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    res_select = [s for s in at.main.selectbox if s.label == "Resolution preset"][0]
    res_select.set_value("Preview — 480x270 @ 15fps").run(timeout=30)

    render_button = [b for b in at.main.button if "Render Video" in b.label][0]
    render_button.click().run(timeout=120)
    assert not at.exception
    assert "video_bytes" in at.session_state
    assert len(at.session_state["video_bytes"]) > 0
    assert len(at.main.error) == 0


def test_video_load_preset_replaces_timeline():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    preset_select = [s for s in at.main.selectbox if "Golden cusp zoom (Delta)" in s.options][0]
    preset_select.set_value("Golden cusp zoom (Delta)").run(timeout=30)
    load_button = [b for b in at.main.button if b.label == "Load preset"][0]
    load_button.click().run(timeout=30)
    assert not at.exception

    kfs = at.session_state["video_keyframes"]
    assert len(kfs) == 8
    assert kfs[0]["region"] == "disk" and kfs[0]["scale"] == 1.02
    assert kfs[-1]["region"] == "halfplane" and kfs[-1]["center_x"] == 0.618
    assert len(_keyframe_expanders(at)) == 8


def test_video_json_export_then_import_round_trips():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    preset_select = [s for s in at.main.selectbox if "Golden cusp zoom (Delta)" in s.options][0]
    preset_select.set_value("Golden cusp zoom (Delta)").run(timeout=30)
    [b for b in at.main.button if b.label == "Load preset"][0].click().run(timeout=30)

    exported = [t for t in at.main.text_area if "Current timeline" in t.label][0].value

    at2 = st_testing.AppTest.from_file(APP_PATH)
    at2.run(timeout=30)
    _switch_to_video_mode(at2)
    import_area = [t for t in at2.main.text_area if "Paste timeline JSON" in t.label][0]
    import_area.set_value(exported).run(timeout=30)
    [b for b in at2.main.button if b.label == "Load from JSON"][0].click().run(timeout=30)
    assert not at2.exception

    original = at.session_state["video_keyframes"]
    roundtripped = at2.session_state["video_keyframes"]
    assert len(roundtripped) == len(original)
    for a, b in zip(original, roundtripped):
        assert {k: v for k, v in a.items() if k != "id"} == {k: v for k, v in b.items() if k != "id"}


def test_video_json_import_rejects_garbage():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    import_area = [t for t in at.main.text_area if "Paste timeline JSON" in t.label][0]
    import_area.set_value("not valid json").run(timeout=30)
    [b for b in at.main.button if b.label == "Load from JSON"][0].click().run(timeout=30)
    assert not at.exception
    # Filtered rather than an exact count: an unrelated "ffmpeg not found" error
    # can also be showing on this page depending on the test environment, and
    # isn't what this test is about.
    assert any("Couldn't load that as a timeline" in e.value for e in at.main.error)


def test_video_30min_preset_loads_with_correct_duration_and_no_safety_warnings():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    preset_select = [s for s in at.main.selectbox if "30-min journey (Delta family)" in s.options][0]
    preset_select.set_value("30-min journey (Delta family)").run(timeout=30)
    [b for b in at.main.button if b.label == "Load preset"][0].click().run(timeout=60)
    assert not at.exception

    kfs = at.session_state["video_keyframes"]
    assert len(kfs) == 14
    assert kfs[0]["time"] == 0
    assert kfs[-1]["time"] == 1800
    assert any(kf["builtin_choice"].startswith("E4") for kf in kfs)
    assert any(kf["builtin_choice"].startswith("E6") for kf in kfs)
    assert any(kf["builtin_choice"].startswith("E8") for kf in kfs)

    duration_caption = [c for c in at.main.caption if "Duration:" in c.value][0]
    assert "1800.0s" in duration_caption.value
    assert len(at.main.warning) == 0  # no zoom-too-deep safety warnings


def test_video_precision_toggle_affects_safety_warnings():
    import json as json_mod

    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    # A keyframe safe at double precision but not at single (loaded via the
    # already-verified JSON import path, since directly mutating the
    # video_keyframes dicts wouldn't take effect -- the widgets' own sticky
    # session-state values win over that on the next rerun).
    borderline_timeline = [
        {"time": 0.0, "region": "halfplane", "center_x": 0.618, "center_y": 0.3, "scale": 0.27, "n_terms": 600},
        {"time": 10.0, "region": "halfplane", "center_x": 0.618, "center_y": 0.01, "scale": 0.005, "n_terms": 600},
    ]
    import_area = [t for t in at.main.text_area if "Paste timeline JSON" in t.label][0]
    import_area.set_value(json_mod.dumps(borderline_timeline)).run(timeout=30)
    [b for b in at.main.button if b.label == "Load from JSON"][0].click().run(timeout=30)
    assert not at.exception
    assert len(at.main.warning) == 0

    precision_radio = [r for r in at.main.radio if r.label == "Precision"][0]
    precision_radio.set_value("Single").run(timeout=30)
    assert not at.exception
    assert len(at.main.warning) == 1
    assert "single precision" in at.main.warning[0].value


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_video_render_with_parallel_workers_and_single_precision():
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _switch_to_video_mode(at)

    res_select = [s for s in at.main.selectbox if s.label == "Resolution preset"][0]
    res_select.set_value("Preview — 480x270 @ 15fps").run(timeout=30)

    precision_radio = [r for r in at.main.radio if r.label == "Precision"][0]
    precision_radio.set_value("Single").run(timeout=30)

    workers_input = [n for n in at.main.number_input if n.label == "Parallel workers"][0]
    workers_input.set_value(2).run(timeout=30)

    render_button = [b for b in at.main.button if "Render Video" in b.label][0]
    render_button.click().run(timeout=120)
    assert not at.exception
    assert "video_bytes" in at.session_state
    assert len(at.session_state["video_bytes"]) > 0
