"""Interactive UI for modforms: pick a form, a region, and a visualization
style from dropdowns, and render it -- or build a soothing zoom video.

Run with:

    streamlit run app.py
"""

import json
import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from modforms import forms, lmfdb, plotting, presets, reproduce
from modforms import video as video_mod
from modforms.coloring import STYLE_LABELS, STYLE_PARAMS, STYLES, custom_colormap
from modforms.qexpansion import QExpansion

st.set_page_config(page_title="mod-forms", page_icon="🌀", layout="wide")

# Trim Streamlit's default top/bottom padding and cap the rendered image to
# the viewport height so the preview fits without scrolling, regardless of
# the underlying pixel resolution (which can be much larger, up to 8K).
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 1rem; }
    div[data-testid="stImage"] img {
        max-height: 78vh !important;
        width: auto !important;
        max-width: 100% !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BUILTIN_FORMS = {
    "Delta — weight 12, level 1": lambda n: forms.delta(n),
    "E4 — Eisenstein, weight 4": lambda n: forms.eisenstein(4, n),
    "E6 — Eisenstein, weight 6": lambda n: forms.eisenstein(6, n),
    "E8 — Eisenstein, weight 8": lambda n: forms.eisenstein(8, n),
    "E10 — Eisenstein, weight 10": lambda n: forms.eisenstein(10, n),
    "E14 — Eisenstein, weight 14": lambda n: forms.eisenstein(14, n),
}

# The same built-in choices, as form-spec dicts for modforms.video.Keyframe
# (used by Video mode) instead of ready-made QExpansion objects.
BUILTIN_FORM_SPECS = {
    "Delta — weight 12, level 1": {"source": "delta"},
    "E4 — Eisenstein, weight 4": {"source": "eisenstein", "weight": 4},
    "E6 — Eisenstein, weight 6": {"source": "eisenstein", "weight": 6},
    "E8 — Eisenstein, weight 8": {"source": "eisenstein", "weight": 8},
    "E10 — Eisenstein, weight 10": {"source": "eisenstein", "weight": 10},
    "E14 — Eisenstein, weight 14": {"source": "eisenstein", "weight": 14},
}

CMAPS = ["cividis", "viridis", "twilight", "plasma", "inferno", "coolwarm", "Paired", "magma"]
CUSTOM_CMAP_DEFAULT_COLORS = ["#000000", "#7b2ff7", "#f72585", "#ffbe0b"]

ASPECT_RATIOS = {
    "Native (undistorted)": None,
    "Square 1:1": 1 / 1,
    "4:3": 4 / 3,
    "3:4 (portrait)": 3 / 4,
    "16:9": 16 / 9,
    "9:16 (portrait)": 9 / 16,
    "21:9 (ultrawide)": 21 / 9,
    "Custom": "custom",
}

HALFPLANE_PRESETS = {
    "Custom": None,
    "Delta box [-1,1]x[0,2]": presets.DELTA_BOX,
    "Delta zoom [0.1,0.4]x[0,0.25]": presets.DELTA_ZOOM,
    "g box [-2.5,2.5]x[0,2]": presets.G_BOX,
    "f105 box [-1,1]x[0,1]": presets.F105_BOX,
    "f10 box [-1,1]x[0,2]": presets.F10_BOX,
}

VIDEO_RESOLUTION_PRESETS = {
    "Preview — 480x270 @ 15fps": (480, 270, 15),
    "HD — 1280x720 @ 30fps": (1280, 720, 30),
    "Full HD — 1920x1080 @ 30fps": (1920, 1080, 30),
    "Custom": None,
}

VIDEO_QUALITY_PRESETS = {"Standard": 23, "High": 18, "Very high": 14}


@st.cache_data(show_spinner=False)
def cached_builtin_form(name, n_terms):
    return BUILTIN_FORMS[name](n_terms)


@st.cache_data(show_spinner="Searching the LMFDB...")
def cached_lmfdb_search(level, weight, limit):
    return lmfdb.search_newforms(level=level, weight=weight, limit=limit)


@st.cache_data(show_spinner="Fetching q-expansion from the LMFDB...")
def cached_lmfdb_qexpansion(label, n_terms):
    form = lmfdb.fetch_qexpansion(label, n_terms=n_terms)
    return form.coeffs, form.start, form.weight, form.level, form.label


def build_output_filename(form, source_kind, region, style):
    symbol = {"delta": "Delta", "lmfdb": getattr(form, "label", None)}.get(source_kind)
    if symbol is None and source_kind == "eisenstein":
        symbol = f"E{int(form.weight)}" if getattr(form, "weight", None) else "E"
    symbol = symbol or "form"
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(symbol)).strip("_") or "form"
    return f"{safe_symbol}_{region}_{style}.png"


_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "^": r"\^{}",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "&": r"\&",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
}


def _latex_escape(s):
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in s)


def build_form_latex(form, source_kind):
    """LaTeX describing exactly what's being plotted: the truncated
    q-expansion actually evaluated, plus weight/level when known -- the
    same (k, N) notation the paper uses for a modular form of weight k on
    a congruence subgroup of level N.
    """
    start = form.start
    end = start + len(form.coeffs) - 1
    if source_kind == "delta":
        symbol, coeff_name = r"\Delta", r"\tau(n)"
    elif source_kind == "eisenstein":
        symbol, coeff_name = rf"E_{{{int(form.weight)}}}", "a_n"
    elif source_kind == "lmfdb":
        symbol = rf"f_{{\text{{{_latex_escape(form.label)}}}}}"
        coeff_name = "a_n"
    else:
        symbol, coeff_name = "f", "a_n"

    latex = rf"{symbol}(z) = \sum_{{n={start}}}^{{{end}}} {coeff_name}\, q^{{n}}, \quad q = e^{{2\pi i z}}"

    meta = []
    if source_kind != "eisenstein" and getattr(form, "weight", None) is not None:
        meta.append(rf"k = {form.weight}")
    if getattr(form, "level", None) is not None:
        meta.append(rf"N = {form.level}")
    if meta:
        latex += r",\quad " + r",\ ".join(meta)
    return latex


def build_domain_latex(region, box, disk_extent):
    if region == "disk":
        return rf"z = \varphi(w), \quad w \in \mathbb{{D}},\ |w| < {disk_extent:.3g}"
    (x0, x1), (y0, y1) = box
    return rf"z = x + iy \in \mathbb{{H}}, \quad x \in [{x0:.3g}, {x1:.3g}],\ y \in [{y0:.3g}, {y1:.3g}]"


def build_settings_caption(style, style_kwargs, cmap_fingerprint, content_shape, final_shape, bg_choice):
    parts = [STYLE_LABELS.get(style, style)]
    if "cmap" in style_kwargs:
        if isinstance(cmap_fingerprint, tuple):
            parts.append(f"custom colormap ({len(cmap_fingerprint[1])} colors)")
        else:
            parts.append(f"colormap: {cmap_fingerprint}")
    if "alpha" in style_kwargs:
        parts.append(f"α = {style_kwargs['alpha']:.2f}")
    if "base" in style_kwargs:
        parts.append(f"base = {style_kwargs['base']:.3g}")
    if "offset" in style_kwargs:
        parts.append(f"offset = {style_kwargs['offset']:.2f}")
    dims = f"{content_shape[1]}×{content_shape[0]} px"
    if final_shape != content_shape:
        dims += f" → {final_shape[1]}×{final_shape[0]} px"
    parts.append(dims)
    parts.append(f"background: {bg_choice.lower()}")
    return " · ".join(parts)


def _hex_to_rgb01(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def render_image_mode():
    form = None
    source_kind = None

    with st.sidebar:
        dark_mode = st.toggle(
            "🌙 Night mode", value=False, help="Dark UI chrome. Plot background is set separately below."
        )
        if dark_mode:
            st.markdown(
                """
                <style>
                .stApp { background-color: #000000; color: #e6e6e6; }
                section[data-testid="stSidebar"] { background-color: #0a0a0a; }
                section[data-testid="stSidebar"] * { color: #e6e6e6; }
                .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span { color: #e6e6e6; }
                </style>
                """,
                unsafe_allow_html=True,
            )

        st.header("1. Form")
        source = st.radio("Source", ["Built-in", "LMFDB", "Upload CSV"], horizontal=True)

        if source == "Built-in":
            n_terms = st.slider("Fourier coefficients (terms)", 20, 800, 400, 20)
            choice = st.selectbox("Form", list(BUILTIN_FORMS))
            form = cached_builtin_form(choice, n_terms)
            source_kind = "delta" if choice.startswith("Delta") else "eisenstein"

        elif source == "LMFDB":
            n_terms = st.slider("Fourier coefficients (terms)", 20, 800, 400, 20)
            st.caption(
                "Either search by level/weight and pick a result below, or type a "
                "label directly (e.g. from the paper) — then click **Load form**."
            )

            with st.expander("Search by level / weight", expanded=True):
                level = st.number_input("Level", min_value=1, value=105, step=1)
                weight = st.number_input("Weight", min_value=1, value=2, step=1)
                if st.button("Search"):
                    try:
                        st.session_state["lmfdb_results"] = cached_lmfdb_search(int(level), int(weight), 20)
                    except lmfdb.LMFDBError as e:
                        st.error(str(e))
                results = st.session_state.get("lmfdb_results")
                if results:
                    labels = [d.get("label") for d in results]
                    st.dataframe(
                        [{"label": d.get("label"), "dim": d.get("dim")} for d in results],
                        width="stretch",
                        hide_index=True,
                    )
                    st.selectbox(
                        "Pick a result → fills in the label below",
                        labels,
                        key="lmfdb_pick_result",
                        on_change=lambda: st.session_state.__setitem__(
                            "lmfdb_label_input", st.session_state["lmfdb_pick_result"]
                        ),
                    )
                elif results == []:
                    st.caption("No results.")

            st.session_state.setdefault("lmfdb_label_input", "5.4.a.a")
            label = st.text_input(
                "LMFDB label",
                key="lmfdb_label_input",
                help="Galois-orbit label (e.g. 5.4.a.a) or embedded label (5.4.a.a.1.1)",
            )
            if st.button("Load form", type="secondary"):
                try:
                    coeffs, start, weight, lvl, resolved_label = cached_lmfdb_qexpansion(label, n_terms)
                    st.session_state["lmfdb_form"] = (coeffs, start, weight, lvl, resolved_label)
                    st.success(f"Loaded {resolved_label} ({len(coeffs)} terms)")
                except (lmfdb.LMFDBError, ValueError) as e:
                    st.error(str(e))

            if "lmfdb_form" in st.session_state:
                coeffs, start, weight, lvl, resolved_label = st.session_state["lmfdb_form"]
                form = QExpansion(coeffs, start=start, weight=weight, level=lvl, label=resolved_label)
                source_kind = "lmfdb"
                st.caption(f"✓ Currently loaded: **{resolved_label}** — click **▶ Render** below to plot it.")

        else:  # Upload CSV
            upload = st.file_uploader(
                "CSV of Fourier coefficients (one per line, starting at q^start)", type="csv"
            )
            start = st.number_input("Exponent of q for first coefficient", value=1, step=1)
            wc1, wc2 = st.columns(2)
            csv_weight = wc1.number_input("Weight (optional)", min_value=0, value=0, step=1, help="0 = unspecified")
            csv_level = wc2.number_input("Level (optional)", min_value=0, value=0, step=1, help="0 = unspecified")
            if upload is not None:
                form = QExpansion.from_csv(
                    upload,
                    start=int(start),
                    weight=int(csv_weight) or None,
                    level=int(csv_level) or None,
                    label=upload.name,
                )
                source_kind = "csv"

        st.header("2. Region")
        region = st.radio(
            "Region",
            ["disk", "halfplane"],
            horizontal=True,
            format_func=lambda r: "Poincaré disk" if r == "disk" else "Upper halfplane",
        )

        box = None
        disk_extent = 1.02
        if region == "halfplane":
            preset_name = st.selectbox("Preset box", list(HALFPLANE_PRESETS))
            preset = HALFPLANE_PRESETS[preset_name]
            if preset is None:
                c1, c2 = st.columns(2)
                x0 = c1.number_input("x0", value=-1.0)
                x1 = c2.number_input("x1", value=1.0)
                y0 = c1.number_input("y0", value=0.0)
                y1 = c2.number_input("y1", value=2.0)
                box = ((x0, x1), (y0, y1))
            else:
                box = preset
                st.caption(f"Box: x ∈ [{box[0][0]}, {box[0][1]}], y ∈ [{box[1][0]}, {box[1][1]}]")
        else:
            disk_extent = st.number_input("Disk plot extent", min_value=1.0, max_value=2.0, value=1.02, step=0.01)

        st.header("3. Style")
        style = st.selectbox("Visualization style", list(STYLES), format_func=lambda k: STYLE_LABELS.get(k, k))

        style_kwargs = {}
        cmap_fingerprint = None
        needed = STYLE_PARAMS.get(style, [])
        if "cmap" in needed:
            cmap_source = st.radio("Colormap", ["Built-in", "Custom"], horizontal=True)
            if cmap_source == "Built-in":
                style_kwargs["cmap"] = st.selectbox("Choose colormap", CMAPS)
                cmap_fingerprint = style_kwargs["cmap"]
            else:
                n_stops = st.slider("Number of colors", 2, 6, 3)
                stop_cols = st.columns(n_stops)
                stops = [
                    stop_cols[i].color_picker(
                        f"#{i + 1}", CUSTOM_CMAP_DEFAULT_COLORS[i % len(CUSTOM_CMAP_DEFAULT_COLORS)]
                    )
                    for i in range(n_stops)
                ]
                cyclic = st.checkbox(
                    "Cyclic (wraps smoothly — good for phase-based styles)", value=True
                )
                style_kwargs["cmap"] = custom_colormap(stops, cyclic=cyclic)
                cmap_fingerprint = ("custom", tuple(stops), cyclic)
        if "alpha" in needed:
            style_kwargs["alpha"] = st.slider("Alpha (magnitude curve)", 0.05, 1.0, 0.25, 0.05)
        if "base" in needed:
            default_base = 7.0 if style == "periodic-log" else 2.0
            style_kwargs["base"] = st.number_input(
                "Base (contour/log spacing)", min_value=1.01, value=default_base, step=0.5
            )
        if "offset" in needed:
            style_kwargs["offset"] = st.slider("Hue offset", 0.0, 1.0, 0.0, 0.05)

        st.header("4. Resolution & Aspect Ratio")
        res = st.slider("Detail (long edge of the plotted content, px)", 128, 8192, 512, 64)
        if res > 2048:
            st.caption(
                f"⚠️ {res}×{res} (~{res * res / 1e6:.0f}M points) will take a while — "
                "roughly a minute around 4K, several minutes at 8K — and can use "
                "several GB of RAM at the top end."
            )

        aspect_choice = st.selectbox("Output aspect ratio", list(ASPECT_RATIOS))
        target_ratio = ASPECT_RATIOS[aspect_choice]
        if target_ratio == "custom":
            rc1, rc2 = st.columns(2)
            aw = rc1.number_input("Width", min_value=1, value=16, step=1)
            ah = rc2.number_input("Height", min_value=1, value=9, step=1)
            target_ratio = aw / ah

        content_shape = (res, res) if region == "disk" else plotting.natural_shape(box or ((-1, 1), (0, 2)), res)
        final_shape = plotting.padded_shape(content_shape, target_ratio) if target_ratio is not None else content_shape
        st.caption(
            f"Plotted content: {content_shape[1]}×{content_shape[0]} px"
            + (f" → padded to {final_shape[1]}×{final_shape[0]} px" if final_shape != content_shape else "")
        )

        st.header("5. Background")
        bg_choice = st.radio(
            "Outside the disk (or masked region)",
            ["White", "Black", "Custom"],
            horizontal=True,
            help="Fill color for points outside the Poincaré disk (disk region) or any aspect-ratio padding.",
        )
        if bg_choice == "White":
            plot_background = (1.0, 1.0, 1.0)
        elif bg_choice == "Black":
            plot_background = (0.0, 0.0, 0.0)
        else:
            bg_hex = st.color_picker("Background color", "#000000")
            plot_background = _hex_to_rgb01(bg_hex)

        render_clicked = st.button("▶ Render", type="primary", width="stretch")

    # Fingerprint of everything that affects the rendered image, so we can tell
    # the user when the on-screen preview no longer matches the current
    # controls (e.g. toggling Night mode doesn't retroactively repaint an
    # already-rendered image -- only the next Render does). Custom colormaps
    # use `cmap_fingerprint` here instead of the actual Colormap object, since
    # a fresh (but equal) object is rebuilt on every rerun.
    fingerprint_style_kwargs = dict(style_kwargs)
    if "cmap" in fingerprint_style_kwargs:
        fingerprint_style_kwargs["cmap"] = cmap_fingerprint
    current_fingerprint = (
        getattr(form, "label", None),
        len(form.coeffs) if form is not None else None,
        region,
        box,
        disk_extent,
        style,
        tuple(sorted(fingerprint_style_kwargs.items())),
        res,
        target_ratio,
        plot_background,
        dark_mode,
    )

    if render_clicked:
        if form is None:
            st.error("No form loaded — pick a built-in form, load an LMFDB label, or upload a CSV.")
        else:
            with st.spinner("Evaluating and rendering..."):
                vals = plotting.evaluate_on_region(
                    form,
                    region=region,
                    box=box or ((-1, 1), (0, 2)),
                    shape=content_shape,
                    disk_extent=disk_extent,
                )
                rgb = plotting.render(vals, style=style, background=plot_background, **style_kwargs)
                if target_ratio is not None:
                    rgb = plotting.pad_to_aspect(rgb, target_ratio, background=plot_background)
                out_shape = rgb.shape[:2]

                metadata = reproduce.build_metadata(
                    form,
                    source_kind,
                    region=region,
                    box=box,
                    disk_extent=disk_extent,
                    style=style,
                    style_kwargs_json=reproduce.style_kwargs_to_json(style_kwargs, cmap_fingerprint),
                    content_shape=content_shape,
                    output_shape=out_shape,
                    target_ratio=target_ratio,
                    background=plot_background,
                )
                st.session_state["last_png"] = plotting.rgb_to_png_bytes(rgb, metadata=metadata)
                st.session_state["last_filename"] = build_output_filename(form, source_kind, region, style)
                st.session_state["last_form_latex"] = build_form_latex(form, source_kind)
                st.session_state["last_domain_latex"] = build_domain_latex(region, box, disk_extent)
                st.session_state["last_settings_caption"] = build_settings_caption(
                    style, style_kwargs, cmap_fingerprint, content_shape, out_shape, bg_choice
                )
                st.session_state["last_render_fingerprint"] = current_fingerprint

    if "last_png" in st.session_state:
        if st.session_state.get("last_render_fingerprint") != current_fingerprint:
            st.warning("Settings changed since this render — click **▶ Render** to update the preview.")
        st.image(st.session_state["last_png"])
        st.latex(st.session_state["last_form_latex"])
        st.latex(st.session_state["last_domain_latex"])
        st.markdown(
            f'<div style="text-align:center; opacity:0.75; font-size:0.85rem; margin-top:-0.6rem;">'
            f'{st.session_state["last_settings_caption"]}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download PNG",
            st.session_state["last_png"],
            file_name=st.session_state.get("last_filename", "modform.png"),
            mime="image/png",
            help=(
                "The PNG embeds everything needed to reproduce it (form, region, style, "
                "resolution, background) as metadata -- run "
                "`python -m modforms.cli replot <file> --out new.png` to re-render it exactly."
            ),
        )
    else:
        st.info("Configure a form in the sidebar and click **Render**.")


def _default_keyframe(next_id, **overrides):
    kf = {
        "id": next_id,
        "time": 0.0,
        "source": "Built-in",
        "builtin_choice": "Delta — weight 12, level 1",
        "lmfdb_label": "5.4.a.a",
        "n_terms": 400,
        "region": "disk",
        "center_x": 0.0,
        "center_y": 0.0,
        "scale": 1.02,
        "style": "colormap-phase-contour",
        "cmap_source": "Built-in",
        "cmap_name": "cividis",
        "custom_stops": list(CUSTOM_CMAP_DEFAULT_COLORS[:3]),
        "cyclic": True,
        "alpha": 0.25,
        "base": 2.0,
        "offset": 0.0,
        "bg_choice": "Black",
        "bg_custom": "#000000",
    }
    kf.update(overrides)
    return kf


def _preset_keyframe(**overrides):
    """Like _default_keyframe, but with no id -- for VIDEO_PRESETS templates
    (ids get assigned fresh when a preset is actually loaded).
    """
    kf = _default_keyframe(0, **overrides)
    del kf["id"]
    return kf


# Ready-made timelines, selectable in the UI without typing anything in.
VIDEO_PRESETS = {
    "Golden cusp zoom (Delta)": [
        _preset_keyframe(time=0.0, region="disk", center_x=0.0, center_y=0.0, scale=1.02, n_terms=600, cmap_name="cividis"),
        _preset_keyframe(time=10.0, region="disk", center_x=0.0, center_y=0.0, scale=1.02, n_terms=600, cmap_name="cividis"),
        _preset_keyframe(time=22.0, region="disk", center_x=0.44, center_y=0.762, scale=0.15, n_terms=600, cmap_name="twilight"),
        _preset_keyframe(time=34.0, region="disk", center_x=-0.762, center_y=0.44, scale=0.15, n_terms=600, cmap_name="viridis"),
        _preset_keyframe(time=48.0, region="halfplane", center_x=0.618, center_y=0.3, scale=0.27, n_terms=600, cmap_name="viridis"),
        _preset_keyframe(time=60.0, region="halfplane", center_x=0.618, center_y=0.08, scale=0.07, n_terms=600, cmap_name="viridis"),
        _preset_keyframe(time=74.0, region="halfplane", center_x=0.618, center_y=0.02, scale=0.018, n_terms=600, cmap_name="plasma"),
        _preset_keyframe(time=90.0, region="halfplane", center_x=0.618, center_y=0.006, scale=0.005, n_terms=600, cmap_name="plasma"),
    ],
    "Simple cusp zoom (default)": [
        _preset_keyframe(time=0.0, region="disk", center_x=0.0, center_y=0.0, scale=1.02, cmap_name="cividis"),
        _preset_keyframe(time=10.0, region="disk", center_x=0.0, center_y=0.85, scale=0.05, cmap_name="twilight"),
    ],
}


def _load_keyframes(keyframe_dicts):
    """Replace the current timeline with (copies of, freshly re-ID'd)
    ``keyframe_dicts`` -- used by both the preset picker and JSON import.
    """
    next_id = st.session_state["video_kf_next_id"]
    loaded = []
    for kf in keyframe_dicts:
        merged = _default_keyframe(next_id)
        merged.update(kf)
        merged["id"] = next_id
        merged["custom_stops"] = list(merged["custom_stops"])
        next_id += 1
        loaded.append(merged)
    st.session_state["video_kf_next_id"] = next_id
    st.session_state["video_keyframes"] = loaded


def _ensure_video_state():
    if "video_keyframes" not in st.session_state:
        st.session_state["video_kf_next_id"] = 2
        st.session_state["video_keyframes"] = [
            _default_keyframe(0),
            _default_keyframe(1, time=10.0, center_y=0.85, scale=0.05, cmap_name="twilight"),
        ]


def _kf_form_spec(kf):
    if kf["source"] == "Built-in":
        spec = dict(BUILTIN_FORM_SPECS[kf["builtin_choice"]])
    else:
        spec = {"source": "lmfdb", "label": kf["lmfdb_label"]}
    spec["n_terms"] = kf["n_terms"]
    return spec


def _kf_cmap_fingerprint(kf):
    if kf["cmap_source"] == "Built-in":
        return video_mod.builtin_cmap(kf["cmap_name"])
    return video_mod.custom_cmap(kf["custom_stops"], cyclic=kf["cyclic"])


def _kf_background(kf):
    if kf["bg_choice"] == "White":
        return (1.0, 1.0, 1.0)
    if kf["bg_choice"] == "Black":
        return (0.0, 0.0, 0.0)
    return _hex_to_rgb01(kf["bg_custom"])


def ui_keyframe_to_video_keyframe(kf):
    """Convert one of this app's plain keyframe dicts (from session_state)
    into a real modforms.video.Keyframe.
    """
    style_kwargs = {}
    needed = STYLE_PARAMS.get(kf["style"], [])
    if "cmap" in needed:
        style_kwargs["cmap"] = _kf_cmap_fingerprint(kf)
    if "alpha" in needed:
        style_kwargs["alpha"] = kf["alpha"]
    if "base" in needed:
        style_kwargs["base"] = kf["base"]
    if "offset" in needed:
        style_kwargs["offset"] = kf["offset"]

    return video_mod.Keyframe(
        time=kf["time"],
        form=_kf_form_spec(kf),
        region=kf["region"],
        center=(kf["center_x"], kf["center_y"]),
        scale=kf["scale"],
        style=kf["style"],
        style_kwargs=style_kwargs,
        background=_kf_background(kf),
    )


def render_keyframe_editor(kf, index, total):
    kid = kf["id"]
    with st.expander(f"Keyframe {index + 1} — t = {kf['time']:.1f}s", expanded=(index >= total - 2)):
        kf["time"] = st.number_input("Time (seconds)", min_value=0.0, value=float(kf["time"]), step=1.0, key=f"kf{kid}_time")

        st.markdown("**Form**")
        kf["source"] = st.radio("Source", ["Built-in", "LMFDB"], horizontal=True, key=f"kf{kid}_source", index=["Built-in", "LMFDB"].index(kf["source"]))
        kf["n_terms"] = st.slider("Terms", 20, 800, kf["n_terms"], 20, key=f"kf{kid}_n_terms")
        if kf["source"] == "Built-in":
            kf["builtin_choice"] = st.selectbox(
                "Form", list(BUILTIN_FORM_SPECS), key=f"kf{kid}_builtin", index=list(BUILTIN_FORM_SPECS).index(kf["builtin_choice"])
            )
        else:
            kf["lmfdb_label"] = st.text_input(
                "LMFDB label", value=kf["lmfdb_label"], key=f"kf{kid}_lmfdb",
                help="Galois-orbit label (e.g. 5.4.a.a) or embedded label (5.4.a.a.1.1). "
                "Use Image mode's search to find one first.",
            )

        st.markdown("**View**")
        kf["region"] = st.radio(
            "Region", ["disk", "halfplane"], horizontal=True, key=f"kf{kid}_region",
            index=["disk", "halfplane"].index(kf["region"]),
            format_func=lambda r: "Poincaré disk" if r == "disk" else "Upper halfplane",
        )
        cc1, cc2, cc3 = st.columns(3)
        x_label = "center Re(w)" if kf["region"] == "disk" else "center x"
        y_label = "center Im(w)" if kf["region"] == "disk" else "center y"
        kf["center_x"] = cc1.number_input(x_label, value=float(kf["center_x"]), format="%.5f", key=f"kf{kid}_cx")
        kf["center_y"] = cc2.number_input(y_label, value=float(kf["center_y"]), format="%.5f", key=f"kf{kid}_cy")
        kf["scale"] = cc3.number_input(
            "scale (half-height)", value=float(kf["scale"]), min_value=1e-6, format="%.5f", key=f"kf{kid}_scale"
        )

        st.markdown("**Style**")
        kf["style"] = st.selectbox(
            "Visualization style", list(STYLES), key=f"kf{kid}_style", index=list(STYLES).index(kf["style"]),
            format_func=lambda k: STYLE_LABELS.get(k, k),
        )
        needed = STYLE_PARAMS.get(kf["style"], [])
        if "cmap" in needed:
            kf["cmap_source"] = st.radio(
                "Colormap", ["Built-in", "Custom"], horizontal=True, key=f"kf{kid}_cmap_source",
                index=["Built-in", "Custom"].index(kf["cmap_source"]),
            )
            if kf["cmap_source"] == "Built-in":
                kf["cmap_name"] = st.selectbox(
                    "Choose colormap", CMAPS, key=f"kf{kid}_cmap_name", index=CMAPS.index(kf["cmap_name"]) if kf["cmap_name"] in CMAPS else 0
                )
            else:
                n_stops = st.slider("Number of colors", 2, 6, len(kf["custom_stops"]), key=f"kf{kid}_n_stops")
                while len(kf["custom_stops"]) < n_stops:
                    kf["custom_stops"].append(CUSTOM_CMAP_DEFAULT_COLORS[len(kf["custom_stops"]) % len(CUSTOM_CMAP_DEFAULT_COLORS)])
                del kf["custom_stops"][n_stops:]
                stop_cols = st.columns(n_stops)
                for i in range(n_stops):
                    kf["custom_stops"][i] = stop_cols[i].color_picker(f"#{i + 1}", kf["custom_stops"][i], key=f"kf{kid}_stop{i}")
                kf["cyclic"] = st.checkbox("Cyclic", value=kf["cyclic"], key=f"kf{kid}_cyclic")
        if "alpha" in needed:
            kf["alpha"] = st.slider("Alpha", 0.05, 1.0, kf["alpha"], 0.05, key=f"kf{kid}_alpha")
        if "base" in needed:
            kf["base"] = st.number_input("Base", min_value=1.01, value=kf["base"], step=0.5, key=f"kf{kid}_base")
        if "offset" in needed:
            kf["offset"] = st.slider("Hue offset", 0.0, 1.0, kf["offset"], 0.05, key=f"kf{kid}_offset")

        st.markdown("**Background**")
        kf["bg_choice"] = st.radio(
            "Outside the disk/box", ["White", "Black", "Custom"], horizontal=True, key=f"kf{kid}_bg",
            index=["White", "Black", "Custom"].index(kf["bg_choice"]),
        )
        if kf["bg_choice"] == "Custom":
            kf["bg_custom"] = st.color_picker("Background color", kf["bg_custom"], key=f"kf{kid}_bg_custom")

        if total > 1:
            if st.button("🗑️ Remove this keyframe", key=f"kf{kid}_remove"):
                st.session_state["video_keyframes"] = [k for k in st.session_state["video_keyframes"] if k["id"] != kid]
                st.rerun()


def render_video_mode():
    _ensure_video_state()

    st.caption(
        "A Timeline is a sequence of Keyframes. Between two consecutive keyframes the view "
        "zooms geometrically (constant perceived zoom speed, eased smoothly) toward the next "
        "one's center/scale; if the form, style, or colormap also changes, the two are "
        "crossfaded across the same span. Two adjacent keyframes with identical settings just "
        "hold that view for a while."
    )

    load_cols = st.columns([3, 1])
    preset_choice = load_cols[0].selectbox("Load a preset timeline", list(VIDEO_PRESETS), label_visibility="collapsed")
    if load_cols[1].button("Load preset", width="stretch"):
        _load_keyframes(VIDEO_PRESETS[preset_choice])
        st.rerun()

    with st.expander("Import / export timeline as JSON"):
        st.caption("Paste a timeline (a JSON list of keyframe objects) someone gave you, or copy your own to save it.")
        import_text = st.text_area("Paste timeline JSON here", height=120, key="video_import_json")
        if st.button("Load from JSON"):
            try:
                loaded = json.loads(import_text)
                if not isinstance(loaded, list) or not loaded:
                    raise ValueError("expected a non-empty JSON list of keyframe objects")
                _load_keyframes(loaded)
                st.success(f"Loaded {len(loaded)} keyframes.")
                st.rerun()
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                st.error(f"Couldn't load that as a timeline: {e}")

        current_json = json.dumps(
            [{k: v for k, v in kf.items() if k != "id"} for kf in st.session_state["video_keyframes"]], indent=2
        )
        st.text_area("Current timeline (copy this to save it)", value=current_json, height=120)

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if not ffmpeg_available:
        st.error(
            "ffmpeg was not found on PATH. Install it to render video (e.g. `winget install ffmpeg` on "
            "Windows, `brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux), then restart this app."
        )

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Timeline")
        keyframes = st.session_state["video_keyframes"]
        for i, kf in enumerate(keyframes):
            render_keyframe_editor(kf, i, len(keyframes))

        if st.button("➕ Add keyframe"):
            last = keyframes[-1]
            new_id = st.session_state["video_kf_next_id"]
            st.session_state["video_kf_next_id"] += 1
            new_kf = dict(last)
            new_kf["id"] = new_id
            new_kf["custom_stops"] = list(last["custom_stops"])
            new_kf["time"] = last["time"] + 8.0
            new_kf["scale"] = max(1e-6, last["scale"] * 0.3)
            st.session_state["video_keyframes"].append(new_kf)
            st.rerun()

    timeline = video_mod.Timeline([ui_keyframe_to_video_keyframe(kf) for kf in st.session_state["video_keyframes"]])

    with right:
        st.subheader("Render settings")
        preset_name = st.selectbox("Resolution preset", list(VIDEO_RESOLUTION_PRESETS))
        preset = VIDEO_RESOLUTION_PRESETS[preset_name]
        if preset is None:
            rc1, rc2 = st.columns(2)
            width = rc1.number_input("Width", min_value=64, value=1280, step=32)
            height = rc2.number_input("Height", min_value=64, value=720, step=32)
            fps = st.slider("FPS", 5, 60, 30)
        else:
            width, height, fps = preset

        quality_name = st.selectbox("Quality", list(VIDEO_QUALITY_PRESETS), index=1)
        crf = VIDEO_QUALITY_PRESETS[quality_name]

        st.caption(f"Duration: {timeline.duration:.1f}s at {width}×{height} @ {fps}fps")

        for w in timeline.check_safety():
            st.warning(w)

        st.divider()
        st.subheader("Preview a single frame")
        preview_t = st.slider("Time (s)", 0.0, max(0.1, timeline.duration), 0.0, key="video_preview_t")
        if st.button("🔍 Preview this frame"):
            with st.spinner("Rendering preview frame..."):
                rgb = video_mod.render_frame(timeline, preview_t, width, height)
                st.session_state["video_preview_png"] = plotting.rgb_to_png_bytes(rgb)
        if "video_preview_png" in st.session_state:
            st.image(st.session_state["video_preview_png"])

        st.divider()
        st.subheader("Render")
        if st.button("⏱ Estimate render time"):
            with st.spinner("Timing a few sample frames..."):
                n_frames, per_frame, total_s = video_mod.estimate_render_time(timeline, fps=fps, width=width, height=height)
            st.info(f"{n_frames} frames, ~{per_frame * 1000:.0f} ms/frame → estimated **{total_s / 60:.1f} min**")

        render_video_clicked = st.button("🎬 Render Video", type="primary", disabled=not ffmpeg_available, width="stretch")

        if render_video_clicked:
            n_frames = max(1, round(timeline.duration * fps))
            progress_bar = st.progress(0.0, text=f"Rendering frame 0/{n_frames}...")

            def on_progress(done, total_frames):
                progress_bar.progress(done / total_frames, text=f"Rendering frame {done}/{total_frames}...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                out_path = Path(tmp_dir) / "video.mp4"
                try:
                    video_mod.render_video(
                        timeline,
                        str(out_path),
                        fps=fps,
                        width=width,
                        height=height,
                        crf=crf,
                        progress=False,
                        progress_callback=on_progress,
                    )
                    st.session_state["video_bytes"] = out_path.read_bytes()
                    st.success("Done!")
                except RuntimeError as e:
                    st.error(str(e))

        if "video_bytes" in st.session_state:
            st.video(st.session_state["video_bytes"])
            st.download_button(
                "Download MP4", st.session_state["video_bytes"], file_name="modform_video.mp4", mime="video/mp4"
            )


st.title("🌀 Visualizing Modular Forms")
st.caption(
    "Based on David Lowry-Duda, *Visualizing Modular Forms*, "
    "[arXiv:2002.05234](https://arxiv.org/abs/2002.05234)."
)

app_mode = st.radio("Mode", ["🖼️ Image", "🎬 Video"], horizontal=True, label_visibility="collapsed")
st.divider()

if app_mode == "🖼️ Image":
    render_image_mode()
else:
    render_video_mode()
