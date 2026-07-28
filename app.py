"""Interactive UI for modforms: pick a form, a region, and a visualization
style from dropdowns, and render it.

Run with:

    streamlit run app.py
"""

import io

import numpy as np
import streamlit as st
from PIL import Image

from modforms import forms, lmfdb, plotting, presets
from modforms.coloring import STYLE_LABELS, STYLE_PARAMS, STYLES
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

CMAPS = ["cividis", "viridis", "twilight", "plasma", "inferno", "coolwarm", "Paired", "magma"]

HALFPLANE_PRESETS = {
    "Custom": None,
    "Delta box [-1,1]x[0,2]": presets.DELTA_BOX,
    "Delta zoom [0.1,0.4]x[0,0.25]": presets.DELTA_ZOOM,
    "g box [-2.5,2.5]x[0,2]": presets.G_BOX,
    "f105 box [-1,1]x[0,1]": presets.F105_BOX,
    "f10 box [-1,1]x[0,2]": presets.F10_BOX,
}


@st.cache_data(show_spinner=False)
def cached_builtin_form(name, n_terms):
    return BUILTIN_FORMS[name](n_terms)


@st.cache_data(show_spinner="Searching the LMFDB...")
def cached_lmfdb_search(level, weight, limit):
    return lmfdb.search_newforms(level=level, weight=weight, limit=limit)


@st.cache_data(show_spinner="Fetching q-expansion from the LMFDB...")
def cached_lmfdb_qexpansion(label, n_terms):
    form = lmfdb.fetch_qexpansion(label, n_terms=n_terms)
    return form.coeffs, form.start, form.label


def rgb_to_png_bytes(rgb):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


st.title("🌀 Visualizing Modular Forms")
st.caption(
    "Based on David Lowry-Duda, *Visualizing Modular Forms*, "
    "[arXiv:2002.05234](https://arxiv.org/abs/2002.05234)."
)

form = None

with st.sidebar:
    dark_mode = st.toggle("🌙 Night mode", value=False, help="Dark UI, and a black plot background instead of white.")
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
                coeffs, start, resolved_label = cached_lmfdb_qexpansion(label, n_terms)
                st.session_state["lmfdb_form"] = (coeffs, start, resolved_label)
                st.success(f"Loaded {resolved_label} ({len(coeffs)} terms)")
            except (lmfdb.LMFDBError, ValueError) as e:
                st.error(str(e))

        if "lmfdb_form" in st.session_state:
            coeffs, start, resolved_label = st.session_state["lmfdb_form"]
            form = QExpansion(coeffs, start=start, label=resolved_label)
            st.caption(f"✓ Currently loaded: **{resolved_label}** — click **▶ Render** below to plot it.")

    else:  # Upload CSV
        upload = st.file_uploader(
            "CSV of Fourier coefficients (one per line, starting at q^start)", type="csv"
        )
        start = st.number_input("Exponent of q for first coefficient", value=1, step=1)
        if upload is not None:
            form = QExpansion.from_csv(upload, start=int(start), label=upload.name)

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
    needed = STYLE_PARAMS.get(style, [])
    if "cmap" in needed:
        style_kwargs["cmap"] = st.selectbox("Colormap", CMAPS)
    if "alpha" in needed:
        style_kwargs["alpha"] = st.slider("Alpha (magnitude curve)", 0.05, 1.0, 0.25, 0.05)
    if "base" in needed:
        default_base = 7.0 if style == "periodic-log" else 2.0
        style_kwargs["base"] = st.number_input("Base (contour/log spacing)", min_value=1.01, value=default_base, step=0.5)
    if "offset" in needed:
        style_kwargs["offset"] = st.slider("Hue offset", 0.0, 1.0, 0.0, 0.05)

    st.header("4. Resolution")
    res = st.slider("Grid size (rows = cols)", 128, 8192, 512, 64)
    if res > 2048:
        st.caption(
            f"⚠️ {res}×{res} (~{res * res / 1e6:.0f}M points) will take a while — "
            "roughly a minute around 4K, several minutes at 8K — and can use "
            "several GB of RAM at the top end."
        )

    render_clicked = st.button("▶ Render", type="primary", width="stretch")

if render_clicked:
    if form is None:
        st.error("No form loaded — pick a built-in form, load an LMFDB label, or upload a CSV.")
    else:
        with st.spinner("Evaluating and rendering..."):
            vals = plotting.evaluate_on_region(
                form,
                region=region,
                box=box or ((-1, 1), (0, 2)),
                shape=(res, res),
                disk_extent=disk_extent,
            )
            plot_background = (0.0, 0.0, 0.0) if dark_mode else (1.0, 1.0, 1.0)
            rgb = plotting.render(vals, style=style, background=plot_background, **style_kwargs)
            st.session_state["last_png"] = rgb_to_png_bytes(rgb)
            label = getattr(form, "label", None) or "custom form"
            st.session_state["last_caption"] = f"{label} — {STYLE_LABELS.get(style, style)}"

if "last_png" in st.session_state:
    st.image(st.session_state["last_png"], caption=st.session_state["last_caption"])
    st.download_button("Download PNG", st.session_state["last_png"], file_name="modform.png", mime="image/png")
else:
    st.info("Configure a form in the sidebar and click **Render**.")
