# mod-forms

Plotting tools for visualizing modular forms, implementing the techniques
surveyed in

> David Lowry-Duda, ["Visualizing Modular Forms"](https://arxiv.org/abs/2002.05234), arXiv:2002.05234.

The paper looks at several ways to turn a complex-valued modular form into
a picture — domain coloring, magnitude-only plots, phase plots, phase
plots with magnitude contours — and introduces a way to plug in
matplotlib's perceptually-uniform colormaps (`viridis`, `cividis`,
`twilight`, ...). This repo implements all of it in vectorized NumPy, on
both the upper halfplane and the Poincare disk.

| Standard domain coloring | Periodic magnitude (LMFDB style) |
|---|---|
| ![](docs/gallery/standard_disk.png) | ![](docs/gallery/periodic_linear_disk.png) |
| **Phase plot with contours** | **Colormap phase plot (cividis)** |
| ![](docs/gallery/phase_contour_disk.png) | ![](docs/gallery/colormap_phase_cividis.png) |

(All four are the Ramanujan Delta function, weight 12 level 1, on the
Poincare disk.)

## Install

```bash
pip install -e .
```

(or just `pip install -r requirements.txt` and add `src/` to your
`PYTHONPATH` — the only dependencies are `numpy` and `matplotlib`.)

## UI

```bash
pip install -e ".[ui]"     # or: pip install -r requirements-ui.txt
streamlit run app.py
```

This opens in your browser at `http://localhost:8501`, with a Mode switch
at the top: **🖼️ Image** for single stills, **🎬 Video** for building an
ambient zoom video (see below) — both without touching any code.

### Image mode

Dropdowns for the form (built-in, LMFDB import, or your own CSV), the
region, the visualization style, and whatever parameters that style
takes, plus a live preview and a PNG download button.

A few extras worth knowing about:

- **Resolution & aspect ratio**: the "Detail" slider controls the actual
  plotted content's resolution (up to 8K); a separate "Output aspect
  ratio" dropdown (Square, 4:3, 16:9, 21:9, custom W:H, ...) letterboxes
  that content onto a canvas of the target shape using the current
  background color, without stretching or cropping anything.
- **Custom colormaps**: for any colormap-based style, switch "Colormap"
  from Built-in to Custom to pick your own 2-6 colors with color pickers
  and build a colormap by interpolating between them (optionally cyclic,
  which matters for phase-based styles).
- **Background**: independent of Night mode (which only affects the UI
  chrome), a "Background" control sets what fills the region outside the
  disk/box — White, Black, or a custom color picker — used for both the
  disk mask and any aspect-ratio padding.
- **Night mode**: a dark UI theme, so the app's own chrome doesn't clash
  with a dark plot background.
- **LaTeX description**: below every render, a centered LaTeX block states
  exactly what was plotted — e.g. `Δ(z) = Σ τ(n) qⁿ, q = e^{2πiz}, k=12,
  N=1` for Delta, or the LMFDB label for an imported form — followed by
  the domain (disk or halfplane box) and a plain-text line of the style,
  colormap, resolution, and background actually used for that render.
- **Self-describing downloads**: every downloaded PNG (from the UI or
  `modforms.cli plot`) embeds everything needed to reproduce it — the
  exact Fourier coefficients (not just a label, so it doesn't depend on
  the LMFDB later), region, style/colormap, resolution, and background —
  as PNG metadata. Re-render one exactly with:

  ```bash
  python -m modforms.cli replot downloaded.png --out again.png
  ```

  (add `--shape ROWS COLS` to reproduce it at a different resolution).
  The filename is also descriptive (e.g. `Delta_disk_phase-contour.png`)
  rather than a generic `modform.png`. Prefer to keep tweaking a shot
  interactively in the UI instead of using `replot`? `modforms.cli
  inspect downloaded.png` prints the same settings in a human-readable
  form mapped onto the UI's own sidebar sections (1. Form, 2. Region, ...)
  so you know exactly what to re-enter by hand.

If you change a control after already rendering, a warning banner tells
you the preview is stale until you click Render again.

## Quick start (CLI / scripting)

```bash
python -m modforms.cli plot --form delta --region disk --style phase-contour --cmap cividis --out delta.png
python -m modforms.cli replot delta.png --out delta_again.png   # reproduce it exactly from its own metadata
python -m modforms.cli inspect delta.png                        # print its settings, e.g. to re-enter them in the UI by hand
```

```python
from modforms import forms, plotting

delta = forms.delta(400)  # 400 Fourier coefficients, as in the paper
plotting.plot_form(
    delta,
    region="disk",              # or "halfplane"
    style="phase-contour",      # see styles below
    out="delta.png",
)
```

Run `python examples/reproduce_delta_gallery.py` to regenerate a full
gallery (all styles, disk + halfplane) into `output/`.

## Video mode: ambient zoom videos

`modforms.video` renders continuous "infinite zoom" style videos —
zooming toward a cusp while crossfading between colormaps and even
switching forms mid-zoom — in the style of ambient fractal-zoom videos on
YouTube.

### In the UI

Switch the app's Mode to **🎬 Video**. Each keyframe (a point in time with
a form, view, style, and colormap) gets its own editable card — add one
with **➕ Add keyframe** (it starts as a copy of the last one, zoomed in a
bit further and 8s later, so you're tweaking rather than starting from
scratch), remove one with the 🗑️ button on its card. Two adjacent
keyframes with the same settings just hold that view; different settings
zoom *and* crossfade between them (see "which forms transition well"
below).

Before committing to a real render: use **🔍 Preview this frame** to check
any single moment instantly, and **⏱ Estimate render time** to see how
long the full thing will actually take (based on timing a few real
sample frames, not a guess) — then pick a resolution preset and hit
**🎬 Render Video**, which shows a live progress bar and, when done, an
inline player plus an MP4 download button. Rendering blocks the app
while it runs (this is a local, single-user tool, so that's fine — just
don't close the tab), so for anything beyond a short/preview-quality
clip, exporting a timeline script and rendering via the CLI in the
background (below) is more practical than waiting in the browser.

**Precision** and **Parallel workers** (next to Quality) control the two
performance levers described below under "Performance" — Parallel
workers defaults to your CPU's core count (frames render independently,
so this is close to a free speedup), Precision defaults to Double (safe);
switching it to Single roughly triples speed at the cost of a shallower
safe zoom depth, and any resulting safety warnings update live.

**Loading a timeline without typing it in**: the "Load a preset timeline"
dropdown has a few ready-made sequences (a short "golden cusp zoom" demo,
and a full "30-min journey" — see below), or expand "Import / export
timeline as JSON" to paste in any timeline as JSON (e.g. one shared by
someone else, or your own saved from the same box, which also lets a
timeline survive a page reload/browser restart, since the running app's
own state otherwise doesn't persist across those).

**For a longer (e.g. half-hour) video**, the "30-min journey (Delta
family)" preset is a full 14-keyframe, 1800-second structure: an
establishing shot, an orbit to a rich boundary region, a dive into the
golden-ratio zoom, a drift through the level-1 family (Delta → E4 → E6 →
E8, panning to a different rich region with each), then a second
golden-ratio dive before returning to the opening shot (loop-friendly).
**Render time is the real constraint at this length** — check "⏱ Estimate
render time" for this specific timeline on your own machine before
committing to anything; it varies a lot by hardware, and going from
Preview resolution to full HD can easily be a 30x-plus difference in
total render time for a video this long.

### From the command line (for full-length/production renders)

A real multi-minute render is thousands of frames and can take minutes to
hours — better suited to a background process than a browser tab. Write
a Python timeline script (see `examples/video_ambient_delta_zoom.py`) and
render it headlessly:

```bash
pip install -e .   # ffmpeg must also be on PATH -- see below
python -m modforms.cli video examples/video_ambient_delta_zoom.py --out preview.mp4 --preview
python -m modforms.cli video examples/video_ambient_delta_zoom.py --out video.mp4 --width 1920 --height 1080 --fps 30
```

`--preview` is a one-flag shortcut for a fast, low-res draft (480×270 @
15fps, fastest x264 preset) — same full timeline duration, just cheap to
render, so you can check the composition/timing before committing to a
full-resolution render. Individual flags (`--width`, `--fps`, ...) still
override just that setting if you want, e.g. `--preview --width 640` for
a slightly bigger draft. Add `--dry-run` on top of either to just print
the frame count and a time estimate without rendering anything.

A timeline is a Python file exporting `TIMELINE` (a list of `Keyframe`, or
a `Timeline`) — see `examples/video_ambient_delta_zoom.py` for a complete,
working one. Each keyframe sets a time, a form, a view (`center`/`scale`
in the disk's or halfplane's own coordinates — no aspect-ratio padding
here, frames are always full-bleed), a style, and colormap. Between two
consecutive keyframes the view zooms geometrically (constant *perceived*
zoom speed, exactly like a Mandelbrot zoom) with eased (smootherstep)
pacing; if the two keyframes differ in form/style/colormap, their renders
crossfade across the same span too. A "held" shot is just two adjacent
keyframes with identical parameters.

```python
from modforms.video import Keyframe, builtin_cmap

DELTA = {"source": "delta", "n_terms": 400}

TIMELINE = [
    Keyframe(time=0.0, form=DELTA, region="disk", center=(0, 0), scale=1.02,
             style="colormap-phase-contour", style_kwargs={"base": 2.0, "cmap": builtin_cmap("cividis")}),
    Keyframe(time=20.0, form=DELTA, region="disk", center=(0.05, 0.85), scale=0.01,
             style="colormap-phase-contour", style_kwargs={"base": 2.0, "cmap": builtin_cmap("twilight")}),
]
```

**Before committing to a long render**, always check `--dry-run` first —
it prints the frame count and a time estimate from a few sample frames,
so you're not guessing at how long a render will actually take. The
bottleneck is the Python/NumPy evaluation of the form, not video
encoding, so resolution matters a lot. Render a `--preview` draft first to
check the composition and timing, then commit to a full-resolution
overnight/background render for the real thing.

**Performance**: frames render in parallel across CPU cores by default
(`--workers N` to control this, `--workers 1` to force sequential) —
since every frame is an independent computation, this is close to a free
speedup (same output, just distributed; verified bit-for-bit identical
against sequential rendering). `--precision single` additionally switches
the core math from complex128 to complex64, measured roughly 3x faster
on the dominant cost (the q-expansion evaluation), at a real accuracy
cost: complex64 has ~7 significant digits vs complex128's ~15-16, so
Fourier terms underflow into rounding error much sooner, meaning deep
zooms show truncation artifacts at a noticeably shallower depth (measured
at roughly 6x shallower for a given term count) than at the default
double precision. `Timeline.check_safety()` accounts for whichever
precision you pass it. Good uses for single precision: fast previews, or
any part of a timeline that doesn't zoom especially deep; not recommended
for the golden-ratio-style deep-zoom passages.

**Zoom depth is not literally infinite.** Unlike Mandelbrot's escape-time
iteration, a modular form's truncated q-expansion doesn't generate
open-ended new detail forever — past a point (governed by float64
precision and how many Fourier terms you evaluated with) truncation
artifacts appear instead of finer structure. `Timeline.check_safety()`
(also run automatically by the `video` CLI command) gives a rough,
non-blocking warning when a keyframe's zoom looks too deep for its term
count.

**Pick an irrational zoom target for sustained detail.** Zooming toward a
specific *rational* boundary point (an actual single cusp) eventually
zooms *inside* that cusp's own horoball, where the form is smooth and the
image goes flat/boring — verified by rendering it. Zooming toward an
*irrational* point instead keeps encountering infinitely many nearby
smaller cusps (a rational's denominator can always be improved), so
detail keeps appearing at every scale — the disk boundary's Ford-circle
tiling made visible. The golden ratio, `(sqrt(5) - 1) / 2 ≈ 0.618`, is
the standard choice for this (its continued fraction `[0; 1, 1, 1, ...]`
converges slowest, giving the most uniform, persistent nesting) — used as
the target in the "Golden cusp zoom" UI preset. In the halfplane model
this is just `center=(0.618, y)` with `y` shrinking toward 0; cuspforms (like
Delta) are what show this richness at all — Eisenstein series approach a
nonzero constant at cusps instead of vanishing, so they go flat/boring
near any boundary point regardless; look for their interesting structure
at their actual zeros instead.

**Installing ffmpeg**: Linux (`apt install ffmpeg` / your package
manager), macOS (`brew install ffmpeg`), Windows (`winget install
ffmpeg` or `choco install ffmpeg`, or download from
[ffmpeg.org](https://ffmpeg.org/download.html) and add it to PATH).

**Which forms transition well.** The built-in Eisenstein series (E4, E6,
E8, E10, E14) and Delta are all level 1, so they share the same single
cusp orbit and periodicity — morphing between them (e.g. Delta → E4 → E6)
feels like the same underlying skeleton changing texture/density rather
than a structural jump, and is the smoothest family to drift through
continuously. Switching to a different level (e.g. an LMFDB import like
`105.2.a.a`) is a bigger structural change — more cusp classes, different
boundary rhythm — better suited to a deliberate "chapter break" than
continuous drifting.

## Styles

Each corresponds to a subsection of the paper:

| `--style` | Paper section | Description |
|---|---|---|
| `standard` | 2.2.1 | hue = phase, brightness = magnitude (0=black) |
| `magnitude` | 2.2.2 | grayscale, magnitude only, no phase |
| `periodic-linear` | 2.2.3 | hue = magnitude mod 1 (LMFDB-style medallions) |
| `periodic-log` | 2.2.4 | hue = log(magnitude) mod 1 |
| `phase` | 2.2.5 | hue = phase only, magnitude ignored |
| `phase-contour` | 2.2.6 | phase plot with magnitude contours (Wegert-style) |
| `colormap-phase` | 3 | phase plot using an arbitrary matplotlib colormap |
| `colormap-phase-contour` | 3 | colormap phase plot + magnitude contours |
| `colormap-magnitude` | 3 | colormap applied to (periodic) magnitude |
| `colormap-standard` | 3 | colormap hue + magnitude-as-brightness |

Every style is a plain function `f(values, **kwargs) -> RGB array` in
`modforms/coloring.py`, so it's easy to add your own.

## Built-in forms

- `delta(n_terms=400)` — the Ramanujan Delta function (weight 12, level 1).
  Computed exactly via the pentagonal number theorem, no external data
  needed.
- `eisenstein(k, n_terms=400)` for `k in {4, 6, 8, 10, 14}` — level-1
  Eisenstein series, also computed exactly.

The paper's other three running examples (`g`, weight 4 level 5; `f_105`,
weight 2 level 105; `f_10`, weight 20 level 10) are LMFDB newforms with
non-trivial coefficients — see "Importing from the LMFDB" below for the
easiest way to pull those in directly, or `examples/custom_form_example.py`
for loading coefficients by hand with `QExpansion.from_list`/`from_csv`.

## Importing from the LMFDB

`modforms.lmfdb` talks to the [LMFDB's public API](https://www.lmfdb.org/api/)
to browse and import newforms directly, so you don't have to copy
coefficients by hand. This is also available in the UI (`streamlit run
app.py`, pick "LMFDB" as the form source) if you'd rather browse and load
by dropdown/search box.

Browse forms by level/weight:

```bash
python -m modforms.cli search --level 105 --weight 2
```

```python
from modforms import lmfdb
lmfdb.print_newforms(level=105, weight=2)
for doc in lmfdb.search_newforms(level=105, weight=2):
    print(doc["label"], doc["dim"])
```

Fetch a form's q-expansion and plot it directly, using either the
Galois-orbit label from the paper (e.g. `5.4.a.a`, which assumes the first
embedding) or a fully embedded label (`5.4.a.a.1.1`):

```bash
python -m modforms.cli plot --lmfdb 5.4.a.a --region disk --style phase-contour --out g.png
```

```python
from modforms import lmfdb, plotting

g = lmfdb.fetch_qexpansion("5.4.a.a", n_terms=400)
plotting.plot_form(g, region="disk", style="phase-contour", out="g.png")
```

`fetch_qexpansion`'s parsing has been confirmed against a live response
(`105.2.a.a.1.1`): documents come back as `{"data": [...]}`, and
coefficients live in `an_normalized` — a list of `[re, im]` pairs starting
at n=1 that are Hecke-normalized (divided by `n**((weight-1)/2)`), which
`fetch_qexpansion` un-normalizes using the document's own `weight` field.
`search_newforms` (the `mf_newforms` collection) hasn't specifically been
checked against a live response, but uses the same defensive pattern: if
a query ever returns a document shaped differently than expected, you get
an `LMFDBSchemaError` showing the actual keys found instead of a bare
`KeyError`.

## How it works

`modforms/grid.py` builds a grid of points on the upper halfplane (a box
`[x0,x1] x [y0,y1]`) or on the Poincare disk (via the Moebius transform
`phi(w) = (1 - i*w) / (w - i)` from Section 2.1, masking points outside the
unit disk). `modforms/qexpansion.py` evaluates a truncated q-expansion
`sum a_n q^n` on that grid with Horner's method. `modforms/coloring.py`
turns the resulting complex array into RGB pixels for each style, using a
from-scratch vectorized RGB<->HSL conversion (`modforms/hsl.py`) to
implement the magnitude-contour trick from Remark 2 of the paper: a
sawtooth in `log_base(|f|) mod 1` is used as a lightness adjustment, so
contours appear as sharp light/dark transitions without ever being
computed explicitly.

## Layout

```
app.py            Streamlit UI (streamlit run app.py)
src/modforms/
  grid.py         halfplane/disk grids, disk<->halfplane Moebius map
  qexpansion.py   evaluate a truncated q-expansion on a grid
  forms.py        built-in forms (Delta, Eisenstein series)
  hsl.py          vectorized RGB <-> HSL
  coloring.py     the plotting styles (Sections 2 and 3 of the paper)
  plotting.py     evaluate_on_region / render / save_png / plot_form
  presets.py      the regions used in the paper's figures
  lmfdb.py        import/browse newforms from the LMFDB's API
  reproduce.py    PNG reproduce-metadata: build it, read it back, replot
  video.py        ambient zoom video timelines (Keyframe, Timeline, render_video)
  cli.py          command-line interface (plot / search / replot / video)
examples/
  video_ambient_delta_zoom.py   a complete, working video timeline
tests/
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The UI tests (`tests/test_app.py`) exercise `app.py` headlessly via
Streamlit's `AppTest` and are skipped automatically if `streamlit` isn't
installed (install with `pip install -e ".[ui]"` to include them).
