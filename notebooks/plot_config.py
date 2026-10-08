"""Central figure configuration for the paper figures produced in ``notebooks/``.

Everything that decides how a figure *looks* -- column widths, font sizes, line widths,
colors, colormaps, output directory -- lives here, so the notebooks only decide *what*
to draw and how tall each panel grid should be.

This module holds the *shared specification* only. Anything that names one experiment's
arms, methods or band limits (per-arm colors, per-arm labels) belongs in that experiment's
notebook, next to the code that defines the arms -- see the config cell of
``collapse_experiment.ipynb``. The palette below (:data:`COLORS`) is the exception: its keys
are semantic roles that mean the same thing in every notebook.

Typical use inside a notebook::

    import plot_config as pc

    pc.apply_style()                                     # once, in the imports cell
    fig, axes = pc.subplots(2, 3, aspect=0.46)           # double column by default
    ...
    pc.save(fig, "toy1_rank_realizations.pdf")

Key conventions
---------------
* **Width** is never given in inches by a notebook: it is either the single-column
  (``columns="single"``, 3.25 in) or the double-column (``columns="double"``, 6.75 in)
  text width of the paper.
* **Height** stays a per-figure choice, but as the ``aspect`` fraction of the width
  (``height = aspect * width``), so a figure keeps its proportions if the widths change.
* Figures use ``layout="constrained"`` and are saved *without* ``bbox_inches="tight"``,
  so the PDF on disk has exactly the requested paper size and needs no rescaling in
  LaTeX (include it with ``\\includegraphics[width=\\linewidth]{...}``).

To customize, either mutate the module-level ``CONFIG`` before calling
:func:`apply_style`, or pass a modified copy::

    pc.CONFIG.tick_size = 6.5                            # global tweak
    pc.apply_style()

    cfg = pc.CONFIG.replace(font_size=9.0)               # one-off variant
    fig, ax = pc.subplots(1, 1, aspect=0.7, cfg=cfg)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the "science" / "no-latex" / "grid" styles

__all__ = [
    "PlotConfig",
    "CONFIG",
    "FIG_DIR",
    "PRESENTATION_DIR",
    "apply_style",
    "width_of",
    "figsize",
    "subplots",
    "save",
    "COLORS",
    "LORA_PALETTE",
    "MAP_COLORS",
    "diverging_cmap",
    "sequential_cmap",
    "temperature_cmap",
    "grey_ramp",
]


# --------------------------------------------------------------------------------------
# output location
# --------------------------------------------------------------------------------------
def _find_project_root(start: Optional[Path] = None) -> Path:
    """Nearest ancestor containing ``.project-root`` (falls back to this file's parent)."""
    start = Path.cwd() if start is None else Path(start)
    for parent in [start, *start.parents]:
        if (parent / ".project-root").exists():
            return parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import configured_path  # noqa: E402

#: Checkpoints and derived diagnostics (``paths.results_dir``).
RESULTS_DIR = configured_path("results_dir")
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
#: Slide variants of paper figures (re-arranged grids, metric subsets). Git-ignored, so
#: nothing written here can be mistaken for -- or overwrite -- a paper figure.
PRESENTATION_DIR = FIG_DIR / "presentation"


# --------------------------------------------------------------------------------------
# the config object
# --------------------------------------------------------------------------------------
@dataclass
class PlotConfig:
    """Every knob that the paper figures share. All sizes are in points, widths in inches."""

    # --- canvas -----------------------------------------------------------------------
    single_column_width: float = 3.25   # \columnwidth  of the paper template
    double_column_width: float = 6.75   # \textwidth    of the paper template
    default_columns: str = "double"      # "single" | "double" -- what subplots() assumes

    # --- style ------------------------------------------------------------------------
    style: Tuple[str, ...] = ("science", "grid")
    font_family: str = "serif"

    # --- font sizes -------------------------------------------------------------------
    font_size: float = 9.0          # base text size (rcParams["font.size"])
    title_size: float = 9.0         # per-axes titles
    suptitle_size: float = 10.0      # figure.suptitle
    label_size: float = 9.0         # x/y axis labels
    tick_size: float = 8.0          # tick labels (also used for colorbar ticks)
    legend_size: float = 8.5        # legend entries
    annotation_size: float = 7.5    # in-axes text / annotate / gridline labels
    colorbar_label_size: float = 8.5

    # --- lines, markers, spines -------------------------------------------------------
    line_width: float = 1.5         # default data line width
    thick_line_width: float = 2.0   # reference curves (truth / empirical)
    thin_line_width: float = 1.0    # guide lines (axhline, axvline, coastlines)
    marker_size: float = 2.0        # observation dots
    axes_line_width: float = 0.5
    tick_width: float = 0.5
    grid_alpha: float = 0.35
    band_alpha: float = 0.25        # fill_between uncertainty bands

    # --- legend box -------------------------------------------------------------------
    legend_frameon: bool = True         # every legend gets a box ...
    legend_fancybox: bool = True        # ... with rounded corners
    legend_framealpha: float = 0.85     # slightly translucent, so data underneath stays visible
    legend_facecolor: str = "white"
    legend_edgecolor: str = "0.7"
    legend_border_width: float = 0.5    # frame line width; matches the axes spines

    # --- constrained-layout padding --------------------------------------------------
    # Keep the edge pads at >= ~0.04 in: below that, panel titles and tick labels of
    # fixed-aspect axes (e.g. cartopy GeoAxes) get clipped by the canvas border.
    layout_h_pad: float = 0.05      # figure edge -> axes decorations, inches
    layout_w_pad: float = 0.05
    layout_hspace: float = 0.02     # space between rows,    fraction of subplot height
    layout_wspace: float = 0.02     # space between columns, fraction of subplot width

    # --- colormaps (resolved lazily so a missing `cmap` package degrades gracefully) ---
    # Three roles with their own entry, so any one of them can be re-pointed without
    # disturbing the others: signed fields (covariance, correlation, anomalies,
    # standardized residuals) use `diverging`, non-negative magnitudes use `sequential`,
    # and *absolute* 2-m temperature uses `temperature` -- the one physical quantity that
    # several notebooks plot, currently on the same vik ramp as the signed fields.
    diverging_cmap: str = "crameri:vik"        # covariance / correlation maps (signed)
    sequential_cmap: str = "crameri:batlow"    # magnitudes: marginal std, log power
    temperature_cmap: str = "crameri:vik"   # absolute t2m fields (cold blue -> warm red)
    diverging_cmap_fallback: str = "RdBu_r"
    sequential_cmap_fallback: str = "viridis"
    temperature_cmap_fallback: str = "RdYlBu_r"

    # --- saving -----------------------------------------------------------------------
    fig_dir: Path = field(default_factory=lambda: FIG_DIR)
    savefig_dpi: int = 400          # only affects rasterized artists inside the PDF
    savefig_format: str = "pdf"
    savefig_transparent: bool = False

    # ---------------------------------------------------------------------------------
    def replace(self, **changes) -> "PlotConfig":
        """A copy with ``changes`` applied (leaves the module-level ``CONFIG`` alone)."""
        return dataclasses.replace(self, **changes)

    def rc_params(self) -> Dict[str, object]:
        """The rcParams this config implies, applied on top of the scienceplots style."""
        return {
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "figure.titlesize": self.suptitle_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "legend.title_fontsize": self.legend_size,
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "axes.linewidth": self.axes_line_width,
            "xtick.major.width": self.tick_width,
            "ytick.major.width": self.tick_width,
            "xtick.minor.width": self.tick_width * 0.75,
            "ytick.minor.width": self.tick_width * 0.75,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": self.axes_line_width,
            "figure.figsize": (self.double_column_width, 0.4 * self.double_column_width),
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": self.layout_h_pad,
            "figure.constrained_layout.w_pad": self.layout_w_pad,
            "figure.constrained_layout.hspace": self.layout_hspace,
            "figure.constrained_layout.wspace": self.layout_wspace,
            "savefig.dpi": self.savefig_dpi,
            "savefig.format": self.savefig_format,
            "savefig.transparent": self.savefig_transparent,
            "savefig.bbox": "standard",   # NOT "tight": that would change the paper size
            "image.cmap": self.sequential_cmap_fallback,
            "legend.frameon": self.legend_frameon,
            "legend.fancybox": self.legend_fancybox,
            "legend.framealpha": self.legend_framealpha,
            "legend.facecolor": self.legend_facecolor,
            "legend.edgecolor": self.legend_edgecolor,
            "patch.linewidth": self.legend_border_width,   # what the legend frame picks up
            "axes.axisbelow": True,
            "pdf.fonttype": 42,           # editable/embeddable TrueType, not Type-3
            "ps.fonttype": 42,
        }


CONFIG = PlotConfig()


def _cfg(cfg: Optional[PlotConfig]) -> PlotConfig:
    return CONFIG if cfg is None else cfg


# --------------------------------------------------------------------------------------
# workaround: the usetex math minus sign disappears from saved PDFs
# --------------------------------------------------------------------------------------
def _patch_type1_minus() -> bool:
    """Stop matplotlib from dropping the math minus when it subsets a usetex Type-1 font.

    ``matplotlib._type1font.Type1Font.subset`` resets slot 0 of the font encoding to
    ``.notdef`` unconditionally. In the Computer Modern symbol fonts (CMSY*) slot 0 *is*
    the math minus, so subsetting deletes the only glyph we asked for: the embedded font
    comes out with an empty ``/CharSet`` and every ``$10^{-6}$``, every negative exponent
    in a log tick label and every ``$-x$`` renders as a blank in the PDF. Inline (Agg)
    output is unaffected -- it rasterizes the dvi directly and never subsets -- which is
    why the figures look correct in the notebook and wrong on disk.

    This reinstalls the method with slot 0 preserved when it is actually used. It is a
    no-op if matplotlib no longer contains the offending line.

    Returns whether the fix is in place.
    """
    try:
        import inspect

        from matplotlib import _type1font
    except Exception:                                    # pragma: no cover - defensive
        return False

    original = getattr(_type1font.Type1Font, "subset", None)
    if original is None:
        return False
    if getattr(original, "_minus_fix", False):           # already patched in this session
        return True
    try:
        if "encoding[0] = '.notdef'" not in inspect.getsource(original):
            return True                                  # upstream fixed it: leave alone
    except OSError:                                      # pragma: no cover - no source
        return False

    def subset(self, characters, name_prefix):
        characters = frozenset(characters)
        encoding = {code: glyph
                    for code, glyph in self.prop["Encoding"].items()
                    if code in characters}
        encoding.setdefault(0, ".notdef")                # <-- the fix: do not overwrite
        todo = set(encoding.values()) | {".notdef"}      # .notdef stays in the subset
        done, seen_subrs = set(), {0, 1, 2, 3}
        while todo:
            glyph = todo.pop()
            called_glyphs, called_subrs = _type1font._CharstringSimulator(self).run(glyph)
            todo.update(called_glyphs - done)
            seen_subrs.update(called_subrs)
            done.add(glyph)
        newparts = self._replace(
            [(x, f'/FontName /{name_prefix}{self.prop["FontName"]} def')
             for x in self._pos["FontName"]]
            + [(self._pos["CharStrings"][0], self._subset_charstrings(done)),
               (self._pos["Subrs"][0], self._subset_subrs(seen_subrs)),
               (self._pos["Encoding"][0], self._postscript_encoding(encoding))]
            + [(x, "") for x in self._pos.get("UniqueID", [])]
        )
        return type(self)((
            newparts[0],
            self._encrypt(newparts[1], "eexec"),
            self.parts[2],
        ))

    subset.__doc__ = original.__doc__
    subset._minus_fix = True
    _type1font.Type1Font.subset = subset
    return True


#: True once the usetex minus-sign workaround above is active (applied by `apply_style`).
USETEX_MINUS_FIX = False


# --------------------------------------------------------------------------------------
# style application and sizing
# --------------------------------------------------------------------------------------
def apply_style(cfg: Optional[PlotConfig] = None, **overrides) -> PlotConfig:
    """Activate the scienceplots style plus this config's rcParams, globally.

    ``overrides`` are applied to a copy of ``cfg`` (default: the module-level ``CONFIG``)
    and the copy is returned, so ``cfg = pc.apply_style(tick_size=6)`` works.

    Also installs the usetex minus-sign workaround (see :func:`_patch_type1_minus`), since
    the ``science`` style turns ``text.usetex`` on and every saved PDF is affected.
    """
    global USETEX_MINUS_FIX
    cfg = _cfg(cfg).replace(**overrides) if overrides else _cfg(cfg)
    plt.style.use(list(cfg.style))
    mpl.rcParams.update(cfg.rc_params())
    USETEX_MINUS_FIX = _patch_type1_minus()
    return cfg


def width_of(columns: Union[str, float, None] = None, cfg: Optional[PlotConfig] = None) -> float:
    """Figure width in inches. ``columns`` is "single", "double", or an explicit width."""
    cfg = _cfg(cfg)
    if columns is None:
        columns = cfg.default_columns
    if isinstance(columns, (int, float)):
        return float(columns)
    key = str(columns).lower()
    if key in ("single", "1", "one", "column"):
        return cfg.single_column_width
    if key in ("double", "2", "two", "full", "text"):
        return cfg.double_column_width
    raise ValueError(f"columns must be 'single', 'double' or a width in inches, got {columns!r}")


def figsize(
    aspect: float,
    columns: Union[str, float, None] = None,
    cfg: Optional[PlotConfig] = None,
) -> Tuple[float, float]:
    """``(width, height)`` in inches, with ``height = aspect * width``.

    The width comes from the column layout; only ``aspect`` is a per-figure choice.
    """
    w = width_of(columns, cfg)
    return (w, aspect * w)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    aspect: float = 0.62,
    columns: Union[str, float, None] = None,
    cfg: Optional[PlotConfig] = None,
    **kwargs,
):
    """``plt.subplots`` with the paper width, an ``aspect``-scaled height and constrained layout.

    Any ``plt.subplots`` keyword still works; passing ``figsize`` or ``layout`` explicitly
    overrides the defaults computed here.
    """
    cfg = _cfg(cfg)
    kwargs.setdefault("figsize", figsize(aspect, columns, cfg))
    kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if kwargs["layout"] == "constrained" and fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(
            h_pad=cfg.layout_h_pad, w_pad=cfg.layout_w_pad,
            hspace=cfg.layout_hspace, wspace=cfg.layout_wspace,
        )
    return fig, axes


def save(fig, name: str, cfg: Optional[PlotConfig] = None, **kwargs) -> Path:
    """Save ``fig`` into the figure directory at *exactly* its paper size.

    No ``bbox_inches="tight"`` on purpose -- constrained layout already fits the labels
    inside the requested canvas, and a tight bbox would silently change the width.
    """
    cfg = _cfg(cfg)
    path = Path(name)
    if not path.is_absolute():
        path = cfg.fig_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)     # e.g. PRESENTATION_DIR on first use
    kwargs.setdefault("dpi", cfg.savefig_dpi)
    kwargs.setdefault("transparent", cfg.savefig_transparent)
    fig.savefig(path, **kwargs)
    return path


# --------------------------------------------------------------------------------------
# colors -- one palette shared by every figure in the paper
# --------------------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    # semantic roles (same meaning in every notebook)
    "kernel_score": "#1b9e77",   # anything trained with the kernel score
    "log_score": "#7b3294",      # anything trained with the raw log score / NLL
    "energy_score": "#e7298a",   # anything trained with the (fair) energy score
    "deterministic": "#0f7ba2",  # point-forecast baseline (MSE / Dirac: no covariance)
    "truth": "#555555",          # closed-form posterior / reference model
    "empirical": "#000000",      # data-derived reference curves
    "signal": "#333333",         # the true signal / observed field
    "observation": "#d73027",    # observation markers
    "guide": "#999999",          # axhline / axvline guides
    # low-rank family (ERA5), ordered by rank
    "lora": "#d95f02",
    "lora_light": "#fdae61",
    "lora_dark": "#a6761d",
}

#: Low-rank ERA5 arms, in the order of ``Config.lora_ranks`` (extend if more ranks are added).
LORA_PALETTE: Tuple[str, ...] = (COLORS["lora"], COLORS["lora_light"], "#e6ab02", COLORS["lora_dark"])

#: Geographic context on any map panel (cartopy features). Muted on purpose: the data
#: colors carry the signal, the geography is background. Shared so the ERA5 grid maps and
#: the station-network maps sit on the same-looking Earth.
MAP_COLORS: Dict[str, str] = {
    "ocean": "#dbe7f0",
    "land": "#f5f3f0",
    "coast": "0.35",
    "border": "0.55",
    "graticule": "0.6",
}


# --------------------------------------------------------------------------------------
# colormaps
# --------------------------------------------------------------------------------------
def _resolve_cmap(name: str, fallback: str):
    """``cmap``-package colormap (e.g. "crameri:vik"), or a matplotlib fallback."""
    try:
        from cmap import Colormap

        return Colormap(name).to_mpl()
    except Exception:
        return plt.get_cmap(fallback)


def diverging_cmap(cfg: Optional[PlotConfig] = None):
    """Signed fields: covariance and correlation maps (zero-centered, perceptually uniform)."""
    cfg = _cfg(cfg)
    return _resolve_cmap(cfg.diverging_cmap, cfg.diverging_cmap_fallback)


def sequential_cmap(cfg: Optional[PlotConfig] = None):
    """Non-negative magnitudes: marginal standard deviation, log power spectra."""
    cfg = _cfg(cfg)
    return _resolve_cmap(cfg.sequential_cmap, cfg.sequential_cmap_fallback)


def temperature_cmap(cfg: Optional[PlotConfig] = None):
    """Absolute 2-m temperature, in K or degC (observations, predictive means, draws).

    Configured separately from :func:`diverging_cmap` (anomalies, correlations,
    standardized residuals) so the temperature ramp can be changed for every notebook at
    once without touching what signed fields look like.
    """
    cfg = _cfg(cfg)
    return _resolve_cmap(cfg.temperature_cmap, cfg.temperature_cmap_fallback)


def grey_ramp(lo: float = 0.72, hi: float = 0.08, name: str = "pc_grey_ramp"):
    """Light-to-dark grey ramp for *strength* encoded on non-data artists.

    Network edges, links or annotations whose weight should read as "more" without
    competing with the data colors: ``lo``/``hi`` are the grey levels (1 = white,
    0 = black) at the low and high end. Neither end is pure white or pure black, so a
    weak edge stays visible on the map background and a strong one does not read as a
    frame line.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, [str(lo), str(hi)])
