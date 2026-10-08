"""Per-task results tables (EUPPBench, ETTh1/2, ETTm1/2, illness, weather,
electricity, traffic, ERA5, Depth).

One table per data set, in the layout of the EUPPBench table, rows in the order
Det, Diag, LorD, Chol, SB (physical space) and SCORE (spectral space), whose
correlation core is fitted with the kernel score after a stage-1 CRPS fit. Diag,
LorD and Chol appear twice, the log-score fit and right below it the
kernel-score fit. ETT tables stack up to four blocks (uni/multivariate x
T96/T192), each ranked on its own; a block is left out when its results are
missing. Illness, weather, electricity and traffic are multivariate only, so
their blocks are labelled by the horizon alone.

Columns follow the order of the figures: MSE, CRPS, ES, VS, KS, NLL/d. Two rows
are equally good on a metric when their seed mean ± sd intervals overlap as
printed (`scored`, `tied`); a cell is bold when no other row of its (block,
metric) beats it, and equally good rows share their rank (`shared_ranks`). Each
metric ranks the rows that report it, and a row's mean rank is over the metrics
it reports (NLL/d is missing for the two reference rows), as in the
cross-data-set `main_rank` table; the best mean rank of a block is bold. Scores
are printed to `DIGITS` decimal places, the mean rank to `RANK_DIGITS`. A cell
whose printed sd exceeds its printed mean in magnitude is unstable: it is
reported, ties no stable cell and is ranked behind every one of them. There is
no cutoff on the value itself (`mrt.NLL_PER_D_LIMIT` is lifted); a log score of
four or more integer digits is only *printed* in scientific notation.

    python notebooks/make_task_tables.py            # writes tables/task_*.tex

ERA5 and Depth come from W&B via the results-table notebook's pickle cache
(`.cache_results_tables.pkl`, refresh it there); EUPPBench and the time series
are read from the results directory (`paths.results_dir`).
"""

import math
import pickle
import sys
from decimal import Decimal
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))
sys.path.insert(0, str(NB_DIR.parent))
import make_results_tables as mrt  # noqa: E402
from src.utils.paths import configured_path  # noqa: E402

SHARED = configured_path("results_dir")
CACHE = NB_DIR / ".cache_results_tables.pkl"
OUT = NB_DIR / "tables"

METRICS = ["mse", "crps", "energy_score", "variogram_score", "kernel_score",
           "nll_per_d"]
LABEL = {"mse": "MSE", "crps": "CRPS", "energy_score": "ES",
         "variogram_score": "VS", "kernel_score": "KS", "nll_per_d": r"NLL/$d$"}

#: Decimal places of every score cell: one precision across all the paper's tables.
DIGITS = 3
#: Decimal places of the mean-rank column.
RANK_DIGITS = 1

#: A log score this large (in magnitude) is printed as `(m ± s) × 10^e`. A print
#: format only: whether a cell competes is `scored`'s stability test, not this.
SCI_ABOVE = 1e3

#: Depth's MSE, CRPS, VS and KS are ~1e-3, which `DIGITS` would print as 0.002 in
#: every row; they are shown multiplied by 1e3 instead, and the caption says so.
DEPTH_SCALE = {"mse": 1e3, "crps": 1e3, "variogram_score": 1e3,
               "kernel_score": 1e3}


def _row(label, score, method, rank=None, objective=None, stage1=None,
         prefix=False):
    def match(c):
        if (not c.method.startswith(method)) if prefix else c.method != method:
            return False
        if (c.rank or None) != rank:
            return False
        if stage1 is not None and c.stage1 != stage1:
            return False
        return c.objective() == objective
    return dict(label=label, score=score, match=match)


#: Each covariance baseline twice, its kernel-score fit right below its
#: log-score fit.
PHYS = [
    _row("Det", "MSE", "deterministic", objective="mse"),
    _row("Diag", "NLL", "phys_diag", objective="nll"),
    _row("Diag", "KS", "phys_diag", objective="kernel_score"),
    _row(r"LorD $r{=}1$", "NLL", "phys_lora", rank="1", objective="nll"),
    _row(r"LorD $r{=}1$", "KS", "phys_lora", rank="1", objective="kernel_score"),
    _row("Chol", "NLL", "phys_cholesky", objective="nll"),
    _row("Chol", "KS", "phys_cholesky", objective="kernel_score"),
    _row("SB", "KS", "generative", objective="kernel_score"),
]

#: SCORE, whichever `whitened_*` core the task runs (Fourier, or the graph
#: Fourier transform with a Matern filter on EUPPBench).
SPECTRAL = [_row("SCORE", "CRPS+KS", "whitened_", objective="kernel_score",
                 stage1="crps", prefix=True)]


def scored(mean, sd, scale=1.0):
    """A `(mean, sd, unstable)` cell, at the precision the tables print it.

    Ties are judged on the printed numbers (`DIGITS` places after `scale`, as
    exact decimals), so every bold cell and rank can be checked against the
    table itself: two cells printing 0.026 ± 0.000 are equally good, whatever
    their fourth decimal. A missing sd (one seed) is zero.

    A cell is **unstable** when its sd exceeds the magnitude of its mean: the
    seeds do not agree on the size of the score, so the mean says nothing and
    the cell must not be read as tying the best one (a fixed cutoff on |nll/d|
    would still let a wide cell such as +331.6 ± 413.0 tie the best log score).
    Only log scores qualify in practice: the five proper scores are
    non-negative with seed spreads far below their means.
    """
    sd = sd or 0.0
    m = Decimal(f"{mean * scale:.{DIGITS}f}")
    s = Decimal(f"{sd * scale:.{DIGITS}f}")
    return (m, s, s > abs(m))


def tied(a, b):
    """Whether two `scored` cells count as equally good.

    They do when their mean ± sd intervals overlap, `|m_a - m_b| <= s_a + s_b`:
    0.500 ± 0.020 ties 0.470 ± 0.020. An unstable cell ties no stable one,
    however wide its sd -- that width is exactly what would otherwise let it tie
    every row of its column (Depth's SCORE at (3.7 ± 8.0) x 10^3).
    """
    if a[2] != b[2]:
        return False
    return abs(a[0] - b[0]) <= a[1] + b[1]


def beats(a, b):
    """Whether cell `a` is better than cell `b`: lower, and not tied with it.

    A stable cell beats every unstable one.
    """
    return not tied(a, b) and (a[2], a[0]) < (b[2], b[0])


def shared_ranks(cells):
    """Ranks, 1 = best, with equally good rows sharing their places.

    `cells` holds a `(mean, sd, unstable)` per row, or None where the row does
    not report the metric; that row is not ranked and gets None. A row's rank is
    one plus the rows that beat it plus half the rows it ties with, so a tie
    splits the places it spans: two rows sharing first get (1 + 2) / 2 = 1.5,
    exactly the average rank of an exact tie. Unlike equality, overlap is not
    transitive (a ~ b and b ~ c without a ~ c); counting pairs keeps the rank
    defined there, and a pool's ranks still sum to n (n + 1) / 2.
    """
    ranks = [None] * len(cells)
    for i, a in enumerate(cells):
        if a is None:
            continue
        others = [b for j, b in enumerate(cells) if j != i and b is not None]
        ranks[i] = (1 + sum(beats(b, a) for b in others)
                    + 0.5 * sum(tied(a, b) for b in others))
    return ranks


def unbeaten(cells):
    """Indices of the rows no other row beats: the bold cells of a column.

    That is the lowest mean and every row tied with it, less any row that
    another tied row beats in turn.
    """
    present = [(i, a) for i, a in enumerate(cells) if a is not None]
    return {i for i, a in present
            if not any(beats(b, a) for j, b in present if j != i)}


def cell(stats, metric):
    """The `scored` cell of one metric of `aggregate`'s stats, or None."""
    mean, sd, _ = stats[metric]
    if mean is None:
        return None
    return scored(mean, sd)


def aggregate(entries, d):
    """`mrt.aggregate` with every seed kept.

    `mrt`'s guard blanks a configuration's whole log-score cell once a seed is
    past `mrt.NLL_PER_D_LIMIT`; these tables report what was measured instead,
    and `scored` decides whether the cell competes.
    """
    limit, mrt.NLL_PER_D_LIMIT = mrt.NLL_PER_D_LIMIT, math.inf
    try:
        return mrt.aggregate(entries, d)
    finally:
        mrt.NLL_PER_D_LIMIT = limit


def _sci(mean, sd):
    """A huge log score as `(mean ± sd) × 10^e`, `e` the exponent of the mean."""
    e = math.floor(math.log10(abs(mean)))
    body = f"{mean / 10 ** e:+.{DIGITS}f}"
    if sd is not None:
        body = f"({body} \\pm {sd / 10 ** e:.{DIGITS}f})"
    return rf"${body} \times 10^{{{e}}}$"


def block_rows(grouped, d, sections):
    """[(section, [(label, score, stats)])] for the rows present in `grouped`."""
    out = []
    for name, rows in sections:
        found = []
        for r in rows:
            hits = [v for v in grouped.values() if r["match"](v[0]["cfg"])]
            if len(hits) > 1:
                raise ValueError(f"{r['label']} matches {len(hits)} configurations")
            if hits:
                found.append((r["label"], r["score"], aggregate(hits[0], d)))
        if found:
            out.append((name, found))
    return out


def scaled(sections, scale):
    """`sections` with each metric in `scale` (mean and sd) multiplied by its factor."""
    def one(stats):
        out = dict(stats)
        for m, k in scale.items():
            mean, sd, n = stats[m]
            if mean is not None:
                out[m] = (mean * k, None if sd is None else sd * k, n)
        return out
    return [(name, [(label, score, one(stats)) for label, score, stats in rs])
            for name, rs in sections]


def render_block(sections, indent=""):
    rows = [row for _, rs in sections for row in rs]
    column = {m: [cell(s, m) for _, _, s in rows] for m in METRICS}
    digits = {m: DIGITS for m in METRICS}
    best = {m: unbeaten(column[m]) for m in METRICS}
    per_metric = {m: shared_ranks(column[m]) for m in METRICS}
    mean_rank = []
    for i in range(len(rows)):
        own = [per_metric[m][i] for m in METRICS if per_metric[m][i] is not None]
        mean_rank.append(sum(own) / len(own))
    best_rank = round(min(mean_rank), RANK_DIGITS)

    lines, i = [], 0
    for si, (name, rs) in enumerate(sections):
        if si and not indent:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{9}}{{l}}{{{indent}\emph{{{name}}}}}\\")
        for label, score, stats in rs:
            cells = []
            for m in METRICS:
                mean, sd, n = stats[m]
                if mean is None:
                    cells.append("-")
                    continue
                if m == "nll_per_d" and abs(mean) >= SCI_ABOVE:
                    body = _sci(mean, sd)
                else:
                    body = mrt._fmt(mean, sd, n, digits[m],
                                    signed=m in mrt._SIGNED_COLUMNS)
                if i in best[m]:
                    body = r"$\mathbf{" + body.strip("$") + "}$"
                cells.append(body)
            rank = f"{mean_rank[i]:.{RANK_DIGITS}f}"
            if round(mean_rank[i], RANK_DIGITS) == best_rank:
                rank = r"\mathbf{" + rank + "}"
            pad = r"\qquad " if indent else ""
            lines.append(f"{pad}{label} & {score} & " + " & ".join(cells)
                         + f" & ${rank}$ \\\\")
            i += 1
    return lines


HEADER = [
    r"\begin{tabular}{llrrrrrrr}",
    r"\toprule",
    "Method & Objective & " + " & ".join(LABEL[m] + r" $\downarrow$" for m in METRICS)
    + r" & $\varnothing$ Rank $\downarrow$ \\",
    r"\midrule",
]


def wrap(body, caption, label):
    return "\n".join([
        r"\begin{table*}[ht]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\small", r"\adjustbox{max width=\linewidth}{%",
        *HEADER, *body, r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])


def caption(task, blocks=True, extra=""):
    """`blocks` is whether the table stacks several ranked blocks."""
    stats = "with mean and standard deviation over five seeds"
    head = (f"Test scores on {task}. Each block is ranked separately {stats}"
            if blocks else f"Test scores on {task}, {stats}")
    return (f"{head}; the best model within one standard deviation per metric"
            f" is bold.{extra}")


def single(grouped, d, task, label, extra="", scale=None):
    sections = block_rows(grouped, d, [("Physical space", PHYS),
                                       ("Spectral space", SPECTRAL)])
    if scale:
        sections = scaled(sections, scale)
    return wrap(render_block(sections), caption(task, blocks=False, extra=extra),
                label)


def stacked(ds, pretty, blocks, extra="", label=None, channels=7, show_mode=True):
    """Time-series table: one ranked block per (mode, horizon) in `blocks`.

    `channels` is the multivariate width, for the per-coordinate log score:
    7 for ETT and illness, 21 for weather, 321 for electricity, 862 for
    traffic.
    `show_mode=False` labels a block by its horizon alone, for the data sets
    that are multivariate only.
    """
    body = []
    for bi, (mode, T) in enumerate(blocks):
        grouped = mrt.fetch_dir(str(SHARED / "time_series" / ds / mode), f"T{T}")
        d = T * (channels if mode == "multivariate" else 1)
        sections = block_rows(grouped, d, [("Physical space", PHYS),
                                           ("Spectral space", SPECTRAL)])
        if bi:
            body.append(r"\cmidrule(l){1-9}")
        title = f"{mode.capitalize()}, $T = {T}$" if show_mode else f"$T = {T}$"
        body.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{title}}}}}\\")
        body += render_block(sections, indent=r"\quad")
    return wrap(body, caption(pretty, extra=extra), label or f"tab:{ds}")


def ett(ds, pretty):
    # A block appears only if its results exist.
    root = SHARED / "time_series" / ds
    blocks = [(m, T) for m in ("univariate", "multivariate") for T in (96, 192)
              if (root / m / f"T{T}").is_dir()]
    return stacked(ds, pretty, blocks)


def illness():
    return stacked("national_illness", r"National illness (multivariate, 7 series)",
                   [("multivariate", 24), ("multivariate", 36)],
                   label="tab:illness", show_mode=False)


def weather():
    return stacked("weather", r"Weather (multivariate, 21 series)",
                   [("multivariate", 96), ("multivariate", 192)],
                   label="tab:weather", channels=21, show_mode=False)


def electricity():
    return stacked("electricity", r"Electricity (multivariate, 321 series)",
                   [("multivariate", 96), ("multivariate", 192)],
                   label="tab:electricity", channels=321, show_mode=False)


def traffic():
    return stacked("traffic", r"Traffic (multivariate, 862 series)",
                   [("multivariate", 96), ("multivariate", 192)],
                   label="tab:traffic", channels=862, show_mode=False)


def main():
    cache = pickle.loads(CACHE.read_bytes())
    tables = {
        "euppbench": single(cache["euppbench"], 115,
                            r"EUPPBench station post-processing ($d=115$)",
                            "tab:euppbench"),
        "era5": single(cache["era5"], mrt.DIMENSION[("era5", None)],
                       r"ERA5 surface temperature ($d = 160 \times 220$)",
                       "tab:era5"),
        "depth": single(cache["depth"], mrt.DIMENSION[("depth_bts_crop", None)],
                        r"NYU Depth v2 (Eigen crop, $d = 426 \times 560$)",
                        "tab:depth",
                        extra=" MSE, CRPS, VS and KS are multiplied by "
                              "$10^{3}$.",
                        scale=DEPTH_SCALE),
        "etth1": ett("etth1", "ETTh1"),
        "etth2": ett("etth2", "ETTh2"),
        "ettm1": ett("ettm1", "ETTm1"),
        "ettm2": ett("ettm2", "ETTm2"),
        "illness": illness(),
        "weather": weather(),
        "electricity": electricity(),
        "traffic": traffic(),
    }
    OUT.mkdir(exist_ok=True)
    for name, tex in tables.items():
        (OUT / f"task_{name}.tex").write_text(tex)
        print(f"% ---- {name} ----\n{tex}")


if __name__ == "__main__":
    main()
