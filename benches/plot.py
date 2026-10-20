"""Plots the results of benches/voxelizer.rs, run by bench.sh.

Usage: python benches/plot.py <directory holding results.csv>

Writes next to the CSV:
  throughput-compress-off.png  points per second of every implementation
  throughput-compress-on.png   the same with compress
  mt-scaling.png               MultiThreaded speedup over SingleThreaded
  latency.png                  median and p95 duration of a frame
Runs that panicked have no numbers, so they are missing from the plots and
listed under them.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# One fixed color and marker per implementation, the same in every plot
IMPLS = {
    "st": ("SingleThreaded", "#2a78d6", "o"),
    "mt": ("MultiThreaded", "#eb6834", "s"),
    "wgpu": ("WgpuAccelerated", "#1baf7a", "^"),
}
# Blue ramp, light to dark, for ordered values (points per frame)
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281"]
NUMBERS = ["frames", "setup_ms", "mean_ms", "median_ms", "p95_ms", "min_ms",
           "max_ms", "mpoints_per_s", "msteps_per_s", "changed_per_frame"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 1,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.facecolor": SURFACE,
})


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["points"] = int(row["points"])
        row["range"] = int(row["range"])
        row["threads"] = int(row["threads"]) if row["threads"] else None
        for key in NUMBERS:
            row[key] = float(row[key]) if row[key] else None
    return rows


def compact(value, _=None):
    for divisor, suffix in ((1e6, "M"), (1e3, "K")):
        if value >= divisor:
            return f"{value / divisor:g}{suffix}"
    return f"{value:g}"


def label(row):
    return row["impl"] if row["threads"] is None else f"mt/{row['threads']}"


def grid(rows, title, subtitle):
    """A figure with one panel per scenario (rows) and range (columns)."""
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    ranges = sorted({row["range"] for row in rows})
    fig, axes = plt.subplots(
        len(scenarios), len(ranges), squeeze=False, layout="constrained",
        figsize=(1.0 + 3.6 * len(ranges), 1.4 + 2.8 * len(scenarios)))
    fig.suptitle(f"{title}\n{subtitle}", x=0.01, ha="left", fontsize=11)
    panels = []
    for r, scenario in enumerate(scenarios):
        for c, range_ in enumerate(ranges):
            ax = axes[r][c]
            ax.set_title(f"{scenario}, range {range_} voxels", loc="left")
            panel = [row for row in rows
                     if row["scenario"] == scenario and row["range"] == range_]
            panels.append((ax, panel, r == len(scenarios) - 1, c == 0))
    return fig, panels


def footnote(fig, rows):
    notes = []
    panicked = [row for row in rows if row["matches"] == "panicked"]
    if panicked:
        names = ", ".join(sorted({label(row) for row in panicked}))
        notes.append(f"{len(panicked)} runs panicked ({names}), their points are "
                     "missing, run.log has the message")
    mismatched = sum(row["matches"] == "no" for row in rows)
    if mismatched:
        notes.append(f"{mismatched} runs returned other voxels than the ref run")
    if notes:
        fig.text(0.01, 0, "\n".join(notes), ha="left", va="top",
                 color=INK_SECONDARY, transform=fig.transFigure)


def points_axis(ax, rows, bottom):
    points = sorted({row["points"] for row in rows})
    ax.set_xscale("log")
    ax.set_xticks(points)
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.xaxis.set_minor_locator(NullLocator())
    # Room for the markers at both ends
    ax.set_xlim(points[0] / 1.6, points[-1] * 1.6)
    if bottom:
        ax.set_xlabel("points per frame")


def throughput(rows, compress, out):
    rows = [row for row in rows if row["compress"] == compress]
    if not rows:
        return
    threads = sorted({row["threads"] for row in rows if row["threads"]})
    fig, panels = grid(
        rows, f"Throughput, compress {compress}",
        "million points per second, higher is better\nMultiThreaded at its best "
        f"thread count ({', '.join(map(str, threads))} tried)")
    handles = {}
    for ax, panel, bottom, left in panels:
        for impl, (name, color, marker) in IMPLS.items():
            best = {}
            for row in panel:
                if row["impl"] == impl and row["mpoints_per_s"] is not None:
                    best[row["points"]] = max(best.get(row["points"], 0),
                                              row["mpoints_per_s"])
            if not best:
                continue
            xs = sorted(best)
            handles[impl] = ax.plot(
                xs, [best[x] for x in xs], color=color, marker=marker,
                linewidth=2, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, solid_capstyle="round",
                solid_joinstyle="round", label=name)[0]
        points_axis(ax, panel, bottom)
        ax.set_ylim(bottom=0)
        if left:
            ax.set_ylabel("Mpoints/s")
    fig.legend(handles=list(handles.values()), loc="outside right upper")
    footnote(fig, rows)
    fig.savefig(out / f"throughput-compress-{compress}.png", bbox_inches="tight")
    plt.close(fig)


def mt_scaling(rows, out):
    rows = [row for row in rows if row["compress"] == "off"]
    st = {(row["scenario"], row["range"], row["points"]): row["mpoints_per_s"]
          for row in rows if row["impl"] == "st" and row["mpoints_per_s"]}
    mt = [row for row in rows if row["impl"] == "mt"]
    if not st or not mt:
        print("mt-scaling.png skipped: needs both st and mt runs")
        return
    points = sorted({row["points"] for row in mt})
    step = (len(RAMP) - 1) / max(len(points) - 1, 1)
    shades = {p: RAMP[round(i * step)] for i, p in enumerate(points)}
    fig, panels = grid(
        mt, "MultiThreaded scaling, compress off",
        "speedup over SingleThreaded, above 1 is faster")
    for ax, panel, bottom, left in panels:
        ax.axhline(1, color=BASELINE, linewidth=1)
        ax.text(1, 1, "SingleThreaded ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color=MUTED, fontsize=8)
        for p in points:
            runs = sorted(
                (row["threads"], row["mpoints_per_s"] / st[key])
                for row in panel
                if row["points"] == p and row["mpoints_per_s"]
                and (key := (row["scenario"], row["range"], p)) in st)
            if runs:
                ax.plot(*zip(*runs), color=shades[p], marker="o", linewidth=2,
                        markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, solid_capstyle="round",
                        solid_joinstyle="round")
        threads = sorted({row["threads"] for row in panel})
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(bottom=0)
        if bottom:
            ax.set_xlabel("threads")
        if left:
            ax.set_ylabel("speedup")
    handles = [Line2D([], [], color=shades[p], marker="o", linewidth=2,
                      markersize=7, markeredgecolor=SURFACE,
                      label=f"{compact(p)} points") for p in points]
    fig.legend(handles=handles, loc="outside right upper")
    footnote(fig, mt)
    fig.savefig(out / "mt-scaling.png", bbox_inches="tight")
    plt.close(fig)


def latency(rows, out):
    rows = [row for row in rows if row["compress"] == "off"]
    if not rows:
        return
    most = max(row["points"] for row in rows)
    rows = [row for row in rows if row["points"] == most]
    fig, panels = grid(
        rows, f"Frame duration, {compact(most)} points per frame, compress off",
        "median, the whisker reaches the 95th percentile; lower is better")
    for ax, panel, bottom, left in panels:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        labels = [label(row) for row in panel]
        ys = range(len(panel))
        right = max((row["p95_ms"] or 0 for row in panel), default=0) or 1
        for y, row in zip(ys, panel):
            if row["median_ms"] is None:
                ax.text(0, y, " panicked", va="center", color=MUTED)
                continue
            color = IMPLS[row["impl"]][1]
            ax.barh(y, row["median_ms"], height=0.55, color=color)
            ax.plot([row["median_ms"], row["p95_ms"]], [y, y],
                    color=INK_SECONDARY, linewidth=1)
            ax.text(row["p95_ms"] + right * 0.02, y, f"{row['median_ms']:.3g}",
                    va="center", color=INK_SECONDARY, fontsize=8)
        ax.set_yticks(list(ys), labels)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, right * 1.18)
        if bottom:
            ax.set_xlabel("ms per frame")
    handles = [Line2D([], [], color=color, linewidth=6, label=name)
               for impl, (name, color, _) in IMPLS.items()
               if any(row["impl"] == impl for row in rows)]
    fig.legend(handles=handles, loc="outside right upper")
    footnote(fig, rows)
    fig.savefig(out / "latency.png", bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    rows = load(out / "results.csv")
    if not rows:
        sys.exit("results.csv has no runs")
    throughput(rows, "off", out)
    throughput(rows, "on", out)
    mt_scaling(rows, out)
    latency(rows, out)
    print(f"plots in {out}")


if __name__ == "__main__":
    main()
