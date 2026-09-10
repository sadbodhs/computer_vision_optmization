#!/usr/bin/env python3
"""Regenerate the report figures from committed data.

Three figures, chosen because the SHAPE carries the insight. Everything else in
the study stays a table, where the values carry it -- a bar chart of four numbers
is decoration, and a bar chart of 0.47348 vs 0.47376 (the mAP comparison) would
be actively misleading either way you scale the axis.

  fig1  latency vs throughput   the Pareto frontier: the study's core tradeoff
  fig2  concurrency scaling     where each flow plateaus, against the engine cap
  fig3  stage decomposition     where per-frame time actually goes

Figures use a transparent background and mid-grey furniture so they stay legible
in both the light and dark themes of the docs site.

Usage:  python3 scripts/make_plots.py [repo_root]
Output: docs/img/*.png
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(REPO, "docs", "img")
os.makedirs(OUT, exist_ok=True)

FG = "#888888"          # readable on both light and dark backgrounds
ENGINE_CAP = 1028       # trtexec batch-1 YOLOv8s, docs/methodology.md

# Colour per flow; the "good" flows get the saturated colours.
COLORS = {
    "A1": "#9aa0a6", "A2": "#e8710a", "B1": "#9aa0a6", "B2": "#1a73e8",
    "C1": "#9aa0a6", "C2": "#12a150", "D": "#a142f4",
}
LABELS = {
    "A1": "A1  C++ TRT, CPU path", "A2": "A2  C++ TRT, full-CUDA",
    "B1": "B1  Triton + C++, raw gRPC", "B2": "B2  Triton + C++, CUDA-shm",
    "C1": "C1  Triton + PyTorch", "C2": "C2  Triton + numpy, sys-shm",
    "D": "D  Triton async, batch-8",
}
ORDER = ["A2", "D", "B2", "C2", "A1", "B1", "C1"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=FG, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=FG, fontsize=10)
    ax.set_ylabel(ylabel, color=FG, fontsize=10)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color(FG)
        s.set_alpha(0.4)
    ax.grid(True, alpha=0.15, color=FG, linewidth=0.6)
    ax.set_facecolor("none")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


def load_capacity():
    rows = defaultdict(list)
    with open(os.path.join(REPO, "results", "capacity_table.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["flow"]].append((int(r["concurrency"]), float(r["fps"]),
                                    float(r["lat_p50_ms"])))
    for k in rows:
        rows[k].sort()
    return rows


def fig_pareto(cap):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for flow in ORDER:
        pts = cap[flow]
        lat = [p[2] for p in pts]
        fps = [p[1] for p in pts]
        ax.plot(lat, fps, "-o", color=COLORS[flow], label=LABELS[flow],
                linewidth=1.8, markersize=5, alpha=0.95)
        ax.annotate(flow, (lat[0], fps[0]), textcoords="offset points",
                    xytext=(-14, 4), color=COLORS[flow], fontsize=9, fontweight="bold")
    ax.axvline(33.3, color=FG, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(34.5, 150, "30 FPS frame budget\n(33.3 ms)", color=FG, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "The tradeoff: latency vs throughput\n(each line is one flow swept over concurrency 1-16)",
          "per-frame latency p50 (ms, log scale)  -  lower is better",
          "throughput (fps)  -  higher is better")
    leg = ax.legend(fontsize=8, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(FG)
    save(fig, "pareto-latency-throughput.png")


def fig_scaling(cap):
    fig, ax = plt.subplots(figsize=(9, 5))
    for flow in ORDER:
        pts = cap[flow]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                color=COLORS[flow], label=LABELS[flow], linewidth=1.8, markersize=5)
    ax.axhline(ENGINE_CAP, color=FG, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(16, ENGINE_CAP - 75, f"batch-1 engine cap  {ENGINE_CAP} qps",
            color=FG, fontsize=8, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "Throughput scaling with concurrency",
          "concurrent streams", "throughput (fps)")
    # outside the axes: inside, it covered the B1/C1 lines
    leg = ax.legend(fontsize=8, frameon=False, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0))
    for t in leg.get_texts():
        t.set_color(FG)
    save(fig, "concurrency-scaling.png")


def fig_stages():
    data = defaultdict(dict)
    with open(os.path.join(REPO, "results", "stage_decomposition.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            data[r["flow"]][r["stage"]] = float(r["ms"])
    flows = ["A2", "B2", "C2"]
    stages = [("host_to_device", "H2D transfer", "#e8710a"),
              ("inference", "inference + transport", "#1a73e8"),
              ("postprocess", "postprocess (NMS)", "#12a150")]
    fig, ax = plt.subplots(figsize=(8, 4))
    left = [0.0] * len(flows)
    for key, label, colour in stages:
        vals = [data[f].get(key, 0.0) for f in flows]
        ax.barh(flows, vals, left=left, label=label, color=colour, height=0.55)
        for i, v in enumerate(vals):
            if v >= 0.5:
                ax.text(left[i] + v / 2, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    for i, total in enumerate(left):
        ax.text(total + 0.08, i, f"{total:.2f} ms", va="center", fontsize=9, color=FG)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.18)
    style(ax, "Where per-frame time goes (capacity mode, concurrency 1)",
          "milliseconds per frame", "")
    # below the axes: inside, it covered the C2 bar
    leg = ax.legend(fontsize=8, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.5, -0.18), ncol=3)
    for t in leg.get_texts():
        t.set_color(FG)
    save(fig, "stage-decomposition.png")


if __name__ == "__main__":
    cap = load_capacity()
    fig_pareto(cap)
    fig_scaling(cap)
    fig_stages()
