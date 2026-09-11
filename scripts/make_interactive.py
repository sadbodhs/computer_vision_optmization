#!/usr/bin/env python3
"""Emit the interactive version of the latency-vs-throughput chart.

Same data as the static PNG (results/capacity_table.tsv), so the two cannot
disagree. Interactivity earns its place on this particular chart for three
reasons the static one cannot solve:

  * seven series overlap in the 1-2 ms cluster -> click the legend to isolate one
  * the plot shows shape but not values      -> hover for fps, latency, concurrency
  * D's 74 ms tail compresses the interesting -> box-zoom into 1-4 ms
    region, which is why the x axis is log

Plotly is loaded from CDN; nothing is vendored. GitHub markdown does not execute
JavaScript, so this is rendered on the docs site only - scripts/prepare_site.py
swaps the PNG for this at build time, and the repo view keeps the PNG.

Usage:  python3 scripts/make_interactive.py [repo_root]
Output: docs/img/pareto_interactive.html
"""
import csv
import json
import os
import sys
from collections import defaultdict

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(REPO, "docs", "img", "pareto_interactive.html")

COLORS = {
    "A1": "#9aa0a6", "A2": "#e8710a", "B1": "#9aa0a6", "B2": "#1a73e8",
    "C1": "#9aa0a6", "C2": "#12a150", "D": "#a142f4",
}
LABELS = {
    "A1": "A1 · C++ TRT, CPU path", "A2": "A2 · C++ TRT, full-CUDA",
    "B1": "B1 · Triton + C++, raw gRPC", "B2": "B2 · Triton + C++, CUDA-shm",
    "C1": "C1 · Triton + PyTorch", "C2": "C2 · Triton + numpy, sys-shm",
    "D": "D · Triton async, batch-8",
}
ORDER = ["A2", "D", "B2", "C2", "A1", "B1", "C1"]

rows = defaultdict(list)
with open(os.path.join(REPO, "results", "capacity_table.tsv")) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows[r["flow"]].append((int(r["concurrency"]), float(r["fps"]),
                                float(r["lat_p50_ms"])))
for k in rows:
    rows[k].sort()

traces = []
for flow in ORDER:
    pts = rows[flow]
    traces.append({
        "x": [p[2] for p in pts],
        "y": [p[1] for p in pts],
        "text": ["concurrency %d" % p[0] for p in pts],
        "name": LABELS[flow],
        "mode": "lines+markers",
        "type": "scatter",
        "line": {"color": COLORS[flow], "width": 2},
        "marker": {"color": COLORS[flow], "size": 8},
        "hovertemplate": ("<b>%s</b><br>%%{text}<br>"
                          "%%{y:.0f} fps<br>%%{x:.2f} ms p50<extra></extra>") % flow,
    })

layout = {
    "title": {"text": "Latency vs throughput — drag to zoom, click the legend to isolate a flow",
              "font": {"size": 15}},
    "xaxis": {"title": {"text": "per-frame latency p50 (ms, log) — lower is better"},
              "type": "log", "gridcolor": "rgba(136,136,136,0.2)",
              "zeroline": False,
              "tickvals": [1, 2, 5, 10, 20, 50], "ticktext": ["1", "2", "5", "10", "20", "50"]},
    "yaxis": {"title": {"text": "throughput (fps) — higher is better"},
              "gridcolor": "rgba(136,136,136,0.2)", "zeroline": False},
    "shapes": [{"type": "line", "x0": 33.3, "x1": 33.3, "yref": "paper", "y0": 0, "y1": 1,
                "line": {"color": "#888", "width": 1, "dash": "dash"}}],
    "annotations": [{"x": 1.523, "y": 0.05, "xref": "x", "yref": "paper",
                     "text": "30 FPS budget (33.3 ms)", "showarrow": False,
                     "font": {"size": 10, "color": "#888"}, "xanchor": "left"}],
    "hovermode": "closest",
    "legend": {"font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 70, "r": 20, "t": 55, "b": 60},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#888"},
}

HTML = """<!doctype html>
<meta charset="utf-8">
<title>Latency vs throughput</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  /* The iframe is a separate document: without color-scheme it defaults to
     light and the UA paints an opaque white canvas under `transparent`,
     which shows as a white box on the site's dark theme. */
  html, body { margin: 0; padding: 0; background: transparent;
               color-scheme: light dark; }
  #chart { width: 100%%; height: 520px; }
</style>
<div id="chart"></div>
<script>
  Plotly.newPlot("chart", %s, %s,
                 {responsive: true, displaylogo: false,
                  modeBarButtonsToRemove: ["lasso2d", "select2d"]});
</script>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML % (json.dumps(traces), json.dumps(layout)))
print("wrote", os.path.relpath(OUT, REPO))
