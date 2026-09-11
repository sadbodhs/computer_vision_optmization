#!/usr/bin/env python3
"""Figure + summary for the model-cost sweep.

The point of the plot is the flat line: the pipeline's non-engine work does not
grow with the model, so its SHARE of the frame collapses as the engine grows.
Two reference lines mark where that share drops under 10% - for A2, whose fixed
cost is the measured constant, and for B1, whose 1.15 ms of raw-gRPC transport
puts the same crossover four times further up the curve.

Usage: python3 scripts/plot_model_scaling.py [repo_root]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
TSV = os.path.join(REPO, "results", "v3", "model_scaling.tsv")
OUT = os.path.join(REPO, "docs", "img", "model-scaling.png")

B1_TRANSPORT_MS = 1.15   # docs/transport.md: cost of shipping [1,84,8400] over raw gRPC

rows = defaultdict(list)
with open(TSV) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows[r["model"]].append(r)

ORDER = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
summary = {}
for m, rs in rows.items():
    g = lambda k: st.mean(float(r[k]) for r in rs)
    summary[m] = dict(fps=g("fps"), p50=g("lat_p50_ms"), infer=g("infer_ms"),
                      h2d=g("h2d_ms"), nms=g("nms_ms"), ne=g("non_engine_ms"),
                      pct=g("non_engine_pct"),
                      ne_sd=st.pstdev([float(r["non_engine_ms"]) for r in rs]))

print("%-9s %8s %8s %8s %11s %7s" % ("model", "fps", "infer", "p50", "non-engine", "share"))
for m in ORDER + ["yolov8s"]:
    s = summary[m]
    print("%-9s %8.1f %8.3f %8.3f %11.4f %6.1f%%"
          % (m, s["fps"], s["infer"], s["p50"], s["ne"], s["pct"]))

allne = [summary[m]["ne"] for m in summary]
K = st.mean(allne)
print("\nnon-engine cost across the ladder: mean %.4f ms, sd %.4f (%.1f%% of mean)"
      % (K, st.pstdev(allne), 100 * st.pstdev(allne) / K))
print("engine span %.2fx   delivered-fps span %.2fx   -> %.0f%% of the model's"
      " advantage is eaten by fixed cost"
      % (summary["yolo11x"]["infer"] / summary["yolo11n"]["infer"],
         summary["yolo11n"]["fps"] / summary["yolo11x"]["fps"],
         100 * (1 - (summary["yolo11n"]["fps"] / summary["yolo11x"]["fps"])
                / (summary["yolo11x"]["infer"] / summary["yolo11n"]["infer"]))))
print("crossover (<10%% of frame): A2 engine > %.2f ms | B1 engine > %.2f ms"
      % (K * 9, B1_TRANSPORT_MS * 9))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

x = [summary[m]["infer"] for m in ORDER]
ax1.plot(x, [summary[m]["ne"] for m in ORDER], "o-", color="#e8710a", lw=2,
         label="A2 non-engine cost (measured)")
ax1.axhline(K, ls="--", lw=1, color="#888")
ax1.set_ylim(0, 0.40)
ax1.set_xlabel("engine time (ms)")
ax1.set_ylabel("non-engine cost per frame (ms)")
ax1.set_title("Fixed, not proportional\n%.3f ms ± %.3f across a %.1fx engine span"
              % (K, st.pstdev(allne), x[-1] / x[0]), fontsize=11)
for m, xi in zip(ORDER, x):
    ax1.annotate(m.replace("yolo11", ""), (xi, summary[m]["ne"]),
                 textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax1.grid(alpha=.25)
ax1.legend(fontsize=9, loc="lower right")

ax2.plot(x, [summary[m]["pct"] for m in ORDER], "o-", color="#e8710a", lw=2,
         label="A2 — zero-copy (%.3f ms fixed)" % K)
xs = [0.5 + i * 0.05 for i in range(280)]
ax2.plot(xs, [100 * B1_TRANSPORT_MS / (v + B1_TRANSPORT_MS) for v in xs],
         "-", color="#9aa0a6", lw=2, label="B1 — raw gRPC (1.15 ms transport)")
ax2.axhline(10, ls="--", lw=1, color="#888")
ax2.annotate("10% of the frame", (10, 10.8), fontsize=9, color="#888")
ax2.set_xscale("log")
ax2.set_xlabel("engine time (ms, log)")
ax2.set_ylabel("plumbing as % of frame time")
ax2.set_title("Where the plumbing stops mattering\nA2 at %.1f ms engine · B1 at %.1f ms"
              % (K * 9, B1_TRANSPORT_MS * 9), fontsize=11)
ax2.grid(alpha=.25)
ax2.legend(fontsize=9)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", os.path.relpath(OUT, REPO))
