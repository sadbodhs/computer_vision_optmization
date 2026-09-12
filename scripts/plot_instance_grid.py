#!/usr/bin/env python3
"""Figure for the instance-count grid.

triton-tuning.md swept instance_group count at one operating point (B2, four
streams) and concluded "count: 2 was the right call". Across the concurrency
range and across both flows, the real finding is a contrast: instances buy B2
a third more throughput and buy D almost nothing, because D already has
batching doing the same job.

Usage: python3 scripts/plot_instance_grid.py [repo_root]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
TSV = os.path.join(REPO, "results", "v3", "instance_grid.tsv")
OUT = os.path.join(REPO, "docs", "img", "instance-grid.png")

d = defaultdict(dict)
for r in csv.DictReader(open(TSV), delimiter="\t"):
    d[r["flow"]][(int(r["instances"]), int(r["concurrency"]))] = (
        float(r["fps"]), float(r["mean_batch"]))

COUNTS = [1, 2, 4]
CONC = [1, 4, 16]
COL = {1: "#9aa0a6", 4: "#1a73e8", 16: "#e8710a"}

print("%-4s %-12s %8s %8s %8s   %s" % ("flow", "concurrency", "count=1", "count=2", "count=4", "gain 1->4"))
for flow in ("B2", "D"):
    for n in CONC:
        v = [d[flow][(c, n)][0] for c in COUNTS]
        print("%-4s %-12d %8.1f %8.1f %8.1f   %+6.1f%%"
              % (flow, n, v[0], v[1], v[2], 100 * (v[2] / v[0] - 1)))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)

for ax, flow, title in ((a1, "B2", "B2 — no batching"), (a2, "D", "D — dynamic batching")):
    for n in CONC:
        ax.plot(COUNTS, [d[flow][(c, n)][0] for c in COUNTS], "o-", lw=2,
                color=COL[n], label="concurrency %d" % n)
    ax.set_xticks(COUNTS)
    ax.set_xlabel("instance_group count")
    ax.set_ylabel("throughput (fps)")
    g = [100 * (d[flow][(4, n)][0] / d[flow][(1, n)][0] - 1) for n in CONC]
    ax.set_title("%s\ngain from 1→4 instances: %s"
                 % (title, ", ".join("%+.0f%%" % x for x in g)), fontsize=11)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1800)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", os.path.relpath(OUT, REPO))

print("\nmean batch formed by D (instances and batch size trade off):")
for n in CONC:
    print("  concurrency %-3d " % n + "  ".join(
        "count=%d: %.2f" % (c, d["D"][(c, n)][1]) for c in COUNTS))
