#!/usr/bin/env python3
"""Figure + summary for the cross-architecture sweep.

The model-cost ladder (docs/model-scaling.md) held architecture constant and
varied cost, and found a fixed 0.256 ms of non-engine work. This sweep does the
opposite - it varies architecture - and finds that the "fixed" cost is not a
constant of the pipeline at all. It is output bytes divided by PCIe bandwidth,
and output shape is an architectural choice that has nothing to do with how
expensive the model is.

Usage: python3 scripts/plot_zoo.py [repo_root]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
TSV = os.path.join(REPO, "results", "v3", "model_zoo.tsv")
OUT = os.path.join(REPO, "docs", "img", "model-zoo.png")

TASK = {
    "resnet50": "classification", "efficientnet_b0": "classification",
    "efficientnet_v2_s": "classification",
    "yolo11n": "detection", "yolo11s": "detection", "yolo11m": "detection",
    "yolo11l": "detection", "yolo11x": "detection", "yolov8s": "detection",
    "rtdetr-l": "detection", "yolov8s-worldv2": "detection",
    "yolo11n-seg": "segmentation", "yolo11s-seg": "segmentation",
    "segformer_b0": "segmentation", "segformer_b2": "segmentation",
    "segformer_b5": "segmentation", "unet_r34": "segmentation",
    "deeplabv3_mnv3": "segmentation",
    "dinov2_l": "backbone/dense", "depth_anything_v2_l": "backbone/dense",
    "sam_vit_b_enc": "backbone/dense", "sam_vit_h_enc": "backbone/dense",
}
COLOR = {"classification": "#12a150", "detection": "#e8710a",
         "segmentation": "#1a73e8", "backbone/dense": "#a142f4"}

rows = []
failed = []
for r in csv.DictReader(open(TSV), delimiter="\t"):
    if not r.get("gpu_ms") or r.get("input") == "FAILED":
        failed.append(r["model"])
        continue
    rows.append(dict(m=r["model"], gpu=float(r["gpu_ms"]), h2d=float(r["h2d_ms"]),
                     d2h=float(r["d2h_ms"]), ob=int(r["out_bytes"]),
                     pct=float(r["transport_pct"]), task=TASK.get(r["model"], "other")))

print("%-22s %-15s %8s %8s %10s %8s %9s"
      % ("model", "task", "gpu_ms", "d2h_ms", "out_KB", "transp%", "GB/s"))
for r in sorted(rows, key=lambda z: z["gpu"]):
    bw = r["ob"] / r["d2h"] / 1e6 if r["d2h"] > 0 else float("nan")
    print("%-22s %-15s %8.3f %8.4f %10.1f %7.1f%% %9.1f"
          % (r["m"], r["task"], r["gpu"], r["d2h"], r["ob"] / 1024, r["pct"], bw))
if failed:
    print("\nbuild failed: %s" % ", ".join(failed))

big = [r for r in rows if r["ob"] > 500_000]
bw = sum(r["ob"] / r["d2h"] for r in big) / len(big) / 1e6
print("\nD2H bandwidth over outputs >500 KB: %.1f GB/s (n=%d)" % (bw, len(big)))
print("output size spans %.0fx; engine time spans %.0fx"
      % (max(r["ob"] for r in rows) / min(r["ob"] for r in rows),
         max(r["gpu"] for r in rows) / min(r["gpu"] for r in rows)))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

for task in COLOR:
    pts = [r for r in rows if r["task"] == task]
    if not pts:
        continue
    a1.scatter([r["ob"] / 1024 for r in pts], [r["d2h"] for r in pts],
               s=46, color=COLOR[task], label=task, zorder=3)
xs = [4, 40000]
a1.plot(xs, [x * 1024 / (bw * 1e6) for x in xs], "--", color="#888", lw=1,
        label="%.0f GB/s" % bw, zorder=2)
a1.annotate("below ~1 MB a copy is\nlatency-bound, not\nbandwidth-bound",
            xy=(9, 0.0042), xytext=(30, 0.0012), fontsize=8, color="#666",
            arrowprops=dict(arrowstyle="->", color="#999", lw=.8))
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("output size (KB, log)"); a1.set_ylabel("D2H time (ms, log)")
a1.set_title("Transport cost is output size / PCIe bandwidth\nnothing else", fontsize=11)
a1.grid(alpha=.25, which="both"); a1.legend(fontsize=8)

order = sorted(rows, key=lambda z: -z["pct"])
a2.barh([r["m"] for r in order], [r["pct"] for r in order],
        color=[COLOR[r["task"]] for r in order])
a2.axvline(10, ls="--", lw=1, color="#888")
a2.invert_yaxis()
a2.set_xlabel("transport as % of frame time (H2D + D2H)")
a2.set_title("Same task, same engine cost, opposite verdict\n"
             "DeepLabV3 57% vs SegFormer-B0 14%", fontsize=11)
a2.tick_params(axis="y", labelsize=7.5)
a2.grid(alpha=.25, axis="x")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", os.path.relpath(OUT, REPO))
