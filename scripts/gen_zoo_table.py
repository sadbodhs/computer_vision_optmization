#!/usr/bin/env python3
"""Render the full model table into docs/model-zoo.md from the TSV.

The page had only excerpt tables, so every model's parameter count and output
size lived in the raw data and nowhere a reader would look. This generates the
complete listing between two markers, so the page can never disagree with
results/v3/model_zoo.tsv.

Usage: python3 scripts/gen_zoo_table.py [repo_root]
"""
import csv
import io
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
TSV = os.path.join(REPO, "results", "v3", "model_zoo.tsv")
DOC = os.path.join(REPO, "docs", "model-zoo.md")
BEGIN, END = "<!-- BEGIN zoo-table -->", "<!-- END zoo-table -->"

NAME = {
    "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_v2_s": "EfficientNetV2-S", "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s", "yolo11m": "YOLO11m", "yolo11l": "YOLO11l",
    "yolo11x": "YOLO11x", "yolov8s": "YOLOv8s", "rtdetr-l": "RT-DETR-L",
    "yolov8s-worldv2": "YOLOv8s-World", "yolo11n-seg": "YOLO11n-seg",
    "yolo11s-seg": "YOLO11s-seg", "segformer_b0": "SegFormer-B0",
    "segformer_b2": "SegFormer-B2", "segformer_b5": "SegFormer-B5",
    "unet_r34": "U-Net-R34", "deeplabv3_mnv3": "DeepLabV3-MNv3",
    "dinov2_l": "DINOv2-L", "depth_anything_v2_l": "Depth Anything V2-L",
    "sam_vit_b_enc": "SAM ViT-B (encoder)", "sam_vit_h_enc": "SAM ViT-H (encoder)",
    "ssdlite_mnv3": "SSDLite-MNv3", "maskrcnn_r50": "Mask R-CNN R50",
}
TASK = {
    "resnet50": "cls", "efficientnet_b0": "cls", "efficientnet_v2_s": "cls",
    "yolo11n": "det", "yolo11s": "det", "yolo11m": "det", "yolo11l": "det",
    "yolo11x": "det", "yolov8s": "det", "rtdetr-l": "det",
    "yolov8s-worldv2": "det (open-vocab)", "ssdlite_mnv3": "det",
    "yolo11n-seg": "seg (inst)", "yolo11s-seg": "seg (inst)",
    "maskrcnn_r50": "seg (inst)",
    "segformer_b0": "seg", "segformer_b2": "seg", "segformer_b5": "seg",
    "unet_r34": "seg", "deeplabv3_mnv3": "seg",
    "dinov2_l": "backbone", "depth_anything_v2_l": "depth",
    "sam_vit_b_enc": "promptable", "sam_vit_h_enc": "promptable",
}


def kb(n):
    n = int(n) / 1024.0
    return "%.1f MB" % (n / 1024) if n >= 1024 else "%.1f KB" % n


rows = list(csv.DictReader(open(TSV), delimiter="\t"))
ok = [r for r in rows if r.get("gpu_ms")]
bad = [r for r in rows if not r.get("gpu_ms")]
ok.sort(key=lambda r: float(r["gpu_ms"]))

out = ["| Model | Task | Params | Input | Output | Engine | Transport |",
       "|---|---|---:|---|---:|---:|---:|"]
for r in ok:
    m = r["model"]
    out.append("| **%s** | %s | %.1f M | %s | %s | %.3f ms | %.1f%% |"
               % (NAME.get(m, m), TASK.get(m, "-"), int(r["params"]) / 1e6,
                  "x".join(r["input"].split("x")[2:]), kb(r["out_bytes"]),
                  float(r["gpu_ms"]), float(r["transport_pct"])))
for r in bad:
    m = r["model"]
    out.append("| *%s* | %s | %.1f M | — | — | *build failed* | — |"
               % (NAME.get(m, m), TASK.get(m, "-"), int(r["params"]) / 1e6))

table = "\n".join(out)
t = io.open(DOC, encoding="utf-8").read()
assert BEGIN in t and END in t, "markers missing in %s" % DOC
pre, rest = t.split(BEGIN, 1)
_, post = rest.split(END, 1)
io.open(DOC, "w", encoding="utf-8").write(pre + BEGIN + "\n\n" + table + "\n\n" + END + post)
print("wrote %d measured + %d failed rows into %s" % (len(ok), len(bad), DOC))
