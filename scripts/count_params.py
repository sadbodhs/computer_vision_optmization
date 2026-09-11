#!/usr/bin/env python3
"""Add parameter counts and input-pixel counts to the model zoo table.

Reads dims from the ONNX initializers WITHOUT loading the weight data - a
TensorProto carries its dims even when the payload lives in external files,
which matters because SAM ViT-H's weights are 2.4 GB of extensionless blobs.

Usage: python3 count_params.py <tsv> <onnx_dir>
"""
import csv
import os
import sys

import onnx

TSV, ONNXDIR = sys.argv[1], sys.argv[2]

rows = list(csv.DictReader(open(TSV), delimiter="\t"))
out = []
for r in rows:
    p = os.path.join(ONNXDIR, r["model"] + ".onnx")
    params = ""
    if os.path.exists(p):
        g = onnx.load(p, load_external_data=False).graph
        n = 0
        for init in g.initializer:
            k = 1
            for d in init.dims:
                k *= d
            n += k
        params = str(n)
    r["params"] = params
    # input pixels: the other half of "how much work is this"
    px = ""
    if r.get("input") and "x" in r["input"]:
        d = [int(v) for v in r["input"].split("x")]
        if len(d) == 4:
            px = str(d[2] * d[3])
    r["input_px"] = px
    out.append(r)

cols = list(rows[0].keys())
with open(TSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
    w.writeheader()
    w.writerows(out)

print("%-22s %12s %10s %10s %12s" % ("model", "params", "Mparams", "in_px", "gpu_ms"))
for r in out:
    if not r["params"]:
        continue
    print("%-22s %12s %10.1f %10s %12s"
          % (r["model"], r["params"], int(r["params"]) / 1e6, r["input_px"],
             r["gpu_ms"] or "FAILED"))
