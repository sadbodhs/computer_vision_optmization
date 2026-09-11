#!/usr/bin/env python3
"""A/B the OUTPUT BINDING precision: FP32 (the default) vs FP16.

Every transport number in docs/model-zoo.md assumes an FP32 output binding,
because that is what trtexec writes by default. But the binding format is a free
choice, and the engines are already 88-98% FP16 internally - so an FP16 output
binding should halve the bytes AND may remove a reformat on the way out.

Whether it costs accuracy is a separate question this does not answer: it
changes the wire format, not the compute, so the values are the same numbers
TensorRT already computed, rounded to the precision it already used internally.

Usage: python3 io_precision.py <model> <out.tsv>
"""
import os
import re
import subprocess
import sys

import onnx

MODEL, OUT = sys.argv[1], sys.argv[2]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
ONNX = "/models/_exp/%s.onnx" % MODEL

g = onnx.load(ONNX, load_external_data=False).graph
n_out = len(g.output)
n_in = len(g.input)


def run(tag, extra):
    eng = "/tmp/io_%s_%s.plan" % (MODEL, tag)
    r = subprocess.run([TRTEXEC, "--onnx=" + ONNX, "--fp16", "--saveEngine=" + eng] + extra,
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    if "PASSED" not in log:
        err = [l for l in log.splitlines() if re.search(r"[Ee]rror|failed", l)][-2:]
        return None, " | ".join(err)[:160]

    def mean(label):
        m = re.search(re.escape(label) + r".*?mean = ([\d.]+) ms", log)
        return float(m.group(1)) if m else float("nan")

    qps = float(re.search(r"Throughput: ([\d.]+) qps", log).group(1))
    os.remove(eng)
    return dict(qps=qps, gpu=mean("GPU Compute Time:"), h2d=mean("H2D Latency:"),
                d2h=mean("D2H Latency:")), None


base, err = run("fp32out", [])
if err:
    print("%-18s baseline FAILED: %s" % (MODEL, err)); sys.exit(0)

fp16, err = run("fp16out", ["--inputIOFormats=" + ",".join(["fp32:chw"] * n_in),
                            "--outputIOFormats=" + ",".join(["fp16:chw"] * n_out)])
if err:
    print("%-18s fp16-out FAILED: %s" % (MODEL, err)); sys.exit(0)

for tag, d in (("fp32", base), ("fp16", fp16)):
    open(OUT, "a").write("\t".join([MODEL, tag, "%.1f" % d["qps"], "%.4f" % d["gpu"],
                                    "%.4f" % d["h2d"], "%.4f" % d["d2h"],
                                    "%.2f" % (100 * (d["h2d"] + d["d2h"]) /
                                              (d["gpu"] + d["h2d"] + d["d2h"]))]) + "\n")

sp = base["d2h"] / fp16["d2h"] if fp16["d2h"] else float("nan")
print("%-18s d2h %7.4f -> %7.4f ms (%.2fx)   qps %7.1f -> %7.1f (%+.1f%%)"
      % (MODEL, base["d2h"], fp16["d2h"], sp, base["qps"], fp16["qps"],
         100 * (fp16["qps"] / base["qps"] - 1)))
