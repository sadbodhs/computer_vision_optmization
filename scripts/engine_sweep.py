#!/usr/bin/env python3
"""Build an FP16 engine from an ONNX file and decompose its per-frame cost.

Runs INSIDE the container. A2 is hardcoded to a 640x640 80-class YOLO head
(IMG/NUM_CLASSES/NUM_ANCHORS in cpp/src/main_cuda.cu), so the pipeline harness
cannot carry a classifier or a segmentation model. trtexec can, and it reports
the split this sweep actually needs:

    H2D  -> cost of getting the input in     (scales with input size)
    GPU  -> the engine itself                 (the denominator)
    D2H  -> cost of getting the output out    (scales with OUTPUT size)

That D2H column is the whole reason for including 4 KB classifiers and 31 MB
segmentation heads in one table: it measures the transport axis directly,
across three orders of magnitude of output, without needing a bespoke pipeline
for each architecture.

These are ENGINE-LEVEL numbers, not A2 pipeline numbers. They are comparable to
each other and to `trtexec` figures elsewhere in the repo; they are NOT
comparable to the A2 columns in results/v3/model_scaling.tsv.

Usage: python3 engine_sweep.py <onnx> <name> <out.tsv>

NOTE ON WHERE TO PUT EXPORTS
----------------------------
Do NOT write ONNX exports under triton/models/ (mounted as /models). That path
is Triton's --model-repository, and Triton treats EVERY subdirectory of it as a
model. A directory of .onnx files has no version subdirectory, so the load
fails, and Triton's default --exit-on-error=true then takes the whole server
down. The failure is delayed and confusing: the running server is unaffected,
so nothing breaks until the next container restart, which may be days later and
for an unrelated reason.

Use a path outside the model repository (this host uses
/home/suchi/sadbodh/model_exports). Engines with a 1/model.plan and no
config.pbtxt are fine - TensorRT plans auto-complete.
"""
import os
import re
import subprocess
import sys

import onnx

ONNX, NAME, OUT = sys.argv[1:4]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
ENGINE = "/tmp/%s.plan" % NAME

DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8}


def shape_of(v):
    return [d.dim_value or 1 for d in v.type.tensor_type.shape.dim]


def nbytes(v):
    n = 1
    for d in shape_of(v):
        n *= d
    return n * DTYPE_BYTES.get(v.type.tensor_type.elem_type, 4)


g = onnx.load(ONNX).graph
in_shape = shape_of(g.input[0])
in_bytes = nbytes(g.input[0])


def engine_io(log):
    """Read output shapes from trtexec's own build log.

    Two earlier approaches were wrong. Reading dims from the ONNX graph yields 1
    per symbolic dim_param, which reported Depth-Anything's ~1 MB output as 0 KB
    while its measured D2H was 0.046 ms - the number this sweep exists to
    measure, quietly wrong. Deserializing the engine with the python bindings
    fails outright: this container has trtexec 10.7 but `import tensorrt` 11.3,
    so the runtime returns None. The build log cannot disagree with the engine
    it just described, and trtexec writes fp32 outputs, which is what D2H moves.
    """
    total, shapes = 0, []
    for dims in re.findall(r"Output binding for \S+ with dimensions ([\dx]+) is created", log):
        shp = [int(d) for d in dims.split("x")]
        n = 1
        for d in shp:
            n *= d
        total += n * 4            # trtexec output format is fp32:CHW
        shapes.append(dims)
    return total, ";".join(shapes)


r = subprocess.run([TRTEXEC, "--onnx=" + ONNX, "--fp16", "--saveEngine=" + ENGINE],
                   capture_output=True, text=True)
log = r.stdout + r.stderr
if "PASSED" not in log:
    tail = [l for l in log.splitlines() if re.search(r"error|Error|failed", l)][-3:]
    print("%s\tBUILD_FAILED\t%s" % (NAME, " | ".join(tail) or "see log"))
    open(OUT, "a").write("%s\tFAILED\t\t\t\t\t\t\t\t\n" % NAME)
    sys.exit(0)


def mean_of(label):
    m = re.search(re.escape(label) + r".*?mean = ([\d.]+) ms", log)
    return float(m.group(1)) if m else float("nan")


out_bytes, out_shapes = engine_io(log)
qps = float(re.search(r"Throughput: ([\d.]+) qps", log).group(1))
h2d, gpu, d2h = mean_of("H2D Latency:"), mean_of("GPU Compute Time:"), mean_of("D2H Latency:")
lat = mean_of("Latency:")
non_engine = h2d + d2h

row = [NAME, "x".join(str(d) for d in in_shape), out_shapes,
       "%d" % in_bytes, "%d" % out_bytes,
       "%.1f" % qps, "%.4f" % gpu, "%.4f" % h2d, "%.4f" % d2h,
       "%.4f" % non_engine, "%.2f" % (100.0 * non_engine / (gpu + non_engine))]
open(OUT, "a").write("\t".join(row) + "\n")
print("%-20s %9.1f qps  gpu %7.3f  h2d %6.3f  d2h %7.3f  out %8.1f KB  transport %5.1f%%"
      % (NAME, qps, gpu, h2d, d2h, out_bytes / 1024.0,
         100.0 * non_engine / (gpu + non_engine)))
os.remove(ENGINE)
