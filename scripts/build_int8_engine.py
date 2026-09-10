#!/usr/bin/env python3
"""Build a CALIBRATED INT8 TensorRT engine that Triton can actually serve.

The obvious route -- ultralytics export(format="engine", int8=True) -- produces a
correctly calibrated engine that Triton then REFUSES to load:

    IRuntime::deserializeCudaEngine: Error Code 1: Serialization
    (Trying to load an engine created with incompatible serialization version.)

TensorRT engines are locked to the exact build that produced them. Ultralytics
pip-installs its own TensorRT to export (tensorrt_cu13 11.3 by default) while the
container serves with native 10.7.0.23. Pinning the pip package to
tensorrt==10.7.0.post1 does not help either: post1 and 10.7.0.23 are different
builds of the same version, and the serialization check is exact.

The route that works is two-stage, and this script runs both:

  1. ultralytics exports with int8=True. Keep its CALIBRATION CACHE -- a small
     text file of per-tensor scales, portable across builds of the same TensorRT
     version -- and its intermediate ONNX. Discard its engine.

  2. the container's OWN trtexec rebuilds from that ONNX and cache:

       trtexec --onnx=<int8 onnx> --int8 --fp16 --calib=<cache> --saveEngine=...

     --fp16 is required: the ultralytics ONNX carries fp16 layer precisions and
     the builder rejects the network without it.

Result: loads in Triton, +32.8% throughput for -1.55 mAP50-95. See
docs/precision.md and docs/accuracy.md.

Calibration images come from the same COCO val2017 directory the evaluation uses.
That overlap is fine for a speed-vs-accuracy delta but not for an absolute
accuracy claim -- noted in docs/accuracy.md.

Usage (inside the Triton container):
  python3 scripts/build_int8_engine.py --weights /models/yolov8s.pt \\
      --images /coco/val2017 --fraction 0.05 --out /models/yolov8s_int8
"""
import argparse
import os
import subprocess
import sys

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

CONFIG = '''name: "{name}"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 0
instance_group [ {{ count: 2 kind: KIND_GPU }} ]
input [ {{ name: "images" data_type: TYPE_FP32 dims: [ 1, 3, 640, 640 ] }} ]
output [ {{ name: "output0" data_type: TYPE_FP32 dims: [ 1, 84, 8400 ] }} ]
'''


def write_cal_yaml(path, images_root):
    """Minimal ultralytics dataset yaml for calibration (images only)."""
    import yaml
    from ultralytics.cfg import ROOT as CFG_ROOT
    src = os.path.join(CFG_ROOT, "cfg", "datasets", "coco.yaml")
    names = yaml.safe_load(open(src))["names"]
    root = os.path.dirname(images_root.rstrip("/"))
    leaf = os.path.basename(images_root.rstrip("/"))
    with open(path, "w") as f:
        f.write("path: %s\ntrain: %s\nval: %s\nnames:\n" % (root, leaf, leaf))
        for k, v in names.items():
            f.write("  %s: %s\n" % (k, v))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/models/yolov8s.pt")
    ap.add_argument("--images", default="/coco/val2017")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="fraction of the image set used for calibration")
    ap.add_argument("--out", default="/models/yolov8s_int8")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-calibration", action="store_true",
                    help="reuse an existing .cache and .int8.onnx next to the weights")
    args = ap.parse_args()

    stem = os.path.splitext(args.weights)[0]
    cache = stem + ".cache"
    onnx_int8 = stem + ".int8.onnx"

    # ---- stage 1: calibrate (ultralytics), keeping cache + ONNX ----
    if not args.skip_calibration:
        from ultralytics import YOLO
        cal_yaml = write_cal_yaml("/tmp/cal.yaml", args.images)
        print("calibrating on %s (fraction=%s)" % (args.images, args.fraction),
              file=sys.stderr)
        YOLO(args.weights).export(
            format="engine", int8=True, data=cal_yaml, imgsz=args.imgsz,
            batch=1, dynamic=False, workspace=8, verbose=False,
            fraction=args.fraction,
        )
    for f in (cache, onnx_int8):
        if not os.path.exists(f):
            sys.exit("expected %s from the ultralytics export; not found" % f)
    print("calibration cache: %s" % cache, file=sys.stderr)

    # ---- stage 2: rebuild with the container's own TensorRT ----
    name = os.path.basename(args.out)
    os.makedirs(os.path.join(args.out, "1"), exist_ok=True)
    plan = os.path.join(args.out, "1", "model.plan")
    cmd = [TRTEXEC, "--onnx=" + onnx_int8, "--int8", "--fp16",
           "--calib=" + cache, "--saveEngine=" + plan,
           "--warmUp=500", "--duration=5"]
    print("+ " + " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(plan):
        sys.stderr.write(r.stdout[-3000:])
        sys.exit("trtexec did not produce an engine")
    for line in r.stdout.splitlines():
        if "Throughput:" in line:
            print(line.strip(), file=sys.stderr)
            break

    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(CONFIG.format(name=name))
    print("installed into Triton model repo: %s" % args.out)
    print("restart the server, then:")
    print("  python3 scripts/accuracy_eval.py --model %s --limit 500 --conf 0.001" % name)


if __name__ == "__main__":
    main()
