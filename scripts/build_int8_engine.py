#!/usr/bin/env python3
"""Build a CALIBRATED INT8 TensorRT engine for YOLOv8s and install it in the
Triton model repo, so INT8 can be scored on the same footing as FP16.

docs/precision.md established a +33.6% throughput *ceiling* for INT8 using an
uncalibrated engine, and deliberately refused to call it a result: without
calibration the detections are meaningless. This produces a real engine, using
COCO val images as the calibration set, so scripts/accuracy_eval.py can measure
what that throughput actually costs in mAP.

Calibration images come from the same COCO val2017 directory the evaluation uses.
That overlap is acceptable for a speed-vs-accuracy delta but would not be
acceptable for an absolute accuracy claim -- noted in docs/accuracy.md.

Usage (inside the Triton container):
  python3 scripts/build_int8_engine.py --weights /models/yolov8s.pt \
      --images /coco/val2017 --fraction 0.05 --out /models/yolov8s_int8
"""
import argparse
import os
import shutil
import sys

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
        f.write(f"path: {root}\ntrain: {leaf}\nval: {leaf}\nnames:\n")
        for k, v in names.items():
            f.write(f"  {k}: {v}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/models/yolov8s.pt")
    ap.add_argument("--images", default="/coco/val2017")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="fraction of the image set used for calibration")
    ap.add_argument("--out", default="/models/yolov8s_int8")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    cal_yaml = write_cal_yaml("/tmp/cal.yaml", args.images)
    print(f"calibration set: {args.images} (fraction={args.fraction})", file=sys.stderr)

    model = YOLO(args.weights)
    engine_path = model.export(
        format="engine", int8=True, data=cal_yaml, imgsz=args.imgsz,
        batch=1, dynamic=False, workspace=8, verbose=False,
        fraction=args.fraction,
    )
    print(f"exported: {engine_path}", file=sys.stderr)

    name = os.path.basename(args.out)
    os.makedirs(os.path.join(args.out, "1"), exist_ok=True)
    shutil.copy(str(engine_path), os.path.join(args.out, "1", "model.plan"))
    with open(os.path.join(args.out, "config.pbtxt"), "w") as f:
        f.write(CONFIG.format(name=name))
    print(f"installed into Triton model repo: {args.out}")
    print("restart the server, then:")
    print(f"  python3 scripts/accuracy_eval.py --model {name} --limit 500 --conf 0.001")


if __name__ == "__main__":
    main()
