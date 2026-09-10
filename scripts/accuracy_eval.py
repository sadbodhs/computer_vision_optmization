#!/usr/bin/env python3
"""COCO mAP for the study's pipeline: does our preprocessing + NMS chain preserve
the model's accuracy, and what does a precision change cost?

Runs COCO val2017 through Triton using the SAME preprocessing contract the
benchmark flows use (centered letterbox, pad 114, RGB, /255) and the SAME
class-aware NMS, then scores with pycocotools.

Two knobs exist specifically to test study choices rather than assume them:
  --resize nearest|linear   the numpy client letterboxes with nearest-neighbour;
                            linear is what reference implementations use.
  --conf                    the flows deploy at 0.25, but mAP is conventionally
                            measured near 0.001. Both are meaningful, for
                            different questions.

Usage (inside the Triton container):
  python3 scripts/accuracy_eval.py --model yolov8s \
      --images /coco/val2017 --ann /coco/annotations/instances_val2017.json \
      --limit 500 --conf 0.001 --resize linear
"""
import argparse
import json
import os
import sys
import time

import numpy as np

IMG = 640
NUM_ANCHORS = 8400

# YOLO class index (0-79) -> COCO category_id (1-90, with gaps)
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90]


def letterbox(img_bgr, resize="linear"):
    """BGR uint8 HxWx3 -> ([1,3,640,640] float32, scale, pad_x, pad_y).

    Same geometry as the benchmark flows: centered, pad 114, BGR->RGB, /255.
    """
    import cv2
    h, w = img_bgr.shape[:2]
    scale = min(IMG / w, IMG / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_NEAREST if resize == "nearest" else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    canvas = np.full((IMG, IMG, 3), 114, dtype=np.uint8)
    px, py = (IMG - nw) // 2, (IMG - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB, /255
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]  # 1,3,640,640
    return chw, scale, px, py


def nms_class_aware(boxes, scores, classes, iou_thr):
    """Same semantics as the flows' NMS: greedy, per class, IoU threshold."""
    keep = []
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    suppressed = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        a = order[i]
        keep.append(a)
        rest = order[i + 1:]
        live = rest[~suppressed[i + 1:]]
        if len(live) == 0:
            continue
        xx1 = np.maximum(x1[a], x1[live]); yy1 = np.maximum(y1[a], y1[live])
        xx2 = np.minimum(x2[a], x2[live]); yy2 = np.minimum(y2[a], y2[live])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[a] + areas[live] - inter, 1e-9)
        same = classes[live] == classes[a]
        bad = live[(iou > iou_thr) & same]
        if len(bad):
            pos = {v: k for k, v in enumerate(order)}
            for b in bad:
                suppressed[pos[b]] = True
    return keep


def postprocess(out, scale, px, py, w0, h0, conf_thr, iou_thr, max_det=100):
    """[1,84,8400] -> COCO detection dicts in ORIGINAL image coordinates."""
    o = out[0]                       # 84,8400
    scores_all = o[4:, :]            # 80,8400
    cls = scores_all.argmax(axis=0)
    conf = scores_all.max(axis=0)
    m = conf > conf_thr
    if not m.any():
        return []
    idx = np.nonzero(m)[0]
    cx, cy, bw, bh = o[0, idx], o[1, idx], o[2, idx], o[3, idx]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    sc, cl = conf[idx].astype(np.float64), cls[idx]

    keep = nms_class_aware(boxes, sc, cl, iou_thr)[:max_det]
    dets = []
    for k in keep:
        x1, y1, x2, y2 = boxes[k]
        # undo letterbox: remove padding, then rescale
        x1 = (x1 - px) / scale; x2 = (x2 - px) / scale
        y1 = (y1 - py) / scale; y2 = (y2 - py) / scale
        x1 = float(np.clip(x1, 0, w0)); x2 = float(np.clip(x2, 0, w0))
        y1 = float(np.clip(y1, 0, h0)); y2 = float(np.clip(y2, 0, h0))
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append({"category_id": COCO_IDS[int(cl[k])],
                     "bbox": [round(x1, 2), round(y1, 2),
                              round(x2 - x1, 2), round(y2 - y1, 2)],
                     "score": round(float(sc[k]), 5)})
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s")
    ap.add_argument("--server", default="localhost:8001")
    ap.add_argument("--images", default="/coco/val2017")
    ap.add_argument("--ann", default="/coco/annotations/instances_val2017.json")
    ap.add_argument("--limit", type=int, default=500, help="0 = all 5000")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU (flows use 0.45)")
    ap.add_argument("--resize", default="linear", choices=["linear", "nearest"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2
    import tritonclient.grpc as grpcclient
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[:args.limit]

    client = grpcclient.InferenceServerClient(url=args.server)
    results, t0 = [], time.perf_counter()
    for n, iid in enumerate(img_ids, 1):
        info = coco.loadImgs(iid)[0]
        path = os.path.join(args.images, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        tensor, scale, px, py = letterbox(img, args.resize)

        inp = [grpcclient.InferInput("images", [1, 3, IMG, IMG], "FP32")]
        inp[0].set_data_from_numpy(tensor)
        res = client.infer(args.model, inp,
                           outputs=[grpcclient.InferRequestedOutput("output0")])
        out = res.as_numpy("output0")

        for d in postprocess(out, scale, px, py, w0, h0, args.conf, args.iou):
            d["image_id"] = iid
            results.append(d)
        if n % 100 == 0:
            print(f"  {n}/{len(img_ids)} images, {len(results)} dets, "
                  f"{n/(time.perf_counter()-t0):.1f} img/s", file=sys.stderr)

    if not results:
        print("no detections produced", file=sys.stderr); sys.exit(1)

    out_path = args.out or f"/tmp/dets_{args.model}_{args.resize}.json"
    json.dump(results, open(out_path, "w"))

    cocoDt = coco.loadRes(out_path)
    e = COCOeval(coco, cocoDt, "bbox")
    e.params.imgIds = img_ids
    e.evaluate(); e.accumulate(); e.summarize()
    print(json.dumps({
        "model": args.model, "images": len(img_ids), "resize": args.resize,
        "conf": args.conf, "nms_iou": args.iou,
        "mAP50_95": round(float(e.stats[0]), 5),
        "mAP50": round(float(e.stats[1]), 5),
        "mAP75": round(float(e.stats[2]), 5),
        "detections": len(results),
    }))


if __name__ == "__main__":
    main()
