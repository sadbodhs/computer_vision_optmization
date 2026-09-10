#!/usr/bin/env python3
"""Reference mAP for the SAME image subset, using ultralytics end-to-end.

This is the control for scripts/accuracy_eval.py. That script measures the
study's pipeline (our letterbox -> TensorRT FP16 engine -> our class-aware NMS);
this one measures ultralytics' own pipeline (its letterbox -> PyTorch model ->
its NMS) over the identical images and scores them with the identical
pycocotools call. The difference between the two is what our custom chain costs.

NMS parameters are forced to match the flows (conf/iou) so the comparison
isolates preprocessing + engine rather than thresholding policy.

Usage (inside the Triton container):
  python3 scripts/accuracy_reference.py --limit 500 --conf 0.001 --iou 0.45
"""
import argparse
import json
import os
import sys

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/models/yolov8s.pt")
    ap.add_argument("--images", default="/coco/val2017")
    ap.add_argument("--ann", default="/coco/annotations/instances_val2017.json")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="/tmp/dets_reference.json")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[:args.limit]

    model = YOLO(args.weights)
    results = []
    paths = [os.path.join(args.images, coco.loadImgs(i)[0]["file_name"]) for i in img_ids]

    B = 32
    for s in range(0, len(paths), B):
        batch_paths = paths[s:s + B]
        batch_ids = img_ids[s:s + B]
        preds = model.predict(batch_paths, imgsz=args.imgsz, conf=args.conf,
                              iou=args.iou, max_det=100, verbose=False, device=0)
        for iid, r in zip(batch_ids, preds):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
                results.append({
                    "image_id": iid,
                    "category_id": COCO_IDS[int(c)],
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(sc), 5),
                })
        print(f"  {min(s+B, len(paths))}/{len(paths)} images, {len(results)} dets",
              file=sys.stderr)

    if not results:
        print("no detections", file=sys.stderr); sys.exit(1)
    json.dump(results, open(args.out, "w"))

    cocoDt = coco.loadRes(args.out)
    e = COCOeval(coco, cocoDt, "bbox")
    e.params.imgIds = img_ids
    e.evaluate(); e.accumulate(); e.summarize()
    print(json.dumps({
        "source": "ultralytics_reference", "weights": args.weights,
        "images": len(img_ids), "conf": args.conf, "nms_iou": args.iou,
        "mAP50_95": round(float(e.stats[0]), 5),
        "mAP50": round(float(e.stats[1]), 5),
        "mAP75": round(float(e.stats[2]), 5),
        "detections": len(results),
    }))


if __name__ == "__main__":
    main()
