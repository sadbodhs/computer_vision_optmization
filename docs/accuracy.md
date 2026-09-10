# Accuracy — does the pipeline preserve the model?

[← index](../README.md) · related: [Methodology](methodology.md) · [Precision](precision.md)

Every other page in this study measures speed. This one asks the question speed
cannot: **does the custom preprocessing → TensorRT → custom NMS chain actually
produce the model's detections?**

COCO val2017, 500 images, scored with `pycocotools`. The study's chain
([`scripts/accuracy_eval.py`](../scripts/accuracy_eval.py)) is compared against
ultralytics running end-to-end on the *same images with the same conf/IoU*
([`scripts/accuracy_reference.py`](../scripts/accuracy_reference.py)), so the
comparison isolates our pipeline rather than thresholding policy.
Raw data: [`results/v3/accuracy.tsv`](../results/v3/accuracy.tsv).

---

## Result

| Pipeline | Engine | Resize | mAP50-95 | mAP50 | vs reference |
|---|---|---|---|---|---|
| ultralytics reference | PyTorch FP32 | linear | **0.47376** | 0.64535 | — |
| study chain | TensorRT FP16 | linear | **0.47348** | 0.64524 | **−0.00028 (−0.06%)** |
| **study chain as shipped** | TensorRT FP16 | **nearest** | 0.46782 | 0.63582 | **−0.00594 (−1.25%)** |

## 1. The chain is sound — the custom code costs essentially nothing

With linear resize, the study's pipeline lands **0.06% from the PyTorch
reference**. That single number validates a lot at once:

- the centered-letterbox geometry (pad 114, /255, RGB) is correct
- **FP16 costs no measurable accuracy** versus FP32 on this model
- the hand-rolled class-aware NMS matches ultralytics' behaviour
- the box un-letterboxing math (remove pad, rescale, clip) is right

That is the reassurance the whole benchmark rested on but never demonstrated. It
is now demonstrated rather than assumed.

## 2. …but as shipped, every flow leaves ~0.6 mAP on the table

The pipeline does **not** use linear resize. It uses nearest-neighbour, and not
only in the Python client — the fused CUDA kernel does too:

```c
// cpp/src/main_cuda.cu — nv12_letterbox_kernel
// nearest-neighbor mapping into source
int u = sx * src_w / nw, v = sy * src_h / nh;
```

and [`triton/client_v2.py`](../triton/client_v2.py) mirrors it in numpy. So this
applies to **A2, B2, C2 and D alike** — the entire study.

The cost is **−0.0059 mAP50-95 (−1.25%)** and **−0.0095 mAP50** against the
reference, for a preprocessing shortcut that buys a handful of memory loads in a
kernel that is not the bottleneck ([stage decomposition](stage-decomposition.md)
puts preprocessing at 0.15 ms against a 0.97 ms engine).

> **This is the finding that justifies having an accuracy axis at all.** No amount
> of throughput measurement could surface it: every flow was *equally* wrong, so
> they all agreed with each other, and detection counts looked plausible. It took
> an external ground truth to see it.

## Caveats

- **500 images, not the full 5000.** Enough to resolve a 1.25% gap, not enough for
  a headline "YOLOv8s scores X" claim. The absolute numbers here run above
  ultralytics' published 44.9 partly for that reason and partly because NMS IoU is
  0.45 (the flows' value) rather than ultralytics' 0.7 default.
- **The comparison is chain-vs-chain**, so it bundles preprocessing, engine
  precision and NMS. The linear-resize row being ~0 means none of those three
  individually hurts; it does not decompose them.
- Measured at `conf 0.001` (the mAP convention). The flows *deploy* at `conf 0.25`,
  which is an operating-point choice and not what mAP measures.
- Only YOLOv8s.

## What this unblocks

INT8 now has somewhere to report to. [Precision](precision.md) established a
**+33.6% throughput ceiling** for INT8 but deliberately refused to call it a
result, because an uncalibrated engine's detections are meaningless. With this
harness in place, a calibrated INT8 engine can be scored on the same 500 images
against the same reference, turning that ceiling into an actual
speed-versus-accuracy trade.

---

[← index](../README.md) · related: [Methodology](methodology.md) · [Precision](precision.md)
