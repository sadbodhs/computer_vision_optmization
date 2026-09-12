# Methodology — how these numbers were produced

[← index](../README.md) · prev: [Use cases](use-cases.md) · next: [Results](results.md)

Everything else in this repo depends on this page being right. The first pass of
this study was wrong precisely because these rules were not in place — see
[STORY §3](../STORY.md) for that post-mortem.

---

## The two measurement modes

- **Capacity** — preprocessed frames replayed flat-out from disk. Measures what the
  pipeline can actually do, with no camera pacing. This is where framework
  overhead shows.
- **RTSP end-to-end** — live 30 fps sources. Measures "can it keep up + latency",
  bounded by the source, not the pipeline.

> **A benchmark that saturates the source measures the source.** The v1 pass
> compared a source-capped C++ number (29 fps of an available 30) against an
> uncapped Triton number (14 fps) and concluded "C++ is 2× faster". It measured
> the camera. Capacity mode exists to make every number mean what it says.

The capacity input is `frames.bin` — a headerless concatenation of
`N × [1,3,640,640]` float32 tensors (4,915,200 bytes/frame), regenerated with
[`scripts/make_frames.sh`](../scripts/make_frames.sh).

## The reference ceiling

`trtexec` on the FP16 YOLOv8s engine is the physical floor for per-frame cost:

| Engine | Per frame | Throughput |
|---|---|---|
| batch-1 | **0.97 ms** | 1028 fps |
| batch-8 | 4.84 ms / 8 frames = **0.61 ms** | 1630 fps effective |

Also measured: YOLOv8n **1490 qps** · YOLO11n **1259 qps**.

Everything in this study is the story of what stands between your camera and
that 0.97 ms.

**0.97 ms is the right baseline here, but it is not a hardware floor.** About
**0.13 ms of it is CPU-side kernel *launch* overhead** rather than GPU compute —
with CUDA Graphs the same engine runs at **0.86 ms / 1166 qps**. Every flow in
this study launches per frame, so 0.97 ms is what they are all measured against;
[CUDA graphs](cuda-graphs.md) covers what changes when you stop doing that.

## How to read the tables

Two independent scores per pipeline: *throughput* and *latency*. A pipeline can
win one and lose the other — that is the whole Triton-vs-C++ story, and the whole
point of [flow D](batching.md).

**Flow D reads differently on purpose**: `1041↑ · wait 6.2 · svc 0.61` means each
frame **waits 6.2 ms** for its batch to fill, then the GPU **services it in
0.61 ms**. D's high fps is *bought with queue wait*.

### Notation

The arrow is used in two senses, and the difference matters:

| Form | Means | Example |
|---|---|---|
| `a → b` **inside a stage or flow name** | *then*, or *into* | `decode→infer→parse`, `BGR→RGB`, `Host→Device` |
| `a → b` **inside a measured cell** | *before → after*, baseline first | `1023 → 1359 fps` is the baseline, then the change being tested |

Where a column is a before/after pair, its header names both sides
(`transport share FP32 → FP16`), so a cell can always be read on its own.

Other conventions: **p50/p95/p99** are percentiles of per-frame latency, never
averages. **fps** is whole-pipeline throughput, not per stream. A **bold** row is
the one the surrounding text is arguing about. *Italic* rows are controls or
failures, not results.

## Preprocessing parity (the contract)

> **Match preprocessing bit-for-bit before comparing pipelines.** In v1 the C++
> side used a bottom-padded letterbox and the Python side a centered one; they
> reported wildly different detection counts on the same video, and every
> downstream comparison was invalid.

| Item | Value |
|---|---|
| **Model** | YOLOv8s (Ultralytics), COCO 80 classes |
| **Input** | 640×640, RGB, /255 (sigmoid baked into export) |
| **Precision** | FP16 engines from the same ONNX (md5-verified) |
| **Engine format** | `.plan` (Triton) / `.engine` (DeepStream) — same serialization |
| **Output** | `[1, 84, 8400]`, conf 0.25, class-aware NMS IoU 0.45 |
| **Also tested** | YOLOv8n (1490 qps, 0.392 mAP) · YOLO11n (1259 qps, 0.416 mAP) — see [accuracy](accuracy.md#5-speed-and-accuracy-together-per-model) |
| **Preproc contract** | centered letterbox, pad 114, BGR→RGB, /255 — identical in every flow |

The single implementation of this contract is
[`triton/client_v2.py:letterbox_nv12_np`](../triton/client_v2.py); the frames.bin
generator imports it rather than re-implementing it, so the two cannot drift.

## Test rig

RTX 3090 (compute 8.6, driver 595.84) · Triton 24.12 · TensorRT 10.7 (10.3 in the
DeepStream container) · DeepStream 7.1 · Docker for everything.

All numbers are medians across ≥3 runs unless noted; variance was <±2% — with one known exception, **A2 at concurrency 2, which is bimodal at ±12%** (see [`results/README.md`](../results/README.md)).

## Known caveats

- Flow C2 at N≥4 in-process hit a cross-thread shm race; valid C2 in-process data
  is N≤3 (see [contention](contention.md)).
- DeepStream postprocess semantics differ (`cluster-mode=2` vs our class-aware
  NMS), so **detection counts** differ between E and the other flows; inference
  cost does not.
- There is **no accuracy (mAP) axis** in this study — every flow runs the same
  FP16 weights, so speed is comparable, but see [roadmap](roadmap.md) for why this
  becomes essential the moment INT8 enters.

---

[← index](../README.md) · prev: [Use cases](use-cases.md) · next: [Results](results.md)
