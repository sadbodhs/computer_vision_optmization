# DeepStream (Flow E) — NVIDIA's integrated stack

[← index](../README.md) · prev: [Stage decomposition](stage-decomposition.md) · next: [Reproduce](reproduce.md)

The control arm: what do you get when you use NVIDIA's own end-to-end pipeline
instead of assembling one?

`nvv4l2decoder → nvstreammux → nvinfer → parser`, no RPC anywhere.

---

## Parity with the other flows

Same ONNX (md5 `0fa8a04298da24d785b29f001a8c139e`, verified), engines rebuilt
in-container with TensorRT 10.3 (vs 10.7 for Triton). `trtexec` parity check:
**1030 vs 1028 qps** — negligible. Preprocessing via `nvinfer` (1/255,
aspect-ratio, symmetric padding) + the marcoslucianops YOLO parser.

> **Postprocess semantics differ**: DeepStream uses `cluster-mode=2` where the
> other flows use class-aware NMS. **Detection counts differ; inference cost does
> not.** Compare E's timings, not its detection totals.

## E1 — single stream

| Flow | Latency p50 (decode→infer→parse) | fps (RTSP source-bound) |
|---|---|---|
| A2 (C++ full-CUDA) | 1.25 ms | ~20-23 |
| B2 (C++→Triton, CUDA shm) | 1.55 ms | ~12 |
| **E1 (DeepStream)** | **1.50 ms** | **45** |
| C2 (Python numpy) | ~2-8 ms | ~8-20 |

E1 sits between A2 and B2 — all in the same league. DeepStream has no RPC at all
and its batch window is negligible at batch=1.

## E2 — multi-stream batched

| Config | Total fps | fps/stream | p50 | GPU util | Reading |
|---|---|---|---|---|---|
| E1: 1 stream (RTSP 30fps) | 45 | 45 | **1.50 ms** | — | decode+infer+parse, no RPC |
| E2: 3 streams | 104 | 34.5 | 3.1 ms | — | batch=3 assembles fast |
| E2: 6 streams | 169 | 28 | 32.5 ms | — | batch wait dominates |
| E2: 8 streams | 226 | 28 | 32.7 ms | **5.4%** | source-bound: 240fps demand, 94% delivered |

**E2 is source-bound, not compute-bound.** 8×30 fps = 240 fps of demand; it
delivers 226 (94%) using **5.4% of the GPU**. The 33 ms p50 is the `nvstreammux`
batch-assembly wait — a batch of 8 fills every ~33 ms when sources arrive at
30 fps — not inference.

## Batch-assembly: deterministic vs demand-driven

| | DeepStream `nvstreammux` | Triton dynamic batching |
|---|---|---|
| Batch fills when | sources tick (fixed rate) | client requests arrive |
| Wait is | deterministic (~33 ms at 30 fps) | depends on client in-flight pattern |

Same trade as [flow D](batching.md), reached by a different road: E2's 33 ms is the
frame interval; D's 6–74 ms is the queue.

## The open comparison

E2 has **not** been run in capacity mode. To compare its batched-inference ceiling
against D's 1665 fps directly, E2 needs the same file-replay treatment (the
`frames.bin` trick). Its per-batch overhead — no RPC at all — should push it above
D. **Untested; stated as a hypothesis, not a result.** See [roadmap](roadmap.md).

## Verdict

For an **edge product on NVIDIA hardware**, E is the pragmatic pick: 1.50 ms/frame
with **zero custom code**, NVDEC→infer integrated, and no server to operate. You
trade the flexibility of a serving layer (model reload, metrics, multi-tenant
scheduling) for an integrated pipeline that just works.

---

[← index](../README.md) · prev: [Stage decomposition](stage-decomposition.md) · next: [Reproduce](reproduce.md)
