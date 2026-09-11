# DeepStream (Flow E) — NVIDIA's integrated stack

[← index](../README.md) · prev: [Results](results.md) · next: [Transport](transport.md)

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

## Capacity mode: attempted, and it measures the harness

The study hypothesised that E2 in capacity mode "should push it above D", since
DeepStream has no RPC at all. That was worth testing, and `ds_bench --mode file`
does it: an `appsrc` per stream pushes raw NV12 frames flat-out from a file into
`nvstreammux`, with `sync=false` on the sink so nothing waits on a clock.

| Config | Engine | Total fps |
|---|---|---|
| 1 stream | fixed batch-1 | 653.0 |
| 1 stream | dynamic b1–8 | 210.5 |
| 3 streams | dynamic b1–8 | 400.9 |
| 8 streams | dynamic b1–8 | 501.3 |
| 8 streams | **fixed batch-8** | **~501** (500.0 / 502.0 / 501.9) |

Against D's 1640 fps that looks like a decisive answer. **It isn't one**, and the
reason is in the next line:

> **GPU utilisation during the 8-stream run: median 0%, max 99%.**

The GPU is idle almost the whole time. Two more tells point the same way: the
fixed batch-8 engine performs *identically* to the dynamic one (~501 fps both),
even though the dynamic engine is 3× slower at batch 1 (210 vs 653 fps) — so the
engine is plainly not the constraint; and the frame count comes out at exactly
6128 on three consecutive runs of a *time-limited* benchmark, which is what a
deterministic feed rate looks like, not a throughput measurement.

**The bottleneck is the test harness.** Each `appsrc` push does a 345 KB CPU
`memcpy` per frame, then `nvvideoconvert` copies sysmem→NVMM, per stream, per
frame. At eight streams that path saturates before DeepStream does, and the
inference sits waiting.

### So the comparison is still open

This is the same trap the study's first pass fell into, in a new costume — see
[methodology](methodology.md): *a benchmark that saturates the source measures the
source.* Here the "source" is our own frame feeder. **E2's batched-inference
ceiling remains unmeasured, and 501 fps must not be quoted as it.**

Closing it properly means removing the feed from the critical path: pushing
pre-allocated NVMM buffers rather than sysmem ones, or using a buffer pool so
there is no per-frame `memcpy` and no format conversion. That is a real change to
`ds_bench`, not a parameter.

**What can be said:** at a single stream with a fixed batch-1 engine, E2 sustains
**653 fps** in capacity mode against a source-bound 45 fps — so the source, not
DeepStream, was the limit in the [E1/E2 tables above](#e2--multi-stream-batched).
That much the exercise did establish. See [roadmap](roadmap.md).

## Verdict

For an **edge product on NVIDIA hardware**, E is the pragmatic pick: 1.50 ms/frame
with **zero custom code**, NVDEC→infer integrated, and no server to operate. You
trade the flexibility of a serving layer (model reload, metrics, multi-tenant
scheduling) for an integrated pipeline that just works.

---

[← index](../README.md) · prev: [Results](results.md) · next: [Transport](transport.md)
