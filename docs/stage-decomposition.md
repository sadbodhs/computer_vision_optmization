# Where does the time go? — stage decomposition

[← index](../README.md) · prev: [Contention](contention.md) · next: [DeepStream](deepstream.md)

Per-stage wall time for one frame, measured inside each flow with per-stage timers
(`stages_ms` in every binary's JSON output). Concurrency 1, YOLOv8s.

---

## Capacity mode (preprocessed frames, no decode — pure pipeline cost)

| Stage | A2 (C++ TRT) | B2 (Triton, CUDA shm) | C2 (Python numpy) |
|---|---|---|---|
| Host→Device transfer | **0.25 ms** (H2D, pinned+async) | 0 (zero-copy shm) | 0 (sys-shm, server copies) |
| Preprocess | 0 (already preprocessed input) | 0 | 0 |
| Inference (GPU) + output handling | **0.98 ms** (incl. compact kernel) | 1.16 ms (gRPC round trip incl. server infer) | ~5.1 ms (gRPC + serialize) |
| Postprocess (NMS, CPU) | 0.01 ms | 0.00 ms | ~1-3 ms (numpy NMS) |
| **Total** | **1.25 ms** | **1.26 ms** | **~6.4 ms** |

The engine itself is 0.97–0.98 ms everywhere. A2 adds 0.25 ms H2D + 0.01 NMS.
B2's zero-copy shm makes the *entire* Triton framework cost **0.18 ms** over A2.
C2's gap is client-side: numpy serialization into gRPC (~3 ms) + Python NMS (~2 ms).

## RTSP end-to-end mode (adds decode; includes source pacing)

| Stage | A2 (C++ full-CUDA) | B2 (Triton, CUDA shm) | C2 (Python numpy) |
|---|---|---|---|
| Decode (NVDEC / pipe) | 0.15-0.4 ms compute (rest = waiting for 30fps frame) | same | ffmpeg pipe ≈ 33.3 ms wall (pacing) |
| Preprocess | **0.15 ms** (fused CUDA kernel) | 0.15 ms | **1.26 ms** (numpy CPU) |
| Infer + transfer | 1.22 ms | 1.71 ms (gRPC) | 5.08 ms (gRPC raw) |
| Postprocess | 0.001 ms | 0.001 ms | ~1-3 ms |
| **GPU-only total** | **≈1.4 ms** | **≈1.9 ms** | **≈7.5 ms** |

## The architectural tax, stacked

Starting from the engine's honest price, each hop adds:

```
raw engine ..................... 0.97 ms   (the GPU's honest price)
+ in-process C++ wrapper ....... +0.26 ms  → A2: 1.23 ms   (our code, CUDA kernels, NMS)
+ Triton server + gRPC ......... +2.0  ms  → B1: 3.27 ms   (copies + scheduling)
+ Python client ................ +3.1  ms  → C1: 6.4  ms   (interpreter, GIL, torch)
```

The +2.0 ms of "Triton server + gRPC" is almost entirely copies, and copies are
fixable — [transport](transport.md) shows it collapsing to ~0.18 ms with CUDA shm.

## Four facts that fall out

**Decode is free at 30 FPS.** NVDEC decodes a frame in ~0.2–0.4 ms; for the rest
of the 33 ms interval the decoder *waits*. In the C++ flows the "decode" timer
reads ~31.9 ms — that is socket-read blocking, not compute. Do not optimize it.

**The fused CUDA preprocess kernel (0.15 ms) is 8× faster than numpy-on-CPU
(1.26 ms)** — and numpy is already the *fast* Python option; torch was worse.

> Measured when the kernel still used nearest-neighbour interpolation. It now
> interpolates bilinearly, which costs **+0.0085 ms** and recovers 1.25% mAP —
> see [accuracy](accuracy.md). The stage remains far from the bottleneck.

**The framework tax lives only in the infer stage.** B2 pays 1.16–1.71 ms where A2
pays 0.98–1.22.

**CPU vs GPU split.** In A2, ~99% of pipeline time is GPU work. In C2, roughly 60%
of client time is CPU (numpy serialize + NMS) — the GPU sits idle waiting for its
next frame.

> **The GPU is almost never the bottleneck at the edge.** Every Triton
> configuration here left it >90% idle until the copies came out. The fight is
> over PCIe round trips, serialization, and interpreter locks.

## Note on the output tensor

Every flow moves a `[1,84,8400]` FP32 output = **2.8 MB per frame**. A2/B2/D
reduce it on-GPU with a compact kernel before it crosses to the host; C2 does not.
Folding NMS into the engine itself would remove this entirely — untested, see
[roadmap](roadmap.md).

---

[← index](../README.md) · prev: [Contention](contention.md) · next: [DeepStream](deepstream.md)
