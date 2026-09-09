# Corrected Comparison Tables — the 4 flows

The raw "fps" table alone misrepresents Flow D (its throughput is inflated by
batch-8 amortization while each frame waits longer). These tables present the
numbers fairly: latency per frame, per-frame GPU cost, and suitability.

## Table 1 — Per-frame view (1 frame through the pipeline, comparable 1:1)

| Flow | Stack | Latency p50 (1 frame) | vs 33.3 ms budget (30 FPS) | Latency p95 | Verdict |
|---|---|---|---|---|---|
| **A2** | C++ TRT, all-CUDA | **1.23 ms** | 3.7% of budget | 1.26 ms | fastest per frame |
| **B2** | C++→Triton, CUDA shm | **1.28 ms** | 3.8% | 1.32 ms | ≈A2, with Triton ops |
| **C2** | Python numpy→Triton, sys-shm | 1.69 ms | 5.1% | 1.86 ms | fastest Python |
| **D** | async→Triton dyn-batch(8) | 6.23 ms | 18.7% | 7.16 ms | pays a queue wait |

## Table 2 — Throughput (system capacity when fed continuously)

| Flow | conc=1 | conc=8 | conc=16 | Effective GPU cap |
|---|---|---|---|---|
| A2 | 809 | — | ~1205* | 1028 qps (batch-1 engine) |
| B2 | 654 | 1131 | 1128 | 1028 (same engine, via server) |
| C2 | 469 | 1037 | 1038 | 1028 (same engine, via server) |
| D | 1041 | 1640 | **1665** | 1649 (batch-8 engine) |

*A2's 1205 at conc=16 comes from 16 CUDA streams overlapping H2D/compute/D2H —
the engine itself is still 1028 qps per stream of work.

## Table 3 — Efficiency per frame (the fair unit for D)

Per-frame GPU cost = GPU pass time ÷ frames per pass. This is where batching wins:

| Flow | Frames per GPU pass | GPU time/pass | **GPU cost per frame** | Client+server overhead per frame |
|---|---|---|---|---|
| A2 | 1 | 0.97 ms | **0.97 ms** | ~0.26 ms (in-process) |
| B2 | 1 | 0.97 ms | **0.97 ms** | ~0.31 ms (CUDA shm round trip) |
| C2 | 1 | 0.97 ms | **0.97 ms** | ~0.72 ms (sys-shm round trip) |
| D  | 8 | 4.84 ms | **0.61 ms** | batch window ~1–5 ms + round trip |

→ D's GPU is **37% cheaper per frame** (0.61 vs 0.97 ms) because one pass shares
weights/memory across 8 frames. That efficiency is the entire reason its fps is
higher; it costs 5–70 ms of queue latency per frame to realize it.

## Table 4 — What to run when (decision table)

| Scenario | Pick | Why |
|---|---|---|
| Live camera, 30 FPS, latency-sensitive (1 stream) | **A2 or B2** | 1.2 ms ≪ 33 ms budget; D's 6–70 ms wait wastes budget |
| Live multi-stream (N cameras) | **B2** | 1128 fps ≫ N×30 fps; Triton gives ops/reload/batching later |
| Python-only stack | **C2** | 1038 fps; sys-shm is the right shm for CPU data |
| Offline video / batch analytics / many users | **D** | highest fps per GPU-second (1665), latency negotiable |
| Model-as-a-service (bursty clients) | **D** | batching amortizes GPU across concurrent users |

## One-line summary per flow

- **A2** — lowest latency, in-process control. 809 fps single, 1.23 ms.
- **B2** — Triton without the tax (CUDA shm): 654–1131 fps, 1.28 ms.
- **C2** — Python without the tax (numpy + sys-shm + processes): 1038 fps, 1.69 ms.
- **D** — GPU-efficient batching: 0.61 ms/frame GPU cost, 1665 fps, 6–70 ms wait.
---

# Flow E — DeepStream (added: closes the loop)

DeepStream 7.1: nvv4l2decoder (NVDEC, NVMM zero-copy) -> nvstreammux (batched) ->
nvinfer -> fakesink. **Same weights**: same ONNX (md5-verified), FP16 engines built
in-container with TRT 10.3 (same ONNX gives 0.969 ms GPU — parity with TRT 10.7
engine 0.970 ms). Same preprocessing via nvinfer (1/255, aspect-ratio, symmetric
padding) + marcoslucianops YOLO parser; postproc semantics differ slightly
(cluster-mode=2 vs our class-aware NMS — detection counts differ, inference cost
does not).

## E1 — single stream (vs latency-optimized flows)

| Flow | Latency p50 (decode->infer->parse) | fps (RTSP source-bound) |
|---|---|---|
| A2 (C++ full-CUDA) | 1.25 ms | ~20-23 |
| B2 (C++->Triton, CUDA shm) | 1.55 ms | ~12 |
| **E1 (DeepStream)** | **1.50 ms** | **45** |
| C2 (Python numpy) | ~2-8 ms | ~8-20 |

E1 latency sits between A2 and B1 — all in the same league; DeepStream has no RPC
at all and its batch-window is negligible at batch=1.

## E2 — multi-stream batched (vs Flow D throughput arm)

| Config | Total fps | fps/stream | Latency p50 | GPU util |
|---|---|---|---|---|
| D: async dyn-batch, conc=16 (file mode, no source cap) | 1665 | 104 | 74 ms | 79% |
| **E2: DeepStream 8 streams (RTSP 30fps sources)** | **226** | 28 | **32.7 ms** | 5.4% |
| E2: DeepStream 6 streams | 169 | 28 | 32.5 ms | - |
| E2: DeepStream 3 streams | 104 | 34.5 | 3.1 ms | - |

**Key reading**: E2 is *source-bound* — 8x30fps = 240 fps demand, and it delivers
226 fps (94% of source capacity) with only 5.4% GPU. The 33 ms p50 is the
streammux batch-assembly wait (batch=8 at 30fps sources fills every ~33 ms), not
inference. To compare against D's 1665 fps, E2 would need the file-replay
capacity mode (same frames.bin trick) — that would show DeepStream's true
batched-inference ceiling, which its per-batch overhead (no RPC at all) should
push above D.

## Notes
- DeepStream container ran TensorRT 10.3 (vs 10.7 in Triton container): engines
  rebuilt from the identical ONNX (md5 0fa8a04298da24d785b29f001a8c139e verified);
  trtexec parity: 1030 vs 1028 qps — negligible.
- DeepStream uses nvstreammux batch-assembly (fixed batch) vs Triton dynamic
  batching (queue+delay): E2's batch wait is deterministic (sources arrive at
  fixed rate), D's depends on client in-flight patterns.
