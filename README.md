# Triton vs Pure TensorRT vs DeepStream — Inference Pipeline Benchmark

One question, answered with measurements: *for the same YOLO model on the same
GPU, which serving pipeline processes a frame fastest, and which delivers the
most frames per second?*

Hardware: RTX 3090 · Triton 24.12 · TensorRT 10.7/10.3 · DeepStream 7.1 ·
**YOLOv8s** FP16 @ 640×640 (identical ONNX, md5-verified across flows).
Everything runs in Docker.

**New here?** Read [STORY.md](STORY.md) — the narrative of how the first answer
turned out to be wrong, and what it took to get a trustworthy one.

---

## Index

| Doc | Question it answers | Headline finding |
|---|---|---|
| [Methodology](docs/methodology.md) | How were these numbers produced? | A benchmark that saturates the source measures the source |
| [Results](docs/results.md) | How fast is each pipeline? | A2 lowest latency (1.23 ms); D highest throughput (1665 fps) |
| [Transport](docs/transport.md) | Which shared memory, and when? | CPU data → sys-shm (3.6×); GPU data → CUDA IPC (3×) |
| [Batching](docs/batching.md) | Is dynamic batching free? | No — 37% cheaper GPU/frame, paid in 6–74 ms queue wait |
| [Contention](docs/contention.md) | What if N pipelines share the GPU? | MPS gives A2 +32% but B2 nothing — Triton's edge inverts once MPS is on |
| [Stage decomposition](docs/stage-decomposition.md) | Where does the time actually go? | With zero-copy, Triton's whole framework costs 0.18 ms |
| [DeepStream](docs/deepstream.md) | What does NVIDIA's own stack do? | 1.50 ms/frame, zero custom code, source-bound at 5.4% GPU |
| [Reproduce](docs/reproduce.md) | How do I run this myself? | Four commands from a clean clone |
| [CUDA graphs](docs/cuda-graphs.md) | Is the engine ceiling real? | No — ~0.13 ms of it is launch overhead; graphs give +15–27% |
| [Triton tuning](docs/triton-tuning.md) | Were the Triton knobs right? | `count:2` validated (+30% over 1); graphs and instances are substitutes |
| [Accuracy](docs/accuracy.md) | Does the pipeline preserve the model? | Yes (−0.06% vs PyTorch) — but nearest-neighbour resize costs every flow ~0.6 mAP |
| [Precision](docs/precision.md) | What could INT8 / sparsity buy? | INT8 ceiling ~+34%; forced 2:4 sparsity ~+1% (speed only, no accuracy) |
| [Roadmap](docs/roadmap.md) | What is *not* covered? | No accuracy axis, no INT8; MPS and CUDA graphs now measured |

**Two reading paths.** Start-to-finish: STORY → Methodology → Results → the four
analysis docs. Or jump straight to the row above that matches your question.

Supporting material: [`results/`](results/README.md) (raw data + provenance) ·
[`docker/`](docker/README.md) (container recipes).

---

## The contenders

Naming used everywhere in this repo:

| ID | Name | Decode | Preprocess | Transport | Inference | Postprocess |
|----|------|--------|-----------|-----------|-----------|-------------|
| **A1** | **C++ TRT, CPU path** | NVDEC → CPU NV12 | swscale + CPU loops | none (in-proc) | TensorRT `enqueueV3` | CPU NMS |
| **A2** | **C++ TRT, full-CUDA** | NVDEC **zero-copy** (GPU frames) | fused CUDA kernel → TRT input buffer | none (in-proc) | TensorRT, same CUDA stream | GPU compact kernel, KB of boxes to host |
| **B1** | **Triton + C++ client** | NVDEC → CPU NV12 | swscale + CPU loops | gRPC (raw payloads) | Triton `tensorrt_plan` | CPU NMS |
| **B2** | **Triton + C++ client + CUDA shm** | NVDEC **zero-copy** | CUDA kernel → **CUDA IPC shm region** | gRPC (handles only) | Triton `tensorrt_plan` | GPU compact kernel on shm output |
| **C1** | Triton + Python torch *(superseded)* | ffmpeg pipe | torch GPU + H2D/D2H | gRPC raw | Triton | torch + NMS |
| **C2** | **Triton + Python numpy + sys-shm** | ffmpeg pipe | **pure numpy** (no GPU) | **system shared memory** | Triton `tensorrt_plan` | numpy NMS, multiprocessing |
| **D** | **Triton + async + dynamic batching** | NVDEC **zero-copy** | CUDA kernel → CUDA shm | gRPC async, 8 in-flight | Triton **batch-8 engine** | GPU compact kernel |
| **E1** | **DeepStream, 1 stream** | `nvv4l2decoder` (NVMM) | `nvinfer` (1/255, AR, sym-pad) | none (GStreamer) | TensorRT via `nvinfer` | marcoslucianops YOLO parser |
| **E2** | **DeepStream, N-stream batched** | same × N | `nvstreammux` batch | none (GStreamer) | TensorRT batch-N | same |

**Reference ceiling**: `trtexec` on the batch-1 engine = **0.97 ms/frame,
1028 fps**; the batch-8 engine = **0.61 ms/frame** (1630 fps effective). Every
number below is the story of what stands between your camera and that 0.97 ms.

## Headline numbers

Capacity mode, YOLOv8s FP16. Full tables: [Results](docs/results.md).

| Flow | Best latency (p50) | Best throughput | In one line |
|---|---|---|---|
| **A2** C++ TRT full-CUDA | **1.23 ms** | 1219 fps | Lowest latency, in-process control, zero dependencies |
| **B2** Triton + CUDA shm | 1.28 ms | 1131 fps | Triton without the tax — ≈A2 latency, plus server ops |
| **C2** Triton + numpy + sys-shm | 1.69 ms | 1038 fps | Python within 0.4 ms of C++ |
| **D** Triton async + batch-8 | 6.2–74 ms wait | **1665–1816 fps** | Highest throughput; latency is the price |
| **E1** DeepStream | 1.50 ms | 45 fps/stream (source-bound) | Zero custom code, integrated NVDEC→infer |
| B1 raw gRPC · C1 torch | 3.27 / 6.4 ms | 496 / 225 fps | What "just use the server" costs if you feed it naively |

**The headline conclusion**: fed properly — CUDA-shm zero-copy plus async clients
keeping batches full — Triton **beats the hand-rolled C++ pipeline by ~50% on
throughput**. The same server with a naive client loses by 8×. Triton's framework
is only as good as its client; its scheduler is the irreplaceable part.

## Decision guide

| Scenario | Pick | Latency (p50/frame) | Throughput | Why this pick |
|---|---|---|---|---|
| Live camera, lowest latency, full control | **A2** — C++ TRT full-CUDA | **1.23 ms** | 809 fps | fastest per frame; zero dependencies |
| Live multi-stream, want a server | **B2** — Triton + CUDA shm | 1.28 ms | 1131 fps | ≈A2 latency + Triton ops (reload, metrics) |
| Python-only team | **C2** — Triton + numpy + sys-shm | 1.69 ms | 1038 fps | within 0.4 ms of C++ with pure-Python client |
| Offline / max throughput, latency negotiable | **D** — Triton async, batch-8 | 6.2–74 ms wait (0.61 ms GPU service) | 1640–1665 fps | cheapest GPU service per frame (0.61 ms) |
| Multi-model production serving | **D** — 3 models × async batch-8 | 29–59 ms wait | **1799–1816 fps** | Triton scheduler has no hand-rolled equivalent |
| Edge product, NVIDIA-supported stack | **E** — DeepStream | 1.50 ms (1 stream) | 45 fps/stream (source-bound) | zero custom code; NVDEC→infer integrated |

At 30 FPS live video (33.3 ms budget) *every* flow keeps up — the differences are
in latency headroom, not capability.

## Six things we'd tell ourselves at the start

1. **A benchmark that saturates the source measures the source.**
2. **Latency and throughput are different products** — D's 1665 fps and its 74 ms
   queue wait are the same number read two ways.
3. **The GPU is almost never the bottleneck at the edge.** The fight is over PCIe
   round trips, serialization, and interpreter locks.
4. **Match preprocessing bit-for-bit before comparing pipelines.**
5. **Dynamic batching requires async in-flight clients** — sync clients pay for
   the window and never collect the benefit.
6. **Shared memory: pick by data location.** CPU → system shm; GPU → CUDA IPC.

## Reproduce

```bash
docker/build.sh                              # build both images from source
scripts/export_models.sh                     # pt -> ONNX -> FP16 .plan
scripts/make_frames.sh videos/real.mp4 500   # capacity-replay input
scripts/benchmark_v2.sh 10 3                  # the full 4-arm sweep
```

Details, per-arm commands, and the source→binary map: [Reproduce](docs/reproduce.md).

## Scope

This study covers the **serving and transport layer**, at FP16, at 640×640, for
detection on a single GPU. It now has an [accuracy axis](docs/accuracy.md), and does not yet
cover in-graph NMS, input-resolution scaling, or application-level tricks like
detect-and-track. MPS, CUDA graphs and the INT8/sparsity *speed ceilings* have
since been measured; calibrated INT8 still needs the accuracy axis. Those limits are enumerated
honestly in the [Roadmap](docs/roadmap.md).
