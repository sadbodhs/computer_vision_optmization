# Triton vs Pure TensorRT vs DeepStream — Inference Pipeline Benchmark

**📖 [Read this as a site](https://sadbodhs.github.io/computer_vision_optimization/overview/)** — searchable, with an interactive version
of the chart below.

One question, answered with measurements: *for the same YOLO model on the same
GPU, which serving pipeline processes a frame fastest, and which delivers the
most frames per second?*

Hardware: RTX 3090 · Triton 24.12 · TensorRT 10.7/10.3 · DeepStream 7.1 ·
**YOLOv8s** FP16 @ 640×640 (identical ONNX, md5-verified across flows).
Everything runs in Docker.

**New to this?** Start with [Introduction](docs/introduction.md) — what these
stacks are, why the choice is hard, and why "fastest" is not a well-formed
question. Building something specific? [Use cases](docs/use-cases.md) maps
robotics, surveillance, manufacturing and the rest onto a pipeline. Then
[STORY.md](STORY.md) for the narrative of how the first answer
turned out to be wrong, and what it took to get a trustworthy one.

![Latency versus throughput for every flow](docs/img/pareto-latency-throughput.png)

*Up and to the left is better.* Each line is one pipeline swept over concurrency
1–16. Full tables in [Results](docs/results.md).

---

## Index

| Doc | Question it answers | Headline finding |
|---|---|---|
| [Introduction](docs/introduction.md) | Why does this choice even matter? | The plumbing costs more than the model; "fastest" depends on latency vs throughput |
| [Use cases](docs/use-cases.md) | Which of these is for *my* problem? | Four binding constraints; in three of them the lowest-latency pipeline is the wrong pick |
| [Methodology](docs/methodology.md) | How were these numbers produced? | A benchmark that saturates the source measures the source |
| [Results](docs/results.md) | How fast is each pipeline? | A2 lowest latency (1.23 ms); D highest throughput (1665 fps) |
| [DeepStream](docs/deepstream.md) | What does NVIDIA's own stack do? | 1.50 ms/frame, zero custom code, source-bound at 5.4% GPU |
| [Transport](docs/transport.md) | Which shared memory, and when? | CPU data → sys-shm (3.6×); GPU data → CUDA IPC (3×) |
| [Stage decomposition](docs/stage-decomposition.md) | Where does the time actually go? | With zero-copy, Triton's whole framework costs 0.18 ms |
| [Model cost](docs/model-scaling.md) | When does the plumbing stop mattering? | Non-engine cost is fixed at 0.256 ms; A2 crosses under 10% at a 2.3 ms engine, B1 not until 10.4 ms |
| [Across architectures](docs/model-zoo.md) | What sets the plumbing cost? | Output bytes / 25 GB/s — and output shape is an architectural choice: DeepLabV3 pays 57% transport, SegFormer-B0 14%, at the same engine cost. Params predict engine time at r&sup2; 0.80 in bulk, but ResNet50 and YOLO11l share 25M and differ 5.8x |
| [Batching](docs/batching.md) | Is dynamic batching free? | No — 37% cheaper GPU/frame, paid in 6–74 ms queue wait |
| [Triton tuning](docs/triton-tuning.md) | Were the Triton knobs right? | `count:2` validated (+30% over 1); graphs and instances are substitutes |
| [CUDA graphs](docs/cuda-graphs.md) | Is the engine ceiling real? | No — ~0.13 ms of it is launch overhead; +11.8% inside A2, and its plateau rises 1167→1249 fps |
| [In-graph NMS](docs/in-graph-nms.md) | Is the 2.82 MB output worth removing? | On raw gRPC yes — +33% despite a 19% slower engine; on zero-copy paths, no |
| [Contention](docs/contention.md) | What if N pipelines share the GPU? | MPS gives A2 +32% but B2 nothing — Triton's edge inverts once MPS is on |
| [Accuracy](docs/accuracy.md) | Does the pipeline preserve the model? | Yes, for every model and both engine shapes; batching is accuracy-free; nearest-neighbour resize cost ~0.6 mAP (fixed) |
| [Precision](docs/precision.md) | Is INT8 worth it? | +32.8% throughput for −1.55 mAP — and it beats downgrading the model; sparsity ~+1%, not worth it |
| [Reproduce](docs/reproduce.md) | How do I run this myself? | Four commands from a clean clone |
| [Roadmap](docs/roadmap.md) | What is *not* covered? | No accuracy axis, no INT8; MPS and CUDA graphs now measured |

**Two reading paths.** Start-to-finish: the table above is in reading order —
why it matters, then how it was measured, then what was measured, then what it
costs. Or jump straight to the row that matches your question.

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

This study covers the **serving and transport layer**, at FP16, for detection on a
single GPU — no longer at a single engine cost, since it now sweeps a
[model-cost ladder](docs/model-scaling.md) and
[22 models across four tasks](docs/model-zoo.md). It has an
[accuracy axis](docs/accuracy.md), including calibrated INT8. MPS, CUDA graphs,
[in-graph NMS](docs/in-graph-nms.md) and the INT8/sparsity ceilings are measured.

**Still not covered:** input resolution (deliberately — a well-understood
quadratic), application-level tricks like detect-and-track, multi-GPU, and an
accuracy axis for segmentation. Those limits are enumerated honestly in the
[Roadmap](docs/roadmap.md).
