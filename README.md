# Triton vs Pure TensorRT vs DeepStream — Inference Pipeline Benchmark

One question, answered with measurements: *for the same YOLO model on the same
GPU, which serving pipeline processes a frame fastest, and which delivers the
most frames per second?*

Hardware: RTX 3090 · Triton 24.12 · TensorRT 10.7/10.3 · DeepStream 7.1 ·
**YOLOv8s** FP16 @ 640×640 (identical ONNX, md5-verified across flows).
Everything runs in Docker.

---

## 1. The contenders (naming used everywhere in this repo)

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

**The zero-copy memory chain** (A2 / B2 / D): NVDEC NV12 (GPU) → kernel reads in
place → writes TRT input / shm (GPU) → infer → compact kernel (GPU) → only ~KB of
candidate boxes cross to host. One CUDA stream end-to-end.

**Two measurement modes:**
- **Capacity** — preprocessed frames replayed flat-out from disk. Measures what the
  pipeline can actually do, with no camera pacing. This is where framework
  overhead shows.
- **RTSP end-to-end** — live 30 fps sources. Measures "can it keep up + latency",
  bounded by the source, not the pipeline.

Reference ceiling: `trtexec` on the batch-1 engine = **0.97 ms/frame, 1028 fps**;
the batch-8 engine = 4.84 ms/8 frames = **0.61 ms/frame (1630 fps effective)**.

---

## 2. Capacity results

**Flow key** (details in Section 1): **A1** C++ TRT, CPU path · **A2** C++ TRT,
full-CUDA zero-copy · **B1** Triton + C++ client, raw gRPC · **B2** Triton + C++
client, CUDA shm · **C1** Triton + Python torch *(superseded)* · **C2** Triton +
Python numpy, sys-shm · **D** Triton async in-flight + batch-8 engines ·
*(E = DeepStream, see its own section)*. All run the **same YOLOv8s FP16 model**.

Two numbers per cell. **`fps↑` = throughput (higher is better) · `ms↓` = per-frame
latency p50 (lower is better)**. Best value per row is **bold**.

| Concurrency | A1 — C++ TRT, CPU path | A2 — C++ TRT, full-CUDA | B1 — Triton+C++ gRPC | B2 — Triton+C++ CUDA-shm | C1 — Triton+PyTorch | C2 — Triton+Py numpy+shm | D — Triton async, batch-8 |
|---|---|---|---|---|---|---|---|
| 1 | 456↑ · 1.32↓ | 809↑ · **1.23↓** | 222↑ · 3.27↓ | 654↑ · 1.28↓ | 144↑ · 6.4↓ | 469↑ · 1.69↓ | **1041↑** · wait 6.2↓svc 0.61 |
| 2 | 793↑ · 1.50↓ | **1219↑ · 1.60↓** | 368↑ · 4.03↓ | 951↑ · 1.83↓ | 198↑ · 9.6↓ | 736↑ · 2.20↓ | **1136↑** · wait 11.1 svc 0.61 |
| 4 | 956↑ · 3.31↓ | **1175↑ · 3.40↓** | 472↑ · 6.90↓ | **1092↑ · 4.02↓** | 218↑ · 10.8↓ | 986↑ · 3.15↓ | **1378↑** · wait 19.2 svc 0.61 |
| 8 | 1055↑ · 6.43↓ | **1187↑ · 6.72↓** | 488↑ · 14.6↓ | **1131↑ · 6.60↓** | 222↑ · 12.0↓ | 1037↑ · 6.86↓ | **1640↑** · wait 36.6 svc 0.61 |
| 16 | 1205↑ · 11.8↓ | 1160↑ · 13.8↓ | 496↑ · 30.4↓ | **1128↑ · 13.5↓** | 225↑ · 14.7↓ | 1038↑ · 14.5↓ | **1665↑** · wait 73.9 svc 0.61 |

**How to read this table**

- `fps↑ · ms↓` — two independent scores per pipeline: *throughput* and *latency*.
  A pipeline can win one and lose the other (that is the whole Triton-vs-C++ story).
- **D reads differently on purpose**: `1041↑ · wait 6.2 · svc 0.61` means each frame
  **waits 6.2 ms** for its batch to fill, then the GPU **services it in 0.61 ms**
  (8 frames share one 4.84 ms pass — 37% cheaper per frame than batch-1's 0.97 ms).
  D's high fps is *bought with queue wait* — at conc=16 a frame waits 74 ms,
  more than 2 camera frames at 30 FPS.
- **Row winners**: for throughput, D (1665) > A2 (1219, saturates ~1200 from conc=2)
  > B2 (1131) > C2 (1038) > B1 (496) > C1 (225). For latency, A2 (1.23 ms at
  conc=1, 1.60 at conc=2) < B2 < C2 < D's wait (6.2-73.9). A2's notable shape: it
  hits its ~1200 fps plateau already at conc=2 (multiple CUDA streams overlap
  H2D/compute/D2H) and latency then scales linearly with N — a batch-1 engine
  cannot exceed its 1028 qps core for long. If your feed is 30 FPS live video,
  latency under 33 ms is what matters — only D at conc≥4 starts eating multiple
  frame budgets.

GPU util at conc=16: A2 81% · B2 84% · C2 82% · D 79% (Triton-without-shm was 52%).

---

## 3. Latency-first view (live 30 FPS camera: budget = 33.3 ms/frame)

| Flow | p50 | % of budget | p95 | fps @ 1 stream | fps @ 3 streams |
|---|---|---|---|---|---|
| A2 | **1.25 ms** | 3.8% | 1.27 ms | ~20-23 | — |
| **E1 DeepStream** | **1.50 ms** | 4.5% | 1.59 ms | 45 | — |
| B2 | 1.55 ms | 4.7% | 1.73 ms | ~12 | — |
| C2 | ~2-8 ms | 6-24% | — | ~8-20 | — |
| D | 6.2-74 ms | 19-220% | — | — | — |

At 30 FPS *every* flow keeps up; the differences are in latency headroom.

---

## 4. Transfer-variant findings (which shared memory, and when)

| Variant | Python client (CPU data) | C++ client (GPU data) | Why |
|---|---|---|---|
| raw gRPC | 130 fps · 7.1 ms | 222 fps · 3.3 ms | protobuf serialize + socket copies |
| **system shm** | **469 fps · 1.69 ms** | — | server reads CPU shm directly; 2 copies gone |
| **CUDA shm** | 330 fps · 2.47 ms | **654 fps · 1.28 ms** | zero-copy only when bytes already on GPU |

- Data on GPU (C++ kernels) → **CUDA IPC shm** (3× vs raw).
- Data on CPU (numpy) → **system shm** (3.6× vs raw); CUDA shm just moves the H2D
  copy client-side.
- Region management: one `cudaMalloc` + one IPC handle **per stream for its
  lifetime** — the shm region *is* the kernel's output buffer; never allocate
  per frame.

---

## 5. Multi-instance contention (N pipelines, one GPU)

| N instances | A2 latency each | B2 latency each | Total fps (A2 / B2) |
|---|---|---|---|
| 1 | 1.24 ms | 1.28 ms | 809 / 654 |
| 2 | 2.10 ms (+69%) | 1.86 ms (+45%) | 950 / 891 |
| 3 | 3.13 ms (+153%) | 2.25-2.30 ms (+76%) | 957 / 985 |

- Latency grows ~linearly with N; total throughput plateaus at the engine cap.
- **Triton shares the GPU more gracefully** (+76% vs +153% at N=3): its shared
  engine pool avoids per-process CUDA context switching.
- C2 in-process streams: +2-3% latency at N=3 (threads share one connection).

---

## 6. DeepStream (E) — NVIDIA's integrated stack, same weights

Same ONNX (md5-verified), engines rebuilt in-container (TRT 10.3; trtexec parity
1030 vs 1028 qps — negligible). `nvv4l2decoder → nvstreammux → nvinfer → parser`.

| Config | Total fps | fps/stream | p50 | GPU util | Reading |
|---|---|---|---|---|---|
| E1: 1 stream (RTSP 30fps) | 45 | 45 | **1.50 ms** | — | decode+infer+parse, no RPC |
| E2: 3 streams | 104 | 34.5 | 3.1 ms | — | batch=3 assembles fast |
| E2: 6 streams | 169 | 28 | 32.5 ms | — | batch wait dominates |
| E2: 8 streams | 226 | 28 | 32.7 ms | **5.4%** | source-bound: 240fps demand, 94% delivered |

E2's 33 ms p50 is the **streammux batch-assembly wait** (batch of 8 fills every
~33 ms at 30 fps sources), not inference. E2 is source-bound — its capacity-mode
sweep (file replay) is future work to compare against D directly.

---

## 7. Triton at its absolute best (zero-copy + async + batching, all 3 models)

D's stack (CUDA shm + async in-flight + batch-8 engines) is "B2 with dynamic
batching". Run across **all three models concurrently** — Triton's real
production scenario, where its scheduler has no hand-rolled equivalent:

| Scenario | Total fps | Latency p50 | vs hand-rolled best |
|---|---|---|---|
| D: 1 model (yolov8s), conc=16 | 1665 | 73.9 ms wait | +38% vs A2 (1205) |
| **D: 3 models × 6 streams each** | **1799** | 29.4 ms | +49% vs A2 |
| **D: 3 models × 6 streams (conc=18)** | **1816** | 58.6 ms | +51% vs A2 |
| D: yolov8n alone, conc=8 | 1988 | 24.1 ms | engine cap 2960 effective |

**This is the answer to "shouldn't Triton win?"**: yes — when fed properly
(CUDA shm zero-copy, async clients keeping batches full), Triton **beats the
hand-rolled C++ pipeline by ~50% on throughput**. The same server with naive
clients (C1: 225 fps) loses 8×. Triton's framework is only as good as its
client; its scheduler is the irreplaceable part.

## 8. Where does the time go? — stage decomposition (conc=1, YOLOv8s)

Per-stage wall time for one frame, measured inside each flow with per-stage timers
(`stages_ms` in every binary's JSON output):

### Capacity mode (preprocessed frames, no decode — pure pipeline cost)

| Stage | A2 (C++ TRT) | B2 (Triton, CUDA shm) | C2 (Python numpy) |
|---|---|---|---|
| Host→Device transfer | **0.25 ms** (H2D, pinned+async) | 0 (zero-copy shm) | 0 (sys-shm, server copies) |
| Preprocess | 0 (already preprocessed input) | 0 | 0 |
| Inference (GPU) + output handling | **0.98 ms** (incl. compact kernel) | 1.16 ms (gRPC round trip incl. server infer) | ~5.1 ms (gRPC + serialize) |
| Postprocess (NMS, CPU) | 0.01 ms | 0.00 ms | ~1-3 ms (numpy NMS) |
| **Total** | **1.25 ms** | **1.26 ms** | **~6.4 ms** |

Reading: the engine itself is 0.97-0.98 ms everywhere. A2 adds 0.25 ms H2D + 0.01 NMS.
B2's zero-copy shm makes the *entire* Triton framework cost 0.18 ms over A2.
C2's gap is client-side: numpy serialization into gRPC (~3 ms) + Python NMS (~2 ms).

### RTSP end-to-end mode (adds decode; includes source pacing)

| Stage | A2 (C++ full-CUDA) | B2 (Triton, CUDA shm) | C2 (Python numpy) |
|---|---|---|---|
| Decode (NVDEC / pipe) | 0.15-0.4 ms compute (rest = waiting for 30fps frame) | same | ffmpeg pipe ≈ 33.3 ms wall (pacing) |
| Preprocess | **0.15 ms** (fused CUDA kernel) | 0.15 ms | **1.26 ms** (numpy CPU) |
| Infer + transfer | 1.22 ms | 1.71 ms (gRPC) | 5.08 ms (gRPC raw) |
| Postprocess | 0.001 ms | 0.001 ms | ~1-3 ms |
| **GPU-only total** | **≈1.4 ms** | **≈1.9 ms** | **≈7.5 ms** |

Key facts:
- **Decode is free at 30 FPS**: NVDEC decodes a frame in ~0.2-0.4 ms; the rest of the
  33 ms frame interval the decoder *waits* for the next packet. In the C++ flows the
  "decode" timer shows ~31.9 ms — that is socket-read blocking, not compute.
- **Preprocess**: the fused CUDA kernel (0.15 ms) is 8× faster than numpy-on-CPU
  (1.26 ms) — and numpy is already the *fast* Python option (torch was worse).
- **The framework tax is only in the infer stage**: B2 pays 1.16-1.71 ms where A2
  pays 0.98-1.22 — with CUDA shm, Triton's entire server costs ~0.2-0.5 ms per frame.
- **CPU vs GPU split**: in A2, ~99% of pipeline time is GPU work; in C2, roughly
  60% of client time is CPU (numpy serialize + NMS) — the GPU is idle waiting.

## 9. Decision guide

| Scenario | Pick | Latency (p50/frame) | Throughput | Why this pick |
|---|---|---|---|---|
| Live camera, lowest latency, full control | **A2** — C++ TRT full-CUDA | **1.23 ms** | 809 fps | fastest per frame; zero dependencies |
| Live multi-stream, want a server | **B2** — Triton + CUDA shm | 1.28 ms | 1131 fps | ≈A2 latency + Triton ops (reload, metrics) |
| Python-only team | **C2** — Triton + numpy + sys-shm | 1.69 ms | 1038 fps | within 0.4 ms of C++ with pure-Python client |
| Offline / max throughput, latency negotiable | **D** — Triton async, batch-8 | 6.2–74 ms wait (0.61 ms GPU service) | 1640–1665 fps | cheapest GPU service per frame (0.61 ms) |
| Multi-model production serving | **D** — 3 models × async batch-8 | 29–59 ms wait | **1799–1816 fps** | Triton scheduler has no hand-rolled equivalent |
| Edge product, NVIDIA-supported stack | **E** — DeepStream | 1.50 ms (1 stream) | 45 fps/stream (source-bound) | zero custom code; NVDEC→infer integrated |

---

## 10. Reproduce

```bash
docker restart triton-server        # image triton-bench:v3 (Triton arm)
# A2: C++ full-CUDA
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_pipeline_cuda --engine model.plan --mode file --file frames.bin --streams 1 --duration 10'
# B2: Triton + CUDA shm
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_cuda --mode file --file frames.bin --model yolov8s --streams 1 --duration 10'
# C2: Python numpy + sys-shm + processes
docker exec triton-server bash -c 'cd /work && python3 client_v2.py --mode file --file cpp/build/frames.bin --model yolov8s --transfer sys --streams 4 --processes 4 --duration 8'
# D: async + dynamic batching (single model)
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_async --model yolov8s_dyn --file frames.bin --streams 8 --duration 10'
# D multi-model (Triton production scenario, 3 models x async batching)
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_async --models yolov8n_dyn,yolov8s_dyn,yolo11n_dyn --file frames.bin --streams 18 --duration 10'
# E: DeepStream (ds-build container)
docker exec ds-build bash -c 'cd /tmp && ./ds_bench --config /opt/ds/model/yolov8s/config_infer_primary_yolov8s.txt --streams 1 --batch 1 --duration 15'
```

Engines are gitignored — regenerate from the same ONNX (ultralytics export →
`trtexec --fp16`) or pull the committed container images.

## 11. Appendix — model & preprocessing contract

| Item | Value |
|---|---|
| **Model** | YOLOv8s (Ultralytics), COCO 80 classes |
| **Input** | 640×640, RGB, /255 (sigmoid baked into export) |
| **Precision** | FP16 engines from the same ONNX (md5-verified) |
| **Engine format** | `.plan` (Triton) / `.engine` (DeepStream) — same serialization |
| **Output** | `[1, 84, 8400]`, conf 0.25, class-aware NMS IoU 0.45 |
| **Also tested** | YOLOv8n (1490 qps) · YOLO11n (1259 qps) |
| **Preproc contract** | centered letterbox, pad 114, BGR→RGB, /255 — identical in every flow |

## 12. Artifacts

- `cpp/src/main_cuda.cu` — A2 · `cpp/src/main.cpp` — A1
- `cpp/src/grpc_client_cuda.cu` — B2/D(sync) · `cpp/src/grpc_client.cpp` — B1
- `cpp/src/grpc_async_client.cu` — D (async, 8 in-flight)
- `cpp/src/ds_bench.cpp` — E1/E2 (DeepStream, RTSP + file modes)
- `triton/client_v2.py` — C2 (+ `--transfer raw|sys|cuda`)
- `triton/models/` — Triton configs · `results/` — raw data + `comparison_tables.md` + `parallel_contention.md`
- Images: `triton-bench:v3`, `ds-build` (DeepStream + YOLO parser)