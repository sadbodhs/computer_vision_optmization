# Triton vs Pure TensorRT vs DeepStream — Inference Pipeline Benchmark

One question, answered with measurements: *for the same YOLO model on the same
GPU, which serving pipeline processes a frame fastest, and which delivers the
most frames per second?*

Hardware: RTX 3090 · Triton 24.12 · TensorRT 10.7/10.3 · DeepStream 7.1 ·
3× YOLO FP16 engines @ 640×640 (identical ONNX, md5-verified across flows).
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

## 2. Capacity results — every cell shows `fps · ms-per-frame`

`ms/frame` = p50 end-to-end latency for one frame at that concurrency.
For **D** it splits into *queue wait* (batch assembly) and *GPU service*
(4.84 ms ÷ 8 = 0.61 ms) — both shown.

| Concurrency | A1 | A2 | B1 | B2 | C1 (torch) | C2 (numpy) | D (async batch-8) |
|---|---|---|---|---|---|---|---|
| 1 | 456 · 1.32 ms | **809 · 1.23 ms** | 222 · 3.27 ms | 654 · 1.28 ms | 144 · 6.4 ms | 469 · 1.69 ms | 1041 · **6.2 ms wait** (0.61 ms svc) |
| 2 | 793 · 1.50 ms | — | 368 · 4.03 ms | 951 · 1.83 ms | 198 · 9.6 ms | 736 · 2.20 ms | 1136 · 11.1 ms wait |
| 4 | 956 · 3.31 ms | — | 472 · 6.90 ms | 1092 · 4.02 ms | 218 · 10.8 ms | 986 · 3.15 ms | 1378 · 19.2 ms wait |
| 8 | 1055 · 6.43 ms | — | 488 · 14.6 ms | 1131 · 6.60 ms | 222 · 12.0 ms | 1037 · 6.86 ms | 1640 · 36.6 ms wait |
| 16 | 1205 · 11.8 ms | — | 496 · 30.4 ms | 1128 · 13.5 ms | 225 · 14.7 ms | 1038 · 14.5 ms | 1665 · **73.9 ms wait** (0.61 ms svc) |

**How to read D:** its fps advantage is *not* a faster pipeline — it processes 8
frames per GPU pass. Each frame's **GPU service time is 0.61 ms** (37% cheaper
than batch-1's 0.97 ms), but each frame **waits 6–74 ms** for its batch-mates.
Throughput ≠ latency: D wins the first column, A2/B2 win the second.

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

## 7. Decision guide

| Scenario | Pick | Why |
|---|---|---|
| Live camera, lowest latency, full control | **A2** | 1.23 ms, no dependencies |
| Live multi-stream, want a server | **B2** | 1.28 ms + Triton ops (reload, metrics) |
| Python-only team | **C2** | 1038 fps; numpy + sys-shm + processes |
| Offline / max throughput / many users | **D** | 0.61 ms/frame GPU cost; latency negotiable |
| Edge product, NVIDIA-supported stack | **E** | 1.5 ms single, batched multi-stream, zero custom code |

---

## 8. Reproduce

```bash
docker restart triton-server        # image triton-bench:v3 (Triton arm)
# A2: C++ full-CUDA
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_pipeline_cuda --engine model.plan --mode file --file frames.bin --streams 1 --duration 10'
# B2: Triton + CUDA shm
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_cuda --mode file --file frames.bin --model yolov8s --streams 1 --duration 10'
# C2: Python numpy + sys-shm + processes
docker exec triton-server bash -c 'cd /work && python3 client_v2.py --mode file --file cpp/build/frames.bin --model yolov8s --transfer sys --streams 4 --processes 4 --duration 8'
# D: async + dynamic batching
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_async --model yolov8s_dyn --file frames.bin --streams 8 --duration 10'
# E: DeepStream (ds-build container)
docker exec ds-build bash -c 'cd /tmp && ./ds_bench --config /opt/ds/model/yolov8s/config_infer_primary_yolov8s.txt --streams 1 --batch 1 --duration 15'
```

Engines are gitignored — regenerate from the same ONNX (ultralytics export →
`trtexec --fp16`) or pull the committed container images.

## 9. Artifacts

- `cpp/src/main_cuda.cu` — A2 · `cpp/src/main.cpp` — A1
- `cpp/src/grpc_client_cuda.cu` — B2/D(sync) · `cpp/src/grpc_client.cpp` — B1
- `cpp/src/grpc_async_client.cu` — D (async, 8 in-flight)
- `cpp/src/ds_bench.cpp` — E1/E2 (DeepStream, RTSP + file modes)
- `triton/client_v2.py` — C2 (+ `--transfer raw|sys|cuda`)
- `triton/models/` — Triton configs · `results/` — raw data + `comparison_tables.md` + `parallel_contention.md`
- Images: `triton-bench:v3`, `ds-build` (DeepStream + YOLO parser)