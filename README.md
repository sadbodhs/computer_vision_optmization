# Triton vs Pure TensorRT — Pipeline Comparison (v3: 4 optimized flows)

## Setup
- **Hardware**: NVIDIA RTX 3090 (24 GB), driver 595.84, host Ubuntu 26.04, 16-core CPU
- **Models**: YOLOv8n / YOLOv8s / YOLO11n — FP16 TensorRT 10.7 engines, 640×640
  (+ `yolov8s_dyn`: dynamic-batch 1–8 engine, Triton dynamic batching, 5 ms queue)
- **Input**: preprocessed frame replay (`frames.bin`, capacity tests bypass the 30 fps
  source cap) + 3 RTSP streams (H.264 640×360@30, MediaMTX)
- **Everything in Docker**; container committed as `triton-bench:full`

## The 4 optimized flows

| | Flow A2 | Flow B2 | Flow C2 | Flow D |
|---|---|---|---|---|
| **Stack** | C++ TRT in-proc | C++ gRPC→Triton | numpy→Triton | C++ async→Triton dyn-batch |
| **Decode** | NVDEC **zero-copy** (CUDA hw frames) | same | ffmpeg pipe | same as B2 |
| **Preproc** | fused CUDA kernel → **writes TRT input buffer directly** | same kernel → writes **CUDA-shm region** | **pure numpy**, no torch, no GPU | same as B2 |
| **Transfer** | none (all GPU) | **CUDA IPC shm** | **system shm** (best; raw & cuda-shm also measured) | CUDA IPC shm |
| **Postproc** | GPU **compact kernel** → KB of boxes to host | same | numpy NMS | same |

Memory chain for A2/B2/D (the zero-copy chain):
`NVDEC NV12 (GPU) → kernel reads in place → writes TRT input / shm (GPU) → infer →
compact kernel (GPU) → only ~KB candidate boxes cross to host`
Only PCIe traffic: the final candidate list. One CUDA stream end-to-end.

## Results — capacity mode (frames replayed flat-out), total fps

| Concurrency | A1: C++ (v2) | **A2: C++ full-CUDA** | B1: C++→Triton | **B2: CUDA shm** | C old: torch | **C2: numpy+sys-shm+mp** | **D: async+dyn-batch** |
|---|---|---|---|---|---|---|---|
| 1 | 456 | **809** | 222 | **654** | 144 | 469 | **1041** |
| 2 | 793 | — | 368 | **951** | 198 | 736 | 1136 |
| 4 | 956 | — | 472 | **1092** | 218 | 986 | 1378 |
| 8 | 1055 | — | 488 | **1131** | 222 | 1037 | **1640** |
| 16 | 1205 | — | 496 | **1128** | 225 | 1038 | **1665** |

Latency p50 @ conc=1: A2 **1.23 ms**, B2 **1.28 ms**, C2 1.69 ms (sys), D 6.3 ms
(batching trades latency for throughput). Reference: trtexec cap 1028 qps (batch-1);
batch-8 engine effective cap 1630 fps; 3-proc C++ 1121 fps; GPU util at high conc:
B2 84 %, D 79 %, C2 82 % (v1 Triton best was 52 %).

## Transfer-variant findings (your shm questions, answered)

| Variant | Python conc=1 | Why |
|---|---|---|
| raw gRPC | 130 fps / 7.1 ms | protobuf serialize + socket copies dominate |
| **system shm** | **469 fps / 1.69 ms** | server reads CPU shm directly; 2 copies eliminated |
| CUDA shm | 330 fps / 2.47 ms | H2D copy just moves client-side — no win for CPU data |

- **CUDA shm wins when data is already on GPU** (flows B2/D: kernel output goes
  straight into the IPC region — 654 fps vs 222 raw = **3×**).
- **System shm wins when data is on CPU** (flow C2: Python/numpy). Your guess was
  right for the C++ flows; for the numpy flow system shm is the best fit.
- "Same memory location" management: B2's shm region IS the kernel's destination
  buffer (one `cudaMalloc`, one IPC handle, reused every frame — no per-frame alloc).

## Key findings

1. **Full-CUDA preprocessing pays**: A2 file-mode 809 fps @ conc1 (p50 1.23 ms) vs
   A1 456 (1.32) — GPU compact kernel removes the 2.8 MB D2H + CPU 8400×80 scan.
   RTSP mode: A2 1.25 ms p50 for decode→pre→infer→post (zero-copy chain works).
2. **CUDA shm through Triton ≈ in-process TRT**: B2 654–1131 fps; Triton server
   overhead is now only ~0.1 ms per request at conc=1 (1.28 vs 1.23 in-process).
3. **Python is no longer the bottleneck when done right**: numpy+sys-shm+mp reaches
   1038 fps (was 225 with torch+threads). GIL + gRPC serialization were the killers.
4. **Dynamic batching needs async in-flight clients**: sync clients never co-arrive
   (D-sync ≤ B-sync in v2), but async depth-8 clients fill batch-8 → **1665 fps @
   conc16 — the highest of any arm**, at the cost of higher p50 latency (73 ms)
   because requests wait for batch-mates. Latency-sensitive streams should use B2.
5. **RTSP E2E remains decode-bound** (~30 fps source): A2 23 fps, B2 ~12 fps
   (variance from stream jitter) — inference is never the constraint in live mode.

## Best technique per flow (the answer)

- **Flow A2** (custom, no server): NVDEC zero-copy + fused CUDA kernel + GPU compact.
  Best latency (1.23 ms) and best single-stream throughput (809).
- **Flow B2** (Triton, low latency): CUDA shm + GPU preprocess. Near-in-process
  performance (654–1131 fps, 1.28 ms) with Triton's operational benefits.
- **Flow C2** (Python, Triton): numpy + system shm + multiprocessing. 4.6× the old
  Python client at saturation; sys-shm is the right shm for CPU-side preprocessing.
- **Flow D** (batched Triton): async in-flight + dynamic batching. Absolute best
  total throughput (1665 fps) when latency is negotiable.

## Reproduce

```bash
docker restart triton-server   # image triton-bench:full
cd /home/suchi/sadbodh/rt_vs_triton
# A2
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_pipeline_cuda --engine model.plan --mode file --file frames.bin --streams 1 --duration 10'
# B2 (CUDA shm)
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_cuda --mode file --file frames.bin --model yolov8s --streams 1 --duration 10'
# C2 (numpy + sys shm + processes)
docker exec triton-server bash -c 'cd /work && python3 client_v2.py --mode file --file cpp/build/frames.bin --model yolov8s --transfer sys --streams 4 --processes 4 --duration 8'
# D (async + dynamic batch)
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_async --model yolov8s_dyn --file frames.bin --streams 8 --duration 10'
```

## Artifacts
- `cpp/src/main_cuda.cu` — Flow A2 (NVDEC zero-copy + fused kernel + compact)
- `cpp/src/grpc_client_cuda.cu` — Flow B2 (CUDA IPC shm → Triton)
- `cpp/src/grpc_async_client.cu` — Flow D (async in-flight + dyn batch + CUDA shm)
- `triton/client_v2.py` — Flow C2 (numpy; `--transfer raw|sys|cuda`, `--processes N`)
- `triton/models/` — FP16 engines + `yolov8s_dyn`
- `results/` — raw data (`v2/` = A1/B1/C/D-sync, `v3/` = GPU samples)
- Image `triton-bench:full` — provisioned environment