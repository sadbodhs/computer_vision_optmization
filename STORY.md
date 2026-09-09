# The Time We Made a GPU Wait: A Benchmarking Story

*How we benchmarked Triton Inference Server against a hand-rolled TensorRT
pipeline, discovered that our first answer was wrong, rebuilt everything,
and ended up with numbers we can actually trust.*

---

## 1. The question

It started with a simple engineering question:

> We run YOLO models on RTSP cameras. Should we serve them through **Triton
> Inference Server** (Python preprocessing, gRPC), or skip the framework and
> write a **pure TensorRT C++ pipeline** ourselves? Which one is actually
> more efficient?

Everyone has an opinion. Vendors have slides. But we had an RTX 3090, a
weekend, and Docker. So we decided to measure it.

The rules we set for ourselves:

- Everything runs in **Docker** — nothing installed on the host, so the
  experiment is reproducible and disposable.
- Same model, same weights, same preprocessing on every side of the
  comparison. Only the *plumbing* differs.
- Use the good stuff on every path: **NVDEC hardware decode**, **zero-copy
  transfers**, pinned memory, real NMS.

## 2. Building the arena

The arena came together quickly:

- **Models**: YOLOv8n, YOLOv8s, YOLO11n exported to ONNX and compiled to FP16
  TensorRT 10.7 engines (640×640, fixed batch 1).
- **Streams**: three fake RTSP cameras — ffmpeg pushing H.264 into MediaMTX —
  plus a `frames.bin` replay file of preprocessed tensors for *capacity* tests
  that bypass the 30 fps source entirely.
- **Triton**: the official 24.12 container serving the engines through the
  `tensorrt_plan` backend.
- **The challenger**: a C++ program linking libavcodec and libnvinfer
  directly — decode, letterbox, infer, NMS, no server, no RPC.

Then came the first benchmark. Triton: **14 fps**. Pure TensorRT: **29 fps**.
"C++ is 2× faster," we wrote, and almost shipped it.

## 3. The first result was a lie

Fortunately we audited ourselves before anyone else did. Three problems:

**The C++ pipeline wasn't faster — the stream was slow.** The RTSP source
delivers 30 frames per second. The C++ pipeline processed 29.1 of them. It
wasn't hitting a pipeline limit; it was hitting a *source* limit. Its true
capacity was unknown. Triton's 14 fps, meanwhile, was genuinely
pipeline-bound. Comparing a capped number to an uncapped one is meaningless.

**The two pipelines weren't doing the same work.** The C++ side preprocessed
on CPU (swscale), the Triton client preprocessed on GPU (torch) with a
*different letterbox* (bottom-padding vs centering) — which is why they
reported wildly different detection counts on the same video. The decodes
differed too (in-process vs ffmpeg subprocess pipes). We weren't comparing
frameworks; we were comparing apples to a fruit salad.

**The "optimizations" were aspirational.** We had promised zero-copy and CUDA
shared memory in the design doc. Neither existed in the code yet.

So we tore it down and rebuilt.

## 4. Rebuilding: decompose, then compare

The insight that reshaped the experiment: *the question "Triton vs TensorRT"
is really three questions stacked on top of each other.* How much does the
GPU inference cost? How much does the server add? How much does Python add?

So we built **four arms** that isolate each layer, all using identical
engines, identical preprocessing (centered letterbox, pad 114, /255, RGB),
and identical class-aware NMS:

| Arm | Pipeline | What it isolates |
|-----|----------|------------------|
| **A2** | C++ → TensorRT, fully on-GPU | the raw engine + our pre/post |
| **B2** | C++ gRPC client → Triton | + gRPC and Triton's server |
| **C2** | Python gRPC client → Triton | + Python client |
| **D** | async C++ → Triton, dynamic batching | batching's value |

And we added a **capacity mode**: instead of waiting on a 30 fps camera, each
arm replays preprocessed frames from disk as fast as the hardware allows. Now
nothing is source-capped and every number means what it says.

## 5. What the GPU actually costs

Before the pipelines, the engine itself. `trtexec` on the FP16 YOLOv8s
engine: **0.97 ms per frame, 1028 frames/second**. That's the physical
ceiling for batch-1 work on this GPU. YOLOv8n does 1490, YOLO11n does 1259.

Everything else in this story is the story of what stands between your
camera and that 0.97 ms.

## 6. The numbers

Capacity mode — total frames/second across all streams, at concurrency N
(single model, YOLOv8s):

| N | A2 (C++ in-proc) | B2 (C++→Triton, CUDA shm) | C2 (Python numpy, sys shm) | D (async, dyn-batch) |
|---|---|---|---|---|
| 1 | **809** | 654 | 469 | 1041 |
| 2 | 793 | 951 | 736 | 1136 |
| 4 | 956 | 1092 | 986 | 1378 |
| 8 | 1055 | 1131 | 1037 | **1640** |
| 16 | 1205 | 1128 | 1038 | **1665** |

And the per-frame latency — the number your video frame actually cares about:

| N=1 | Latency p50 | vs 33.3 ms frame budget at 30 FPS |
|-----|------------|-----------------------------------|
| A2 | **1.23 ms** | 3.7% |
| B2 | **1.28 ms** | 3.8% |
| C2 | 1.69 ms | 5.1% |
| D  | 6.23 ms | 18.7% (batching window) |

## 7. Where the time actually goes

Starting from the 0.97 ms engine, each architectural hop adds a tax:

```
raw engine ..................... 0.97 ms   (the GPU's honest price)
+ in-process C++ wrapper ....... +0.26 ms  → A2: 1.23 ms   (our code, CUDA kernels, NMS)
+ Triton server + gRPC ......... +2.0  ms  → B1: 3.27 ms   (copies + scheduling)
+ Python client ................ +3.1  ms  → C1: 6.4  ms   (interpreter, GIL, torch)
```

Two lessons fell out of this decomposition.

**First, Triton's tax is mostly copies, and copies are fixable.** The
server-side breakdown (from Triton's own instrumentation) showed ~1.27 ms
copying the input to the GPU and ~1.25 ms copying the output back — the
inference itself is only 1.2 ms. When we gave the client a **CUDA shared
memory** region — a buffer the client's CUDA kernel writes and the server
reads through an IPC handle, with *no copy in either direction* — B2's
throughput jumped from 222 to **654 fps**, and its latency dropped to
1.28 ms, just 0.05 ms above running the engine in-process. Triton's entire
framework, given zero-copy buffers, costs about a tenth of a millisecond.

**Second, Python's tax is not copies — it's the GIL.** Swapping the torch
preprocessing for pure numpy and adding shared memory got the Python client
to 469 fps (from 144). But its multi-stream scaling had a hard ceiling at
~225 fps with threads. Moving to processes unlocked it: **1038 fps at 16
streams**. Python can be fast; it just can't be *concurrent* inside one
interpreter.

## 8. The batching trap

Dynamic batching looks like free lunch: the batch-8 engine costs 4.84 ms for
eight frames — 0.61 ms each, **37% cheaper per frame** than running solo.
Our first attempt at flow D, though, came out *slower* than the plain
batch-1 server.

The reason is that batching is a bus, and buses need passengers to arrive
together. Our synchronous clients sent one request, blocked, read the
answer, then sent the next — so requests trickled in one at a time, the
5 ms batching window expired empty, and every request paid the window's
latency while riding alone. The fix was an **async client with eight
requests in flight** per stream. Suddenly requests co-arrive, batches fill,
and the same server delivers **1665 fps — the highest number in the entire
study, and the only configuration that beats in-process C++**.

The catch is honest and visible in the latency column: those 1665 frames
each waited 6–74 ms in the queue for their batch-mates. Batching converts
latency into throughput. For offline analytics that's the best trade in the
study. For a live camera with a 33 ms frame budget, it's a bus that misses
its stop.

## 9. What happens when multiple inferences run at once

We then ran N independent pipeline instances on the same GPU and watched
each one's latency:

| N instances | A2 latency each | B2 latency each | Total throughput |
|---|---|---|---|
| 1 | 1.24 ms | 1.28 ms | 809 / 654 fps |
| 2 | 2.10 ms | 1.86 ms | 950 / 891 fps |
| 3 | 3.13 ms | 2.25–2.30 ms | 957 / ~985 fps |

Three observations:

- **Latency grows linearly with N** — the GPU time-slices between contexts.
  1.24 → 2.10 → 3.13 ms is almost exactly 1×, 1.7×, 2.5×.
- **Total throughput plateaus at the engine cap** (~950–990 fps). Parallel
  instances divide the pie; they don't grow it. The GPU is not "more GPU"
  because there are more processes.
- **Triton shares more gracefully than raw CUDA contexts.** At N=3, B2's
  latency grew +76% while A2's grew +153%. The server's shared engine pool
  avoids the context-switch cost that independent processes pay each other.

For the record, six live 30 fps cameras need 180 fps of aggregate capacity —
every configuration here covers that with room to spare. Contention only
becomes real when aggregate demand approaches a thousand frames per second.

## 10. The shm question, answered empirically

We went into this believing CUDA shared memory would always win. The data
disagreed, in an instructive way:

| Transfer | Python client (CPU-resident frames) | C++ client (GPU-resident frames) |
|----------|-------------------------------------|----------------------------------|
| raw gRPC | 130 fps | 222 fps |
| system shm | **469 fps** | — |
| CUDA shm | 330 fps | **654 fps** |

Shared memory always beats raw gRPC (it eliminates the protobuf
serialization and the socket copies). But *which* shared memory depends on
where the bytes already live: **system shm** for CPU-side numpy data,
**CUDA IPC shm** for GPU-side kernel output. The copy you eliminate is the
copy that matched the data's location. In flow B2 the shm region *is* the
kernel's destination buffer — one `cudaMalloc`, one IPC handle, reused for
every frame of the stream.

## 11. What we'd tell ourselves at the start

1. **A benchmark that saturates the source measures the source.** Replay
   preprocessed input flat-out, or your "2× faster" is really "the camera ran
   at 30 fps."
2. **Latency and throughput are different products.** Flow D's 1665 fps and
   6–74 ms queue wait are the same number read two ways. Pick per use case:
   B2 for live streams, D for offline throughput, A2 when you want no
   dependencies, C2 when your team is Python-only.
3. **The GPU is almost never the bottleneck at the edge.** Every Triton
   configuration in this study left it >90% idle until we removed the copies.
   The fight is over PCIe round trips, serialization, and interpreter locks.
4. **Match preprocessing bit-for-bit before comparing pipelines** — a
   different letterbox silently changes detection counts and invalidates
   everything downstream.
5. **Dynamic batching requires async in-flight clients.** Sync clients pay
   for the batching window and never collect the benefit.
6. **Shared memory: pick by data location.** CPU data → system shm; GPU data
   → CUDA IPC shm. And reuse one registered region per stream for the life
   of the stream — never allocate per frame.

## 12. Epilogue: reproducibility

Everything — both pipelines, the Triton model repo, the benchmark harness,
the raw results — lives in this repository and runs inside a committed
Docker image (`triton-bench:v3`). The models are regenerated from
`ultralytics` exports + `trtexec`; the engines themselves are gitignored.

```bash
docker restart triton-server   # image: triton-bench:v3
./scripts/benchmark_v2.sh 10 3 # the full 4-arm sweep
```

Raw data: `results/` (v1 = the flawed first pass, kept for honesty;
v2 = the corrected sweep; `parallel_contention.md` = the multi-instance
study; `comparison_tables.md` = the fair tables).

---

*Measured on an RTX 3090 (driver 595.84), Triton 24.12, TensorRT 10.7,
3× YOLO FP16 engines @ 640×640. All numbers are medians across ≥3 runs
unless noted; variance was <±2%.*