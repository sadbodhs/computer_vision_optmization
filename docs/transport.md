# Transport — which shared memory, and when

[← index](../README.md) · prev: [DeepStream](deepstream.md) · next: [Stage decomposition](stage-decomposition.md)

The question: once your frames are preprocessed, how do you get them into the
server without paying for it twice?

---

## The finding

| Variant | Python client (CPU data) | C++ client (GPU data) | Why |
|---|---|---|---|
| raw gRPC | 130 fps · 7.1 ms | 222 fps · 3.3 ms | protobuf serialize + socket copies |
| **system shm** | **469 fps · 1.69 ms** | — | server reads CPU shm directly; 2 copies gone |
| **CUDA shm** | 330 fps · 2.47 ms | **654 fps · 1.28 ms** | zero-copy only when bytes already on GPU |

**Pick by data location, not by which sounds faster:**

- Data on GPU (C++ CUDA kernels) → **CUDA IPC shm** (3× vs raw).
- Data on CPU (numpy) → **system shm** (3.6× vs raw). CUDA shm here is *worse*
  than system shm — it just moves the H2D copy to the client side.

We went in believing CUDA shared memory would always win. It doesn't. The copy you
eliminate has to be the copy that matched where the data already lived.

## Why raw gRPC costs what it does

Triton's own server-side instrumentation showed, per request:

- ~1.27 ms copying the input to the GPU
- ~1.25 ms copying the output back
- ~1.2 ms of actual inference

The framework was spending more time on copies than on the model. Giving the
client a CUDA shared-memory region — a buffer the client's kernel writes and the
server reads through an IPC handle, with **no copy in either direction** — took B2
from 222 to **654 fps** and its latency to 1.28 ms, just 0.05 ms above running the
engine in-process.

> **Triton's entire framework, given zero-copy buffers, costs about a tenth of a
> millisecond.** The framework was never the problem; the copies were.

## Region management (the part people get wrong)

One `cudaMalloc` + one IPC handle **per stream, for the stream's lifetime**. The
shm region *is* the preprocessing kernel's output buffer — the kernel writes
straight into the region the server will read.

**Never allocate or register per frame.** Registration is a server round trip;
doing it per frame reintroduces exactly the overhead shm exists to remove.

## The zero-copy chain (A2 / B2 / D)

```
NVDEC NV12 (GPU) → fused kernel reads in place → writes TRT input / shm (GPU)
                 → infer → compact kernel (GPU) → only ~KB of candidate boxes to host
```

One CUDA stream end-to-end. The full `[1,84,8400]` output tensor (2.8 MB) never
crosses to the host — a GPU compact kernel reduces it to the handful of candidate
boxes above threshold first.

## Caveat

Multi-process C2 requires unique shm region names per process (handled in
`client_v2.py` via `run_tag`); sharing region names across threads caused a data
race that invalidated the N≥4 in-process numbers. See
[contention](contention.md#caveats).

---

[← index](../README.md) · prev: [DeepStream](deepstream.md) · next: [Stage decomposition](stage-decomposition.md)
