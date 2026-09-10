# Batching — the trap, and Triton at its best

[← index](../README.md) · prev: [Transport](transport.md) · next: [Contention](contention.md)

Dynamic batching is where Triton either wins the whole study or loses to a
200-line C++ program. Which one depends entirely on the client.

---

## The trap

The batch-8 engine costs 4.84 ms for eight frames — 0.61 ms each, **37% cheaper
per frame** than running solo. Free lunch, apparently.

Our first attempt at flow D came out *slower* than the plain batch-1 server.

**Batching is a bus, and buses need passengers to arrive together.** The
synchronous clients sent one request, blocked, read the answer, then sent the
next. Requests trickled in one at a time, the 5 ms batching window expired empty,
and every request paid the window's latency while riding alone.

> **Dynamic batching requires async in-flight clients.** A sync client pays for
> the batching window and never collects the benefit.

The fix: an **async client with eight requests in flight** per stream. Requests
co-arrive, batches fill, and the same server delivers **1665 fps — the highest
number in the entire study, and the only configuration that beats in-process
C++.**

## The bill, in the latency column

Those 1665 frames each waited 6–74 ms in the queue for their batch-mates:

| Concurrency | Total fps | Queue wait (p50) | GPU service/frame |
|---|---|---|---|
| 1 | 1041 | 6.2 ms | 0.61 ms |
| 2 | 1136 | 11.1 ms | 0.61 ms |
| 4 | 1378 | 19.2 ms | 0.61 ms |
| 8 | 1640 | 36.6 ms | 0.61 ms |
| 16 | 1665 | 73.9 ms | 0.61 ms |

**Batching converts latency into throughput.** For offline analytics that is the
best trade in the study. For a live camera with a 33 ms frame budget, it is a bus
that misses its stop — at conc=16 a frame waits more than two full camera frames.

## Triton at its absolute best

Flow D is "B2 with dynamic batching": CUDA shm zero-copy + async in-flight +
batch-8 engines. Run across **all three models concurrently** — Triton's real
production scenario, where its scheduler has no hand-rolled equivalent:

| Scenario | Total fps | Latency p50 | vs hand-rolled best |
|---|---|---|---|
| D: 1 model (yolov8s), conc=16 | 1665 | 73.9 ms wait | +38% vs A2 (1205) |
| **D: 3 models × 6 streams each** | **1799** | 29.4 ms | +49% vs A2 |
| **D: 3 models × 6 streams (conc=18)** | **1816** | 58.6 ms | +51% vs A2 |
| D: yolov8n alone, conc=8 | 1988 | 24.1 ms | engine cap 2960 effective |

### This is the answer to "shouldn't Triton win?"

Yes — when fed properly. With CUDA shm zero-copy and async clients keeping batches
full, Triton **beats the hand-rolled C++ pipeline by ~50% on throughput**. The
same server with naive clients (C1: 225 fps) loses by 8×.

**Triton's framework is only as good as its client; its scheduler is the
irreplaceable part.** You can hand-roll zero-copy and NVDEC in an afternoon. You
cannot hand-roll a multi-model scheduler that keeps a GPU 80% busy across three
engines and eighteen streams.

## Configuration

The batching side is a Triton model config, not client code:

```protobuf
max_batch_size: 8
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 5000
}
instance_group [ { count: 2 kind: KIND_GPU } ]
```

See [`triton/models/yolov8s_dyn/config.pbtxt`](../triton/models/yolov8s_dyn/config.pbtxt).
`max_queue_delay_microseconds` is the bus timetable: how long a partly-full batch
waits for stragglers. 5 ms is the floor of D's 6.2 ms conc=1 latency.

> Not yet swept: `instance_group count`, `preferred_batch_size`, and the queue
> delay were fixed at these values throughout. See [roadmap](roadmap.md).

---

[← index](../README.md) · prev: [Transport](transport.md) · next: [Contention](contention.md)
