# Introduction — why this exists

[← index](../README.md) · next: [Use cases](use-cases.md)

If you already know why someone would agonise over Triton vs TensorRT vs
DeepStream, skip to [Methodology](methodology.md). This page is for everyone else.

---

## The situation

You have cameras — a handful, or a few dozen — and a GPU. You want to run object
detection on every frame: count people, spot vehicles, flag defects on a line.

The model is the easy part. YOLO is a solved problem; you download weights and it
works. **The hard part is everything around the model**, and that is where the
performance actually goes.

A frame has to be decoded from the camera's H.264 stream, resized and normalised
into the exact tensor shape the network expects, moved onto the GPU, run through
the network, and the raw output turned back into boxes. The network is one step
in that chain and frequently not the expensive one.

## What you are choosing between

NVIDIA offers several ways to build that chain, all first-party, all fast on the
box, and it is genuinely unobvious which to use.

| | What it is |
|---|---|
| **TensorRT** | A compiler and runtime. Takes your trained model and builds an *engine* optimised for one specific GPU. Not a server — you call it from your own program. |
| **Triton Inference Server** | A server that hosts models. Your application sends frames over the network (gRPC) and gets detections back. Handles batching, multiple models, metrics, hot reload. |
| **DeepStream** | An integrated video pipeline. Decode → batch → infer → parse, wired together as a GStreamer graph. You write configuration rather than code. |

And the fourth option, always present: **write it yourself** — link TensorRT and
the video decoder directly into one process and skip the server entirely.

They are not alternatives in the tidy sense. Triton *runs* TensorRT engines.
DeepStream *runs* TensorRT engines. The question is not which inference library
is fastest — it is the same library underneath in every case. **The question is
what the plumbing around it costs.**

## Why you cannot just read the benchmarks

Vendor numbers are real and also not about you. They typically report a model's
throughput under ideal batched conditions on a datacentre card. Your situation is
eight 30 fps cameras and a latency budget.

And the layers interact in ways no single number captures:

- A server that looks slow may be slow *because of how you fed it*, not because
  of the server. One flow in this study goes from 222 fps to 654 fps with no
  change to the server at all — only to the client.
- A pipeline that looks fast may be fast because **nothing was asking it for more
  work**. Measure a 30 fps camera and you will measure 30 fps, whatever is behind
  it.

That second trap is the one this study fell into first, and
[STORY.md](../STORY.md) is the account of catching it.

## Two numbers that are not the same thing

The single most useful idea here, and the one most easily lost:

> **Throughput** is how many frames per second the system can process.
> **Latency** is how long *one* frame waits from arrival to answer.

They trade against each other, and optimising one can wreck the other. The
clearest case in this study is batching: collecting 8 frames and running them as
one GPU pass is **37% cheaper per frame** — excellent throughput. But each frame
now waits for seven others to show up. At high concurrency that wait reaches
**74 ms**, more than two frames of a 30 fps camera.

For offline video analytics that is the best trade available. For a live camera
it is a bus that misses its stop.

So "fastest" is not a well-formed question. **Fastest at what?**

## The frame budget

A useful anchor. A 30 fps camera gives you a new frame every **33.3 ms**. If your
pipeline takes longer than that per frame, you fall behind permanently.

Every latency figure here can be read against that budget. It reframes the whole
comparison: at 1.2 ms per frame you are using 3.7% of it and the difference
between two pipelines stops mattering; at 74 ms you have already lost.

## What this study does

One model, one GPU, identical preprocessing, nine pipeline variants — and only
the plumbing differs. Each variant isolates one layer, so the cost of that layer
can be named:

- the engine alone (the floor nothing can beat)
- a hand-rolled C++ pipeline around it
- the same work through a server
- the same again with the transport fixed
- the server used properly, with batching and async clients
- NVIDIA's integrated stack for comparison

Plus the parts people usually skip: what it costs in **accuracy**, what happens
when several pipelines **share one GPU**, and which optimisations are free versus
which are paid for somewhere else.

## What you can take from it

- **If you just want an answer**, the [decision guide](../README.md#decision-guide)
  maps common situations to a pipeline.
- **If you are choosing a stack**, [Results](results.md) has the tables and the
  latency-vs-throughput picture.
- **If you are debugging something slow**, [Stage decomposition](stage-decomposition.md)
  shows where per-frame time actually goes, and [Transport](transport.md) covers
  the single biggest avoidable cost.
- **If you are about to benchmark anything yourself**, read
  [Methodology](methodology.md) first. The traps it describes are not exotic; they
  are the default outcome.

A recurring theme is worth stating in advance: **the GPU is almost never the
bottleneck at the edge.** In nearly every configuration measured here it sat idle
while the pipeline fought PCIe transfers, protobuf serialisation and interpreter
locks. Optimising the model is usually not where the frames are.

---

[← index](../README.md) · next: [Use cases](use-cases.md)
