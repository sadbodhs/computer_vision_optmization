# CUDA Graphs — how much of the "engine ceiling" is launch overhead?

[← index](../README.md) · prev: [Triton tuning](triton-tuning.md) · next: [In-graph NMS](in-graph-nms.md)

The study treats `trtexec` on the batch-1 engine — **0.97 ms / ~1028 qps** for
YOLOv8s — as the physical floor, "the GPU's honest price." That framing is wrong
in one specific way: a meaningful slice of it is **CPU-side kernel-launch
overhead**, not GPU compute. CUDA Graphs replays a whole iteration as one
pre-recorded graph and removes most of it.

Script: [`scripts/cuda_graphs.sh`](../scripts/cuda_graphs.sh) ·
raw data: [`results/v3/cuda_graphs.tsv`](../results/v3/cuda_graphs.tsv)
(3 repeats per cell, `trtexec --useCudaGraph`, data transfers included).

---

## Result

| Model | plain (qps) | `--useCudaGraph` (qps) | Gain |
|---|---|---|---|
| YOLOv8n | 1494.2 | **1840.8** | **+23.2%** |
| YOLOv8s | 1014.6 | **1165.5** | **+14.9%** |
| YOLO11n | 1253.0 | **1592.0** | **+27.0%** |

The plain column reproduces the published caps (1490 / 1028 / 1259), so this is a
like-for-like A/B.

## The gain is a fixed cost, not a percentage

Converted to per-frame time, the *absolute* saving is nearly identical across
three models of quite different cost:

| Model | plain ms/frame | graph ms/frame | **saved** |
|---|---|---|---|
| YOLOv8n | 0.669 | 0.543 | **0.126 ms** |
| YOLOv8s | 0.986 | 0.858 | **0.128 ms** |
| YOLO11n | 0.798 | 0.628 | **0.170 ms** |

That is the tell: **CUDA Graphs removes a roughly constant ~0.13 ms of
per-iteration launch overhead.** It reads as +15% on YOLOv8s and +23% on YOLOv8n
only because the same fixed cost is a larger share of a cheaper model's budget.

> **The corollary matters for model selection.** Launch overhead does not shrink
> when you pick a smaller model, so the speedup from downsizing is always less
> than the FLOP reduction suggests — unless you also remove the launch cost.

## Measured: graphs inside the A2 pipeline

The projection above is now a measurement. `trt_pipeline_cuda` takes a
`--cuda-graph` flag that captures the inference region — `enqueueV3`, the
`d_count` memset, the compact kernel, the count D2H — once and replays it per
frame. The input H2D stays outside the graph: capture freezes every argument,
and that copy's source pointer walks through `frames.bin`. The captured region
is therefore exactly the `infer` stage the report already breaks out.

Script: [`cuda_graphs_pipeline.sh`](../scripts/cuda_graphs_pipeline.sh) ·
raw data: [`results/v3/cuda_graphs_pipeline.tsv`](../results/v3/cuda_graphs_pipeline.tsv).
Same binary, one flag apart, arms **interleaved** (off, on, off, on…) so drift
hits both equally instead of accumulating in whichever ran second.

| streams | graph | fps | p50 | p95 | infer window |
|---|---|---|---|---|---|
| 1 | off | 799.1 | 1.247 ms | 1.268 ms | 0.982 ms |
| 1 | **on** | **893.8** (+11.8%) | **1.115 ms** | **1.134 ms** (−10.6%) | **0.851 ms** |
| 2 | off | 1121.0 | 1.770 ms | 2.026 ms | 1.452 ms |
| 2 | **on** | **1250.5** (+11.6%) | **1.596 ms** | **1.650 ms** (−18.6%) | 1.330 ms |
| 4 | off | 1166.9 | 3.426 ms | 3.875 ms | 2.797 ms |
| 4 | **on** | **1248.9** (+7.0%) | **3.205 ms** | **3.440 ms** (−11.2%) | 2.375 ms |

5 repeats at 1 stream, 3 at 2 and 4. Detections per frame held at **1.597 in
every single run, both arms** — the invariant that separates "faster" from
"quietly doing less work".

### The projection was right, and the reason is better than the number

At concurrency 1 the captured region falls 0.982 → 0.851 ms: a saving of
**0.1315 ms** against the ~0.13 ms projected from the engine table.

The more interesting part is *which* numbers those are. Compare them to the
`trtexec` row at the top of this page:

| | `trtexec` | A2's infer stage |
|---|---|---|
| plain | 0.986 ms | 0.982 ms |
| with graphs | 0.858 ms | 0.851 ms |

**A2's inference stage is the engine benchmark**, to within 0.5%. Nothing was
hiding inside it — no stray synchronisation, no framework cost, nothing for the
pipeline to be blamed for. That is worth stating plainly because it is the one
place in this whole study where a hand-rolled pipeline reaches the synthetic
number exactly.

### The percentage is diluted, and that is the transferable part

+14.9% at the engine becomes **+11.8%** end-to-end. Nothing went wrong; the
saving is a fixed 0.13 ms, and A2's per-frame budget also contains 0.256 ms of
H2D and 0.015 ms of host NMS that a graph cannot touch.

> **The rule: your gain is ~0.13 ms divided by your total per-frame time**, not
> the percentage printed on the engine benchmark. The leaner everything else is,
> the more of the headline you keep. A pipeline spending 6 ms per frame keeps
> about 2% of it.

This is the same fixed-cost logic as the model-selection corollary above, turned
on the pipeline instead of the model.

### Graphs raise the plateau — it was partly launch-bound

A2 saturates at **~1167 fps** without graphs and **~1249 fps** with them. The
ceiling that concurrency alone could not push past was not the GPU running out
of work; some of it was the CPU failing to feed it fast enough.

### And the gain shrinks with concurrency, exactly as it did in Triton

+11.8% → +11.6% → +7.0% across 1, 2 and 4 streams. [Triton
tuning](triton-tuning.md) found the same shape against `instance_group count`
(+13.6% at 1, +3.1% at 2) and gave the reason: **graphs and concurrency are
substitutes.** Both fill the gaps between launches, so whichever you apply
second recovers less. Seeing it reproduce in-process, with no server involved,
is good evidence the explanation is right rather than a Triton artefact.

One honest caveat on the table: at 2 and 4 streams the `infer` window includes
time queued behind other threads, so the deltas there are not purely launch
overhead. The clean measurement of the fixed cost is the concurrency-1 row.

### Tail latency improves more than throughput

p95 falls 10.6–18.6% while p50 falls ~10%, and the run-to-run spread collapses:
without graphs the 2-stream arm ranged 1079–1192 fps across three repeats; with
graphs, 1243–1257. Replaying one pre-recorded graph is simply more predictable
than issuing a dozen launches per frame, and that shows up in the tail before it
shows up in the mean. [Triton tuning](triton-tuning.md#3-where-graphs-still-earn-their-place-tail-latency)
reached the same conclusion from the other side.

**If you are latency-bound rather than throughput-bound, this is the strongest
result on the page** — and per [Use cases](use-cases.md#1-closed-loop-something-moves-because-of-the-detection),
closed-loop deployments are judged on exactly this.

## What this does *not* yet show

- **A2 is measured (above); A1, B1 and D are not.** A1 is the CPU path, where
  0.13 ms against a 1.32 ms frame would be worth roughly what it is in A2. B1/B2
  launch inside the *server*, so Triton's own graph support is the lever there and
  it is already measured. Flow D is the real gap — see the shape note below.
- **Triton's own CUDA-graph support: now measured** — see
  [Triton tuning](triton-tuning.md). Short version: the engine-level gain arrives
  intact at `instance_group count: 1` (+13.6%) but is mostly gone at `count: 2`
  (+3.1%), because multiple instances already hide the launch overhead that graphs
  remove. The p95 tail improvement survives regardless.
- **Graphs constrain shape changes.** A captured graph is fixed-shape; the
  dynamic-batch `_dyn` engines used by flow D would need one graph per batch size.
  Not explored here.

## How this changes the reference ceiling

[Methodology](methodology.md#the-reference-ceiling) presents 0.97 ms / 1028 qps as
the floor. More precisely:

| | YOLOv8s batch-1 |
|---|---|
| Ceiling as published (per-launch) | 0.97 ms · 1028 qps |
| Ceiling with launch overhead removed | **0.86 ms · 1166 qps** |

The published number remains the right baseline for *this study*, because every
flow measured here launches per frame. It just is not a hardware floor.

---

[← index](../README.md) · prev: [Triton tuning](triton-tuning.md) · next: [In-graph NMS](in-graph-nms.md)
