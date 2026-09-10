# CUDA Graphs — how much of the "engine ceiling" is launch overhead?

[← index](../README.md) · related: [Methodology](methodology.md) · [Stage decomposition](stage-decomposition.md)

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

## What this does *not* yet show

- **No pipeline binary uses CUDA Graphs.** A1/A2/B1/B2/D all launch per frame.
  Applying graphs inside `main_cuda.cu` means capturing the inference + compact
  kernel into a graph and replaying it, which is a code change that has **not**
  been made or measured. A2's engine stage measures ~0.98 ms, so ~0.13 ms is the
  plausible headroom — a *projection from this table, not a measurement*.
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

[← index](../README.md) · related: [Methodology](methodology.md) · [Stage decomposition](stage-decomposition.md)
