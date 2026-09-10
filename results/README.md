# results/ — provenance

[← index](../README.md)

What each file is, and how it maps to the tables in the top-level README/STORY.

## Layout

| Path | Pass | What it is |
|---|---|---|
| `*_single_*.json`, `*_3stream_*.json` | **v1 (flawed)** | The first, source-capped pass (STORY §3). Kept for honesty. `latency_ms_est` here is `1000/fps`, a derived reciprocal — **not** a measured p50. Do not cite these as pipeline capacity. |
| `v2/all_*.tsv` | **v2 (corrected)** | The 4-arm capacity + RTSP sweep from `scripts/benchmark_v2.sh`. One JSON per repeat, `arm|json`. |
| `v2/gpu_*.csv`, `gpu_*.csv`, `v3/gpu_*.csv` | v2/v3 | GPU util/mem samples (`scripts/gpu_sample.sh`) taken during runs. |
| `comparison_tables.md` | v2 | The fair tables (per-frame latency, per-frame GPU cost) + Flow E (DeepStream). |
| `parallel_contention.md` | v3 | Multi-instance contention study (`scripts/parallel_test.sh`). |
| `v3/triton_knobs.tsv` | v3 | `instance_group count` x CUDA graphs sweep on B2 (`scripts/triton_knobs.sh`), 3 repeats per cell. |
| `v3/precision_ceilings.tsv` | v3 | INT8 / sparsity **speed ceilings** (`scripts/precision_ceilings.sh`). Throughput only — INT8 built without calibration, accuracy invalid. |
| `v3/cuda_graphs.tsv` | v3 | CUDA Graphs A/B across all three engines (`scripts/cuda_graphs.sh`), 3 repeats per cell. |
| `v3/mps_contention_N3.tsv` | v3 | CUDA MPS A/B for A2 at N=3 (`scripts/mps_contention.sh`), 3 repeats per condition. MPS off ~960 fps / 3.115 ms; MPS on ~1266 fps / 2.476 ms. |

## Reading the v2 TSVs (important)

`arm|json`. In the **first** committed `all_*.tsv` files the `A_cap_*` and
`trtexec_*` rows are **blank** — that harness (v2.1) invoked the CPU-path binary
name (`trt_pipeline`) and an empty engine-cap capture, and those arms produced no
JSON. The A2 and engine-ceiling numbers in the README were taken from **direct
binary runs**, not those rows.

`scripts/benchmark_v2.sh` has since been corrected (v2.2) to invoke the exact
binaries the published tables report — `trt_pipeline_cuda` (A2), `trt_grpc_cuda`
(B2), `client_v2.py` (C2), `trt_grpc_async` (D) — and two bugs that produced
those blank rows were fixed: the `trtexec` cap parsed a fixed awk column (which
picked up the `[I]` log-level tag), and `client_v2.py` deadlocked the C2 arm
(see the commit for `for...else` / shm-lifecycle details).

## The complete run — `v2/all_20260910_084552.tsv`

**Every arm populated, no blank rows.** 82 rows: A2/B2/C2/D × concurrency
1/2/4/8/16 × 3 repeats, plus RTSP arms, 3-process arms, and engine caps.

Engine ceilings now captured (vs the README's published figures):

| Engine | measured | published |
|---|---|---|
| yolov8s (batch-1) | **1022.98** qps | 1028 |
| yolov8n | **1490.79** qps | 1490 |
| yolo11n | **1252.58** qps | 1259 |

Capacity medians vs the published table — **18 of 20 cells within ±2%**, the
stated run-to-run variance:

| Flow | conc 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| A2 | 796 (−1.6%) | 1084 (**−11.1%**) | 1168 (−0.6%) | 1178 (−0.7%) | 1154 (−0.5%) |
| B2 | 645 (−1.4%) | 938 (−1.4%) | 1090 (−0.2%) | 1109 (−2.0%) | 1114 (−1.2%) |
| C2 | 464 (−1.0%) | 747 (+1.5%) | 1000 (+1.4%) | 1047 (+1.0%) | 1026 (−1.2%) |
| D | 1047 (+0.6%) | 1129 (−0.6%) | 1276 (**−7.4%**) | 1617 (−1.4%) | 1672 (+0.4%) |

**Two cells fall outside variance and are not being papered over:**

- **A2 @ conc=2: 1084 vs 1219 published (−11%).** This matters because the
  published claim is that A2 "hits its ~1200 fps plateau already at conc=2". In
  this run the plateau starts at conc=4 (1168) instead. The plateau itself
  reproduces; *where it begins* does not.
- **D @ conc=4: 1276 vs 1378 (−7.4%).** D's batch-fill behaviour is the most
  client-timing-sensitive arm in the study, so it is the most run-dependent.

One re-run does not overturn a published median, and the tables have **not** been
edited on the strength of it. Both cells need more repeats before either number
is called wrong — noted here so the discrepancy is visible rather than buried.

## Regenerating

```bash
scripts/make_frames.sh videos/real.mp4 500   # capacity-replay input
scripts/benchmark_v2.sh 10 3                  # full sweep -> results/v2/all_<ts>.tsv
scripts/parallel_test.sh 8 3                  # contention -> results/v3/
```
