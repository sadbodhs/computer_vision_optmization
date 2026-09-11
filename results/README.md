# results/ — provenance

[← index](../README.md)

What each file is, and how it maps to the tables in the top-level README/STORY.

## Layout

| Path | Pass | What it is |
|---|---|---|
| `*_single_*.json`, `*_3stream_*.json` | **v1 (flawed)** | The first, source-capped pass (STORY §3). Kept for honesty. `latency_ms_est` here is `1000/fps`, a derived reciprocal — **not** a measured p50. Do not cite these as pipeline capacity. |
| `v2/all_*.tsv` | **v2 (corrected)** | The 4-arm capacity + RTSP sweep from `scripts/benchmark_v2.sh`. One JSON per repeat, `arm|json`. |
| `v2/gpu_*.csv`, `gpu_*.csv`, `v3/gpu_*.csv` | v2/v3 | GPU util/mem samples (`scripts/gpu_sample.sh`) taken during runs. |
| `capacity_table.tsv` | — | The canonical capacity numbers (flow × concurrency → fps, latency) as data, so `scripts/make_plots.py` can regenerate the figures. Mirrors the table in `docs/results.md`. |
| `stage_decomposition.tsv` | — | Per-stage per-frame times behind the stage-decomposition figure. |
| `comparison_tables.md` | v2 | The fair tables (per-frame latency, per-frame GPU cost) + Flow E (DeepStream). |
| `parallel_contention.md` | v3 | Multi-instance contention study (`scripts/parallel_test.sh`). |
| `v3/accuracy.tsv` | v3 | COCO val2017 mAP (500 imgs): study chain vs ultralytics reference (`scripts/accuracy_eval.py`, `scripts/accuracy_reference.py`). |
| `v3/deepstream_capacity.tsv` | v3 | E2 capacity-mode attempt. **Harness-bound, not a DeepStream ceiling** - GPU median 0% at 8 streams. Do not quote 501 fps as E2 capacity. |
| `v3/model_scaling.tsv` | v3 | The model-cost ladder (YOLO11 n/s/m/l/x + yolov8s control) through A2, 3 interleaved repeats (`scripts/model_scaling.sh`). `non_engine_ms` is p50 minus engine time - the column the page is about. |
| `v3/in_graph_nms.tsv` | v3 | Output-size A/B: engine throughput and raw-gRPC round trip for the stock `[1,84,8400]` head vs an in-graph-NMS `[1,300,6]` build (`scripts/probe_transport.py`). |
| `v3/batching_knobs.tsv` | v3 | `preferred_batch_size` x `max_queue_delay_microseconds` sweep on flow D at concurrency 1 and 8 (`scripts/batching_knobs.sh`). |
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

**Two cells fell outside variance. Both have since been re-run 5 times, and they
turned out to have different explanations:**

**A2 @ conc=2 — the cell is unstable, not wrong.** Five repeats:
`1226 · 1213 · 1087 · 1101 · 1107` fps. It is *bimodal*: it either reaches ~1220
or settles near ~1100, spread **138.6 fps (12.5%)**. Both the published 1219 and
the 1084 re-run are inside its range, so neither is an error — but this cell does
not honour the study's stated "<±2% variance", and it is the least reliable number
in the capacity table. That matters because it is the cell behind the claim that
**A2 saturates at concurrency 2**; on the low runs the plateau does not begin
until concurrency 4.

**D @ conc=4 — stable, and the published figure is ~3% high.** Five repeats:
`1343 · 1337 · 1336 · 1335 · 1349` fps, spread **13.6 fps (1.0%)**. The reliable
value is **~1337**, against 1378 published. Well within the kind of drift you get
across separate sessions, and the shape of D's curve is unaffected.

The published tables are left as they are — they are a real run, and re-running
does not make an earlier honest measurement retroactively wrong. What changes is
the confidence attached: treat A2 @ conc=2 as ±12%, and D @ conc=4 as ~1337.

## Regenerating## Regenerating

```bash
scripts/make_frames.sh videos/real.mp4 500   # capacity-replay input
scripts/benchmark_v2.sh 10 3                  # full sweep -> results/v2/all_<ts>.tsv
scripts/parallel_test.sh 8 3                  # contention -> results/v3/
```
