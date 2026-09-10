# results/ — provenance

What each file is, and how it maps to the tables in the top-level README/STORY.

## Layout

| Path | Pass | What it is |
|---|---|---|
| `*_single_*.json`, `*_3stream_*.json` | **v1 (flawed)** | The first, source-capped pass (STORY §3). Kept for honesty. `latency_ms_est` here is `1000/fps`, a derived reciprocal — **not** a measured p50. Do not cite these as pipeline capacity. |
| `v2/all_*.tsv` | **v2 (corrected)** | The 4-arm capacity + RTSP sweep from `scripts/benchmark_v2.sh`. One JSON per repeat, `arm|json`. |
| `v2/gpu_*.csv`, `gpu_*.csv`, `v3/gpu_*.csv` | v2/v3 | GPU util/mem samples (`scripts/gpu_sample.sh`) taken during runs. |
| `comparison_tables.md` | v2 | The fair tables (per-frame latency, per-frame GPU cost) + Flow E (DeepStream). |
| `parallel_contention.md` | v3 | Multi-instance contention study (`scripts/parallel_test.sh`). |

## Reading the v2 TSVs (important)

`arm|json`. In the **first** committed `all_*.tsv` files the `A_cap_*` and
`trtexec_*` rows are **blank** — that harness (v2.1) invoked the CPU-path binary
name (`trt_pipeline`) and an empty engine-cap capture, and those arms produced no
JSON. The A2 and engine-ceiling numbers in the README were taken from **direct
binary runs**, not those rows.

`scripts/benchmark_v2.sh` has since been corrected (v2.2) to invoke the exact
binaries the published tables report — `trt_pipeline_cuda` (A2), `trt_grpc_cuda`
(B2), `client_v2.py` (C2), `trt_grpc_async` (D) — so a fresh
`scripts/benchmark_v2.sh 10 3` now populates every arm, including A2 and the
`trtexec_*` engine caps. Re-run it to regenerate a complete TSV.

Spot-check (RTX 3090, YOLOv8s FP16, capacity mode, single stream), which
reproduces the headline A2/B2 figures:

| Arm | fps | p50 |
|---|---|---|
| A2 `trt_pipeline_cuda` | ~798–809 | 1.24 ms |
| B2 `trt_grpc_cuda` | ~644–654 | 1.26 ms |

## Regenerating

```bash
scripts/make_frames.sh videos/real.mp4 500   # capacity-replay input
scripts/benchmark_v2.sh 10 3                  # full sweep -> results/v2/all_<ts>.tsv
scripts/parallel_test.sh 8 3                  # contention -> results/v3/
```
