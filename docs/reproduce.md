# Reproduce

[← index](../README.md) · prev: [Precision](precision.md) · next: [Roadmap](roadmap.md)

Everything runs in Docker; nothing is installed on the host. Full container
recipe: [`docker/README.md`](../docker/README.md).

---

## From a clean clone

```bash
docker/build.sh                              # build triton-bench:v3 + ds-build:latest
scripts/export_models.sh                     # pt -> ONNX -> FP16 .plan (all 6 engines)
docker run -d --name triton-server --gpus all --shm-size=1g --network host \
  -v "$PWD/triton/models:/models" -v "$PWD/videos:/work/videos" \
  triton-bench:v3 tritonserver --model-repository=/models
scripts/make_frames.sh videos/real.mp4 500   # -> cpp/build/frames.bin (capacity input)
scripts/benchmark_v2.sh 10 3                  # the full 4-arm sweep
```

> Do not bind-mount the repo over `/work`: the image carries the sources *and*
> the compiled binaries in `/work/cpp/build`, and the mount would hide them.

Engines, ONNX, and `frames.bin` are gitignored — GPU-specific or large, but fully
regenerable by the two scripts above. The images rebuild from source, so nothing
depends on a committed image surviving.

## Individual arms

```bash
docker restart triton-server        # image triton-bench:v3 (Triton arm)
# A2: C++ full-CUDA
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_pipeline_cuda --engine model.plan --mode file --file frames.bin --streams 1 --duration 10'
# B2: Triton + CUDA shm
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_cuda --mode file --file frames.bin --model yolov8s --streams 1 --duration 10'
# C2: Python numpy + sys-shm + processes
docker exec triton-server bash -c 'cd /work && python3 client_v2.py --mode file --file cpp/build/frames.bin --model yolov8s --transfer sys --streams 4 --processes 4 --duration 8'
# D: async + dynamic batching (single model)
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_async --model yolov8s_dyn --file frames.bin --streams 8 --duration 10'
# D multi-model (Triton production scenario, 3 models x async batching)
docker exec triton-server bash -c 'cd /work/cpp/build && ./trt_grpc_async --models yolov8n_dyn,yolov8s_dyn,yolo11n_dyn --file frames.bin --streams 18 --duration 10'
# E: DeepStream (ds-build container)
docker exec ds-build bash -c 'cd /tmp && ./ds_bench --config /opt/ds/model/yolov8s/config_infer_primary_yolov8s.txt --streams 1 --batch 1 --duration 15'
```

## Other harnesses

```bash
scripts/parallel_test.sh 8 3    # multi-instance contention, N=3 -> results/v3/
scripts/pa_sweep.sh yolov8s "1 2 4 8 16"   # perf_analyzer sweep
scripts/gpu_sample.sh out.csv 30           # GPU util/mem sampling during a run
```

## Source → binary map

| Source | Binary | Flow |
|---|---|---|
| `cpp/src/main.cpp` | `trt_pipeline` | A1 — C++ TRT, CPU path |
| `cpp/src/main_cuda.cu` | `trt_pipeline_cuda` | A2 — C++ TRT, full-CUDA |
| `cpp/src/grpc_client.cpp` | `trt_grpc_client` | B1 — Triton + C++, raw gRPC |
| `cpp/src/grpc_client_cuda.cu` | `trt_grpc_cuda` | B2 (and D-sync) — CUDA shm |
| `cpp/src/grpc_async_client.cu` | `trt_grpc_async` | D — async, 8 in-flight |
| `cpp/src/ds_bench.cpp` | `ds_bench` | E1/E2 — DeepStream |
| `triton/client_v2.py` | — | C2 (`--transfer raw\|sys\|cuda`) |

The first five build from [`cpp/CMakeLists.txt`](../cpp/CMakeLists.txt);
`ds_bench` builds in the DeepStream container via
[`docker/build_ds_bench.sh`](../docker/build_ds_bench.sh).

## Other artifacts

- [`triton/models/`](../triton/models) — Triton model configs (fixed + `_dyn`)
- [`results/`](../results/README.md) — raw data, with provenance notes
- Images: `triton-bench:v3`, `ds-build:latest`

---

[← index](../README.md) · prev: [Precision](precision.md) · next: [Roadmap](roadmap.md)
