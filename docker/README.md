# Docker — rebuild and run everything from the repo

[← index](../README.md)

Every arm runs in a container; nothing is installed on the host. Previously the
containers were only recoverable from committed images (`triton-bench:v3`,
`ds-build`). These Dockerfiles rebuild them from source, so the repo alone is
enough. All commands run **from the repo root**.

> Requires an NVIDIA GPU + the NVIDIA Container Toolkit (`--gpus all`), and access
> to `nvcr.io` base images. Tested target: RTX 3090, driver 595.84.

## 1. Build the images

```bash
docker/build.sh            # triton-bench:v3 + ds-build:latest
# or individually: docker/build.sh triton | docker/build.sh deepstream
```

## 2. Regenerate the models (pt → ONNX → FP16 .plan)

Engines are gitignored (GPU/driver-specific) and rebuilt in-container:

```bash
scripts/export_models.sh   # writes triton/models/<m>/1/model.plan for all 3 models
```

## 3. Start the Triton server

```bash
docker run -d --name triton-server --gpus all --shm-size=1g --network host \
  -v "$PWD/triton/models:/models" -v "$PWD:/work" \
  triton-bench:v3 tritonserver --model-repository=/models
```

## 4. Generate the capacity-replay input (frames.bin)

`frames.bin` is preprocessed tensors replayed flat-out; it is gitignored and
regenerated from a source video (identical preprocessing to the C2 numpy path):

```bash
scripts/make_frames.sh videos/real.mp4 500   # -> cpp/build/frames.bin
```

## 5. Run the benchmarks

```bash
scripts/benchmark_v2.sh 10 3       # 4-arm capacity + RTSP sweep, 3 repeats
scripts/parallel_test.sh 8 3       # multi-instance contention, N=3
```

DeepStream (Flow E) uses the `ds-build` image with the YOLO-parser project mounted:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" -v "$PWD/../deepstream-yolov11:/opt/ds" \
  ds-build:latest \
  /work/cpp/build/ds_bench --config /opt/ds/model/yolov8s/config_infer_primary_yolov8s.txt \
    --streams 1 --batch 1 --duration 15
```

## Image contents

| Image | Base | Adds |
|---|---|---|
| `triton-bench:v3` | `tritonserver:24.12-py3` | C++ client SDK + perf_analyzer (from `-sdk`, in `/opt`), ffmpeg dev, `tritonclient[all]`/numpy/ultralytics, all `cpp/` binaries in `/work/cpp/build` |
| `ds-build:latest` | `deepstream:7.1-triton-multiarch` | GStreamer dev, compiled `ds_bench` |

## Notes / things to verify on your host

- The Triton C++ client SDK path is assumed at `/opt` (`-DTRITON_CLIENT_ROOT=/opt`).
  If `cmake` reports the client SDK not found, point it at the right prefix.
- `ds_bench` link flags (`docker/build_ds_bench.sh`) target DeepStream 7.1's default
  layout (`/opt/nvidia/deepstream/deepstream`); adjust if your SDK differs.
- `--shm-size=1g` matters for the CUDA/system shared-memory arms (B2/C2/D).
