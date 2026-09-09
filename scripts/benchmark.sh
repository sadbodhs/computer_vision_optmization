#!/bin/bash
# Benchmark harness: Triton (Python pre/post) vs Pure TensorRT (C++)
# Usage: ./benchmark.sh [duration_seconds]
set -e
DURATION=${1:-15}
BASE=/home/suchi/sadbodh/rt_vs_triton
RESULTS=$BASE/results
mkdir -p $RESULTS
TS=$(date +%Y%m%d_%H%M%S)

echo "=== Benchmark started at $TS, duration=${DURATION}s ==="

# ---- Step 1: single stream, single model (yolov8s) ----
echo ""
echo "### STEP 1: Single stream (cam1) x yolov8s ###"

# Triton pipeline
echo "[Triton] running..."
docker exec triton-server bash -c "cd /work && python3 client.py --model yolov8s --duration $DURATION" \
  > $RESULTS/triton_single_$TS.json 2>/dev/null
echo "[Triton] $(cat $RESULTS/triton_single_$TS.json)"

# C++ pipeline
echo "[C++ TRT] running..."
docker exec triton-server bash -c "cd /work/cpp/build && ./trt_pipeline --engine model.plan --url rtsp://localhost:8554/cam1 --duration $DURATION" \
  > $RESULTS/cpp_single_$TS.json 2>/dev/null
echo "[C++ TRT] $(cat $RESULTS/cpp_single_$TS.json)"

# ---- Step 2: 3 streams x 3 models ----
echo ""
echo "### STEP 2: 3 streams x 3 models (yolov8n, yolov8s, yolo11n) ###"

# Triton: 3 streams, 3 models (one client process, 3 threads)
echo "[Triton 3-stream] running..."
docker exec triton-server bash -c "cd /work && python3 client.py --models yolov8n,yolov8s,yolo11n --streams 3 --duration $DURATION" \
  > $RESULTS/triton_3stream_$TS.json 2>/dev/null
echo "[Triton 3-stream] $(cat $RESULTS/triton_3stream_$TS.json)"

# C++: 3 independent processes, one per stream+model
echo "[C++ 3-pipeline] running..."
docker cp $BASE/scripts/run_cpp3.sh triton-server:/work/run_cpp3.sh 2>/dev/null
docker exec triton-server bash -c "bash /work/run_cpp3.sh $DURATION" \
  > $RESULTS/cpp_3stream_$TS.json 2>/dev/null
echo "[C++ 3-pipeline]"
cat $RESULTS/cpp_3stream_$TS.json

echo ""
echo "=== Results saved to $RESULTS (suffix $TS) ==="
