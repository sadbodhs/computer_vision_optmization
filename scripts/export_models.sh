#!/bin/bash
# Regenerate all model artifacts, in Docker, from nothing but the repo:
#   ultralytics .pt  ->  ONNX  ->  FP16 TensorRT .plan (batch-1 and batch-8)
# Output lands in the Triton model repo layout the configs expect:
#   triton/models/<m>/1/model.plan          (max_batch_size 0, fixed [1,3,640,640])
#   triton/models/<m>_dyn/1/model.plan      (max_batch_size 8, dynamic batching)
#
# Engines are GPU/driver-specific and gitignored — this is how anyone rebuilds them.
# Usage: scripts/export_models.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm --gpus all -v "$ROOT:/work" \
  nvcr.io/nvidia/tritonserver:24.12-py3 bash -c '
set -euo pipefail
pip install -q ultralytics onnx onnxsim 2>&1 | tail -1
cd /work/triton/models
TRTEXEC=/usr/src/tensorrt/bin/trtexec

for m in yolov8n yolov8s yolo11n; do
  echo "=== $m ==="
  # ultralytics always writes "<m>.onnx" next to the weights, so run the two exports
  # in separate dirs to keep both without any rename juggling.
  mkdir -p _exp_fixed _exp_dyn

  # 1a) fixed batch-1 ONNX — md5 of this ONNX is the parity anchor shared with DeepStream
  ( cd _exp_fixed && yolo export model=$m.pt format=onnx imgsz=640 dynamic=false simplify=True 2>&1 | tail -2 )
  md5sum _exp_fixed/$m.onnx
  # 1b) dynamic-batch ONNX for the batch-8 engine
  ( cd _exp_dyn && yolo export model=$m.pt format=onnx imgsz=640 dynamic=True simplify=True 2>&1 | tail -2 )

  # 2) fixed batch-1 engine  -> models/<m>/1/model.plan  (config: max_batch_size 0, [1,3,640,640])
  mkdir -p $m/1
  $TRTEXEC --onnx=_exp_fixed/$m.onnx --fp16 --saveEngine=$m/1/model.plan 2>&1 | grep -E "Throughput|Engine" | tail -2

  # 3) dynamic batch-8 engine -> models/<m>_dyn/1/model.plan  (config: max_batch_size 8)
  mkdir -p ${m}_dyn/1
  $TRTEXEC --onnx=_exp_dyn/$m.onnx --fp16 \
    --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 \
    --saveEngine=${m}_dyn/1/model.plan 2>&1 | grep -E "Throughput|Engine" | tail -2

  rm -rf _exp_fixed _exp_dyn
done
echo "=== done ==="
ls -R yolov8n yolov8s yolo11n yolov8n_dyn yolov8s_dyn yolo11n_dyn 2>/dev/null | grep -E "plan|:" || true
'
