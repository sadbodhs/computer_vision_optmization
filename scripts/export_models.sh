#!/bin/bash
set -e
docker run --rm --gpus all   -v /home/suchi/sadbodh/rt_vs_triton:/work   nvcr.io/nvidia/tritonserver:24.12-py3 bash -c '
pip install -q ultralytics onnx onnxsim 2>&1 | tail -1
cd /work/triton/models
for m in yolov8n yolov8s yolo11n; do
  mkdir -p $m/1
  yolo export model=$m.pt format=onnx imgsz=640 dynamic=false simplify=True 2>&1 | tail -2
done
ls -la
'
