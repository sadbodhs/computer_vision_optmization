#!/bin/bash
# Run 3 independent C++ TRT pipelines (one per stream+model)
DURATION=${1:-15}
cd /work/cpp/build
cp /models/yolov8n/1/model.plan n.plan
cp /models/yolov8s/1/model.plan s.plan
cp /models/yolo11n/1/model.plan l.plan

./trt_pipeline --engine n.plan --url rtsp://localhost:8554/cam1 --duration $DURATION > /tmp/cpp1.json 2>/dev/null &
P1=$!
./trt_pipeline --engine s.plan --url rtsp://localhost:8554/cam2 --duration $DURATION > /tmp/cpp2.json 2>/dev/null &
P2=$!
./trt_pipeline --engine l.plan --url rtsp://localhost:8554/cam3 --duration $DURATION > /tmp/cpp3.json 2>/dev/null &
P3=$!
wait $P1 $P2 $P3
cat /tmp/cpp1.json /tmp/cpp2.json /tmp/cpp3.json
