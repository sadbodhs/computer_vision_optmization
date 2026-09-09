#!/bin/bash
# Run 3 independent pure-TRT pipelines (capacity file mode) concurrently
DURATION=${1:-10}
cd /work/cpp/build
cp /models/yolov8n/1/model.plan n.plan
cp /models/yolov8s/1/model.plan s.plan
cp /models/yolo11n/1/model.plan l.plan

./trt_pipeline --engine n.plan --mode file --file frames.bin --streams 1 --duration $DURATION > /tmp/c1.json 2>/dev/null &
P1=$!
./trt_pipeline --engine s.plan --mode file --file frames.bin --streams 1 --duration $DURATION > /tmp/c2.json 2>/dev/null &
P2=$!
./trt_pipeline --engine l.plan --mode file --file frames.bin --streams 1 --duration $DURATION > /tmp/c3.json 2>/dev/null &
P3=$!
wait $P1 $P2 $P3
cat /tmp/c1.json /tmp/c2.json /tmp/c3.json