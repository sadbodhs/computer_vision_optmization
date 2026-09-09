#!/bin/bash
# Rigorous benchmark harness v2.1: 4 arms x concurrency x REPEATS
set -u
DURATION=${1:-10}
REPEATS=${2:-3}
R=/home/suchi/sadbodh/rt_vs_triton/results/v2
mkdir -p $R
TS=$(date +%Y%m%d_%H%M%S)
OUTF=$R/all_$TS.tsv
echo "# arm|json" > $OUTF

docker_exec() { docker exec triton-server bash -c "$1" 2>/dev/null | tail -1; }

run_arm() { # run_arm <name> <docker-cmd-string>
  local NAME=$1; local CMD=$2
  for i in $(seq $REPEATS); do
    local OUT
    OUT=$(docker_exec "$CMD")
    echo "$NAME|$OUT" >> $OUTF
    echo "[$NAME r$i] $OUT"
  done
}

D=/work/cpp/build
echo "=== Benchmark v2.1: duration=${DURATION}s repeats=$REPEATS ==="

# 0) trtexec engine caps
for m in yolov8n yolov8s yolo11n; do
  T=$(docker_exec "/usr/src/tensorrt/bin/trtexec --loadEngine=/models/$m/1/model.plan --warmUp=200 --duration=3 2>/dev/null | grep 'Throughput:' | head -1 | awk '{print \$2}'")
  echo "trtexec_$m|{\"cap_qps\":$T}" >> $OUTF
  echo "[trtexec_$m] cap_qps=$T"
done

# 1) capacity sweeps
for N in 1 2 4 8 16; do
  run_arm "A_cap_s$N" "cd $D && ./trt_pipeline    --engine model.plan --mode file --file frames.bin --streams $N --duration $DURATION"
  run_arm "B_cap_s$N" "cd $D && ./trt_grpc_client --mode file --file frames.bin --model yolov8s     --streams $N --duration $DURATION"
  run_arm "C_cap_s$N" "cd /work && python3 client.py --mode file --file cpp/build/frames.bin --model yolov8s --streams $N --duration $DURATION"
  run_arm "D_cap_s$N" "cd $D && ./trt_grpc_client --mode file --file frames.bin --model yolov8s_dyn --streams $N --duration $DURATION"
done

# 2) 3-process pure TRT capacity (3 independent pipelines)
for i in $(seq $REPEATS); do
  OUT=$(docker_exec "bash /work/run_cpp3_cap.sh $DURATION" | tr '\n' ' ')
  echo "A3proc_cap|$OUT" >> $OUTF
  echo "[A3proc_cap r$i] $OUT"
done

# 3) RTSP end-to-end parity (source-capped 30fps)
run_arm "A_rtsp"  "cd $D && ./trt_pipeline    --engine model.plan --url rtsp://localhost:8554/cam1 --streams 1 --duration $DURATION"
run_arm "B_rtsp"  "cd $D && ./trt_grpc_client --mode rtsp --url rtsp://localhost:8554/cam1 --model yolov8s --streams 1 --duration $DURATION"
run_arm "C_rtsp"  "cd /work && python3 client.py --model yolov8s --streams 1 --duration $DURATION"
run_arm "C_rtsp3" "cd /work && python3 client.py --models yolov8n,yolov8s,yolo11n --streams 3 --duration $DURATION"

# 4) 3-process pure TRT RTSP (the original step-2 comparison)
for i in $(seq $REPEATS); do
  OUT=$(docker_exec "bash /work/run_cpp3.sh $DURATION" | tr '\n' ' ')
  echo "A3proc_rtsp|$OUT" >> $OUTF
  echo "[A3proc_rtsp r$i] $OUT"
done

echo "=== saved to $OUTF ==="