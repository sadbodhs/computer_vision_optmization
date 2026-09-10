#!/bin/bash
# Parallel-instance contention test: N independent pipeline instances on one GPU.
# Each instance = 1 process in capacity mode; each reports its own fps + latency.
# Usage: parallel_test.sh <duration> <N>
set -u
DURATION=${1:-8}
N=${2:-3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repo root, wherever it is checked out
R=$ROOT/results/v3
mkdir -p $R
OUTF=$R/parallel_contention_N${N}.tsv
D=/work/cpp/build

cat > /tmp/launch_A2.sh << EOF
cd $D
cp /models/yolov8s/1/model.plan a2.plan
for i in \$(seq $N); do
  ./trt_pipeline_cuda --engine a2.plan --mode file --file frames.bin --streams 1 --duration $DURATION > /tmp/par_A2_\$i.json 2>/dev/null &
done
wait
EOF

cat > /tmp/launch_B2.sh << EOF
cd $D
for i in \$(seq $N); do
  ./trt_grpc_cuda --mode file --file frames.bin --model yolov8s --streams 1 --duration $DURATION > /tmp/par_B2_\$i.json 2>/dev/null &
done
wait
EOF

cat > /tmp/launch_C2.sh << EOF
cd /work
for i in \$(seq $N); do
  python3 client_v2.py --mode file --file cpp/build/frames.bin --model yolov8s --transfer sys --streams 1 --duration $DURATION > /tmp/par_C2_\$i.json 2>/dev/null &
done
wait
EOF

echo "flow,instance,json" | tee $OUTF
for FLOW in A2 B2 C2; do
  echo "=== $FLOW x $N ==="
  bash /tmp/launch_$FLOW.sh 2>/dev/null
  for i in $(seq $N); do
    J=$(cat /tmp/par_${FLOW}_$i.json 2>/dev/null | tail -1)
    echo "$FLOW,$i,$J" | tee -a $OUTF
    echo "  [$FLOW #$i] $J"
  done
done