#!/bin/bash
# CUDA Graphs A/B on the batch-1 engines: how much of the "engine ceiling" is
# actually kernel-launch overhead rather than compute?
#
# The study's reference ceiling (0.97 ms / 1028 qps for YOLOv8s) was measured with
# ordinary per-iteration launches. --useCudaGraph replays the whole iteration as a
# single graph, removing per-launch CPU cost.
#
# Usage: scripts/cuda_graphs.sh [duration] [repeats]
set -euo pipefail
DURATION=${1:-5}
REPEATS=${2:-2}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R=$ROOT/results/v3
mkdir -p "$R"
OUTF=$R/cuda_graphs.tsv
CONTAINER=${CONTAINER:-triton-server}
TRTEXEC=/usr/src/tensorrt/bin/trtexec

echo -e "model\tmode\trepeat\tthroughput_qps\tlatency_median_ms" > "$OUTF"

for m in yolov8n yolov8s yolo11n; do
  for mode in plain graph; do
    FLAG=""; [ "$mode" = "graph" ] && FLAG="--useCudaGraph"
    for r in $(seq $REPEATS); do
      out=$(docker exec $CONTAINER $TRTEXEC --loadEngine=/models/$m/1/model.plan \
              --warmUp=500 --duration=$DURATION $FLAG 2>/dev/null)
      tp=$(echo "$out" | grep "Throughput:" | head -1 \
           | awk '{for(i=1;i<=NF;i++) if($i=="Throughput:"){print $(i+1); exit}}')
      lat=$(echo "$out" | grep -E "\[I\] Latency:" | head -1 \
            | grep -oE "median = [0-9.]+" | awk '{print $3}')
      echo -e "$m\t$mode\t$r\t$tp\t$lat" | tee -a "$OUTF"
    done
  done
done

echo "=== saved to $OUTF ==="
python3 - "$OUTF" <<'PY'
import sys, collections, statistics
rows = collections.defaultdict(list)
for i, line in enumerate(open(sys.argv[1])):
    if i == 0: continue
    m, mode, _, tp, lat = line.split("\t")
    rows[(m, mode)].append(float(tp))
print(f"{'model':10} {'plain':>10} {'graph':>10} {'gain':>8}")
for m in ("yolov8n", "yolov8s", "yolo11n"):
    p = statistics.median(rows[(m, "plain")]); g = statistics.median(rows[(m, "graph")])
    print(f"{m:10} {p:10.1f} {g:10.1f} {(g-p)/p*100:+7.1f}%")
PY
