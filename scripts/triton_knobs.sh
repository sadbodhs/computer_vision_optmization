#!/bin/bash
# Sweep two Triton model-config knobs the study fixed without justification:
#   * instance_group count  (was hardcoded to 2 everywhere)
#   * optimization { cuda { graphs: true } }
#
# CUDA Graphs were measured at the ENGINE level in docs/cuda-graphs.md (+15-27%).
# This asks whether that survives the serving path, i.e. whether flow B2 sees it.
#
# Measures B2 (trt_grpc_cuda, CUDA shared memory) against yolov8s at fixed
# concurrency. The server is restarted for each config (MODE_NONE = no hot reload).
#
# The original config.pbtxt is restored on exit, including on Ctrl-C / error.
#
# Usage: scripts/triton_knobs.sh [duration] [streams]
set -euo pipefail
DURATION=${1:-8}
STREAMS=${2:-4}
REPEATS=${REPEATS:-3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R=$ROOT/results/v3
mkdir -p "$R"
OUTF=$R/triton_knobs.tsv
CONTAINER=${CONTAINER:-triton-server}
CFG=$ROOT/triton/models/yolov8s/config.pbtxt
BACKUP=$(mktemp)

cp "$CFG" "$BACKUP"
restore() {
  echo "--- restoring original config.pbtxt ---" >&2
  cp "$BACKUP" "$CFG"; rm -f "$BACKUP"
  docker restart $CONTAINER > /dev/null 2>&1 || true
  wait_ready || true
}
trap restore EXIT INT TERM

wait_ready() {
  for i in $(seq 1 45); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready 2>/dev/null)" = "200" ] && return 0
    sleep 2
  done
  return 1
}

write_cfg() {  # write_cfg <instance_count> <graphs true|false>
  local COUNT=$1 GRAPHS=$2
  cat > "$CFG" <<EOF
name: "yolov8s"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 0
instance_group [ { count: $COUNT kind: KIND_GPU } ]
input [ { name: "images" data_type: TYPE_FP32 dims: [ 1, 3, 640, 640 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 1, 84, 8400 ] } ]
EOF
  if [ "$GRAPHS" = "true" ]; then
    cat >> "$CFG" <<EOF
optimization { cuda { graphs: true } }
EOF
  fi
}

echo -e "instances\tgraphs\tstreams\tfps\tlat_p50_ms\tlat_p95_ms" > "$OUTF"

for COUNT in 1 2 4; do
  for GRAPHS in false true; do
    write_cfg "$COUNT" "$GRAPHS"
    docker restart $CONTAINER > /dev/null
    if ! wait_ready; then
      echo -e "$COUNT\t$GRAPHS\t$STREAMS\tSERVER_FAILED\t\t" | tee -a "$OUTF"
      continue
    fi
    for rep in $(seq $REPEATS); do
      JSON=$(docker exec $CONTAINER bash -c \
        "cd /work/cpp/build && ./trt_grpc_cuda --mode file --file frames.bin --model yolov8s \
         --streams $STREAMS --duration $DURATION 2>/dev/null | tail -1")
      read -r FPS P50 P95 <<<"$(echo "$JSON" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(d["fps"], d["lat_ms_p50"], d["lat_ms_p95"])
except Exception:
    print("PARSE_FAIL", "", "")
')"
      echo -e "$COUNT\t$GRAPHS\t$STREAMS\t$FPS\t$P50\t$P95" | tee -a "$OUTF"
    done
  done
done

echo "=== saved to $OUTF ==="
