#!/bin/bash
# Sweep the dynamic-batching knobs the study fixed at [4,8] / 5000us and never
# justified: max_queue_delay_microseconds, and whether preferred_batch_size
# matters once the delay is set.
#
# Flow D (trt_grpc_async, 8 in-flight) against yolov8s_dyn. The question the
# study leaves open is whether its 6.2ms floor at concurrency 1 is the 5ms
# batching window, and what shortening or lengthening it actually buys.
#
# Restores the original config on exit, including on error/interrupt.
#
# Usage: scripts/batching_knobs.sh [duration] [streams]
set -euo pipefail
DURATION=${1:-8}
STREAMS=${2:-8}
REPEATS=${REPEATS:-3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R=$ROOT/results/v3
mkdir -p "$R"
OUTF=$R/batching_knobs.tsv
CONTAINER=${CONTAINER:-triton-server}
CFG=$ROOT/triton/models/yolov8s_dyn/config.pbtxt
BACKUP=$(mktemp)

cp "$CFG" "$BACKUP"
restore() {
  echo "--- restoring yolov8s_dyn/config.pbtxt ---" >&2
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

write_cfg() {  # write_cfg <preferred list> <delay us>
  cat > "$CFG" <<EOF
name: "yolov8s_dyn"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 8
dynamic_batching {
  preferred_batch_size: [ $1 ]
  max_queue_delay_microseconds: $2
}
instance_group [ { count: 2 kind: KIND_GPU } ]
input [ { name: "images" data_type: TYPE_FP32 dims: [ 3, 640, 640 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 84, 8400 ] } ]
EOF
}

echo -e "preferred\tdelay_us\tstreams\tfps\tlat_p50_ms\tlat_p95_ms" > "$OUTF"

# delay sweep at the study's preferred_batch_size, then two alternative shapes
for spec in "4, 8:1000" "4, 8:5000" "4, 8:20000" "8:5000" "2, 4:5000"; do
  PREF="${spec%%:*}"; DELAY="${spec##*:}"
  write_cfg "$PREF" "$DELAY"
  docker restart $CONTAINER > /dev/null
  if ! wait_ready; then
    echo -e "$PREF\t$DELAY\t$STREAMS\tSERVER_FAILED\t\t" | tee -a "$OUTF"
    continue
  fi
  for rep in $(seq $REPEATS); do
    JSON=$(docker exec $CONTAINER bash -c \
      "cd /work/cpp/build && ./trt_grpc_async --model yolov8s_dyn --file frames.bin \
       --streams $STREAMS --duration $DURATION 2>/dev/null | tail -1")
    read -r FPS P50 P95 <<<"$(echo "$JSON" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(d["fps"], d["lat_ms_p50"], d["lat_ms_p95"])
except Exception:
    print("PARSE_FAIL", "", "")
')"
    echo -e "$PREF\t$DELAY\t$STREAMS\t$FPS\t$P50\t$P95" | tee -a "$OUTF"
  done
done

echo "=== saved to $OUTF ==="
