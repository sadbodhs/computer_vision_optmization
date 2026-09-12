#!/usr/bin/env bash
# instance_group count ACROSS the concurrency range, for B2 and D.
#
# triton-tuning.md swept count at a single operating point - B2 at concurrency 4
# - and concluded "count: 2 was the right call". That is one cell of a grid. The
# interesting question is where extra instances do anything at all: with one
# request in flight the second instance is necessarily idle, so count cannot
# matter at concurrency 1, and the published B2-vs-A2 latency comparison at
# concurrency 1 gives Triton two instances against A2's single execution context.
#
# For D there is a second reason: instances and dynamic batching interact. Two
# instances means two batches can execute at once, which changes what the
# scheduler is able to form. Mean batch size is recorded per cell.
set -euo pipefail
DURATION=${DURATION:-8}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-$ROOT/results/v3/instance_grid.tsv}
C=${CONTAINER:-triton-server}

CFG_B2=$ROOT/triton/models/yolov8s/config.pbtxt
CFG_D=$ROOT/triton/models/yolov8s_dyn/config.pbtxt
B1=$(mktemp); B2=$(mktemp)
cp "$CFG_B2" "$B1"; cp "$CFG_D" "$B2"
restore() { cp "$B1" "$CFG_B2"; cp "$B2" "$CFG_D"; rm -f "$B1" "$B2"
            docker restart $C >/dev/null 2>&1 || true; ready || true; }
trap restore EXIT INT TERM

ready() { for i in $(seq 1 45); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready 2>/dev/null)" = "200" ] && return 0
  sleep 2; done; return 1; }

met() { curl -s localhost:8002/metrics | awk -v m="$1" -v mod="$2" '$0 ~ "^"m"\\{model=\""mod"\"" {print $2}'; }

write_b2() { cat > "$CFG_B2" <<EOC
name: "yolov8s"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 0
instance_group [ { count: $1 kind: KIND_GPU } ]
input [ { name: "images" data_type: TYPE_FP32 dims: [ 1, 3, 640, 640 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 1, 84, 8400 ] } ]
EOC
}
write_d() { cat > "$CFG_D" <<EOC
name: "yolov8s_dyn"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 8
dynamic_batching { preferred_batch_size: [ 4, 8 ] max_queue_delay_microseconds: 5000 }
instance_group [ { count: $1 kind: KIND_GPU } ]
input [ { name: "images" data_type: TYPE_FP32 dims: [ 3, 640, 640 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 84, 8400 ] } ]
EOC
}

mkdir -p "$(dirname "$OUT")"
printf 'flow\tinstances\tconcurrency\tfps\tlat_p50_ms\tmean_batch\n' > "$OUT"

for COUNT in 1 2 4; do
  write_b2 "$COUNT"; write_d "$COUNT"
  docker restart $C >/dev/null; ready || { echo "server down at count=$COUNT" >&2; continue; }
  sleep 5
  for N in 1 4 16; do
    for FLOW in B2 D; do
      if [ "$FLOW" = "B2" ]; then BIN="./trt_grpc_cuda --mode file --model yolov8s"; MOD=yolov8s
      else BIN="./trt_grpc_async --model yolov8s_dyn"; MOD=yolov8s_dyn; fi
      sleep 6
      n0=$(met nv_inference_count "$MOD"); e0=$(met nv_inference_exec_count "$MOD")
      J=$(docker exec $C bash -lc "cd /work/cpp/build && $BIN --file frames.bin --streams $N --duration $DURATION" 2>/dev/null) || J=""
      n1=$(met nv_inference_count "$MOD"); e1=$(met nv_inference_exec_count "$MOD")
      [ -z "$J" ] && { echo "  $FLOW count=$COUNT N=$N FAILED" >&2; continue; }
      echo "$J" | python3 -c "
import json,sys
d=json.load(sys.stdin)
n=${n1:-0}-${n0:-0}; e=${e1:-0}-${e0:-0}
b=(n/e) if e else 0
open('$OUT','a').write('%s\t%s\t%s\t%.1f\t%.3f\t%.2f\n'%('$FLOW','$COUNT','$N',d['fps'],d['lat_ms_p50'],b))
print('  %-3s count=%s N=%-3s %8.1f fps  p50 %7.3f ms  batch %.2f'%('$FLOW','$COUNT','$N',d['fps'],d['lat_ms_p50'],b))
"
    done
  done
done
echo "wrote $OUT" >&2
