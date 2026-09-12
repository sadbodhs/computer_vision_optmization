#!/usr/bin/env bash
# What batch size does Triton ACTUALLY form for flow D?
#
# docs/results.md prints "svc 0.61" on every D row - the same constant five
# times - which is the trtexec batch-8 per-frame cost, not a measurement of
# what the server achieved at that concurrency. If the batch were really full
# at concurrency 1, D would run at ~1640 fps; it runs at 1041. So the batch is
# not full, and the table implies otherwise.
#
# Triton's own counters settle it:
#   nv_inference_count / exec_count       = mean batch size actually formed
#   compute_infer_duration_us / requests  = real GPU service per frame
#   queue_duration_us / requests          = real queue wait per frame
set -euo pipefail
M=${MODEL:-yolov8s_dyn}
DURATION=${DURATION:-10}
OUT=${1:-results/v3/batch_achieved.tsv}

scrape() {  # $1 = metric name -> value for $M
  curl -s localhost:8002/metrics \
    | awk -v m="$1" -v mod="$M" '$0 ~ "^"m"\\{model=\""mod"\"" {print $2}'
}

mkdir -p "$(dirname "$OUT")"
printf 'concurrency\tfps\trequests\texecutions\tmean_batch\tsvc_us_per_frame\tqueue_us_per_frame\tcompute_us_per_exec\n' > "$OUT"

for N in 1 2 4 8 16; do
  # Let the previous arm's queue drain. Without this, a high-concurrency run
  # starts while the server is still working through the last one's backlog -
  # which produced a 554 fps reading for N=16 that re-runs put at ~1620.
  sleep "${SETTLE:-8}"
  r0=$(scrape nv_inference_count); e0=$(scrape nv_inference_exec_count)
  c0=$(scrape nv_inference_compute_infer_duration_us); q0=$(scrape nv_inference_queue_duration_us)

  fps=$(docker exec triton-server bash -lc \
    "cd /work/cpp/build && ./trt_grpc_async --model $M --file frames.bin --streams $N --duration $DURATION" \
    2>/dev/null | python3 -c 'import json,sys; print("%.1f"%json.load(sys.stdin)["fps"])')

  r1=$(scrape nv_inference_count); e1=$(scrape nv_inference_exec_count)
  c1=$(scrape nv_inference_compute_infer_duration_us); q1=$(scrape nv_inference_queue_duration_us)

  python3 - "$N" "$fps" "$r0" "$r1" "$e0" "$e1" "$c0" "$c1" "$q0" "$q1" "$OUT" <<'PY'
import sys
N, fps = sys.argv[1], sys.argv[2]
r0, r1, e0, e1, c0, c1, q0, q1 = (float(x) for x in sys.argv[3:11])
out = sys.argv[11]
req, ex, comp, que = r1 - r0, e1 - e0, c1 - c0, q1 - q0
row = [N, fps, "%.0f" % req, "%.0f" % ex,
       "%.2f" % (req / ex if ex else 0),
       "%.1f" % (comp / req if req else 0),
       "%.1f" % (que / req if req else 0),
       "%.1f" % (comp / ex if ex else 0)]
open(out, "a").write("\t".join(row) + "\n")
print("  N=%-3s %8s fps  requests %7.0f  execs %7.0f  mean batch %5.2f  "
      "svc %6.1f us/frame  queue %8.1f us/frame"
      % (N, fps, req, ex, req / ex if ex else 0,
         comp / req if req else 0, que / req if req else 0))
PY
done
echo "wrote $OUT" >&2
