#!/usr/bin/env bash
# The same A2 pipeline, swept over a model cost ladder.
#
# Every number elsewhere in this repo is YOLOv8s, so every conclusion is stated
# at one engine cost. This asks what those conclusions are worth at other costs:
# the pipeline's non-engine work (H2D, GPU compact kernel, host NMS) is a FIXED
# cost, so its share of the frame shrinks as the engine grows, and at some point
# the plumbing this study is about stops mattering.
#
# The ladder is one model family on purpose - YOLO11 n/s/m/l/x have identical
# output shape, NMS and launch pattern, so only cost varies and nothing is
# confounded. yolov8s rides along as the control that ties this sweep to every
# published number.
#
# Repeats are INTERLEAVED across models so thermal drift hits every rung equally
# rather than accumulating in whichever ran last.
set -euo pipefail

CONTAINER=${CONTAINER:-triton-server}
MODELS=${MODELS:-"yolo11n yolo11s yolo11m yolo11l yolo11x yolov8s"}
DURATION=${DURATION:-10}
REPEATS=${REPEATS:-3}
OUT=${1:-results/v3/model_scaling.tsv}
ROW=${ROW:-scripts/_model_row.py}

mkdir -p "$(dirname "$OUT")"
printf 'model\trepeat\tfps\tlat_p50_ms\tlat_p95_ms\tlat_p99_ms\tinfer_ms\th2d_ms\tnms_ms\tnon_engine_ms\tnon_engine_pct\tframes\tdets\tdets_per_frame\n' > "$OUT"

for r in $(seq 1 "$REPEATS"); do
  echo "--- repeat $r ---" >&2
  for m in $MODELS; do
    docker exec "$CONTAINER" bash -lc \
      "cd /work/cpp/build && ./trt_pipeline_cuda --engine /models/$m/1/model.plan \
       --mode file --file frames.bin --streams 1 --duration $DURATION" 2>/dev/null \
      | python3 "$ROW" "$m" "$r" "$OUT"
  done
done

echo "wrote $OUT" >&2
