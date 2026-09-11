#!/usr/bin/env bash
# A2 with and without in-process CUDA graphs, in capacity mode.
#
# The published CUDA-graph result is a `trtexec --useCudaGraph` number: a loop
# with nothing around the engine. This runs the same engine inside the actual
# A2 pipeline, where an H2D copy, a GPU compact kernel and a host NMS share the
# stream, and asks how much of the gain survives.
#
# Same binary, one flag apart, and the arms are INTERLEAVED so thermal drift or
# a noisy neighbour hits both equally instead of accumulating in whichever ran
# second.
set -euo pipefail

CONTAINER=${CONTAINER:-triton-server}
ENGINE=${ENGINE:-/models/yolov8s/1/model.plan}
DURATION=${DURATION:-10}
REPEATS=${REPEATS:-5}
STREAMS=${STREAMS:-1}
OUT=${1:-results/v3/cuda_graphs_pipeline.tsv}
ROW=${ROW:-scripts/_cuda_graph_row.py}

mkdir -p "$(dirname "$OUT")"
printf 'flow\tgraph\trepeat\tstreams\tfps\tlat_p50_ms\tlat_p95_ms\tlat_p99_ms\tinfer_ms\th2d_ms\tnms_ms\tframes\tdets\tdets_per_frame\tgraph_active\n' > "$OUT"

for r in $(seq 1 "$REPEATS"); do
  for arm in off on; do
    flag=""
    [ "$arm" = "on" ] && flag="--cuda-graph"
    docker exec "$CONTAINER" bash -lc \
      "cd /work/cpp/build && ./trt_pipeline_cuda --engine $ENGINE --mode file \
       --file frames.bin --streams $STREAMS --duration $DURATION $flag" 2>/dev/null \
      | python3 "$ROW" "$arm" "$r" "$STREAMS" "$OUT"
  done
done

echo "wrote $OUT" >&2
