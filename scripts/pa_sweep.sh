#!/bin/bash
# perf_analyzer sweep per model per concurrency — one process at a time (avoids OOM)
# Usage: pa_sweep.sh <model> <concurrency_list e.g. "1 2 4 8 16">
M=$1
shift
CONCS="$@"
for c in $CONCS; do
  if [ "$c" = "1" ]; then
    /opt/bin/perf_analyzer -m $M -i grpc -u localhost:8001 -b 1 \
      --concurrency-range $c --shape images:1,3,640,640 \
      > /tmp/pa_${M}_${c}.txt 2>&1
  else
    /opt/bin/perf_analyzer -m $M -i grpc -u localhost:8001 -b 1 --async \
      --concurrency-range $c --shape images:1,3,640,640 \
      > /tmp/pa_${M}_${c}.txt 2>&1
  fi
  tp=$(grep "Throughput:" /tmp/pa_${M}_${c}.txt | head -1 | awk '{print $2}')
  lat=$(grep "Avg latency:" /tmp/pa_${M}_${c}.txt | head -1 | awk '{print $3}')
  echo "$M conc=$c throughput=$tp infer/s avg_latency_us=$lat"
done